#!/usr/bin/env python3
"""Calibrate a camera from photographs of the on-screen tag grid.

The grid is a calibration target in its own right, and a better one than a
printed chessboard here: its feature positions are addressed in screen pixels
so they are exact, every corner carries a marker ID so partial views still
contribute, and it is the same target the pose test uses, through the same
capture path. A phone that calibrates on one target and runs on another has
had two chances to change lens, resolution or processing in between.

What this cannot do is fix a target that is not planar, so the panel must be
flat and the photos must be of the panel itself - not a photo of a photo.

Shooting guide (15-25 photos):
  - Lock the phone to ONE lens (usually 1x). Ultrawide/tele switching mid-set
    silently mixes two different cameras into one intrinsic model.
  - Shoot JPEG, not HEIC - OpenCV cannot read HEIC. Turn off portrait mode,
    HDR and any "scene optimisation"; they warp geometry non-linearly.
  - Keep every photo the same resolution and orientation.
  - Vary the tilt. A planar target photographed only face-on cannot separate
    focal length from distance - they trade off exactly. Aim for tilts from
    face-on out to 40-50 degrees, in several directions, plus a couple of
    rolls, and fill the frame with the panel.

Usage:
    python tools/calibrate.py --photos photos/calib --out calib.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import starnav  # noqa: E402

# A planar target constrains k3 very weakly; leaving it free lets it soak up
# noise and distort the edges of the model. Fixed by default.
DEFAULT_FLAGS = cv2.CALIB_FIX_K3

# Below this, a view carries too little of the grid to be worth its own
# extrinsic parameters.
MIN_MARKERS_PER_VIEW = 6

# CLAUDE.md's acceptance threshold for calibration quality.
MAX_ACCEPTABLE_RMS_PX = 0.5

# A planar-target set with no oblique views is degenerate in focal length.
MIN_TILT_SPREAD_DEG = 20.0


def collect_views(paths, detector, marker_map, offsets):
    """Detect the grid in each photo and build its point correspondences.

    Object points are built in the marker plane's OWN frame, at Z = 0, rather
    than at the ceiling height. calibrateCamera's planar initialisation
    (initCameraMatrix2D) assumes a Z = 0 target; handing it a constant non-zero
    Z is still a plane but defeats that path. Only the intrinsics are wanted
    here, and those do not care where the plane sits in the world.
    """
    plane_map = dict(marker_map, ceiling_height_m=0.0)
    object_points, image_points, used, skipped = [], [], [], []
    image_size = None

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            skipped.append((path.name, "unreadable (HEIC? use JPEG)"))
            continue

        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            # Intrinsics are tied to a specific sensor readout. Mixing sizes or
            # orientations produces one model that fits neither.
            skipped.append((path.name, f"size {size[0]}x{size[1]} != {image_size[0]}x{image_size[1]}"))
            continue

        corners, ids, _ = detector.detectMarkers(image)
        known, _unknown = starnav.split_known_unknown(corners, ids, set(marker_map["markers"]))
        if len(known) < MIN_MARKERS_PER_VIEW:
            skipped.append((path.name, f"only {len(known)} mapped markers"))
            continue

        obj, img, _ids = starnav.build_correspondences(known, plane_map, offsets)
        object_points.append(obj.astype(np.float32))
        image_points.append(img.astype(np.float32))
        used.append((path.name, len(known)))

    return object_points, image_points, image_size, used, skipped


def tilt_angles_deg(rvecs):
    """Angle between the target plane's normal and the camera's optical axis.

    This is the diversity that makes a planar calibration well posed, so it is
    reported rather than assumed.
    """
    angles = []
    for rvec in rvecs:
        rotation, _ = cv2.Rodrigues(rvec)
        normal_in_camera = rotation @ np.array([0.0, 0.0, 1.0])
        angles.append(np.degrees(np.arccos(min(1.0, abs(float(normal_in_camera[2]))))))
    return np.array(angles)


def per_view_errors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    return np.array([
        starnav.reprojection_error(obj, img, rvec, tvec, camera_matrix, dist_coeffs)
        for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs)
    ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="calibrate from photos of the screen tag grid")
    parser.add_argument("--photos", type=Path, required=True, help="folder of calibration photos")
    parser.add_argument("--markers", type=Path, default=Path("config/markers_screen.json"),
                        help="marker map the photos show")
    parser.add_argument("--hall", type=Path, default=Path("config/hall.json"),
                        help="source of the dictionary and detector settings")
    parser.add_argument("--out", type=Path, default=Path("calib.npz"),
                        help="where to write the intrinsics")
    parser.add_argument("--free-k3", action="store_true",
                        help="let the third radial term float (rarely justified on a planar target)")
    parser.add_argument("--rational", action="store_true",
                        help="use the rational distortion model, for very wide lenses")
    parser.add_argument("--refine-win", type=int, default=None,
                        help="corner refinement window; raise it for high-resolution stills")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hall = starnav.load_json(args.hall)
    marker_map = starnav.load_marker_map(args.markers)

    detection_cfg = dict(hall["detection"])
    if args.refine_win:
        detection_cfg["corner_refine_win_size"] = args.refine_win
    detector = starnav.make_detector(detection_cfg)

    paths = sorted(p for p in args.photos.iterdir()
                   if p.suffix.lower() in starnav.IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"no readable images in {args.photos} "
                         f"(supported: {', '.join(starnav.IMAGE_EXTENSIONS)})")

    offsets = starnav.corner_offsets(marker_map["tag_size_m"] / 2.0)
    object_points, image_points, image_size, used, skipped = collect_views(
        paths, detector, marker_map, offsets)

    print(f"photos found : {len(paths)}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    if len(object_points) < 5:
        raise SystemExit(f"only {len(object_points)} usable views; need at least 5, "
                         f"and 15-25 for a trustworthy result")
    print(f"usable views : {len(object_points)} at {image_size[0]}x{image_size[1]}")
    print(f"markers/view : min {min(n for _, n in used)}, max {max(n for _, n in used)}")

    flags = 0 if args.free_k3 else DEFAULT_FLAGS
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags)

    errors = per_view_errors(object_points, image_points, rvecs, tvecs,
                             camera_matrix, dist_coeffs)
    tilts = tilt_angles_deg(rvecs)
    focal_x, focal_y = camera_matrix[0, 0], camera_matrix[1, 1]
    hfov = 2 * np.degrees(np.arctan(image_size[0] / (2 * focal_x)))

    # Two statistics, both worth reporting, easily confused. OpenCV's rms is
    # sqrt(sum(dx^2 + dy^2) / n_corners) pooled over every corner in every
    # view, so a handful of bad corners pull it up hard. The per-view figure
    # averages distances instead. A large gap between them means outliers
    # rather than uniformly poor corners - usually a few blurred or
    # grazing-angle tags, which is worth knowing before recapturing everything.
    pooled_mean = float(np.mean([e for e in errors]))
    print(f"\nRMS reprojection : {rms:.4f} px   (pooled over all corners)")
    print(f"per-view mean    : median {np.median(errors):.3f}, worst {errors.max():.3f} px "
          f"({used[int(np.argmax(errors))][0]}), mean of views {pooled_mean:.3f}")
    print(f"focal length     : fx {focal_x:.1f}, fy {focal_y:.1f} px  "
          f"(aspect {focal_y / focal_x:.4f}, HFOV {hfov:.1f} deg)")
    print(f"principal point  : {camera_matrix[0, 2]:.1f}, {camera_matrix[1, 2]:.1f} "
          f"(centre is {image_size[0] / 2:.0f}, {image_size[1] / 2:.0f})")
    print(f"distortion       : {np.array2string(dist_coeffs.ravel(), precision=4)}")
    print(f"view tilt spread : {tilts.min():.0f} to {tilts.max():.0f} deg")

    if rms > MAX_ACCEPTABLE_RMS_PX:
        print(f"\nREJECT: RMS {rms:.3f} px exceeds the {MAX_ACCEPTABLE_RMS_PX} px "
              f"threshold. Recapture rather than proceeding - every position error "
              f"downstream inherits this. Usual causes: mixed lenses, HDR or "
              f"portrait mode left on, motion blur, or a non-flat panel.")
    if tilts.max() - tilts.min() < MIN_TILT_SPREAD_DEG:
        print(f"\nWARNING: only {tilts.max() - tilts.min():.0f} deg of tilt variation. "
              f"A planar target shot near face-on cannot separate focal length from "
              f"distance; fx will be confidently wrong. Add oblique views.")

    np.savez(args.out, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
             image_size=np.array(image_size), rms=rms, n_views=len(object_points))
    print(f"\nwrote {args.out}")
    print("evaluate on photos this calibration did NOT see - reusing the same set "
          "reports the fit, not the accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
