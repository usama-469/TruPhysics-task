#!/usr/bin/env python3
"""Refine surveyed marker positions from the images that see them.

Calibration treats the marker map as truth. When the map is a tape measure held
overhead, its error - a few millimetres per tag - has nowhere to go but into the
intrinsics, and it shows up as a reprojection residual that no amount of
reshooting will reduce. On the ceiling rig that floor was about 9 px.

This solves for the marker positions as well, by alternating two steps that are
each easy on their own:

  1. Hold the map fixed, run cv2.calibrateCamera -> intrinsics and one pose
     per view.
  2. Hold the cameras fixed, move each marker to wherever best explains the
     corners observed for it across every view it appears in.

cv2.calibrateCameraRO does this in one call, but only for a pattern that is
fully visible in every view. Here each view sees a different 4-6 of the 9 tags,
so the alternation is what makes it work.

Two things pin the solution down, because otherwise the map and the cameras
could drift together:

  scale       - the tag's own size. Corners sit at fixed offsets from a marker
                centre, so tag_size_m anchors the metric scale and only the
                centres are free. Get tag_size_m wrong and this will happily
                refine a map that is uniformly the wrong size.
  pose        - after each update the refined map is rigidly aligned back onto
                the surveyed one. That removes the free translation and
                rotation while leaving the SHAPE correction, so the output
                stays in the frame you measured rather than sliding into an
                arbitrary one.

Usage:
    python tools/refine_map.py --photos photos/ceiling_calib --out config/markers.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import starnav  # noqa: E402

# Step used for the numerical Jacobian of image position against marker
# position. Small enough to be linear over the step, large enough that the
# projection difference is not lost in floating point.
JACOBIAN_STEP_M = 1e-4


def collect(paths, detector, known_ids, min_markers):
    """Per view, the image corners of every mapped marker it shows."""
    views, image_size, skipped = [], None, []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            skipped.append((path.name, "unreadable"))
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            skipped.append((path.name, f"size {size[0]}x{size[1]}"))
            continue
        corners, ids, _ = detector.detectMarkers(image)
        known, _ = starnav.split_known_unknown(corners, ids, known_ids)
        if len(known) < min_markers:
            skipped.append((path.name, f"only {len(known)} markers"))
            continue
        views.append((path.name, {mid: c.reshape(4, 2).astype(np.float64)
                                  for mid, c in known}))
    return views, image_size, skipped


def as_state(entry, plane_z):
    """A marker as (x, y, z, yaw), whatever length it was stored at."""
    values = list(entry) + [plane_z, 0.0]
    return values[0], values[1], (values[2] if len(entry) > 2 else plane_z),         (values[3] if len(entry) > 3 else 0.0)


def marker_corners(state, offsets):
    """The four world corners of a marker at (x, y, z, yaw)."""
    x, y, z, yaw = state[0], state[1], state[2], state[3]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[x + dx * c - dy * s, y + dx * s + dy * c, z]
                     for dx, dy in offsets])


def build_points(views, positions, offsets, plane_z):
    """Object/image point arrays for calibrateCamera, from the current map."""
    object_points, image_points = [], []
    for _, seen in views:
        obj, img = [], []
        for mid in sorted(seen):
            world = marker_corners(as_state(positions[mid], plane_z), offsets)
            for point, corner in zip(world, seen[mid]):
                obj.append(tuple(point))
                img.append(corner)
        object_points.append(np.array(obj, np.float32))
        image_points.append(np.array(img, np.float32))
    return object_points, image_points


def update_positions(views, positions, offsets, plane_z, rvecs, tvecs,
                     camera_matrix, dist_coeffs, refine_z=False, refine_yaw=False):
    """Move each marker to whatever best explains its observed corners.

    Moving a marker centre translates all four of its corners equally in the
    world, so each marker is a two-parameter least-squares problem: accumulate
    the normal equations over every view that sees it and solve a 2x2 system.
    The Jacobian is taken numerically because cv2.projectPoints differentiates
    with respect to the pose and the intrinsics, not the object points.
    """
    # Optionally solve the marker's height too. Printed sheets taped to a wall
    # are not exactly coplanar - paper curls, tape has thickness, walls are not
    # flat - and that deviation is invisible to a purely 2D refinement, which
    # leaves it stranded in the residual.
    dims = 2 + int(refine_z) + int(refine_yaw)
    normals = {mid: (np.zeros((dims, dims)), np.zeros(dims)) for mid in positions}
    for (_, seen), rvec, tvec in zip(views, rvecs, tvecs):
        for mid, observed in seen.items():
            state = np.array(as_state(positions[mid], plane_z))
            base = marker_corners(state, offsets)
            projected, _ = cv2.projectPoints(base, rvec, tvec, camera_matrix, dist_coeffs)
            residual = (projected.reshape(-1, 2) - observed).ravel()

            jacobian = np.empty((residual.size, dims))
            for axis in range(dims):
                bumped = state.copy()
                # yaw is an angle, so it gets an angular step
                bumped[axis] += JACOBIAN_STEP_M if axis < 3 else 1e-3
                moved, _ = cv2.projectPoints(marker_corners(bumped, offsets),
                                             rvec, tvec, camera_matrix, dist_coeffs)
                step = JACOBIAN_STEP_M if axis < 3 else 1e-3
                jacobian[:, axis] = ((moved.reshape(-1, 2) - projected.reshape(-1, 2))
                                     / step).ravel()

            jtj, jtr = normals[mid]
            normals[mid] = (jtj + jacobian.T @ jacobian,
                            jtr + jacobian.T @ residual)

    updated = {}
    for mid, (jtj, jtr) in normals.items():
        try:
            # Gauss-Newton step. A marker seen in too few views gives a
            # singular system; leave it where the survey put it.
            step = np.linalg.solve(jtj, -jtr)
        except np.linalg.LinAlgError:
            step = np.zeros(2)
        state = np.array(as_state(positions[mid], plane_z))
        state[:dims] += step
        updated[mid] = tuple(state)
    return updated


def align_to(positions, reference):
    """Put the refined map back into the surveyed frame (Umeyama, 2D).

    Rotation, translation AND scale are all free parameters of the joint
    problem - the cameras and the focal length can absorb any of them - so
    without this the refined map drifts. Scale needs removing as much as the
    other two: in an early version it was left free on the argument that
    tag_size_m anchors it, and the map quietly grew 6 % while the plane moved
    90 mm closer, which is the same picture to a camera.

    So the SHAPE of the map is refined and its overall size is held to the
    survey. A tape across a 1.8 m grid is a better length reference than a
    printed tag whose size depends on what the printer did. Any residual scale
    error then lands in the focal length, where section 11 of the evaluation
    shows it costs range rather than X/Y.
    """
    ids = sorted(positions)
    p = np.array([positions[i][:2] for i in ids])
    q = np.array([reference[i] for i in ids])
    pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
    u, sigma, vt = np.linalg.svd(pc.T @ qc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:            # reflection is not a rigid move
        vt[-1] *= -1
        sigma[-1] *= -1
        rotation = vt.T @ u.T
    variance = (pc ** 2).sum()
    scale = sigma.sum() / variance if variance > 0 else 1.0
    shifted = scale * (rotation @ pc.T).T + q.mean(axis=0)
    out = {}
    for k, i in enumerate(ids):
        rest = tuple(float(v) for v in list(positions[i])[2:])
        out[i] = (float(shifted[k][0]), float(shifted[k][1])) + rest
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="refine marker positions from images")
    parser.add_argument("--photos", type=Path, required=True,
                        help="folder of images of the mounted grid")
    parser.add_argument("--hall", type=Path, default=Path("config/hall.json"))
    parser.add_argument("--markers", type=Path, default=None,
                        help="starting map. Defaults to hall.json's 'markers' key.")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the refined map (default: overwrite input)")
    parser.add_argument("--calib-out", type=Path, default=None,
                        help="also write the intrinsics this produced")
    parser.add_argument("--refine-yaw", action="store_true",
                        help="also solve each tag's own rotation in the plane. Tape a "
                             "sheet up by hand and it lands a few degrees off; on a "
                             "150 mm tag 4 degrees moves a corner about 3.7 mm.")
    parser.add_argument("--rational", action="store_true",
                        help="rational distortion model (k4-k6). Worth trying on a "
                             "wide phone lens the 5-parameter model cannot fit.")
    parser.add_argument("--free-k3", action="store_true",
                        help="let k3 vary instead of fixing it at zero")
    parser.add_argument("--refine-z", action="store_true",
                        help="also solve each marker's height. Tags taped to a wall "
                             "are not exactly coplanar, and that shows up as an "
                             "irreducible residual if it is not modelled.")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--min-markers", type=int, default=4)
    parser.add_argument("--refine-win", type=int, default=None)
    parser.add_argument("--convention", nargs=2, type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hall = starnav.load_json(args.hall)
    markers_path = args.markers or Path(hall.get("markers", "config/markers.json"))
    marker_map = starnav.load_marker_map(markers_path)
    surveyed = dict(marker_map["markers"])
    # Solve in the marker plane's OWN frame at Z = 0. calibrateCamera's planar
    # initialisation refuses a rig at constant non-zero Z, and where the plane
    # sits in the world changes neither the intrinsics nor the marker layout.
    ceiling_height = marker_map["ceiling_height_m"]
    plane_z = 0.0

    detection_cfg = dict(hall["detection"])
    if args.refine_win:
        detection_cfg["corner_refine_win_size"] = args.refine_win
    detector = starnav.make_detector(detection_cfg)
    convention = args.convention or hall.get("pose", {}).get("convention", (0, 0))
    offsets = starnav.corner_offsets(marker_map["tag_size_m"] / 2.0,
                                     int(convention[0]), bool(convention[1]))

    paths = sorted(p for p in args.photos.iterdir()
                   if p.suffix.lower() in starnav.IMAGE_EXTENSIONS)
    views, image_size, skipped = collect(paths, detector, set(surveyed), args.min_markers)
    if len(views) < 5:
        raise SystemExit(f"only {len(views)} usable views; need at least 5")

    print(f"map          : {markers_path}")
    print(f"convention   : rotation {convention[0]}, mirror {bool(convention[1])}")
    print(f"views        : {len(views)} of {len(paths)} at {image_size[0]}x{image_size[1]}")
    print(f"tag size     : {marker_map['tag_size_m']} m  (this sets the scale; it is "
          f"NOT refined)\n")

    positions = dict(surveyed)
    camera_matrix = dist_coeffs = None
    print(f"{'iter':>5}{'RMS px':>10}{'largest move mm':>18}")
    for iteration in range(args.iterations):
        object_points, image_points = build_points(views, positions, offsets, plane_z)
        flags = 0 if args.free_k3 or args.rational else cv2.CALIB_FIX_K3
        if args.rational:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if camera_matrix is not None:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, camera_matrix, dist_coeffs,
            flags=flags)

        moved = align_to(update_positions(views, positions, offsets, plane_z,
                                          rvecs, tvecs, camera_matrix, dist_coeffs,
                                          args.refine_z, args.refine_yaw),
                         surveyed)
        largest = max(np.hypot(moved[i][0] - positions[i][0],
                               moved[i][1] - positions[i][1]) for i in positions)
        positions = moved
        print(f"{iteration:>5}{rms:>10.4f}{1000 * largest:>18.3f}")

    object_points, image_points = build_points(views, positions, offsets, plane_z)
    final_flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, camera_matrix, dist_coeffs,
        flags=final_flags)

    # Per-view residuals: a flat spread is a model that fits every view equally
    # badly (survey, distortion). A wide spread points at particular frames -
    # motion blur, or rolling-shutter skew from a handheld shot.
    per_view = np.array([
        starnav.reprojection_error(o, i, r, t, camera_matrix, dist_coeffs)
        for o, i, r, t in zip(object_points, image_points, rvecs, tvecs)])
    print(f"per-view err : median {np.median(per_view):.3f}, "
          f"p10 {np.percentile(per_view, 10):.3f}, p90 {np.percentile(per_view, 90):.3f}, "
          f"worst {per_view.max():.3f} px")

    print(f"\nfinal RMS    : {rms:.4f} px")
    print(f"focal length : fx {camera_matrix[0, 0]:.1f}, fy {camera_matrix[1, 1]:.1f}")
    print(f"principal pt : {camera_matrix[0, 2]:.1f}, {camera_matrix[1, 2]:.1f}")

    print(f"\n{'id':>4}{'x':>10}{'y':>9}{'moved mm':>10}{'dz mm':>9}{'yaw deg':>9}")
    shifts = []
    for mid in sorted(positions):
        sx, sy = surveyed[mid]
        rx, ry, rz, ryaw = as_state(positions[mid], plane_z)
        shift = 1000 * float(np.hypot(rx - sx, ry - sy))
        shifts.append(shift)
        print(f"{mid:>4}{rx:>10.4f}{ry:>9.4f}{shift:>10.2f}"
              f"{1000 * (rz - plane_z):>9.2f}{np.degrees(ryaw):>9.2f}")
    print(f"\nmarker moves : median {np.median(shifts):.2f} mm, worst {max(shifts):.2f} mm")
    print("               Compare that against your survey uncertainty. Moves far "
          "larger than\n               your tape error mean a mis-measured tag, not "
          "a refinement.")

    out = args.out or markers_path
    payload = ['{', f'  "tag_size_m": {marker_map["tag_size_m"]:.6f},',
               f'  "ceiling_height_m": {ceiling_height:.3f},',
               f'  "survey_uncertainty_m": {max(0.001, np.median(shifts) / 1000):.4f},',
               '  "markers": {']
    rows = []
    for mid in sorted(positions):
        x, y, z, yaw = as_state(positions[mid], plane_z)
        if args.refine_yaw:
            rows.append(f'    "{mid}": [{x:.4f}, {y:.4f}, '
                        f'{ceiling_height + z - plane_z:.4f}, {yaw:.5f}]')
        elif args.refine_z:
            rows.append(f'    "{mid}": [{x:.4f}, {y:.4f}, '
                        f'{ceiling_height + z - plane_z:.4f}]')
        else:
            rows.append(f'    "{mid}": [{x:.4f}, {y:.4f}]')
    payload.append(",\n".join(rows))
    payload += ['  }', '}']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(payload) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    if args.calib_out:
        np.savez(args.calib_out, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                 image_size=np.array(image_size), rms=rms, n_views=len(views))
        print(f"wrote {args.calib_out}")
    print("\nThese positions are fitted to THESE images. Evaluate on a set the "
          "refinement did not see,\nor the reported accuracy is the fit rather than "
          "the accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
