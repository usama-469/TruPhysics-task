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

Two targets are supported. The tag grid is the default and the better one for
the reasons above. A plain chessboard is also accepted, because it is the
classic target and findChessboardCornersSB localises it very precisely - but
the whole board must be visible in every view, so frame-corner coverage has to
come from moving the camera rather than from letting the target run off frame.
That matters: distortion is largest at the frame edges, and it can only be
estimated where there are corners.

Usage:
    python tools/calibrate.py --photos photos/calib --out calib.npz
    python tools/calibrate.py --photos photos/calib_v2 --chessboard 9x6 --square-mm 43.9
"""

from __future__ import annotations

import argparse
import struct
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


# EXIF tags worth reading. A phone records which lens took each frame, so
# "did it switch lenses on me?" is a question the files can answer directly
# instead of being inferred from a bad residual.
EXIF_FOCAL_LENGTH = 0x920A
EXIF_FOCAL_35MM = 0xA405
EXIF_LENS_MODEL = 0xA434
EXIF_MODEL = 0x0110
EXIF_SUB_IFD = 0x8769
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}


def _parse_ifd(block, offset, endian, into):
    """Read one IFD's entries into `into` as {tag: (type, count, raw bytes)}."""
    if offset + 2 > len(block):
        return
    (entries,) = struct.unpack(endian + "H", block[offset:offset + 2])
    for index in range(entries):
        at = offset + 2 + index * 12
        if at + 12 > len(block):
            return
        tag, kind, count = struct.unpack(endian + "HHI", block[at:at + 8])
        size = _TYPE_SIZE.get(kind, 0) * count
        if not size:
            continue
        if size <= 4:
            payload = block[at + 8:at + 8 + size]
        else:
            (pointer,) = struct.unpack(endian + "I", block[at + 8:at + 12])
            payload = block[pointer:pointer + size]
        into[tag] = (kind, count, payload)


def read_exif(path):
    """Camera model, lens and focal length from a JPEG, or {} if absent.

    Hand-rolled rather than pulling in a dependency: the project is Python +
    OpenCV + NumPy and this needs five tags out of the APP1 segment, which is a
    TIFF header sitting a few bytes into the file.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read(384 * 1024)
    except OSError:
        return {}
    if data[:2] != b"\xff\xd8":
        return {}

    cursor, tiff = 2, None
    while cursor + 4 <= len(data) and data[cursor] == 0xFF:
        marker = data[cursor + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            cursor += 2
            continue
        if marker == 0xDA:  # start of scan; metadata is all behind us
            break
        (length,) = struct.unpack(">H", data[cursor + 2:cursor + 4])
        if marker == 0xE1 and data[cursor + 4:cursor + 10] == b"Exif\x00\x00":
            tiff = data[cursor + 10:cursor + 2 + length]
            break
        cursor += 2 + length
    if not tiff or len(tiff) < 8:
        return {}

    endian = "<" if tiff[:2] == b"II" else ">" if tiff[:2] == b"MM" else None
    if endian is None:
        return {}
    (first,) = struct.unpack(endian + "I", tiff[4:8])
    raw = {}
    _parse_ifd(tiff, first, endian, raw)
    if EXIF_SUB_IFD in raw:
        (_, _, payload) = raw[EXIF_SUB_IFD]
        (sub_offset,) = struct.unpack(endian + "I", payload[:4])
        _parse_ifd(tiff, sub_offset, endian, raw)

    def text(tag):
        if tag not in raw:
            return None
        return raw[tag][2].split(b"\x00")[0].decode("ascii", "replace").strip() or None

    def number(tag):
        if tag not in raw:
            return None
        kind, _, payload = raw[tag]
        try:
            if kind == 5 and len(payload) >= 8:
                num, den = struct.unpack(endian + "II", payload[:8])
                return num / den if den else None
            if kind == 3 and len(payload) >= 2:
                return float(struct.unpack(endian + "H", payload[:2])[0])
        except struct.error:
            return None
        return None

    return {"model": text(EXIF_MODEL), "lens": text(EXIF_LENS_MODEL),
            "focal_mm": number(EXIF_FOCAL_LENGTH),
            "focal_35mm": number(EXIF_FOCAL_35MM)}


def report_exif(paths) -> None:
    """Flag a set shot through more than one lens or focal length.

    Intrinsics describe ONE optical configuration. A set that mixes the main
    and ultra-wide cameras, or that a digital zoom crept into, is two cameras
    averaged into one model - and it shows up as a residual no amount of
    reshooting the same way will fix.
    """
    seen = [(path.name, read_exif(path)) for path in paths]
    tagged = [(name, info) for name, info in seen if info and info.get("focal_mm")]
    if not tagged:
        print("exif         : none readable (fine - it is only a cross-check)")
        return

    groups = {}
    for name, info in tagged:
        key = (info.get("lens") or "?", round(info["focal_mm"], 2))
        groups.setdefault(key, []).append(name)

    if len(groups) == 1:
        (lens, focal), names = next(iter(groups.items()))
        print(f"exif         : {len(names)} photos, one lens - {lens} @ {focal} mm")
        return

    print(f"exif         : *** {len(groups)} DIFFERENT OPTICAL CONFIGURATIONS ***")
    for (lens, focal), names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        listing = ", ".join(names[:4]) + (" ..." if len(names) > 4 else "")
        print(f"               {len(names):>3} x {focal:>6} mm  {lens}")
        print(f"                   {listing}")
    print("               Intrinsics describe ONE lens. Keep the largest group and")
    print("               delete the rest, or reshoot without letting the camera")
    print("               switch - do not pinch-zoom, and stay far enough away that")
    print("               macro mode does not cut in.")


def collect_views(paths, detector, marker_map, offsets, min_markers=None):
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
        if len(known) < (min_markers or MIN_MARKERS_PER_VIEW):
            skipped.append((path.name, f"only {len(known)} mapped markers"))
            continue

        obj, img, _ids = starnav.build_correspondences(known, plane_map, offsets)
        object_points.append(obj.astype(np.float32))
        image_points.append(img.astype(np.float32))
        used.append((path.name, len(known)))

    return object_points, image_points, image_size, used, skipped


def collect_chessboard_views(paths, pattern, square_m: float):
    """Detect the chessboard in each photo and build its correspondences.

    findChessboardCornersSB is tried first: it is the sector-based detector, it
    returns corners already at sub-pixel accuracy, and it is both faster and
    more tolerant of blur and uneven lighting than the classic pipeline on the
    large stills a phone produces. The classic detector plus cornerSubPix is
    kept as a fallback for boards it declines.

    Object points sit at Z = 0 in the board's own frame. The square size scales
    the tvecs and nothing else - the intrinsics this script exists to find come
    out identical whatever value is passed.
    """
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

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
            skipped.append((path.name, f"size {size[0]}x{size[1]} != "
                                       f"{image_size[0]}x{image_size[1]}"))
            continue

        found, corners = cv2.findChessboardCornersSB(
            image, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not found:
            found, corners = cv2.findChessboardCorners(
                image, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                corners = cv2.cornerSubPix(
                    image, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        if not found:
            skipped.append((path.name, f"no {cols}x{rows} board found - whole board "
                                       f"must be visible, in focus, unclipped"))
            continue

        object_points.append(grid.copy())
        image_points.append(corners.reshape(-1, 2).astype(np.float32))
        used.append((path.name, cols * rows))

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
    parser.add_argument("--min-markers", type=int, default=None,
                        help=f"markers a view must show to be used (default "
                             f"{MIN_MARKERS_PER_VIEW}). Lower it for a small grid: an "
                             f"oblique view of a 9-tag ceiling often shows only 4-5, and "
                             f"those are exactly the views that make the solve well posed.")
    parser.add_argument("--chessboard", default=None, metavar="COLSxROWS",
                        help="calibrate from a chessboard instead of the tag grid. "
                             "COLSxROWS are INNER corner counts, e.g. 9x6.")
    parser.add_argument("--square-mm", type=float, default=None,
                        help="chessboard square size, as printed by tools/chessboard.py. "
                             "Scales the tvecs only; the intrinsics are unaffected.")
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

    paths = sorted(p for p in args.photos.iterdir()
                   if p.suffix.lower() in starnav.IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"no readable images in {args.photos} "
                         f"(supported: {', '.join(starnav.IMAGE_EXTENSIONS)})")

    if args.chessboard:
        if args.square_mm is None:
            raise SystemExit("--chessboard also needs --square-mm "
                             "(tools/chessboard.py prints it)")
        try:
            pattern = tuple(int(v) for v in args.chessboard.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--chessboard must look like 9x6, got '{args.chessboard}'")
        target = f"chessboard {pattern[0]}x{pattern[1]} inner corners at {args.square_mm} mm"
        object_points, image_points, image_size, used, skipped = collect_chessboard_views(
            paths, pattern, args.square_mm / 1000.0)
    else:
        marker_map = starnav.load_marker_map(args.markers)
        detection_cfg = dict(hall["detection"])
        if args.refine_win:
            detection_cfg["corner_refine_win_size"] = args.refine_win
        detector = starnav.make_detector(detection_cfg)
        offsets = starnav.corner_offsets(marker_map["tag_size_m"] / 2.0)
        target = f"tag grid from {args.markers}"
        object_points, image_points, image_size, used, skipped = collect_views(
            paths, detector, marker_map, offsets, args.min_markers)

    print(f"target       : {target}")
    print(f"photos found : {len(paths)}")
    report_exif(paths)
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    if len(object_points) < 5:
        raise SystemExit(f"only {len(object_points)} usable views; need at least 5, "
                         f"and 15-25 for a trustworthy result")
    print(f"usable views : {len(object_points)} at {image_size[0]}x{image_size[1]}")
    label = "corners/view" if args.chessboard else "markers/view"
    print(f"{label} : min {min(n for _, n in used)}, max {max(n for _, n in used)}")

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
    # Per photo, because "the RMS is bad" is not actionable but "these three
    # photos are bad and these twelve are flat-on" is. tilt_deg is the angle
    # between the camera's aim and the target's face: 0 means the board imaged
    # as a clean rectangle and contributed no shape information at all.
    print(f"\n{'photo':<26}{'corners':>8}{'tilt_deg':>10}{'err_px':>9}   note")
    order = np.argsort(-errors)
    flat = 0
    for view_index in range(len(used)):
        name, corner_count = used[view_index]
        tilt = tilts[view_index]
        note = []
        if tilt < 10.0:
            note.append("FLAT-ON, adds no tilt")
            flat += 1
        if view_index in order[:3] and errors[view_index] > 2 * np.median(errors):
            note.append("worst - consider deleting")
        print(f"{name[:25]:<26}{corner_count:>8}{tilt:>10.1f}{errors[view_index]:>9.3f}"
              f"   {', '.join(note)}")

    print(f"\nRMS reprojection : {rms:.4f} px   (pooled over all corners)")
    print(f"per-view mean    : median {np.median(errors):.3f}, worst {errors.max():.3f} px "
          f"({used[int(np.argmax(errors))][0]}), mean of views {pooled_mean:.3f}")
    print(f"focal length     : fx {focal_x:.1f}, fy {focal_y:.1f} px  "
          f"(aspect {focal_y / focal_x:.4f}, HFOV {hfov:.1f} deg)")
    print(f"principal point  : {camera_matrix[0, 2]:.1f}, {camera_matrix[1, 2]:.1f} "
          f"(centre is {image_size[0] / 2:.0f}, {image_size[1] / 2:.0f})")
    print(f"distortion       : {np.array2string(dist_coeffs.ravel(), precision=4)}")
    print(f"view tilt spread : {tilts.min():.0f} to {tilts.max():.0f} deg")

    if rms <= MAX_ACCEPTABLE_RMS_PX:
        print(f"\nPASS: RMS {rms:.3f} px is within the {MAX_ACCEPTABLE_RMS_PX} px "
              f"threshold.")
    elif rms <= 1.0:
        print(f"\nMARGINAL: RMS {rms:.3f} px is over the {MAX_ACCEPTABLE_RMS_PX} px "
              f"threshold but under 1.0. Check `worst` above - if it sits well above "
              f"the median, a few bad photos are carrying it and deleting those "
              f"usually clears it without a full recapture.")
    else:
        print(f"\nREJECT: RMS {rms:.3f} px is over 1.0 px.")
        # Which cause to name depends on the tilt spread, because the two big
        # failures look nothing alike. Too little tilt gives a LOW residual with
        # a confidently wrong fx; a target that is not planar gives a HIGH
        # residual, because no single set of intrinsics can fit a curved
        # surface. Measured on synthetic sets: a 12 mm bow across a 600 mm panel
        # produces ~1.1 px and biases fx by ~3.7%, and an 1800R curve (~25 mm)
        # produces ~2.2 px and ~8%. Corner noise does NOT do this - heavy blur
        # and noise move the residual by hundredths of a pixel, not whole ones.
        if tilts.max() - tilts.min() >= MIN_TILT_SPREAD_DEG:
            print(f"        Tilt spread is fine ({tilts.max() - tilts.min():.0f} deg), so "
                  f"this is NOT a shooting-angle problem and more photos will not fix "
                  f"it. A residual this size means the model cannot fit the "
                  f"observations, which is systematic. In order of likelihood:")
            print(f"          1. THE TARGET IS NOT FLAT. A curved monitor is the usual "
                  f"culprit - lay a straightedge on the screen and look for a gap. "
                  f"Around {rms:.1f} px corresponds to roughly "
                  f"{rms * 11:.0f} mm of bow, and biases fx by a few percent.")
            print(f"          2. Mixed lenses - a pinch-zoom that switched to tele or "
                  f"a digital crop partway through the set.")
            print(f"          3. HDR, portrait or night mode left on; they warp geometry "
                  f"non-linearly.")
            print(f"        Printing the board on paper and taping it to something flat "
                  f"sidesteps all of this. The square size does not affect the "
                  f"intrinsics, so it need not be exact.")
        else:
            print(f"        Tilt spread is only {tilts.max() - tilts.min():.0f} deg. "
                  f"Fix that first - step to the side and TURN back to aim at the "
                  f"board, so it images as a trapezoid rather than a rectangle.")
        print(f"        Every position error downstream inherits this.")
    if flat > len(used) / 2:
        print(f"\n{flat} of {len(used)} photos are flat-on to the target (tilt under 10 "
              f"deg). Sliding the camera sideways does not create tilt - only TURNING "
              f"it to aim from an angle does. Step to the side, then turn back to aim "
              f"at the board, so it images as a trapezoid rather than a rectangle.")
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
