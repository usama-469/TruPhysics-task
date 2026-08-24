#!/usr/bin/env python3
"""Handedness check with a fixed camera and no tape measure.

The camera never moves. The TAGS move, by an exact number of screen pixels, and
the reported camera position is read at each step. A tag shift is known to a
fraction of a millimetre and costs nothing to repeat, which a hand-held tape
measure is not.

Every step is solved against ONE fixed marker map - the map written at offset
zero. The solver therefore believes the tags never moved, and the only way it
can explain a pattern that slid 100 mm right is a camera that slid 100 mm LEFT:

    reported camera X changes by MINUS the tag shift.

Shifting the tags and regenerating the map to match is a different experiment:
the world frame moves with the tags, the camera really is where it always was,
and the reported X does not change. Both columns print, because the pair of them
is what makes the sign unambiguous.

Two steps, because the camera is a phone rather than a webcam:

    1. python tools/eval_shift.py --display --offsets-mm 0 100 -100
       Puts each tag position on screen at native size. Photograph each one
       WITHOUT MOVING THE CAMERA, press any key to advance. Writes the map.

    2. python tools/eval_shift.py --photos photos/handedness \\
           --offsets-mm 0 100 -100 --assume-hfov-deg 65
       Reads the photos in filename order, pairs them with the offsets in the
       order they were displayed, and prints the verdict.

Handedness without a calibration:
  The SIGN does not depend on the intrinsics. It falls out of the corner
  ordering and the local->world composition, so a guessed focal length gets the
  direction right while getting every millimetre wrong. --assume-hfov-deg builds
  a camera matrix from the photo width and an assumed field of view for exactly
  that case; verified correct for focal lengths from half to double the truth.
  Distances printed in that mode are NOT metric and the tool says so.

One tag or several:
  A single tag answers the SIGN question and nothing else: its pose is
  rotation-ambiguous, and about a degree of rotation error at 0.75 m is over a
  centimetre of lateral error. A 3x2 grid pins the rotation through the shared
  plane and brings the same measurement inside ~2 mm. Prefer the grid; the sign
  is firmer and it costs nothing.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import starnav  # noqa: E402
import screen_tags  # noqa: E402

WINDOW = "starnav - shift test (photograph, then any key)"


def plan_steps(args, geometry):
    """Grid layout plus the achieved shift of each step.

    The requested shift is rounded to whole pixels, so what is displayed is not
    exactly what was asked for. Every comparison uses the achieved value: a
    0.3 mm rounding error is the same size as the effect being measured.
    """
    ruler_px = 0 if args.no_ruler else screen_tags.RULER_STRIP_PX
    tag_px, tag_size_mm = screen_tags.snap_tag_size(args.tag_size_mm, geometry["px_per_mm"])
    layout = screen_tags.build_layout(geometry, tag_px, args.quiet_cells,
                                      None, args.cols, args.rows, ruler_px)
    steps = []
    for wanted_mm in args.offsets_mm:
        offset_px = int(round(wanted_mm * geometry["px_per_mm"]))
        steps.append((offset_px, offset_px / geometry["px_per_mm"]))
    return layout, tag_px, tag_size_mm, steps


def display(args, geometry, dictionary, layout, tag_px, tag_size_mm, steps) -> int:
    """Step through the tag positions, one keypress at a time. No camera needed."""
    # The reference map comes from offset 0 whether or not offset 0 is one of
    # the displayed steps, so every shift below is absolute against one frame.
    _, centres_zero = screen_tags.render_screen(
        dictionary, geometry, layout, tag_px, args.background, (0, 0))
    if not centres_zero:
        raise SystemExit("the tags do not fit on the panel at offset 0")
    args.markers.parent.mkdir(parents=True, exist_ok=True)
    screen_tags.write_marker_map(args.markers, centres_zero, geometry, tag_size_mm,
                                 args.camera_distance_m, args.ruler_tolerance_mm)

    n_tags = len(centres_zero)
    print(f"panel   : {geometry['width_px']}x{geometry['height_px']}, "
          f"{geometry['px_per_mm']:.4f} px/mm ({args.diagonal_in}\" assumed)")
    print(f"tags    : {layout['cols']}x{layout['rows']} = {n_tags} at {tag_size_mm:.2f} mm")
    print(f"map     : {args.markers}")
    print(f"steps   : {[round(mm, 2) for _, mm in steps]} mm\n")
    print("Photograph each screen WITHOUT MOVING THE CAMERA, then press any key.")
    print("Shoot JPEG, not HEIC - OpenCV cannot read HEIC.")
    print("Keep every photo the same orientation and resolution.")
    print("Check the ruler bar measures 100 mm before the first shot.\n")

    for index, (offset_px, actual_mm) in enumerate(steps):
        canvas, centres = screen_tags.render_screen(
            dictionary, geometry, layout, tag_px, args.background, (offset_px, 0))
        if len(centres) != n_tags:
            raise SystemExit(
                f"a shift of {actual_mm:.1f} mm pushes tags off the panel "
                f"({len(centres)} of {n_tags} left). Use a smaller shift, fewer "
                f"columns, or a smaller --tag-size-mm.")
        # The caption sits top-left, clear of every tag and its quiet zone, so
        # it cannot change what is detected.
        cv2.putText(canvas, f"shot {index + 1} of {len(steps)}   shift {actual_mm:+.2f} mm",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, screen_tags.BLACK, 2, cv2.LINE_AA)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW, canvas)
        print(f"  shot {index + 1}/{len(steps)}: shift {actual_mm:+7.2f} mm - "
              f"photograph now, then press any key")
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    offsets = " ".join(str(mm) for mm in args.offsets_mm)
    print(f"\nCopy the {len(steps)} photos into a folder in the order taken, then run:")
    print(f"  python tools/eval_shift.py --photos <folder> --offsets-mm {offsets} "
          f"--assume-hfov-deg 65")
    return 0


def group_photos(paths, n_steps: int):
    """Split photos across the displayed positions, in filename order.

    Filename order is shooting order for any camera that numbers sequentially,
    and it is the only ordering a folder of stills carries. Equal groups mean
    several shots per position are allowed and get averaged.
    """
    if len(paths) % n_steps:
        raise SystemExit(
            f"{len(paths)} photos do not divide evenly across {n_steps} positions. "
            f"Shoot the same number at each, and remove any extras.")
    per_step = len(paths) // n_steps
    return [paths[i * per_step:(i + 1) * per_step] for i in range(n_steps)]


def solve_group(paths, detector, marker_map, camera_matrix, dist_coeffs, offsets,
                expected_size):
    """Median pose over the photos taken at one tag position.

    The median, not the mean: one photo that drops a corner produces an outlier
    metres away, and a single one of those would drag a mean far enough to hide
    a real result.
    """
    known_ids = set(marker_map["markers"])
    xs, ys, zs, residuals, counts = [], [], [], [], []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    {path.name}: unreadable (HEIC? shoot JPEG), skipped")
            continue
        size = (image.shape[1], image.shape[0])
        if size != expected_size:
            # Intrinsics belong to one sensor readout; mixing sizes or
            # orientations makes the pixel coordinates mean different things.
            print(f"    {path.name}: {size[0]}x{size[1]} != "
                  f"{expected_size[0]}x{expected_size[1]}, skipped")
            continue
        corners, ids, _rejected = detector.detectMarkers(image)
        known, _unknown = starnav.split_known_unknown(corners, ids, known_ids)
        pose = starnav.pose_from_detections(known, marker_map, camera_matrix,
                                            dist_coeffs, offsets)
        if pose is None:
            print(f"    {path.name}: no mapped markers detected, skipped")
            continue
        xs.append(pose["x"])
        ys.append(pose["y"])
        zs.append(pose["z"])
        residuals.append(pose["reproj_px"])
        counts.append(pose["n_markers"])

    if not xs:
        return None
    return {
        "n": len(xs),
        "x": float(np.median(xs)), "y": float(np.median(ys)), "z": float(np.median(zs)),
        "sigma_x_mm": float(np.std(xs)) * 1000.0,
        "markers": int(np.median(counts)),
        "reproj_px": float(np.median(residuals)),
    }


def intrinsics_for(args, photo_size):
    """Real intrinsics if calibrated, otherwise a guess good enough for the sign."""
    if args.calib.exists():
        camera_matrix, dist_coeffs, calib_size = starnav.load_calibration(args.calib)
        if calib_size and tuple(calib_size) != photo_size:
            raise SystemExit(
                f"calib.npz was captured at {calib_size[0]}x{calib_size[1]} but these "
                f"photos are {photo_size[0]}x{photo_size[1]}. Intrinsics belong to one "
                f"sensor readout; recalibrate, or pass --assume-hfov-deg for sign only.")
        return camera_matrix, dist_coeffs, True

    if args.assume_hfov_deg is None:
        raise SystemExit(
            f"{args.calib} not found. Run tools/calibrate.py for metric results, or "
            f"pass --assume-hfov-deg for a sign-only check that does not need it.")
    focal = (photo_size[0] / 2.0) / math.tan(math.radians(args.assume_hfov_deg) / 2.0)
    # Principal point assumed dead centre and distortion assumed zero. Both are
    # wrong on any real lens, and neither changes which DIRECTION the reported
    # position moves - the only claim this mode makes.
    camera_matrix = np.array([[focal, 0.0, photo_size[0] / 2.0],
                              [0.0, focal, photo_size[1] / 2.0],
                              [0.0, 0.0, 1.0]])
    return camera_matrix, np.zeros(5), False


def analyse(args, steps) -> int:
    marker_map = starnav.load_marker_map(args.markers)
    offsets = starnav.corner_offsets(marker_map["tag_size_m"] / 2.0)

    paths = sorted(p for p in args.photos.iterdir()
                   if p.suffix.lower() in starnav.IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(
            f"no readable images in {args.photos}. iPhone shoots HEIC by default and "
            f"OpenCV cannot read it - Settings > Camera > Formats > Most Compatible.")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise SystemExit(f"could not read {paths[0]} (HEIC? shoot JPEG)")
    photo_size = (first.shape[1], first.shape[0])
    camera_matrix, dist_coeffs, metric = intrinsics_for(args, photo_size)

    detection_cfg = dict(starnav.load_json(args.hall)["detection"])
    if args.refine_win:
        detection_cfg["corner_refine_win_size"] = args.refine_win
    detector = starnav.make_detector(detection_cfg)

    print(f"photos  : {len(paths)} in {args.photos}, {photo_size[0]}x{photo_size[1]}")
    print(f"map     : {args.markers} ({len(marker_map['markers'])} tags at "
          f"{marker_map['tag_size_m'] * 1000:.2f} mm)")
    if metric:
        print(f"calib   : {args.calib}")
    else:
        print(f"calib   : NONE - assuming {args.assume_hfov_deg:.0f} deg HFOV, "
              f"f={camera_matrix[0, 0]:.0f} px")
        print("          SIGN-ONLY MODE. Directions are trustworthy, distances are NOT.")
    print()

    results = []
    for (_, actual_mm), group in zip(steps, group_photos(paths, len(steps))):
        print(f"  shift {actual_mm:+7.2f} mm: {', '.join(p.name for p in group)}")
        sample = solve_group(group, detector, marker_map, camera_matrix, dist_coeffs,
                             offsets, photo_size)
        if sample is None:
            raise SystemExit(
                f"no usable photo at shift {actual_mm:+.1f} mm. Check framing, focus "
                f"and glare on the panel.")
        sample["shift_mm"] = actual_mm
        results.append(sample)

    print()
    report(results, metric)
    return 0


def report(results, metric: bool = True) -> None:
    base_x_mm = results[0]["x"] * 1000.0
    base_shift = results[0]["shift_mm"]

    print(f"{'shift_mm':>9}{'n':>4}{'tags':>6}{'x_mm':>10}{'dx_mm':>9}{'expected':>10}"
          f"{'error_mm':>10}{'x_upd_mm':>10}{'reproj':>8}")
    for row in results:
        x_mm = row["x"] * 1000.0
        dx = x_mm - base_x_mm
        expected = -(row["shift_mm"] - base_shift)
        # X as it would read against a map REGENERATED at this offset: the tags
        # moved in the world too, so the camera should sit exactly where it did
        # at the first step. This column is the control - it should not move.
        x_updated = x_mm + row["shift_mm"]
        print(f"{row['shift_mm']:>9.2f}{row['n']:>4}{row['markers']:>6}{x_mm:>10.2f}"
              f"{dx:>9.2f}{expected:>10.2f}{dx - expected:>10.2f}{x_updated:>10.2f}"
              f"{row['reproj_px']:>8.3f}")

    z_list = ", ".join(f"{row['z'] * 1000.0:.1f}" for row in results)
    print(f"\nZ (panel distance) : {z_list} mm")

    # The verdict this whole script exists for. Every step whose shift differs
    # from the baseline votes on the direction, and they must all agree.
    votes = [(row["shift_mm"] - base_shift, row["x"] * 1000.0 - base_x_mm)
             for row in results[1:] if abs(row["shift_mm"] - base_shift) > 1.0]
    if votes:
        agreed = (all(dx < 0 for shift, dx in votes if shift > 0)
                  and all(dx > 0 for shift, dx in votes if shift < 0))
        print("\nHANDEDNESS: shifting the tags +X moved reported camera X "
              + ("NEGATIVE, as expected." if agreed else "the WRONG WAY."))
        if agreed:
            print("            World X is not mirrored. corner_offsets rotation 0, "
                  "mirror False holds for this rig.")
        else:
            print("            MIRRORED. Either the map's axis or the physical "
                  "mounting runs opposite to what starnav.corner_offsets assumes; "
                  "re-run tools/eval_photos.py to search all 8 conventions.")

    if not metric:
        print("\nRun without a calibration: the direction above is the result. "
              "Ignore every millimetre on this page.")
    print("\nSign recorded here is the handedness result. Re-verify it with tags "
          "overhead before trusting it in the hall.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="tag-shift handedness check: PC displays, phone photographs")
    parser.add_argument("--display", action="store_true",
                        help="step through the tag positions for photographing")
    parser.add_argument("--photos", type=Path, default=None,
                        help="folder of photos to analyse, in shooting order")
    parser.add_argument("--offsets-mm", type=float, nargs="+", default=[0.0, 100.0, -100.0],
                        help="tag X shifts in mm; must match between the two runs")
    parser.add_argument("--markers", type=Path, default=Path("config/markers_shift.json"),
                        help="reference map, written by --display at offset 0")
    parser.add_argument("--hall", type=Path, default=Path("config/hall.json"))
    parser.add_argument("--calib", type=Path, default=Path("calib.npz"))
    parser.add_argument("--assume-hfov-deg", type=float, default=None,
                        help="analyse without calib.npz for a SIGN-ONLY check, using a "
                             "camera matrix built from this horizontal field of view "
                             "(iPhone main camera ~65). Distances are then not metric.")
    parser.add_argument("--tag-size-mm", type=float, default=90.0)
    parser.add_argument("--cols", type=int, default=3,
                        help="grid columns (default 3). A grid pins rotation; one tag "
                             "reads the sign but wanders in magnitude.")
    parser.add_argument("--rows", type=int, default=2, help="grid rows (default 2)")
    parser.add_argument("--camera-distance-m", type=float, default=0.0,
                        help="written as ceiling_height_m; 0 makes the solved camera Z "
                             "read the panel distance directly")
    parser.add_argument("--diagonal-in", type=float, default=27.0)
    parser.add_argument("--resolution", default=None, help="WxH, overrides auto-detection")
    parser.add_argument("--refine-win", type=int, default=None,
                        help="corner refinement window; raise to ~9 for large stills")
    parser.add_argument("--background", type=int, default=screen_tags.DEFAULT_BACKGROUND)
    parser.add_argument("--quiet-cells", type=float, default=screen_tags.DEFAULT_QUIET_CELLS)
    parser.add_argument("--ruler-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--no-ruler", action="store_true")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.display == bool(args.photos):
        raise SystemExit("pass exactly one of --display (to shoot) or --photos (to analyse)")

    if args.resolution:
        width_px, height_px = (int(v) for v in args.resolution.lower().split("x"))
    elif args.display:
        detected = screen_tags.detect_screen_px()
        if detected is None:
            raise SystemExit("could not detect the screen; pass --resolution WxH")
        width_px, height_px = detected
    else:
        # Analysis needs only the step list; the map already carries the
        # geometry that mattered, and the same rounding reproduces the shifts.
        width_px, height_px = 1920, 1080

    geometry = screen_tags.screen_geometry(width_px, height_px, args.diagonal_in)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    layout, tag_px, tag_size_mm, steps = plan_steps(args, geometry)

    if args.display:
        return display(args, geometry, dictionary, layout, tag_px, tag_size_mm, steps)
    return analyse(args, steps)


if __name__ == "__main__":
    raise SystemExit(main())
