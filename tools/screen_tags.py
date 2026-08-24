#!/usr/bin/env python3
"""Display an ArUco tag grid on a monitor at an exact physical size.

The monitor stands in for the ceiling. This is strictly better than tape and
paper for testing the *algorithm*, because the marker positions are addressed
in screen pixels and are therefore exact to a fraction of a millimetre - the
survey error row of the error budget drops to nearly zero, so whatever jitter
is left is the algorithm's own. It is strictly worse for testing the *system*,
because the range is 1 m rather than 12 m and every range-dependent error term
shrinks with it. See the feasibility notes at the bottom of this file.

World frame for the screen rig (right-handed, and identical in structure to the
hall frame so `starnav.py` needs no changes):

    X - screen right
    Y - screen DOWN
    Z - into the screen, i.e. from the camera towards the markers

Y points down the screen rather than up because the frame must be right-handed
with +Z running from the camera to the marker plane, exactly as the hall frame
has +Z running from the floor-level camera up to the ceiling. Getting this
wrong is the handedness trap: mirror Z and the object-point winding reverses,
and X comes out negated. With this frame, marker world XY is simply the screen
position in metres, and the mapping transfers to a real ceiling unchanged as
long as the printed tags are taped up in the same rotational orientation they
are rendered in here.

Usage:
    python tools/screen_tags.py                       # detect screen, show fullscreen
    python tools/screen_tags.py --tag-size-mm 40      # smaller tags, denser grid
    python tools/screen_tags.py --offset-mm 100 0     # exact 100 mm shift, scale test
    python tools/screen_tags.py --no-show             # write PNG + JSON only
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path

import cv2
import numpy as np

MM_PER_INCH = 25.4
WHITE = 255
BLACK = 0

# DICT_4X4 tags are 4 data cells plus a 1-cell black border on each side.
DICT_4X4_CELLS = 6

# ArUco needs a white quiet zone to find the tag's outer contour; one bit-cell
# is the documented minimum. On a screen there is no tape to eat the border, so
# the minimum is enough and the saved space goes into more tags.
DEFAULT_QUIET_CELLS = 1

# The real system is a 0.72 m tag at 12 m. Everything about pose conditioning
# follows from that ratio, not from the absolute numbers, so the recommended
# camera distance is chosen to reproduce it.
REFERENCE_TAG_OVER_RANGE = 0.72 / 12.0

# Background outside each tag's quiet zone. Mid-grey rather than white: a
# full-screen white panel at 1 m drives the camera's auto-exposure down and
# blooms the tag edges, which biases sub-pixel corner refinement. Each tag
# still sits on its own white pad, so the quiet zone is preserved.
DEFAULT_BACKGROUND = 110

# Ruler drawn at the bottom of the screen so the assumed px/mm can be checked
# against a physical ruler. 100 mm is long enough to make a 1 % error visible.
RULER_LENGTH_MM = 100.0
RULER_STRIP_PX = 90


def detect_screen_px():
    """Physical pixel size of the primary display, or None if not detectable.

    Process DPI awareness has to be set first: without it Windows reports
    *logical* pixels scaled by the display setting, so a 150 % scaled 1920-wide
    panel reports 1280 and every millimetre downstream is 1.5x wrong.
    """
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        return None


def screen_geometry(width_px: int, height_px: int, diagonal_in: float) -> dict:
    """Physical size of the panel from its diagonal and pixel aspect ratio.

    The aspect ratio is taken from the resolution rather than assumed to be
    16:9, so 16:10 and ultrawide panels come out right.
    """
    diagonal_mm = diagonal_in * MM_PER_INCH
    aspect = math.hypot(width_px, height_px)
    width_mm = diagonal_mm * width_px / aspect
    height_mm = diagonal_mm * height_px / aspect
    return {
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "px_per_mm": width_px / width_mm,
    }


def snap_tag_size(tag_size_mm: float, px_per_mm: float) -> tuple[int, float]:
    """Round the tag to a whole number of screen pixels per bit-cell.

    A tag whose cell size is not an integer number of screen pixels gets its
    cell boundaries resampled by the renderer, which softens the black/white
    transitions unevenly across the tag and puts a systematic bias into
    sub-pixel corner refinement. Snapping to a multiple of DICT_4X4_CELLS makes
    every cell an identical block of screen pixels with no interpolation
    anywhere, so the displayed edges are as sharp as the panel allows.
    """
    wanted_px = tag_size_mm * px_per_mm
    cell_px = max(1, int(round(wanted_px / DICT_4X4_CELLS)))
    tag_px = cell_px * DICT_4X4_CELLS
    return tag_px, tag_px / px_per_mm


def build_layout(geometry: dict, tag_px: int, quiet_cells: float,
                 pitch_mm: float | None, cols: int | None, rows: int | None,
                 ruler_px: int = RULER_STRIP_PX) -> dict:
    """Work out how many tags fit and where their centres land."""
    cell_px = tag_px // DICT_4X4_CELLS
    pad_px = int(round(cell_px * quiet_cells))
    footprint_px = tag_px + 2 * pad_px

    pitch_px = footprint_px if pitch_mm is None else int(round(pitch_mm * geometry["px_per_mm"]))
    if pitch_px < footprint_px:
        raise ValueError(
            f"pitch {pitch_mm} mm is smaller than one tag plus its quiet zone "
            f"({footprint_px / geometry['px_per_mm']:.1f} mm) - the quiet zones would overlap"
        )

    # The ruler strip is reserved out of the usable height, not overlaid, so it
    # can never clip a quiet zone.
    usable_h = geometry["height_px"] - ruler_px
    fit_cols = max(1, (geometry["width_px"] - footprint_px) // pitch_px + 1)
    fit_rows = max(1, (usable_h - footprint_px) // pitch_px + 1)
    cols = fit_cols if cols is None else min(cols, fit_cols)
    rows = fit_rows if rows is None else min(rows, fit_rows)

    grid_w = (cols - 1) * pitch_px + footprint_px
    grid_h = (rows - 1) * pitch_px + footprint_px
    origin_x = (geometry["width_px"] - grid_w) // 2
    origin_y = (usable_h - grid_h) // 2

    return {
        "cols": int(cols), "rows": int(rows),
        "cell_px": cell_px, "pad_px": pad_px, "footprint_px": footprint_px,
        "pitch_px": int(pitch_px), "ruler_px": int(ruler_px),
        "origin_x": int(origin_x), "origin_y": int(origin_y),
    }


def best_fill(geometry: dict, quiet_cells: float, capacity: int, ruler_px: int,
              min_cell_px: int = 6) -> int:
    """Largest tag size that packs the most tags onto the panel.

    "Fill the screen" is a trade-off, not a maximum: shrinking the tags fits
    more of them, but every tag also has to survive being photographed, so the
    useful answer is the *largest* tag size that still reaches the highest tag
    count the dictionary can address. Search descending and keep the first
    candidate that ties the best count, which is by construction the biggest.

    The floor of 6 px per bit-cell is a detection limit, not an aesthetic one:
    below roughly 3-4 screen pixels per cell the panel's own subpixel structure
    starts to blur cell boundaries into each other.
    """
    largest_cell = geometry["height_px"] // DICT_4X4_CELLS
    best_count, best_cell = 0, min_cell_px
    for cell_px in range(largest_cell, min_cell_px - 1, -1):
        layout = build_layout(geometry, cell_px * DICT_4X4_CELLS, quiet_cells,
                              None, None, None, ruler_px)
        count = layout["cols"] * layout["rows"]
        if count > capacity:
            continue
        if count > best_count:
            best_count, best_cell = count, cell_px
    return best_cell * DICT_4X4_CELLS


def render_screen(dictionary, geometry: dict, layout: dict, tag_px: int,
                  background: int, offset_px: tuple[int, int]):
    """Draw the full-screen image and return it with each tag's centre in px."""
    canvas = np.full((geometry["height_px"], geometry["width_px"]), background, np.uint8)
    centres = {}

    marker_id = 0
    for row in range(layout["rows"]):
        for col in range(layout["cols"]):
            x0 = layout["origin_x"] + col * layout["pitch_px"] + offset_px[0]
            y0 = layout["origin_y"] + row * layout["pitch_px"] + offset_px[1]
            pad = layout["pad_px"]
            # Skip anything the offset has pushed off-screen rather than
            # clipping it: a half-drawn tag is either undetectable or, worse,
            # detectable with a corrupted corner.
            if x0 < 0 or y0 < 0 or x0 + layout["footprint_px"] > geometry["width_px"] \
                    or y0 + layout["footprint_px"] > geometry["height_px"] - layout["ruler_px"]:
                marker_id += 1
                continue

            canvas[y0:y0 + layout["footprint_px"], x0:x0 + layout["footprint_px"]] = WHITE
            tag = cv2.aruco.generateImageMarker(dictionary, marker_id, tag_px)
            canvas[y0 + pad:y0 + pad + tag_px, x0 + pad:x0 + pad + tag_px] = tag
            # Centre of the black square - the point markers.json stores.
            centres[marker_id] = (x0 + pad + tag_px / 2.0, y0 + pad + tag_px / 2.0)
            marker_id += 1

    if layout["ruler_px"]:
        _draw_ruler(canvas, geometry, layout["ruler_px"])
    return canvas, centres


def _draw_ruler(canvas, geometry: dict, strip_px: int) -> None:
    """A known-length bar for checking px/mm against a physical ruler.

    Everything downstream scales linearly with px_per_mm. If the panel is not
    the assumed diagonal, or the OS is scaling the window, this bar will not
    measure 100 mm and every reported position is wrong by the same ratio.
    """
    length_px = int(round(RULER_LENGTH_MM * geometry["px_per_mm"]))
    y = geometry["height_px"] - strip_px // 2
    x0 = (geometry["width_px"] - length_px) // 2
    cv2.rectangle(canvas, (x0 - 20, y - 34), (x0 + length_px + 20, y + 26), WHITE, -1)
    cv2.line(canvas, (x0, y), (x0 + length_px, y), BLACK, 2)
    for x in (x0, x0 + length_px):
        cv2.line(canvas, (x, y - 14), (x, y + 14), BLACK, 2)
    cv2.putText(canvas, f"{RULER_LENGTH_MM:.0f}.0 mm - measure this", (x0, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1, cv2.LINE_AA)


def write_marker_map(path: Path, centres: dict, geometry: dict, tag_size_m: float,
                     camera_distance_m: float, ruler_tolerance_mm: float) -> dict:
    """Write a markers.json in the same schema starnav.py already reads.

    World XY is the screen position in metres with the origin at the top-left
    of the active area, X right and Y down (see the module docstring). No code
    in the pipeline changes for the screen rig - only this file does.
    """
    mm_per_px = 1.0 / geometry["px_per_mm"]
    markers = {
        str(marker_id): [round(cx * mm_per_px / 1000.0, 6), round(cy * mm_per_px / 1000.0, 6)]
        for marker_id, (cx, cy) in sorted(centres.items())
    }
    payload = {
        "tag_size_m": round(tag_size_m / 1000.0, 6),
        # The marker plane sits at Z = camera_distance, so a camera placed at
        # exactly that distance solves to Z ~ 0. Any residual in the reported
        # Z is range error, which makes it a free scale check.
        "ceiling_height_m": camera_distance_m,
        # Screen pixel addressing is exact; the real uncertainty is whether the
        # panel is the size we assumed, which the on-screen ruler bounds.
        "survey_uncertainty_m": round(ruler_tolerance_mm / RULER_LENGTH_MM
                                      * geometry["width_mm"] / 1000.0, 6),
        "markers": markers,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def report(geometry: dict, layout: dict, tag_size_mm: float, camera_distance_m: float,
           n_tags: int, camera_width_px: int, camera_hfov_deg: float) -> None:
    px_per_mm = geometry["px_per_mm"]
    pitch_mm = layout["pitch_px"] / px_per_mm
    spread_x_mm = (layout["cols"] - 1) * pitch_mm
    spread_y_mm = (layout["rows"] - 1) * pitch_mm

    print(f"panel        : {geometry['width_px']}x{geometry['height_px']} px, "
          f"{geometry['width_mm']:.1f} x {geometry['height_mm']:.1f} mm "
          f"({px_per_mm:.3f} px/mm)")
    print(f"tag          : {tag_size_mm:.2f} mm = {layout['cell_px'] * DICT_4X4_CELLS} px "
          f"({layout['cell_px']} px per bit-cell, no resampling)")
    print(f"grid         : {layout['cols']} x {layout['rows']} = {n_tags} tags, "
          f"pitch {pitch_mm:.1f} mm, spread {spread_x_mm:.0f} x {spread_y_mm:.0f} mm")

    matched_m = tag_size_mm / 1000.0 / REFERENCE_TAG_OVER_RANGE
    print(f"\nangular match to the 12 m hall (0.72 m tag / 12 m = {REFERENCE_TAG_OVER_RANGE:.3f}):")
    print(f"  put the camera {matched_m:.2f} m from the screen")

    # Focal length implied by the camera's horizontal field of view, used only
    # to predict how many pixels a tag will occupy. It is a sanity figure, not
    # a substitute for calibration.
    focal_px = (camera_width_px / 2.0) / math.tan(math.radians(camera_hfov_deg) / 2.0)
    tag_px_at_range = focal_px * (tag_size_mm / 1000.0) / matched_m
    view_w_mm = 2000.0 * matched_m * math.tan(math.radians(camera_hfov_deg) / 2.0)
    print(f"  at {camera_width_px}px / {camera_hfov_deg:.0f}deg HFOV that is "
          f"{tag_px_at_range:.0f} px per tag ({tag_px_at_range / DICT_4X4_CELLS:.1f} px per cell)")
    print(f"  field of view there is {view_w_mm:.0f} mm wide vs a "
          f"{geometry['width_mm']:.0f} mm panel "
          f"({'whole panel visible' if view_w_mm > geometry['width_mm'] else 'PANEL WILL BE CROPPED'})")
    print(f"  baseline/range = {spread_x_mm / 1000.0 / matched_m:.2f} "
          f"(hall: ~5 m of tags at 12 m = 0.42)")

    print(f"\ncheck before trusting any number:")
    print(f"  1. the on-screen bar measures {RULER_LENGTH_MM:.0f} mm with a physical ruler")
    print(f"  2. the panel is FLAT, not curved - a 1800R curve bows the centre ~25 mm "
          f"out of plane and breaks the coplanar-marker assumption outright")
    print(f"  3. calibration (calib.npz) was captured with this same camera at a "
          f"similar range, through the same capture path")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ArUco tag grid displayed at an exact physical size")
    parser.add_argument("--diagonal-in", type=float, default=27.0, help="panel diagonal in inches")
    parser.add_argument("--resolution", default=None,
                        help="WxH physical pixels; auto-detected if omitted")
    parser.add_argument("--tag-size-mm", type=float, default=60.0,
                        help="target tag edge; snapped to a whole number of px per bit-cell")
    parser.add_argument("--fill", action="store_true",
                        help="ignore --tag-size-mm and pack as many tags onto the panel as "
                             "the dictionary can address, using the largest tag that does so")
    parser.add_argument("--no-ruler", action="store_true",
                        help="drop the ruler strip and use the full panel height")
    parser.add_argument("--pitch-mm", type=float, default=None,
                        help="centre-to-centre spacing (default: as tight as the quiet zones allow)")
    parser.add_argument("--cols", type=int, default=None, help="cap the column count")
    parser.add_argument("--rows", type=int, default=None, help="cap the row count")
    parser.add_argument("--quiet-cells", type=float, default=DEFAULT_QUIET_CELLS,
                        help="white border in bit-cells (1 is the ArUco minimum)")
    parser.add_argument("--background", type=int, default=DEFAULT_BACKGROUND,
                        help="grey level outside the tags, 0-255 (lower = less bloom)")
    parser.add_argument("--camera-distance-m", type=float, default=1.0,
                        help="intended camera-to-screen distance, written as ceiling_height_m")
    parser.add_argument("--offset-mm", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("DX", "DY"),
                        help="shift the whole pattern by an exact amount (scale/translation test)")
    parser.add_argument("--ruler-tolerance-mm", type=float, default=0.5,
                        help="how precisely you can read the on-screen ruler; sets survey_uncertainty_m")
    parser.add_argument("--camera-width-px", type=int, default=1280,
                        help="camera resolution, for the predicted tag size only")
    parser.add_argument("--camera-hfov-deg", type=float, default=60.0,
                        help="camera horizontal field of view, for the prediction only")
    parser.add_argument("--dictionary", default="DICT_4X4_50",
                        help="ArUco dictionary; its size caps how many tags fit")
    parser.add_argument("--out-map", type=Path, default=Path("config/markers_screen.json"),
                        help="marker map written for the rest of the pipeline")
    parser.add_argument("--out-png", type=Path, default=Path("tags/screen_grid.png"),
                        help="copy of the displayed image, for the report")
    parser.add_argument("--no-show", action="store_true", help="write the files and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.resolution:
        width_px, height_px = (int(v) for v in args.resolution.lower().split("x"))
    else:
        detected = detect_screen_px()
        if detected is None:
            raise SystemExit("could not detect the screen; pass --resolution WxH")
        width_px, height_px = detected

    geometry = screen_geometry(width_px, height_px, args.diagonal_in)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    capacity = len(dictionary.bytesList)
    ruler_px = 0 if args.no_ruler else RULER_STRIP_PX

    if args.fill:
        tag_px = best_fill(geometry, args.quiet_cells, capacity, ruler_px)
        tag_size_mm = tag_px / geometry["px_per_mm"]
    else:
        tag_px, tag_size_mm = snap_tag_size(args.tag_size_mm, geometry["px_per_mm"])

    layout = build_layout(geometry, tag_px, args.quiet_cells, args.pitch_mm,
                          args.cols, args.rows, ruler_px)
    needed = layout["cols"] * layout["rows"]
    if needed > capacity:
        raise SystemExit(
            f"grid needs {needed} IDs but {args.dictionary} only has {capacity}; "
            f"use --cols/--rows, a larger --tag-size-mm, or a bigger dictionary"
        )

    offset_px = tuple(int(round(v * geometry["px_per_mm"])) for v in args.offset_mm)
    canvas, centres = render_screen(dictionary, geometry, layout, tag_px,
                                    args.background, offset_px)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_png), canvas)
    write_marker_map(args.out_map, centres, geometry, tag_size_mm,
                     args.camera_distance_m, args.ruler_tolerance_mm)

    report(geometry, layout, tag_size_mm, args.camera_distance_m, len(centres),
           args.camera_width_px, args.camera_hfov_deg)
    if any(args.offset_mm):
        print(f"\npattern shifted by {args.offset_mm[0]:+.1f}, {args.offset_mm[1]:+.1f} mm "
              f"({offset_px[0]:+d}, {offset_px[1]:+d} px). The reported camera XY must move "
              f"by the same amount with the opposite sign.")
    print(f"\nwrote {args.out_png} and {args.out_map}")

    if args.no_show:
        return 0

    # Displayed through a fullscreen OpenCV window rather than an image viewer:
    # a viewer will fit-to-window and resample, which changes the physical tag
    # size without saying so. Fullscreen at native resolution is 1:1.
    window = "starnav - screen tags (any key to close)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
