#!/usr/bin/env python3
"""Display a chessboard on the monitor at a known square size, for calibration.

Shown through a fullscreen OpenCV window at native resolution rather than saved
and opened in a viewer, for the same reason the tag grid is: a viewer will
fit-to-window and resample, which changes the physical square size without
saying so. The squares are snapped to a whole number of screen pixels, so no
square is resampled unevenly across its face - uneven resampling puts a
systematic bias into sub-pixel corner refinement, which is the one thing
calibration cannot tolerate.

The square size in mm is printed, but note what it does and does not affect:
scaling the object points scales only the recovered tvecs. fx, fy, cx, cy and
the distortion coefficients come out identical whatever you claim the squares
measure. So a wrong panel diagonal costs nothing HERE - it costs later, in
markers.json, where tag_size_m sets the scale of every reported position.

Usage:
    python tools/chessboard.py                       # 9x6 inner corners
    python tools/chessboard.py --inner 7x5 --margin-mm 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_tags  # noqa: E402

WINDOW = "starnav - chessboard (any key to close)"


def build_board(geometry: dict, inner_cols: int, inner_rows: int, margin_mm: float):
    """Render the board centred on the panel, squares snapped to whole pixels.

    OpenCV counts INNER corners, so a 9x6 board has 10x7 squares. The count is
    deliberately not square: an NxN board is rotationally ambiguous by 90
    degrees and the corner ordering can flip between views, which quietly
    corrupts the correspondences.
    """
    if inner_cols == inner_rows:
        raise SystemExit(
            f"--inner {inner_cols}x{inner_rows} is square and therefore rotationally "
            f"ambiguous; use different counts, e.g. 9x6")

    squares_x, squares_y = inner_cols + 1, inner_rows + 1
    margin_px = int(round(margin_mm * geometry["px_per_mm"]))
    available_w = geometry["width_px"] - 2 * margin_px
    available_h = geometry["height_px"] - 2 * margin_px
    square_px = int(min(available_w // squares_x, available_h // squares_y))
    if square_px < 10:
        raise SystemExit(
            f"squares would be {square_px} px - too small to localise. Use fewer "
            f"inner corners or a smaller --margin-mm.")

    board_w = square_px * squares_x
    board_h = square_px * squares_y
    # White page, black squares. findChessboardCorners needs a light quiet zone
    # all the way around the board, which the centring margin provides.
    canvas = np.full((geometry["height_px"], geometry["width_px"]),
                     screen_tags.WHITE, np.uint8)
    origin_x = (geometry["width_px"] - board_w) // 2
    origin_y = (geometry["height_px"] - board_h) // 2
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2:
                continue
            y0 = origin_y + row * square_px
            x0 = origin_x + col * square_px
            canvas[y0:y0 + square_px, x0:x0 + square_px] = screen_tags.BLACK

    return canvas, square_px, (board_w, board_h)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="display a chessboard for calibration")
    parser.add_argument("--inner", default="9x6",
                        help="INNER corner counts, COLSxROWS (default 9x6). OpenCV counts "
                             "inner corners, so 9x6 draws 10x7 squares.")
    parser.add_argument("--margin-mm", type=float, default=20.0,
                        help="white border around the board; the board needs a light "
                             "quiet zone to be found at all")
    parser.add_argument("--diagonal-in", type=float, default=27.0)
    parser.add_argument("--resolution", default=None, help="WxH, overrides auto-detection")
    parser.add_argument("--out-png", type=Path, default=Path("tags/chessboard.png"),
                        help="copy of what was displayed, for the report")
    parser.add_argument("--no-show", action="store_true", help="write the file and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        inner_cols, inner_rows = (int(v) for v in args.inner.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--inner must look like 9x6, got '{args.inner}'")

    if args.resolution:
        width_px, height_px = (int(v) for v in args.resolution.lower().split("x"))
    else:
        detected = screen_tags.detect_screen_px()
        if detected is None:
            raise SystemExit("could not detect the screen; pass --resolution WxH")
        width_px, height_px = detected

    geometry = screen_tags.screen_geometry(width_px, height_px, args.diagonal_in)
    canvas, square_px, (board_w, board_h) = build_board(
        geometry, inner_cols, inner_rows, args.margin_mm)
    square_mm = square_px / geometry["px_per_mm"]

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_png), canvas)

    print(f"panel        : {width_px}x{height_px} px, "
          f"{geometry['width_mm']:.1f} x {geometry['height_mm']:.1f} mm "
          f"({geometry['px_per_mm']:.4f} px/mm)")
    print(f"board        : {inner_cols}x{inner_rows} inner corners "
          f"= {inner_cols + 1}x{inner_rows + 1} squares")
    print(f"square       : {square_mm:.3f} mm = {square_px} px (whole pixels, no resampling)")
    print(f"board size   : {board_w / geometry['px_per_mm']:.1f} x "
          f"{board_h / geometry['px_per_mm']:.1f} mm")
    print(f"wrote        : {args.out_png}")

    print("\nShoot 15+ photos, then:")
    print(f"  python tools/calibrate.py --photos photos/calib_v2 "
          f"--chessboard {inner_cols}x{inner_rows} --square-mm {square_mm:.3f}")
    print("\nVary the TILT more than the count. A planar target shot only face-on")
    print("cannot separate focal length from distance - they trade off exactly.")
    print("Aim for face-on out to 40-50 degrees, in several directions, plus a")
    print("couple of rolls, and push the board into each corner of the frame.")
    print("A plain chessboard must be FULLY visible in every shot, so get corner")
    print("coverage by moving the camera, not by letting the board run off frame.")
    print("\nSame capture path as the demo: one lens at 1x, JPEG not HEIC, HDR and")
    print("portrait mode off, one resolution and orientation throughout.")

    if args.no_show:
        return 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
