#!/usr/bin/env python3
"""Generate printable ArUco tags at an exact physical size.

The pose solver's scale comes entirely from `tag_size_m`. If the printed tag is
5 % smaller than the config says, every reported position is 5 % wrong in range
and nothing in the pipeline can detect it. So this script sizes the image in
pixels from the requested physical size and DPI, and prints the numbers needed
to check the result against a ruler after printing.

Usage:
    python tools/generate_tags.py                       # every ID in markers.json
    python tools/generate_tags.py --ids 0 1 2 --dpi 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MM_PER_INCH = 25.4
WHITE = 255
BLACK = 0

# ArUco needs a white quiet zone around the tag to find its outer contour. One
# bit-cell is the documented minimum; DICT_4X4 tags are 4 data bits plus a
# 1-bit black border on each side, so the tag is 6 cells wide and one quiet
# cell is tag_size/6. Wider is better - the sheet gets taped to a ceiling and
# tape tends to eat the border - but a 150 mm tag plus two quiet cells is
# 250 mm and no longer fits A4, so the default is one cell and the width is
# reported in mm so the operator can check it against their paper.
DEFAULT_QUIET_ZONE_CELLS = 1
DICT_4X4_CELLS = 6

# A4 minus a typical 5 mm unprintable margin. Only used to warn.
A4_PRINTABLE_MM = (200.0, 287.0)
A4_FULL_MM = (210.0, 297.0)

LABEL_HEIGHT_PX_PER_INCH = 0.45  # label strip height, in inches of paper
TICK_LENGTH_FRACTION = 0.5       # ruler tick length as a fraction of the quiet zone


def load_marker_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_tag(dictionary, marker_id: int, tag_size_m: float, dpi: int,
               quiet_cells: float):
    """Render one tag with quiet zone, ruler ticks and a label.

    The black square (including the tag's own 1-cell black border) is exactly
    `tag_size_m` across, because that is the edge `cv2.aruco` measures and the
    edge whose four corners become the object points in solvePnP. Measuring the
    white paper edge instead is the classic off-by-one-cell scale error.
    """
    # metres -> millimetres -> inches -> pixels
    tag_px = int(round((tag_size_m * 1000.0 / MM_PER_INCH) * dpi))

    # Quiet zone in pixels, keyed to the tag's own cell size so it scales with
    # any tag size or DPI.
    cell_px = tag_px / DICT_4X4_CELLS
    quiet_px = int(round(cell_px * quiet_cells))
    label_px = int(round(LABEL_HEIGHT_PX_PER_INCH * dpi))

    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, tag_px)

    height = tag_px + 2 * quiet_px + label_px
    width = tag_px + 2 * quiet_px
    sheet = np.full((height, width), WHITE, dtype=np.uint8)
    sheet[quiet_px:quiet_px + tag_px, quiet_px:quiet_px + tag_px] = marker

    # Ruler ticks aligned with the black square's edges. Put a ruler across a
    # facing pair and the reading must equal tag_size_m; if it does not, the
    # print dialog scaled the page and the tag is unusable.
    tick = int(round(quiet_px * TICK_LENGTH_FRACTION))
    for offset in (quiet_px, quiet_px + tag_px - 1):
        cv2.line(sheet, (offset, quiet_px - tick), (offset, quiet_px - 1), BLACK, 1)
        cv2.line(sheet, (quiet_px - tick, offset), (quiet_px - 1, offset), BLACK, 1)

    label = (f"ID {marker_id}   edge {tag_size_m * 1000:.0f} mm "
             f"(black square, between ticks)   {dpi} dpi   PRINT AT 100%")
    # Scale the font to the sheet so the label is readable at any tag size.
    font_scale = width / 1400.0
    cv2.putText(sheet, label, (quiet_px, height - label_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK,
                max(1, int(round(font_scale * 2))), cv2.LINE_AA)
    return sheet


def on_page(sheet, dpi: int, page_mm=A4_FULL_MM):
    """Centre the tag sheet on a full page-sized canvas.

    For printers that will not let you disable "fit to page". A page-sized image
    maps onto the page as a single uniform scale instead of an unknown one, so
    whatever shrink the driver applies hits every tag identically and one ruler
    measurement recovers it. It does not make the print exact - only measuring
    does that - it makes it predictable and the same on every sheet.
    """
    page_w = int(round(page_mm[0] / MM_PER_INCH * dpi))
    page_h = int(round(page_mm[1] / MM_PER_INCH * dpi))
    height, width = sheet.shape[:2]
    if width > page_w or height > page_h:
        raise SystemExit(
            f"sheet {width}x{height} px does not fit a {page_mm[0]}x{page_mm[1]} mm "
            f"page at {dpi} dpi - use a smaller --tag-size-m or --quiet-cells")
    page = np.full((page_h, page_w), WHITE, dtype=np.uint8)
    top, left = (page_h - height) // 2, (page_w - width) // 2
    page[top:top + height, left:left + width] = sheet
    return page


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="printable ArUco tags at an exact physical size")
    parser.add_argument("--markers", default="config/markers.json", type=Path,
                        help="source of tag_size_m and the marker IDs")
    parser.add_argument("--hall", default="config/hall.json", type=Path,
                        help="source of the ArUco dictionary name")
    parser.add_argument("--out", default="tags", type=Path, help="output folder")
    parser.add_argument("--dpi", default=300, type=int, help="printer resolution")
    parser.add_argument("--quiet-cells", type=float, default=DEFAULT_QUIET_ZONE_CELLS,
                        help="white border width in tag bit-cells (1 is the ArUco minimum)")
    parser.add_argument("--ids", nargs="*", type=int, default=None,
                        help="subset of IDs to render (default: all in markers.json)")
    parser.add_argument("--page", choices=("none", "a4"), default="none",
                        help="a4 centres each tag on a full A4 canvas, for printers "
                             "that cannot disable 'fit to page'")
    parser.add_argument("--tag-size-m", type=float, default=None,
                        help="override tag_size_m, e.g. to print a larger test tag")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    marker_cfg = load_marker_config(args.markers)
    hall_cfg = load_marker_config(args.hall)

    dictionary_name = hall_cfg["detection"]["dictionary"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    tag_size_m = args.tag_size_m if args.tag_size_m is not None else marker_cfg["tag_size_m"]
    ids = args.ids if args.ids else sorted(int(k) for k in marker_cfg["markers"])

    args.out.mkdir(parents=True, exist_ok=True)
    sheet = None
    for marker_id in ids:
        sheet = render_tag(dictionary, marker_id, tag_size_m, args.dpi, args.quiet_cells)
        if args.page == "a4":
            sheet = on_page(sheet, args.dpi)
        path = args.out / f"tag_{marker_id:03d}_{tag_size_m * 1000:.0f}mm.png"
        cv2.imwrite(str(path), sheet)
        print(f"{path}  {sheet.shape[1]}x{sheet.shape[0]} px")

    sheet_mm = (sheet.shape[1] / args.dpi * MM_PER_INCH,
                sheet.shape[0] / args.dpi * MM_PER_INCH)
    print(f"\nsheet size   : {sheet_mm[0]:.0f} x {sheet_mm[1]:.0f} mm")
    if args.page == "a4":
        # A page-sized image is supposed to exceed the printable area: the
        # driver shrinks it to the margins, by one factor, on every sheet.
        # That is the whole point of the mode, so the size warning does not apply.
        shrink = min(A4_PRINTABLE_MM[0] / sheet_mm[0], A4_PRINTABLE_MM[1] / sheet_mm[1])
        print(f"page mode    : full A4. 'Fit to page' shrinks this by about "
              f"{shrink:.3f} - the same on every sheet - so the tag should land "
              f"near {tag_size_m * 1000 * shrink:.0f} mm.")
        print("               That is an ESTIMATE from typical margins. Measure "
              "tick-to-tick and put the real number in tag_size_m.")
    elif sheet_mm[0] > A4_PRINTABLE_MM[0] or sheet_mm[1] > A4_PRINTABLE_MM[1]:
        print(f"WARNING      : larger than the A4 printable area "
              f"({A4_PRINTABLE_MM[0]:.0f} x {A4_PRINTABLE_MM[1]:.0f} mm). Printing this "
              f"will silently scale the tag and corrupt tag_size_m. Use larger paper, "
              f"a smaller --quiet-cells, or a smaller --tag-size-m.")

    # OpenCV cannot write the PNG pHYs chunk, so these files carry no DPI
    # metadata and viewers will guess. Print at 100% / "actual size" with page
    # scaling off, then check the tick-to-tick distance with a ruler.
    print(f"\n{len(ids)} tags, {dictionary_name}, {tag_size_m * 1000:.0f} mm edge at {args.dpi} dpi")
    if args.page == "a4":
        print("let 'fit to page' scale it, then measure tick-to-tick before taping up")
    else:
        print("print at 100% (no 'fit to page'), then measure tick-to-tick before taping up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
