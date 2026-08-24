#!/usr/bin/env python3
"""Render the report's figures from whatever data is currently on disk.

A script rather than a folder of one-off screenshots: the numbers will change
when the rig is recaptured, and a figure that silently belongs to an older run
is worse than no figure. Re-run this after every capture.

Drawn with OpenCV primitives only, per the project's dependency rule - no
matplotlib. The charts are deliberately plain.

Usage:
    python tools/make_figures.py --clip clips/marker_test_v1.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import starnav  # noqa: E402

# Light palette: these end up on a white document page, not a dark terminal.
BG = (247, 249, 250)
PANEL = (255, 255, 255)
INK = (45, 42, 40)
MUTED = (135, 130, 125)
GRID = (225, 222, 218)
GREEN = (70, 160, 60)
RED = (60, 60, 205)
BLUE = (180, 115, 45)
FONT = cv2.FONT_HERSHEY_SIMPLEX

FIGURE_WIDTH = 1600
MARGIN = 60


def text(img, string, origin, scale=0.6, colour=INK, thickness=1):
    cv2.putText(img, string, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def title_block(img, title, subtitle=None):
    text(img, title, (MARGIN, 52), 0.95, INK, 2)
    if subtitle:
        text(img, subtitle, (MARGIN, 84), 0.58, MUTED, 1)


def crop_to_detections(image, detections, padding=0.12):
    """Trim to the region the markers occupy, plus a margin.

    Handheld shots put the panel in a corner of a frame that is mostly room, so
    an uncropped figure spends most of its pixels on the desk. The margin is
    proportional rather than fixed so it survives any frame size.
    """
    if not detections:
        return image
    points = np.vstack([corners.reshape(4, 2) for _, corners in detections])
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    height, width = image.shape[:2]
    x0 = int(max(0, x0 - pad_x)); x1 = int(min(width, x1 + pad_x))
    y0 = int(max(0, y0 - pad_y)); y1 = int(min(height, y1 + pad_y))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return image
    return image[y0:y1, x0:x1]


def sample_clip(path: Path, detector, marker_map, every: int):
    """Walk the clip once, returning per-frame marker counts and some frames."""
    known_ids = set(marker_map["markers"])
    capture = cv2.VideoCapture(str(path))
    counts, frames, index = [], [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            known, unknown = starnav.split_known_unknown(corners, ids, known_ids)
            counts.append((index, len(known)))
            frames.append((index, frame, known, unknown))
        index += 1
    capture.release()
    return counts, frames


# --------------------------------------------------------------------------
# Figure 1 - detection contact sheet
# --------------------------------------------------------------------------

def figure_detection(frames, viz_cfg, columns=4, rows=2) -> np.ndarray:
    """Annotated frames spanning the clip, showing what the detector found."""
    wanted = columns * rows
    # Evenly spaced in time among frames that saw anything, rather than the
    # frames with the most markers. Picking the best frames would make the
    # figure an advert instead of evidence - the sparse views are exactly the
    # ones a reader should see.
    seen = [f for f in frames if f[2]]
    picks = seen[::max(1, len(seen) // wanted)][:wanted]

    cell_w = (FIGURE_WIDTH - 2 * MARGIN - (columns - 1) * 16) // columns
    sample = crop_to_detections(picks[0][1], picks[0][2] + picks[0][3], padding=0.18)
    cell_h = int(cell_w * sample.shape[0] / sample.shape[1])
    height = 130 + rows * (cell_h + 34) + MARGIN
    sheet = np.full((height, FIGURE_WIDTH, 3), BG, np.uint8)
    title_block(sheet, "Figure 1 - Stage 1 detection",
                "Green = ID present in the marker map. Red = detected but unmapped.")

    for n, (index, frame, known, unknown) in enumerate(picks):
        annotated = frame.copy()
        starnav.draw_detections(annotated, known, viz_cfg["known_color_bgr"])
        starnav.draw_detections(annotated, unknown, viz_cfg["unknown_color_bgr"])
        annotated = crop_to_detections(annotated, known + unknown, padding=0.18)
        cell = cv2.resize(annotated, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        x = MARGIN + (n % columns) * (cell_w + 16)
        y = 130 + (n // columns) * (cell_h + 34)
        sheet[y:y + cell_h, x:x + cell_w] = cell
        cv2.rectangle(sheet, (x, y), (x + cell_w, y + cell_h), GRID, 1)
        text(sheet, f"frame {index}   {len(known)} markers", (x, y + cell_h + 22), 0.52, MUTED)
    return sheet


# --------------------------------------------------------------------------
# Figure 2 - markers per frame
# --------------------------------------------------------------------------

def figure_marker_counts(counts) -> np.ndarray:
    """Marker count against time - the clip's usable fraction, at a glance."""
    height = 620
    fig = np.full((height, FIGURE_WIDTH, 3), BG, np.uint8)
    values = np.array([c for _, c in counts])
    indices = np.array([i for i, _ in counts])

    plot_l, plot_r = MARGIN + 70, FIGURE_WIDTH - MARGIN
    plot_t, plot_b = 150, height - 110
    cv2.rectangle(fig, (plot_l, plot_t), (plot_r, plot_b), PANEL, -1)

    usable = int((values >= 2).sum())
    title_block(fig, "Figure 2 - Mapped markers visible per frame",
                f"{usable}/{len(values)} sampled frames ({usable / len(values) * 100:.0f}%) "
                f"have the 2+ markers a fused solve needs.")

    top = max(4, int(values.max()))
    for step in range(0, top + 1, max(1, top // 5)):
        y = int(plot_b - (step / top) * (plot_b - plot_t))
        cv2.line(fig, (plot_l, y), (plot_r, y), GRID, 1)
        text(fig, str(step), (plot_l - 42, y + 5), 0.5, MUTED)

    points = [(int(plot_l + (i / indices.max()) * (plot_r - plot_l)),
               int(plot_b - (v / top) * (plot_b - plot_t)))
              for i, v in zip(indices, values)]
    cv2.fillPoly(fig, [np.array([(plot_l, plot_b)] + points + [(plot_r, plot_b)])], (222, 238, 220))
    cv2.polylines(fig, [np.array(points)], False, GREEN, 2, cv2.LINE_AA)

    # The 2-marker line is the threshold the whole quality story hangs on, so
    # it goes on top of the series rather than under it.
    y_two = int(plot_b - (2 / top) * (plot_b - plot_t))
    cv2.line(fig, (plot_l, y_two), (plot_r, y_two), RED, 1, cv2.LINE_AA)
    label = "2 markers - fused solve threshold"
    (label_w, _), _ = cv2.getTextSize(label, FONT, 0.48, 1)
    cv2.rectangle(fig, (plot_r - label_w - 14, y_two - 24),
                  (plot_r - 4, y_two - 4), PANEL, -1)
    text(fig, label, (plot_r - label_w - 10, y_two - 10), 0.48, RED)

    text(fig, "frame number", ((plot_l + plot_r) // 2 - 60, plot_b + 44), 0.55, MUTED)
    text(fig, "markers", (MARGIN - 10, plot_t - 14), 0.55, MUTED)
    text(fig, f"median {np.median(values):.0f}   max {values.max()}   "
              f"zero-marker frames {int((values == 0).sum())}",
         (plot_l, height - 46), 0.55, INK)
    return fig


# --------------------------------------------------------------------------
# Figure 3 - handedness
# --------------------------------------------------------------------------

def figure_handedness(frame, known, marker_map, camera_matrix, dist_coeffs,
                      ranked) -> np.ndarray:
    """Why the winning convention wins, shown rather than asserted.

    Left: the correct convention's reprojected corners land on the detected
    ones. Right: the mirrored convention's land nowhere near, because no rigid
    camera pose can reproduce a reversed winding. The bar chart underneath is
    the same fact as a number.
    """
    panels = []
    for label, (rotation, mirror), colour in (
            ("correct: rotation 0, mirror False", (0, False), BLUE),
            ("mirrored: rotation 0, mirror True", (0, True), RED)):
        offsets = starnav.corner_offsets(marker_map["tag_size_m"] / 2.0, rotation, mirror)
        pose = starnav.pose_from_detections(known, marker_map, camera_matrix,
                                            dist_coeffs, offsets)
        object_points, image_points, _ = starnav.build_correspondences(
            known, marker_map, offsets)
        projected, _ = cv2.projectPoints(object_points, pose["rvec"], pose["tvec"],
                                         camera_matrix, dist_coeffs)

        canvas = frame.copy()
        for (ox, oy) in image_points:
            cv2.circle(canvas, (int(ox), int(oy)), 5, GREEN, 1, cv2.LINE_AA)
        for (px, py) in projected.reshape(-1, 2):
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.drawMarker(canvas, (int(px), int(py)), colour, cv2.MARKER_TILTED_CROSS,
                               9, 1, cv2.LINE_AA)
        panels.append((label, crop_to_detections(canvas, known), pose["reproj_px"]))

    panel_w = (FIGURE_WIDTH - 2 * MARGIN - 40) // 2
    panel_h = int(panel_w * panels[0][1].shape[0] / panels[0][1].shape[1])
    chart_h = 230
    height = 150 + panel_h + 60 + chart_h + MARGIN
    fig = np.full((height, FIGURE_WIDTH, 3), BG, np.uint8)
    title_block(fig, "Figure 3 - Resolving the corner handedness",
                "Circles = detected corners. Crosses = corners reprojected from the solved pose.")

    for n, (label, canvas, residual) in enumerate(panels):
        x = MARGIN + n * (panel_w + 40)
        fig[150:150 + panel_h, x:x + panel_w] = cv2.resize(
            canvas, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(fig, (x, 150), (x + panel_w, 150 + panel_h), GRID, 1)
        text(fig, label, (x, 150 + panel_h + 26), 0.58, INK)
        text(fig, f"mean reprojection error {residual:.2f} px",
             (x, 150 + panel_h + 50), 0.55, GREEN if n == 0 else RED)

    # Bar chart of all eight candidates.
    base = 150 + panel_h + 90
    text(fig, "All 8 candidate conventions, mean reprojection error (px)",
         (MARGIN, base + 20), 0.6, INK)
    bar_top, bar_bottom = base + 40, base + chart_h - 40
    worst = max(r for r, _, _ in ranked)
    slot = (FIGURE_WIDTH - 2 * MARGIN) // len(ranked)
    for n, (residual, rotation, mirror) in enumerate(ranked):
        x = MARGIN + n * slot
        bar = int((residual / worst) * (bar_bottom - bar_top))
        colour = GREEN if n == 0 else (200, 200, 205)
        cv2.rectangle(fig, (x + 12, bar_bottom - bar), (x + slot - 22, bar_bottom), colour, -1)
        text(fig, f"{residual:.1f}", (x + 12, bar_bottom - bar - 8), 0.5, INK)
        text(fig, f"r{rotation} {'mir' if mirror else 'nom'}", (x + 12, bar_bottom + 22), 0.48, MUTED)
    return fig


# --------------------------------------------------------------------------
# Figure 4 - world frame
# --------------------------------------------------------------------------

def figure_world_frame() -> np.ndarray:
    """The screen rig's axes and how they map onto the hall's.

    Worth a diagram because the Y-down choice looks arbitrary until the
    right-handedness constraint is drawn next to it.
    """
    height = 860
    fig = np.full((height, FIGURE_WIDTH, 3), BG, np.uint8)
    title_block(fig, "Figure 4 - World frame for the screen rig",
                "Right-handed, with +Z running from the camera to the marker plane - "
                "structurally identical to the hall.")

    # Generous headroom to the left and above the panel: the axis labels live
    # outside it, so nothing is drawn over the tags.
    panel_x, panel_y, panel_w, panel_h = 260, 250, 560, 330
    cv2.rectangle(fig, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 58, 56), -1)
    for row in range(3):
        for col in range(5):
            tx = panel_x + 40 + col * 100
            ty = panel_y + 45 + row * 95
            cv2.rectangle(fig, (tx, ty), (tx + 52, ty + 52), (245, 245, 245), -1)
            cv2.rectangle(fig, (tx + 9, ty + 9), (tx + 43, ty + 43), (30, 30, 30), -1)
    text(fig, "27in panel, 50 tags at known screen positions",
         (panel_x, panel_y + panel_h + 34), 0.55, MUTED)

    origin = (panel_x, panel_y)
    cv2.arrowedLine(fig, (origin[0], origin[1] - 26), (origin[0] + 210, origin[1] - 26),
                    RED, 3, cv2.LINE_AA, tipLength=0.12)
    text(fig, "+X  screen right", (origin[0] + 224, origin[1] - 19), 0.6, RED)
    cv2.arrowedLine(fig, (origin[0] - 26, origin[1]), (origin[0] - 26, origin[1] + 200),
                    GREEN, 3, cv2.LINE_AA, tipLength=0.13)
    text(fig, "+Y", (origin[0] - 82, origin[1] + 110), 0.6, GREEN)
    text(fig, "screen", (origin[0] - 128, origin[1] + 140), 0.52, GREEN)
    text(fig, "down", (origin[0] - 116, origin[1] + 162), 0.52, GREEN)

    cv2.circle(fig, origin, 13, BLUE, 2, cv2.LINE_AA)
    cv2.line(fig, (origin[0] - 9, origin[1] - 9), (origin[0] + 9, origin[1] + 9), BLUE, 2, cv2.LINE_AA)
    cv2.line(fig, (origin[0] - 9, origin[1] + 9), (origin[0] + 9, origin[1] - 9), BLUE, 2, cv2.LINE_AA)
    text(fig, "origin, +Z into the screen", (origin[0] - 34, origin[1] - 62), 0.58, BLUE)

    camera_x = panel_x + panel_w + 320
    camera_y = panel_y + panel_h // 2
    cv2.rectangle(fig, (camera_x - 45, camera_y - 34), (camera_x + 45, camera_y + 34), INK, -1)
    cv2.circle(fig, (camera_x, camera_y), 20, BG, -1)
    text(fig, "camera", (camera_x - 34, camera_y + 66), 0.58, INK)
    cv2.arrowedLine(fig, (camera_x - 52, camera_y), (panel_x + panel_w + 16, camera_y),
                    MUTED, 2, cv2.LINE_AA, tipLength=0.05)
    text(fig, "0.75 m", (panel_x + panel_w + 118, camera_y - 16), 0.6, MUTED)

    notes = [
        "Y points DOWN the screen so the frame stays right-handed once +Z is fixed pointing",
        "from the camera towards the markers.",
        "",
        "In the hall the same frame is X right, Y forward, Z up: the camera sits below the",
        "ceiling plane and looks along +Z at it. Same arrangement, plane rotated.",
        "",
        "Marker world XY is therefore just the tag's position on the panel, in metres, with the",
        "origin at the top-left of the active area.",
    ]
    for n, line in enumerate(notes):
        text(fig, line, (MARGIN, 660 + n * 24), 0.55, INK)
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="render report figures from the current data")
    parser.add_argument("--clip", type=Path, default=Path("clips/marker_test_v1.mp4"),
                        help="video or image folder for figures 1 and 2")
    parser.add_argument("--photos", type=Path, default=Path("photos/eval"),
                        help="stills for the handedness figure")
    parser.add_argument("--calib", type=Path, default=Path("calib.npz"),
                        help="intrinsics; figure 3 is skipped without them")
    parser.add_argument("--hall", type=Path, default=Path("config/hall.json"),
                        help="source of the dictionary and marker map path")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"),
                        help="output folder for the figures")
    parser.add_argument("--every", type=int, default=3, help="sample every Nth clip frame")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hall = starnav.load_json(args.hall)
    marker_map = starnav.load_marker_map(Path(hall["markers"]))
    detector = starnav.make_detector(hall["detection"])
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    if args.clip.exists():
        counts, frames = sample_clip(args.clip, detector, marker_map, args.every)
        for name, figure in (("fig1_detection.png", figure_detection(frames, hall["viz"])),
                             ("fig2_markers_per_frame.png", figure_marker_counts(counts))):
            cv2.imwrite(str(args.out / name), figure)
            written.append(name)

    grid = Path("tags/screen_grid.png")
    if grid.exists():
        cv2.imwrite(str(args.out / "fig0_screen_grid.png"), cv2.imread(str(grid)))
        written.append("fig0_screen_grid.png")

    if args.calib.exists() and args.photos.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import eval_photos  # noqa: E402

        camera_matrix, dist_coeffs, image_size = starnav.load_calibration(args.calib)
        paths = eval_photos.load_photos(args.photos)
        detections = eval_photos.detect_all(paths, detector, set(marker_map["markers"]), image_size)
        usable = {k: v for k, v in detections.items() if len(v) >= 8}
        if usable:
            ranked = eval_photos.resolve_convention(usable, marker_map, camera_matrix, dist_coeffs)
            best_name = max(usable, key=lambda k: len(usable[k]))
            frame = cv2.imread(str(args.photos / best_name))
            figure = figure_handedness(frame, usable[best_name], marker_map,
                                       camera_matrix, dist_coeffs, ranked)
            cv2.imwrite(str(args.out / "fig3_handedness.png"), figure)
            written.append("fig3_handedness.png")

    cv2.imwrite(str(args.out / "fig4_world_frame.png"), figure_world_frame())
    written.append("fig4_world_frame.png")

    for name in sorted(written):
        path = args.out / name
        image = cv2.imread(str(path))
        print(f"{path}  {image.shape[1]}x{image.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
