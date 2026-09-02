#!/usr/bin/env python3
"""Raspberry Pi camera check - detect ArUco tags, or collect calibration views.

    python3 starnav_pi.py                       marker detection; watch at :8000
    python3 starnav_pi.py --calibrate 9x6       collect chessboard views instead
    python3 starnav_pi.py --selfcheck           no camera needed

The Pi has no screen, so the annotated feed goes out as MJPEG over HTTP and any browser
on the network is the display. That is stdlib only; nothing to install beyond OpenCV.

Calibration is a two-step job on purpose. This script only CAPTURES views - it watches
for a chessboard and saves a frame whenever the board has moved somewhere new, so you
just walk it around in front of the lens and watch the counter in the browser. The solve
is already written: tools/calibrate.py does it, with the RMS threshold, the per-view
residuals and the tilt-spread check the report needs.

    python3 starnav_pi.py --calibrate 9x6        # until the counter stops rising
    scp -r usama@starnav.local:~/calib_views .   # bring the JPEGs back
    python tools/calibrate.py --photos calib_views --chessboard 9x6 --square-mm 25

Print the board with tools/chessboard.py, and MEASURE a square with calipers rather than
trusting the printer's scaling. Capture happens through the same VideoCapture path the
demo uses, which is the part that has to match; where the solve runs does not matter.

Setup on the Pi (the apt build of OpenCV is usually too old for cv2.aruco.ArucoDetector):

    pip install opencv-contrib-python        # needs >= 4.7, and contrib for cv2.aruco
    python3 -c "import cv2; print(cv2.__version__, hasattr(cv2.aruco, 'ArucoDetector'))"

A USB webcam is --source 0. The CSI Camera Module is not a V4L2 index under libcamera
and needs picamera2 instead; that is a different capture path, not a flag.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

BOUNDARY = "frame"


def banner(frame, text):
    cv2.putText(frame, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA)


def markers_view(dictionary_name):
    """Outline and label every ArUco marker. Returns a process(frame) callable."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name)))

    def process(frame):
        corners, ids, _rejected = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # Outline plus the printed ID for every hit - the whole labelling job in one call.
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        banner(frame, f"{0 if ids is None else len(ids)} markers")

    return process


def chessboard_view(pattern, outdir, wanted, min_move):
    """Save a frame whenever the board reaches a pose we have not captured yet.

    A calibration set is only as good as its variety: the same board in the same place
    twenty times pins down nothing. Views are accepted on having moved across the frame
    or changed apparent size, which is a cheap stand-in for "this is a new pose". It
    cannot see TILT, which is what actually makes a planar solve well posed, so the
    overlay asks for it and calibrate.py measures the spread afterwards and complains.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    seen = []  # (centre_xy, span_px) per accepted view

    def process(frame):
        # Plain flags, not EXHAUSTIVE: this detection only decides whether the frame is
        # worth keeping. calibrate.py re-detects on the saved JPEG with the accurate
        # settings, so nothing here has to be sub-pixel.
        found, corners = cv2.findChessboardCornersSB(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
        if found and len(seen) < wanted:
            points = corners.reshape(-1, 2)
            centre, span = points.mean(axis=0), float(np.ptp(points, axis=0).max())
            gap = min_move * frame.shape[1]
            if all(np.hypot(*(centre - c)) > gap or abs(span - s) > gap for c, s in seen):
                path = outdir / f"view_{len(seen) + 1:02d}.jpg"
                cv2.imwrite(str(path), frame)  # clean frame, before anything is drawn on it
                seen.append((centre, span))
                print(f"captured {path}  ({len(seen)}/{wanted})")
        if found:
            cv2.drawChessboardCorners(frame, pattern, corners, found)
        banner(frame, f"{len(seen)}/{wanted} views captured - stop and run calibrate.py"
               if len(seen) >= wanted else
               f"{len(seen)}/{wanted} views - move and TILT the board")

    return process


class Stream(BaseHTTPRequestHandler):
    """Serves the annotated feed as multipart JPEG, which browsers render natively."""

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        try:
            while True:
                ok, frame = self.server.camera.read()
                if not ok:
                    break
                self.server.process(frame)
                ok, jpeg = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg.tobytes())
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer closed the tab


def selfcheck():
    """Both detectors, on images drawn in memory. No camera involved."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[50:250, 50:250] = cv2.aruco.generateImageMarker(dictionary, 7, 200)
    _corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(canvas)
    assert ids is not None and ids.flatten().tolist() == [7], ids

    # 10x7 squares = 9x6 inner corners, on a white quiet zone the detector needs.
    squares = (np.indices((7, 10)).sum(axis=0) % 2 * 255).astype(np.uint8)
    board = np.full((420, 540), 255, np.uint8)
    board[60:340, 80:480] = np.kron(squares, np.ones((40, 40), np.uint8))
    found, corners = cv2.findChessboardCornersSB(board, (9, 6))
    assert found and len(corners) == 54, (found, None if not found else len(corners))

    print("selfcheck ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0",
                        help="camera index, video file, or stream URL")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--size", nargs=2, type=int, default=(1280, 720),
                        metavar=("W", "H"))
    parser.add_argument("--fourcc", default="MJPG",
                        help="capture format. High-resolution USB cameras only offer "
                             "their useful frame rates in MJPG; the YUYV the driver "
                             "picks by default caps them at a few fps. '-' to leave alone.")
    parser.add_argument("--calibrate", default=None, metavar="COLSxROWS",
                        help="collect chessboard views instead of detecting markers. "
                             "COLSxROWS are INNER corner counts, e.g. 9x6.")
    parser.add_argument("--views", type=int, default=20,
                        help="how many calibration views to collect (15-25 is sensible)")
    parser.add_argument("--outdir", type=Path, default=Path("calib_views"),
                        help="where calibration views are written")
    parser.add_argument("--min-move", type=float, default=0.10,
                        help="how far the board must move to count as a new view, "
                             "as a fraction of frame width")
    parser.add_argument("--selfcheck", action="store_true",
                        help="verify both detectors work, then exit")
    args = parser.parse_args()

    if args.selfcheck:
        return selfcheck()

    camera = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not camera.isOpened():
        raise SystemExit(f"could not open camera {args.source!r}")
    # Format before size: the driver offers different resolutions per format, so asking
    # for a size first and the format second can leave you with neither.
    if args.fourcc != "-":
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.size[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.size[1])
    print(f"capture: {camera.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}"
          f"x{camera.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {camera.get(cv2.CAP_PROP_FPS):.0f} fps")

    if args.calibrate:
        cols, rows = (int(n) for n in args.calibrate.lower().split("x"))
        process = chessboard_view((cols, rows), args.outdir, args.views, args.min_move)
        print(f"collecting {args.views} views of a {cols}x{rows} board -> {args.outdir}")
    else:
        process = markers_view(args.dict)

    # ponytail: single-threaded, so one viewer at a time. ThreadingHTTPServer plus a
    # shared latest-frame if more than one person needs to watch.
    server = HTTPServer(("", args.port), Stream)
    server.camera, server.process = camera, process
    print(f"watch at http://<pi-address>:{args.port}/   (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()


if __name__ == "__main__":
    main()
