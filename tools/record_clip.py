#!/usr/bin/env python3
"""Record a clip from the camera as lossless PNG frames.

Exists because of a practical constraint of the screen rig: with one monitor,
a fullscreen tag pattern covers the terminal and the starnav windows, so there
is no way to watch a live run. Capture first, replay through starnav after -
which is also what the spec wants anyway, since a recorded input makes a result
reproducible without the rig.

PNG frames rather than a video file, deliberately. Every lossy codec (MJPG,
H.264) works by discarding high-frequency detail, and a fiducial marker is
almost entirely high-frequency detail at its corners. The compression ringing
lands exactly where sub-pixel corner refinement is looking, so it biases the
corner estimate and therefore the pose. `starnav.py` already accepts a folder
of images as a source, so this costs nothing downstream.

Capture goes through starnav's own FrameSource, so the camera is opened with
the same backend and the same focus/exposure locks as a live run. A clip
captured through a different path than the calibration is not usable.

Usage:
    python tools/record_clip.py --out clips/run1 --seconds 30 --delay 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Run from anywhere: the repo root holds starnav.py, this file sits in tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import starnav  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="record camera frames as lossless PNGs")
    parser.add_argument("--hall", default="config/hall.json", type=Path,
                        help="runtime config; supplies the camera locks")
    parser.add_argument("--source", default=None, help="override hall.json source")
    parser.add_argument("--out", default=Path("clips/run"), type=Path, help="output folder")
    parser.add_argument("--seconds", type=float, default=30.0, help="capture duration")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="grace period before capture starts, to bring the tag "
                             "window up and get hands out of frame")
    parser.add_argument("--max-frames", type=int, default=2000,
                        help="hard cap, so a mistake cannot fill the disk")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hall = starnav.load_json(args.hall)
    source_spec = args.source if args.source is not None else hall["source"]
    source = starnav.FrameSource(source_spec, hall["camera"])
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"source  : {source.kind}")
    print(f"output  : {args.out}")
    for remaining in range(int(args.delay), 0, -1):
        print(f"starting in {remaining}...", flush=True)
        time.sleep(1.0)

    # Discard the first few frames: most cameras deliver a couple of frames at
    # the pre-lock exposure before the settings applied at open take effect.
    for _ in range(5):
        source.read()

    print(f"recording {args.seconds:.0f} s", flush=True)
    started = time.perf_counter()
    count = 0
    try:
        while count < args.max_frames:
            elapsed = time.perf_counter() - started
            if elapsed >= args.seconds:
                break
            frame = source.read()
            if frame is None:
                print("source ended early")
                break
            # Zero-padded so the folder replays in capture order: an image
            # sequence has no timestamps, filename order is all there is.
            cv2.imwrite(str(args.out / f"frame_{count:06d}.png"), frame)
            count += 1
    finally:
        source.release()

    duration = time.perf_counter() - started
    print(f"wrote {count} frames in {duration:.1f} s ({count / max(duration, 1e-6):.1f} fps)")
    print(f"replay with: python starnav.py --source {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
