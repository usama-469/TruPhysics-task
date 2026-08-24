# CLAUDE.md — Star Navigation

Ceiling-marker indoor positioning prototype. Upward-facing camera reads fiducial
markers on the ceiling and outputs absolute X/Y position in a hall coordinate frame.

This is an internship evaluation task with a 3-day budget. Correctness and
legibility beat performance. Do not optimize anything until it works and is measured.

---

## Hardware reality

- **Target hardware (not available):** Raspberry Pi 5 + upward camera, industrial hall,
  ceiling height up to 12 m, target accuracy ±5 cm.
- **Actual dev hardware:** laptop running Python, plus either the laptop webcam or a
  phone camera streamed in as a video device. Printed ArUco tags taped to a ~2.5 m
  ceiling in a hand-measured grid.
- **Therefore:** everything is a *scaled demonstrator*. Marker size is scaled to
  preserve angular size (15 cm tag at 2.5 m ≈ 72 cm tag at 12 m). The code path must
  be the same one that would run on the Pi — no laptop-only shortcuts.

Never write code that assumes the 12 m case is being tested. Never hardcode the
ceiling height; read it from config.

---

## Coordinate conventions — read before touching pose code

- **World frame:** origin at a marked corner of the test area. X right, Y forward,
  Z **up**. Units are **metres** everywhere. Never centimetres, never pixels, in any
  stored or transmitted value.
- **Ceiling plane:** all markers lie at Z = `ceiling_height`, facing downward.
- **Camera frame:** OpenCV convention — X right, Y down, Z along the optical axis.
  The camera looks up, so world-Z-up and camera-Z-forward are roughly antiparallel.
- **solvePnP output:** `rvec`, `tvec` map **world → camera**. To get the camera in world:
  ```python
  R, _ = cv2.Rodrigues(rvec)
  C = (-R.T @ tvec).ravel()   # camera position in world coords
  ```
  `C[0]`, `C[1]` are the reported X/Y. Yaw comes from `R.T`.
- **Marker corner order:** `cv2.aruco` returns corners clockwise starting top-left *as
  seen in the image*. Because the markers are viewed from below, the mapping from image
  corners to world object points has a handedness trap. **Verify empirically with one
  tag at a known position before building anything on top of it.** If X or Y comes out
  mirrored or negated, the bug is here, not in the solver.

---

## Architecture

Single Python process, one loop. Keep modules small and independently testable.

```
starnav/
  config.py          # loads hall config + marker map, no logic
  calibrate.py       # chessboard intrinsic calibration -> calib.npz
  detector.py        # frame -> list of (marker_id, image_corners)
  pose.py            # detections + map + intrinsics -> (x, y, yaw, quality)
  viz.py             # OpenCV-drawn 2D hall map window
  publish.py         # UDP JSON emitter
  main.py            # capture loop wiring the above
  tools/
    generate_tags.py # printable tag sheet PDF/PNG at a specified physical size
    eval_static.py   # repeatability: N samples at rest -> mean, std, drift
    eval_translation.py # known displacement -> measured displacement, scale error
config/
  hall.json          # ceiling height, tag size, camera source, UDP target
  markers.json       # {"id": [x, y]} in metres, plus survey_uncertainty_m
```

## Pose estimation approach

- **Preferred:** collect 3D–2D correspondences from **all** visible markers into a
  single `cv2.solvePnP` call. One fused pose from all corners. Do **not** compute a
  pose per marker and average the results — that discards the joint geometry and
  handles outliers badly.
- **Single marker fallback:** `cv2.SOLVEPNP_IPPE_SQUARE`. Flag these frames as
  low-quality in the output; single-tag pose at long range is badly conditioned and
  rotation-ambiguous.
- **Planar structure:** all markers share a known plane. This is the main source of
  conditioning improvement with ≥2 markers. Note it in comments where exploited.
- **Smoothing:** none in v1. Add an exponential filter or constant-velocity Kalman only
  *after* raw jitter has been measured and logged. Smoothing before measuring hides the
  numbers the report needs.
- Every emitted pose carries a `quality` field: number of markers used, mean
  reprojection error in pixels, and a coarse accuracy estimate in metres.

## Visualization

OpenCV drawing into a second named window. Deliberately plain — the spec says polish
is not required. Must show: hall bounds, known marker positions as squares, detected
markers highlighted, current position as a dot with a short heading line, a faint trail
of the last ~100 poses, and a text block with X, Y, yaw, detected IDs, FPS, and marker
count. No matplotlib, no pygame, no separate render thread.

## Output interface

UDP JSON to `127.0.0.1:5005`, one datagram per frame:
```json
{"t": 1723459200.12, "x": 12.43, "y": 6.81, "yaw_deg": 92.4,
 "markers": [1, 4, 5], "reproj_px": 0.38, "acc_est_m": 0.04}
```
ROS 2 is the eventual target; keep the payload flat so a bridge node is trivial.

---

## Measurement obligations

The written report is graded alongside the code. These scripts exist to produce numbers
for it, so keep their output copy-pasteable:

1. **Calibration quality** — RMS reprojection error in px. Reject and recapture if > 0.5.
2. **Repeatability** — camera stationary, 200 samples, report σ in mm for X and Y.
   This is the algorithm's noise floor and is independent of tag placement error.
3. **Scale accuracy** — slide the camera a tape-measured 1.00 m, report measured
   displacement. Tests scale; also independent of absolute tag placement.
4. **Marker placement error** — record how tags were surveyed and the resulting
   uncertainty. Carry it as its own row in the error budget. Do not conflate it with
   algorithm error.

## Known error terms (context for the report, not for the code)

Camera tilt dominates at height: 1° of tilt at 12 m is roughly 21 cm of lateral error.
Pixel/GSD error is comfortable by comparison (~3 mm/px at 12 m with a ~3900 px focal
length). Conclusion the report should reach: ±5 cm at 12 m requires levelled mounting,
an IMU, or multi-marker geometry constraining orientation — not more resolution.

---

## Working rules

- Python 3, OpenCV, NumPy. No other dependencies without asking.
- Config over constants. No magic numbers in function bodies.
- Camera source is swappable via config: webcam index, phone stream URL, or a video file.
  A recorded video file must work as an input so results are reproducible offline.
- If a phone is used as the camera, calibration **must** be captured through the same
  streaming path as the demo. Re-encoding, cropping, and digital stabilization all
  silently invalidate intrinsics.
- Lock focus and exposure where the API allows. Autofocus hunting mid-run causes
  position jumps that look like algorithm failure.
- Log to CSV every frame. The report needs the raw data.
- Comment the *why* on anything geometric. The reader is evaluating reasoning.
- When something is a scaled-demo compromise, mark it `# SCALED-DEMO:` with the note on
  what changes on real hardware. These comments get collected into the report's
  "next steps" section.
