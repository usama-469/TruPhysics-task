# Task spec — Star Navigation v1 (minimal core)

Deliberately minimal. One file, no OCR, no line detection, no filtering, no threading.
The goal is a working position estimate I can read and defend line by line in under an hour.

---

## Problem

An upward-facing camera moves beneath a ceiling of ArUco markers whose world positions
are already known. Compute the **camera's** position in the world frame, continuously,
from the video feed.

The marker map is an input, not something to detect. Marker IDs come from the ArUco
encoding, not from reading text. There are no grid lines anywhere.

## Input

- A video source: webcam index, video file path, or image sequence folder.
  All three must work — video files make results reproducible without the ceiling rig.
- `calib.npz`: camera matrix and distortion coefficients from a prior calibration step.
- `markers.json`: `{"tag_size_m": 0.15, "ceiling_height_m": 2.5, "markers": {"0": [0.0, 0.0], "1": [1.0, 0.0], ...}}`
  Positions are the marker centres in metres, world X/Y. All markers lie on the ceiling plane.

## Core algorithm

For each frame:

1. **Detect** — `cv2.aruco.detectMarkers` with `DICT_4X4_50`. Returns IDs and image corners.
2. **Discard** any detected ID not present in `markers.json`.
3. **Build correspondences** — for each known marker, generate its four corner positions
   in world coordinates from its centre, `tag_size_m` and `ceiling_height_m`. Pair them
   with the four detected image corners. N markers gives 4N point pairs.
4. **Solve** — one `cv2.solvePnP` call over *all* pairs at once. Not one call per marker
   averaged afterwards; the joint solve uses the full geometry.
   Single marker visible: fall back to `cv2.SOLVEPNP_IPPE_SQUARE` and flag the frame as
   low quality (single-tag pose is ill-conditioned and rotation-ambiguous).
5. **Invert to world** — `solvePnP` returns world→camera. The camera position is
   `C = -R.T @ tvec` where `R, _ = cv2.Rodrigues(rvec)`. Report `C[0]`, `C[1]`. Yaw from `R.T`.
6. **Quality** — mean reprojection error in pixels, plus the marker count.

No smoothing. Raw per-frame output only; jitter must be visible so it can be measured.

## Coordinate conventions

World: X right, Y forward, Z up, origin at a marked corner of the test area, metres.
Camera: OpenCV convention, X right, Y down, Z along the optical axis.

The camera views the markers from below, which puts a handedness trap in the mapping from
detected image corners to world object points. **Verify with one marker at a known
position before building anything on top.** Mirrored or negated X/Y means the bug is here,
not in the solver.

## Output

Per frame, printed and appended to a CSV:

```
frame  x_m  y_m  yaw_deg  n_markers  ids  reproj_px  acc_est_m
```

Plus two OpenCV windows:

- **Camera view** — the frame with detected markers outlined and IDs drawn.
- **Hall map** — plain 2D top-down plot: known marker positions as small squares,
  detected ones highlighted, current camera position as a dot with a short heading line,
  a faint trail of the last 100 positions, and a text block showing X, Y, yaw, IDs, FPS.
  OpenCV drawing primitives only. Polish is explicitly not required.

## Accuracy reporting

Report `acc_est_m` per frame, derived rather than asserted:

- Ground sampling distance at the ceiling: `ceiling_height_m / focal_length_px`.
- Multiply by the mean reprojection error in px to get an approximate position uncertainty.
- Scale by roughly `1/sqrt(n_markers)` to reflect multi-marker averaging.
- State in a comment that this excludes marker placement survey error and camera tilt,
  which are handled separately in the report's error budget.

Never report precision finer than the ground sampling distance supports.

## Constraints

- Python 3, OpenCV, NumPy. Nothing else.
- Single file is fine at this stage; split only when it exceeds readability.
- No magic numbers in function bodies — everything from config.
- Comment the *why* on every geometric step. The reasoning is what's being evaluated.
- Where a choice exists only because this is a 2.5 m scaled rig rather than a 12 m hall,
  mark it `# SCALED-DEMO:` with a note on what changes on real hardware.

## Build order

Do these in sequence and stop after each so I can run it:

1. Detection only — draw outlines and IDs on the live feed. Confirms camera and dictionary.
2. Single-marker pose — one tag at a known position, print XY. **Handedness check here.**
3. Multi-marker fused solve — verify XY is stable and sensible with 3+ tags visible.
4. Hall map window and CSV logging.
5. Accuracy estimate and quality metrics.
