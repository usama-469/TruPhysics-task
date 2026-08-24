# Star Navigation — ceiling-marker indoor positioning

An upward-facing camera reads ArUco markers whose world positions are already known,
and reports the **camera's** absolute X/Y in a hall coordinate frame.

Target hardware is a Raspberry Pi 5 under a 12 m industrial ceiling. What actually runs
here is a scaled demonstrator: a 27" monitor stands in for the ceiling, at 0.75 m instead
of 12 m, with tag size scaled to hold the same angular geometry. **The code path is the
same one that would run on the Pi** — only two numbers in a config file differ.

Python 3, OpenCV, NumPy. Nothing else.

---

## Quick start

Two workflows. Pick by what you're testing.

### A. Screen rig — geometry and accuracy

```bash
python tools/screen_tags.py --fill --camera-distance-m 0.75   # tags on the monitor
# photograph the panel: ~20 shots into photos/calib, ~20 different ones into photos/eval
python tools/calibrate.py   --photos photos/calib --refine-win 9
python tools/eval_photos.py --photos photos/eval  --refine-win 9
```

Gives you: camera intrinsics, the resolved corner handedness, and position error in
millimetres.

### B. Live or recorded run — the pipeline itself

```bash
python starnav.py --source 0                        # webcam
python starnav.py --source clips/marker_test_v1.mp4 # video file
python starnav.py --source clips/run1               # folder of images
```

`q` or `ESC` quits, `space` pauses, `s` saves a snapshot to `logs/snapshots/`.

---

## Layout

```
starnav.py              the pipeline: config, capture, detection, pose
config/hall.json        runtime knobs — source, marker map path, camera, detector, viz
config/markers.json     marker map for printed tags on a real ceiling
config/markers_screen.json   marker map for the monitor (written by screen_tags.py)
calib.npz               camera intrinsics (written by calibrate.py)

tools/screen_tags.py    display the tag grid on a monitor at an exact physical size
tools/generate_tags.py  printable tags at an exact physical size, for a real ceiling
tools/calibrate.py      intrinsics from photos of the grid
tools/eval_photos.py    resolve handedness, measure error
tools/record_clip.py    capture lossless frames from the camera
tools/make_figures.py   render the report figures

photos/calib/  photos/eval/   your stills
clips/                        videos and recorded frame folders
logs/                         CSV output and snapshots
docs/figures/                 generated figures
```

---

## Generating the tag grid

This is the target everything else measures against, so it is worth understanding.

### The basic call

```bash
python tools/screen_tags.py --fill
```

Detects your monitor, computes its physical size, renders a grid, and shows it
**fullscreen**. Press any key to close. It also writes two files:

- `tags/screen_grid.png` — a copy of what was displayed, for the report
- `config/markers_screen.json` — where every tag is, in metres, which is what the
  pose solver reads

On the current 27" 1920×1080 panel `--fill` produces **50 tags at 44.83 mm**, 10×5,
59.8 mm pitch.

### Why it is displayed rather than saved and opened

An image viewer will fit-to-window and resample. That silently changes the physical tag
size, and since `tag_size_m` is the only thing setting the scale of the whole system,
every position downstream would be wrong by that same fraction with nothing to reveal it.
A fullscreen OpenCV window at native resolution is 1:1.

For the same reason the script sets process DPI awareness before asking Windows for the
resolution — without it, a display running at 125 % scaling reports 1536 instead of 1920
and every millimetre is off by a quarter.

### Check this before trusting any number

**Measure the ruler bar at the bottom of the screen with a physical ruler.** It should
read 100 mm. If it doesn't, the assumed panel size is wrong and everything scales with it.
Fix by passing the true diagonal:

```bash
python tools/screen_tags.py --fill --diagonal-in 24
```

**The panel must be flat.** A curved 1800R monitor bows about 25 mm out of plane across
600 mm, which breaks the coplanar-marker assumption outright. There is no flag for this.

### Sizing the tags

Two modes, mutually exclusive:

```bash
python tools/screen_tags.py --fill                  # most tags the dictionary can address
python tools/screen_tags.py --tag-size-mm 60        # a specific size
```

Either way the size is snapped to a whole number of screen pixels per bit-cell. A tag
whose cells land on fractional pixels gets resampled unevenly across its face, which puts
a systematic bias into sub-pixel corner refinement. The printed size in the summary is
the snapped one, not what you asked for.

`--fill` is capped by the dictionary: `DICT_4X4_50` has 50 IDs, so 50 tags. Asking for a
denser grid fails with a message rather than silently reusing IDs.

### Where to stand

The summary prints a recommended camera distance:

```
angular match to the 12 m hall (0.72 m tag / 12 m = 0.060):
  put the camera 0.75 m from the screen
```

That reproduces the real system's tag-size-to-range ratio, which is what governs pose
conditioning. It also tells you how many pixels a tag will occupy at that distance and
whether the whole panel stays in frame.

You do not have to obey it — shooting closer gives bigger tags and an easier problem than
the hall, shooting further gives a harder one. Just say which you did in the report.

### The other options

| Flag | Does |
|---|---|
| `--camera-distance-m 0.75` | Written into the map as `ceiling_height_m`. Set it to where you'll actually stand — then a solved `z` near 0 means the range is right, and any offset is a direct readout of focal-length error. |
| `--offset-mm 100 0` | Shifts the whole pattern by an exact amount. The absolute scale test: shoot from a fixed spot before and after, and the reported camera XY must move by the same distance in the opposite direction. |
| `--background 110` | Grey level outside the tags. Lower reduces bloom from a bright panel. Each tag keeps its own white quiet zone regardless. |
| `--quiet-cells 1` | White border in bit-cells. 1 is the ArUco minimum and the default. |
| `--pitch-mm`, `--cols`, `--rows` | Manual layout instead of automatic packing. |
| `--no-ruler` | Drops the ruler strip, uses the full panel height. Only after you have verified the scale once. |
| `--resolution 1920x1080` | Override auto-detection, e.g. for a second monitor. |
| `--no-show` | Write the files without displaying. |

### If you have one monitor

The fullscreen tag window covers the terminal and `starnav.py`'s windows, so there is no
way to watch a live run. Record instead:

```bash
python tools/record_clip.py --out clips/run1 --seconds 30 --delay 5
python starnav.py --source clips/run1
```

The 5-second delay is for bringing the tag window up and getting your hands out of frame.

---

## Script reference

### `starnav.py` — the pipeline

The whole positioning path: config loading, capture, detection, pose. Everything else in
`tools/` imports from here rather than reimplementing, so what the tools measure is what
actually runs.

```bash
python starnav.py --source 0
python starnav.py --source clip.mp4 --calib calib.npz
python starnav.py --convention 0 1          # if eval_photos says mirror True
```

Behaviour depends on whether `calib.npz` exists:

- **absent** → detection only. Outlines and IDs, green for mapped, red for detected but
  unmapped. This is the right first test: it confirms the camera, the dictionary and the
  map before any geometry is involved.
- **present** → adds pose. Overlay shows x, y, yaw, reprojection error, an accuracy
  estimate, and flags single-marker frames as low quality.

It degrades rather than guessing a focal length, because guessed intrinsics produce
plausible-looking positions that are wrong by however far the guess was off.

Functions worth knowing if you're reading the code:

| Function | Does |
|---|---|
| `FrameSource` | One interface over webcam index, video file, stream URL, and image folder. Applies focus/exposure locks to live cameras and prints what the driver actually accepted. |
| `make_detector` | Builds the ArUco detector from config, including sub-pixel corner refinement. |
| `split_known_unknown` | Separates detections into markers that are in the map and those that aren't. |
| `corner_offsets(half, rotation, mirror)` | The 8 candidate corner conventions. `mirror` is the handedness trap. |
| `build_correspondences` | N markers → 4N 3D–2D point pairs for one solve. |
| `pose_from_detections` | The whole pose step. Returns x, y, z, yaw, marker count, reprojection error, accuracy estimate. |
| `solve_multi` | Fused solve over every visible marker at once. |
| `solve_single` | One-marker fallback via `IPPE_SQUARE`. Ill-conditioned by nature — flagged low quality. |
| `accuracy_estimate` | Ground sampling distance × reprojection error ÷ √markers. Excludes tilt and survey error by construction. |

### `tools/calibrate.py` — intrinsics

```bash
python tools/calibrate.py --photos photos/calib --refine-win 9
```

Calibrates from photos of the tag grid and writes `calib.npz`. The grid is a better
calibration target than a printed chessboard here: its feature positions are exact,
every corner carries an ID so partial views still contribute, and it is the same target
through the same capture path as the actual test.

Reports RMS reprojection error, per-view errors, focal length, principal point,
distortion, and the spread of viewing angles. **Rejects above 0.5 px** — recapture rather
than proceed, since every position error downstream inherits it.

Shooting: 15–25 photos, one lens only, JPEG not HEIC, HDR and portrait mode off, same
resolution and orientation throughout, tilts from face-on out to 40–50°. A planar target
shot only face-on cannot separate focal length from distance; the script warns if your
tilt spread is under 20°.

`--free-k3` and `--rational` change the distortion model; you rarely want either.

### `tools/eval_photos.py` — handedness and error

```bash
python tools/eval_photos.py --photos photos/eval --refine-win 9
```

Answers two questions and writes `logs/eval_photos.csv`.

**Handedness.** ArUco's corner order maps to world object points in one of 8 ways —
4 mounting rotations × 2 windings. The script solves all 8 and ranks them by residual.
The right one lands at a fraction of a pixel; the rest are one to two orders of magnitude
worse, because no rigid camera pose can reproduce a reversed winding. Pass the winner to
`starnav.py --convention ROTATION MIRROR`. A margin under 3× means inconclusive — usually
too few markers per photo, or too little variety in viewing angle.

**Error.** There is no ground truth for where the phone was, but there is exact ground
truth for where every marker is, so markers are held out and predicted:

- *leave-one-out* — solve from all other markers, predict the held-out one. This is the
  **noise floor**, not the accuracy: the retained markers pin the pose so tightly that
  systematic error cancels. Expect true error several times this.
- *extrapolation* — solve from a small central cluster, predict outward. This is the one
  that exposes focal-length error, residual distortion and panel bow, and it reports
  whether the error grows with distance, which is the signature separating systematic
  error from noise.

Neither covers camera tilt or survey error. For an absolute check use `--offset-mm`.

Use **different photos** from the calibration set. Reusing them reports the fit, not the
accuracy.

### `tools/screen_tags.py` — the grid

See the section above.

### `tools/generate_tags.py` — printable tags

```bash
python tools/generate_tags.py --dpi 300
```

For when you move to a real ceiling. Renders each tag at an exact physical size with
ruler ticks, and warns if the sheet exceeds A4's printable area — printing an oversized
sheet makes the driver scale it, which corrupts `tag_size_m` invisibly. Print at 100 %,
no fit-to-page, and measure tick-to-tick before taping anything up.

### `tools/record_clip.py` — capture

```bash
python tools/record_clip.py --out clips/run1 --seconds 30 --delay 5
```

Records lossless PNG frames through `starnav.FrameSource`, so the same backend and the
same focus/exposure locks as a live run. PNG rather than video because every lossy codec
discards high-frequency detail, and a fiducial marker is almost entirely high-frequency
detail at its corners — the ringing lands exactly where sub-pixel refinement is looking.

### `tools/make_figures.py` — report figures

```bash
python tools/make_figures.py --clip clips/marker_test_v1.mp4
```

Writes `docs/figures/`: the target, a detection contact sheet, marker count over time,
the handedness comparison, and the world-frame diagram. Re-run after every capture — a
figure quietly belonging to an older run is worse than no figure. OpenCV drawing only,
no matplotlib.

---

## Config

### `config/hall.json`

| Key | Notes |
|---|---|
| `source` | Default input: webcam index as a string, video path, stream URL, or image folder. `--source` overrides. |
| `markers` | Path to the marker map. Switching between the screen rig and a real ceiling is a change here, not a flag on every command. |
| `camera.backend` | `auto`, `dshow`, `msmf`, `v4l2`, `ffmpeg`. On Windows the default MSMF often ignores property writes; try `dshow` if the startup readout says `IGNORED BY DRIVER`. |
| `camera.autofocus`, `focus`, `auto_exposure`, `exposure` | Locks. Autofocus hunting mid-run shifts the apparent tag size and looks exactly like algorithm failure. Values are raw driver values and backends disagree about them, hence the readout at startup. |
| `detection.corner_refinement` | `subpix` by default. Not cosmetic — corner error propagates roughly linearly into position. |
| `detection.corner_refine_win_size` | 5 suits webcam frames. Raise to ~9 for high-resolution stills, or pass `--refine-win`. |
| `viz.*` | Window name, colours, text layout. |
| `playback.wait_ms` | Frame delay. `loop_video` replays finite sources. |

### `config/markers*.json`

```json
{
  "tag_size_m": 0.044829,
  "ceiling_height_m": 0.75,
  "survey_uncertainty_m": 0.002989,
  "markers": { "0": [0.029886, 0.034556], "1": [0.089659, 0.034556] }
}
```

Marker centres in metres, world X/Y, all on the plane `Z = ceiling_height_m`.
`survey_uncertainty_m` is how well the marker positions themselves are known — a separate
row in the error budget, deliberately never folded into `acc_est_m`.

Metres everywhere. Never centimetres, never pixels, in any stored or transmitted value.

---

## Coordinate frames

**Hall:** X right, Y forward, Z up, origin at a marked corner. Markers on the ceiling at
`Z = ceiling_height_m`, facing down. The camera sits below and looks along +Z at them.

**Screen rig:** X screen-right, Y screen-**down**, Z into the screen. Y points down so the
frame stays right-handed once +Z is fixed running from camera to marker plane — the same
arrangement as the hall with the plane stood on its edge. Marker world XY is therefore
just the tag's position on the panel, with the origin at the top-left of the active area.

`solvePnP` returns world→camera. The camera in world coordinates is:

```python
R, _ = cv2.Rodrigues(rvec)
C = (-R.T @ tvec).ravel()
```

The corner-order mapping has a handedness trap. Do not reason about it — resolve it with
`tools/eval_photos.py`, which is what the empirical check in the spec asks for.

---

## Reading the output

| Field | Means |
|---|---|
| `reproj_px` | Mean reprojection error. Fit quality. A tilted camera reprojects its own wrong pose perfectly, so a small value does **not** mean a correct position. |
| `acc_est_m` | Ground sampling distance × reprojection error ÷ √markers. Excludes tilt and survey error. |
| `n_markers` | 1 means the ill-conditioned fallback ran; treat that frame as low quality. |
| `range_m` | Solved camera-to-plane distance. Tape-measure one shot and compare — a range off by a fixed fraction is a focal-length error, and it scales X and Y by the same fraction. |

Error at this range does not transfer to the hall directly. Both dominant terms scale
with range — corner noise as range/focal, tilt as range × tan — so convert to an angle
and multiply by 12 m. `eval_photos.py` prints that projection.

Camera tilt dominates at height: **1° of tilt at 12 m is about 21 cm of lateral error.**
Pixel resolution is comfortable by comparison. ±5 cm at 12 m needs levelled mounting, an
IMU, or multi-marker geometry constraining orientation — not more megapixels.

---

## Status

Built: detection, single-marker pose, multi-marker fused solve, accuracy estimate,
calibration, handedness resolution, error measurement, figures.

Not built yet: the hall map window (2D top-down plot with trail), per-frame CSV logging
from the live loop, and the UDP JSON publisher.

No smoothing anywhere, deliberately. Raw per-frame output only, so jitter stays visible
and measurable. A filter goes in after the noise floor is logged, not before.

---

## Gotchas

- **Stale `calib.npz`.** If calibration was rejected, delete the file. `starnav.py` will
  otherwise use it and report confident nonsense.
- **Resolution mismatch.** Intrinsics belong to one sensor readout. `eval_photos.py`
  hard-skips photos that don't match — "0 usable photos" usually means mixed lenses,
  orientations or resolutions.
- **HEIC.** OpenCV cannot read it. Shoot JPEG.
- **Phone video.** Compression plus stabilisation, which drifts the effective intrinsics
  frame to frame so no single calibration is valid. Turn stabilisation off, or use stills.
- **Overexposed panel.** Blown whites cut the detection rate hard. Screen brightness
  around 30–40 %, in a lit room, with camera exposure locked.
- **Reusing calibration photos for evaluation.** Reports the fit, not the accuracy.
