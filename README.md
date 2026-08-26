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

### Pose model: 6-DOF or 4-DOF

`pose.assume_level` in `hall.json` picks the solver, and `--assume-level 0|1` overrides it
for one run. The active choice is printed at startup, because it is an assertion about the
physical rig rather than a tuning knob.

- **6-DOF (`false`)** — `cv2.solvePnP` over every visible marker's corners at once, with
  `SOLVEPNP_IPPE_SQUARE` as the single-marker fallback. Makes no assumption about mounting.
- **4-DOF (`true`)** — for a mount that holds the optical axis normal to the marker plane.
  The world→image map is then a pure **similarity** — scale, in-plane rotation, two
  translations — which is linear and solvable in closed form. No iteration, no branch, no
  rotation ambiguity.

The 4-DOF path exists because a level mount is the *worst* case for the unconstrained
solver: a fronto-parallel square is exactly where `IPPE_SQUARE` degenerates, and `solvePnP`
spends three rotational degrees of freedom estimating a tilt that is known to be zero.
Measured on synthetic corners with 0.15 px of noise, median lateral error:

| | 6-DOF | 4-DOF | 4-DOF reproj |
|---|---|---|---|
| 1 tag, level | 6.20 mm | **0.05 mm** | 0.132 px |
| 6 tags, level | 0.43 mm | **0.02 mm** | 0.180 px |
| 1 tag, 0.5° tilt | 6.13 mm | 6.63 mm | 0.229 px |
| 6 tags, 0.5° tilt | 0.43 mm | 6.87 mm | 0.542 px |
| 6 tags, 2° tilt | 0.34 mm | 26.93 mm | 2.136 px |

Two things to read off that. On a genuinely level mount the constrained solve is one to two
orders of magnitude better, and it rescues the single-marker case that is otherwise
rotation-ambiguous. But when the promise is broken it degrades exactly as
`range × tan(tilt)` predicts — 0.5° at 0.75 m is 6.5 mm — while the 6-DOF path is unmoved,
because it models the tilt.

**The residual polices the assumption.** A tilted square images as a trapezoid and no
similarity can reproduce one, so `reproj_px` climbs with tilt in 4-DOF mode (0.18 → 0.54 →
2.14 above) while staying flat in 6-DOF mode. In this mode `reproj_px` is a *levelness
monitor*, not merely a fit statistic — and it is the only self-check the constrained solver
has. Its sensitivity falls with range: at 0.75 m a 1° tilt moves a 0.2 m tag's image by over
a pixel, but at 12 m a 0.24° tilt moves a 0.72 m tag by ~0.06 px, under the noise floor.
Close in it will catch a bad mount; at ceiling height it will not, and levelling has to be
verified mechanically.

For context on how tight that is: **±5 cm at 12 m needs tilt under 0.24°**, roughly 4 mm of
sag across a 1 m bracket.

### Measure your tilt, don't assume it

Every pose carries `tilt_deg` — the angle between the optical axis and the marker plane's
normal — and it appears as a column in the per-still table. Run 6-DOF on a capture and read
it off, rather than guessing whether the level assumption is defensible. The 4-DOF solve
costs `range × tan(tilt_deg)`, so the number converts directly into the error you would be
accepting.

Read it from a grid, not from one tag. Measured against known tilts:

| True tilt | From 1 tag | From 6 tags |
|---|---|---|
| 0.0° | 0.67° ±0.55 | 0.02° ±0.02 |
| 1.0° | 1.17° ±0.45 | 0.99° ±0.02 |
| 3.0° | 3.07° ±0.51 | 3.00° ±0.03 |
| 8.0° | 8.17° ±0.60 | 8.00° ±0.03 |

A single tag inherits its own rotation ambiguity and reads ~0.7° when the truth is zero, so
it cannot confirm a level mount. A grid is exact to a few hundredths of a degree.

**`assume_level` defaults to `false`**, because a handheld capture always has a few degrees
of tilt and the bench photos are handheld. Turn it on for a camera that is genuinely fixed
and levelled — the eventual Pi mount — not for photographs taken by hand.

One consolation for handheld work: a tilt that stays *constant* across a set of shots
produces a constant offset that cancels in any differential measurement. That is why the
handedness result above was valid despite being shot by hand — it reads `dx` between
positions, never absolute X.

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
python tools/calibrate.py --photos photos/calib --refine-win 9          # tag grid
python tools/calibrate.py --photos photos/calib_v2 \
    --chessboard 9x6 --square-mm 42.339                                 # chessboard
```

Calibrates from photos of a planar target and writes `calib.npz`. Two targets work:

- **Tag grid** (default). Feature positions are exact, every corner carries an ID so
  partial views still contribute, and it is the same target through the same capture path
  as the actual test.
- **Chessboard** (`--chessboard COLSxROWS --square-mm S`, counts are *inner* corners).
  Uses `findChessboardCornersSB`, falling back to the classic detector plus `cornerSubPix`.
  The whole board must be visible in every view, so frame-corner coverage has to come from
  moving the camera. That matters — distortion is largest at the frame edges and can only
  be estimated where there are corners.

The square size scales the recovered `tvecs` and nothing else: `fx`, `fy`, `cx`, `cy` and
the distortion coefficients come out identical whatever you pass. A wrong panel diagonal
costs nothing here — it costs later, in `markers.json`.

Reports RMS reprojection error, per-view errors, focal length, principal point,
distortion, and the spread of viewing angles. **PASS at ≤ 0.5 px**, *marginal* between 0.5
and 1.0 — check `worst` against the median, since a few bad photos are usually carrying it
— and **reject above 1.0 px**.

### RMS is necessary, not sufficient

A low RMS does not mean a good calibration. Verified against synthetic photos generated
from known intrinsics (fx 1256.0, cx 810.0, cy 590.0, k1 0.120, k2 −0.250):

| Capture | RMS | Recovered fx | Verdict |
|---|---|---|---|
| 15 views, 0–30° tilt spread | 0.102 px | **1255.9** (0.008% out) | PASS |
| 15 views, 0–3° tilt spread | 0.169 px | **1294.8** (3.1% out) | PASS, but *warned* |

The second one **passes the RMS gate while `fx` is 3.1% wrong** — and a 3.1% focal error
is a 3.1% error on every distance reported, forever. A planar target shot near face-on
cannot separate focal length from distance; the two trade off exactly, and the residual
stays small because the wrong answer still fits. This is why the script prints the tilt
spread and warns below 20°. **Read that line before believing the RMS.**

Shooting: 15–25 photos, one lens only, JPEG not HEIC, HDR and portrait mode off, same
resolution and orientation throughout, tilts from face-on out to 40–50° in several
directions, plus a couple of rolls.

`--free-k3` and `--rational` change the distortion model; you rarely want either.

### `tools/chessboard.py` — chessboard target

```bash
python tools/chessboard.py                       # 9x6 inner corners
python tools/chessboard.py --inner 7x5 --margin-mm 30
```

Displays a chessboard fullscreen at native resolution and prints the square size in mm,
then the exact `calibrate.py` command to run. On the 27" 1920×1080 panel, 9×6 inner
corners gives **42.339 mm squares at 136 px**, snapped to whole pixels so no square is
resampled unevenly across its face — uneven resampling biases sub-pixel corner refinement,
which is the one thing calibration cannot tolerate.

Inner-corner counts must differ (9×6, not 6×6): a square board is rotationally ambiguous
and the corner ordering can flip between views, silently corrupting correspondences. The
script refuses a square pattern.

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

### `tools/eval_shift.py` — handedness from a fixed camera

Two runs, because the camera is a phone rather than a webcam:

```bash
python tools/eval_shift.py --display --offsets-mm 0 100 -100
python tools/eval_shift.py --photos photos/handedness --offsets-mm 0 100 -100 \
    --assume-hfov-deg 65
```

`--display` puts each tag position on screen at native size and waits for a keypress, so
you photograph each one without moving the camera. `--photos` reads them back in filename
order, pairs them with the offsets, and prints the sign verdict.

Needs no calibration. The sign falls out of the corner ordering and the local→world
composition, so a guessed focal length gets the direction right while getting every
distance wrong — and the tool says so on every line it prints in that mode.

Prefer the default 3×2 grid over a single tag. One square is rotation-ambiguous, and about
a degree of rotation error at 0.75 m is over a centimetre of lateral error; a grid pins
rotation through the shared plane. Measured in simulation on a 100 mm shift:

| Grid | X error at rest | Worst shift error |
|---|---|---|
| 1×1, 200 mm | 17.1 mm | 54.0 mm |
| 3×2, 90 mm | 1.2 mm | 1.7 mm |

A mirrored capture path does not produce a wrong sign — `cv2.aruco` will not detect a
mirrored tag at all, so it fails at detection. Use the phone's rear camera, and shoot JPEG:
OpenCV cannot read HEIC.

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
| `pose.assume_level` | `true` constrains the solve to 4 DOF for a mount whose optical axis is normal to the marker plane; `false` uses full 6-DOF `solvePnP`. An assertion about the rig — see *Pose model* above. |
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

The corner-order mapping has a handedness trap. Do not reason about it — resolve it
empirically, with `tools/eval_shift.py` or `tools/eval_photos.py`.

### Measured handedness — screen rig, 2026-08-24

| Axis | Maps to | Status |
|---|---|---|
| World **+X** | screen right, as seen by a viewer facing the panel | **verified by displacement** |
| World **+Y** | screen **down** | not displacement-tested — see below |
| World **+Z** | into the screen, away from the camera | sign confirmed: solved camera Z is negative |

**Verdict: world X is not mirrored.** `starnav.corner_offsets` rotation 0, mirror False
holds for this rig, which is the default `--convention 0 0`.

**How it was verified.** The camera was fixed and the *grid* was displaced a known amount,
rather than moving the camera against a tape measure — a shift of a whole number of screen
pixels is known to a fraction of a millimetre and costs nothing to repeat. Six tags
(3×2, 89.66 mm) were photographed at three positions from one fixed iPhone, each step
solved against the single map written at offset 0. The solver therefore believes the tags
never moved, so a pattern sliding 100 mm right can only be explained by a camera sliding
100 mm left:

```
 shift_mm   n  tags      x_mm    dx_mm  expected  error_mm  x_upd_mm  reproj
     0.00   1     6    237.17     0.00     -0.00      0.00    237.17   2.721
    99.93   1     6    137.88   -99.28    -99.93      0.65    237.82   3.157
   -99.93   1     6    303.64    66.48     99.93    -33.46    203.71   2.753

Z (panel distance) : -815.2, -811.8, -815.5 mm
```

Both directions agree on the sign, and the margin is comfortable: flipping the verdict
would need errors of 99 mm and 66 mm against a worst observed 33 mm.

**What this does not establish.** Three things, deliberately:

- **Y was never displaced.** `eval_shift.py` shifts X only, so the Y row above is inherited
  from how `screen_tags.py` writes the map, plus the synthetic assertion in
  `test_starnav.py`. It has not been confirmed against a real photograph.
- **The millimetres are not metric.** This ran in sign-only mode with no `calib.npz`,
  assuming zero distortion and a focal length guessed from a 65° field of view. The
  −33.46 mm outlier and the ~2.7 px residuals are that missing distortion model, not
  algorithm error; in simulation with correct intrinsics the same test residuals were
  0.21–0.25 px. The verdict is robust to the guess — it came out identical for every
  assumed field of view from 35° to 110°.
- **Ceiling markers are still unverified.** This is a vertical panel viewed from the front.
  Re-run the check with tags overhead before trusting the sign in the hall.

The steady Z across all three shots (3.7 mm spread) is what rules out the phone being
knocked between exposures.

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

Handedness is **resolved and recorded** — see *Measured handedness* under Coordinate
frames. World +X is screen-right and not mirrored, verified by displacing the grid a known
100 mm with the camera fixed. Y has not been displacement-tested.

Pointing `--source` at a folder of stills prints one `x_m y_m z_m` row per photo, which is
the readout the sign check is made from. `python test_starnav.py` runs the self-checks:
the single-marker sign convention, and the fronto-parallel degeneracy that `IPPE_SQUARE`
falls into at exactly 0° tilt.

Not built yet: the hall map window (2D top-down plot with trail), per-frame CSV logging
from the live loop, and the UDP JSON publisher. `calib.npz` was deleted — the previous one
was derived from video frames and its 1.63 px RMS failed the 0.5 px gate — so nothing
metric runs until a fresh calibration exists.

The pose solver is selectable: 6-DOF `solvePnP`, or a 4-DOF similarity fit for a level
mount (`pose.assume_level`, currently **true**). The evaluation rig assumes no tilt; tilt
handling comes later, and the 4-DOF residual is what flags the assumption breaking in the
meantime.

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
