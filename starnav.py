#!/usr/bin/env python3
"""Star Navigation v1 - ceiling-marker indoor positioning.

An upward-facing camera moves beneath a ceiling of ArUco markers whose world
positions are known in advance (config/markers.json). From the marker
detections we recover the *camera's* position in the world frame.

BUILD STAGES 1-3 OF 5.
    1. Detection - marker outlines and IDs on the feed. Confirms the source
       opens, the dictionary matches the printed tags, and the IDs overhead
       match the IDs in the map. Every pose bug is cheaper to find after this.
    2. Single-marker pose, with the handedness of the image-corner to
       world-point mapping resolved empirically by tools/eval_photos.py.
    3. Multi-marker fused solve - all visible markers' corners into one
       solvePnP.
    Stages 4 (hall map window, per-frame CSV) and 5 (quality metrics in the
    live output) are still to come. `acc_est_m` already exists here and is
    exercised by tools/eval_photos.py.

    Pose is skipped and the loop runs as stage 1 if no calib.npz is present -
    there is no metric geometry without intrinsics, and guessing them would
    produce confident nonsense.

Coordinate conventions:
    World  - X right, Y forward, Z up, origin at a marked corner, metres.
    Camera - OpenCV convention, X right, Y down, Z along the optical axis.
    All markers lie on the ceiling plane at Z = ceiling_height_m, facing down.
    The camera sits on the -Z side of that plane and looks along +Z towards it.
    (The screen rig in tools/screen_tags.py is the same arrangement with the
    plane stood on its edge; see that file for how the axes map.)

Usage:
    python starnav.py                        # source from config/hall.json
    python starnav.py --source 0             # webcam index
    python starnav.py --source clip.mp4      # video file (reproducible runs)
    python starnav.py --source frames/       # folder of images
    python starnav.py --markers config/markers_screen.json --calib calib.npz
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Keys are UI, not tunables, so they stay here rather than in the config file.
KEY_QUIT = (ord("q"), 27)  # q or ESC
KEY_PAUSE = ord(" ")
KEY_SNAPSHOT = ord("s")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Named capture backends. On Windows the default (MSMF) is slow to open and
# silently ignores most property writes; DSHOW usually honours focus/exposure
# locks. Which one works is a per-machine fact, so the choice lives in config.
CAPTURE_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "ffmpeg": cv2.CAP_FFMPEG,
}

CORNER_REFINEMENT = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_marker_map(path: Path) -> dict:
    """Read markers.json and normalise it.

    JSON object keys are always strings; ArUco returns integer IDs. Converting
    once here means the per-frame code can compare IDs directly instead of
    stringifying inside the loop.
    """
    raw = load_json(path)
    for key in ("tag_size_m", "ceiling_height_m", "markers"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key '{key}'")

    positions = {}
    for marker_id, xy in raw["markers"].items():
        if len(xy) != 2:
            raise ValueError(
                f"{path}: marker {marker_id} must be [x, y] in metres "
                f"(Z is fixed at ceiling_height_m for every marker)"
            )
        positions[int(marker_id)] = (float(xy[0]), float(xy[1]))

    return {
        "tag_size_m": float(raw["tag_size_m"]),
        "ceiling_height_m": float(raw["ceiling_height_m"]),
        # Carried through for the report's error budget. This is a property of
        # how the tags were surveyed, never of the algorithm, so it stays a
        # separate number and is deliberately not folded into acc_est_m.
        "survey_uncertainty_m": float(raw.get("survey_uncertainty_m", 0.0)),
        "markers": positions,
    }


# --------------------------------------------------------------------------
# Video input
# --------------------------------------------------------------------------

class FrameSource:
    """One interface over the three input kinds the spec requires.

    Webcam index, video file (or stream URL), and image folder all have to
    work: the ceiling rig is not always available, and replaying a recorded
    clip is the only way to get a repeatable number out of a run.

    `is_live` matters downstream - focus and exposure locks only apply to a
    real camera, and only a live feed can drop frames.
    """

    def __init__(self, spec, camera_cfg: dict):
        self.spec = str(spec)
        self.kind = "unknown"
        self.is_live = False
        self._capture = None
        self._image_paths = []
        self._image_index = 0

        path = Path(self.spec)
        if self.spec.isdigit():
            self._open_camera(int(self.spec), camera_cfg)
        elif path.is_dir():
            self._open_image_folder(path)
        else:
            self._open_video(self.spec)

    # -- webcam ------------------------------------------------------------

    def _open_camera(self, index: int, camera_cfg: dict) -> None:
        backend_name = camera_cfg.get("backend", "auto")
        if backend_name not in CAPTURE_BACKENDS:
            raise ValueError(f"unknown camera backend '{backend_name}'")
        self._capture = cv2.VideoCapture(index, CAPTURE_BACKENDS[backend_name])
        if not self._capture.isOpened():
            raise RuntimeError(
                f"could not open webcam index {index} via backend "
                f"'{backend_name}' - try another index or backend in hall.json"
            )
        self.is_live = True
        self.kind = f"webcam {index} ({backend_name})"
        self._apply_camera_settings(camera_cfg)

    def _apply_camera_settings(self, camera_cfg: dict) -> None:
        """Lock resolution, focus and exposure, then report what actually stuck.

        Autofocus hunting changes the principal distance mid-run, which changes
        the apparent tag size and therefore the recovered range and position.
        It looks exactly like algorithm failure. Auto-exposure does the same
        thing more subtly, by varying motion blur on the corners.

        The locks are attempted, not trusted: most backends accept the write
        and ignore it, so every property is read back and printed. A run whose
        intrinsics are invalid should be obvious at startup, not in the report.
        """
        # Order matters: autofocus must be disabled before a focus value is
        # accepted, and likewise auto_exposure before exposure.
        properties = [
            ("frame_width", cv2.CAP_PROP_FRAME_WIDTH),
            ("frame_height", cv2.CAP_PROP_FRAME_HEIGHT),
            ("autofocus", cv2.CAP_PROP_AUTOFOCUS),
            ("focus", cv2.CAP_PROP_FOCUS),
            ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
            ("exposure", cv2.CAP_PROP_EXPOSURE),
        ]
        print("camera settings (requested -> actual):")
        for name, prop in properties:
            wanted = camera_cfg.get(name)
            if wanted is None:
                continue
            self._capture.set(prop, float(wanted))
            actual = self._capture.get(prop)
            status = "ok" if abs(actual - float(wanted)) < 1e-6 else "IGNORED BY DRIVER"
            print(f"  {name:<13} {wanted} -> {actual} [{status}]")

    # -- video file / stream ----------------------------------------------

    def _open_video(self, spec: str) -> None:
        self._capture = cv2.VideoCapture(spec)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source '{spec}'")
        # A network stream (phone camera) is live even though it opens like a
        # file. Calibration for such a source has to be captured through the
        # same streaming path: re-encoding, cropping and digital stabilisation
        # all change the intrinsics without changing anything visible.
        self.is_live = "://" in spec
        self.kind = f"{'stream' if self.is_live else 'video file'} '{spec}'"

    # -- image folder ------------------------------------------------------

    def _open_image_folder(self, folder: Path) -> None:
        # Sorted by filename: an image sequence has no timestamps of its own,
        # so lexical order is the only ordering available. Zero-pad exported
        # frame numbers or the sequence will play out of order.
        self._image_paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._image_paths:
            raise RuntimeError(
                f"no images in '{folder}' (looked for {IMAGE_EXTENSIONS})"
            )
        self.kind = f"image folder '{folder}' ({len(self._image_paths)} frames)"

    # -- common ------------------------------------------------------------

    def read(self):
        """Return the next frame, or None when the source is exhausted."""
        if self._capture is not None:
            ok, frame = self._capture.read()
            return frame if ok else None

        if self._image_index >= len(self._image_paths):
            return None
        frame = cv2.imread(str(self._image_paths[self._image_index]))
        self._image_index += 1
        return frame

    def rewind(self) -> bool:
        """Restart a finite source. False if this source cannot be replayed."""
        if self._image_paths:
            self._image_index = 0
            return True
        if self._capture is not None and not self.is_live:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        return False

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def make_detector(detection_cfg: dict) -> cv2.aruco.ArucoDetector:
    """Build the ArUco detector.

    DICT_4X4_50 by config: the smallest dictionary that covers the tag count
    here. Fewer bits per tag means larger, more robust bit cells for a given
    printed size, which is what matters when the tag is a small blob at range.

    Sub-pixel corner refinement is on by default and is not cosmetic. Corner
    localisation error propagates roughly linearly into the recovered position:
    at the real 12 m ceiling one pixel is about 3 mm on the ceiling plane, so
    the difference between whole-pixel and ~0.1 px corners is the difference
    between a millimetres-level and a sub-millimetre contribution to the error
    budget. It costs a few percent of frame time and is measured, not assumed.
    """
    dictionary_name = detection_cfg["dictionary"]
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary '{dictionary_name}'")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    params = cv2.aruco.DetectorParameters()
    refinement = detection_cfg["corner_refinement"]
    if refinement not in CORNER_REFINEMENT:
        raise ValueError(f"unknown corner_refinement '{refinement}'")
    params.cornerRefinementMethod = CORNER_REFINEMENT[refinement]
    params.cornerRefinementWinSize = int(detection_cfg["corner_refine_win_size"])
    params.cornerRefinementMaxIterations = int(detection_cfg["corner_refine_max_iterations"])
    params.cornerRefinementMinAccuracy = float(detection_cfg["corner_refine_min_accuracy"])

    return cv2.aruco.ArucoDetector(dictionary, params)


def split_known_unknown(corners, ids, known_ids):
    """Partition detections into markers that are in the map and those that are not.

    Algorithm step 2 is "discard any detected ID not present in markers.json":
    an unmapped tag has no world position, so it cannot contribute a
    correspondence. They are kept separately rather than dropped outright so
    the operator can see a tag that is physically on the ceiling but missing
    from the map, which is the most common surveying mistake.
    """
    known, unknown = [], []
    if ids is None:
        return known, unknown
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        bucket = known if int(marker_id) in known_ids else unknown
        bucket.append((int(marker_id), marker_corners))
    return known, unknown


# --------------------------------------------------------------------------
# Pose  (build stages 2 and 3)
# --------------------------------------------------------------------------

def load_calibration(path: Path):
    """Read camera_matrix and dist_coeffs from calib.npz."""
    data = np.load(str(path))
    camera_matrix = data["camera_matrix"].astype(np.float64)
    dist_coeffs = data["dist_coeffs"].astype(np.float64).ravel()
    image_size = tuple(int(v) for v in data["image_size"]) if "image_size" in data else None
    return camera_matrix, dist_coeffs, image_size


def corner_offsets(half: float, rotation: int = 0, mirror: bool = False) -> np.ndarray:
    """XY offsets of a marker's four corners, in the order ArUco reports them.

    ArUco returns corners clockwise *as printed on the tag*, starting at the
    tag's own top-left. Corner index 0 is therefore a fixed physical corner,
    independent of how the camera is rotated - so the only freedom here is how
    the tag was physically placed:

      `rotation` - the tag was mounted turned by k * 90 degrees. Cyclically
                   shifts which physical corner is index 0.
      `mirror`   - the winding is reversed. This is the handedness trap. It is
                   fixed by geometry, not by choice: the winding that appears
                   clockwise in the image depends on which side of the marker
                   plane the camera sits on. Get it wrong and the solve does
                   not fail quietly, it fails with a huge reprojection error
                   (see tools/eval_photos.py, which searches all 8 combinations
                   and reports the residual for each).

    The base ordering assumes world +Y runs the same way the tag's printed
    "down" does, and the camera sits on the -Z side of the marker plane, which
    is the case for both the hall (camera below, ceiling above) and the screen
    rig (camera in front, +Z into the screen).
    """
    base = [(-half, -half), (+half, -half), (+half, +half), (-half, +half)]
    if mirror:
        base = [(-x, y) for x, y in base]
    rotation %= 4
    return np.array(base[rotation:] + base[:rotation], dtype=np.float64)


def build_correspondences(detections, marker_map: dict, offsets: np.ndarray):
    """Turn detected markers into 3D-2D point pairs for a single solve.

    N markers give 4N pairs, all fed to one solvePnP. Solving each marker
    separately and averaging would throw away exactly the thing that makes the
    multi-marker case well conditioned: the markers' positions *relative to
    each other* are known, so their combined outline constrains camera
    orientation far more tightly than any single tag's four corners can.
    """
    height = marker_map["ceiling_height_m"]
    object_points, image_points, used_ids = [], [], []
    for marker_id, corners in detections:
        centre_x, centre_y = marker_map["markers"][marker_id]
        for (dx, dy), image_corner in zip(offsets, corners.reshape(4, 2)):
            # Every marker lies on the same plane, Z = ceiling_height. That
            # shared plane is the planar structure the solver exploits.
            object_points.append((centre_x + dx, centre_y + dy, height))
            image_points.append(image_corner)
        used_ids.append(marker_id)
    return (np.array(object_points, dtype=np.float64),
            np.array(image_points, dtype=np.float64),
            used_ids)


def reprojection_error(object_points, image_points, rvec, tvec,
                       camera_matrix, dist_coeffs) -> float:
    """Mean pixel distance between observed and reprojected corners."""
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))


def solve_multi(object_points, image_points, camera_matrix, dist_coeffs):
    """Fused solve over every visible marker's corners.

    SOLVEPNP_ITERATIVE is used deliberately: for a coplanar point set OpenCV
    initialises it from a homography - the closed-form planar solution - and
    then refines by Levenberg-Marquardt. That homography init is where the
    known common plane buys its conditioning, so this is a planar solver in one
    call rather than a generic one that happens to be given planar data.
    """
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix,
                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec) if ok else (None, None)


def solve_single(marker_id: int, corners, marker_map: dict,
                 camera_matrix, dist_coeffs):
    """Fallback when only one marker is visible. Ill-conditioned by nature.

    This assumes the nominal corner convention (rotation 0, no mirror). The
    convention search in tools/eval_photos.py uses multi-marker frames only,
    which is the right place to resolve it anyway: a single square is
    self-similar under the mirror, so it cannot distinguish the two.

    SOLVEPNP_IPPE_SQUARE will not accept the world-frame object points used
    above: it requires the marker's own canonical local square, centred on the
    origin in the Z=0 plane, in the order (-L,+L), (+L,+L), (+L,-L), (-L,-L).
    Feeding it world points silently produces a badly wrong pose with a large
    residual. So the pose is solved in the marker's local frame and then
    composed with where that marker sits in the world.

    Note the local frame has +Y "up" the printed tag while the world frame has
    +Y running the other way, hence the diag(1,-1,-1) - a 180 degree turn about
    X, which is the same flip that puts the tag face-on to a camera on the -Z
    side of the plane.
    """
    half = marker_map["tag_size_m"] / 2.0
    local = np.array([[-half, +half, 0.0], [+half, +half, 0.0],
                      [+half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)
    image_points = corners.reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(local, image_points, camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None

    rotation_local_to_camera, _ = cv2.Rodrigues(rvec)
    rotation_local_to_world = np.diag([1.0, -1.0, -1.0])
    centre_x, centre_y = marker_map["markers"][marker_id]
    translation_local_to_world = np.array(
        [[centre_x], [centre_y], [marker_map["ceiling_height_m"]]], dtype=np.float64)

    rotation_world_to_camera = rotation_local_to_camera @ rotation_local_to_world.T
    translation_world_to_camera = tvec - rotation_world_to_camera @ translation_local_to_world
    rvec_world, _ = cv2.Rodrigues(rotation_world_to_camera)
    return rvec_world, translation_world_to_camera


def pose_from_detections(detections, marker_map: dict, camera_matrix, dist_coeffs,
                         offsets: np.ndarray):
    """Camera pose in world coordinates, or None if nothing usable is visible."""
    if not detections:
        return None

    object_points, image_points, used_ids = build_correspondences(
        detections, marker_map, offsets)

    single = len(used_ids) == 1
    if single:
        marker_id, corners = detections[0]
        rvec, tvec = solve_single(marker_id, corners, marker_map,
                                  camera_matrix, dist_coeffs)
    else:
        rvec, tvec = solve_multi(object_points, image_points, camera_matrix, dist_coeffs)
    if rvec is None:
        return None

    # solvePnP returns the world -> camera transform. The camera's position in
    # the world is the origin of the camera frame expressed in world
    # coordinates, which is the inverse transform applied to zero.
    rotation, _ = cv2.Rodrigues(rvec)
    camera_in_world = (-rotation.T @ tvec).ravel()

    # Heading of the camera's image-"up" axis, projected into the marker plane.
    # The optical axis itself points at the ceiling and projects to nothing, so
    # it cannot serve as a heading; image-up is the axis a vehicle would call
    # forward.
    image_up_in_world = rotation.T @ np.array([0.0, -1.0, 0.0])
    yaw_deg = float(np.degrees(np.arctan2(image_up_in_world[1], image_up_in_world[0])))

    error_px = reprojection_error(object_points, image_points, rvec, tvec,
                                  camera_matrix, dist_coeffs)
    range_m = abs(marker_map["ceiling_height_m"] - camera_in_world[2])

    return {
        "x": float(camera_in_world[0]),
        "y": float(camera_in_world[1]),
        "z": float(camera_in_world[2]),
        "yaw_deg": yaw_deg,
        "n_markers": len(used_ids),
        "ids": used_ids,
        "reproj_px": error_px,
        "range_m": range_m,
        "single_marker": single,
        "acc_est_m": accuracy_estimate(error_px, len(used_ids), range_m, camera_matrix),
        "rvec": rvec,
        "tvec": tvec,
    }


def accuracy_estimate(error_px: float, n_markers: int, range_m: float,
                      camera_matrix) -> float:
    """Coarse per-frame position uncertainty in metres, derived not asserted.

    Ground sampling distance is how many metres on the marker plane one pixel
    covers: range / focal length. Multiplying by the mean reprojection error
    converts the residual the solver actually saw into a distance. Dividing by
    sqrt(n_markers) reflects that independent corner noise averages down across
    markers.

    The range used is the solved camera-to-plane distance rather than the
    nominal ceiling height, because that is the distance the pixels were
    actually projected over; the two agree when the camera sits near Z = 0.

    This deliberately EXCLUDES marker placement survey error and camera tilt.
    Tilt in particular dominates at height - one degree at 12 m is about 21 cm
    of lateral error, which no reprojection residual will reveal, because a
    tilted camera reprojects its own wrong pose perfectly. Both belong in the
    report's error budget as separate rows.
    """
    focal_px = float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0)
    ground_sampling_distance = range_m / focal_px
    return float(ground_sampling_distance * error_px / math.sqrt(max(1, n_markers)))


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

class FpsMeter:
    """Frame rate over a sliding window.

    A window rather than an instantaneous 1/dt: single-frame timings on a
    webcam are dominated by USB jitter and are unreadable on screen.
    """

    def __init__(self, window_frames: int):
        self._stamps = deque(maxlen=max(2, int(window_frames)))

    def tick(self) -> None:
        self._stamps.append(time.perf_counter())

    def value(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


def draw_detections(frame, detections, color_bgr) -> None:
    """Outline each detected marker and label it with its ID."""
    if not detections:
        return
    # drawDetectedMarkers wants the shapes detectMarkers produced, and writes
    # the ID next to each tag for us.
    corners = [marker_corners for _, marker_corners in detections]
    ids = np.array([[marker_id] for marker_id, _ in detections])
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, tuple(color_bgr))


def draw_text_block(frame, lines, viz_cfg) -> None:
    """Overlay status text, drawn twice for legibility.

    A ceiling view is mostly bright, so plain white text disappears against it.
    A dark pass underneath is cheaper than compositing a background panel.
    """
    x, y = viz_cfg["text_origin_px"]
    scale = viz_cfg["font_scale"]
    step = viz_cfg["text_line_height_px"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, line in enumerate(lines):
        origin = (x, y + index * step)
        cv2.putText(frame, line, origin, font, scale,
                    tuple(viz_cfg["text_shadow_bgr"]), 3, cv2.LINE_AA)
        cv2.putText(frame, line, origin, font, scale,
                    tuple(viz_cfg["text_color_bgr"]), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Star Navigation v1 - detection and pose from ceiling markers"
    )
    parser.add_argument("--hall", default="config/hall.json", type=Path,
                        help="runtime config: source, markers path, camera, detector, viz")
    parser.add_argument("--markers", default=None, type=Path,
                        help="marker map: tag size, ceiling height, marker world positions. "
                             "Defaults to hall.json's 'markers' key, so switching between "
                             "the screen rig and the real ceiling is a config edit rather "
                             "than a flag on every command.")
    parser.add_argument("--source", default=None,
                        help="override hall.json source: webcam index, video path, "
                             "stream URL, or image folder")
    parser.add_argument("--calib", default="calib.npz", type=Path,
                        help="intrinsics; without them the loop runs detection only")
    parser.add_argument("--convention", nargs=2, type=int, default=(0, 0),
                        metavar=("ROTATION", "MIRROR"),
                        help="corner convention, as resolved by tools/eval_photos.py")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hall = load_json(args.hall)
    markers_path = args.markers or Path(hall.get("markers", "config/markers.json"))
    marker_map = load_marker_map(markers_path)
    viz_cfg = hall["viz"]
    playback_cfg = hall["playback"]

    known_ids = set(marker_map["markers"])
    source_spec = args.source if args.source is not None else hall["source"]
    source = FrameSource(source_spec, hall["camera"])
    detector = make_detector(hall["detection"])
    fps = FpsMeter(viz_cfg["fps_window_frames"])

    snapshot_dir = Path(hall["output"]["snapshot_dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Pose needs intrinsics. Rather than substitute a guessed focal length -
    # which would produce plausible-looking positions that are wrong by
    # whatever fraction the guess was off - the loop degrades to stage 1.
    camera_matrix = dist_coeffs = None
    offsets = corner_offsets(marker_map["tag_size_m"] / 2.0,
                             int(args.convention[0]), bool(args.convention[1]))
    if args.calib.exists():
        camera_matrix, dist_coeffs, calib_size = load_calibration(args.calib)
        captured_at = f" (captured at {calib_size[0]}x{calib_size[1]})" if calib_size else ""
        print(f"calibration : {args.calib}{captured_at}")
    else:
        print(f"calibration : {args.calib} not found - DETECTION ONLY, no pose")

    print(f"source      : {source.kind}")
    print(f"dictionary  : {hall['detection']['dictionary']}")
    print(f"mapped tags : {sorted(known_ids)}")
    # SCALED-DEMO: the rig is a ~2.5 m ceiling with 15 cm tags, chosen to hold
    # angular size roughly constant against the real case - a 15 cm tag at
    # 2.5 m subtends about the same angle as a 72 cm tag at 12 m. On real
    # hardware only these two config numbers change; no code path differs.
    print(f"tag size    : {marker_map['tag_size_m']} m at "
          f"{marker_map['ceiling_height_m']} m ceiling")
    print("keys        : q/ESC quit, space pause, s snapshot")

    frame = None
    frame_index = 0
    paused = False
    try:
        while True:
            if not paused:
                frame = source.read()
                if frame is None:
                    if playback_cfg["loop_video"] and source.rewind():
                        continue
                    print(f"source exhausted after {frame_index} frames")
                    break

                fps.tick()
                # Detection runs on greyscale; the colour frame is display only.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _rejected = detector.detectMarkers(gray)
                known, unknown = split_known_unknown(corners, ids, known_ids)

                draw_detections(frame, known, viz_cfg["known_color_bgr"])
                draw_detections(frame, unknown, viz_cfg["unknown_color_bgr"])

                lines = [
                    f"frame {frame_index}   fps {fps.value():5.1f}",
                    f"mapped   {len(known)}: {[i for i, _ in known]}",
                    f"unmapped {len(unknown)}: {[i for i, _ in unknown]}",
                ]
                if camera_matrix is not None:
                    pose = pose_from_detections(known, marker_map, camera_matrix,
                                                dist_coeffs, offsets)
                    if pose is None:
                        lines.append("no pose: no mapped markers in view")
                    else:
                        lines.append(f"x {pose['x']:+.3f}  y {pose['y']:+.3f}  "
                                     f"yaw {pose['yaw_deg']:+6.1f}")
                        lines.append(f"reproj {pose['reproj_px']:.2f} px   "
                                     f"acc~{pose['acc_est_m'] * 1000:.0f} mm"
                                     + ("   SINGLE MARKER - low quality"
                                        if pose["single_marker"] else ""))
                else:
                    lines.append("no calib.npz: detection only, no pose")
                draw_text_block(frame, lines, viz_cfg)

                cv2.imshow(viz_cfg["camera_window"], frame)
                frame_index += 1

            key = cv2.waitKey(playback_cfg["wait_ms"]) & 0xFF
            if key in KEY_QUIT:
                break
            if key == KEY_PAUSE:
                paused = not paused
            elif key == KEY_SNAPSHOT and frame is not None:
                path = snapshot_dir / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
