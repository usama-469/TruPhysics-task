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
       world-point mapping resolved empirically - by tools/eval_photos.py on
       multi-marker photos, or by tools/eval_shift.py from a fixed camera.
       Pointing --source at a folder of stills prints one X/Y/Z row per photo,
       which is the readout the sign check is made from.
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
import csv
import json
import math
import socket
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
        # Stills are a measurement set - a handful of deliberate photographs -
        # so each one's pose is printed as a table row. A live feed or a video
        # would flood the terminal and its record is the window, not stdout.
        self.is_stills = False
        self.frame_name = ""
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
        self.is_stills = True
        self.kind = f"image folder '{folder}' ({len(self._image_paths)} frames)"

    # -- common ------------------------------------------------------------

    def read(self):
        """Return the next frame, or None when the source is exhausted."""
        if self._capture is not None:
            ok, frame = self._capture.read()
            return frame if ok else None

        if self._image_index >= len(self._image_paths):
            return None
        path = self._image_paths[self._image_index]
        self.frame_name = path.name
        frame = cv2.imread(str(path))
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


def solve_level(object_points, image_points, camera_matrix, dist_coeffs):
    """4-DOF pose for a camera whose optical axis is normal to the marker plane.

    When the mount guarantees no tilt, the only remaining freedoms are the two
    lateral offsets, the range, and the rotation about the optical axis. The
    world-to-image map is then a pure SIMILARITY - scale, in-plane rotation,
    translation - rather than a homography:

        [u - cx]         [ cos t   sin t ] [X - Cx]
        [v - cy] = (f/d) [-sin t   cos t ] [Y - Cy]

    which is linear in (a, b, tx, ty) = (s cos t, s sin t, ...) and solvable in
    closed form. That is the whole point. A general solvePnP would spend three
    rotational degrees of freedom estimating a tilt that is known to be zero,
    and a fronto-parallel square is precisely where SOLVEPNP_IPPE_SQUARE is
    degenerate - so on a level rig the unconstrained solver is at its WORST
    exactly where this one is at its best.

    Distortion has to be removed first: it is the one part of the imaging model
    that is not a similarity, so leaving it in would be absorbed into the fit as
    a wrong scale and a wrong centre.

    The fit is overdetermined even for a single marker - four corners are eight
    equations against four unknowns - so the residual still carries information.
    A tilted square images as a trapezoid, which no similarity can reproduce, so
    reproj_px becomes a TILT DETECTOR here rather than merely a fit statistic.
    Its sensitivity falls with range: at 0.75 m a 1 degree tilt perturbs a 0.2 m
    tag's image by over a pixel, but at 12 m a 0.24 degree tilt moves a 0.72 m
    tag by ~0.06 px, which is under the noise floor. Close in it will catch a
    bad mount; at ceiling height it will not, and levelling has to be verified
    mechanically.
    """
    # Every marker lies on one plane, so a single Z serves for the whole solve.
    plane_z = float(object_points[0, 2])
    focal_px = float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0)

    ideal = cv2.undistortPoints(image_points.reshape(-1, 1, 2), camera_matrix,
                                dist_coeffs, P=camera_matrix).reshape(-1, 2)
    u = ideal[:, 0] - camera_matrix[0, 2]
    v = ideal[:, 1] - camera_matrix[1, 2]
    world_x, world_y = object_points[:, 0], object_points[:, 1]

    # u =  a*X + b*Y + tx
    # v = -b*X + a*Y + ty
    rows = len(world_x)
    design = np.zeros((2 * rows, 4), dtype=np.float64)
    rhs = np.empty(2 * rows, dtype=np.float64)
    design[0::2, 0] = world_x
    design[0::2, 1] = world_y
    design[0::2, 2] = 1.0
    rhs[0::2] = u
    design[1::2, 0] = world_y
    design[1::2, 1] = -world_x
    design[1::2, 3] = 1.0
    rhs[1::2] = v

    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    a, b, tx, ty = solution
    scale = math.hypot(a, b)
    if scale <= 1e-9:
        return None, None

    angle = math.atan2(b, a)
    rotation_2d = np.array([[math.cos(angle), math.sin(angle)],
                            [-math.sin(angle), math.cos(angle)]])
    # scale = focal / distance-to-plane, so the range falls straight out of it.
    distance = focal_px / scale
    centre_xy = -np.linalg.inv(rotation_2d) @ np.array([tx, ty]) / scale

    # Rebuild the world->camera pair the rest of the pipeline expects, so the
    # inversion, the yaw and the reprojection error are computed by exactly the
    # same code as the unconstrained path - including projecting through the
    # FULL model, which is what lets the residual reveal a violated assumption.
    rotation = np.array([[math.cos(angle), math.sin(angle), 0.0],
                         [-math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    camera_in_world = np.array([centre_xy[0], centre_xy[1], plane_z - distance])
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = (-rotation @ camera_in_world).reshape(3, 1)
    return rvec, tvec


def pose_from_detections(detections, marker_map: dict, camera_matrix, dist_coeffs,
                         offsets: np.ndarray, assume_level: bool = False):
    """Camera pose in world coordinates, or None if nothing usable is visible."""
    if not detections:
        return None

    object_points, image_points, used_ids = build_correspondences(
        detections, marker_map, offsets)

    single = len(used_ids) == 1
    if assume_level:
        # SCALED-DEMO: valid only while the mount holds the optical axis normal
        # to the marker plane. On real hardware this is a mechanical promise,
        # not a software one - see solve_level for how far reproj_px can be
        # trusted to police it.
        rvec, tvec = solve_level(object_points, image_points, camera_matrix, dist_coeffs)
    elif single:
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

    # Angle between the optical axis and the marker plane's normal. This is the
    # term that dominates at height - 1 degree at 12 m is about 21 cm laterally -
    # so it is reported per frame rather than assumed. It also says whether
    # pose.assume_level is honest for a given rig: the 4-DOF solve costs
    # range * tan(this), and a handheld capture is never near zero.
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(float(rotation[2, 2]))))))

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
        "level_assumed": assume_level,
        "tilt_deg": tilt_deg,
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


# Hall map  (build stage 4)
# --------------------------------------------------------------------------

class HallMap:
    """Top-down 2D view of the marker plane and the camera's track across it.

    Deliberately plain OpenCV drawing; the spec says polish is not required.
    The map earns its place because it makes a wrong pose obvious at a glance -
    a mirrored X, a tag mapped to the wrong grid cell, a jump the moment a
    marker leaves view - none of which are visible in a scrolling column of
    numbers.

    The extent is derived from the marker map rather than configured. The tags
    *are* the surveyed area, so a hand-entered hall rectangle would be a second
    copy of the same fact, free to drift out of agreement with the first.
    """

    def __init__(self, marker_map: dict, viz_cfg: dict):
        xs = [x for x, _ in marker_map["markers"].values()]
        ys = [y for _, y in marker_map["markers"].values()]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        self.tag_size = marker_map["tag_size_m"]
        self.markers = marker_map["markers"]

        self.cfg = viz_cfg
        self.trail = deque(maxlen=viz_cfg["map_trail_len"])

        # Canvas size is fixed once, from the aspect of the surveyed area, so
        # the window never resizes mid-run. The margin is deliberately not in
        # here: it scales both axes equally and would cancel. max(..., tag_size)
        # keeps a single-marker or single-row map from collapsing to zero.
        span_x = max(self.bounds[2] - self.bounds[0], self.tag_size)
        span_y = max(self.bounds[3] - self.bounds[1], self.tag_size)
        longest = max(span_x, span_y)
        self.width_px = max(1, round(viz_cfg["map_size_px"] * span_x / longest))
        self.height_px = max(1, round(viz_cfg["map_size_px"] * span_y / longest))

        self._fit()

    def _view(self):
        """The world rectangle to draw: marker extent, widened by the trail.

        The camera is not guaranteed to be inside the tags - on the screen rig
        it sits well outside them - and a map that silently clips the dot it
        exists to show is worse than no map.

        Widening by the *trail* rather than by every pose ever seen is what
        keeps this stable. A single ill-conditioned single-marker frame can
        place the camera metres away; grow-only would let that one outlier zoom
        the map out permanently, whereas here it drops out of view once it ages
        past the trail length and the scale recovers on its own.
        """
        min_x, min_y, max_x, max_y = self.bounds
        for x, y in self.trail:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
        return min_x, min_y, max_x, max_y

    def _fit(self) -> None:
        """Recompute scale and origin so the view fits the fixed canvas.

        One scale for both axes, so a metre reads the same length horizontally
        and vertically and a non-square area is drawn undistorted. Whatever
        room is left over on the other axis becomes centring slack.
        """
        min_x, min_y, max_x, max_y = self._view()
        span_x = max(max_x - min_x, self.tag_size)
        span_y = max(max_y - min_y, self.tag_size)
        margin = max(span_x, span_y) * self.cfg["map_margin_frac"]
        self.scale = min(self.width_px / (span_x + 2 * margin),
                         self.height_px / (span_y + 2 * margin))
        self.origin = (min_x - (self.width_px / self.scale - span_x) / 2.0,
                       min_y - (self.height_px / self.scale - span_y) / 2.0)

    def to_px(self, x: float, y: float):
        """World metres -> map pixels.

        World +Y is "forward" and is drawn up the image, but image rows
        increase downward, hence the subtraction rather than a second scale.
        Dropping that flip mirrors the map vertically and reads exactly like a
        sign error in the solver, which is the wrong place to go looking.
        """
        return (int(round((x - self.origin[0]) * self.scale)),
                int(round(self.height_px - (y - self.origin[1]) * self.scale)))

    def render(self, pose, detected_ids, frame_index: int, fps_value: float):
        cfg = self.cfg
        if pose is not None:
            self.trail.append((pose["x"], pose["y"]))
        self._fit()
        canvas = np.full((self.height_px, self.width_px, 3),
                         cfg["map_bg_bgr"], dtype=np.uint8)

        # Surveyed extent. Top-left in pixels is (min_x, max_y) in world.
        min_x, min_y, max_x, max_y = self.bounds
        cv2.rectangle(canvas, self.to_px(min_x, max_y), self.to_px(max_x, min_y),
                      tuple(cfg["map_bounds_bgr"]), 1)

        # Markers at true physical size, so the drawing carries the same scale
        # as the geometry: a tag that looks wrong here is placed wrong.
        half = self.tag_size / 2.0
        detected = set(detected_ids)
        for marker_id, (cx, cy) in sorted(self.markers.items()):
            seen = marker_id in detected
            color = cfg["map_detected_bgr"] if seen else cfg["known_color_bgr"]
            cv2.rectangle(canvas, self.to_px(cx - half, cy + half),
                          self.to_px(cx + half, cy - half),
                          tuple(color), -1 if seen else 1)
            cv2.putText(canvas, str(marker_id), self.to_px(cx + half, cy + half),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg["map_label_scale"],
                        tuple(color), 1, cv2.LINE_AA)

        # Unsmoothed trail: every wobble in it is real measurement noise, and
        # the point of v1 is that it stays visible.
        if len(self.trail) > 1:
            points = np.array([self.to_px(x, y) for x, y in self.trail], dtype=np.int32)
            cv2.polylines(canvas, [points], False, tuple(cfg["map_trail_bgr"]), 1, cv2.LINE_AA)

        lines = [f"frame {frame_index}   fps {fps_value:5.1f}"]
        if pose is None:
            lines.append("no pose")
        else:
            centre = self.to_px(pose["x"], pose["y"])
            cv2.circle(canvas, centre, cfg["map_dot_px"], tuple(cfg["map_pose_bgr"]), -1)
            # Heading is drawn a fixed number of pixels long rather than a
            # fixed number of metres: it is a direction indicator, and a metric
            # length would vanish on one rig and dominate the other. The minus
            # on sin is the same image-Y-down flip as to_px, in pixel space.
            angle = math.radians(pose["yaw_deg"])
            tip = (int(round(centre[0] + cfg["map_heading_px"] * math.cos(angle))),
                   int(round(centre[1] - cfg["map_heading_px"] * math.sin(angle))))
            cv2.line(canvas, centre, tip, tuple(cfg["map_pose_bgr"]), 2, cv2.LINE_AA)

            lines.append(f"x {pose['x']:+.3f}  y {pose['y']:+.3f}  yaw {pose['yaw_deg']:+6.1f}")
            lines.append(f"markers {pose['n_markers']}: {pose['ids']}")
            lines.append(f"reproj {pose['reproj_px']:.2f} px   "
                         f"acc~{pose['acc_est_m'] * 1000:.1f} mm"
                         + ("   SINGLE MARKER" if pose["single_marker"] else ""))
        # Solid strip behind the status text. The camera view gets away with
        # a shadow pass alone, but here the text lands on top of bright filled
        # marker squares and their ID labels, which shadowing cannot rescue.
        header_px = cfg["text_origin_px"][1] + len(lines) * cfg["text_line_height_px"]
        cv2.rectangle(canvas, (0, 0), (self.width_px, header_px - cfg["text_line_height_px"] // 2),
                      tuple(cfg["map_bg_bgr"]), -1)
        draw_text_block(canvas, lines, cfg)
        return canvas


# --------------------------------------------------------------------------
# Output  (build stage 4: per-frame CSV, and the UDP feed)
# --------------------------------------------------------------------------

# z_m is logged alongside x/y because it is the free scale check: the marker
# map puts the plane at a known Z, so the solved camera Z is a direct readout
# of range error, and a range off by a fixed fraction means the focal length is
# off by that same fraction - which scales X and Y with it.
CSV_COLUMNS = ["t", "frame", "x_m", "y_m", "z_m", "yaw_deg", "tilt_deg", "n_markers",
               "ids", "reproj_px", "acc_est_m", "single_marker", "level_assumed"]


def open_csv(path: Path):
    """Open the per-frame log for append, writing a header only if new.

    Append rather than truncate: overwriting would silently destroy the
    previous run, and the report is built from these rows. Runs stay separable
    without a filename scheme because `frame` restarts at 0 and `t` jumps.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if fresh:
        writer.writerow(CSV_COLUMNS)
    return handle, writer


def csv_row(timestamp: float, frame_index: int, pose):
    """One row per frame, including frames with no pose.

    Frames without a pose are logged as blanks rather than skipped: the
    fraction of frames that yielded a fix is itself a result, and it cannot be
    recovered from a file that only kept the successes.
    """
    if pose is None:
        return [f"{timestamp:.3f}", frame_index, "", "", "", "", "", 0, "", "", "", "", ""]
    return [f"{timestamp:.3f}", frame_index,
            f"{pose['x']:.6f}", f"{pose['y']:.6f}", f"{pose['z']:.6f}",
            f"{pose['yaw_deg']:.3f}", f"{pose['tilt_deg']:.3f}",
            pose["n_markers"], " ".join(str(i) for i in pose["ids"]),
            f"{pose['reproj_px']:.4f}", f"{pose['acc_est_m']:.6f}",
            int(pose["single_marker"]), int(pose["level_assumed"])]


def make_publisher(udp_cfg: dict):
    """Return a send(pose) callable. UDP JSON, one datagram per fix.

    Fire-and-forget by design: a consumer that stalls or dies must not slow the
    capture loop or block a frame. A dropped datagram costs one pose and the
    next is along in ~30 ms. The payload stays flat so a ROS 2 bridge node is a
    field-by-field copy.
    """
    if not udp_cfg.get("enabled", True):
        return lambda pose: None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = (udp_cfg["host"], int(udp_cfg["port"]))

    def send(pose) -> None:
        payload = {"t": time.time(), "x": pose["x"], "y": pose["y"],
                   "yaw_deg": pose["yaw_deg"], "markers": pose["ids"],
                   "reproj_px": pose["reproj_px"], "acc_est_m": pose["acc_est_m"]}
        sock.sendto(json.dumps(payload).encode("utf-8"), address)

    return send


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
    parser.add_argument("--assume-level", type=int, choices=(0, 1), default=None,
                        help="override hall.json pose.assume_level. 1 constrains the "
                             "solve to 4 DOF for a level mount; 0 uses full solvePnP.")
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

    pose_cfg = hall.get("pose", {})
    assume_level = (bool(pose_cfg.get("assume_level", False))
                    if args.assume_level is None else bool(args.assume_level))

    known_ids = set(marker_map["markers"])
    source_spec = args.source if args.source is not None else hall["source"]
    source = FrameSource(source_spec, hall["camera"])
    detector = make_detector(hall["detection"])
    fps = FpsMeter(viz_cfg["fps_window_frames"])

    output_cfg = hall["output"]
    snapshot_dir = Path(output_cfg["snapshot_dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    hall_map = HallMap(marker_map, viz_cfg)
    csv_path = Path(output_cfg["csv_path"])
    csv_handle, csv_writer = open_csv(csv_path)
    publish = make_publisher(output_cfg["udp"])

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
    # Printed loudly: this is an assertion about the physical rig, not a
    # tuning knob, and a wrong one biases every position silently.
    print(f"csv log     : {csv_path}")
    udp_cfg = output_cfg["udp"]
    print(f"udp feed    : {udp_cfg['host']}:{udp_cfg['port']}"
          if udp_cfg.get("enabled", True) else "udp feed    : disabled")
    print(f"pose model  : {'4-DOF, MOUNT ASSUMED LEVEL' if assume_level else '6-DOF solvePnP'}")
    print("keys        : q/ESC quit, space pause, s snapshot")
    if source.is_stills:
        # Copy-pasteable straight into the report. Z is here alongside X and Y
        # because it is the free scale check: the marker map pins the plane at
        # a known Z, so the solved camera Z is a direct readout of range error.
        print(f"\n{'photo':<24}{'x_m':>9}{'y_m':>9}{'z_m':>9}"
              f"{'yaw':>7}{'tilt':>7}{'n':>3}{'reproj':>8}{'acc_mm':>8}")

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
                pose = None
                if camera_matrix is not None:
                    pose = pose_from_detections(known, marker_map, camera_matrix,
                                                dist_coeffs, offsets, assume_level)
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

                if source.is_stills:
                    name = source.frame_name[:23]
                    if pose is None:
                        print(f"{name:<24}{'no pose':>9}")
                    else:
                        print(f"{name:<24}{pose['x']:>9.4f}{pose['y']:>9.4f}"
                              f"{pose['z']:>9.4f}{pose['yaw_deg']:>7.1f}"
                              f"{pose['tilt_deg']:>7.1f}{pose['n_markers']:>3}"
                              f"{pose['reproj_px']:>8.3f}"
                              f"{pose['acc_est_m'] * 1000:>8.2f}")

                cv2.imshow(viz_cfg["camera_window"], frame)
                cv2.imshow(viz_cfg["map_window"],
                           hall_map.render(pose, [i for i, _ in known],
                                           frame_index, fps.value()))

                # Log before publishing: the CSV is the record the report is
                # written from, and a send that raises must not cost the row.
                csv_writer.writerow(csv_row(time.time(), frame_index, pose))
                if pose is not None:
                    publish(pose)

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
        csv_handle.close()
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
