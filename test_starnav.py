#!/usr/bin/env python3
"""Self-checks for the pose solvers. Run: python test_starnav.py

Both failure modes here are silent ones. A mirrored or negated axis produces
confident, plausible-looking numbers; so does a 4-DOF level solve running on a
mount that is not actually level. These assertions pin the signs and the level
solve's self-check in software, and run without a camera, a calibration or a
printed tag.

Multi-marker pose accuracy is not tested here - tools/eval_photos.py measures it
against real photographs, which is a stronger check than any assertion.
"""

import json
import math
import socket

import cv2
import numpy as np

import starnav

# Stand-in intrinsics. 480x848 portrait, ~51 deg HFOV, chosen so a 0.2 m tag at
# 0.75 m stays inside the frame even after the 0.2 m lateral move below.
CAMERA_MATRIX = np.array([[500.0, 0.0, 240.0],
                          [0.0, 500.0, 424.0],
                          [0.0, 0.0, 1.0]])
NO_DISTORTION = np.zeros(5)

# The single-tag rig, as config/markers_single.json describes it: one tag
# centred on the panel, plane at Z = 0 so the solved camera Z reads the tape
# distance directly.
SINGLE = {"tag_size_m": 0.199865, "ceiling_height_m": 0.0,
          "markers": {0: (0.298863, 0.154101)}}
TAPE_DISTANCE_M = 0.75

VIZ = json.loads(open("config/hall.json", encoding="utf-8").read())["viz"]
HALL = {"tag_size_m": 0.15, "ceiling_height_m": 2.5,
        "markers": {0: (0.0, 0.0), 1: (2.0, 0.0), 4: (1.0, 1.0)}}

# A handheld photo is never exactly square-on to the panel, and it must not be:
# SOLVEPNP_IPPE_SQUARE is degenerate for a perfectly fronto-parallel square,
# where its two candidate poses collapse into one and it can return the wrong
# one. A few degrees of tilt resolves it completely - see
# test_face_on_is_degenerate_but_flagged for the failure it avoids.
TILT_DEG = 8.0


def tilt_about_camera_x(degrees):
    angle = np.radians(degrees)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), -np.sin(angle)],
                     [0.0, np.sin(angle), np.cos(angle)]])


def synthetic_detection(camera_xyz, rotation=None):
    """Project the single tag's corners as seen from a known camera position.

    The camera's axes start aligned to the world's - OpenCV's X right / Y down
    / Z forward onto the screen rig's X right / Y down / Z into the panel - and
    are then tilted by TILT_DEG, so what is exercised is the translation and
    its signs.

    This checks that solve_single's local->world composition is self-consistent
    with corner_offsets. It CANNOT confirm the physical corner ordering that
    cv2.aruco reports for a tag viewed from a given side - that is what the
    photographs resolve, and why CLAUDE.md insists the handedness check is
    empirical.
    """
    rotation = tilt_about_camera_x(TILT_DEG) if rotation is None else rotation
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = (-rotation @ np.array(camera_xyz, dtype=float)).reshape(3, 1)

    centre_x, centre_y = SINGLE["markers"][0]
    offsets = starnav.corner_offsets(SINGLE["tag_size_m"] / 2.0)
    world = np.array([[centre_x + dx, centre_y + dy, SINGLE["ceiling_height_m"]]
                      for dx, dy in offsets])
    image, _ = cv2.projectPoints(world, rvec, tvec, CAMERA_MATRIX, NO_DISTORTION)
    return [(0, image.reshape(1, 4, 2))]


def solve(camera_xyz, rotation=None):
    pose = starnav.pose_from_detections(
        synthetic_detection(camera_xyz, rotation), SINGLE, CAMERA_MATRIX,
        NO_DISTORTION, starnav.corner_offsets(SINGLE["tag_size_m"] / 2.0))
    assert pose is not None and pose["single_marker"]
    return pose


def test_single_marker_signs():
    """The stage-2 sign check, in software. Predicts what the photos should show."""
    centre_x, centre_y = SINGLE["markers"][0]
    # Camera on the -Z side of the panel, centred on the tag and tilted.
    base = solve((centre_x, centre_y, -TAPE_DISTANCE_M))
    assert abs(base["x"] - centre_x) < 1e-6, base["x"]
    assert abs(base["y"] - centre_y) < 1e-6, base["y"]
    # Z is negative: the camera sits in front of a plane pinned at Z = 0, so
    # the MAGNITUDE of z is the tape distance.
    assert abs(base["z"] + TAPE_DISTANCE_M) < 1e-6, base["z"]

    # corner_offsets and solve_single must agree on corner order, or the
    # residual blows up even though the pose itself is fine.
    assert base["reproj_px"] < 1e-3, base["reproj_px"]

    # Move right 0.20 m. World X runs right, so X increases.
    right = solve((centre_x + 0.20, centre_y, -TAPE_DISTANCE_M))
    assert abs((right["x"] - base["x"]) - 0.20) < 1e-6, right["x"] - base["x"]
    assert abs(right["y"] - base["y"]) < 1e-6

    # Raise 0.20 m. This map's world Y runs DOWN the panel, so moving the
    # camera up DECREASES Y. That sign is the whole point of the check.
    up = solve((centre_x, centre_y - 0.20, -TAPE_DISTANCE_M))
    assert abs((up["y"] - base["y"]) + 0.20) < 1e-6, up["y"] - base["y"]
    assert abs(up["x"] - base["x"]) < 1e-6


def level_shot(camera_xyz, roll_deg=0.0, tilt_deg=0.0):
    """Project the tag from a camera that may or may not honour the level promise."""
    roll, tilt = np.radians(roll_deg), np.radians(tilt_deg)
    about_z = np.array([[np.cos(roll), np.sin(roll), 0.0],
                        [-np.sin(roll), np.cos(roll), 0.0],
                        [0.0, 0.0, 1.0]])
    rotation = tilt_about_camera_x(tilt_deg) @ about_z
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = (-rotation @ np.array(camera_xyz, dtype=float)).reshape(3, 1)
    centre_x, centre_y = SINGLE["markers"][0]
    offsets = starnav.corner_offsets(SINGLE["tag_size_m"] / 2.0)
    world = np.array([[centre_x + dx, centre_y + dy, SINGLE["ceiling_height_m"]]
                      for dx, dy in offsets])
    image, _ = cv2.projectPoints(world, rvec, tvec, CAMERA_MATRIX, NO_DISTORTION)
    return starnav.pose_from_detections(
        [(0, image.reshape(1, 4, 2))], SINGLE, CAMERA_MATRIX, NO_DISTORTION,
        offsets, assume_level=True)


def test_level_solve_beats_pnp_when_level():
    """On a level mount the 4-DOF solve is exact where solvePnP is degenerate.

    A fronto-parallel square is the worst case for SOLVEPNP_IPPE_SQUARE and the
    best case for the similarity fit, which is the whole reason the constrained
    path exists.
    """
    centre_x, centre_y = SINGLE["markers"][0]
    camera = (centre_x - 0.05, centre_y + 0.03, -TAPE_DISTANCE_M)
    pose = level_shot(camera, roll_deg=23.0)

    assert pose is not None and pose["level_assumed"]
    assert abs(pose["x"] - camera[0]) < 1e-4, pose["x"]
    assert abs(pose["y"] - camera[1]) < 1e-4, pose["y"]
    assert abs(pose["z"] - camera[2]) < 1e-4, pose["z"]
    assert pose["reproj_px"] < 1e-3, pose["reproj_px"]


def test_level_solve_residual_reveals_tilt():
    """The 4-DOF residual polices the assumption the 4-DOF solve depends on.

    A tilted square images as a trapezoid and no similarity can reproduce one,
    so violating the level promise shows up in reproj_px - unlike the 6-DOF
    path, which models the tilt and stays quiet about it. This is the only
    self-check the constrained solver has, so it must keep working.
    """
    centre_x, centre_y = SINGLE["markers"][0]
    camera = (centre_x, centre_y, -TAPE_DISTANCE_M)
    level = level_shot(camera, roll_deg=10.0, tilt_deg=0.0)
    tilted = level_shot(camera, roll_deg=10.0, tilt_deg=2.0)

    assert level["reproj_px"] < 1e-3, level["reproj_px"]
    assert tilted["reproj_px"] > 10 * max(level["reproj_px"], 1e-6)

    # Unmodelled tilt costs range * tan(tilt) - at 0.75 m and 2 deg, ~26 mm.
    # Measured as a combined lateral error, not per-axis: a tilt about the
    # camera's X axis displaces the image vertically, so which of world X or Y
    # absorbs it depends entirely on the roll angle.
    offset = math.hypot(tilted["x"] - camera[0], tilted["y"] - camera[1])
    expected = TAPE_DISTANCE_M * math.tan(math.radians(2.0))
    assert abs(offset - expected) < 0.005, (offset, expected)


def test_face_on_is_degenerate_but_flagged():
    """A perfectly square-on single tag is unsolvable - but it says so.

    IPPE_SQUARE's two candidate poses coincide when the square is exactly
    fronto-parallel, and it can return the mirrored one: here the camera comes
    back at +Z, on the far side of the panel it is actually in front of.

    What makes that survivable is that the wrong pose does NOT reproject
    cleanly. reproj_px jumps by nine orders of magnitude, so the frame is
    flagged rather than silently believed. This asserts the alarm still works;
    it is not an endorsement of shooting square-on. Add a few degrees of tilt.
    """
    centre_x, centre_y = SINGLE["markers"][0]
    face_on = solve((centre_x, centre_y, -TAPE_DISTANCE_M), np.eye(3))
    tilted = solve((centre_x, centre_y, -TAPE_DISTANCE_M))

    assert tilted["reproj_px"] < 1e-3, tilted["reproj_px"]
    assert face_on["reproj_px"] > 1.0, face_on["reproj_px"]
    # X and Y still land correctly; it is Z whose sign flips, which is exactly
    # why the stage-2 check reads Z and not just X/Y.
    assert abs(face_on["x"] - centre_x) < 1e-6
    assert face_on["z"] > 0 > tilted["z"]


def test_map_transform():
    hall_map = starnav.HallMap(HALL, VIZ)

    # Which way +Y draws is a property of the world frame, not a taste, and
    # both directions have to work. A right-handed wall or ceiling survey ends
    # up Y-DOWN, and drawing that with +Y up puts the markers in the right
    # place while sending the moving dot the wrong way - correct-looking and
    # wrong, which is the worst kind of bug.
    assert hall_map.to_px(1.0, 0.0)[0] > hall_map.to_px(0.0, 0.0)[0]
    if hall_map.y_down:
        assert hall_map.to_px(0.0, 1.0)[1] > hall_map.to_px(0.0, 0.0)[1]
    else:
        assert hall_map.to_px(0.0, 1.0)[1] < hall_map.to_px(0.0, 0.0)[1]

    # and the flag actually reverses it
    flipped = starnav.HallMap(HALL, dict(VIZ, map_y_down=not hall_map.y_down))
    assert ((flipped.to_px(0.0, 1.0)[1] > flipped.to_px(0.0, 0.0)[1])
            != (hall_map.to_px(0.0, 1.0)[1] > hall_map.to_px(0.0, 0.0)[1]))

    # Every marker lands inside the plot area, below the header band - the
    # regression that hid the top row of markers under the status text.
    for x, y in HALL["markers"].values():
        col, row = hall_map.to_px(x, y)
        assert 0 < col < hall_map.width_px, (x, y, col)
        assert hall_map.header_px < row < hall_map.header_px + hall_map.height_px, (x, y, row)

    # One scale for both axes: a metre is the same length either way.
    span_x = hall_map.to_px(1.0, 0.0)[0] - hall_map.to_px(0.0, 0.0)[0]
    span_y = abs(hall_map.to_px(0.0, 0.0)[1] - hall_map.to_px(0.0, 1.0)[1])
    assert abs(span_x - span_y) <= 1, (span_x, span_y)


def test_map_renders():
    hall_map = starnav.HallMap(HALL, VIZ)
    pose = {"x": 1.0, "y": 0.5, "yaw_deg": 90.0, "n_markers": 2, "ids": [0, 4],
            "reproj_px": 0.31, "acc_est_m": 0.012, "single_marker": False}
    for frame in (hall_map.render(None, [], 0, 0.0),
                  hall_map.render(pose, [0, 4], 1, 30.0)):
        # canvas = reserved header band + the plot area
        assert frame.shape == (hall_map.header_px + hall_map.height_px,
                               hall_map.width_px, 3)
    assert len(hall_map.trail) == 1  # only the frame that had a pose


def test_csv_row():
    pose = {"x": 1.25, "y": -0.5, "z": 2.5, "yaw_deg": 92.4, "tilt_deg": 1.7,
            "n_markers": 3, "ids": [1, 4, 5], "reproj_px": 0.38,
            "acc_est_m": 0.04, "single_marker": False, "level_assumed": False}
    row = starnav.csv_row(1723459200.12, 7, pose)
    assert len(row) == len(starnav.CSV_COLUMNS)
    assert row[starnav.CSV_COLUMNS.index("ids")] == "1 4 5"  # space-joined, no quoting
    assert row[starnav.CSV_COLUMNS.index("tilt_deg")] == "1.700"

    # No-pose frames are still logged - the fraction of frames that yielded a
    # fix is itself a result, and it cannot be recovered from a file that only
    # kept the successes.
    blank = starnav.csv_row(1723459200.12, 8, None)
    assert len(blank) == len(starnav.CSV_COLUMNS)
    assert blank[1] == 8


def test_publisher_sends():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(2.0)
    host, port = listener.getsockname()

    send = starnav.make_publisher({"enabled": True, "host": host, "port": port})
    send({"x": 12.43, "y": 6.81, "yaw_deg": 92.4, "ids": [1, 4, 5],
          "reproj_px": 0.38, "acc_est_m": 0.04})

    payload = json.loads(listener.recv(4096).decode("utf-8"))
    listener.close()
    assert set(payload) == {"t", "x", "y", "yaw_deg", "markers", "reproj_px", "acc_est_m"}
    assert payload["markers"] == [1, 4, 5] and payload["x"] == 12.43

    starnav.make_publisher({"enabled": False})({"x": 0.0})  # disabled = silent, not a crash


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok  {name}")
    print("all checks passed")
