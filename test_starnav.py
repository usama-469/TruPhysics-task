#!/usr/bin/env python3
"""Self-checks for the single-marker pose. Run: python test_starnav.py

Stage 2 is the sign check, and its failure mode is silent: a mirrored or negated
axis produces confident, plausible-looking numbers. These assertions pin the
signs in software so the photographs have something to be compared against, and
they run without a camera, a calibration or a printed tag.

Multi-marker pose accuracy is not tested here - tools/eval_photos.py measures it
against real photographs, which is a stronger check than any assertion.
"""

import cv2
import numpy as np

import starnav

# Stand-in intrinsics. 480x848 portrait, ~51 deg HFOV, chosen so a 0.2 m tag at 0.75 m stays inside the frame even after the
# 0.2 m lateral move the sign check makes.
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


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok  {name}")
    print("all checks passed")
