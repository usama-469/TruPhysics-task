#!/usr/bin/env python3
"""Resolve the corner handedness and measure position error from still photos.

Two questions, answered from a folder of photographs of the on-screen tag grid.

1. HANDEDNESS. The mapping from ArUco's image corner order to world object
   points has 8 candidates: 4 mounting rotations x 2 windings. Only one is
   geometrically right. Rather than reason about it, all 8 are solved and the
   residuals compared - the correct one lands at a fraction of a pixel and the
   rest are one to two orders of magnitude worse. This is the empirical check
   the spec asks for, and it is decisive because a mirrored winding cannot be
   absorbed by any rigid camera pose.

2. ERROR. There is no ground truth for where the phone was standing, but there
   is exact ground truth for where every marker is. So each marker is held out
   in turn: solve the pose from the others, back-project the held-out marker's
   corners onto the marker plane, and compare against its known position. The
   residual is a position error in millimetres, on the same plane and in the
   same units as the quantity the system exists to report.

   That number is honest about everything except the two terms it cannot see -
   survey error (near zero here, the map is pixel-addressed) and camera tilt
   (invisible to any residual, because a tilted camera reprojects its own wrong
   pose perfectly).

Usage:
    python tools/eval_photos.py --photos photos/eval --calib calib.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import starnav  # noqa: E402

# A held-out marker needs enough company left over for the remaining solve to
# be well conditioned; below this the "error" measures the solve, not the map.
MIN_MARKERS_FOR_HOLDOUT = 6

# Real ceiling height the demo is standing in for. Errors from both dominant
# terms - corner noise (range/focal x pixels) and tilt (range x tan) - are
# linear in range, so an angular error measured here projects to the hall by
# simple proportion. Stated as an assumption, not a claim.
HALL_HEIGHT_M = 12.0


def load_photos(folder: Path):
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in starnav.IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"no images in {folder} "
                         f"(supported: {', '.join(starnav.IMAGE_EXTENSIONS)}; HEIC will not load)")
    return paths


def detect_all(paths, detector, known_ids, expected_size):
    """Detect once per photo; the convention search then reuses the results."""
    detections = {}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  {path.name}: unreadable, skipped")
            continue
        size = (image.shape[1], image.shape[0])
        if expected_size is not None and size != expected_size:
            # Not a warning. Intrinsics are in pixels of a specific readout; a
            # different resolution or orientation makes every number nonsense.
            print(f"  {path.name}: {size[0]}x{size[1]} but calib.npz is "
                  f"{expected_size[0]}x{expected_size[1]} - SKIPPED")
            continue
        corners, ids, _ = detector.detectMarkers(image)
        known, _ = starnav.split_known_unknown(corners, ids, known_ids)
        detections[path.name] = known
    return detections


def mean_residual(detections, marker_map, camera_matrix, dist_coeffs, offsets):
    """Mean reprojection error across all photos for one corner convention."""
    residuals = []
    for known in detections.values():
        if len(known) < 2:
            continue
        pose = starnav.pose_from_detections(known, marker_map, camera_matrix,
                                            dist_coeffs, offsets)
        if pose is not None:
            residuals.append(pose["reproj_px"])
    return float(np.mean(residuals)) if residuals else float("inf")


def resolve_convention(detections, marker_map, camera_matrix, dist_coeffs):
    """Rank all 8 corner conventions by residual and report the winner."""
    half = marker_map["tag_size_m"] / 2.0
    results = []
    for mirror in (False, True):
        for rotation in range(4):
            offsets = starnav.corner_offsets(half, rotation, mirror)
            results.append((mean_residual(detections, marker_map, camera_matrix,
                                          dist_coeffs, offsets), rotation, mirror))
    results.sort()
    return results


def backproject_to_plane(image_points, rvec, tvec, camera_matrix, dist_coeffs, plane_z):
    """Intersect the rays through given pixels with the marker plane.

    This is the inverse of what the solver did, and it is what turns a pixel
    residual into a distance on the plane: undistort to normalised camera rays,
    rotate them into the world, then walk each ray from the camera centre until
    it reaches Z = plane_z.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    camera_in_world = (-rotation.T @ tvec).ravel()
    normalised = cv2.undistortPoints(
        image_points.reshape(-1, 1, 2).astype(np.float64), camera_matrix, dist_coeffs
    ).reshape(-1, 2)
    rays_camera = np.hstack([normalised, np.ones((len(normalised), 1))])
    rays_world = (rotation.T @ rays_camera.T).T
    scale = (plane_z - camera_in_world[2]) / rays_world[:, 2]
    return camera_in_world + scale[:, None] * rays_world


def holdout_errors(known, marker_map, camera_matrix, dist_coeffs, offsets):
    """Leave-one-marker-out position error, in millimetres on the marker plane."""
    if len(known) < MIN_MARKERS_FOR_HOLDOUT + 1:
        return np.array([])

    plane_z = marker_map["ceiling_height_m"]
    errors = []
    for index, (marker_id, corners) in enumerate(known):
        others = known[:index] + known[index + 1:]
        pose = starnav.pose_from_detections(others, marker_map, camera_matrix,
                                            dist_coeffs, offsets)
        if pose is None:
            continue
        on_plane = backproject_to_plane(corners.reshape(4, 2), pose["rvec"], pose["tvec"],
                                        camera_matrix, dist_coeffs, plane_z)
        measured_centre = on_plane[:, :2].mean(axis=0)
        truth = np.array(marker_map["markers"][marker_id])
        errors.append(np.linalg.norm(measured_centre - truth) * 1000.0)
    return np.array(errors)


def cluster_holdout_errors(known, marker_map, camera_matrix, dist_coeffs, offsets,
                           cluster_size: int):
    """Solve from a small central cluster, then predict every other marker.

    Leave-one-out measures the wrong thing on a dense grid. Holding out one
    marker while keeping forty-nine leaves the pose almost perfectly pinned, so
    the residual only shows corner noise - and worse, any *systematic* error is
    invisible, because a focal length that is 0.1 % wrong shifts the retained
    markers and the held-out one together and cancels out.

    Solving from a tight central cluster and predicting outward removes that
    cancellation. The prediction is an extrapolation, so focal length error,
    residual distortion and any bow in the panel all show up, and they show up
    growing with distance from the cluster - which is the signature that tells
    those apart from noise.

    Returns (errors_mm, radii_m), radius measured from the cluster's centroid
    on the marker plane.
    """
    if len(known) < cluster_size + 2:
        return np.array([]), np.array([])

    image_centres = np.array([corners.reshape(4, 2).mean(axis=0) for _, corners in known])
    centroid = image_centres.mean(axis=0)
    order = np.argsort(np.linalg.norm(image_centres - centroid, axis=1))
    cluster = [known[i] for i in order[:cluster_size]]
    outside = [known[i] for i in order[cluster_size:]]

    pose = starnav.pose_from_detections(cluster, marker_map, camera_matrix,
                                        dist_coeffs, offsets)
    if pose is None:
        return np.array([]), np.array([])

    plane_z = marker_map["ceiling_height_m"]
    cluster_centre = np.mean([marker_map["markers"][mid] for mid, _ in cluster], axis=0)
    errors, radii = [], []
    for marker_id, corners in outside:
        on_plane = backproject_to_plane(corners.reshape(4, 2), pose["rvec"], pose["tvec"],
                                        camera_matrix, dist_coeffs, plane_z)
        measured = on_plane[:, :2].mean(axis=0)
        truth = np.array(marker_map["markers"][marker_id])
        errors.append(np.linalg.norm(measured - truth) * 1000.0)
        radii.append(np.linalg.norm(truth - cluster_centre))
    return np.array(errors), np.array(radii)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="handedness resolution and error measurement from stills")
    parser.add_argument("--photos", type=Path, required=True,
                        help="folder of stills to evaluate; must NOT be the calibration set")
    parser.add_argument("--calib", type=Path, default=Path("calib.npz"),
                        help="intrinsics from calibrate.py")
    parser.add_argument("--markers", type=Path, default=Path("config/markers_screen.json"),
                        help="marker map the photos show")
    parser.add_argument("--hall", type=Path, default=Path("config/hall.json"),
                        help="source of the dictionary and detector settings")
    parser.add_argument("--csv", type=Path, default=Path("logs/eval_photos.csv"),
                        help="per-photo results, one row each")
    parser.add_argument("--refine-win", type=int, default=None,
                        help="corner refinement window; raise it for high-resolution stills")
    parser.add_argument("--cluster-size", type=int, default=4,
                        help="markers in the central cluster for the extrapolation test")
    parser.add_argument("--convention", nargs=2, type=int, default=None,
                        metavar=("ROTATION", "MIRROR"),
                        help="skip the search and force a convention")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hall = starnav.load_json(args.hall)
    marker_map = starnav.load_marker_map(args.markers)
    camera_matrix, dist_coeffs, image_size = starnav.load_calibration(args.calib)

    detection_cfg = dict(hall["detection"])
    if args.refine_win:
        detection_cfg["corner_refine_win_size"] = args.refine_win
    detector = starnav.make_detector(detection_cfg)

    paths = load_photos(args.photos)
    print(f"photos: {len(paths)} in {args.photos}")
    detections = detect_all(paths, detector, set(marker_map["markers"]), image_size)
    usable = {name: known for name, known in detections.items() if len(known) >= 2}
    if not usable:
        raise SystemExit("no photo showed 2 or more mapped markers")
    counts = [len(k) for k in usable.values()]
    print(f"usable: {len(usable)} photos, {min(counts)}-{max(counts)} markers each\n")

    # -- 1. handedness -----------------------------------------------------
    half = marker_map["tag_size_m"] / 2.0
    if args.convention:
        rotation, mirror = int(args.convention[0]), bool(args.convention[1])
        print(f"convention forced: rotation {rotation}, mirror {mirror}")
    else:
        ranked = resolve_convention(usable, marker_map, camera_matrix, dist_coeffs)
        print("corner convention search (mean reprojection error over all photos):")
        for residual, rot, mir in ranked:
            marker = "  <-- best" if (residual, rot, mir) == ranked[0] else ""
            print(f"  rotation {rot}  mirror {str(mir):<5}  {residual:9.3f} px{marker}")
        _, rotation, mirror = ranked[0]
        runner_up = ranked[1][0]
        margin = runner_up / max(ranked[0][0], 1e-9)
        print(f"\nverdict: rotation {rotation}, mirror {mirror} "
              f"({margin:.0f}x better than the next candidate)")
        if margin < 3:
            print("  INCONCLUSIVE - the candidates are too close. Usually means too few "
                  "markers per photo or too little viewing-angle variety.")
        if mirror:
            print("  NOTE: the winning convention is mirrored relative to the nominal "
                  "one. Either the map's Y axis or the physical mounting runs the "
                  "opposite way from what starnav.corner_offsets assumes.")

    offsets = starnav.corner_offsets(half, rotation, mirror)

    # -- 2. pose and error per photo ---------------------------------------
    print(f"\n{'photo':<22}{'n':>4}{'x_m':>9}{'y_m':>9}{'z_m':>8}{'yaw':>7}"
          f"{'reproj':>8}{'acc_mm':>8}{'loo_mm':>8}{'extrap_mm':>11}")
    rows = []
    all_holdout = []
    all_cluster, all_radii = [], []
    for name, known in usable.items():
        pose = starnav.pose_from_detections(known, marker_map, camera_matrix,
                                            dist_coeffs, offsets)
        if pose is None:
            continue
        errors = holdout_errors(known, marker_map, camera_matrix, dist_coeffs, offsets)
        median_holdout = float(np.median(errors)) if errors.size else float("nan")
        all_holdout.append(errors)

        cluster_err, cluster_rad = cluster_holdout_errors(
            known, marker_map, camera_matrix, dist_coeffs, offsets, args.cluster_size)
        median_cluster = float(np.median(cluster_err)) if cluster_err.size else float("nan")
        all_cluster.append(cluster_err)
        all_radii.append(cluster_rad)

        print(f"{name[:21]:<22}{pose['n_markers']:>4}{pose['x']:>9.4f}{pose['y']:>9.4f}"
              f"{pose['z']:>8.4f}{pose['yaw_deg']:>7.1f}{pose['reproj_px']:>8.3f}"
              f"{pose['acc_est_m'] * 1000:>8.2f}{median_holdout:>8.2f}{median_cluster:>11.2f}")
        rows.append({
            "photo": name, "n_markers": pose["n_markers"],
            "x_m": round(pose["x"], 6), "y_m": round(pose["y"], 6),
            "z_m": round(pose["z"], 6), "yaw_deg": round(pose["yaw_deg"], 3),
            "range_m": round(pose["range_m"], 4),
            "reproj_px": round(pose["reproj_px"], 4),
            "acc_est_m": round(pose["acc_est_m"], 6),
            "loo_median_mm": round(median_holdout, 3),
            "extrap_median_mm": round(median_cluster, 3),
            "ids": " ".join(str(i) for i in pose["ids"]),
        })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # -- 3. summary --------------------------------------------------------
    def pool(chunks):
        kept = [c for c in chunks if c.size]
        return np.concatenate(kept) if kept else np.array([])

    loo = pool(all_holdout)
    extrapolation = pool(all_cluster)
    radii = pool(all_radii)
    ranges = np.array([r["range_m"] for r in rows])
    median_range_mm = float(np.median(ranges)) * 1000.0

    def project_to_hall(error_mm: float) -> float:
        """Same angular error, seen from the real ceiling height."""
        return error_mm / median_range_mm * HALL_HEIGHT_M * 1000.0

    print(f"\nreprojection  : median {np.median([r['reproj_px'] for r in rows]):.3f} px")
    print(f"range         : {ranges.min():.2f} to {ranges.max():.2f} m from the panel")
    print("                tape-measure one shot and compare: a range that is off by a "
          "fixed fraction is a focal-length error, and it scales every X and Y too")

    if loo.size:
        print(f"\nleave-one-out : median {np.median(loo):.2f} mm, p95 "
              f"{np.percentile(loo, 95):.2f} mm ({loo.size} samples)")
        print("                This is the NOISE FLOOR, not the accuracy. Every other "
              "marker stays in the solve, so the pose barely moves and any systematic "
              "error cancels. Expect the true position error to be several times this.")

    if extrapolation.size:
        print(f"\nextrapolation : median {np.median(extrapolation):.2f} mm, p95 "
              f"{np.percentile(extrapolation, 95):.2f} mm, worst "
              f"{extrapolation.max():.2f} mm ({extrapolation.size} samples)")
        print(f"                solved from {args.cluster_size} central markers, predicting "
              f"outward to {radii.max():.2f} m away")
        print(f"                -> {project_to_hall(np.median(extrapolation)):.0f} mm at "
              f"{HALL_HEIGHT_M:.0f} m if the error stays angular")
        # A systematic term grows with distance from the cluster; noise does not.
        # Splitting at the median radius separates the two without fitting anything.
        if radii.size == extrapolation.size and radii.max() > radii.min():
            near = extrapolation[radii <= np.median(radii)]
            far = extrapolation[radii > np.median(radii)]
            if near.size and far.size:
                growth = np.median(far) / max(np.median(near), 1e-9)
                print(f"                near half {np.median(near):.2f} mm vs far half "
                      f"{np.median(far):.2f} mm ({growth:.1f}x)")
                if growth > 2.0:
                    print("                growing with distance = SYSTEMATIC. Suspect focal "
                          "length, residual distortion, an inaccurate panel size, or a "
                          "panel that is not flat - not corner noise.")

    print("\nneither number covers camera tilt or survey error. For an absolute check, "
          "\nre-run screen_tags.py with --offset-mm and shoot from the same spot: the "
          "\nreported XY must move by exactly that much, in the opposite direction.")
    print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
