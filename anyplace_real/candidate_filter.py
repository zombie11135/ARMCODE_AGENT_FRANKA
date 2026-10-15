"""Offline AnyPlace candidate geometry filter and TCP waypoint composer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .geometry import rotation_checks, transform_points, write_json, write_ply
from .run_inference import extract_predictions


def _inside_convex_support(points_xy: np.ndarray, hull: ConvexHull, margin_m: float) -> np.ndarray:
    # scipy hull equations use outward, unit-length normals: A*x + b <= 0.
    return np.all(
        points_xy @ hull.equations[:, :2].T + hull.equations[:, 2][None, :] <= -margin_m,
        axis=1,
    )


def evaluate_candidate(
    candidate: np.ndarray,
    target_proxy: np.ndarray,
    support_points: np.ndarray,
    clearance_m: float = 0.001,
    support_margin_m: float = 0.003,
    minimum_inside_fraction: float = 0.95,
    contact_band_m: float = 0.003,
    obstacle_points: Optional[np.ndarray] = None,
    obstacle_clearance_m: float = 0.005,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    candidate = np.asarray(candidate, dtype=np.float64)
    target_proxy = np.asarray(target_proxy, dtype=np.float64)
    support_points = np.asarray(support_points, dtype=np.float64)
    if candidate.shape != (4, 4) or target_proxy.ndim != 2 or target_proxy.shape[1] != 3:
        raise ValueError("invalid candidate or target proxy shape")
    if support_points.ndim != 2 or support_points.shape[1] != 3 or len(support_points) < 3:
        raise ValueError("support_points must contain at least three Nx3 points")

    support_z = float(np.median(support_points[:, 2]))
    transformed = transform_points(target_proxy, candidate)
    correction_z = support_z + float(clearance_m) - float(np.min(transformed[:, 2]))
    correction = np.eye(4, dtype=np.float64)
    correction[2, 3] = correction_z
    corrected = correction @ candidate
    transformed = transform_points(target_proxy, corrected)

    hull = ConvexHull(support_points[:, :2])
    inside = _inside_convex_support(transformed[:, :2], hull, float(support_margin_m))
    inside_fraction = float(np.mean(inside))
    center = np.mean(transformed, axis=0)
    center_supported = bool(_inside_convex_support(center[None, :2], hull, 0.0)[0])
    contact = np.abs(transformed[:, 2] - (support_z + clearance_m)) <= contact_band_m
    contact_points = transformed[contact]
    contact_supports_com = False
    if len(contact_points) >= 3:
        try:
            contact_hull = ConvexHull(contact_points[:, :2])
            contact_supports_com = bool(
                _inside_convex_support(center[None, :2], contact_hull, 0.0)[0]
            )
        except Exception:
            contact_supports_com = False

    obstacle_clearance = None
    obstacle_collision_free = True
    if obstacle_points is not None and len(obstacle_points):
        tree = cKDTree(np.asarray(obstacle_points, dtype=np.float64))
        obstacle_clearance = float(np.min(tree.query(transformed, k=1)[0]))
        obstacle_collision_free = bool(obstacle_clearance >= obstacle_clearance_m)

    rotation = rotation_checks(corrected)
    rotation_valid = bool(
        abs(rotation["determinant"] - 1.0) <= 1e-3
        and rotation["orthogonality_frobenius_error"] <= 1e-3
    )
    passed = bool(
        rotation_valid
        and inside_fraction >= minimum_inside_fraction
        and center_supported
        and contact_supports_com
        and obstacle_collision_free
    )
    metrics = {
        "passed_geometry_filter": passed,
        "support_surface_median_z_m": support_z,
        "vertical_projection_correction_m": correction_z,
        "minimum_clearance_after_projection_m": float(np.min(transformed[:, 2]) - support_z),
        "target_inside_eroded_support_fraction": inside_fraction,
        "minimum_inside_fraction": minimum_inside_fraction,
        "center_of_geometry_supported": center_supported,
        "contact_point_count": int(len(contact_points)),
        "contact_patch_supports_center": contact_supports_com,
        "obstacle_clearance_m": obstacle_clearance,
        "obstacle_collision_free": obstacle_collision_free,
        "rotation": rotation,
        "limitations": [
            "The support model is a gravity-aligned plane plus a 2-D convex hull.",
            "Robot/gripper/held-object IK and swept-volume collision are not evaluated here.",
        ],
    }
    return corrected, metrics


def run(
    predictions_path: Path,
    target_proxy_path: Path,
    support_path: Path,
    output_dir: Path,
    object_input_transform_path: Optional[Path] = None,
    tcp_object_transform_path: Optional[Path] = None,
    obstacle_path: Optional[Path] = None,
    clearance_m: float = 0.001,
    approach_height_m: float = 0.10,
) -> dict:
    predictions = extract_predictions(predictions_path.resolve())
    target_proxy = np.load(str(target_proxy_path.resolve()))
    support = np.load(str(support_path.resolve()))
    obstacles = np.load(str(obstacle_path.resolve())) if obstacle_path else None
    object_input = (
        np.load(str(object_input_transform_path.resolve()))
        if object_input_transform_path
        else None
    )
    tcp_object = (
        np.load(str(tcp_object_transform_path.resolve()))
        if tcp_object_transform_path
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    corrected_candidates = []
    for index, candidate in enumerate(predictions):
        try:
            corrected, metrics = evaluate_candidate(
                candidate,
                target_proxy,
                support,
                clearance_m=clearance_m,
                obstacle_points=obstacles,
            )
            corrected_candidates.append(corrected)
            candidate_rows.append({"index": index, "corrected_transform": corrected.tolist(), **metrics})
        except Exception as exc:
            corrected_candidates.append(np.asarray(candidate, dtype=np.float64))
            candidate_rows.append(
                {"index": index, "passed_geometry_filter": False, "error": str(exc)}
            )

    passing = [row for row in candidate_rows if row.get("passed_geometry_filter")]
    passing.sort(
        key=lambda row: (
            -row["target_inside_eroded_support_fraction"],
            abs(row["vertical_projection_correction_m"]),
            row["index"],
        )
    )
    selected = passing[0] if passing else None
    waypoints = None
    planning_blockers = []
    if selected is None:
        planning_blockers.append("No AnyPlace candidate passed the tabletop geometry filter.")
    if object_input is None:
        planning_blockers.append("T_base_object_input is missing.")
    if tcp_object is None:
        planning_blockers.append("Per-grasp T_tcp_object is missing.")
    if selected is not None and object_input is not None and tcp_object is not None:
        object_goal = np.asarray(selected["corrected_transform"]) @ object_input
        tcp_goal = object_goal @ np.linalg.inv(tcp_object)
        preplace = tcp_goal.copy()
        preplace[2, 3] += float(approach_height_m)
        retreat = preplace.copy()
        waypoints = {
            "frame": "robot_base",
            "T_base_tcp_preplace": preplace.tolist(),
            "T_base_tcp_place": tcp_goal.tolist(),
            "T_base_tcp_retreat": retreat.tolist(),
            "release": "GT200_OPEN_PLACEHOLDER_NOT_EXECUTABLE",
        }

    if selected is not None:
        selected_points = transform_points(target_proxy, np.asarray(selected["corrected_transform"]))
        np.save(str(output_dir / "selected_corrected_object_motion.npy"), np.asarray(selected["corrected_transform"]))
        np.save(str(output_dir / "selected_transformed_collision_proxy.npy"), selected_points)
        write_ply(output_dir / "selected_transformed_collision_proxy.ply", selected_points)

    # This stage deliberately cannot authorize hardware motion. A separate
    # robot-model planner must validate IK and swept collisions first.
    planning_blockers.append("Robot IK and full swept-volume collision validation have not run.")
    planning_blockers.append("This offline tool contains no xArm or GT200 command path.")
    report = {
        "status": "offline_candidate_filter_complete",
        "offline_only": True,
        "robot_connection_attempted": False,
        "execution_allowed": False,
        "prediction_count": int(len(predictions)),
        "passing_geometry_candidate_count": len(passing),
        "selected_candidate_index": None if selected is None else selected["index"],
        "candidates": candidate_rows,
        "tcp_waypoint_proposal": waypoints,
        "planning_ready_for_robot_kinematics": bool(waypoints is not None),
        "planning_blockers": planning_blockers,
        "input_semantics": {
            "predictions": "AnyPlace object-motion transforms in robot-base gravity frame",
            "target_proxy": "complete metric collision proxy in the same input pose/frame",
            "support": "support/base_obj points in robot-base gravity frame",
            "T_base_object_input": "object frame used by the target proxy before placement",
            "T_tcp_object": "per-grasp held-object relation, object coordinates mapped into TCP",
        },
    }
    write_json(output_dir / "candidate_filter_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter all AnyPlace candidates offline")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--target-proxy", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--T-base-object-input", dest="object_input", type=Path)
    parser.add_argument("--T-tcp-object", dest="tcp_object", type=Path)
    parser.add_argument("--obstacles", type=Path)
    parser.add_argument("--clearance-m", type=float, default=0.001)
    parser.add_argument("--approach-height-m", type=float, default=0.10)
    args = parser.parse_args()
    report = run(
        args.predictions,
        args.target_proxy,
        args.support,
        args.output_dir,
        object_input_transform_path=args.object_input,
        tcp_object_transform_path=args.tcp_object,
        obstacle_path=args.obstacles,
        clearance_m=args.clearance_m,
        approach_height_m=args.approach_height_m,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
