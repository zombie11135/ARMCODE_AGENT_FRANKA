from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from .geometry import (
    cloud_stats,
    load_json,
    rotation_checks,
    save_cloud_overlay,
    transform_points,
    write_json,
    write_ply,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_predictions(path: Path) -> np.ndarray:
    raw = np.load(str(path), allow_pickle=True)
    if raw.shape == (1,) and isinstance(raw[0], dict):
        predictions = raw[0]["relative"]
    elif raw.shape == () and isinstance(raw.item(), dict):
        predictions = raw.item()["relative"]
    else:
        raise ValueError("unrecognized upstream output structure: {}".format(raw.shape))
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 2:
        predictions = predictions[None, ...]
    if predictions.ndim != 3 or predictions.shape[1:] != (4, 4):
        raise ValueError("upstream relative transforms must have shape Kx4x4")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run upstream AnyPlace offline; never sends robot commands")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--anyplace-root", default=Path("/home/wpy/anyplace-main"), type=Path)
    parser.add_argument("--data-root", default=Path("/home/wpy/anyplace_data"), type=Path)
    parser.add_argument("--checkpoint", default=Path("/home/wpy/anyplace_data/anyplace_ckpts/anyplace_multitask/model.pth"), type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-refine-iters", default=10, type=int)
    parser.add_argument("--init-k", default=5, type=int)
    parser.add_argument("--candidate-index", default=0, type=int)
    parser.add_argument("--model-name", default="anyplace_stage1")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    anyplace_root = args.anyplace_root.resolve()
    checkpoint = args.checkpoint.resolve()
    base_points = np.load(str(input_dir / "base_obj_pcd.npy"))
    target_points = np.load(str(input_dir / "target_obj_pcd.npy"))
    if len(base_points) < 8192 or len(target_points) < 1024:
        raise ValueError("AnyPlace requires >=8192 base and >=1024 target points for this upstream implementation")

    os.environ["ANYPLACE_SOURCE_DIR"] = str(anyplace_root / "anyplace")
    os.environ["ANYPLACE_DATA_DIR"] = str(args.data_root.resolve())
    from anyplace.inference import inference_official
    from anyplace.utils import config_util

    config_path = anyplace_root / "anyplace/config/full_eval_cfgs/anyplace_inference/vial_insertion/anyplace_diffusion_molmocrop_multitask.yaml"
    runtime = config_util.load_config(str(config_path), demo_train_eval="eval")
    runtime["debug"] = False
    runtime["debug_data"] = False
    runtime["port_vis"] = 6000
    runtime["seed"] = args.seed
    runtime["local_dataset_dir"] = None
    runtime["export_viz"] = False
    runtime["export_viz_dirname"] = "rpdiff_export_viz"
    runtime["compute_coverage"] = False
    runtime["out_coverage_dirname"] = None
    runtime["new_meshcat"] = True
    runtime["experiment"]["parent_crop"] = False
    runtime["experiment"]["eval"].update({
        "data_path": str(input_dir),
        "ckpt_path": str(checkpoint),
        "base_obj_file_name": "base_obj_pcd",
        "target_obj_file_name": "target_obj_pcd",
        "use_molmo": False,
        "model_name": args.model_name,
        "n_refine_iters": args.n_refine_iters,
        "init_k_val": args.init_k,
    })
    runtime = config_util.recursive_attr_dict(runtime)

    output_path = input_dir / (args.model_name + "_relative_pose_prediction_real.npy")
    started = time.time()
    inference_official.main(runtime)
    elapsed = time.time() - started
    predictions = extract_predictions(output_path)
    if not 0 <= args.candidate_index < len(predictions):
        raise IndexError("candidate index outside returned prediction set")
    selected = predictions[args.candidate_index]
    transformed = transform_points(target_points, selected)
    np.save(str(input_dir / "placement_transform.npy"), selected)
    np.save(str(input_dir / "target_obj_pcd_transformed.npy"), transformed)
    write_ply(input_dir / "target_obj_pcd_transformed.ply", transformed)
    save_cloud_overlay(input_dir / "placement_overlay.png", base_points, target_points, transformed)

    preprocess_report_path = input_dir / "preprocess_report.json"
    preprocess_report = load_json(preprocess_report_path) if preprocess_report_path.exists() else {}
    coordinate_checks = preprocess_report.get("coordinate_checks", {})
    input_frame = coordinate_checks.get("actual_anyplace_input_frame", "official_dataset_world")
    transform_base = None
    t_robot_camera_path = input_dir / "T_legacy_xarmbase_camera.npy"
    if input_frame == "legacy_xarm_base_gravity_z_up" and t_robot_camera_path.exists():
        # The prepared model inputs already live in the legacy xArm base frame;
        # no camera-frame conjugation is required.
        transform_base = selected.copy()
        np.save(str(input_dir / "T_legacy_xarmbase_object_motion.npy"), transform_base)

    geometry_validation = None
    if input_frame == "legacy_xarm_base_gravity_z_up":
        base_min_xy = np.min(base_points[:, :2], axis=0)
        base_max_xy = np.max(base_points[:, :2], axis=0)
        base_surface_z = float(np.median(base_points[:, 2]))
        target_min_z = float(np.min(transformed[:, 2]))
        inside_xy = (
            (transformed[:, 0] >= base_min_xy[0])
            & (transformed[:, 0] <= base_max_xy[0])
            & (transformed[:, 1] >= base_min_xy[1])
            & (transformed[:, 1] <= base_max_xy[1])
        )
        clearance = target_min_z - base_surface_z
        geometry_validation = {
            "method": "gravity-z support-plane smoke check on the manually cropped table patch",
            "base_surface_median_z_m": base_surface_z,
            "transformed_target_min_z_m": target_min_z,
            "minimum_clearance_m": clearance,
            "estimated_penetration_m": max(0.0, -clearance),
            "target_points_inside_support_xy_aabb_fraction": float(np.mean(inside_xy)),
            "pass_thresholds": {"maximum_penetration_m": 0.005, "minimum_inside_xy_fraction": 0.95},
            "passed": bool(clearance >= -0.005 and float(np.mean(inside_xy)) >= 0.95),
            "limitation": "Not a mesh collision or stability test; the target cloud is a partial single-view surface.",
        }

    if input_frame == "legacy_xarm_base_gravity_z_up":
        planning_blockers = [
            "Repository hardware is xArm plus custom GT200 gripper, not the requested historical Franka plus Robotiq stack.",
            "The controller TCP offset is zero and no measured gripper/TCP-to-object transform is available.",
            "Hand-eye file has no residual, date, or camera/robot serial binding provenance.",
            "The saved target point cloud is a partial single-view surface, not a complete object geometry.",
        ]
    else:
        planning_blockers = ["Official dataset inference is a model smoke test, not a real-robot plan."]
    if geometry_validation is not None and not geometry_validation["passed"]:
        planning_blockers.append(
            "Selected model candidate fails the support-plane smoke check (estimated penetration {:.4f} m).".format(
                geometry_validation["estimated_penetration_m"]
            )
        )
    offline_plan = {
        "kind": "offline_object_motion_proposal",
        "execution_allowed": False,
        "robot_waypoints": None,
        "gripper_commands": None,
        "object_motion_frame": input_frame,
        "object_motion_transform": selected.tolist(),
        "geometry_validation": geometry_validation,
        "blockers": planning_blockers,
    }
    write_json(input_dir / "offline_plan.json", offline_plan)

    report = {
        "status": "offline_inference_complete",
        "upstream_entry": "anyplace.inference.inference_official.main",
        "upstream_root": str(anyplace_root),
        "upstream_commit": "3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed,
        "runtime_seconds": elapsed,
        "configuration": {
            "n_refine_iters": args.n_refine_iters,
            "init_k_val": args.init_k,
            "parent_crop": False,
            "use_molmo": False,
            "task_specific_crop": "disabled; the prepared base_obj is already an explicitly reviewed crop",
        },
        "prediction_count": int(len(predictions)),
        "selected_candidate_index": args.candidate_index,
        "candidate_selection_warning": "No success classifier is active; index 0 is a smoke-test proposal, not a validated optimum.",
        "transform_direction": "T_input_placed_target maps original target points into placed target points in the shared AnyPlace input frame",
        "anyplace_input_frame": input_frame,
        "T_input_placed_target": selected.tolist(),
        "rotation_check": rotation_checks(selected),
        "homogeneous_last_row_max_abs_error": float(np.max(np.abs(selected[3] - np.array([0, 0, 0, 1])))),
        "target_before": cloud_stats(target_points),
        "target_after": cloud_stats(transformed),
        "base_obj": cloud_stats(base_points),
        "T_legacy_xarmbase_object_motion": None if transform_base is None else transform_base.tolist(),
        "geometry_validation": geometry_validation,
        "offline_plan_ready_for_execution": False,
        "planning_blockers": planning_blockers,
        "robot_command_generated": False,
        "outputs": {
            "raw_prediction": str(output_path),
            "selected_transform_npy": str(input_dir / "placement_transform.npy"),
            "transformed_target_ply": str(input_dir / "target_obj_pcd_transformed.ply"),
            "overlay": str(input_dir / "placement_overlay.png"),
            "offline_plan": str(input_dir / "offline_plan.json"),
        },
    }
    write_json(input_dir / "inference_report.json", report)
    print(input_dir / "inference_report.json")


if __name__ == "__main__":
    main()
