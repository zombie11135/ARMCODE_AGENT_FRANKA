from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from .geometry import (
    cloud_stats,
    deterministic_subsample,
    load_json,
    load_mask,
    normalized_intrinsics,
    pose_mrad_to_transform,
    resolve_path,
    rotation_checks,
    rgbd_points,
    save_mask_review,
    transform_points,
    write_json,
    write_ply,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare recorded RGB-D for offline AnyPlace inference")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    project_root = resolve_path(manifest_path.parent, manifest.get("project_root", ".")).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    color_path = resolve_path(project_root, manifest["rgbd"]["color_path"])
    depth_path = resolve_path(project_root, manifest["rgbd"]["depth_path"])
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None:
        raise FileNotFoundError(color_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if manifest["rgbd"].get("depth_aligned_to_color") is not True:
        raise ValueError("manifest must explicitly confirm depth_aligned_to_color=true")

    intrinsics = normalized_intrinsics(manifest["rgbd"]["aligned_intrinsics"])
    depth_unit = float(manifest["rgbd"]["depth_unit_m_per_raw_unit"])
    target_spec = manifest["segmentation"]["target"]
    base_spec = manifest["segmentation"]["base_obj"]
    target_mask = load_mask(target_spec, depth.shape, project_root)
    base_mask = load_mask(base_spec, depth.shape, project_root)

    target_points, target_colors, target_valid = rgbd_points(
        depth, color, target_mask, intrinsics, depth_unit,
        float(target_spec["min_depth_m"]), float(target_spec["max_depth_m"]),
    )
    base_points, base_colors, base_valid = rgbd_points(
        depth, color, base_mask, intrinsics, depth_unit,
        float(base_spec["min_depth_m"]), float(base_spec["max_depth_m"]),
    )
    target_points, target_colors = deterministic_subsample(target_points, target_colors, int(target_spec.get("max_points", 30000)))
    base_points, base_colors = deterministic_subsample(base_points, base_colors, int(base_spec.get("max_points", 30000)))
    if len(target_points) < 1024:
        raise ValueError("target point cloud has fewer than 1024 valid points")
    if len(base_points) < 8192:
        raise ValueError("base_obj point cloud has fewer than 8192 valid points")
    target_points_camera = target_points.copy()

    np.save(str(output_dir / "target_mask_valid.npy"), target_valid)
    np.save(str(output_dir / "base_mask_valid.npy"), base_valid)
    write_ply(output_dir / "target_obj_pcd_camera.ply", target_points, target_colors)
    write_ply(output_dir / "base_obj_pcd_camera.ply", base_points, base_colors)
    save_mask_review(output_dir / "segmentation_review.png", color, target_valid, base_valid)

    coordinate_checks = {
        "sensor_point_cloud_frame": manifest["coordinates"]["sensor_point_cloud_frame"],
        "requested_anyplace_input_frame": manifest["coordinates"]["anyplace_input_frame"],
        "depth_backprojection": "x=(u-cx)z/fx, y=(v-cy)z/fy, z=depth; OpenCV optical frame",
        "depth_unit_m_per_raw_unit": depth_unit,
        "base_obj_semantics": "AnyPlace parent/support object; not the robot base frame",
        "target_camera": cloud_stats(target_points),
        "base_obj_camera": cloud_stats(base_points),
    }

    handeye_path_value = manifest["coordinates"].get("handeye_path")
    ee_pose = manifest["coordinates"].get("capture_ee_pose_mrad")
    if handeye_path_value and ee_pose:
        handeye_path = resolve_path(project_root, handeye_path_value)
        with handeye_path.open("r", encoding="utf-8") as stream:
            handeye = yaml.safe_load(stream)
        if handeye.get("frame_type") != "camera_to_ee":
            raise ValueError("stage-1 adapter currently requires camera_to_ee hand-eye data")
        t_ee_camera = np.eye(4, dtype=np.float64)
        t_ee_camera[:3, :3] = np.asarray(handeye["rotation"], dtype=np.float64)
        t_ee_camera[:3, 3] = np.asarray(handeye["translation"], dtype=np.float64)
        if handeye.get("unit", "meter") in ("mm", "millimeter"):
            t_ee_camera[:3, 3] /= 1000.0
        t_robot_ee = pose_mrad_to_transform(ee_pose)
        t_robot_camera = t_robot_ee @ t_ee_camera
        np.save(str(output_dir / "T_legacy_xarmbase_camera.npy"), t_robot_camera)
        if manifest["coordinates"]["anyplace_input_frame"] != "legacy_xarm_base_gravity_z_up":
            raise ValueError("capture transform is present but AnyPlace input frame is not legacy_xarm_base_gravity_z_up")
        target_points = transform_points(target_points, t_robot_camera)
        base_points = transform_points(base_points, t_robot_camera)
        coordinate_checks.update({
            "actual_anyplace_input_frame": "legacy_xarm_base_gravity_z_up",
            "handeye_direction": "T_ee_camera maps camera coordinates into end-effector coordinates",
            "T_ee_camera": t_ee_camera.tolist(),
            "T_ee_camera_rotation_check": rotation_checks(t_ee_camera),
            "capture_ee_pose_mrad": list(ee_pose),
            "T_legacy_xarmbase_camera": t_robot_camera.tolist(),
            "T_legacy_xarmbase_camera_rotation_check": rotation_checks(t_robot_camera),
            "roundtrip_max_abs_m": float(np.max(np.abs(transform_points(transform_points(target_points_camera[:128], t_robot_camera), np.linalg.inv(t_robot_camera)) - target_points_camera[:128]))),
            "robot_frame_warning": "This is the legacy xArm base, not AnyPlace base_obj and not a Franka base.",
            "target_anyplace_input": cloud_stats(target_points),
            "base_obj_anyplace_input": cloud_stats(base_points),
        })
    else:
        raise ValueError("AnyPlace gravity-frame input requires capture-time EE pose and hand-eye calibration")

    # Upstream AnyPlace consumes meter-scale point clouds in a shared,
    # gravity-aligned world frame. Camera-frame clouds above are retained only
    # for visual audit; the arrays below are the actual model inputs.
    np.save(str(output_dir / "target_obj_pcd.npy"), target_points)
    np.save(str(output_dir / "base_obj_pcd.npy"), base_points)
    write_ply(output_dir / "target_obj_pcd.ply", target_points, target_colors)
    write_ply(output_dir / "base_obj_pcd.ply", base_points, base_colors)
    write_ply(output_dir / "local_crop_pcd.ply", base_points, base_colors)

    report = {
        "status": "prepared_offline",
        "manifest": str(manifest_path),
        "source_color": str(color_path),
        "source_depth": str(depth_path),
        "segmentation_assistance": manifest["segmentation"].get("assistance"),
        "coordinate_checks": coordinate_checks,
        "outputs": {
            "target_obj_pointcloud": str(output_dir / "target_obj_pcd.ply"),
            "base_obj_pointcloud": str(output_dir / "base_obj_pcd.ply"),
            "local_crop": str(output_dir / "local_crop_pcd.ply"),
            "segmentation_review": str(output_dir / "segmentation_review.png"),
        },
    }
    write_json(output_dir / "preprocess_report.json", report)
    print(output_dir / "preprocess_report.json")


if __name__ == "__main__":
    main()
