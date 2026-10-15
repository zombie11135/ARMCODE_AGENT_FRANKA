"""Fuse saved masked RGB-D views in a common robot-base frame, offline only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from .geometry import (
    cloud_stats,
    fit_tabletop_cuboid,
    load_json,
    load_mask,
    normalized_intrinsics,
    pose_mmrad_to_transform,
    resolve_path,
    rgbd_points,
    sample_cuboid_surface,
    transform_points,
    voxel_downsample,
    write_json,
    write_ply,
)


def _load_transform(frame: Dict[str, Any], project_root: Path, shared: Dict[str, Any]) -> np.ndarray:
    if frame.get("T_base_camera") is not None:
        transform = np.asarray(frame["T_base_camera"], dtype=np.float64)
    else:
        pose = frame.get("flange_pose_mmrad")
        handeye_value = frame.get("T_flange_camera")
        if handeye_value is None:
            handeye_value = shared.get("T_flange_camera")
        if handeye_value is None:
            handeye_path = frame.get("handeye_path") or shared.get("handeye_path")
            if not handeye_path:
                raise ValueError("frame requires T_base_camera or flange pose plus hand-eye")
            with resolve_path(project_root, handeye_path).open("r", encoding="utf-8") as stream:
                handeye = yaml.safe_load(stream)
            if handeye.get("frame_type") != "camera_to_ee":
                raise ValueError("hand-eye file must explicitly be camera_to_ee/T_flange_camera")
            handeye_value = np.eye(4, dtype=np.float64)
            handeye_value[:3, :3] = np.asarray(handeye["rotation"], dtype=np.float64)
            handeye_value[:3, 3] = np.asarray(handeye["translation"], dtype=np.float64)
            if handeye.get("unit", "meter") in ("mm", "millimeter"):
                handeye_value[:3, 3] /= 1000.0
        if pose is None:
            raise ValueError("flange_pose_mmrad is required when T_base_camera is absent")
        transform = pose_mmrad_to_transform(pose) @ np.asarray(handeye_value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("camera transform must be 4x4")
    return transform


def run(manifest_path: Path, output_dir: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("example_only", False):
        raise ValueError("example_only=true: replace paths/poses and set it to false")
    project_root = resolve_path(manifest_path.parent, manifest.get("project_root", ".")).resolve()
    shared = manifest.get("shared", {})
    frame_reports = []
    all_points = []
    all_colors = []

    for index, frame in enumerate(manifest.get("frames") or []):
        if frame.get("depth_aligned_to_color") is not True:
            raise ValueError("every frame must explicitly confirm depth_aligned_to_color=true")
        color_path = resolve_path(project_root, frame["color_path"])
        depth_path = resolve_path(project_root, frame["depth_path"])
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise FileNotFoundError("missing RGB-D frame {} / {}".format(color_path, depth_path))
        intrinsics = normalized_intrinsics(frame.get("aligned_intrinsics") or shared["aligned_intrinsics"])
        mask = load_mask(frame["target_mask"], depth.shape, project_root)
        points, colors, valid = rgbd_points(
            depth,
            color,
            mask,
            intrinsics,
            float(frame.get("depth_unit_m_per_raw_unit", shared["depth_unit_m_per_raw_unit"])),
            float(frame.get("min_depth_m", shared.get("min_depth_m", 0.10))),
            float(frame.get("max_depth_m", shared.get("max_depth_m", 1.50))),
        )
        transform = _load_transform(frame, project_root, shared)
        points_base = transform_points(points, transform)
        all_points.append(points_base)
        all_colors.append(colors)
        frame_reports.append(
            {
                "index": index,
                "color_path": str(color_path),
                "depth_path": str(depth_path),
                "valid_target_pixels": int(np.count_nonzero(valid)),
                "cloud_base": cloud_stats(points_base),
                "T_base_camera": transform.tolist(),
            }
        )
    if len(all_points) < 2:
        raise ValueError("multiview fusion requires at least two saved views")
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    voxel_size = float(manifest.get("fusion", {}).get("voxel_size_m", 0.0025))
    points, colors = voxel_downsample(points, colors, voxel_size)
    if len(points) < 100:
        raise ValueError("fused target cloud has fewer than 100 points")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "observed_fused_cloud.npy"), points)
    np.save(str(output_dir / "observed_fused_colors.npy"), colors)
    write_ply(output_dir / "observed_fused_cloud.ply", points, colors)

    cuboid_report: Optional[dict] = None
    cuboid = manifest.get("cuboid_proxy")
    if cuboid:
        dimensions = [float(value) for value in cuboid["dimensions_m"]]
        transform = fit_tabletop_cuboid(
            points,
            dimensions,
            float(cuboid["support_plane_z_m"]),
        )
        proxy = sample_cuboid_surface(
            dimensions,
            transform,
            float(cuboid.get("surface_spacing_m", voxel_size)),
        )
        np.save(str(output_dir / "complete_collision_proxy.npy"), proxy)
        np.save(str(output_dir / "T_base_object_input.npy"), transform)
        write_ply(output_dir / "complete_collision_proxy.ply", proxy)
        cuboid_report = {
            "kind": "measured-dimensions_tabletop_cuboid_proxy",
            "dimensions_m": dimensions,
            "support_plane_z_m": float(cuboid["support_plane_z_m"]),
            "T_base_object_input": transform.tolist(),
            "point_count": int(len(proxy)),
            "orientation_warning": "PCA yaw has a 180-degree sign ambiguity; resolve it from package artwork if task semantics require it.",
            "not_measured_surface_warning": "Unobserved faces are a parametric collision proxy, not measured depth.",
        }

    report = {
        "status": "offline_multiview_fusion_complete",
        "offline_only": True,
        "robot_connection_attempted": False,
        "manifest": str(manifest_path),
        "input_frame_count": len(frame_reports),
        "registration": "capture-time robot kinematics plus hand-eye; no unconstrained ICP drift",
        "voxel_size_m": voxel_size,
        "observed_fused_cloud": cloud_stats(points),
        "frames": frame_reports,
        "cuboid_proxy": cuboid_report,
        "outputs": {
            "observed_fused_cloud": str(output_dir / "observed_fused_cloud.ply"),
            "complete_collision_proxy": (
                str(output_dir / "complete_collision_proxy.ply") if cuboid_report else None
            ),
            "T_base_object_input": (
                str(output_dir / "T_base_object_input.npy") if cuboid_report else None
            ),
        },
    }
    write_json(output_dir / "multiview_fusion_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse saved masked RGB-D views offline")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
