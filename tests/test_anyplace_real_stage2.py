import json
from pathlib import Path

import cv2
import numpy as np

from anyplace_real.candidate_filter import evaluate_candidate, run as run_candidate_filter
from anyplace_real.geometry import (
    fit_pivot_tcp,
    fit_tabletop_cuboid,
    sample_cuboid_surface,
    voxel_downsample,
)
from anyplace_real.multiview_fusion import run as run_multiview_fusion
from anyplace_real.held_object_relation import average_tcp_object_relations
from anyplace_real.tcp_calibration_audit import run as run_tcp_audit


def _rotation_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_pivot_fit_recovers_flange_tcp_translation() -> None:
    tcp = np.array([0.012, -0.007, 0.148])
    pivot = np.array([0.35, 0.04, 0.22])
    transforms = []
    for angles in ((0.0, 0.0, 0.0), (0.3, -0.2, 0.1), (-0.25, 0.35, -0.2), (0.2, 0.25, 0.4), (-0.4, -0.2, 0.2)):
        transform = np.eye(4)
        transform[:3, :3] = _rotation_xyz(*angles)
        transform[:3, 3] = pivot - transform[:3, :3] @ tcp
        transforms.append(transform)
    result = fit_pivot_tcp(transforms)
    assert result["linear_system_rank"] == 6
    assert np.allclose(result["tcp_translation_flange_m"], tcp, atol=1e-10)
    assert result["max_residual_m"] < 1e-10


def test_tcp_audit_writes_full_transform_only_with_orientation(tmp_path: Path) -> None:
    tcp = np.array([0.012, -0.007, 0.148])
    pivot = np.array([0.35, 0.04, 0.22])
    teaching = []
    angles_list = (
        (0.0, 0.0, 0.0),
        (0.3, -0.2, 0.1),
        (-0.25, 0.35, -0.2),
        (0.2, 0.25, 0.4),
    )
    for index, angles in enumerate(angles_list):
        rotation = _rotation_xyz(*angles)
        translation = pivot - rotation @ tcp
        teaching.append(
            {
                "name": "p{}".format(index),
                "pose": [*(translation * 1000.0), *angles],
            }
        )
    manifest = {
        "example_only": False,
        "pose_semantics": "T_base_flange_mmrad",
        "teaching_poses_mmrad": teaching,
        "tcp_orientation_rpy_rad": [0.0, 0.0, 0.0],
        "acceptance": {"minimum_orientation_spread_deg": 20.0},
    }
    path = tmp_path / "pivot.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = run_tcp_audit(path, tmp_path / "tcp")
    assert report["acceptance"]["ready_for_planning"] is True
    assert (tmp_path / "tcp/T_flange_tcp.npy").exists()


def test_voxel_fusion_and_tabletop_cuboid_proxy() -> None:
    points = np.array([[0.001, 0.001, 0.03], [0.0015, 0.001, 0.03], [0.10, 0.02, 0.03]])
    colors = np.ones_like(points)
    reduced, reduced_colors = voxel_downsample(points, colors, 0.01)
    assert len(reduced) == 2
    assert reduced_colors.shape == reduced.shape

    rng = np.random.default_rng(2)
    observed = np.column_stack(
        (
            rng.uniform(-0.06, 0.06, 500),
            rng.uniform(-0.035, 0.035, 500),
            np.full(500, 0.03),
        )
    )
    transform = fit_tabletop_cuboid(observed, [0.12, 0.07, 0.03], 0.0)
    proxy = sample_cuboid_surface([0.12, 0.07, 0.03], transform, spacing_m=0.01)
    assert np.isclose(transform[2, 3], 0.015)
    assert np.isclose(proxy[:, 2].min(), 0.0)
    assert np.isclose(proxy[:, 2].max(), 0.03)


def test_multiview_fusion_cli_core_keeps_observed_and_proxy_separate(tmp_path: Path) -> None:
    height, width = 24, 32
    color = np.full((height, width, 3), 120, np.uint8)
    depth = np.full((height, width), 500, np.uint16)
    mask = np.ones((height, width), np.uint8)
    frames = []
    for index, translation_x in enumerate((0.0, 0.01)):
        color_path = tmp_path / "color_{}.png".format(index)
        depth_path = tmp_path / "depth_{}.png".format(index)
        mask_path = tmp_path / "mask_{}.npy".format(index)
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_path), depth)
        np.save(mask_path, mask)
        camera = np.eye(4)
        camera[0, 3] = translation_x
        frames.append(
            {
                "color_path": color_path.name,
                "depth_path": depth_path.name,
                "target_mask": {"kind": "npy", "path": mask_path.name},
                "depth_aligned_to_color": True,
                "T_base_camera": camera.tolist(),
            }
        )
    manifest = {
        "example_only": False,
        "project_root": ".",
        "shared": {
            "depth_unit_m_per_raw_unit": 0.001,
            "aligned_intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 16.0, "cy": 12.0},
            "min_depth_m": 0.2,
            "max_depth_m": 0.8,
        },
        "frames": frames,
        "fusion": {"voxel_size_m": 0.002},
        "cuboid_proxy": {
            "dimensions_m": [0.13, 0.09, 0.03],
            "support_plane_z_m": 0.0,
            "surface_spacing_m": 0.01,
        },
    }
    path = tmp_path / "multiview.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = run_multiview_fusion(path, tmp_path / "fusion")
    assert report["input_frame_count"] == 2
    assert report["cuboid_proxy"]["not_measured_surface_warning"]
    assert (tmp_path / "fusion/observed_fused_cloud.npy").exists()
    assert (tmp_path / "fusion/complete_collision_proxy.npy").exists()


def test_candidate_filter_projects_to_support_and_never_authorizes_execution(tmp_path: Path) -> None:
    identity = np.eye(4)
    target = sample_cuboid_surface([0.08, 0.05, 0.04], identity, spacing_m=0.01)
    grid_x, grid_y = np.meshgrid(np.linspace(-0.2, 0.2, 30), np.linspace(-0.2, 0.2, 30))
    support = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)))
    corrected, metrics = evaluate_candidate(identity, target, support, clearance_m=0.001)
    assert metrics["passed_geometry_filter"]
    assert np.isclose(metrics["minimum_clearance_after_projection_m"], 0.001)
    assert np.isclose(corrected[2, 3], 0.021)

    predictions_path = tmp_path / "predictions.npy"
    np.save(predictions_path, np.array([{"relative": np.asarray([identity])}], dtype=object))
    target_path = tmp_path / "target.npy"
    support_path = tmp_path / "support.npy"
    object_path = tmp_path / "T_base_object.npy"
    held_path = tmp_path / "T_tcp_object.npy"
    np.save(target_path, target)
    np.save(support_path, support)
    np.save(object_path, identity)
    np.save(held_path, identity)
    report = run_candidate_filter(
        predictions_path,
        target_path,
        support_path,
        tmp_path / "out",
        object_input_transform_path=object_path,
        tcp_object_transform_path=held_path,
    )
    assert report["passing_geometry_candidate_count"] == 1
    assert report["planning_ready_for_robot_kinematics"] is True
    assert report["tcp_waypoint_proposal"] is not None
    assert report["execution_allowed"] is False
    assert report["robot_connection_attempted"] is False


def test_held_object_relation_average_reports_small_scatter() -> None:
    expected = np.eye(4)
    expected[:3, 3] = [0.01, -0.02, 0.08]
    relations = []
    for offset in (-0.0005, 0.0, 0.0005):
        relation = expected.copy()
        relation[0, 3] += offset
        relations.append(relation)
    mean, scatter = average_tcp_object_relations(relations)
    assert np.allclose(mean, expected)
    assert scatter["translation_rms_m"] < 0.001
    assert scatter["rotation_rms_deg"] < 1e-9
