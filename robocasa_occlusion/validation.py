"""Acceptance checks on rendered evidence, never used to make policy decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    results = manifest["results"]
    checks = {}
    if manifest.get("object_profile") == "graspable":
        screen = manifest["top_grasp_size_screen"]
        for name, row in screen["objects"].items():
            width = min(row["size_m"][:2])
            checks[f"grasp_size/{name}/opening_with_clearance"] = bool(
                width + 2 * screen["clearance_each_side_m"] <= screen["nominal_opening_m"] + 1e-9)
    for case in ["clear", "label_occluded", "fully_occluded"]:
        metrics = results[case]
        checks[f"{case}/settled"] = metrics["physics"]["all_objects_settled"]
        for camera in ["inspection", "overhead"]:
            root = output / case / "observation"
            depth = np.load(root / f"{camera}_depth_m.npy")
            rgb = np.asarray(Image.open(root / f"{camera}_rgb.png"))
            checks[f"{case}/{camera}/valid_rgbd"] = bool(
                rgb.shape[:2] == depth.shape and np.isfinite(depth).all()
                and (depth > 0).all() and rgb.std() > 10)
            checks[f"{case}/{camera}/target_area_in_frame"] = metrics[camera]["identity_rectangle_in_frame_fraction"] > .99
        checks[f"{case}/no_truth_files_in_observation"] = not any(
            "segmentation" in p.name or "identity" in p.name or "state" in p.name
            for p in (output / case / "observation").iterdir())
    for camera in ["inspection", "overhead"]:
        checks[f"{camera}/clear_identity_visible"] = results["clear"][camera]["identity_rectangle_visible_fraction"] > .95
        checks[f"{camera}/occluded_identity_hidden"] = results["label_occluded"][camera]["identity_rectangle_visible_fraction"] < .02
        checks[f"{camera}/partial_object_still_visible"] = results["label_occluded"][camera]["target_visible_pixels"] > 100
        checks[f"{camera}/full_object_hidden"] = results["fully_occluded"][camera]["target_visible_pixels"] == 0
    if "left_camera_deployment" in manifest:
        deployment = manifest["left_camera_deployment"]
        external_poses = []
        for case in ["clear", "label_occluded", "fully_occluded"]:
            root = output / case / "observation"
            camera_info = []
            for camera in ["left_oblique", "robot0_eye_in_hand"]:
                info = json.loads((root / f"{camera}_camera.json").read_text())
                camera_info.append(info)
                depth = np.load(root / f"{camera}_depth_m.npy")
                rgb = np.asarray(Image.open(root / f"{camera}_rgb.png"))
                checks[f"{case}/{camera}/valid_rgbd"] = bool(
                    rgb.shape == (info["height"], info["width"], 3)
                    and depth.shape == rgb.shape[:2] and np.isfinite(depth).all()
                    and (depth > 0).all() and rgb.std() > 10)
                world_camera = np.asarray(info["T_world_from_camera_optical"])
                world_base = np.asarray(info["T_world_from_arm_base"])
                base_camera = np.asarray(info["T_arm_base_from_camera_optical"])
                checks[f"{case}/{camera}/extrinsics_consistent"] = bool(
                    np.allclose(world_base @ base_camera, world_camera, atol=1e-8))
            left, wrist = camera_info
            pose = np.asarray(left["T_world_from_camera_optical"])
            external_poses.append(pose)
            setup_base = np.asarray(deployment["T_world_from_arm_base_at_setup"])
            setup_position = (np.linalg.inv(setup_base) @ pose)[:3, 3]
            checks[f"{case}/left_camera_requested_position"] = bool(
                np.allclose(setup_position, deployment["position_base_m"], atol=1e-6))
            direction = np.asarray(deployment["lookat_world_m"]) - pose[:3, 3]
            direction /= np.linalg.norm(direction)
            checks[f"{case}/left_camera_looks_at_work_area"] = bool(
                np.allclose(pose[:3, 2], direction, atol=1e-6))
            checks[f"{case}/paired_views_same_capture"] = bool(
                left["capture_group_id"] == wrist["capture_group_id"]
                and left["capture_timestamp_s"] == wrist["capture_timestamp_s"]
                and left["capture_timestamp_s"] == results[case]["physics"]["simulation_time_s"])
        checks["left_camera_fixed_in_world_across_conditions"] = all(
            np.allclose(external_poses[0], pose, atol=1e-8) for pose in external_poses[1:])
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "rendered geometry, RGB-D, settled physics, camera deployment and configured jaw-width budget; not executed grasp or agent success"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("runs/robocasa_graspable"))
    args = parser.parse_args()
    result = validate(args.output_dir)
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
