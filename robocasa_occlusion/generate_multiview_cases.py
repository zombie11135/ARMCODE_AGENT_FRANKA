"""Render varied physical arrangements using the saved official RoboCasa model.

Evaluation truth uses counterfactual renders with other task objects hidden.
It does not use the inference algorithm's point projection or plane fitting.
These are initialized/settled states, not robot-executed clearing trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .probe_multiview import CAMERAS


def generate(scene_dir, output_dir, count=24, seed=20261020, width=640, height=480):
    manifest = json.loads((Path(scene_dir) / "manifest.json").read_text())
    if manifest.get("object_profile") != "graspable":
        raise ValueError("this arrangement generator requires the graspable object-size profile")
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if Path(output_dir).resolve() in {Path(scene_dir).resolve(), (Path(scene_dir)/"clear").resolve()}:
        raise ValueError("output must differ from the source scene directories")
    from .rendering import configure_renderer
    configure_renderer()
    import mujoco
    from .probe_multiview import geometry_groups
    source = Path(scene_dir) / "clear"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "cases.json").exists():
        raise FileExistsError("existing case suite; choose another output directory")
    model = mujoco.MjModel.from_xml_path(str((source / "evaluation/scene.xml").resolve()))
    data = mujoco.MjData(model)
    state = np.load(source / "evaluation/physics_state.npz", allow_pickle=False)
    base_qpos = state["qpos"].copy()
    groups = geometry_groups(source)
    original_groups = model.geom_group.copy()
    addresses = {}
    velocities = {}
    for name in groups:
        joint = model.joint(name + "_joint0")
        addresses[name] = int(joint.qposadr[0])
        velocities[name] = int(joint.dofadr[0])
    # Preserve the specified physical external-camera deployment exactly.
    left_meta = json.loads((source / "observation/left_oblique_camera.json").read_text())
    left = model.camera("left_oblique")
    transform = np.asarray(left_meta["T_world_from_camera_optical"])
    left.pos[:] = transform[:3, 3]
    left.quat[:] = Rotation.from_matrix(transform[:3, :3] @ np.diag([1, -1, -1])).as_quat()[[3, 0, 1, 2]]
    rng = np.random.default_rng(seed)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0  # visual geoms, same convention as the scene exporter
    option.geomgroup[5] = 0  # reserved for evaluator-only counterfactual hiding
    renderer = mujoco.Renderer(model, height=height, width=width)
    rows = []

    def render(camera, mode):
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        if mode == "depth":
            renderer.enable_depth_rendering()
        elif mode == "segmentation":
            renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = renderer.render().copy()
        # Native MuJoCo renderer is top-left, metric optical Z; segmentation is
        # (object ID, object type), unlike the saved robosuite adapter.
        return image[..., ::-1].copy() if mode == "segmentation" else image

    try:
        for index in range(count):
            scenario = ("clear", "label_occluded", "fully_occluded")[index % 3]
            case_id = f"case_{index:03d}"
            root = output_dir / case_id
            policy, evaluation = root / "observation", root / "evaluation"
            policy.mkdir(parents=True, exist_ok=False)
            evaluation.mkdir()
            data.qpos[:] = base_qpos
            data.qvel[:] = 0
            data.time = 0
            yaw = {}
            for name, address in addresses.items():
                delta = rng.uniform(-.025, .025, 2)
                data.qpos[address:address+2] += delta
                yaw[name] = float(rng.uniform(-35, 35))
                rotation = Rotation.from_euler("z", yaw[name], degrees=True) * Rotation.from_quat(
                    base_qpos[address+3:address+7][[1, 2, 3, 0]])
                data.qpos[address+3:address+7] = rotation.as_quat()[[3, 0, 1, 2]]
            target = f"carton_{(index // 3) % 3}"
            if scenario != "clear":
                blocker = "blocker_small" if scenario == "label_occluded" else "blocker_large"
                back, front = addresses[target], addresses[blocker]
                data.qpos[front:front+2] = data.qpos[back:back+2] + rng.uniform(-.004, .004, 2)
                data.qpos[front+2] = data.qpos[back+2] + (.025 + (.025 if blocker == "blocker_small" else .04)) / 2 + .001
                # Keep the center supported while varying the overlap and orientation.
                relative_yaw = float(rng.uniform(-8, 8))
                rotation = Rotation.from_euler("z", relative_yaw, degrees=True) * Rotation.from_quat(
                    data.qpos[back+3:back+7][[1, 2, 3, 0]])
                data.qpos[front+3:front+7] = rotation.as_quat()[[3, 0, 1, 2]]
            # Settle objects under gravity; hold non-task joints, including the
            # robot, at the saved pose. This is not a robot control-policy test.
            fixed = np.ones(model.nq, bool)
            for address in addresses.values():
                fixed[address:address+7] = False
            moving = np.zeros(model.nv, bool)
            for address in velocities.values():
                moving[address:address+6] = True
            fixed_qpos = data.qpos.copy()
            for _ in range(250):
                data.qpos[fixed] = fixed_qpos[fixed]
                data.qvel[~moving] = 0
                mujoco.mj_step(model, data)
            data.qpos[fixed] = fixed_qpos[fixed]
            data.qvel[~moving] = 0
            mujoco.mj_forward(model, data)
            raw_segments = {}
            counts = {}
            for camera in CAMERAS:
                rgb, depth, seg = render(camera, "rgb"), render(camera, "depth"), render(camera, "segmentation")
                cv2.imwrite(str(policy / f"{camera}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.save(policy / f"{camera}_depth_m.npy", depth)
                np.save(evaluation / f"{camera}_segmentation.npy", seg)
                camera_id = model.camera(camera).id
                focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
                pose = np.eye(4)
                pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
                pose[:3, 3] = data.cam_xpos[camera_id]
                meta = {
                    "camera_id": camera, "capture_group_id": f"geometry_suite_{seed}_{case_id}",
                    "capture_timestamp_s": float(data.time), "timestamp_domain": "simulation",
                    "width": width, "height": height, "depth_aligned_to_color": True,
                    "K": [[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]],
                    "T_world_from_camera_optical": pose.tolist(),
                    "synchronization": "same_simulation_state_no_steps_between_views",
                }
                (policy / f"{camera}_camera.json").write_text(json.dumps(meta, indent=2))
                raw_segments[camera] = seg
                counts[camera] = {
                    name: int(((seg[..., 0] == 5) & np.isin(seg[..., 1], ids)).sum())
                    for name, ids in groups.items()
                }
            truth = []
            # Remove other task objects only for evaluator renders. Objects and
            # their full silhouettes are never passed to inference.
            for background, ids in groups.items():
                model.geom_group[:] = original_groups
                for other, other_ids in groups.items():
                    if other != background:
                        model.geom_group[other_ids] = 5
                for camera in CAMERAS:
                    isolated = render(camera, "segmentation")
                    silhouette = (isolated[..., 0] == 5) & np.isin(isolated[..., 1], ids)
                    area = int(silhouette.sum())
                    for foreground, other_ids in groups.items():
                        if foreground == background:
                            continue
                        actual = raw_segments[camera]
                        covered = silhouette & (actual[..., 0] == 5) & np.isin(actual[..., 1], other_ids)
                        pixels = int(covered.sum())
                        if pixels >= 6 and pixels / max(1, area) >= .03:
                            truth.append({
                                "foreground": foreground, "background": background,
                                "view": camera, "occluded_pixels": pixels,
                                "counterfactual_silhouette_pixels": area,
                                "background_observed_in_any_view": any(c[background] > 0 for c in counts.values()),
                            })
            model.geom_group[:] = original_groups
            (evaluation / "scene.xml").symlink_to((source / "evaluation/scene.xml").resolve())
            np.savez(evaluation / "physics_state.npz", qpos=data.qpos.copy(), qvel=data.qvel.copy(), time=data.time)
            metrics = {
                "case_id": case_id, "scenario": scenario, "target": target,
                "truth_method": "counterfactual_object_isolation_renders_not_inference_projection",
                "truth_min_pixels": 6, "truth_min_fraction": .03,
                "visible_pixels": counts, "occlusion_relations": truth,
                "max_object_linear_speed_m_s": max(float(np.linalg.norm(data.qvel[v:v+3])) for v in velocities.values()),
                "max_object_angular_speed_rad_s": max(float(np.linalg.norm(data.qvel[v+3:v+6])) for v in velocities.values()),
            }
            (evaluation / "ground_truth.json").write_text(json.dumps(metrics, indent=2))
            rows.append(metrics)
            print(f"{case_id}: {scenario}, target {target}, {len(truth)} view-specific truth edges", flush=True)
    finally:
        renderer.close()
    report = {
        "seed": seed, "case_count": count, "resolution": [width, height],
        "source_scene": str(source.resolve()), "camera_positions_changed": False,
        "variation": "independent XY +/-25 mm, yaw +/-35 deg, 3 target slots, blocker offset +/-4 mm and relative yaw +/-8 deg",
        "physics": "objects settled 250 MuJoCo steps; non-task joints held at saved state",
        "scope": "one kitchen and object-size profile; no robot-executed grasp or clearing",
        "cases": rows,
    }
    (output_dir / "cases.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("runs/robocasa_graspable"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/occlusion_multiview/geometry_holdout"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    generate(args.scene_dir, args.output_dir, args.count, args.seed)


if __name__ == "__main__":
    main()
