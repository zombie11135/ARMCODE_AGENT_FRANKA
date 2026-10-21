"""Build, render and inspect the fixed kitchen scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .assets import (generate_assets, chinese_font, DEFAULT_OBJECT_PROFILE,
                     OBJECT_PROFILES, top_grasp_size_screen)
from .cameras import ARM_BASE_BODY, CAMERA_NAMES, LEFT_CAMERA, LeftCameraDeployment


def make_env(args, assets):
    from .rendering import configure_renderer
    configure_renderer()
    from robosuite.controllers import load_composite_controller_config
    from .environment import OcclusionCounter

    return OcclusionCounter(
        assets=assets, scenario=args.scenario, scene_seed=args.seed,
        left_camera_deployment=LeftCameraDeployment(
            position_base_m=tuple(getattr(args, "left_camera_position", (-.25, .45, .89))),
            lookat_base_m=getattr(args, "left_camera_lookat", None),
            fovy_deg=getattr(args, "left_camera_fovy", 60.0)),
        robots="PandaOmron", controller_configs=load_composite_controller_config(robot="PandaOmron"),
        layout_ids=[args.layout], style_ids=[args.style],
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        camera_names=["inspection"], initialization_noise=None,
        render_collision_mesh=False, render_visual_mesh=True, ignore_done=True,
    )


def camera_capture(env, camera, width, height):
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map,
    )
    rgb, raw_depth = env.sim.render(width, height, camera_name=camera, depth=True)
    segmentation = env.sim.render(width, height, camera_name=camera, segmentation=True)
    return (rgb[::-1].copy(), get_real_depth_map(env.sim, raw_depth)[::-1].astype(np.float32),
            segmentation[::-1].copy(),
            get_camera_intrinsic_matrix(env.sim, camera, height, width),
            get_camera_extrinsic_matrix(env.sim, camera))


def label_visibility(env, segmentation, intrinsic, camera_to_world):
    """Evaluator only: sample the printed identity rectangle and test z-buffer ownership."""
    key = env.target_key
    body_id = env.obj_body_id[key]
    asset = env.scene_assets["cartons"][0]
    carton_size = asset["size_m"]
    x1, y1, x2, y2 = asset["text_bbox_px"]
    w, h = asset["image_size"]
    u, v = np.meshgrid(np.linspace(x1, x2, 60), np.linspace(y1, y2, 30))
    local = np.column_stack([(u.ravel() / w - .5) * carton_size[0],
                             (.5 - v.ravel() / h) * carton_size[1],
                             np.full(u.size, carton_size[2] / 2 + .00015)])
    rotation = env.sim.data.body_xmat[body_id].reshape(3, 3)
    world = local @ rotation.T + env.sim.data.body_xpos[body_id]
    camera = (np.linalg.inv(camera_to_world) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
    pixels = camera @ intrinsic.T
    pixels = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(int)
    image_h, image_w = segmentation.shape[:2]
    valid = ((camera[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < image_w)
             & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_h))
    label_id = env.sim.model.geom_name2id(f"{key}_label")
    visible = np.zeros(len(pixels), dtype=bool)
    visible[valid] = segmentation[pixels[valid, 1], pixels[valid, 0], 1] == label_id
    geom_ids = [env.sim.model.geom_name2id(n) for n in env.objects[key].visual_geoms]
    target_pixels = int(np.isin(segmentation[:, :, 1], geom_ids).sum())
    return {"target_visible_pixels": target_pixels,
            "identity_rectangle_visible_fraction": float(visible.mean()),
            "identity_rectangle_in_frame_fraction": float(valid.mean()),
            "method": "ground_truth_geometry_and_segmentation_for_evaluation_only",
            "ocr_verified": False}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))


def capture_scene(env, root, width, height):
    root.mkdir(parents=True, exist_ok=True)
    policy = root / "observation"
    truth = root / "evaluation"
    policy.mkdir(exist_ok=True)
    truth.mkdir(exist_ok=True)
    metrics = {}
    capture_group = uuid.uuid4().hex
    capture_time = float(env.sim.data.time)
    world_from_base = env.arm_base_pose()
    for camera in CAMERA_NAMES:
        rgb, depth, segmentation, intrinsic, extrinsic = camera_capture(env, camera, width, height)
        Image.fromarray(rgb).save(policy / f"{camera}_rgb.png")
        np.save(policy / f"{camera}_depth_m.npy", depth)
        np.save(truth / f"{camera}_segmentation.npy", segmentation)
        write_json(policy / f"{camera}_camera.json", {
            "camera_id": camera, "capture_group_id": capture_group,
            "capture_timestamp_s": capture_time, "timestamp_domain": "simulation",
            "synchronization": "same_simulation_state_no_steps_between_views",
            "width": width, "height": height, "K": intrinsic,
            "T_world_from_camera_optical": extrinsic, "depth_units": "metres",
            "T_world_from_arm_base": world_from_base, "arm_base_body": ARM_BASE_BODY,
            "T_arm_base_from_camera_optical": np.linalg.inv(world_from_base) @ extrinsic,
            "depth_definition": "positive optical Z, not Euclidean range",
            "depth_aligned_to_color": True,
            "image_origin": "top_left", "optical_axes": "x right, y down, z forward",
            **({"deployment": env.left_camera_setup} if camera == LEFT_CAMERA else {}),
        })
        metrics[camera] = label_visibility(env, segmentation, intrinsic, extrinsic)
    object_state = {}
    for name, obj in env.objects.items():
        body = env.obj_body_id[name]
        velocity = env.sim.data.get_joint_qvel(obj.joints[0]).copy()
        object_state[name] = {"position_world_m": env.sim.data.body_xpos[body].copy(),
                              "quaternion_wxyz": env.sim.data.body_xquat[body].copy(),
                              "linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
                              "angular_speed_rad_s": float(np.linalg.norm(velocity[3:]))}
    metrics["physics"] = {"objects": object_state,
                          "all_objects_settled": all(v["linear_speed_m_s"] < .01 and
                                                     v["angular_speed_rad_s"] < .1 for v in object_state.values()),
                          "simulation_time_s": float(env.sim.data.time)}
    write_json(truth / "metrics.json", metrics)
    write_json(truth / "identity.json", {"target": env.target_key, "label_order": env.label_order,
                                         "target_name": "布洛芬", "scenario": env.scenario})
    (truth / "scene.xml").write_text(env.sim.model.get_xml())
    np.savez(truth / "physics_state.npz", qpos=env.sim.data.qpos.copy(), qvel=env.sim.data.qvel.copy(),
             time=np.array(env.sim.data.time))
    return metrics


def camera_comparison(output, scenarios, font_path=None):
    """Paired external/wrist observations; every column uses one frozen scene."""
    font, _ = chinese_font(font_path, 23)
    titles = {"clear": "无遮挡", "label_occluded": "文字遮挡",
              "fully_occluded": "检查相机中完全遮挡"}
    cameras = [(LEFT_CAMERA, "左侧固定相机"), ("robot0_eye_in_hand", "腕部相机")]
    canvas = Image.new("RGB", (640 * len(scenarios), 530 * len(cameras)), (245, 247, 249))
    draw = ImageDraw.Draw(canvas)
    for row, (camera, title) in enumerate(cameras):
        for column, scenario in enumerate(scenarios):
            source = Image.open(output / scenario / f"observation/{camera}_rgb.png").convert("RGB")
            source.thumbnail((640, 480))
            x, y = column * 640, row * 530
            canvas.paste(source, (x + (640 - source.width) // 2, y + 50))
            draw.text((x + 15, y + 10), f"{title} · {titles[scenario]}", font=font, fill=(30, 40, 50))
    canvas.save(output / "camera_comparison.png")


def gallery(output, scenarios, font_path=None):
    font, _ = chinese_font(font_path, 26)
    titles = {"clear": "无遮挡", "label_occluded": "身份文字被遮挡", "fully_occluded": "目标完全被遮挡"}
    panels = []
    for scenario in scenarios:
        source = Image.open(output / scenario / "observation/inspection_rgb.png").convert("RGB")
        source.thumbnail((640, 480))
        panel = Image.new("RGB", (640, 530), (245, 247, 249))
        panel.paste(source, ((640 - source.width) // 2, 50))
        ImageDraw.Draw(panel).text((24, 10), titles[scenario], font=font, fill=(30, 40, 50))
        panels.append(panel)
    canvas = Image.new("RGB", (640 * len(panels), 530))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * 640, 0))
    canvas.save(output / "gallery.png")


def versions():
    import mujoco, robosuite, robocasa
    result = {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__, "robocasa": robocasa.__version__}
    for name, module in [("robosuite", robosuite), ("robocasa", robocasa)]:
        repo = Path(module.__file__).resolve().parent.parent
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        result[f"{name}_commit"] = p.stdout.strip() if p.returncode == 0 else None
    return result


def main():
    parser = argparse.ArgumentParser(description="RoboCasa365 Chinese-label occlusion scene (simulation only)")
    parser.add_argument("command", choices=["assets", "render", "view"])
    parser.add_argument("--output-dir", type=Path,
                        help="Default: runs/robocasa_graspable (legacy_wide: runs/robocasa_occlusion)")
    parser.add_argument("--object-profile", choices=OBJECT_PROFILES, default=DEFAULT_OBJECT_PROFILE,
                        help="graspable: top-grasp width <= 6 cm; legacy_wide: original large boxes")
    parser.add_argument("--scenario", choices=["clear", "label_occluded", "fully_occluded"], default="label_occluded")
    parser.add_argument("--all", action="store_true", help="Render the three matched initial conditions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=14)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--font")
    parser.add_argument("--camera", choices=CAMERA_NAMES,
                        default="inspection", help="Initial camera for the interactive viewer")
    parser.add_argument("--left-camera-position", type=float, nargs=3, default=[-.25, .45, .89],
                        metavar=("X", "Y", "Z"), help="Optical-center position in arm-base frame, metres")
    parser.add_argument("--left-camera-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Look-at point in arm-base frame, metres; default: static work-area center")
    parser.add_argument("--left-camera-fovy", type=float, default=60.0,
                        help="Vertical field of view in degrees; approximate until real intrinsics are supplied")
    parser.add_argument("--view-seconds", type=float, default=0,
                        help="Close viewer after this many seconds; 0 keeps it open")
    args = parser.parse_args()
    output = (args.output_dir or Path("runs/robocasa_graspable" if args.object_profile == "graspable"
                                    else "runs/robocasa_occlusion")).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        old_profile = json.loads(manifest_path.read_text()).get("object_profile", "legacy_wide")
        if old_profile != args.object_profile:
            raise ValueError("use a separate output directory when changing object profile")
    assets = generate_assets(output / "assets", args.font, args.object_profile)
    if args.command == "assets":
        print(output / "assets/assets.json")
        return
    env = make_env(args, assets)
    try:
        env.reset()
        env.settle()
        if args.command == "view":
            import time
            import queue
            import mujoco.viewer
            keys = queue.SimpleQueue()
            cameras = list(CAMERA_NAMES)
            camera_index = cameras.index(args.camera)
            print("Viewer: 1 clear, 2 label occluded, 3 fully occluded; C cycle cameras, F free camera.", flush=True)
            print("Changing scenarios initializes a new condition; it is not a robot clearing action.", flush=True)
            with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data,
                                              key_callback=keys.put) as viewer:
                viewer.cam.fixedcamid = env.sim.model.camera_name2id(args.camera)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer_start = time.monotonic()
                while viewer.is_running():
                    if args.view_seconds > 0 and time.monotonic() - viewer_start >= args.view_seconds:
                        break
                    start = time.monotonic()
                    with viewer.lock():
                        while not keys.empty():
                            key = keys.get()
                            if key in (49, 50, 51):
                                env.set_scenario(["clear", "label_occluded", "fully_occluded"][key - 49], settle=False)
                            elif key == 67:
                                camera_index = (camera_index + 1) % len(cameras)
                                viewer.cam.fixedcamid = env.sim.model.camera_name2id(cameras[camera_index])
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            elif key == 70:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                                viewer.cam.lookat[:] = env.world([0, 0, .03])
                                viewer.cam.distance = 1.5
                                viewer.cam.elevation = -50
                                viewer.cam.azimuth = np.degrees(env.counter.rot) - 90
                        env.step(np.zeros(env.action_spec[0].shape))
                    viewer.sync()
                    time.sleep(max(0, 1 / env.control_freq - (time.monotonic() - start)))
            return
        scenarios = ["clear", "label_occluded", "fully_occluded"] if args.all else [args.scenario]
        # Every case starts from the same robot/fixture state, not the previous case.
        initial_state = env.sim.get_state()
        results = {}
        for scenario in scenarios:
            env.sim.set_state(initial_state)
            env.set_scenario(scenario)
            results[scenario] = capture_scene(env, output / scenario, args.width, args.height)
            print(scenario, json.dumps(results[scenario]["inspection"]), flush=True)
        gallery(output, scenarios, args.font)
        camera_comparison(output, scenarios, args.font)
        from .rendering import renderer_info
        write_json(output / "manifest.json", {
            "versions": versions(), "scene_seed": args.seed, "layout": args.layout, "style": args.style,
            "object_profile": args.object_profile, "top_grasp_size_screen": top_grasp_size_screen(assets),
            "development_scope": {
                "planned_actions": ["observe", "pick_and_place"],
                "pushing_in_scope": False, "agent_loop_implemented": False,
            },
            "rendering": renderer_info(),
            "camera_names": list(CAMERA_NAMES), "left_camera_deployment": env.left_camera_setup,
            "counter": env.counter.name, "scenario_generation": "independent initialized states; not a robot clearing demo",
            "autonomous_clearing_implemented": False, "robot_grasp_verified": False,
            "font": str(chinese_font(args.font)[1]), "results": results,
        })
        print(f"Gallery: {output / 'gallery.png'}")
        print(f"External / wrist views: {output / 'camera_comparison.png'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
