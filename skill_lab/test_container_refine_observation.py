import argparse
import time
from pprint import pprint

from core.context import RuntimeContext
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
from skill_lib.combound.place_to_container import PlaceToContainerSkill


DEFAULT_INTRINSICS = {
    "fx": 909.59,
    "fy": 909.80,
    "cx": 658.97,
    "cy": 355.42,
}


def print_event(event: dict, verbose: bool = False):
    event_type = event.get("type")
    if not verbose and event_type not in ["warning", "error"]:
        return

    if event_type == "image":
        print(f"[Image] {event.get('content')}: {event.get('image_path')}")
    elif event_type == "text":
        print(f"[Text] {event.get('content')}")
    elif event_type == "warning":
        print(f"[Warning] {event.get('content')}")
    elif event_type == "error":
        print(f"[Error] {event.get('content')}")
    else:
        print(event)


def check_code(arm, code, label: str = "") -> bool:
    if code != 0:
        print(f"[xArm][ERROR] {label} failed, code={code}")
        print(f"[xArm] state={arm.state}, error={arm.error_code}, warn={arm.warn_code}")
        return False
    return True


def init_arm(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    time.sleep(0.5)
    print("[xArm] connected:", arm.connected)
    print("[xArm] state:", arm.state)
    print("[xArm] error:", arm.error_code)
    print("[xArm] warn:", arm.warn_code)

    arm.clean_warn()
    arm.clean_error()
    check_code(arm, arm.motion_enable(enable=True), "motion_enable")
    check_code(arm, arm.set_mode(0), "set_mode")
    check_code(arm, arm.set_state(0), "set_state")
    time.sleep(0.5)
    return arm


def move_home(arm, home_pose, speed: float, acc: float):
    print("[xArm] move home:", home_pose)
    code = arm.set_position(
        x=home_pose[0],
        y=home_pose[1],
        z=home_pose[2],
        roll=home_pose[3],
        pitch=home_pose[4],
        yaw=home_pose[5],
        speed=speed,
        mvacc=acc,
        wait=True,
        is_radian=True,
    )
    return check_code(arm, code, "move_home")


def summarize_observation(title: str, data: dict):
    if not data:
        print(f"\n=== {title} ===")
        print("<empty>")
        return

    selected = data.get("selected_detection") or {}
    print(f"\n=== {title} ===")
    print("image_path:", data.get("image_path"))
    print("depth_path:", data.get("depth_path"))
    print("object_id:", selected.get("object_id"))
    print("bbox:", selected.get("bbox"))
    print("score:", selected.get("score"))
    print("phrase:", selected.get("phrase"))
    print("source_prompt:", selected.get("source_prompt"))
    print("prompt_hits:", selected.get("prompt_hits"))
    print("color_score:", selected.get("color_score"))
    print("center_pixel:", data.get("center_pixel"))
    print("center_depth_m:", data.get("center_depth_m"))
    print("depth_estimation:")
    pprint(data.get("depth_estimation"))
    print("center_base_xyz_m:")
    pprint(data.get("center_base_xyz"))
    print("place_pose_mmrad:")
    pprint(data.get("place_pose_mmrad"))


def main():
    parser = argparse.ArgumentParser(
        description="只测试容器第一次观测和移动到容器上方后的二次观测。"
    )
    parser.add_argument("--container_name", default="tray")
    parser.add_argument("--container_color", default="pink")
    parser.add_argument("--execute_observation", action="store_true")
    parser.add_argument("--home_first", action="store_true")
    parser.add_argument("--home_pose", type=float, nargs=6, default=[345.0, 0.0, 450.0, 3.14, 0.0, -1.57])
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--capture_dir", default="runs/current_capture")
    parser.add_argument("--camera_warmup_s", type=float, default=2.0)
    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--detector_device", default="cuda")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--detection_index", type=int, default=0)
    parser.add_argument("--depth_window_px", type=int, default=25)
    parser.add_argument("--center_offset_x", type=float, default=0.0)
    parser.add_argument("--center_offset_y", type=float, default=0.0)
    parser.add_argument("--refine_observe_z_mm", type=float, default=450.0)
    parser.add_argument("--refine_wait_s", type=float, default=0.8)
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument("--verbose_events", action="store_true")
    args = parser.parse_args()

    camera_service = RealSenseDualCameraService(
        global_config=CameraConfig(
            name="global",
            serial=args.global_serial,
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial=args.wrist_serial,
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=True,
            depth_width=1280,
            depth_height=720,
            depth_fps=30,
        ),
        save_dir=args.capture_dir,
        show_window=True,
    )

    detector_service = GroundingDINODetectorService(
        model_name_or_path=args.grounding_dino_model,
        device=args.detector_device,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    arm = None
    try:
        camera_service.start()
        time.sleep(args.camera_warmup_s)

        if args.execute_observation:
            arm = init_arm(args.xarm_ip)
            if args.home_first and not move_home(arm, args.home_pose, args.move_speed, args.move_acc):
                return

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            xarm=arm,
            event_callback=lambda event: print_event(event, verbose=args.verbose_events),
        )

        skill = PlaceToContainerSkill(context)
        result = skill.run(
            container_name=args.container_name,
            container_color=args.container_color,
            camera_name="wrist",
            detection_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            detection_index=args.detection_index,
            center_offset_ratio=(args.center_offset_x, args.center_offset_y),
            depth_window_px=args.depth_window_px,
            xarm_ip=args.xarm_ip,
            handeye_config_path=args.handeye_config,
            refine_observation=args.execute_observation,
            refine_only_when_execute=False,
            refine_observe_z_mm=args.refine_observe_z_mm,
            refine_wait_s=args.refine_wait_s,
            execute=False,
            move_speed=args.move_speed,
            move_acc=args.move_acc,
            fallback_intrinsics=DEFAULT_INTRINSICS,
        )

        print("\n=== Container Refine Observation Result ===")
        print("success:", result.success)
        print("message:", result.message)
        print("error:", result.error)

        data = result.data or {}
        summarize_observation("First Observation", data.get("first_observation") or data)

        if data.get("refine_observe_pose_mmrad"):
            print("\nrefine_observe_pose_mmrad:")
            pprint(data.get("refine_observe_pose_mmrad"))

        summarize_observation("Refined Observation", data.get("refined_observation"))

        if result.success:
            print("\n=== Final Selected Place Plan ===")
            print("used_refined_observation:", bool(data.get("refined_observation")))
            print("selected_bbox:", (data.get("selected_detection") or {}).get("bbox"))
            print("center_pixel:", data.get("center_pixel"))
            print("center_depth_m:", data.get("center_depth_m"))
            print("depth_method:", (data.get("depth_estimation") or {}).get("method"))
            print("place_pose_mmrad:")
            pprint(data.get("place_pose_mmrad"))

    finally:
        camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
