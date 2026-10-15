import argparse
import subprocess
import time
from pprint import pprint

from core.context import RuntimeContext
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.realsense_dual_camera_service import (
    CameraConfig,
    RealSenseDualCameraService,
)
from skill_lib.combound.place_to_container import PlaceToContainerSkill


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


def call_grip_service(
    value: int,
    ros_ws: str,
    service_name: str,
    timeout: float,
    ignore_error: bool,
) -> bool:
    cmd = f"source {ros_ws}/devel/setup.bash && rosservice call {service_name} {int(value)}"
    print(f"[Gripper] {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

        if result.stdout.strip():
            print("[Gripper] output:")
            print(result.stdout)

        if result.returncode != 0:
            print(f"[Gripper][WARN] rosservice returned code={result.returncode}")
            return bool(ignore_error)

        return True

    except subprocess.TimeoutExpired:
        print(f"[Gripper][WARN] rosservice timeout: {service_name}")
        return bool(ignore_error)
    except Exception as e:
        print(f"[Gripper][WARN] rosservice failed: {e}")
        return bool(ignore_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container_name", default="tray")
    parser.add_argument("--container_color", default=None)
    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--detection_index", type=int, default=0)
    parser.add_argument("--depth_window_px", type=int, default=25)
    parser.add_argument("--center_offset_x", type=float, default=0.0)
    parser.add_argument("--center_offset_y", type=float, default=0.0)
    parser.add_argument("--place_clearance_mm", type=float, default=250.0)
    parser.add_argument("--approach_clearance_mm", type=float, default=120.0)
    parser.add_argument("--lift_after_place_mm", type=float, default=120.0)
    parser.add_argument("--refine_observation", action="store_true", default=True)
    parser.add_argument("--no_refine_observation", action="store_false", dest="refine_observation")
    parser.add_argument("--refine_only_when_execute", action="store_true", default=True)
    parser.add_argument("--refine_in_dry_run", action="store_false", dest="refine_only_when_execute")
    parser.add_argument("--refine_observe_z_mm", type=float, default=400.0)
    parser.add_argument("--refine_wait_s", type=float, default=0.8)
    parser.add_argument("--execute_place", action="store_true")
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    parser.add_argument("--verbose_events", action="store_true")
    args = parser.parse_args()

    camera_service = RealSenseDualCameraService(
        global_config=CameraConfig(
            name="global",
            serial="213522251050",
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial="243222070053",
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=True,
            depth_width=1280,
            depth_height=720,
            depth_fps=30,
        ),
        save_dir="runs/current_capture",
        show_window=True,
    )

    detector_service = GroundingDINODetectorService(
        model_name_or_path=args.grounding_dino_model,
        device="cuda",
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    arm = None

    try:
        camera_service.start()
        time.sleep(2.0)

        if args.execute_place:
            arm = init_arm(args.xarm_ip)

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            event_callback=lambda event: print_event(event, verbose=args.verbose_events),
        )
        context.xarm = arm

        gripper_open = None
        if args.execute_place and not args.disable_grip:
            gripper_open = lambda: call_grip_service(
                value=args.grip_open_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
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
            place_clearance_mm=args.place_clearance_mm,
            approach_clearance_mm=args.approach_clearance_mm,
            lift_after_place_mm=args.lift_after_place_mm,
            refine_observation=args.refine_observation,
            refine_only_when_execute=args.refine_only_when_execute,
            refine_observe_z_mm=args.refine_observe_z_mm,
            refine_wait_s=args.refine_wait_s,
            execute=args.execute_place,
            move_speed=args.move_speed,
            move_acc=args.move_acc,
            gripper_open=gripper_open,
            fallback_intrinsics={
                "fx": 909.59,
                "fy": 909.80,
                "cx": 658.97,
                "cy": 355.42,
            },
        )

        if not result.success:
            print("\n=== Failed ===")
            print("error:", result.error)
            return

        data = result.data
        selected = data["selected_detection"]

        print("\n=== Place Plan ===")
        print("selected_object_id:", selected.get("object_id"))
        print("selected_bbox:", selected.get("bbox"))
        print("container_color:", data.get("container_color"))
        print("color_score:", selected.get("color_score"))
        print("center_pixel:", data.get("center_pixel"))
        print("center_depth_m:", data.get("center_depth_m"))
        print("center_base_xyz:")
        pprint(data.get("center_base_xyz"))
        if data.get("refined_observation"):
            print("used_refined_observation: True")
            print("refine_observe_pose_mmrad:")
            pprint(data.get("refine_observe_pose_mmrad"))
        else:
            print("used_refined_observation: False")
        print("approach_pose_mmrad:")
        pprint(data.get("approach_pose_mmrad"))
        print("place_pose_mmrad:")
        pprint(data.get("place_pose_mmrad"))
        print("lift_pose_mmrad:")
        pprint(data.get("lift_pose_mmrad"))

        if data.get("execution"):
            print("execution:", data["execution"])

    finally:
        camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
