import argparse
import subprocess
import time
from pprint import pprint

from core.context import RuntimeContext
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.ocr_service import PaddleOCRService
from services.realsense_dual_camera_service import (
    CameraConfig,
    RealSenseDualCameraService,
)
from skill_lib.combound.pick_medicine_and_place_to_container import (
    PickMedicineAndPlaceToContainerSkill,
)


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
    parser.add_argument("--medicine_query", default="布洛芬")
    parser.add_argument("--container_name", default="tray")
    parser.add_argument("--container_color", default=None)
    parser.add_argument("--execute", action="store_true")

    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument(
        "--anygrasp_dir",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
    )

    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument(
        "--pre_grasp_pose",
        type=float,
        nargs=6,
        default=[350, 0, 450, 3.14, 0.0, -1.57],
        metavar=("x", "y", "z", "rx", "ry", "rz"),
    )
    parser.add_argument("--no_pre_grasp_pose", action="store_true")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)

    parser.add_argument("--container_box_threshold", type=float, default=0.25)
    parser.add_argument("--container_text_threshold", type=float, default=0.20)
    parser.add_argument("--container_detection_index", type=int, default=0)
    parser.add_argument("--container_depth_window_px", type=int, default=25)
    parser.add_argument("--no_container_refine", action="store_true")
    parser.add_argument("--container_refine_observe_z_mm", type=float, default=400.0)
    parser.add_argument("--container_refine_wait_s", type=float, default=0.8)
    parser.add_argument("--place_clearance_mm", type=float, default=250.0)

    parser.add_argument("--grasp_box_threshold", type=float, default=0.22)
    parser.add_argument("--grasp_text_threshold", type=float, default=0.18)
    parser.add_argument("--grasp_min_score", type=float, default=0.50)
    parser.add_argument("--grasp_top_k", type=int, default=50)
    parser.add_argument("--grasp_random_select_top_k", type=int, default=5)
    parser.add_argument("--bbox_filter_inner_margin_ratio", type=float, default=0.15)
    parser.add_argument("--no_verify_ocr", action="store_true")
    parser.add_argument("--ocr_verify_strict_mode", action="store_true")
    parser.add_argument("--ocr_verify_fuzzy_threshold", type=float, default=0.72)
    parser.add_argument("--visualize_selected_grasp", action="store_true")
    parser.add_argument("--visualize_bbox_filtered_grasps", action="store_true")

    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--gripper_close_wait_s", type=float, default=1.5)
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
        threshold=min(args.container_box_threshold, args.grasp_box_threshold),
        text_threshold=min(args.container_text_threshold, args.grasp_text_threshold),
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    from services.anygrasp_service import AnyGraspService

    anygrasp_service = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )

    arm = None

    try:
        camera_service.start()
        time.sleep(2.0)

        if args.execute:
            arm = init_arm(args.xarm_ip)

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            ocr=ocr_service,
            anygrasp=anygrasp_service,
            event_callback=lambda event: print_event(event, verbose=args.verbose_events),
        )
        context.xarm = arm

        gripper_open = None
        gripper_close = None
        if args.execute and not args.disable_grip:
            gripper_open = lambda: call_grip_service(
                value=args.grip_open_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
            )
            gripper_close = lambda: call_grip_service(
                value=args.grip_close_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
            )

        skill = PickMedicineAndPlaceToContainerSkill(context)
        result = skill.run(
            medicine_query=args.medicine_query,
            container_name=args.container_name,
            container_color=args.container_color,
            execute=args.execute,
            xarm_ip=args.xarm_ip,
            handeye_config_path=args.handeye_config,
            pre_grasp_pose_mmrad=None if args.no_pre_grasp_pose else args.pre_grasp_pose,
            move_speed=args.move_speed,
            move_acc=args.move_acc,
            gripper_open=gripper_open,
            gripper_close=gripper_close,
            gripper_close_wait_s=args.gripper_close_wait_s,
            container_detection_threshold=args.container_box_threshold,
            container_text_threshold=args.container_text_threshold,
            container_detection_index=args.container_detection_index,
            container_depth_window_px=args.container_depth_window_px,
            container_refine_observation=not args.no_container_refine,
            container_refine_observe_z_mm=args.container_refine_observe_z_mm,
            container_refine_wait_s=args.container_refine_wait_s,
            place_clearance_mm=args.place_clearance_mm,
            grasp_detection_threshold=args.grasp_box_threshold,
            grasp_text_threshold=args.grasp_text_threshold,
            grasp_min_score=args.grasp_min_score,
            grasp_top_k=args.grasp_top_k,
            grasp_random_select_top_k=args.grasp_random_select_top_k,
            verify_ocr=not args.no_verify_ocr,
            ocr_verify_strict_mode=args.ocr_verify_strict_mode,
            ocr_verify_fuzzy_threshold=args.ocr_verify_fuzzy_threshold,
            bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
            visualize_selected_grasp=args.visualize_selected_grasp,
            visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
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
            print("stage:", result.data.get("stage"))
            return

        data = result.data
        place_plan = data["place_plan"]
        grasp_data = data["grasp_result"]
        grasp_plan = grasp_data.get("grasp_plan", {})

        print("\n=== Pick And Place Result ===")
        print("container_bbox:", place_plan["selected_detection"].get("bbox"))
        print("container_center_base_xyz:")
        pprint(place_plan.get("center_base_xyz"))
        print("cached_place_pose_mmrad:")
        pprint(place_plan.get("place_pose_mmrad"))
        print("grasp_pose_mmrad:")
        pprint(grasp_plan.get("best_grasp_xarm_mmrad"))
        print("grasp_choose_reason:", grasp_plan.get("choose_reason"))
        print("place_execution:", data.get("place_execution"))

    finally:
        camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
