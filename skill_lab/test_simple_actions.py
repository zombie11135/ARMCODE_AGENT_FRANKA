import argparse
import subprocess
import time
from pprint import pprint

from core.context import RuntimeContext
from services.anygrasp_service import AnyGraspService
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.ocr_service import PaddleOCRService
from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
from skill_lib.simple_actions import (
    SimpleAction,
    SimpleRobotActions,
    SimpleRobotConfig,
    grasp,
    observe,
    place,
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


def parse_action_spec(spec: str) -> SimpleAction:
    text = str(spec or "").strip()
    if ":" not in text:
        raise ValueError(f"动作格式应为 op:target，例如 grasp:布洛芬，当前为 {text!r}")
    op, target = text.split(":", 1)
    op = op.strip().lower()
    target = target.strip()
    if op in ["observe", "obs", "观察"]:
        return observe(target)
    if op in ["grasp", "pick", "抓取", "抓"]:
        return grasp(target)
    if op in ["place", "put", "放置", "放"]:
        return place(target)
    raise ValueError(f"未知动作 op: {op}")


def parse_sequence(text: str):
    return [parse_action_spec(item) for item in str(text).split(",") if item.strip()]


def build_context(args, arm):
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
        threshold=min(args.grasp_box_threshold, 0.25),
        text_threshold=min(args.grasp_text_threshold, 0.20),
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    anygrasp_service = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )

    context = RuntimeContext(
        camera=camera_service,
        detector=detector_service,
        ocr=ocr_service,
        anygrasp=anygrasp_service,
        xarm=arm,
        event_callback=lambda event: print_event(event, verbose=args.verbose_events),
    )
    return context, camera_service


def main():
    parser = argparse.ArgumentParser(description="Test concise observe/grasp/place action facade.")
    parser.add_argument(
        "--sequence",
        default="observe:抗病毒口服液,grasp:抗病毒口服液,place:粉色盘子",
        help="逗号分隔动作，例如 observe:布洛芬,grasp:布洛芬,place:粉色盘子",
    )
    parser.add_argument("--execute", action="store_true", help="真实执行机械臂动作；默认只规划/干跑。")
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
    parser.add_argument(
        "--anygrasp_dir",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
    )

    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument("--observe_z_mm", type=float, default=450.0)
    parser.add_argument("--observe_scan_y_offsets_mm", type=float, nargs="*", default=[0.0, -150.0, 150.0])
    parser.add_argument("--place_slot_spacing_mm", type=float, default=35.0)
    parser.add_argument("--disable_place_slot_offsets", action="store_true")
    parser.add_argument("--grasp_box_threshold", type=float, default=0.22)
    parser.add_argument("--grasp_text_threshold", type=float, default=0.18)
    parser.add_argument("--grasp_min_score", type=float, default=0.50)
    parser.add_argument("--grasp_top_k", type=int, default=50)
    parser.add_argument(
        "--grasp_selection_mode",
        default="best_good",
        choices=["best_good", "random_good", "random_top_k_good"],
    )
    parser.add_argument("--grasp_random_select_top_k", type=int, default=5)
    parser.add_argument("--no_verify_ocr", action="store_true")
    parser.add_argument("--visualize_selected_grasp", action="store_true")
    parser.add_argument("--visualize_bbox_filtered_grasps", action="store_true")
    parser.add_argument("--disable_nesyro_gate", action="store_true")
    parser.add_argument("--nesyro_gate_threshold", type=float, default=0.7)
    parser.add_argument("--nesyro_gate_max_probe", type=int, default=3)
    parser.add_argument("--nesyro_gate_trace_path", default="runs/current_plan/gate_trace.jsonl")
    parser.add_argument("--enable_vlm_occlusion_gate", action="store_true")
    parser.add_argument("--vlm_occlusion_base_url", default="")
    parser.add_argument("--vlm_occlusion_api_key", default="")
    parser.add_argument("--vlm_occlusion_model", default="")
    parser.add_argument("--vlm_occlusion_timeout", type=int, default=120)
    parser.add_argument("--vlm_occlusion_max_tokens", type=int, default=500)
    parser.add_argument("--vlm_occlusion_temperature", type=float, default=0.0)

    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    parser.add_argument("--verbose_events", action="store_true")
    args = parser.parse_args()

    actions = parse_sequence(args.sequence)
    print("\n=== Simple Action Sequence ===")
    for index, action in enumerate(actions, start=1):
        print(f"{index}. {action.op}({action.target})")

    arm = None
    context = None
    camera_service = None
    try:
        if args.execute:
            arm = init_arm(args.xarm_ip)

        context, camera_service = build_context(args, arm)
        camera_service.start()
        time.sleep(args.camera_warmup_s)

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

        robot = SimpleRobotActions(
            context=context,
            config=SimpleRobotConfig(
                execute=args.execute,
                xarm_ip=args.xarm_ip,
                handeye_config_path=args.handeye_config,
                move_speed=args.move_speed,
                move_acc=args.move_acc,
                observe_z_mm=args.observe_z_mm,
                observe_scan_y_offsets_mm=list(args.observe_scan_y_offsets_mm),
                place_slot_spacing_mm=args.place_slot_spacing_mm,
                enable_place_slot_offsets=not args.disable_place_slot_offsets,
                grasp_min_score=args.grasp_min_score,
                grasp_box_threshold=args.grasp_box_threshold,
                grasp_text_threshold=args.grasp_text_threshold,
                grasp_top_k=args.grasp_top_k,
                grasp_selection_mode=args.grasp_selection_mode,
                grasp_random_select_top_k=args.grasp_random_select_top_k,
                verify_ocr=not args.no_verify_ocr,
                visualize_selected_grasp=args.visualize_selected_grasp,
                visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
                enable_nesyro_gate=not args.disable_nesyro_gate,
                nesyro_gate_threshold=args.nesyro_gate_threshold,
                nesyro_gate_max_probe=args.nesyro_gate_max_probe,
                nesyro_gate_trace_path=args.nesyro_gate_trace_path,
                enable_vlm_occlusion_gate=args.enable_vlm_occlusion_gate,
                vlm_occlusion_base_url=args.vlm_occlusion_base_url,
                vlm_occlusion_api_key=args.vlm_occlusion_api_key,
                vlm_occlusion_model=args.vlm_occlusion_model,
                vlm_occlusion_timeout=args.vlm_occlusion_timeout,
                vlm_occlusion_max_tokens=args.vlm_occlusion_max_tokens,
                vlm_occlusion_temperature=args.vlm_occlusion_temperature,
            ),
            gripper_open=gripper_open,
            gripper_close=gripper_close,
        )

        result = robot.run(actions)
        print("\n=== Simple Actions Result ===")
        print("success:", result.success)
        print("message:", result.message)
        print("error:", result.error)
        print("cached_place_plans:", list(robot.place_plans.keys()))
        print("observations:", list(robot.observations.keys()))
        print("data:")
        pprint(result.data)

    finally:
        if camera_service is not None:
            camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
