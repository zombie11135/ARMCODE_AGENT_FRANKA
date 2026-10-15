import argparse
import time
from pprint import pprint

from core.context import RuntimeContext
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.ocr_service import PaddleOCRService
from services.realsense_dual_camera_service import (
    CameraConfig,
    RealSenseDualCameraService,
)
from skill_lib.combound.inspect_medicine_candidates import InspectMedicineCandidatesSkill


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--box_threshold", type=float, default=0.26)
    parser.add_argument("--text_threshold", type=float, default=0.22)
    parser.add_argument("--max_candidates", type=int, default=5)
    parser.add_argument("--execute_observation", action="store_true")
    parser.add_argument("--observe_z_mm", type=float, default=400.0)
    parser.add_argument("--observe_wait_s", type=float, default=0.8)
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument("--allow_unknown_text_targets", action="store_true")
    parser.add_argument("--no_deduplicate_confirmed", action="store_true")
    parser.add_argument("--duplicate_iou_threshold", type=float, default=0.35)
    parser.add_argument("--duplicate_containment_threshold", type=float, default=0.80)
    parser.add_argument("--min_prompt_hits", type=int, default=2)
    parser.add_argument("--min_candidate_area_px", type=float, default=12000.0)
    parser.add_argument("--max_candidate_area_ratio", type=float, default=0.55)
    parser.add_argument("--edge_margin_px", type=int, default=8)
    parser.add_argument("--max_close_observations", type=int, default=2)
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

    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    arm = None

    try:
        camera_service.start()
        time.sleep(2.0)

        if args.execute_observation:
            arm = init_arm(args.xarm_ip)

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            ocr=ocr_service,
            xarm=arm,
            event_callback=lambda event: print_event(event, verbose=args.verbose_events),
        )

        result = InspectMedicineCandidatesSkill(context).run(
            camera_name="wrist",
            detection_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_candidates=args.max_candidates,
            execute_observation=args.execute_observation,
            observe_z_mm=args.observe_z_mm,
            observe_wait_s=args.observe_wait_s,
            xarm_ip=args.xarm_ip,
            handeye_config_path=args.handeye_config,
            move_speed=args.move_speed,
            move_acc=args.move_acc,
            allow_unknown_text_targets=args.allow_unknown_text_targets,
            deduplicate_confirmed=not args.no_deduplicate_confirmed,
            duplicate_iou_threshold=args.duplicate_iou_threshold,
            duplicate_containment_threshold=args.duplicate_containment_threshold,
            min_prompt_hits=args.min_prompt_hits,
            min_candidate_area_px=args.min_candidate_area_px,
            max_candidate_area_ratio=args.max_candidate_area_ratio,
            edge_margin_px=args.edge_margin_px,
            max_close_observations=args.max_close_observations,
            fallback_intrinsics={
                "fx": 909.59,
                "fy": 909.80,
                "cx": 658.97,
                "cy": 355.42,
            },
        )

        print("\n=== Inspect Medicine Candidates Result ===")
        print("success:", result.success)
        print("message:", result.message)
        print("error:", result.error)

        data = result.data or {}

        print("\n=== Confirmed Targets ===")
        for target in data.get("confirmed_targets", []):
            print("- object_id:", target.get("object_id"))
            print("  medicine_query:", target.get("medicine_query"))
            print("  bbox:", target.get("bbox"))
            print("  ocr_text:", target.get("ocr_text"))
            print("  confirm_reason:", target.get("confirm_reason"))

        print("\n=== Unknown Candidates ===")
        for candidate in data.get("unknown_candidates", []):
            print("- object_id:", candidate.get("object_id"))
            print("  bbox:", candidate.get("bbox"))
            print("  ocr_text:", candidate.get("ocr_text"))
            print("  reason:", candidate.get("confirm_reason"))
            if candidate.get("warnings"):
                print("  warnings:")
                pprint(candidate.get("warnings"))

        print("\n=== Duplicate Groups ===")
        for group in data.get("duplicate_groups", []):
            pprint(group)

        print("\n=== Rejected Detections ===")
        for det in data.get("rejected_detections", []):
            print("- object_id:", det.get("object_id"))
            print("  bbox:", det.get("bbox"))
            print("  reason:", det.get("reject_reason"))
            print("  prompt_hits:", det.get("prompt_hits"))

    finally:
        camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
