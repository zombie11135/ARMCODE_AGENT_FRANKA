import time
from pprint import pprint
import argparse
import subprocess

from core.context import RuntimeContext
from services.realsense_dual_camera_service import (
    CameraConfig,
    RealSenseDualCameraService,
)
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.ocr_service import PaddleOCRService
from services.sam3_bbox_segmenter_service import SAM3BBoxSegmenterService
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill


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
    parser.add_argument("--query", default="布洛芬")
    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--sam3_model_dir", default="/home/wpy/models/sam3")
    parser.add_argument("--box_threshold", type=float, default=0.22)
    parser.add_argument("--text_threshold", type=float, default=0.18)
    parser.add_argument("--min_score", type=float, default=0.50)
    parser.add_argument("--use_sam_for_selection", action="store_true")
    parser.add_argument("--plan_grasp", action="store_true")
    parser.add_argument("--execute_grasp", action="store_true")
    parser.add_argument(
        "--anygrasp_dir",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
    )
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--grasp_selection_mode",
        default="random_top_k_good",
        choices=["random_good", "random_top_k_good", "best_good"],
    )
    parser.add_argument("--random_select_top_k", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--fallback_to_raw_when_no_good", action="store_true")
    parser.add_argument("--visualize_grasps", action="store_true")
    parser.add_argument("--visualize_top_k", type=int, default=20)
    parser.add_argument("--visualize_best_only", action="store_true")
    parser.add_argument("--visualize_selected_grasp", action="store_true", default=True)
    parser.add_argument("--no_visualize_selected_grasp", action="store_false", dest="visualize_selected_grasp")
    parser.add_argument("--visualize_bbox_filtered_grasps", action="store_true")
    parser.add_argument("--visualize_bbox_filtered_top_k", type=int, default=100)
    parser.add_argument("--verbose_events", action="store_true")
    parser.add_argument("--lims_margin", type=float, default=0.03)
    parser.add_argument("--bbox_margin_px", type=int, default=16)
    parser.add_argument("--infer_grasps_on_full_cloud", action="store_true", default=True)
    parser.add_argument("--use_bbox_roi_points", action="store_false", dest="infer_grasps_on_full_cloud")
    parser.add_argument("--filter_grasps_by_bbox", action="store_true", default=True)
    parser.add_argument("--no_filter_grasps_by_bbox", action="store_false", dest="filter_grasps_by_bbox")
    parser.add_argument("--bbox_filter_margin_px", type=int, default=0)
    parser.add_argument("--bbox_filter_inner_margin_ratio", type=float, default=0.20)
    parser.add_argument("--no_verify_ocr", action="store_true")
    parser.add_argument("--ocr_verify_strict_mode", action="store_true")
    parser.add_argument("--ocr_verify_fuzzy_threshold", type=float, default=0.72)
    parser.add_argument("--fallback_to_roi_points_when_bbox_empty", action="store_true", default=True)
    parser.add_argument("--no_fallback_to_roi_points_when_bbox_empty", action="store_false", dest="fallback_to_roi_points_when_bbox_empty")
    parser.add_argument("--min_mask_points", type=int, default=200)
    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--gripper_close_wait_s", type=float, default=1.5)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    args = parser.parse_args()

    if args.execute_grasp:
        args.plan_grasp = True

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

    segmenter_service = None
    if args.use_sam_for_selection:
        segmenter_service = SAM3BBoxSegmenterService(
            model_dir=args.sam3_model_dir,
            checkpoint_path=None,
            device="cuda",
            confidence_threshold=0.0,
            use_autocast=True,
            enable_inst_interactivity=True,
            compile_model=False,
        )

    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    anygrasp_service = None
    if args.plan_grasp:
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

        if args.execute_grasp:
            arm = init_arm(args.xarm_ip)

        context = RuntimeContext(
            camera=camera_service,
            detector=detector_service,
            segmenter=segmenter_service,
            ocr=ocr_service,
            anygrasp=anygrasp_service,
            event_callback=lambda event: print_event(event, verbose=args.verbose_events),
        )
        context.xarm = arm

        skill = FindAndGraspMedicineSkill(context)

        gripper_open = None
        gripper_close = None
        if args.execute_grasp and not args.disable_grip:
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

        result = skill.run(
            query=args.query,
            camera_name="wrist",
            min_score=args.min_score,
            detection_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            use_sam_for_selection=args.use_sam_for_selection,
            use_mask_for_grasp=False,
            plan_grasp=args.plan_grasp,
            execute=args.execute_grasp,
            xarm_ip=args.xarm_ip,
            move_speed=args.move_speed,
            move_acc=args.move_acc,
            gripper_open=gripper_open,
            gripper_close=gripper_close,
            gripper_close_wait_s=args.gripper_close_wait_s,
            handeye_config_path=args.handeye_config,
            top_k=args.top_k,
            grasp_selection_mode=args.grasp_selection_mode,
            random_select_top_k=args.random_select_top_k,
            random_seed=args.random_seed,
            fallback_to_raw_when_no_good=args.fallback_to_raw_when_no_good,
            visualize_grasps=args.visualize_grasps,
            visualize_top_k=args.visualize_top_k,
            visualize_best_only=args.visualize_best_only,
            visualize_selected_grasp=args.visualize_selected_grasp,
            visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
            visualize_bbox_filtered_top_k=args.visualize_bbox_filtered_top_k,
            lims_margin=args.lims_margin,
            bbox_margin_px=args.bbox_margin_px,
            infer_grasps_on_full_cloud=args.infer_grasps_on_full_cloud,
            filter_grasps_by_bbox=args.filter_grasps_by_bbox,
            bbox_filter_margin_px=args.bbox_filter_margin_px,
            bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
            verify_ocr=not args.no_verify_ocr,
            ocr_verify_strict_mode=args.ocr_verify_strict_mode,
            ocr_verify_fuzzy_threshold=args.ocr_verify_fuzzy_threshold,
            fallback_to_roi_points_when_bbox_empty=args.fallback_to_roi_points_when_bbox_empty,
            min_mask_points=args.min_mask_points,
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
            data = result.data or {}
            if data.get("stage"):
                print("stage:", data.get("stage"))
            if data.get("grasp_plan", {}).get("choose_reason"):
                print("choose_reason:", data["grasp_plan"].get("choose_reason"))
            return

        data = result.data
        selected = data["selected_instance"]

        if args.plan_grasp:
            print("\n=== Grasp Plan ===")
            grasp_plan = data.get("grasp_plan", {})
            print("roi_source:", grasp_plan.get("roi_source"))
            print("num_candidates:", grasp_plan.get("num_candidates"))
            print("num_bbox_filtered_grasps:", grasp_plan.get("num_bbox_filtered_grasps"))
            print("bbox_filter_inner_margin_ratio:", grasp_plan.get("bbox_filter_inner_margin_ratio"))
            print("fallback_to_roi_points_used:", grasp_plan.get("fallback_to_roi_points_used"))
            print("lims:", grasp_plan.get("lims"))
            print("best_grasp_xarm_mmrad:")
            pprint(grasp_plan.get("best_grasp_xarm_mmrad"))
            print("approach_pose_mmrad:")
            pprint(grasp_plan.get("approach_pose_mmrad"))
            print("lift_pose_mmrad:")
            pprint(grasp_plan.get("lift_pose_mmrad"))
            print("choose_reason:", grasp_plan.get("choose_reason"))
            if grasp_plan.get("selected_visualization_error"):
                print("selected_visualization_error:", grasp_plan.get("selected_visualization_error"))
            execution = data.get("execution")
            if execution:
                print("execution:", execution)
        else:
            print("\n=== Selected Medicine Box ===")
            print("object_id:", selected.get("object_id"))
            print("bbox:", selected.get("bbox"))
            print("ocr_text:", selected.get("ocr_text"))
            print("score:", data["selection_result"].get("score"))

    finally:
        camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
