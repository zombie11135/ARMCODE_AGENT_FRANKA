from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.context import RuntimeContext
from nesyro_gate import GateDecision, SkillName, Step, build_task_spec_for_pick_place, nesyro_gate
from skill_lib.simple_actions import SimpleAction, SimpleRobotActions, SimpleRobotConfig


def print_event(event: dict, verbose: bool = False) -> None:
    event_type = event.get("type")
    if not verbose and event_type not in {"warning", "error", "image"}:
        return
    if event_type == "image":
        print(f"[Image] {event.get('content')}: {event.get('image_path')}")
    elif event_type == "text":
        print(f"[Text] {event.get('content')}")
    elif event_type == "warning":
        print(f"[Warning] {event.get('content')}")
    elif event_type == "error":
        print(f"[Error] {event.get('content')}")


def build_context(args):
    from services.grounding_dino_detector_service import GroundingDINODetectorService
    from services.ocr_service import PaddleOCRService
    from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService

    camera = RealSenseDualCameraService(
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
        show_window=not args.no_window,
    )
    detector = GroundingDINODetectorService(
        model_name_or_path=args.grounding_dino_model,
        device=args.detector_device,
        threshold=args.grasp_box_threshold,
        text_threshold=args.grasp_text_threshold,
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )
    ocr = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )
    context = RuntimeContext(
        camera=camera,
        detector=detector,
        ocr=ocr,
        event_callback=lambda event: print_event(event, verbose=args.verbose_events),
    )
    return context, camera


def print_dependency_hint() -> None:
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except Exception as exc:
        print(
            "[NeSyRoGate][WARN] numpy/cv2 不可用，"
            f"基于 depth 的前景遮挡估计会被跳过：{exc}"
        )


def print_gate_result(result, scene, target: str) -> None:
    obj = scene.objects.get(target)
    print("\n=== NeSyRo Gate Observe Test ===")
    print(f"target: {target}")
    print(f"decision: {result.decision.value}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"csc: {result.csc:.3f}")
    print(f"lc: {result.lc:.3f}")
    print(f"reason: {result.reason}")
    print(f"failed_conditions: {result.failed_conditions}")
    print(f"suggested_probe: {result.suggested_probe}")
    print(f"suggested_constraint: {result.suggested_constraint}")

    if obj is None:
        print("scene_object: <missing>")
        return

    print("\n--- SceneMemory Object ---")
    print(f"visible: {obj.visible}")
    print(f"reachable: {obj.reachable}")
    print(f"graspable: {obj.graspable}")
    print(f"occlusion_level: {obj.occlusion_level:.3f}")
    print(f"detector_confidence: {obj.detector_confidence:.3f}")
    print(f"ocr_match_score: {obj.ocr_match_score:.3f}")
    print(f"vlm_match_score: {obj.vlm_match_score:.3f}")
    print(f"ocr_text: {obj.ocr_text}")


def run_vlm_occlusion_if_enabled(args, context, robot, observe_result) -> None:
    if not args.use_vlm_occlusion:
        return

    observation = robot.observations.setdefault(
        args.target,
        {
            "kind": "medicine",
            "target": args.target,
            "observation_failed": True,
            "error": observe_result.error,
            "result": observe_result.data,
        },
    )
    selected = observation.get("selected_instance") or {}
    grasp_input = observation.get("grasp_input") or {}
    result_data = observation.get("result") or {}
    image_path = (
        result_data.get("image_path")
        or grasp_input.get("color_path")
        or selected.get("image_path")
        or _latest_capture_image(args.capture_dir)
    )
    bbox = selected.get("bbox") or grasp_input.get("bbox")

    if not image_path:
        print("[VLM-Occlusion][WARN] no image_path available; skip VLM occlusion check")
        return

    from skill_lib.evaluation.verify_target_occlusion_vlm import VerifyTargetOcclusionVLMSkill

    print(f"\n[VLM-Occlusion] image={image_path}, bbox={bbox}")
    vlm_result = VerifyTargetOcclusionVLMSkill(context).run(
        target_object_name=args.target,
        image_path=image_path,
        bbox=bbox,
        base_url=args.vlm_base_url,
        api_key=args.vlm_api_key,
        model_name=args.vlm_model,
        timeout=args.vlm_timeout,
        max_tokens=args.vlm_max_tokens,
        temperature=args.vlm_temperature,
    )
    if not vlm_result.success:
        print("[VLM-Occlusion][WARN] failed:", vlm_result.error)
        observation["vlm_occlusion_error"] = vlm_result.error
        return

    observation["vlm_occlusion"] = vlm_result.data
    if isinstance(observation.get("result"), dict):
        observation["result"]["vlm_occlusion"] = vlm_result.data
    print("[VLM-Occlusion] result:")
    print(json.dumps(vlm_result.data, ensure_ascii=False, indent=2, default=str))


def _latest_capture_image(capture_dir: str) -> str:
    candidates = []
    root = Path(capture_dir)
    for pattern in ["*wrist*color*.png", "*wrist*color*.jpg", "wrist_color.png"]:
        candidates.extend(root.glob(pattern))
    if not candidates:
        return ""
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(latest)


def save_snapshot(args, result, scene) -> None:
    if not args.output_json:
        return
    obj = scene.objects.get(args.target)
    payload = {
        "target": args.target,
        "decision": result.decision.value,
        "confidence": result.confidence,
        "csc": result.csc,
        "lc": result.lc,
        "reason": result.reason,
        "failed_conditions": result.failed_conditions,
        "suggested_probe": result.suggested_probe,
        "suggested_constraint": result.suggested_constraint,
        "scene_object": obj.__dict__ if obj else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved: {path}")


def write_failure_report_if_needed(args, result, scene, observe_result) -> None:
    if result.decision not in {
        GateDecision.REPLAN,
        GateDecision.LOCAL_REPAIR,
        GateDecision.ABORT,
    }:
        return
    if not args.failure_report_path:
        return

    obj = scene.objects.get(args.target)
    report = {
        "success": False,
        "goal": args.goal,
        "failed_action": {"op": "grasp", "target": args.target},
        "failed_action_index": 0,
        "completed_actions": [],
        "medicine_query": args.target,
        "failure_type": _failure_type_from_gate_decision(result.decision),
        "error": result.reason,
        "failed_conditions": list(result.failed_conditions),
        "suggested_constraint": result.suggested_constraint,
        "recommendation": "replan",
        "evidence": {
            "gate": {
                "decision": result.decision.value,
                "confidence": result.confidence,
                "csc": result.csc,
                "lc": result.lc,
                "reason": result.reason,
                "failed_conditions": list(result.failed_conditions),
                "suggested_probe": result.suggested_probe,
                "suggested_constraint": result.suggested_constraint,
            },
            "scene_object": obj.__dict__ if obj else None,
            "observe_success": bool(observe_result.success),
            "observe_error": observe_result.error,
            "observe_data": observe_result.data,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = Path(args.failure_report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[failure] gate failure report saved: {path}")


def _failure_type_from_gate_decision(decision: GateDecision) -> str:
    if decision == GateDecision.LOCAL_REPAIR:
        return "nesyro_gate_local_repair"
    if decision == GateDecision.ABORT:
        return "nesyro_gate_abort"
    return "nesyro_gate_replan"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observe a medicine target and run NeSyRo Gate without grasp execution."
    )
    parser.add_argument("--target", required=True, help="药盒目标，例如 布洛芬 / 蒙脱石散 / 抗病毒口服液")
    parser.add_argument("--plate", default="blue tray", help="任务里的目标盘子名，仅用于 TaskSpec")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max_probe", type=int, default=0, help="单独测试时默认不触发 probe 重试")
    parser.add_argument("--output_json", default="runs/current_plan/nesyro_gate_observe_result.json")
    parser.add_argument("--failure_report_path", default="runs/current_plan/failure_report.json")
    parser.add_argument("--goal", default="找到目标药盒并放入目标容器")

    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--capture_dir", default="runs/current_capture")
    parser.add_argument("--camera_warmup_s", type=float, default=2.0)
    parser.add_argument("--no_window", action="store_true")

    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--detector_device", default="cuda")
    parser.add_argument("--grasp_box_threshold", type=float, default=0.22)
    parser.add_argument("--grasp_text_threshold", type=float, default=0.18)
    parser.add_argument("--grasp_min_score", type=float, default=0.50)
    parser.add_argument("--no_verify_ocr", action="store_true")
    parser.add_argument("--verbose_events", action="store_true")

    parser.add_argument("--use_vlm_occlusion", action="store_true")
    parser.add_argument(
        "--vlm_base_url",
        default=os.getenv(
            "VLM_OCCLUSION_BASE_URL",
            os.getenv("ARM_AGENT_API_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
        ),
    )
    parser.add_argument(
        "--vlm_api_key",
        default=os.getenv(
            "VLM_OCCLUSION_API_KEY",
            os.getenv("ARM_AGENT_API_KEY", os.getenv("PLANNER_API_KEY", "")),
        ),
    )
    parser.add_argument(
        "--vlm_model",
        default=os.getenv(
            "VLM_OCCLUSION_MODEL",
            os.getenv("ARM_AGENT_API_MODEL", os.getenv("PLANNER_API_MODEL", "")),
        ),
    )
    parser.add_argument("--vlm_timeout", type=int, default=120)
    parser.add_argument("--vlm_temperature", type=float, default=0.0)
    parser.add_argument("--vlm_max_tokens", type=int, default=500)
    args = parser.parse_args()

    if args.use_vlm_occlusion and not (args.vlm_base_url and args.vlm_api_key and args.vlm_model):
        parser.error(
            "--use_vlm_occlusion requires VLM config. Pass --vlm_base_url, "
            "--vlm_api_key, --vlm_model, or set VLM_OCCLUSION_* / ARM_AGENT_* / PLANNER_* env vars."
        )

    print_dependency_hint()

    context, camera = build_context(args)
    try:
        camera.start()
        time.sleep(float(args.camera_warmup_s))

        robot = SimpleRobotActions(
            context=context,
            config=SimpleRobotConfig(
                execute=False,
                grasp_min_score=args.grasp_min_score,
                grasp_box_threshold=args.grasp_box_threshold,
                grasp_text_threshold=args.grasp_text_threshold,
                verify_ocr=not args.no_verify_ocr,
                enable_nesyro_gate=False,
                nesyro_gate_trace_path="",
            ),
        )

        print(f"\n[observe] target={args.target}")
        observe_result = robot.observe(args.target)
        if not observe_result.success:
            print("[observe] failed")
            print("error:", observe_result.error)
            print("data:", json.dumps(observe_result.data, ensure_ascii=False, indent=2, default=str))
            robot.observations[args.target] = {
                "kind": "medicine",
                "target": args.target,
                "observation_failed": True,
                "error": observe_result.error,
                "result": observe_result.data,
            }

        run_vlm_occlusion_if_enabled(args, context, robot, observe_result)

        scene = robot.build_scene_memory(current_holding=None)
        task = build_task_spec_for_pick_place(
            target_object_name=args.target,
            target_plate_name=args.plate,
            target_object_id=args.target,
            target_plate_id=args.plate,
        )
        result = nesyro_gate(
            Step(SkillName.GRASP, {"object": args.target}),
            scene,
            task,
            threshold=args.threshold,
            max_probe=args.max_probe,
        )

        print_gate_result(result, scene, args.target)
        save_snapshot(args, result, scene)
        write_failure_report_if_needed(args, result, scene, observe_result)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
