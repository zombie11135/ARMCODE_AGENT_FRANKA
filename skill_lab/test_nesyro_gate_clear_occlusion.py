from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.context import RuntimeContext
from core.skill_base import SkillResult
from nesyro_gate import GateDecision, SkillName, Step, build_task_spec_for_pick_place, nesyro_gate
from skill_lib.simple_actions import FALLBACK_INTRINSICS, SimpleRobotActions, SimpleRobotConfig


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


def build_context(args, arm=None):
    from services.anygrasp_service import AnyGraspService
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
    ocr = PaddleOCRService(lang="ch", device="cpu", use_textline_orientation=True)
    anygrasp = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )
    context = RuntimeContext(
        camera=camera,
        detector=detector,
        ocr=ocr,
        anygrasp=anygrasp,
        xarm=arm,
        event_callback=lambda event: print_event(event, verbose=args.verbose_events),
    )
    return context, camera


def init_arm(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    time.sleep(0.5)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
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
            print(result.stdout)
        if result.returncode != 0:
            print(f"[Gripper][WARN] rosservice returned code={result.returncode}")
            return bool(ignore_error)
        return True
    except Exception as e:
        print(f"[Gripper][WARN] rosservice failed: {e}")
        return bool(ignore_error)


def run_vlm_occlusion(args, context, robot, observe_result) -> None:
    result_payload = observe_result.data if isinstance(observe_result.data, dict) else {}
    observation = robot.observations.setdefault(
        args.target,
        {
            "kind": "medicine",
            "target": args.target,
            "observation_failed": True,
            "error": observe_result.error,
            "result": result_payload,
        },
    )
    selected = observation.get("selected_instance") or {}
    grasp_input = observation.get("grasp_input") or {}
    result_data = observation.get("result") or {}
    image_path = (
        result_data.get("image_path")
        or _as_dict(result_data.get("last_failure_data")).get("image_path")
        or grasp_input.get("color_path")
        or selected.get("image_path")
        or latest_capture_file(args.capture_dir, "wrist_color.png")
    )
    bbox = selected.get("bbox") or grasp_input.get("bbox")
    if not image_path:
        raise RuntimeError(
            "No image_path available for VLM occlusion check. "
            "The target may be fully occluded, but no wrist_color.png fallback was found."
        )
    result_data.setdefault("image_path", image_path)

    from skill_lib.evaluation.verify_target_occlusion_vlm import VerifyTargetOcclusionVLMSkill

    print(f"\n[VLM-Occlusion] image={image_path}, bbox={bbox}")
    result = VerifyTargetOcclusionVLMSkill(context).run(
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
    if not result.success:
        raise RuntimeError(result.error)
    observation["vlm_occlusion"] = result.data
    if isinstance(observation.get("result"), dict):
        observation["result"]["vlm_occlusion"] = result.data
    print(json.dumps(result.data, ensure_ascii=False, indent=2, default=str))


def run_gate(args, robot):
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
        max_probe=0,
    )
    obj = scene.objects.get(args.target)
    print("\n=== Gate ===")
    print(f"decision: {result.decision.value}")
    print(f"confidence: {result.confidence:.3f}, csc={result.csc:.3f}, lc={result.lc:.3f}")
    print(f"failed_conditions: {result.failed_conditions}")
    print(f"suggested_constraint: {result.suggested_constraint}")
    if obj:
        print(f"scene: visible={obj.visible}, reachable={obj.reachable}, graspable={obj.graspable}, occlusion={obj.occlusion_level:.3f}")
    return result, scene


def add_detector_blocker_candidates(args, context, observation: Dict[str, Any]) -> None:
    result_data = observation.setdefault("result", {})
    image_path = (
        result_data.get("image_path")
        or _as_dict(result_data.get("last_failure_data")).get("image_path")
        or latest_capture_file(args.capture_dir, "wrist_color.png")
    )
    if not image_path:
        print("[blocker][WARN] no image available for blocker detector candidates")
        return
    try:
        detector = context.require("detector")
        merged: List[Dict[str, Any]] = []
        for prompt in args.blocker_detector_prompts:
            detected = detector.detect(
                image_path=image_path,
                target_name=prompt,
                threshold=float(args.blocker_detector_threshold),
                text_threshold=float(args.blocker_detector_text_threshold),
                max_objects=int(args.blocker_detector_max_objects),
                enable_filter=False,
            )
            for det in detected.get("detections", []) or []:
                det = dict(det)
                det["source"] = f"blocker_detector:{prompt}"
                merged.append(det)
        merged = dedupe_candidates_by_bbox(merged)
        result_data["blocker_detections"] = merged
        print(f"[blocker] detector candidates: {len(merged)}")
    except Exception as e:
        print(f"[blocker][WARN] detector candidate search failed: {e}")


def select_blocker_candidate(args, observation: Dict[str, Any], manual_bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    if manual_bbox:
        return {"object_id": "manual_blocker", "bbox": manual_bbox, "source": "manual"}

    result_data = observation.get("result") or {}
    selected = observation.get("selected_instance") or result_data.get("selected_instance") or {}
    target_bbox = selected.get("bbox")

    candidates = collect_bbox_candidates(result_data)
    selected_id = str(selected.get("object_id") or "")
    depth = load_depth_for_blocker_selection(args, observation)

    if not target_bbox:
        if not candidates:
            raise RuntimeError(
                "Cannot select blocker automatically: target has no bbox and no detector "
                "candidates were preserved. Use --blocker_bbox x1 y1 x2 y2."
            )
        blocker = select_largest_nearby_candidate(candidates, depth)
        blocker["source"] = f"{blocker.get('source', 'candidate')}:fallback_no_target_bbox"
        return blocker

    target_area = max(1.0, _bbox_area(target_bbox))
    expanded_target = expand_bbox(target_bbox, float(args.blocker_search_expand_ratio))
    target_depth = median_depth_in_bbox(depth, target_bbox)
    scored = []
    for cand in candidates:
        bbox = cand.get("bbox")
        if not bbox:
            continue
        if selected_id and str(cand.get("object_id") or "") == selected_id:
            continue
        cand_area = max(1.0, _bbox_area(bbox))
        if cand_area < target_area * float(args.blocker_min_area_ratio):
            continue
        inter = _bbox_intersection_area(target_bbox, bbox)
        expanded_inter = _bbox_intersection_area(expanded_target, bbox)
        if inter <= 0 and expanded_inter <= 0:
            continue
        cand_area = max(1.0, _bbox_area(bbox))
        iou = inter / max(1.0, target_area + cand_area - inter)
        overlap_target = inter / target_area
        overlap_candidate_in_expanded = expanded_inter / cand_area
        if iou >= 0.85:
            continue
        if overlap_target <= 0.02 and overlap_candidate_in_expanded <= 0.20:
            continue
        cand_depth = median_depth_in_bbox(depth, bbox)
        depth_bonus = 0.0
        if target_depth is not None and cand_depth is not None:
            # RealSense depth is camera distance in meters. Smaller means closer to camera,
            # which is usually the object on top in the wrist top-down view.
            if cand_depth + float(args.blocker_depth_closer_margin_m) < target_depth:
                depth_bonus = 1.0
            else:
                depth_bonus = -0.4
        area_bonus = min(cand_area / target_area, 3.0) * 0.2
        score = (
            1.4 * overlap_target
            + 0.8 * overlap_candidate_in_expanded
            + area_bonus
            + depth_bonus
            - 0.5 * iou
        )
        cand = dict(cand)
        cand["blocker_selection_score"] = round(float(score), 4)
        cand["target_depth_median"] = target_depth
        cand["candidate_depth_median"] = cand_depth
        scored.append((score, cand))

    if not scored:
        raise RuntimeError("No overlapping blocker candidate found. Use --blocker_bbox x1 y1 x2 y2.")
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def select_largest_nearby_candidate(candidates: List[Dict[str, Any]], depth) -> Dict[str, Any]:
    del depth
    if not candidates:
        raise RuntimeError("No blocker candidates available.")
    candidates = [dict(c) for c in candidates if c.get("bbox")]
    candidates.sort(key=lambda cand: _bbox_area(cand.get("bbox")), reverse=True)
    return candidates[0]


def collect_bbox_candidates(result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_sources = [result_data]
    last_failure = _as_dict(result_data.get("last_failure_data"))
    if last_failure:
        data_sources.append(last_failure)
    for attempt in result_data.get("failure_attempts", []) or []:
        attempt_data = _as_dict(_as_dict(attempt).get("data"))
        if attempt_data:
            data_sources.append(attempt_data)

    candidates = []
    seen = set()
    for data in data_sources:
        for source_key in ["blocker_detections", "detections", "instances", "ranked_candidates"]:
            for item in data.get(source_key, []) or []:
                cand = item.get("instance") if isinstance(item, dict) and isinstance(item.get("instance"), dict) else item
                if isinstance(cand, dict) and cand.get("bbox"):
                    cand = dict(cand)
                    cand["source"] = cand.get("source") or source_key
                    key = tuple(round(float(v), 2) for v in cand["bbox"])
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(cand)
    return candidates


def dedupe_candidates_by_bbox(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()
    for cand in candidates:
        if not isinstance(cand, dict) or not cand.get("bbox"):
            continue
        key = tuple(round(float(v), 1) for v in cand["bbox"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def latest_capture_file(capture_dir: str, filename: str) -> Optional[str]:
    path = Path(capture_dir) / filename
    return str(path) if path.exists() else None


def existing_capture_file(capture_dir: str, *filenames: str) -> Optional[str]:
    for filename in filenames:
        path = Path(capture_dir) / filename
        if path.exists():
            return str(path)
    return None


def load_depth_for_blocker_selection(args, observation: Dict[str, Any]):
    grasp_input = _as_dict(observation.get("grasp_input"))
    result_data = _as_dict(observation.get("result"))
    depth_npy_path = (
        grasp_input.get("depth_npy_path")
        or existing_capture_file(args.capture_dir, "wrist_depth.npy", "wrist_depth_meter.npy")
    )
    depth_path = (
        grasp_input.get("depth_path")
        or existing_capture_file(args.capture_dir, "wrist_depth.png")
    )
    depth_scale = (
        grasp_input.get("depth_scale")
        or result_data.get("depth_scale")
        or 1000.0
    )
    try:
        import cv2
        import numpy as np

        if depth_npy_path and Path(str(depth_npy_path)).exists():
            depth = np.load(str(depth_npy_path)).astype("float32")
            if np.nanmax(depth) > 10:
                depth = depth / 1000.0
            return depth
        if depth_path and Path(str(depth_path)).exists():
            raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                return None
            raw = raw.astype("float32")
            depth_scale = float(depth_scale)
            return raw * depth_scale if 0 < depth_scale < 1 else raw / depth_scale
    except Exception as e:
        print(f"[blocker][WARN] depth loading for blocker selection failed: {e}")
    return None


def median_depth_in_bbox(depth, bbox: Any) -> Optional[float]:
    if depth is None or not bbox:
        return None
    try:
        import numpy as np

        h, w = depth.shape[:2]
        x1, y1, x2, y2 = clip_bbox(bbox, w, h)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)]
        if valid.size < 20:
            return None
        return float(np.median(valid))
    except Exception:
        return None


def expand_bbox(bbox: Any, ratio: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ratio = max(1.0, float(ratio))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * 0.5 * ratio
    half_h = (y2 - y1) * 0.5 * ratio
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def clip_bbox(bbox: Any, image_width: int, image_height: int) -> List[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))
    return [x1, y1, x2, y2]


def blocker_bbox_for_grasp(args, blocker: Dict[str, Any]) -> List[float]:
    bbox = [float(v) for v in blocker["bbox"]]
    shrink = max(0.0, min(0.45, float(args.blocker_bbox_shrink_ratio)))
    if shrink <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    dx = (x2 - x1) * shrink
    dy = (y2 - y1) * shrink
    return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]


def build_fallback_grasp_input(args, observation: Dict[str, Any], blocker: Dict[str, Any]) -> Dict[str, Any]:
    result_data = observation.get("result") or {}
    image_path = (
        result_data.get("image_path")
        or _as_dict(result_data.get("last_failure_data")).get("image_path")
        or latest_capture_file(args.capture_dir, "wrist_color.png")
    )
    depth_path = existing_capture_file(args.capture_dir, "wrist_depth.png")
    depth_npy_path = existing_capture_file(args.capture_dir, "wrist_depth.npy", "wrist_depth_meter.npy")
    if not image_path:
        raise RuntimeError("Cannot build blocker grasp input: missing wrist color image.")
    if not depth_path and not depth_npy_path:
        raise RuntimeError("Cannot build blocker grasp input: missing wrist depth image/npy.")
    return {
        "object_id": blocker.get("object_id") or "blocker",
        "target_name": "occluding_object",
        "query": "occluding_object",
        "color_path": image_path,
        "image_path": image_path,
        "depth_path": depth_path,
        "depth_npy_path": depth_npy_path,
        "mask_path": None,
        "mask_npy_path": None,
        "bbox": blocker_bbox_for_grasp(args, blocker),
        "depth_intrinsics": FALLBACK_INTRINSICS,
        "color_intrinsics": FALLBACK_INTRINSICS,
        "camera_intrinsics": FALLBACK_INTRINSICS,
        "depth_scale": 1000.0,
        "ocr_text": None,
        "ocr_texts": None,
    }


def build_blocker_grasp_input(args, observation: Dict[str, Any], blocker: Dict[str, Any]) -> Dict[str, Any]:
    result_data = observation.get("result") or {}
    base = dict(observation.get("grasp_input") or result_data.get("grasp_input") or {})
    if not base:
        return build_fallback_grasp_input(args, observation, blocker)
    base.update(
        {
            "object_id": blocker.get("object_id") or "blocker",
            "target_name": "occluding_object",
            "query": "occluding_object",
            "bbox": blocker_bbox_for_grasp(args, blocker),
            "mask_path": None,
            "mask_npy_path": None,
            "ocr_text": None,
            "ocr_texts": None,
        }
    )
    return base


def plan_blocker_grasp(args, context, grasp_input: Dict[str, Any]) -> SkillResult:
    from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill

    return PlanGraspWithAnyGraspSkill(context).run(
        grasp_input=grasp_input,
        xarm_ip=args.xarm_ip,
        handeye_config_path=args.handeye_config,
        top_k=args.grasp_top_k,
        grasp_selection_mode=args.grasp_selection_mode,
        random_select_top_k=args.grasp_random_select_top_k,
        filter_grasps_by_bbox=True,
        bbox_filter_margin_px=args.bbox_filter_margin_px,
        bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
        collision_detection=not args.disable_anygrasp_collision_detection,
        visualize_selected_grasp=args.visualize_selected_grasp,
        visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
    )


def execute_clear_to_buffer(args, context, plan: Dict[str, Any], gripper_open, gripper_close) -> Dict[str, Any]:
    arm = context.require("xarm")
    if not ensure_xarm_ready(arm, "before clear-to-buffer"):
        return {"success": False, "error": "xArm is not ready before clear-to-buffer"}

    approach = plan.get("approach_pose_mmrad")
    grasp_pose = plan.get("best_grasp_xarm_mmrad")
    lift = plan.get("lift_pose_mmrad")
    for name, pose in [("approach", approach), ("grasp", grasp_pose), ("lift", lift)]:
        if pose is None or len(pose) != 6:
            return {"success": False, "error": f"missing {name} pose"}

    if gripper_open and not gripper_open():
        return {"success": False, "error": "gripper open failed"}
    for label, pose in [("approach", approach), ("grasp", grasp_pose)]:
        if not move_xarm(arm, pose, args.move_speed, args.move_acc):
            return {"success": False, "error": f"move {label} failed"}
    if gripper_close and not gripper_close():
        return {"success": False, "error": "gripper close failed"}
    time.sleep(float(args.gripper_close_wait_s))
    if not move_xarm(arm, lift, args.move_speed, args.move_acc):
        return {"success": False, "error": "lift blocker failed"}

    buffer_pose = list(args.buffer_pose_mmrad)
    buffer_approach = list(buffer_pose)
    buffer_approach[2] += float(args.buffer_approach_clearance_mm)
    for label, pose in [("buffer_approach", buffer_approach), ("buffer", buffer_pose)]:
        if not move_xarm(arm, pose, args.move_speed, args.move_acc):
            return {"success": False, "error": f"move {label} failed"}
    if gripper_open and not gripper_open():
        return {"success": False, "error": "release blocker failed"}
    if not move_xarm(arm, buffer_approach, args.move_speed, args.move_acc):
        return {"success": False, "error": "lift from buffer failed"}
    return {"success": True}


def move_xarm(arm, pose: List[float], speed: float, acc: float) -> bool:
    if not ensure_xarm_ready(arm, "before move"):
        return False
    x, y, z, rx, ry, rz = normalize_pose_mmrad(pose)
    print(f"[xArm] move pose={[x, y, z, rx, ry, rz]}")
    code = arm.set_position(x=x, y=y, z=z, roll=rx, pitch=ry, yaw=rz, speed=speed, mvacc=acc, wait=True)
    if code != 0:
        print(f"[xArm][ERROR] set_position failed code={code}, pose={[x, y, z, rx, ry, rz]}")
        print_xarm_status(arm, "after failed set_position")
        return False
    return True


def ensure_xarm_ready(arm, label: str) -> bool:
    state = getattr(arm, "state", None)
    error = getattr(arm, "error_code", None)
    warn = getattr(arm, "warn_code", None)
    print(f"[xArm] {label}: state={state}, error={error}, warn={warn}")
    if error not in (None, 0):
        print(f"[xArm][ERROR] controller has error_code={error}; clear/reset robot before moving.")
        return False
    if state == 4:
        print("[xArm][ERROR] controller is stopped (state=4); clear error and set_state(0) before moving.")
        return False
    return True


def print_xarm_status(arm, label: str) -> None:
    print(
        f"[xArm] {label}: "
        f"state={getattr(arm, 'state', None)}, "
        f"error={getattr(arm, 'error_code', None)}, "
        f"warn={getattr(arm, 'warn_code', None)}"
    )


def normalize_pose_mmrad(pose: List[float]) -> List[float]:
    values = [float(v) for v in pose]
    values[3] = normalize_angle_rad(values[3])
    values[4] = normalize_angle_rad(values[4])
    values[5] = normalize_angle_rad(values[5])
    return values


def normalize_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_args():
    parser = argparse.ArgumentParser(description="Test NeSyRo Gate occlusion clearing to a fixed buffer pose.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--plate", default="blue tray")
    parser.add_argument("--execute", action="store_true", help="Actually move xArm and gripper. Default is dry-run planning only.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--blocker_bbox", type=float, nargs=4, default=None)
    parser.add_argument("--blocker_bbox_shrink_ratio", type=float, default=0.08)
    parser.add_argument("--blocker_search_expand_ratio", type=float, default=1.8)
    parser.add_argument("--blocker_min_area_ratio", type=float, default=0.50)
    parser.add_argument("--blocker_depth_closer_margin_m", type=float, default=0.01)
    parser.add_argument("--blocker_detector_prompts", nargs="*", default=["medicine_box", "snack", "box", "package", "object on top of another object"])
    parser.add_argument("--blocker_detector_threshold", type=float, default=0.20)
    parser.add_argument("--blocker_detector_text_threshold", type=float, default=0.16)
    parser.add_argument("--blocker_detector_max_objects", type=int, default=12)
    parser.add_argument("--buffer_pose_mmrad", type=float, nargs=6, default=[280.0, -260.0, 170.0, 3.14, 0.0, -1.57])
    parser.add_argument("--buffer_approach_clearance_mm", type=float, default=120.0)

    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--capture_dir", default="runs/current_capture")
    parser.add_argument("--camera_warmup_s", type=float, default=2.0)
    parser.add_argument("--no_window", action="store_true")
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument("--handeye_config", default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)

    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--detector_device", default="cuda")
    parser.add_argument("--anygrasp_dir", default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection")
    parser.add_argument("--anygrasp_ckpt", default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar")
    parser.add_argument("--grasp_box_threshold", type=float, default=0.22)
    parser.add_argument("--grasp_text_threshold", type=float, default=0.18)
    parser.add_argument("--grasp_min_score", type=float, default=0.50)
    parser.add_argument("--grasp_top_k", type=int, default=50)
    parser.add_argument("--grasp_selection_mode", default="best_good", choices=["best_good", "random_good", "random_top_k_good"])
    parser.add_argument("--grasp_random_select_top_k", type=int, default=5)
    parser.add_argument("--bbox_filter_margin_px", type=int, default=0)
    parser.add_argument("--bbox_filter_inner_margin_ratio", type=float, default=0.05)
    parser.add_argument("--disable_anygrasp_collision_detection", action="store_true")
    parser.add_argument("--visualize_selected_grasp", action="store_true")
    parser.add_argument("--visualize_bbox_filtered_grasps", action="store_true")

    parser.add_argument("--vlm_base_url", default=os.getenv("VLM_OCCLUSION_BASE_URL", os.getenv("ARM_AGENT_API_BASE_URL", os.getenv("PLANNER_API_BASE_URL", ""))))
    parser.add_argument("--vlm_api_key", default=os.getenv("VLM_OCCLUSION_API_KEY", os.getenv("ARM_AGENT_API_KEY", os.getenv("PLANNER_API_KEY", ""))))
    parser.add_argument("--vlm_model", default=os.getenv("VLM_OCCLUSION_MODEL", os.getenv("ARM_AGENT_API_MODEL", os.getenv("PLANNER_API_MODEL", ""))))
    parser.add_argument("--vlm_timeout", type=int, default=120)
    parser.add_argument("--vlm_temperature", type=float, default=0.0)
    parser.add_argument("--vlm_max_tokens", type=int, default=500)

    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    parser.add_argument("--gripper_close_wait_s", type=float, default=1.5)
    parser.add_argument("--verbose_events", action="store_true")
    args = parser.parse_args()

    if not (args.vlm_base_url and args.vlm_api_key and args.vlm_model):
        parser.error("VLM occlusion config required: pass --vlm_base_url, --vlm_api_key, --vlm_model or set env vars.")
    return args


def main() -> None:
    args = parse_args()
    arm = init_arm(args.xarm_ip) if args.execute else None
    context, camera = build_context(args, arm=arm)
    try:
        camera.start()
        time.sleep(float(args.camera_warmup_s))

        gripper_open = None
        gripper_close = None
        if args.execute and not args.disable_grip:
            gripper_open = lambda: call_grip_service(args.grip_open_value, args.ros_ws, args.grip_service, args.grip_timeout, args.ignore_grip_error)
            gripper_close = lambda: call_grip_service(args.grip_close_value, args.ros_ws, args.grip_service, args.grip_timeout, args.ignore_grip_error)

        robot = SimpleRobotActions(
            context=context,
            config=SimpleRobotConfig(
                execute=False,
                grasp_min_score=args.grasp_min_score,
                grasp_box_threshold=args.grasp_box_threshold,
                grasp_text_threshold=args.grasp_text_threshold,
                enable_nesyro_gate=False,
                nesyro_gate_trace_path="",
            ),
        )

        print(f"\n[Step 1] observe target: {args.target}")
        observe_result = robot.observe(args.target)
        if not observe_result.success:
            robot.observations[args.target] = {
                "kind": "medicine",
                "target": args.target,
                "observation_failed": True,
                "error": observe_result.error,
                "result": observe_result.data,
            }
        run_vlm_occlusion(args, context, robot, observe_result)
        gate_result, _ = run_gate(args, robot)
        if gate_result.decision != GateDecision.REPLAN or "target_occluded" not in gate_result.failed_conditions:
            print("[Done] Gate did not request occlusion clearing.")
            return

        observation = robot.observations[args.target]
        add_detector_blocker_candidates(args, context, observation)
        blocker = select_blocker_candidate(args, observation, manual_bbox=args.blocker_bbox)
        print("\n[Step 2] selected blocker candidate:")
        print(json.dumps(blocker, ensure_ascii=False, indent=2, default=str))

        grasp_input = build_blocker_grasp_input(args, observation, blocker)
        print("\n[Step 3] plan blocker grasp")
        plan_result = plan_blocker_grasp(args, context, grasp_input)
        if not plan_result.success:
            print("[blocker] grasp planning failed:", plan_result.error)
            return
        print("[blocker] grasp planned.")
        print("best_grasp_xarm_mmrad:", plan_result.data.get("best_grasp_xarm_mmrad"))
        print("buffer_pose_mmrad:", args.buffer_pose_mmrad)

        if not args.execute:
            print("\n[Dry-run] Add --execute to actually move blocker to buffer.")
            return

        print("\n[Step 4] execute clear-to-buffer")
        execution = execute_clear_to_buffer(args, context, plan_result.data, gripper_open, gripper_close)
        print(json.dumps(execution, ensure_ascii=False, indent=2))
        if not execution.get("success"):
            return

        print("\n[Step 5] reobserve target after clearing")
        robot.observations.pop(args.target, None)
        observe_result = robot.observe(args.target)
        if observe_result.success:
            run_vlm_occlusion(args, context, robot, observe_result)
            run_gate(args, robot)
        else:
            print("[reobserve] failed:", observe_result.error)
    finally:
        camera.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
