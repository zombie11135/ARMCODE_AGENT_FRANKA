from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.skill_base import BaseSkill, SkillResult
from runtime.observation_memory import scene_mutation


FALLBACK_INTRINSICS = {
    "fx": 909.59,
    "fy": 909.80,
    "cx": 658.97,
    "cy": 355.42,
}


class ClearOcclusionToBufferSkill(BaseSkill):
    """Detect, grasp, and move a target's blocking object to a fixed buffer."""

    name = "clear_occlusion_to_buffer"
    description = "检测目标上方遮挡物，使用 AnyGrasp 抓取并搬运到固定 buffer 位姿。"

    def run(
        self,
        target_name: str,
        observation: Dict[str, Any],
        execute: bool = False,
        capture_dir: str = "runs/current_capture",
        xarm_ip: str = "192.168.1.237",
        handeye_config_path: str = "config/calibration/wrist_handeye.yaml",
        buffer_pose_mmrad: Optional[List[float]] = None,
        buffer_approach_clearance_mm: float = 120.0,
        move_speed: float = 50.0,
        move_acc: float = 500.0,
        gripper_open: Optional[Callable[[], bool]] = None,
        gripper_close: Optional[Callable[[], bool]] = None,
        gripper_close_wait_s: float = 1.5,
        blocker_bbox: Optional[List[float]] = None,
        blocker_bbox_shrink_ratio: float = 0.08,
        blocker_search_expand_ratio: float = 1.8,
        blocker_min_area_ratio: float = 0.50,
        blocker_depth_closer_margin_m: float = 0.01,
        blocker_detector_prompts: Optional[List[str]] = None,
        blocker_detector_threshold: float = 0.20,
        blocker_detector_text_threshold: float = 0.16,
        blocker_detector_max_objects: int = 12,
        allow_unlocalized_target: bool = False,
        blocker_max_image_area_ratio: float = 0.50,
        blocker_max_target_area_ratio: float = 6.0,
        unlocalized_vlm_min_confidence: float = 0.85,
        unlocalized_detector_min_confidence: float = 0.25,
        min_depth_valid_ratio: float = 0.35,
        foreground_depth_band_m: float = 0.018,
        min_foreground_support_ratio: float = 0.15,
        grasp_top_k: int = 50,
        grasp_selection_mode: str = "best_good",
        grasp_random_select_top_k: int = 5,
        grasp_random_seed: Optional[int] = None,
        grasp_pose_options: Optional[Dict[str, float]] = None,
        bbox_filter_margin_px: int = 0,
        bbox_filter_inner_margin_ratio: float = 0.05,
        collision_detection: bool = True,
        visualize_selected_grasp: bool = False,
        visualize_bbox_filtered_grasps: bool = False,
        **kwargs,
    ) -> SkillResult:
        del kwargs
        try:
            resolved_buffer_pose = list(
                buffer_pose_mmrad
                or [280.0, -260.0, 170.0, 3.14, 0.0, -1.57]
            )
            if len(resolved_buffer_pose) != 6 or not all(
                math.isfinite(float(value)) for value in resolved_buffer_pose
            ):
                return _failure(
                    "occlusion_recovery_config_invalid",
                    "buffer_pose_mmrad 必须是 6 个有限数值 [x,y,z,rx,ry,rz]。",
                )
            if execute and (not callable(gripper_open) or not callable(gripper_close)):
                return _failure(
                    "occlusion_recovery_gripper_unavailable",
                    "真实清障必须配置可用的夹爪打开和关闭回调。",
                )

            result_data = _as_dict(observation.get("result"))
            grasp_input = _as_dict(observation.get("grasp_input"))
            image_path = _resolve_image_path(observation, capture_dir)
            if not image_path:
                return _failure(
                    "occlusion_recovery_image_missing",
                    "清障恢复缺少 wrist RGB 图像。",
                )

            vlm_occlusion = _as_dict(
                observation.get("vlm_occlusion")
                or result_data.get("vlm_occlusion")
            )
            blocker_description = _vlm_blocker_description(vlm_occlusion)
            detected = self._detect_blocker_candidates(
                image_path=image_path,
                prompts=blocker_detector_prompts,
                blocker_description=blocker_description,
                threshold=blocker_detector_threshold,
                text_threshold=blocker_detector_text_threshold,
                max_objects=blocker_detector_max_objects,
            )
            candidates = _collect_bbox_candidates(result_data, detected)
            depth = _load_depth(
                capture_dir=capture_dir,
                grasp_input=grasp_input,
                result_data=result_data,
            )
            if depth is None:
                return _failure(
                    "occlusion_recovery_depth_missing",
                    "清障恢复缺少可读取的同帧 wrist 深度图，禁止规划抓取。",
                    recommendation="move_camera_farther_and_reobserve",
                )
            blocker = _select_blocker_candidate(
                observation=observation,
                candidates=candidates,
                depth=depth,
                manual_bbox=blocker_bbox,
                search_expand_ratio=blocker_search_expand_ratio,
                min_area_ratio=blocker_min_area_ratio,
                depth_closer_margin_m=blocker_depth_closer_margin_m,
                allow_unlocalized_target=allow_unlocalized_target,
                image_size=_image_size(image_path),
                max_image_area_ratio=blocker_max_image_area_ratio,
                max_target_area_ratio=blocker_max_target_area_ratio,
                vlm_occlusion=vlm_occlusion,
                unlocalized_vlm_min_confidence=unlocalized_vlm_min_confidence,
                unlocalized_detector_min_confidence=unlocalized_detector_min_confidence,
            )
            selection_vis_path = _write_selection_visualization(
                image_path=image_path,
                output_dir=capture_dir,
                target_bbox=_target_bbox(observation),
                candidates=candidates,
                selected=blocker,
                target_name=target_name,
            )
            if selection_vis_path and self.context is not None:
                try:
                    self.context.emit_image(
                        image_path=selection_vis_path,
                        caption="清障候选与最终抓取 bbox",
                    )
                except Exception:
                    pass
            if blocker is None:
                return _failure(
                    "occlusion_recovery_blocker_not_found",
                    "没有找到可安全清除的遮挡物候选。",
                    detector_candidates=candidates,
                    blocker_description=blocker_description,
                    selection_visualization=selection_vis_path,
                )

            print(
                "[OcclusionRecovery] selected blocker "
                f"bbox={blocker.get('bbox')}, "
                f"confidence={float(blocker.get('confidence') or 0.0):.3f}, "
                f"score={float(blocker.get('blocker_selection_score') or 0.0):.3f}, "
                f"source={blocker.get('source')}, "
                f"visualization={selection_vis_path}"
            )

            planned_bbox = _shrink_bbox(
                blocker["bbox"],
                blocker_bbox_shrink_ratio,
            )
            valid_ratio = _depth_valid_ratio(depth, planned_bbox)
            if valid_ratio is not None and valid_ratio < float(min_depth_valid_ratio):
                return _failure(
                    "occlusion_recovery_depth_invalid",
                    (
                        "遮挡物 ROI 深度质量不足，禁止使用损坏点云规划抓取："
                        f"valid_ratio={valid_ratio:.3f} < {float(min_depth_valid_ratio):.3f}"
                    ),
                    blocker=blocker,
                    planned_bbox=planned_bbox,
                    depth_valid_ratio=valid_ratio,
                    recommendation="move_camera_farther_and_reobserve",
                )

            foreground = _build_foreground_depth_mask(
                depth=depth,
                bbox=planned_bbox,
                output_dir=capture_dir,
                depth_band_m=foreground_depth_band_m,
            )
            if (
                foreground is None
                or float(foreground.get("support_ratio") or 0.0)
                < float(min_foreground_support_ratio)
            ):
                support_ratio = (
                    None if foreground is None else foreground.get("support_ratio")
                )
                return _failure(
                    "occlusion_recovery_foreground_depth_invalid",
                    (
                        "遮挡物近层点云支撑不足，AnyGrasp 可能转而抓到下层目标，"
                        f"禁止执行：support_ratio={support_ratio}, "
                        f"required={float(min_foreground_support_ratio):.3f}"
                    ),
                    blocker=blocker,
                    planned_bbox=planned_bbox,
                    depth_valid_ratio=valid_ratio,
                    foreground_depth=foreground,
                    selection_visualization=selection_vis_path,
                    recommendation="change_viewpoint_and_reobserve",
                )

            blocker_grasp_input = _build_blocker_grasp_input(
                observation=observation,
                capture_dir=capture_dir,
                blocker=blocker,
                planned_bbox=planned_bbox,
                mask_path=foreground.get("mask_path"),
                mask_npy_path=foreground.get("mask_npy_path"),
            )
            pose_options = dict(grasp_pose_options or {})
            allowed = {'target_rx', 'target_ry', 'max_rx_tilt_deg', 'max_ry_tilt_deg', 'lam_rx', 'lam_ry'}
            if set(pose_options) - allowed:
                raise ValueError(f"Unknown grasp pose options: {sorted(set(pose_options) - allowed)}")
            plan_result = self._plan_grasp(
                **pose_options,
                grasp_input=blocker_grasp_input,
                xarm_ip=xarm_ip,
                handeye_config_path=handeye_config_path,
                top_k=grasp_top_k,
                grasp_selection_mode=grasp_selection_mode,
                random_select_top_k=grasp_random_select_top_k,
                random_seed=grasp_random_seed,
                bbox_filter_margin_px=bbox_filter_margin_px,
                bbox_filter_inner_margin_ratio=bbox_filter_inner_margin_ratio,
                collision_detection=collision_detection,
                visualize_selected_grasp=visualize_selected_grasp,
                visualize_bbox_filtered_grasps=visualize_bbox_filtered_grasps,
            )
            if not plan_result.success:
                return _failure(
                    "occlusion_recovery_grasp_planning_failed",
                    plan_result.error or "遮挡物 AnyGrasp 规划失败。",
                    blocker=blocker,
                    planned_bbox=planned_bbox,
                    depth_valid_ratio=valid_ratio,
                    grasp_plan_data=plan_result.data,
                )

            data = {
                "failure_type": None,
                "blocker": blocker,
                "blocker_description": blocker_description,
                "planned_bbox": planned_bbox,
                "selection_visualization": selection_vis_path,
                "depth_valid_ratio": valid_ratio,
                "foreground_depth": foreground,
                "grasp_plan": plan_result.data,
                "buffer_pose_mmrad": resolved_buffer_pose,
                "dry_run": not execute,
            }
            if not execute:
                return SkillResult(
                    success=True,
                    data=data,
                    message="遮挡物抓取与 buffer 搬运已完成 dry-run 规划。",
                )

            execution = self._execute_to_buffer(
                plan=plan_result.data,
                buffer_pose_mmrad=data["buffer_pose_mmrad"],
                buffer_approach_clearance_mm=buffer_approach_clearance_mm,
                move_speed=move_speed,
                move_acc=move_acc,
                gripper_open=gripper_open,
                gripper_close=gripper_close,
                gripper_close_wait_s=gripper_close_wait_s,
            )
            data["execution"] = execution
            if not execution.get("success"):
                data["failure_type"] = "occlusion_recovery_execution_failed"
                return SkillResult(
                    success=False,
                    data=data,
                    error=execution.get("error") or "遮挡物搬运执行失败。",
                )
            return SkillResult(
                success=True,
                data=data,
                message="遮挡物已搬运到 buffer。",
            )
        except Exception as exc:
            return _failure(
                "occlusion_recovery_error",
                f"清障恢复异常：{exc}",
            )

    def _detect_blocker_candidates(
        self,
        image_path: str,
        prompts: Optional[List[str]],
        blocker_description: Optional[str],
        threshold: float,
        text_threshold: float,
        max_objects: int,
    ) -> List[Dict[str, Any]]:
        detector = self.context.require("detector")
        generic_prompts = list(
            prompts
            or [
                "medicine_box",
                "snack",
                "box",
                "package",
                "object on top of another object",
            ]
        )
        prompt_specs = []
        if blocker_description:
            prompt_specs.append((blocker_description, "vlm_blocker_description"))
        prompt_specs.extend((prompt, f"blocker_detector:{prompt}") for prompt in generic_prompts)
        merged: List[Dict[str, Any]] = []
        for prompt, source in prompt_specs:
            result = detector.detect(
                image_path=image_path,
                target_name=prompt,
                threshold=float(threshold),
                text_threshold=float(text_threshold),
                max_objects=int(max_objects),
                enable_filter=False,
            )
            for item in result.get("detections", []) or []:
                candidate = dict(item)
                candidate["source"] = source
                candidate["source_prompt"] = prompt
                merged.append(candidate)
        return _dedupe_candidates(merged)

    def _plan_grasp(self, grasp_input: Dict[str, Any], **kwargs) -> SkillResult:
        from skill_lib.manipulation.plan_grasp_with_anygrasp import (
            PlanGraspWithAnyGraspSkill,
        )

        return PlanGraspWithAnyGraspSkill(self.context).run(
            grasp_input=grasp_input,
            filter_grasps_by_bbox=True,
            **kwargs,
        )

    @scene_mutation('clear_occlusion')
    def _execute_to_buffer(
        self,
        plan: Dict[str, Any],
        buffer_pose_mmrad: List[float],
        buffer_approach_clearance_mm: float,
        move_speed: float,
        move_acc: float,
        gripper_open: Optional[Callable[[], bool]],
        gripper_close: Optional[Callable[[], bool]],
        gripper_close_wait_s: float,
    ) -> Dict[str, Any]:
        arm = self.context.require("xarm")
        if not _xarm_ready(arm):
            return {"success": False, "error": "xArm is not ready for occlusion recovery"}

        approach = plan.get("approach_pose_mmrad")
        grasp_pose = plan.get("best_grasp_xarm_mmrad")
        lift = plan.get("lift_pose_mmrad")
        for name, pose in [("approach", approach), ("grasp", grasp_pose), ("lift", lift)]:
            if not isinstance(pose, (list, tuple)) or len(pose) != 6:
                return {"success": False, "error": f"missing {name} pose"}

        if gripper_open and not gripper_open():
            return {"success": False, "error": "gripper open failed"}
        for label, pose in [("approach", approach), ("grasp", grasp_pose)]:
            if not _move_xarm(arm, pose, move_speed, move_acc):
                return {"success": False, "error": f"move {label} failed"}
        if gripper_close and not gripper_close():
            return {"success": False, "error": "gripper close failed"}
        time.sleep(float(gripper_close_wait_s))
        if not _move_xarm(arm, lift, move_speed, move_acc):
            return {"success": False, "error": "lift blocker failed"}

        buffer_pose = list(buffer_pose_mmrad)
        buffer_approach = list(buffer_pose)
        buffer_approach[2] += float(buffer_approach_clearance_mm)
        for label, pose in [("buffer_approach", buffer_approach), ("buffer", buffer_pose)]:
            if not _move_xarm(arm, pose, move_speed, move_acc):
                return {"success": False, "error": f"move {label} failed"}
        if gripper_open and not gripper_open():
            return {"success": False, "error": "release blocker failed"}
        if not _move_xarm(arm, buffer_approach, move_speed, move_acc):
            return {"success": False, "error": "lift from buffer failed"}
        return {"success": True}


def _failure(failure_type: str, error: str, **data) -> SkillResult:
    return SkillResult(
        success=False,
        data={"failure_type": failure_type, **data},
        error=error,
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_image_path(observation: Dict[str, Any], capture_dir: str) -> Optional[str]:
    result = _as_dict(observation.get("result"))
    grasp_input = _as_dict(observation.get("grasp_input"))
    selected = _as_dict(observation.get("selected_instance"))
    candidates = [
        result.get("image_path"),
        _as_dict(result.get("last_failure_data")).get("image_path"),
        grasp_input.get("color_path"),
        selected.get("image_path"),
        str(Path(capture_dir) / "wrist_color.png"),
    ]
    for path_text in candidates:
        if path_text and Path(str(path_text)).exists():
            return str(path_text)
    return None


def _vlm_blocker_description(vlm_occlusion: Dict[str, Any]) -> Optional[str]:
    if not vlm_occlusion.get("blocking_object_visible"):
        return None
    description = str(vlm_occlusion.get("blocking_object_description") or "").strip()
    return description or None


def _target_bbox(observation: Dict[str, Any]) -> Optional[List[float]]:
    result_data = _as_dict(observation.get("result"))
    selected = _as_dict(
        observation.get("selected_instance") or result_data.get("selected_instance")
    )
    bbox = selected.get("bbox") or _as_dict(observation.get("grasp_input")).get("bbox")
    return list(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None


def _image_size(image_path: str) -> Optional[List[int]]:
    try:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        return [int(width), int(height)]
    except Exception:
        return None


def _collect_bbox_candidates(
    result_data: Dict[str, Any],
    detected: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates = list(detected)
    sources = [result_data, _as_dict(result_data.get("last_failure_data"))]
    for attempt in result_data.get("failure_attempts", []) or []:
        sources.append(_as_dict(_as_dict(attempt).get("data")))
    for data in sources:
        for source_key in ["detections", "instances", "ranked_candidates"]:
            for item in data.get(source_key, []) or []:
                candidate = (
                    item.get("instance")
                    if isinstance(item, dict) and isinstance(item.get("instance"), dict)
                    else item
                )
                if isinstance(candidate, dict) and candidate.get("bbox"):
                    candidate = dict(candidate)
                    candidate["source"] = candidate.get("source") or source_key
                    candidates.append(candidate)
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for candidate in candidates:
        bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
        if not bbox or len(bbox) != 4:
            continue
        key = tuple(round(float(value), 1) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _select_blocker_candidate(
    observation: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    depth,
    manual_bbox: Optional[List[float]],
    search_expand_ratio: float,
    min_area_ratio: float,
    depth_closer_margin_m: float,
    allow_unlocalized_target: bool,
    image_size: Optional[List[int]] = None,
    max_image_area_ratio: float = 0.50,
    max_target_area_ratio: float = 6.0,
    vlm_occlusion: Optional[Dict[str, Any]] = None,
    unlocalized_vlm_min_confidence: float = 0.85,
    unlocalized_detector_min_confidence: float = 0.25,
) -> Optional[Dict[str, Any]]:
    if manual_bbox:
        return {"object_id": "manual_blocker", "bbox": list(manual_bbox), "source": "manual"}

    target_bbox = _target_bbox(observation)
    image_area = _image_area(image_size)
    if not target_bbox:
        vlm_occlusion = _as_dict(vlm_occlusion)
        vlm_verified = (
            bool(vlm_occlusion.get("target_occluded"))
            and bool(vlm_occlusion.get("blocking_object_visible"))
            and bool(_vlm_blocker_description(vlm_occlusion))
            and float(vlm_occlusion.get("confidence") or 0.0)
            >= float(unlocalized_vlm_min_confidence)
        )
        if not candidates or (not allow_unlocalized_target and not vlm_verified):
            return None
        scored = []
        for raw in candidates:
            candidate = dict(raw)
            bbox = candidate.get("bbox")
            if not bbox or _candidate_too_large(
                bbox=bbox,
                image_area=image_area,
                max_image_area_ratio=max_image_area_ratio,
            ):
                continue
            confidence = float(candidate.get("confidence") or 0.0)
            description_grounded = candidate.get("source") == "vlm_blocker_description"
            if vlm_verified and (
                not description_grounded
                or confidence < float(unlocalized_detector_min_confidence)
            ):
                continue
            if not vlm_verified and not allow_unlocalized_target:
                continue
            area_ratio = _bbox_area(bbox) / image_area if image_area else 0.0
            score = confidence + (0.8 if description_grounded else 0.0) - 0.3 * area_ratio
            candidate["blocker_selection_score"] = round(float(score), 4)
            candidate["selection_mode"] = (
                "vlm_description_grounded_unlocalized"
                if vlm_verified
                else "explicit_unlocalized_fallback"
            )
            scored.append((score, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    target_area = max(1.0, _bbox_area(target_bbox))
    expanded_target = _expand_bbox(target_bbox, search_expand_ratio)
    target_depth = _median_depth(depth, target_bbox)
    scored = []
    for raw in candidates:
        candidate = dict(raw)
        bbox = candidate.get("bbox")
        if not bbox:
            continue
        candidate_area = max(1.0, _bbox_area(bbox))
        if _candidate_too_large(
            bbox=bbox,
            image_area=image_area,
            max_image_area_ratio=max_image_area_ratio,
        ):
            continue
        if candidate_area < target_area * float(min_area_ratio):
            continue
        if candidate_area > target_area * float(max_target_area_ratio):
            continue
        intersection = _bbox_intersection_area(target_bbox, bbox)
        expanded_intersection = _bbox_intersection_area(expanded_target, bbox)
        if intersection <= 0 and expanded_intersection <= 0:
            continue
        iou = intersection / max(1.0, target_area + candidate_area - intersection)
        overlap_target = intersection / target_area
        overlap_expanded = expanded_intersection / candidate_area
        if iou >= 0.85:
            continue
        if overlap_target <= 0.02 and overlap_expanded <= 0.20:
            continue

        candidate_depth = _median_depth(depth, bbox)
        depth_bonus = 0.0
        if target_depth is not None and candidate_depth is not None:
            depth_bonus = (
                1.0
                if candidate_depth + float(depth_closer_margin_m) < target_depth
                else -0.4
            )
        score = (
            1.6 * overlap_target
            + 0.8 * overlap_expanded
            + 0.1 * min(candidate_area / target_area, 3.0)
            + depth_bonus
            + (0.35 if candidate.get("source") == "vlm_blocker_description" else 0.0)
            + 0.2 * float(candidate.get("confidence") or 0.0)
            - 0.5 * iou
        )
        candidate["blocker_selection_score"] = round(float(score), 4)
        candidate["target_depth_median"] = target_depth
        candidate["candidate_depth_median"] = candidate_depth
        scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _image_area(image_size: Optional[List[int]]) -> Optional[float]:
    if not image_size or len(image_size) != 2:
        return None
    width, height = [float(value) for value in image_size]
    return width * height if width > 0 and height > 0 else None


def _candidate_too_large(
    bbox: List[float],
    image_area: Optional[float],
    max_image_area_ratio: float,
) -> bool:
    if not image_area:
        return False
    return _bbox_area(bbox) / image_area > float(max_image_area_ratio)


def _write_selection_visualization(
    image_path: str,
    output_dir: str,
    target_bbox: Optional[List[float]],
    candidates: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    target_name: str,
) -> Optional[str]:
    try:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            return None
        for index, candidate in enumerate(candidates, start=1):
            bbox = candidate.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (150, 150, 150), 1)
            label = f"C{index} {float(candidate.get('confidence') or 0.0):.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )
        if target_bbox:
            x1, y1, x2, y2 = [int(round(float(value))) for value in target_bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 3)
            cv2.putText(
                image,
                f"TARGET {target_name}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        if selected and selected.get("bbox"):
            x1, y1, x2, y2 = [int(round(float(value))) for value in selected["bbox"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 4)
            score = float(selected.get("blocker_selection_score") or 0.0)
            cv2.putText(
                image,
                f"BLOCKER score={score:.2f}",
                (x1, min(image.shape[0] - 8, y2 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
        output_path = Path(output_dir) / "occlusion_recovery_blocker_selection.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            return None
        return str(output_path)
    except Exception:
        return None


def _load_depth(capture_dir: str, grasp_input: Dict[str, Any], result_data: Dict[str, Any]):
    depth_npy_path = grasp_input.get("depth_npy_path")
    depth_path = grasp_input.get("depth_path")
    if not depth_npy_path:
        for name in ["wrist_depth.npy", "wrist_depth_meter.npy"]:
            candidate = Path(capture_dir) / name
            if candidate.exists():
                depth_npy_path = str(candidate)
                break
    if not depth_path:
        candidate = Path(capture_dir) / "wrist_depth.png"
        if candidate.exists():
            depth_path = str(candidate)
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
            scale = float(grasp_input.get("depth_scale") or result_data.get("depth_scale") or 1000.0)
            return raw * scale if 0 < scale < 1 else raw / scale
    except Exception:
        return None
    return None


def _median_depth(depth, bbox: List[float]) -> Optional[float]:
    if depth is None:
        return None
    try:
        import numpy as np

        h, w = depth.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)]
        return float(np.median(valid)) if valid.size >= 20 else None
    except Exception:
        return None


def _depth_valid_ratio(depth, bbox: List[float]) -> Optional[float]:
    if depth is None:
        return None
    try:
        import numpy as np

        h, w = depth.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        valid = np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)
        return float(valid.sum() / patch.size)
    except Exception:
        return None


def _build_foreground_depth_mask(
    depth,
    bbox: List[float],
    output_dir: str,
    depth_band_m: float,
) -> Optional[Dict[str, Any]]:
    if depth is None:
        return None
    try:
        import cv2
        import numpy as np

        height, width = depth.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
        patch = depth[y1:y2, x1:x2]
        valid = np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)
        values = patch[valid]
        if values.size < 200:
            return None

        near_depth = float(np.percentile(values, 10.0))
        max_depth = near_depth + max(0.005, float(depth_band_m))
        foreground_patch = valid & (patch <= max_depth)
        binary = (foreground_patch.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 1:
            return None
        component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = labels == component_index
        component_area = int(component.sum())
        support_ratio = float(component_area / max(1, patch.size))

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = component.astype(np.uint8)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        mask_npy_path = output_path / "occlusion_recovery_foreground_mask.npy"
        mask_path = output_path / "occlusion_recovery_foreground_mask.png"
        np.save(str(mask_npy_path), mask)
        cv2.imwrite(str(mask_path), mask * 255)
        return {
            "mask_path": str(mask_path),
            "mask_npy_path": str(mask_npy_path),
            "near_depth_m": near_depth,
            "max_depth_m": max_depth,
            "support_ratio": support_ratio,
            "component_area_px": component_area,
        }
    except Exception:
        return None


def _build_blocker_grasp_input(
    observation: Dict[str, Any],
    capture_dir: str,
    blocker: Dict[str, Any],
    planned_bbox: List[float],
    mask_path: Optional[str] = None,
    mask_npy_path: Optional[str] = None,
) -> Dict[str, Any]:
    result_data = _as_dict(observation.get("result"))
    base = dict(observation.get("grasp_input") or result_data.get("grasp_input") or {})
    if not base:
        image_path = _resolve_image_path(observation, capture_dir)
        depth_path = str(Path(capture_dir) / "wrist_depth.png")
        if not Path(depth_path).exists():
            depth_path = None
        depth_npy_path = None
        for name in ["wrist_depth.npy", "wrist_depth_meter.npy"]:
            candidate = Path(capture_dir) / name
            if candidate.exists():
                depth_npy_path = str(candidate)
                break
        if not image_path or (not depth_path and not depth_npy_path):
            raise RuntimeError("清障抓取缺少同帧 wrist RGB-D。")
        base = {
            "color_path": image_path,
            "image_path": image_path,
            "depth_path": depth_path,
            "depth_npy_path": depth_npy_path,
            "depth_intrinsics": FALLBACK_INTRINSICS,
            "color_intrinsics": FALLBACK_INTRINSICS,
            "camera_intrinsics": FALLBACK_INTRINSICS,
            "depth_scale": 1000.0,
        }
    base.update(
        {
            "object_id": blocker.get("object_id") or "blocker",
            "target_name": "occluding_object",
            "query": "occluding_object",
            "bbox": planned_bbox,
            "mask_path": mask_path,
            "mask_npy_path": mask_npy_path,
            "ocr_text": None,
            "ocr_texts": None,
        }
    )
    return base


def _bbox_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )


def _expand_bbox(bbox: List[float], ratio: float) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    ratio = max(1.0, float(ratio))
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w = (x2 - x1) * 0.5 * ratio
    half_h = (y2 - y1) * 0.5 * ratio
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def _shrink_bbox(bbox: List[float], ratio: float) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    ratio = max(0.0, min(0.45, float(ratio)))
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]


def _clip_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    return [
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    ]


def _normalize_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _move_xarm(arm, pose: List[float], speed: float, acc: float) -> bool:
    x, y, z, rx, ry, rz = [float(value) for value in pose]
    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=_normalize_angle(rx),
        pitch=_normalize_angle(ry),
        yaw=_normalize_angle(rz),
        speed=float(speed),
        mvacc=float(acc),
        wait=True,
    )
    return code == 0


def _xarm_ready(arm) -> bool:
    return getattr(arm, "error_code", 0) in (None, 0) and getattr(arm, "state", 0) != 4
