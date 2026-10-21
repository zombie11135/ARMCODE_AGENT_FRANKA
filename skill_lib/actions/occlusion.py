"""VLM occlusion evidence, blocker recovery and active-view retries.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.actions import SimpleAction
from core.skill_base import SkillResult
from services.object_names import display_name
from skill_lib.actions.observation_cache import invalidate_after_manipulation, require_fresh_observation
from runtime.observation_memory import fresh_camera

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _existing_path(path_text: str) -> Optional[str]:
    path = Path(path_text)
    return str(path) if path.exists() else None


def _should_run_occlusion_recovery(
    robot: SimpleRobotActions,
    action: SimpleAction,
    failed_conditions: List[str],
    require_occlusion: bool = True,
) -> bool:
    if action.op != "grasp":
        return False
    if require_occlusion and "target_occluded" not in failed_conditions:
        return False
    if not robot.config.enable_occlusion_recovery or not robot.config.execute:
        return False
    count = int(robot.occlusion_recovery_counts.get(action.target, 0))
    return count < max(0, int(robot.config.occlusion_recovery_max_attempts or 0))


def _run_occlusion_recovery(robot: SimpleRobotActions, target: str) -> SkillResult:
    robot.occlusion_recovery_counts[target] = int(
        robot.occlusion_recovery_counts.get(target, 0)
    ) + 1
    print(
        "[NeSyRoGate][Recovery] "
        f"clear occlusion for {target}, "
        f"attempt={robot.occlusion_recovery_counts[target]}"
    )

    result = robot._invoke_clear_occlusion_skill(target)
    active_view_attempts = [robot._recovery_attempt_record("current", result)]
    if result.success or not robot._should_try_active_occlusion_view(result):
        result.data.setdefault("active_view_attempts", active_view_attempts)
        print(
            "[NeSyRoGate][Recovery] "
            f"success={result.success}, error={result.error or '-'}"
        )
        return result

    last_result = result
    cfg = robot.config
    for offset_y in cfg.occlusion_active_view_y_offsets_mm:
        pose = list(cfg.home_pose_mmrad)
        pose[1] += float(offset_y)
        pose[2] = max(float(pose[2]), float(cfg.occlusion_active_view_z_mm))
        print(
            "[NeSyRoGate][Recovery][ActiveView] "
            f"move pose={pose}"
        )
        if not robot._move_xarm(pose):
            active_view_attempts.append(
                {
                    "view": {"pose_mmrad": pose},
                    "success": False,
                    "error": "move active view failed",
                }
            )
            continue
        if cfg.occlusion_active_view_wait_s > 0:
            time.sleep(float(cfg.occlusion_active_view_wait_s))

        with fresh_camera(getattr(robot.context, 'camera', None), 'occlusion_active_view_probe'):
            observe_result = robot._observe_medicine_once(target)
        if not observe_result.success:
            robot.observations[target] = {
                "kind": "medicine",
                "target": target,
                "observation_failed": True,
                "error": observe_result.error,
                "result": observe_result.data if isinstance(observe_result.data, dict) else {},
            }
        vlm_result = robot._run_vlm_occlusion_gate_if_enabled(target)
        if not vlm_result.success:
            active_view_attempts.append(
                {
                    "view": {"pose_mmrad": pose},
                    "success": False,
                    "error": vlm_result.error,
                    "observe_success": observe_result.success,
                }
            )
            continue

        observation = robot.observations.get(target) or {}
        vlm_occlusion = _as_dict(observation.get("vlm_occlusion"))
        if (
            robot.config.enable_vlm_occlusion_gate
            and observe_result.success
            and vlm_occlusion.get("target_occluded") is False
            and bool(vlm_occlusion.get("grasp_safe", True))
        ):
            resolved = SkillResult(
                success=True,
                data={
                    "recovery_action": "active_view_resolved_occlusion",
                    "no_blocker_moved": True,
                    "view_pose_mmrad": pose,
                },
                message="主动视角确认目标当前无需清障。",
            )
            active_view_attempts.append(
                robot._recovery_attempt_record({"pose_mmrad": pose}, resolved)
            )
            resolved.data["active_view_attempts"] = active_view_attempts
            return resolved

        view_result = robot._invoke_clear_occlusion_skill(target)
        active_view_attempts.append(
            robot._recovery_attempt_record({"pose_mmrad": pose}, view_result)
        )
        last_result = view_result
        if view_result.success:
            view_result.data["active_view_attempts"] = active_view_attempts
            print(
                "[NeSyRoGate][Recovery][ActiveView] "
                f"success at pose={pose}"
            )
            return view_result

    last_result.data.setdefault("active_view_attempts", active_view_attempts)
    print(
        "[NeSyRoGate][Recovery] "
        f"success=False, error={last_result.error or '-'}"
    )
    return last_result


@invalidate_after_manipulation('clear_occlusion')
def _invoke_clear_occlusion_skill(robot: SimpleRobotActions, target: str) -> SkillResult:
    from skill_lib.manipulation.clear_occlusion_to_buffer import (
        ClearOcclusionToBufferSkill,
    )

    observation = robot.observations.get(target)
    if not isinstance(observation, dict):
        return SkillResult(
            success=False,
            data={"failure_type": "occlusion_recovery_observation_missing"},
            error=f"清障恢复缺少目标观测：{target}",
        )

    cfg = robot.config
    return ClearOcclusionToBufferSkill(robot.context).run(
        target_name=display_name(robot.context, target),
        observation=observation,
        execute=cfg.execute,
        capture_dir=cfg.occlusion_recovery_capture_dir,
        xarm_ip=cfg.xarm_ip,
        handeye_config_path=cfg.handeye_config_path,
        buffer_pose_mmrad=cfg.occlusion_buffer_pose_mmrad,
        buffer_approach_clearance_mm=cfg.occlusion_buffer_approach_clearance_mm,
        move_speed=cfg.move_speed,
        move_acc=cfg.move_acc,
        gripper_open=robot.gripper_open,
        gripper_close=robot.gripper_close,
        gripper_close_wait_s=cfg.gripper_close_wait_s,
        blocker_bbox_shrink_ratio=cfg.occlusion_blocker_bbox_shrink_ratio,
        blocker_search_expand_ratio=cfg.occlusion_blocker_search_expand_ratio,
        blocker_min_area_ratio=cfg.occlusion_blocker_min_area_ratio,
        blocker_depth_closer_margin_m=cfg.occlusion_blocker_depth_closer_margin_m,
        blocker_detector_prompts=cfg.occlusion_blocker_detector_prompts,
        blocker_detector_threshold=cfg.occlusion_blocker_detector_threshold,
        blocker_detector_text_threshold=cfg.occlusion_blocker_detector_text_threshold,
        blocker_detector_max_objects=cfg.occlusion_blocker_detector_max_objects,
        allow_unlocalized_target=cfg.occlusion_allow_unlocalized_target,
        blocker_max_image_area_ratio=cfg.occlusion_blocker_max_image_area_ratio,
        blocker_max_target_area_ratio=cfg.occlusion_blocker_max_target_area_ratio,
        unlocalized_vlm_min_confidence=cfg.occlusion_unlocalized_vlm_min_confidence,
        unlocalized_detector_min_confidence=cfg.occlusion_unlocalized_detector_min_confidence,
        min_depth_valid_ratio=cfg.occlusion_min_depth_valid_ratio,
        foreground_depth_band_m=cfg.occlusion_foreground_depth_band_m,
        min_foreground_support_ratio=cfg.occlusion_min_foreground_support_ratio,
        grasp_top_k=cfg.grasp_top_k,
        grasp_selection_mode=cfg.grasp_selection_mode,
        grasp_random_select_top_k=cfg.grasp_random_select_top_k,
        grasp_random_seed=cfg.grasp_random_seed,
        grasp_pose_options=cfg.grasp_pose_options,
        bbox_filter_margin_px=cfg.bbox_filter_margin_px,
        bbox_filter_inner_margin_ratio=cfg.bbox_filter_inner_margin_ratio,
        collision_detection=cfg.collision_detection,
        visualize_selected_grasp=cfg.visualize_selected_grasp,
        visualize_bbox_filtered_grasps=cfg.visualize_bbox_filtered_grasps,
    )


def _should_try_active_occlusion_view(robot: SimpleRobotActions, result: SkillResult) -> bool:
    if not robot.config.occlusion_active_view_enabled or not robot.config.execute:
        return False
    failure_type = str((result.data or {}).get("failure_type") or "")
    return failure_type in {
        "occlusion_recovery_blocker_not_found",
        "occlusion_recovery_depth_missing",
        "occlusion_recovery_depth_invalid",
        "occlusion_recovery_foreground_depth_invalid",
    }


def _recovery_attempt_record(view: Any, result: SkillResult) -> Dict[str, Any]:
    return {
        "view": view,
        "success": result.success,
        "error": result.error,
        "data": dict(result.data or {}),
    }


@require_fresh_observation('after_occlusion_recovery')
def _refresh_after_occlusion_recovery(robot: SimpleRobotActions, target: str) -> SkillResult:
    robot.observations.pop(target, None)
    result = robot.observe(target)
    if result.success:
        return result
    robot.observations[target] = {
        "kind": "medicine",
        "target": target,
        "observation_failed": True,
        "error": result.error,
        "result": result.data if isinstance(result.data, dict) else {},
    }
    return result


def _run_vlm_occlusion_gate_if_enabled(robot: SimpleRobotActions, target: str) -> SkillResult:
    if not robot.config.enable_vlm_occlusion_gate:
        return SkillResult(success=True, data={"vlm_occlusion_gate": {"enabled": False}})
    if not (
        robot.config.vlm_occlusion_base_url
        and robot.config.vlm_occlusion_api_key
        and robot.config.vlm_occlusion_model
    ):
        return SkillResult(
            success=False,
            data={
                "failure_type": "nesyro_gate_vlm_config_missing",
                "target": target,
                "vlm_occlusion_gate": {
                    "enabled": True,
                    "base_url_present": bool(robot.config.vlm_occlusion_base_url),
                    "api_key_present": bool(robot.config.vlm_occlusion_api_key),
                    "model_present": bool(robot.config.vlm_occlusion_model),
                },
            },
            error=(
                "NeSyRo Gate VLM occlusion check is enabled, but VLM API "
                "configuration is incomplete."
            ),
        )

    observation = robot.observations.get(target)
    if not isinstance(observation, dict):
        return SkillResult(
            success=False,
            data={
                "failure_type": "nesyro_gate_vlm_observation_missing",
                "target": target,
            },
            error=f"NeSyRo Gate VLM occlusion check has no observation for {target}.",
        )

    selected = observation.get("selected_instance") or {}
    grasp_input = observation.get("grasp_input") or {}
    result_data = observation.get("result") or {}
    image_path = (
        result_data.get("image_path")
        or _as_dict(result_data.get("last_failure_data")).get("image_path")
        or grasp_input.get("color_path")
        or selected.get("image_path")
        or _existing_path("runs/current_capture/wrist_color.png")
    )
    bbox = selected.get("bbox") or grasp_input.get("bbox")
    if not image_path:
        return SkillResult(
            success=False,
            data={
                "failure_type": "nesyro_gate_vlm_image_missing",
                "target": target,
                "observation": observation,
            },
            error=f"NeSyRo Gate VLM occlusion check has no image for {target}.",
        )

    try:
        from skill_lib.evaluation.verify_target_occlusion_vlm import (
            VerifyTargetOcclusionVLMSkill,
        )

        print(f"[NeSyRoGate][VLM] occlusion check target={target}, image={image_path}, bbox={bbox}")
        result = VerifyTargetOcclusionVLMSkill(robot.context).run(
            target_object_name=display_name(robot.context, target),
            image_path=image_path,
            bbox=bbox,
            base_url=robot.config.vlm_occlusion_base_url,
            api_key=robot.config.vlm_occlusion_api_key,
            model_name=robot.config.vlm_occlusion_model,
            timeout=robot.config.vlm_occlusion_timeout,
            max_tokens=robot.config.vlm_occlusion_max_tokens,
            temperature=robot.config.vlm_occlusion_temperature,
        )
        if not result.success:
            observation["vlm_occlusion_error"] = result.error
            return SkillResult(
                success=False,
                data={
                    "failure_type": "nesyro_gate_vlm_occlusion_failed",
                    "target": target,
                    "vlm_error": result.error,
                    "vlm_data": result.data,
                },
                error=f"NeSyRo Gate VLM occlusion check failed: {result.error}",
            )
        observation["vlm_occlusion"] = result.data
        if isinstance(observation.get("result"), dict):
            observation["result"]["vlm_occlusion"] = result.data
        print(
            "[NeSyRoGate][VLM] "
            f"occluded={result.data.get('target_occluded')} "
            f"level={result.data.get('occlusion_level')} "
            f"grasp_safe={result.data.get('grasp_safe')} "
            f"confidence={result.data.get('confidence')}"
        )
        return SkillResult(
            success=True,
            data={"vlm_occlusion_gate": result.data},
        )
    except Exception as e:
        observation["vlm_occlusion_error"] = str(e)
        return SkillResult(
            success=False,
            data={
                "failure_type": "nesyro_gate_vlm_occlusion_error",
                "target": target,
                "vlm_error": str(e),
            },
            error=f"NeSyRo Gate VLM occlusion check error: {e}",
        )
