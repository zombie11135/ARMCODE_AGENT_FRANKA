"""Target observation, container caching and scanning across viewpoints.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, TYPE_CHECKING

from core.actions import SimpleAction
from core.skill_base import SkillResult
from skill_lib.config import FALLBACK_INTRINSICS
from skill_lib.actions.targets import container_key, parse_container_spec
from runtime.observation_memory import memory_for
from skill_lib.actions.observation_cache import reuse_observation, recall_observation

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


def preobserve_containers(robot: SimpleRobotActions, actions: Iterable[SimpleAction]) -> SkillResult:
    seen = set()
    results = []
    for action in actions:
        if action.op != "place":
            continue
        spec = parse_container_spec(action.target)
        key = container_key(spec)
        if key in seen:
            continue
        seen.add(key)
        result = robot.observe_container(action.target)
        results.append(
            {
                "container": action.target,
                "key": key,
                "success": result.success,
                "error": result.error,
                "data": result.data,
            }
        )
        if not result.success:
            return SkillResult(
                success=False,
                data={"preobserved_containers": results, "failed_container": action.target},
                error=f"preobserve container failed: {action.target}: {result.error}",
            )
        if robot.config.execute:
            home_result = robot.home()
            if not home_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "preobserved_containers": results,
                        "failed_container": action.target,
                        "home_after_container": home_result.data,
                    },
                    error=f"return home after container observation failed: {home_result.error}",
                )
    return SkillResult(success=True, data={"preobserved_containers": results})


@reuse_observation('container')
def observe_container(robot: SimpleRobotActions, container: str) -> SkillResult:
    spec = parse_container_spec(container)
    key = container_key(spec)
    if key in robot.place_plans and memory_for(robot.context.camera) is None:
        return SkillResult(success=True, data=robot.place_plans[key], message="container plan already cached")

    from skill_lib.combound.place_to_container import PlaceToContainerSkill

    cfg = robot.config
    result = PlaceToContainerSkill(robot.context).run(
        container_name=spec["name"],
        container_color=spec.get("color"),
        camera_name="wrist",
        detection_threshold=0.25,
        text_threshold=0.20,
        detection_index=0,
        depth_window_px=25,
        xarm_ip=cfg.xarm_ip,
        handeye_config_path=cfg.handeye_config_path,
        place_clearance_mm=cfg.place_clearance_mm,
        approach_clearance_mm=cfg.place_approach_clearance_mm,
        lift_after_place_mm=cfg.place_lift_after_mm,
        refine_observation=cfg.container_refine_observation,
        refine_only_when_execute=not cfg.execute,
        refine_observe_z_mm=cfg.container_refine_observe_z_mm,
        refine_wait_s=cfg.container_refine_wait_s,
        execute=False,
        move_speed=cfg.move_speed,
        move_acc=cfg.move_acc,
        gripper_open=None,
        fallback_intrinsics=None if robot.context.transforms is not None else FALLBACK_INTRINSICS,
    )
    return result


def observe(robot: SimpleRobotActions, obj: str) -> SkillResult:
    references = getattr(robot.context, "object_references", None)
    if references is not None and references.resolve(obj):
        return robot._observe_medicine_action(obj)
    if robot._looks_like_container(obj):
        return robot._observe_container_action(obj)
    return robot._observe_medicine_action(obj)


def _observe_container_action(robot: SimpleRobotActions, container: str) -> SkillResult:
    hit = recall_observation(robot, 'container', container)
    if hit is not None:
        return hit
    cached = robot.observations.get(container)
    if cached and cached.get("place_plan"):
        result = robot._move_to_cached_container_observe_pose(container, cached["place_plan"])
        if result.success and memory_for(robot.context.camera) is not None:
            return robot.observe_container(container)
        return result
    return robot._scan_until_found(container, robot.observe_container)


def _observe_medicine_action(robot: SimpleRobotActions, obj: str) -> SkillResult:
    hit = recall_observation(robot, 'medicine', obj)
    if hit is not None:
        return hit
    cached = robot.observations.get(obj)
    if cached and cached.get("selected_instance") and cached.get("grasp_input"):
        move_result = robot._move_to_cached_medicine_observe_pose(obj, cached)
        if not move_result.success:
            return move_result
        return robot._observe_medicine_once(obj)
    return robot._scan_until_found(obj, robot._observe_medicine_once)


@reuse_observation('medicine')
def _observe_medicine_once(robot: SimpleRobotActions, obj: str) -> SkillResult:
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

    cfg = robot.config
    result = FindAndGraspMedicineSkill(robot.context).run(
        query=str(obj),
        camera_name="wrist",
        min_score=cfg.grasp_min_score,
        detection_threshold=cfg.grasp_box_threshold,
        text_threshold=cfg.grasp_text_threshold,
        use_sam_for_selection=False,
        use_mask_for_grasp=False,
        plan_grasp=False,
        execute=False,
        verify_ocr=cfg.verify_ocr,
        ocr_verify_strict_mode=cfg.ocr_verify_strict_mode,
        ocr_verify_fuzzy_threshold=cfg.ocr_verify_fuzzy_threshold,
        fallback_intrinsics=None if robot.context.transforms is not None else FALLBACK_INTRINSICS,
        xarm_ip=cfg.xarm_ip,
        handeye_config_path=cfg.handeye_config_path,
    )
    return result


def _scan_until_found(
    robot: SimpleRobotActions,
    target: str,
    observe_fn: Callable[[str], SkillResult],
) -> SkillResult:
    errors = []
    failure_attempts = []
    last_failure_data = {}
    for offset_y in robot.config.observe_scan_y_offsets_mm:
        move_result = robot._move_to_scan_pose(offset_y)
        if not move_result.success:
            errors.append(move_result.error)
            failure_attempts.append(
                {
                    "scan_offset_y_mm": offset_y,
                    "success": False,
                    "error": move_result.error,
                    "data": move_result.data,
                }
            )
            continue

        if robot.config.observe_wait_s > 0:
            time.sleep(float(robot.config.observe_wait_s))

        result = observe_fn(target)
        if result.success:
            result.data.setdefault("scan_offset_y_mm", offset_y)
            return result
        errors.append(result.error)
        last_failure_data = result.data if isinstance(result.data, dict) else {}
        failure_attempts.append(
            {
                "scan_offset_y_mm": offset_y,
                "success": False,
                "error": result.error,
                "data": last_failure_data,
            }
        )

    return SkillResult(
        success=False,
        data={
            "target": target,
            "scan_offsets_y_mm": list(robot.config.observe_scan_y_offsets_mm),
            "last_failure_data": last_failure_data,
            "failure_attempts": failure_attempts,
        },
        error=f"observe failed after scan poses: {errors}",
    )


def _invalidate_container_cache(robot: SimpleRobotActions, container: str) -> None:
    spec = parse_container_spec(container)
    key = container_key(spec)
    robot.place_plans.pop(key, None)
    robot.observations.pop(container, None)
