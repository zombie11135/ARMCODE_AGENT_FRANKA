"""Cached placement execution and optional placement slot offsets.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.skill_base import SkillResult
from skill_lib.actions.targets import container_key, parse_container_spec, placement_slot_offset_mm
from skill_lib.actions.observation_cache import invalidate_after_manipulation
from runtime.observation_memory import scene_mutation

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


@invalidate_after_manipulation('place')
def place(robot: SimpleRobotActions, container: str) -> SkillResult:
    spec = parse_container_spec(container)
    key = container_key(spec)
    if key not in robot.place_plans:
        result = robot.observe_container(container)
        if not result.success:
            return result

    cfg = robot.config
    plan = robot._offset_place_plan(robot.place_plans[key], robot.place_counts.get(key, 0))
    robot.place_counts[key] = robot.place_counts.get(key, 0) + 1

    if not cfg.execute:
        return SkillResult(success=True, data={"place_plan": plan, "dry_run": True})

    execution = robot._execute_cached_place(plan)
    return SkillResult(
        success=bool(execution.get("success")),
        data={"place_plan": plan, "execution": execution},
        error=None if execution.get("success") else execution.get("error"),
    )


@scene_mutation('place')
def _execute_cached_place(robot: SimpleRobotActions, place_plan: Dict[str, Any]) -> Dict[str, Any]:
    for label, pose in [
        ("approach", place_plan["approach_pose_mmrad"]),
        ("place", place_plan["place_pose_mmrad"]),
    ]:
        if not robot._move_xarm(pose):
            return {"success": False, "error": f"move {label} pose failed"}

    if robot.gripper_open and not robot.gripper_open():
        return {"success": False, "error": "gripper open failed"}

    if not robot._move_xarm(place_plan["lift_pose_mmrad"]):
        return {"success": False, "error": "move lift pose failed"}

    return {"success": True}


def _offset_place_plan(robot: SimpleRobotActions, place_plan: Dict[str, Any], slot_index: int) -> Dict[str, Any]:
    cfg = robot.config
    if not cfg.enable_place_slot_offsets:
        return dict(place_plan)

    offset_y = placement_slot_offset_mm(slot_index, cfg.place_slot_spacing_mm)
    shifted = dict(place_plan)
    for key in ["approach_pose_mmrad", "place_pose_mmrad", "lift_pose_mmrad"]:
        pose = list(shifted[key])
        pose[1] += offset_y
        shifted[key] = pose
    shifted["place_slot_index"] = int(slot_index)
    shifted["place_slot_offset_y_mm"] = float(offset_y)
    return shifted
