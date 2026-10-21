"""Gate evaluation, scene adaptation and observe probes.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.actions import SimpleAction
from core.skill_base import SkillResult
from skill_lib.actions.observation_cache import require_fresh_observation

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


def gate_before_action(
    robot: SimpleRobotActions,
    action: SimpleAction,
    current_holding: Optional[str] = None,
    target_plate: Optional[str] = None,
    previous_action_result: Optional[float] = None,
    step_index: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> SkillResult:
    """Run NeSyRo Gate before an action and perform safe observe probes.

    In ``gate_mode="legacy"`` (the default) this keeps the exact original
    CSCxLC decision flow. ``learned`` / ``shadow`` are handled through
    :class:`nesyro_gate.unified_gate.UnifiedNeSyRoGate`; every decision is
    also recorded by the gate experience logger (never raises).
    """

    if not robot.config.enable_nesyro_gate:
        return SkillResult(
            success=True,
            data={"gate": {"enabled": False, "action": action.__dict__}},
        )

    if action.op == "place" and not current_holding:
        return SkillResult(
            success=False,
            data={
                "failure_type": "nesyro_gate_empty_gripper",
                "action": action.__dict__,
                "current_holding": current_holding,
            },
            error=f"NeSyRo Gate blocked place({action.target}): gripper is not holding an object.",
        )

    if action.op == "grasp" and not robot._has_cached_observation(action.target):
        probe_result = robot._run_gate_probe(
            action=action,
            current_holding=current_holding,
            suggested_probe="observe_target_region",
            reason="missing cached medicine observation before grasp",
        )
        if not probe_result.success:
            return probe_result

    if action.op == "grasp":
        vlm_result = robot._run_vlm_occlusion_gate_if_enabled(action.target)
        if not vlm_result.success:
            return vlm_result

    gate_attempts: List[Dict[str, Any]] = []
    occlusion_recovery_attempts: List[Dict[str, Any]] = []
    max_attempts = max(
        1,
        int(robot.config.nesyro_gate_max_probe or 0)
        + int(robot.config.occlusion_recovery_max_attempts or 0)
        + 1,
    )

    from nesyro_gate import ProbeAction

    for _ in range(max_attempts):
        scene_memory = robot.build_scene_memory(current_holding=current_holding)
        step = robot._action_to_gate_step(action, current_holding=current_holding)
        task_spec = robot._action_to_gate_task_spec(
            action=action,
            current_holding=current_holding,
            target_plate=target_plate,
        )
        gate_input = robot._build_gate_input(
            action=action,
            current_holding=current_holding,
            scene_memory=scene_memory,
            previous_action_result=previous_action_result,
            step_index=step_index,
            total_steps=total_steps,
        )

        result = robot._get_unified_gate().decide(
            step=step,
            scene_memory=scene_memory,
            task_spec=task_spec,
            gate_input=gate_input,
        )
        robot.gate_probe_counts.update(scene_memory.probe_counts)

        record = robot._gate_record(
            action=action,
            current_holding=current_holding,
            gate_result=result,
            scene_memory=scene_memory,
        )
        gate_attempts.append(record)
        robot._append_gate_trace(record)
        robot._last_gate_decision_id = robot._log_gate_decision(
            unified_result=result,
            action=action,
        )

        mode_suffix = "" if result.gate_mode == "legacy" else f" mode={result.effective_mode}"
        print(
            "[NeSyRoGate] "
            f"{action.op}({action.target}) -> {result.action.value} "
            f"conf={result.confidence:.3f} csc={result.legacy.csc:.3f} lc={result.legacy.lc:.3f} "
            f"reason={result.reason}{mode_suffix}"
        )

        if result.action == ProbeAction.EXECUTE:
            return SkillResult(
                success=True,
                data={
                    "gate": record,
                    "gate_attempts": gate_attempts,
                    "occlusion_recovery_attempts": occlusion_recovery_attempts,
                },
            )

        if result.action == ProbeAction.VISUAL_PROBE:
            probe_result = robot._run_gate_probe(
                action=action,
                current_holding=current_holding,
                suggested_probe=result.legacy.suggested_probe,
                reason=result.reason,
            )
            if not probe_result.success:
                probe_result.data.setdefault("gate_attempts", gate_attempts)
                return probe_result
            if action.op == "grasp":
                vlm_result = robot._run_vlm_occlusion_gate_if_enabled(action.target)
                if not vlm_result.success:
                    vlm_result.data.setdefault("gate_attempts", gate_attempts)
                    return vlm_result
            continue

        if result.action == ProbeAction.MANIPULATION_PROBE:
            # Phase-1: only reachable from a learned checkpoint. The legacy
            # flow never produces this action. Degrade to the existing
            # occlusion-recovery path when enabled, otherwise to a safe
            # visual probe.
            if robot._should_run_occlusion_recovery(
                action,
                result.legacy.failed_conditions,
                require_occlusion=False,
            ):
                recovery_result = robot._run_occlusion_recovery(action.target)
                recovery_record = {
                    "attempt": robot.occlusion_recovery_counts.get(action.target, 0),
                    "success": recovery_result.success,
                    "error": recovery_result.error,
                    "data": recovery_result.data,
                }
                occlusion_recovery_attempts.append(recovery_record)
                if recovery_result.success:
                    refresh_result = robot._refresh_after_occlusion_recovery(action.target)
                    recovery_record["refresh"] = {
                        "success": refresh_result.success,
                        "error": refresh_result.error,
                        "data": refresh_result.data,
                    }
                    vlm_result = robot._run_vlm_occlusion_gate_if_enabled(action.target)
                    if not vlm_result.success:
                        vlm_result.data.setdefault("gate_attempts", gate_attempts)
                        vlm_result.data.setdefault(
                            "occlusion_recovery_attempts",
                            occlusion_recovery_attempts,
                        )
                        return vlm_result
                    continue
            probe_result = robot._run_gate_probe(
                action=action,
                current_holding=current_holding,
                suggested_probe="observe_target_region",
                reason=result.reason,
            )
            if not probe_result.success:
                probe_result.data.setdefault("gate_attempts", gate_attempts)
                return probe_result
            if action.op == "grasp":
                vlm_result = robot._run_vlm_occlusion_gate_if_enabled(action.target)
                if not vlm_result.success:
                    vlm_result.data.setdefault("gate_attempts", gate_attempts)
                    return vlm_result
            continue

        if robot._should_run_occlusion_recovery(action, result.legacy.failed_conditions):
            recovery_result = robot._run_occlusion_recovery(action.target)
            recovery_record = {
                "attempt": robot.occlusion_recovery_counts.get(action.target, 0),
                "success": recovery_result.success,
                "error": recovery_result.error,
                "data": recovery_result.data,
            }
            occlusion_recovery_attempts.append(recovery_record)
            if recovery_result.success:
                refresh_result = robot._refresh_after_occlusion_recovery(action.target)
                recovery_record["refresh"] = {
                    "success": refresh_result.success,
                    "error": refresh_result.error,
                    "data": refresh_result.data,
                }
                vlm_result = robot._run_vlm_occlusion_gate_if_enabled(action.target)
                if not vlm_result.success:
                    vlm_result.data.setdefault("gate_attempts", gate_attempts)
                    vlm_result.data.setdefault(
                        "occlusion_recovery_attempts",
                        occlusion_recovery_attempts,
                    )
                    return vlm_result
                continue

        return SkillResult(
            success=False,
            data={
                "failure_type": robot._gate_failure_type(result.legacy.decision.value),
                "action": action.__dict__,
                "gate": record,
                "gate_attempts": gate_attempts,
                "occlusion_recovery_attempts": occlusion_recovery_attempts,
                "failed_conditions": result.legacy.failed_conditions,
                "suggested_constraint": result.legacy.suggested_constraint,
            },
            error=(
                f"NeSyRo Gate blocked {action.op}({action.target}): "
                f"{result.reason}; conditions={result.legacy.failed_conditions}"
            ),
        )

    return SkillResult(
        success=False,
        data={
            "failure_type": "nesyro_gate_probe_exhausted",
            "action": action.__dict__,
            "gate_attempts": gate_attempts,
            "occlusion_recovery_attempts": occlusion_recovery_attempts,
        },
        error=f"NeSyRo Gate exhausted probes for {action.op}({action.target}).",
    )


def build_scene_memory(robot: SimpleRobotActions, current_holding: Optional[str] = None):
    """Build a SceneMemory snapshot from cached observations and plans."""

    from nesyro_gate import build_scene_memory_from_simple_robot

    return build_scene_memory_from_simple_robot(
        observations=robot.observations,
        place_plans=robot.place_plans,
        place_counts=robot.place_counts,
        gripper_holding=current_holding,
        probe_counts=robot.gate_probe_counts,
    )


def _action_to_gate_step(
    robot: SimpleRobotActions,
    action: SimpleAction,
    current_holding: Optional[str],
):
    from nesyro_gate import SkillName, Step

    if action.op == "observe":
        return Step(name=SkillName.OBSERVE, args={"query": action.target})
    if action.op == "grasp":
        return Step(name=SkillName.GRASP, args={"object": action.target})
    if action.op == "place":
        return Step(
            name=SkillName.PLACE,
            args={"object": current_holding, "plate": action.target},
        )
    raise ValueError(f"unknown action op for NeSyRo Gate: {action.op}")


def _action_to_gate_task_spec(
    robot: SimpleRobotActions,
    action: SimpleAction,
    current_holding: Optional[str],
    target_plate: Optional[str],
):
    from nesyro_gate import build_task_spec_for_pick_place

    if action.op == "place":
        return build_task_spec_for_pick_place(
            target_object_name=current_holding or "target medicine",
            target_plate_name=action.target,
            target_object_id=current_holding,
            target_plate_id=action.target,
        )
    return build_task_spec_for_pick_place(
        target_object_name=action.target,
        target_plate_name=target_plate or "target container",
        target_object_id=action.target if action.op == "grasp" else None,
        target_plate_id=target_plate,
    )


def _get_unified_gate(robot: SimpleRobotActions):
    """Return the cached UnifiedNeSyRoGate for the current config.

    Feature-reuse guard: the facade (and any loaded learned checkpoint)
    is built once per (mode, checkpoint, threshold, max_probe) key and
    reused across attempts, so learned mode never re-loads the model or
    re-runs perception.
    """

    key = (
        str(getattr(robot.config, "gate_mode", "legacy") or "legacy"),
        str(getattr(robot.config, "learned_gate_checkpoint_path", "") or ""),
        float(getattr(robot.config, "nesyro_gate_threshold", 0.7)),
        int(getattr(robot.config, "nesyro_gate_max_probe", 3)),
    )
    if robot._unified_gate is None or robot._unified_gate_key != key:
        from nesyro_gate import UnifiedNeSyRoGate

        robot._unified_gate = UnifiedNeSyRoGate(
            mode=key[0],
            checkpoint_path=key[1],
            threshold=key[2],
            max_probe=key[3],
        )
        robot._unified_gate_key = key
    return robot._unified_gate


def _build_gate_input(
    robot: SimpleRobotActions,
    action: SimpleAction,
    current_holding: Optional[str],
    scene_memory,
    previous_action_result: Optional[float],
    step_index: Optional[int],
    total_steps: Optional[int],
):
    """Build a GateInput purely from already-available runtime state.

    No perception or VLM calls happen here: everything is read from the
    SceneMemory snapshot and the runtime bookkeeping.
    """

    from nesyro_gate import GateInput

    obj = scene_memory.get_object(action.target)
    memory_features: Dict[str, Any] = {}
    visibility: Optional[float] = None
    occlusion_score: Optional[float] = None
    if obj is not None:
        memory_features = {
            "detector_confidence": float(obj.detector_confidence),
            "ocr_match_score": float(obj.ocr_match_score),
            "vlm_match_score": float(obj.vlm_match_score),
            "reachable": 1.0 if obj.reachable else 0.0,
            "graspable": 1.0 if obj.graspable else 0.0,
            "in_gripper": 1.0 if obj.in_gripper else 0.0,
            "free_space": 1.0 if obj.free_space else 0.0,
            "placeable": 1.0 if obj.placeable else 0.0,
            "visible": 1.0 if obj.visible else 0.0,
        }
        visibility = 1.0 if obj.visible else 0.0
        occlusion_score = float(obj.occlusion_level)

    probe_counts = dict(scene_memory.probe_counts or {})
    step_key = f"{action.op}({action.target})"
    total = max(1, int(total_steps or 1))
    index = int(step_index) if step_index is not None else None

    return GateInput(
        current_step={"op": action.op, "target": action.target},
        task_stage={
            "step_index": index,
            "total_steps": total_steps,
            "task_progress": (index / total) if index is not None else None,
        },
        target_confidence=None,  # enriched from the legacy CSC after decide
        target_visibility=visibility,
        occlusion_score=occlusion_score,
        pose_confidence=None,  # not provided by the current pipeline
        blocker_confidence=None,  # not provided before occlusion recovery
        blocker_graspability=None,
        blocker_target_relation=None,
        prerequisite_state={},  # enriched from the legacy LC after decide
        previous_action_result=previous_action_result,
        visual_probe_count=int(probe_counts.get(step_key, 0)),
        manipulation_probe_count=int(robot.occlusion_recovery_counts.get(action.target, 0)),
        memory_features=memory_features,
        task_progress=(index / total) if index is not None else None,
        extra={"current_holding": current_holding},
    )


@require_fresh_observation('gate_probe')
def _run_gate_probe(
    robot: SimpleRobotActions,
    action: SimpleAction,
    current_holding: Optional[str],
    suggested_probe: Optional[str],
    reason: str,
) -> SkillResult:
    """Execute the closest existing observe skill for a gate probe request."""

    del current_holding

    probe_target = action.target
    if action.op == "place":
        robot._invalidate_container_cache(action.target)

    print(
        "[NeSyRoGate] probe "
        f"{suggested_probe or 'observe_global_table'} for {action.op}({action.target}): {reason}"
    )
    result = robot.observe(probe_target)
    if result.success:
        return SkillResult(
            success=True,
            data={
                "gate_probe": suggested_probe,
                "probe_target": probe_target,
                "probe_result": result.data,
            },
        )
    if action.op == "grasp":
        robot.observations[probe_target] = {
            "kind": "medicine",
            "target": probe_target,
            "observation_failed": True,
            "error": result.error,
            "result": result.data if isinstance(result.data, dict) else {},
        }
        return SkillResult(
            success=True,
            data={
                "gate_probe": suggested_probe,
                "probe_target": probe_target,
                "probe_failed_conservative_scene_memory": True,
                "probe_error": result.error,
                "probe_data": result.data,
            },
        )
    return SkillResult(
        success=False,
        data={
            "failure_type": "nesyro_gate_probe_failed",
            "gate_probe": suggested_probe,
            "probe_target": probe_target,
            "probe_error": result.error,
            "probe_data": result.data,
        },
        error=(
            f"NeSyRo Gate probe {suggested_probe or 'observe'} failed "
            f"for {probe_target}: {result.error}"
        ),
    )


def _has_cached_observation(robot: SimpleRobotActions, target: str) -> bool:
    cached = robot.observations.get(target)
    if not isinstance(cached, dict):
        return False
    if cached.get("kind") == "medicine":
        return bool(cached.get("selected_instance") and cached.get("grasp_input"))
    if cached.get("kind") == "container":
        return bool(cached.get("place_plan"))
    return False


def _gate_failure_type(decision: str) -> str:
    if decision == "LOCAL_REPAIR":
        return "nesyro_gate_local_repair"
    if decision == "REPLAN":
        return "nesyro_gate_replan"
    if decision == "ABORT":
        return "nesyro_gate_abort"
    return "nesyro_gate_blocked"
