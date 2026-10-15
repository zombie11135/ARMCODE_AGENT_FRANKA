"""Gate trace serialization and decision/outcome experience logging.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from core.actions import SimpleAction

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


def _get_gate_logger(robot: SimpleRobotActions):
    """Return the cached gate experience logger (never raises)."""

    if robot._gate_logger is None:
        from nesyro_gate import GateExperienceLogger

        robot._gate_logger = GateExperienceLogger(
            path=str(getattr(robot.config, "gate_experience_log_path", "") or ""),
        )
    return robot._gate_logger


def _log_gate_decision(robot: SimpleRobotActions, unified_result, action: SimpleAction) -> Optional[str]:
    """Append one gate_decision experience record; never raises."""

    from nesyro_gate import decision_record

    logger = robot._get_gate_logger()
    decision_id = logger.new_record_id()
    record = decision_record(
        decision_id=decision_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        episode_id=logger.episode_id,
        gate_mode=unified_result.gate_mode,
        effective_mode=unified_result.effective_mode,
        fell_back=unified_result.fell_back,
        current_step={"op": action.op, "target": action.target},
        gate_input=unified_result.gate_input,
        selected_action=unified_result.action.value,
        action_scores=unified_result.action_scores,
        confidence=unified_result.confidence,
        legacy_decision=unified_result.legacy.decision.value,
        learned_decision=(
            unified_result.learned.action.value if unified_result.learned else None
        ),
        reason=unified_result.reason,
        trace_reference=str(robot.config.nesyro_gate_trace_path or "") or None,
    )
    logger.append(record)
    return decision_id


def log_gate_outcome(
    robot: SimpleRobotActions,
    *,
    primitive_success: Optional[bool] = None,
    skill_success: Optional[bool] = None,
    execution_failure_type: Optional[str] = None,
    task_progress_before: Optional[float] = None,
    task_progress_after: Optional[float] = None,
    visual_change: Optional[Dict[str, Any]] = None,
    trace_reference: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> bool:
    """Append one gate_outcome record for the last gate decision.

    ``visual_change`` is an optional slot; the current pipeline does not
    compute occlusion deltas, so it stays None until a later phase.
    Never raises; returns whether the line was written.
    """

    from nesyro_gate import outcome_record

    logger = robot._get_gate_logger()
    decision_id = decision_id or robot._last_gate_decision_id
    if decision_id is None:
        return False
    record = outcome_record(
        decision_id=decision_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        episode_id=logger.episode_id,
        primitive_success=primitive_success,
        skill_success=skill_success,
        task_progress_before=task_progress_before,
        task_progress_after=task_progress_after,
        execution_failure_type=execution_failure_type,
        visual_change=visual_change,
        trace_reference=trace_reference,
    )
    return logger.append(record)


def _gate_record(
    robot: SimpleRobotActions,
    action: SimpleAction,
    current_holding: Optional[str],
    gate_result,
    scene_memory,
) -> Dict[str, Any]:
    """Build the per-attempt gate record (legacy fields + unified view).

    ``gate_result`` is a UnifiedGateResult. The original field names
    (decision/confidence/csc/lc/...) keep their legacy values so legacy
    traces stay byte-comparable with the pre-phase-1 format; the new
    unified keys are appended additively.
    """

    objects = {}
    for object_id, obj in scene_memory.objects.items():
        objects[str(object_id)] = {
            "name": obj.name,
            "object_type": obj.object_type,
            "visible": obj.visible,
            "detector_confidence": round(float(obj.detector_confidence), 4),
            "ocr_match_score": round(float(obj.ocr_match_score), 4),
            "identity_source": obj.identity_source,
            "identity_verified": obj.identity_verified,
            "identity_match_score": round(float(obj.identity_match_score), 4),
            "vlm_match_score": round(float(obj.vlm_match_score), 4),
            "reachable": obj.reachable,
            "graspable": obj.graspable,
            "occlusion_level": round(float(obj.occlusion_level), 4),
            "in_gripper": obj.in_gripper,
            "free_space": obj.free_space,
            "placeable": obj.placeable,
        }

    legacy = gate_result.legacy
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action": action.__dict__,
        "current_holding": current_holding,
        "decision": legacy.decision.value,
        "confidence": round(float(legacy.confidence), 4),
        "csc": round(float(legacy.csc), 4),
        "lc": round(float(legacy.lc), 4),
        "reason": legacy.reason,
        "failed_conditions": list(legacy.failed_conditions),
        "suggested_probe": legacy.suggested_probe,
        "suggested_constraint": legacy.suggested_constraint,
        "probe_counts": dict(scene_memory.probe_counts),
        "scene_objects": objects,
        # Phase-1 unified view (additive keys only).
        "gate_action": gate_result.action.value,
        "gate_mode": gate_result.gate_mode,
        "effective_mode": gate_result.effective_mode,
        "fell_back": gate_result.fell_back,
        "action_scores": gate_result.action_scores,
        "learned": gate_result.learned.to_dict() if gate_result.learned else None,
    }
    return record


def _append_gate_trace(robot: SimpleRobotActions, record: Dict[str, Any]) -> None:
    path_text = str(robot.config.nesyro_gate_trace_path or "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(robot._safe_json_for_gate(record), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[NeSyRoGate][WARN] 写入 gate trace 失败：{e}")


def _safe_json_for_gate(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_for_gate(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_for_gate(v) for v in value]
    if hasattr(value, "tolist"):
        return _safe_json_for_gate(value.tolist())
    return str(value)
