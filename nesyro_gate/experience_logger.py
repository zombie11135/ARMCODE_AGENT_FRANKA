"""Append-only JSONL logger for gate experience records.

The logger NEVER raises: a logging failure (bad path, disk error, ...) is
swallowed and reported as a False return value, so logging can never block
or fail the robot main flow.

Two record kinds share one file:

- ``gate_decision``: one per gate evaluation (features, selected action,
  scores, legacy decision, learned decision, trace reference).
- ``gate_outcome``: one per executed action, referencing the last decision
  via ``decision_id`` (primitive/skill success, failure type, progress).

Both only carry information the runtime already has; no extra VLM or
perception calls are made for logging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class GateExperienceLogger:
    """Append JSON lines to a single experience file. Never raises."""

    def __init__(self, path: Optional[str] = None, episode_id: Optional[str] = None):
        self.path = Path(path) if path else None
        self.episode_id = episode_id or time.strftime("%Y%m%dT%H%M%S")
        self._record_counter = 0

    def new_record_id(self, prefix: str = "gate") -> str:
        """Return a unique id for one gate decision within this episode."""

        self._record_counter += 1
        return f"{prefix}-{self.episode_id}-{self._record_counter:04d}"

    def append(self, record: Dict[str, Any]) -> bool:
        """Append one JSON line. Returns True on success, False on failure."""

        if not self.path:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return True
        except Exception:
            return False


def decision_record(
    decision_id: str,
    timestamp: str,
    episode_id: str,
    gate_mode: str,
    effective_mode: str,
    fell_back: bool,
    current_step: Dict[str, Any],
    gate_input,
    selected_action: str,
    action_scores: Optional[Dict[str, float]],
    confidence: float,
    legacy_decision: str,
    learned_decision: Optional[str],
    reason: str,
    trace_reference: Optional[str],
) -> Dict[str, Any]:
    """Build one ``gate_decision`` experience record."""

    step = current_step if isinstance(current_step, dict) else dict(current_step or {})
    features = gate_input.to_dict() if hasattr(gate_input, "to_dict") else gate_input
    return {
        "event": "gate_decision",
        "decision_id": decision_id,
        "timestamp": timestamp,
        "episode_id": episode_id,
        "gate_mode": gate_mode,
        "effective_mode": effective_mode,
        "fell_back": fell_back,
        "step_id": step.get("step_id"),
        "current_step": step,
        "gate_input_features": features,
        "selected_action": selected_action,
        "action_scores": action_scores,
        "confidence": confidence,
        "legacy_decision": legacy_decision,
        "learned_decision": learned_decision,
        "reason": reason,
        "trace_reference": trace_reference,
    }


def outcome_record(
    decision_id: str,
    timestamp: str,
    episode_id: str,
    primitive_success: Optional[bool],
    skill_success: Optional[bool],
    task_progress_before: Optional[float],
    task_progress_after: Optional[float],
    execution_failure_type: Optional[str],
    visual_change: Optional[Dict[str, Any]],
    trace_reference: Optional[str],
) -> Dict[str, Any]:
    """Build one ``gate_outcome`` experience record.

    ``visual_change`` is kept as an optional slot; the current pipeline does
    not compute occlusion deltas, so it stays None until a later phase.
    """

    return {
        "event": "gate_outcome",
        "decision_id": decision_id,
        "timestamp": timestamp,
        "episode_id": episode_id,
        "primitive_success": primitive_success,
        "skill_success": skill_success,
        "task_progress_before": task_progress_before,
        "task_progress_after": task_progress_after,
        "execution_failure_type": execution_failure_type,
        "visual_change": visual_change,
        "trace_reference": trace_reference,
    }
