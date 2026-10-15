"""UnifiedNeSyRoGate: mode-aware facade over the legacy and learnable gates.

Mode semantics:

- ``legacy``  : only the original CSCxLC gate runs and drives the robot.
- ``learned`` : the MLP gate drives the robot; a missing checkpoint (or any
  predict failure) automatically falls back to legacy and is recorded as
  ``effective_mode="learned_fallback_legacy"``.
- ``shadow``  : legacy drives the robot; the MLP gate is evaluated for
  logging only and its output never changes robot behavior.

The legacy gate is evaluated in every mode (it is a cheap rule evaluation
with no perception calls) so experience logs always carry the legacy
decision for comparison, exactly as required by the phase-1 logging schema.

Note on side effects: legacy evaluation keeps its original side effects
(probe-count increment and SceneMemory write-backs), also in learned/shadow
mode. This is intentional in phase 1: the legacy decision must remain fully
comparable across modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from nesyro_gate.interface import GateInput, GateMode, GateOutput, ProbeAction
from nesyro_gate.legacy_gate import LegacyNeSyRoGate, legacy_action_for, legacy_action_scores
from nesyro_gate.learnable_gate import LearnableNeSyRoGate
from nesyro_gate.types import GateResult, SceneMemory, Step, TaskSpec


@dataclass
class UnifiedGateResult:
    """One gate decision with legacy + learnable views.

    ``legacy`` is always computed so experience logs can compare decisions
    across gate modes. ``learned`` is present only when the learnable gate
    ran (learned/shadow mode with a valid checkpoint). ``gate_input`` is the
    enriched feature snapshot that was actually consumed and logged.
    """

    action: ProbeAction
    confidence: float
    action_scores: Optional[Dict[str, float]]
    reason: str
    debug_info: Dict[str, Any]
    legacy: GateResult
    learned: Optional[GateOutput]
    gate_mode: str
    effective_mode: str
    fell_back: bool
    gate_input: GateInput = field(default_factory=GateInput)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "action_scores": self.action_scores,
            "reason": self.reason,
            "debug_info": self.debug_info,
            "gate_mode": self.gate_mode,
            "effective_mode": self.effective_mode,
            "fell_back": self.fell_back,
            "legacy_decision": self.legacy.decision.value,
            "legacy_confidence": self.legacy.confidence,
            "learned": self.learned.to_dict() if self.learned else None,
        }


def _parse_mode(mode: str) -> str:
    value = str(mode or "legacy").strip().lower()
    if value not in {GateMode.LEGACY.value, GateMode.LEARNED.value, GateMode.SHADOW.value}:
        raise ValueError(f"unknown gate_mode: {mode!r}")
    return value


class UnifiedNeSyRoGate:
    """Dispatch between legacy / learned / shadow gate modes."""

    def __init__(
        self,
        mode: str = "legacy",
        checkpoint_path: str = "",
        threshold: float = 0.7,
        max_probe: int = 3,
    ):
        self.mode = GateMode(_parse_mode(mode))
        self.checkpoint_path = str(checkpoint_path or "")
        self.legacy_gate = LegacyNeSyRoGate(threshold=threshold, max_probe=max_probe)
        self.learnable_gate = LearnableNeSyRoGate(checkpoint_path=self.checkpoint_path or None)

    @property
    def learnable_available(self) -> bool:
        """Whether the learnable gate can run in this process."""

        return self.learnable_gate.available

    @staticmethod
    def _enrich_gate_input(
        gate_input: Optional[GateInput],
        legacy_result: GateResult,
    ) -> GateInput:
        """Fill GateInput fields from the just-computed legacy result.

        The legacy gate computes CSC/LC internally; enriching the input
        afterwards keeps one canonical snapshot for logging and for the
        learned feature vector without recomputing anything.
        """

        if gate_input is None:
            gate_input = GateInput()
        if gate_input.target_confidence is None:
            gate_input.target_confidence = float(legacy_result.csc)
        prerequisite = dict(gate_input.prerequisite_state or {})
        prerequisite.setdefault("lc", float(legacy_result.lc))
        prerequisite.setdefault("failed_conditions", list(legacy_result.failed_conditions))
        gate_input.prerequisite_state = prerequisite
        return gate_input

    def decide(
        self,
        step: Step,
        scene_memory: SceneMemory,
        task_spec: TaskSpec,
        gate_input: Optional[GateInput] = None,
    ) -> UnifiedGateResult:
        legacy_result = self.legacy_gate.decide(
            step=step,
            scene_memory=scene_memory,
            task_spec=task_spec,
            gate_input=gate_input,
        )
        gate_input = self._enrich_gate_input(gate_input, legacy_result)

        learned_output: Optional[GateOutput] = None
        fell_back = False
        effective_mode = self.mode.value

        if self.mode in (GateMode.LEARNED, GateMode.SHADOW):
            if self.learnable_gate.available:
                try:
                    learned_output = self.learnable_gate.predict(gate_input)
                except Exception as exc:
                    learned_output = None
                    fell_back = True
                    if self.mode == GateMode.LEARNED:
                        effective_mode = "learned_fallback_legacy"
                    gate_input.extra.setdefault("learned_error", str(exc))
            else:
                fell_back = True
                if self.mode == GateMode.LEARNED:
                    effective_mode = "learned_fallback_legacy"

        if self.mode == GateMode.LEARNED and learned_output is not None:
            action = learned_output.action
            confidence = learned_output.confidence
            action_scores = learned_output.action_scores
            reason = learned_output.reason
        else:
            action = legacy_action_for(legacy_result.decision)
            confidence = float(legacy_result.confidence)
            action_scores = legacy_action_scores(legacy_result.decision)
            reason = legacy_result.reason

        return UnifiedGateResult(
            action=action,
            confidence=confidence,
            action_scores=action_scores,
            reason=reason,
            debug_info={
                "legacy": {
                    "csc": legacy_result.csc,
                    "lc": legacy_result.lc,
                    "failed_conditions": list(legacy_result.failed_conditions),
                    "suggested_probe": legacy_result.suggested_probe,
                    "suggested_constraint": legacy_result.suggested_constraint,
                },
                "learnable_available": self.learnable_gate.available,
                "learnable_load_error": self.learnable_gate.load_error,
            },
            legacy=legacy_result,
            learned=learned_output,
            gate_mode=self.mode.value,
            effective_mode=effective_mode,
            fell_back=fell_back,
            gate_input=gate_input,
        )
