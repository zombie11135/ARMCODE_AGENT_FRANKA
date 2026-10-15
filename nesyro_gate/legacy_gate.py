"""LegacyNeSyRoGate: thin wrapper that keeps the original CSCxLC rule gate.

All computation, thresholds and side effects (probe counts, SceneMemory
write-backs) stay exactly as in nesyro_gate.gate.nesyro_gate(); this wrapper
only translates the result into the unified gate interface.

The wrapper is intentionally free of any new logic so that the legacy
equivalence tests can compare its output field-by-field with the original
function.
"""

from __future__ import annotations

from typing import Dict, Optional

from nesyro_gate.gate import nesyro_gate
from nesyro_gate.interface import GateInput, ProbeAction
from nesyro_gate.types import GateDecision, GateResult, SceneMemory, Step, TaskSpec

# Legacy decisions keep their current runtime meaning:
# - PROBE_REQUIRED is always a safe visual observe probe;
# - LOCAL_REPAIR / REPLAN / ABORT currently behave as replan-triggering
#   failures in the runtime (see SimpleRobotActions.gate_before_action).
LEGACY_DECISION_TO_ACTION = {
    GateDecision.EXECUTE: ProbeAction.EXECUTE,
    GateDecision.PROBE_REQUIRED: ProbeAction.VISUAL_PROBE,
    GateDecision.LOCAL_REPAIR: ProbeAction.REPLAN,
    GateDecision.REPLAN: ProbeAction.REPLAN,
    GateDecision.ABORT: ProbeAction.REPLAN,
}


def legacy_action_for(decision: GateDecision) -> ProbeAction:
    """Map a legacy GateDecision to the unified ProbeAction."""

    return LEGACY_DECISION_TO_ACTION[decision]


def legacy_action_scores(decision: GateDecision) -> Dict[str, float]:
    """Deterministic one-hot scores for a legacy decision.

    The legacy gate has no soft scores; a one-hot vector keeps experience
    logs comparable with learned-gate probability vectors.
    """

    action = legacy_action_for(decision)
    return {member.value: (1.0 if member == action else 0.0) for member in ProbeAction}


class LegacyNeSyRoGate:
    """Original rule gate exposed behind the unified interface.

    ``threshold`` and ``max_probe`` are forwarded verbatim to the unchanged
    :func:`nesyro_gate.nesyro_gate` function.
    """

    name = "legacy"

    def __init__(self, threshold: float = 0.7, max_probe: int = 3):
        self.threshold = float(threshold)
        self.max_probe = int(max_probe)

    def decide(
        self,
        step: Step,
        scene_memory: SceneMemory,
        task_spec: TaskSpec,
        gate_input: Optional[GateInput] = None,
    ) -> GateResult:
        """Run the unchanged legacy gate and return its raw GateResult.

        ``gate_input`` is accepted for interface compatibility and ignored:
        the legacy gate only consumes step / scene_memory / task_spec.
        """

        del gate_input
        return nesyro_gate(
            step=step,
            scene_memory=scene_memory,
            task_spec=task_spec,
            threshold=self.threshold,
            max_probe=self.max_probe,
        )
