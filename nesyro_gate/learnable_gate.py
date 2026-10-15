"""LearnableNeSyRoGate: tiny MLP over low-cost runtime features.

Phase-1 skeleton. The model is deliberately small (2 hidden layers, 64-128
units), runs on CPU, and consumes only features already available in the
runtime / SceneMemory — it never triggers extra perception or VLM calls.

torch is an existing indirect dependency of this project (GroundingDINO /
SAM3). It is imported lazily so importing this module never requires torch;
if torch or the checkpoint is missing, the gate reports ``available=False``
and callers fall back to the legacy gate.

Checkpoint format: a plain torch state_dict (dict of tensors) compatible
with :func:`build_mlp`. Saving a checkpoint for later training is:

    model = build_mlp(len(GATE_FEATURE_NAMES), hidden_units)
    torch.save(model.state_dict(), "logs/gate_experience/learned_gate.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nesyro_gate.interface import GateInput, GateOutput, ProbeAction

# Fixed feature order shared by feature extraction, checkpoint format and
# (future) training. Missing values are mapped to 0.0.
GATE_FEATURE_NAMES: List[str] = [
    "target_confidence",
    "target_visibility",
    "occlusion_score",
    "pose_confidence",
    "blocker_confidence",
    "blocker_graspability",
    "blocker_target_relation",
    "prerequisite_lc",
    "previous_action_result",
    "visual_probe_count",
    "manipulation_probe_count",
    "task_progress",
    "detector_confidence",
    "ocr_match_score",
    "vlm_match_score",
    "reachable",
    "graspable",
    "in_gripper",
    "free_space",
    "placeable",
    "is_observe",
    "is_grasp",
    "is_place",
]

ACTION_ORDER: Tuple[ProbeAction, ...] = (
    ProbeAction.EXECUTE,
    ProbeAction.VISUAL_PROBE,
    ProbeAction.MANIPULATION_PROBE,
    ProbeAction.REPLAN,
)

DEFAULT_HIDDEN_UNITS: Tuple[int, int] = (128, 64)


def build_mlp(input_dim: int, hidden_units: Sequence[int] = DEFAULT_HIDDEN_UNITS):
    """Build the tiny MLP: Linear -> ReLU -> Linear -> ReLU -> 4-way logits."""

    import torch.nn as nn

    layers: List[Any] = []
    previous = int(input_dim)
    for units in hidden_units:
        layers.append(nn.Linear(previous, int(units)))
        layers.append(nn.ReLU())
        previous = int(units)
    layers.append(nn.Linear(previous, len(ACTION_ORDER)))
    return nn.Sequential(*layers)


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gate_input_to_feature_vector(gate_input: Optional[GateInput]) -> List[float]:
    """Convert a GateInput into the fixed-order numeric feature vector.

    Booleans become 0.0/1.0; unknown numeric values become 0.0. The same
    ordering is used by :data:`GATE_FEATURE_NAMES` so future training reads
    the identical schema.
    """

    if gate_input is None:
        gate_input = GateInput()
    memory = dict(gate_input.memory_features or {})
    step = dict(gate_input.current_step or {})
    skill = str(step.get("skill") or step.get("name") or "").lower()
    prerequisite = dict(gate_input.prerequisite_state or {})

    values: Dict[str, Any] = {
        "target_confidence": gate_input.target_confidence,
        "target_visibility": gate_input.target_visibility,
        "occlusion_score": gate_input.occlusion_score,
        "pose_confidence": gate_input.pose_confidence,
        "blocker_confidence": gate_input.blocker_confidence,
        "blocker_graspability": gate_input.blocker_graspability,
        "blocker_target_relation": gate_input.blocker_target_relation,
        "prerequisite_lc": prerequisite.get("lc"),
        "previous_action_result": gate_input.previous_action_result,
        "visual_probe_count": gate_input.visual_probe_count,
        "manipulation_probe_count": gate_input.manipulation_probe_count,
        "task_progress": gate_input.task_progress,
        "detector_confidence": memory.get("detector_confidence"),
        "ocr_match_score": memory.get("ocr_match_score"),
        "vlm_match_score": memory.get("vlm_match_score"),
        "reachable": memory.get("reachable"),
        "graspable": memory.get("graspable"),
        "in_gripper": memory.get("in_gripper"),
        "free_space": memory.get("free_space"),
        "placeable": memory.get("placeable"),
        "is_observe": 1.0 if skill == "observe" else 0.0,
        "is_grasp": 1.0 if skill == "grasp" else 0.0,
        "is_place": 1.0 if skill == "place" else 0.0,
    }
    return [_to_float(values[name]) for name in GATE_FEATURE_NAMES]


class LearnableNeSyRoGate:
    """Small MLP gate; unavailable without a valid torch checkpoint.

    Unavailable gates raise :class:`RuntimeError` from :meth:`predict`;
    the :class:`nesyro_gate.unified_gate.UnifiedNeSyRoGate` facade turns that
    into an automatic legacy fallback.
    """

    name = "learnable"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hidden_units: Sequence[int] = DEFAULT_HIDDEN_UNITS,
    ):
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.hidden_units = tuple(int(unit) for unit in hidden_units)
        self.available = False
        self.model = None
        self.load_error: Optional[str] = None
        self._try_load_checkpoint()

    def _try_load_checkpoint(self) -> None:
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            self.load_error = "no checkpoint file"
            return
        try:
            import torch
        except Exception as exc:
            self.load_error = f"torch unavailable: {exc}"
            return
        try:
            model = build_mlp(len(GATE_FEATURE_NAMES), self.hidden_units)
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.available = True
        except Exception as exc:
            self.load_error = f"failed to load checkpoint: {exc}"
            self.model = None

    def predict(self, gate_input: Optional[GateInput] = None) -> GateOutput:
        """Return 4-way probabilities for a GateInput (CPU inference).

        Raises :class:`RuntimeError` when unavailable; callers must fall
        back to the legacy gate in that case.
        """

        if not self.available or self.model is None:
            raise RuntimeError(self.load_error or "learnable gate unavailable")

        import torch

        vector = gate_input_to_feature_vector(gate_input)
        with torch.no_grad():
            logits = self.model(torch.tensor([vector], dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()

        scores = {
            action.value: float(probability)
            for action, probability in zip(ACTION_ORDER, probabilities)
        }
        best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
        best_action = ACTION_ORDER[best_index]
        return GateOutput(
            action=best_action,
            confidence=float(probabilities[best_index]),
            action_scores=scores,
            reason=f"learnable gate argmax over {len(GATE_FEATURE_NAMES)} features",
            debug_info={
                "feature_names": GATE_FEATURE_NAMES,
                "features": vector,
                "checkpoint_path": self.checkpoint_path,
                "hidden_units": list(self.hidden_units),
            },
        )
