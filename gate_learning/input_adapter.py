"""Policy-only input contract for the RGB-D + task multimodal Gate.

Evaluator truth, scenario classes and simulator body identifiers are rejected
at this boundary.  The model may consume raw sensor observations, task text,
robot/task bookkeeping and action availability, but never hidden labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ACTION_NAMES: Tuple[str, ...] = (
    "EXECUTE",
    "VISUAL_PROBE",
    "MANIPULATION_PROBE",
    "REPLAN",
)

# Deliberately excludes detector/OCR/VLM scores and evaluator truth.  Validity
# is explicit for values which can be unknown; unknown is not silently zero.
MULTIMODAL_STATE_FEATURE_NAMES: Tuple[str, ...] = (
    "previous_action_result",
    "previous_action_result_valid",
    "visual_probe_count",
    "manipulation_probe_count",
    "task_progress",
    "task_progress_valid",
    "gripper_holding",
    "buffer_occupied",
    "is_observe",
    "is_grasp",
    "is_place",
    "remaining_sim_time_fraction",
    "remaining_sim_time_valid",
    "execute_available",
    "visual_probe_available",
    "manipulation_probe_available",
    "replan_available",
)

_FORBIDDEN_INPUT_KEYS = {
    "evaluator_state",
    "evaluator_metrics",
    "scene_type",
    "scenario_type",
    "true_blocker",
    "body_id",
    "target_body_id",
    "optimal_action",
}


@dataclass
class MultimodalGateInput:
    """One decision-time sample before frozen-encoder preprocessing."""

    rgb: np.ndarray
    depth_m: np.ndarray
    task_text: str
    state_vector: np.ndarray
    action_mask: np.ndarray
    history_rgb: Tuple[np.ndarray, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        state = np.asarray(self.state_vector)
        mask = np.asarray(self.action_mask)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must have shape HxWx3, got {rgb.shape}")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"depth_m must be aligned to rgb and have shape {rgb.shape[:2]}, got {depth.shape}"
            )
        if not np.issubdtype(rgb.dtype, np.integer):
            raise ValueError("rgb must contain integer sensor values")
        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size and float(finite_depth.min()) < 0.0:
            raise ValueError("depth_m must use non-negative metres")
        if state.shape != (len(MULTIMODAL_STATE_FEATURE_NAMES),):
            raise ValueError(
                f"state_vector must have {len(MULTIMODAL_STATE_FEATURE_NAMES)} values"
            )
        if mask.shape != (len(ACTION_NAMES),):
            raise ValueError(f"action_mask must have {len(ACTION_NAMES)} values")
        if not str(self.task_text).strip():
            raise ValueError("task_text must not be empty")
        forbidden = _FORBIDDEN_INPUT_KEYS.intersection(self.metadata)
        if forbidden:
            raise ValueError(f"hidden/evaluator fields are forbidden: {sorted(forbidden)}")
        for index, frame in enumerate(self.history_rgb):
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"history_rgb[{index}] must have shape HxWx3")


def _float_with_validity(value: Any) -> Tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    try:
        return float(value), 1.0
    except (TypeError, ValueError):
        return 0.0, 0.0


def _normalise_skill(value: Any) -> str:
    skill = str(value or "").strip().lower()
    if skill.startswith("skillname."):
        skill = skill.split(".", 1)[1]
    return skill


def build_policy_state_vector(
    *,
    robot_state: Mapping[str, Any],
    memory: Mapping[str, Any],
    action_mask: Mapping[str, Any],
    current_step: Any,
    task_progress: Optional[float] = None,
    previous_action_result: Optional[float] = None,
    remaining_sim_time_fraction: Optional[float] = None,
    policy_counters: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Build the non-visual state without confidence or hidden-truth fields."""

    for source in (robot_state, memory, action_mask):
        forbidden = _FORBIDDEN_INPUT_KEYS.intersection(source)
        if forbidden:
            raise ValueError(f"hidden/evaluator fields are forbidden: {sorted(forbidden)}")

    previous, previous_valid = _float_with_validity(previous_action_result)
    progress, progress_valid = _float_with_validity(task_progress)
    remaining, remaining_valid = _float_with_validity(remaining_sim_time_fraction)
    skill = current_step
    if isinstance(current_step, Mapping):
        # Accept both the real runtime's {op: ...} and simulator/name schema.
        skill = current_step.get("skill") or current_step.get("name") or current_step.get("op")
    skill = _normalise_skill(skill)

    probe_counts = memory.get("probe_counts") or {}
    policy_counters = policy_counters or {}
    visual_count = policy_counters.get(
        "visual_probes", memory.get("visual_probe_count", probe_counts.get("visual_probe", 0))
    )
    manipulation_count = memory.get(
        "manipulation_probe_count",
        policy_counters.get("manipulation_probes", probe_counts.get("manipulation_probe", 0)),
    )
    holding = robot_state.get("holding")
    if holding is None:
        holding = memory.get("gripper_holding")

    values = (
        previous,
        previous_valid,
        float(visual_count or 0),
        float(manipulation_count or 0),
        progress,
        progress_valid,
        1.0 if holding else 0.0,
        1.0 if memory.get("buffer_occupied", False) else 0.0,
        1.0 if skill == "observe" else 0.0,
        1.0 if skill in {"grasp", "grasp_target"} else 0.0,
        1.0 if skill in {"place", "place_target"} else 0.0,
        remaining,
        remaining_valid,
        1.0 if action_mask.get("EXECUTE", False) else 0.0,
        1.0 if action_mask.get("VISUAL_PROBE", False) else 0.0,
        1.0 if action_mask.get("MANIPULATION_PROBE", False) else 0.0,
        1.0 if action_mask.get("REPLAN", False) else 0.0,
    )
    return np.asarray(values, dtype=np.float32)


def from_sim_policy_observation(
    observation: Mapping[str, Any],
    *,
    task_text: str,
    history_rgb: Sequence[np.ndarray] = (),
    remaining_sim_time_fraction: Optional[float] = None,
) -> MultimodalGateInput:
    """Adapt ``GateOccludedPickPlaceEnv.get_gate_observation()`` safely."""

    forbidden = _FORBIDDEN_INPUT_KEYS.intersection(observation)
    if forbidden:
        raise ValueError(f"top-level evaluator fields are forbidden: {sorted(forbidden)}")
    policy = observation.get("policy_observation")
    if not isinstance(policy, Mapping):
        raise ValueError("policy_observation is required")
    action_mask = observation.get("action_mask")
    if not isinstance(action_mask, Mapping):
        raise ValueError("action_mask is required")
    memory = policy.get("memory") or {}
    robot_state = policy.get("robot_state") or {}
    task_state = policy.get("task_state") or {}
    current_step = task_state.get("node") or memory.get("task_node") or policy.get("current_step")
    previous = memory.get("last_action_result")
    task_progress = task_state.get("task_progress", memory.get("task_progress"))
    if remaining_sim_time_fraction is None:
        remaining_sim_time_fraction = task_state.get("remaining_sim_time_fraction")
    state = build_policy_state_vector(
        robot_state=robot_state,
        memory=memory,
        action_mask=action_mask,
        current_step=current_step,
        task_progress=task_progress,
        previous_action_result=previous,
        remaining_sim_time_fraction=remaining_sim_time_fraction,
        policy_counters=policy.get("policy_counters") or {},
    )
    item = MultimodalGateInput(
        rgb=np.asarray(policy["rgb"]).copy(),
        depth_m=np.asarray(policy["depth_m"], dtype=np.float32).copy(),
        task_text=str(task_text),
        state_vector=state,
        action_mask=np.asarray([bool(action_mask.get(name, False)) for name in ACTION_NAMES]),
        history_rgb=tuple(np.asarray(frame).copy() for frame in history_rgb[-4:]),
        metadata={
            "camera_index": policy.get("camera_index"),
            "artifact_refs": dict(policy.get("artifact_refs") or {}),
            "input_provenance": "policy_observation_only",
            "depth_unit": "m",
        },
    )
    item.validate()
    return item


def load_rgb_depth(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one aligned RGB/depth artifact pair produced by the simulator."""

    from PIL import Image

    rgb = np.asarray(Image.open(Path(rgb_path)).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(Path(depth_path)), dtype=np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"unaligned RGB/depth artifacts: {rgb.shape} vs {depth.shape}")
    return rgb, depth
