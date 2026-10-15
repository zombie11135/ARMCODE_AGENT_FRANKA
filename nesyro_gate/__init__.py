"""Rule-based NeSyRo Gate prototype for high-level robot task planning.

Phase-1 additions (unified interface / learnable skeleton / experience
logging) are exported alongside the unchanged rule gate.
"""

from nesyro_gate.experience_logger import (
    GateExperienceLogger,
    decision_record,
    outcome_record,
)
from nesyro_gate.gate import nesyro_gate
from nesyro_gate.interface import GateInput, GateMode, GateOutput, ProbeAction
from nesyro_gate.learnable_gate import LearnableNeSyRoGate
from nesyro_gate.legacy_gate import (
    LegacyNeSyRoGate,
    legacy_action_for,
    legacy_action_scores,
)
from nesyro_gate.scene_memory_builder import (
    build_scene_memory_from_simple_robot,
    build_task_spec_for_pick_place,
    container_observation_to_object_state,
    container_plan_to_object_state,
    medicine_observation_to_object_state,
)
from nesyro_gate.types import (
    GateDecision,
    GateResult,
    ObjectState,
    SceneMemory,
    SkillName,
    Step,
    TaskSpec,
)
from nesyro_gate.unified_gate import UnifiedGateResult, UnifiedNeSyRoGate

__all__ = [
    "GateDecision",
    "GateExperienceLogger",
    "GateInput",
    "GateMode",
    "GateOutput",
    "GateResult",
    "LearnableNeSyRoGate",
    "LegacyNeSyRoGate",
    "ObjectState",
    "ProbeAction",
    "SceneMemory",
    "SkillName",
    "Step",
    "TaskSpec",
    "UnifiedGateResult",
    "UnifiedNeSyRoGate",
    "build_scene_memory_from_simple_robot",
    "build_task_spec_for_pick_place",
    "container_observation_to_object_state",
    "container_plan_to_object_state",
    "decision_record",
    "legacy_action_for",
    "legacy_action_scores",
    "medicine_observation_to_object_state",
    "nesyro_gate",
    "outcome_record",
]
