"""Shared data types for the rule-based NeSyRo Gate prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GateDecision(str, Enum):
    """Possible gate decisions before executing an instruction node."""

    EXECUTE = "EXECUTE"
    PROBE_REQUIRED = "PROBE_REQUIRED"
    LOCAL_REPAIR = "LOCAL_REPAIR"
    REPLAN = "REPLAN"
    ABORT = "ABORT"


class SkillName(str, Enum):
    """High-level skills currently supported by the prototype."""

    OBSERVE = "observe"
    GRASP = "grasp"
    PLACE = "place"


@dataclass
class Step:
    """One instruction graph node emitted by a planner."""

    name: SkillName
    args: dict[str, Any]
    step_id: str | None = None


@dataclass
class TaskSpec:
    """Task-level constraints used to verify instruction graph nodes."""

    target_object_name: str
    target_plate_name: str
    allowed_skills: list[SkillName]
    target_object_id: str | None = None
    target_plate_id: str | None = None


@dataclass
class ObjectState:
    """Scene-memory state for one perceived object."""

    object_id: str
    name: str | None
    object_type: str
    visible: bool
    detector_confidence: float
    ocr_text: str | None
    ocr_match_score: float
    vlm_match_score: float
    reachable: bool
    graspable: bool
    occlusion_level: float
    in_gripper: bool = False
    free_space: bool = True
    placeable: bool = True
    identity_source: str | None = None
    identity_match_score: float = 0.0
    identity_verified: bool = False
    fixed_for_task: bool = False


@dataclass
class SceneMemory:
    """Mutable symbolic and perceptual memory used by the gate."""

    objects: dict[str, ObjectState]
    probe_counts: dict[str, int] = field(default_factory=dict)
    searched_regions: set[str] = field(default_factory=set)
    gripper_holding: str | None = None

    def get_object(self, object_id: str) -> ObjectState | None:
        """Return an object state by id, if present."""

        return self.objects.get(object_id)

    def is_visible(self, object_id: str) -> bool:
        """Return whether an object exists and is visible."""

        obj = self.get_object(object_id)
        return bool(obj and obj.visible)

    def is_reachable(self, object_id: str) -> bool:
        """Return whether an object exists and is reachable."""

        obj = self.get_object(object_id)
        return bool(obj and obj.reachable)

    def is_graspable(self, object_id: str) -> bool:
        """Return whether an object exists and is graspable."""

        obj = self.get_object(object_id)
        return bool(obj and obj.graspable)

    def is_severely_occluded(self, object_id: str, threshold: float = 0.6) -> bool:
        """Return whether an object's occlusion level exceeds the threshold."""

        obj = self.get_object(object_id)
        return bool(obj and obj.occlusion_level > threshold)

    def in_gripper(self, object_id: str) -> bool:
        """Return whether the object is currently held by the gripper."""

        obj = self.get_object(object_id)
        return bool((obj and obj.in_gripper) or self.gripper_holding == object_id)

    def plate_has_free_space(self, plate_id: str) -> bool:
        """Return whether a plate or container has free placement space."""

        plate = self.get_object(plate_id)
        return bool(plate and plate.free_space)

    def increment_probe_count(self, step_key: str) -> int:
        """Increment and return the probe count for a step or probe key."""

        self.probe_counts[step_key] = self.get_probe_count(step_key) + 1
        return self.probe_counts[step_key]

    def get_probe_count(self, step_key: str) -> int:
        """Return how many probes have already been requested for a key."""

        return self.probe_counts.get(step_key, 0)


@dataclass
class GateResult:
    """Gate output for one instruction graph node."""

    decision: GateDecision
    confidence: float
    csc: float
    lc: float
    reason: str
    failed_conditions: list[str]
    suggested_probe: str | None = None
    suggested_constraint: str | None = None
