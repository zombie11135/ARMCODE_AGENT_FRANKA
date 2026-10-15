"""Unified gate interface types shared by the legacy and learnable gates.

Phase-1 addition. These types do NOT change the legacy CSCxLC rule gate:
they only define the common contract that future learned gates implement.
Fields that the current pipeline cannot provide stay None / default and are
recorded explicitly in gate experience logs (missing evidence is not a
positive signal).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ProbeAction(str, Enum):
    """Unified gate decision used by both legacy and learnable gates."""

    EXECUTE = "EXECUTE"
    VISUAL_PROBE = "VISUAL_PROBE"
    MANIPULATION_PROBE = "MANIPULATION_PROBE"
    REPLAN = "REPLAN"


class GateMode(str, Enum):
    """Configured gate mode.

    legacy   - only the original CSCxLC gate drives the robot.
    learned  - the learnable MLP gate drives the robot; a missing checkpoint
               automatically falls back to legacy.
    shadow   - legacy drives the robot; the learnable gate is evaluated for
               logging only and its output never changes robot behavior.
    """

    LEGACY = "legacy"
    LEARNED = "learned"
    SHADOW = "shadow"


@dataclass
class GateInput:
    """Low-cost feature snapshot consumed by every gate implementation.

    Only information already available in the runtime / SceneMemory is used.
    The learnable gate never triggers extra perception or VLM calls.

    Missing values stay None (or 0 counts) and default to 0.0 in the learned
    feature vector; the experience log records the raw values as-is.
    """

    current_step: Dict[str, Any] = field(default_factory=dict)
    task_stage: Dict[str, Any] = field(default_factory=dict)
    # semantic confidence (CSC) and symbolic precondition state (LC)
    target_confidence: Optional[float] = None
    target_visibility: Optional[float] = None
    occlusion_score: Optional[float] = None
    pose_confidence: Optional[float] = None  # not provided by current pipeline
    blocker_confidence: Optional[float] = None  # not provided before recovery
    blocker_graspability: Optional[float] = None  # not provided before recovery
    blocker_target_relation: Optional[float] = None  # not provided yet
    prerequisite_state: Dict[str, Any] = field(default_factory=dict)
    # -1.0 failed / 0.0 unknown / 1.0 succeeded
    previous_action_result: Optional[float] = None
    visual_probe_count: int = 0
    manipulation_probe_count: int = 0
    memory_features: Dict[str, Any] = field(default_factory=dict)
    task_progress: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy for experience logging."""

        return asdict(self)


@dataclass
class GateOutput:
    """Unified gate output produced by legacy or learnable gates."""

    action: ProbeAction
    confidence: float
    action_scores: Optional[Dict[str, float]] = None
    reason: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy for experience logging."""

        return asdict(self)
