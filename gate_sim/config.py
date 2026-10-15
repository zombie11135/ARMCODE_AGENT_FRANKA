"""Frozen experiment configuration for GateOccludedPickPlace-v0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class CameraConfig:
    width: int = 192
    height: int = 144
    fov_degrees: float = 58.0
    near: float = 0.03
    far: float = 2.0
    # Reachable wrist-camera positions.  Only the active one is rendered.
    viewpoints: Tuple[Tuple[float, float, float], ...] = (
        (0.10, -0.36, 0.44),
        (0.03, 0.22, 0.50),
        (0.18, 0.16, 0.50),
    )
    look_at: Tuple[float, float, float] = (0.18, 0.02, 0.025)


@dataclass(frozen=True)
class PhysicsConfig:
    timestep_seconds: float = 1.0 / 240.0
    solver_iterations: int = 150
    settle_steps: int = 360
    settle_linear_velocity_mps: float = 0.015
    settle_angular_velocity_radps: float = 0.08
    stable_duration_seconds: float = 0.50
    task_time_limit_seconds: float = 45.0
    compute_time_limit_seconds: float = 180.0
    position_tolerance_m: float = 0.004
    orientation_tolerance_rad: float = 0.03


@dataclass(frozen=True)
class WorkspaceConfig:
    target_half_extents: Tuple[float, float, float] = (0.035, 0.022, 0.030)
    distractor_half_extents: Tuple[float, float, float] = (0.040, 0.026, 0.042)
    goal_center: Tuple[float, float, float] = (0.30, 0.22, 0.012)
    goal_inner_half_extents: Tuple[float, float] = (0.095, 0.080)
    goal_wall_height: float = 0.035
    buffer_center: Tuple[float, float, float] = (0.32, -0.23, 0.006)
    buffer_half_extents: Tuple[float, float] = (0.085, 0.070)
    buffer_capacity: int = 1
    search_region_center: Tuple[float, float] = (0.15, 0.015)
    search_region_half_extents: Tuple[float, float] = (0.16, 0.16)


@dataclass(frozen=True)
class SkillConfig:
    motion_steps: int = 180
    finger_steps: int = 100
    approach_height_m: float = 0.18
    lift_height_m: float = 0.22
    # Panda grasptarget link sits above the lower finger pads in this URDF.
    grasp_clearance_m: float = 0.030
    open_width_m: float = 0.040
    closed_width_m: float = 0.000
    bilateral_contact_required: bool = True
    # A fixed constraint is created only after bilateral finger/object contact.
    # It models a stiff force-closure grasp and is removed before release.
    contact_latch_enabled: bool = True
    max_visual_probes: int = 2
    max_manipulation_probes: int = 1


@dataclass(frozen=True)
class GateSimConfig:
    env_id: str = "GateOccludedPickPlace-v0"
    scene_config_version: str = "gate-occlusion-scenes-v3-daily-target"
    executor_version: str = "panda-contact-latch-v1"
    perception_version: str = "rgb-depth-color-seg-v2-textured-clutter"
    camera: CameraConfig = field(default_factory=CameraConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    object_colors_rgb: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "target_red": (220, 45, 45),
            "target_daily_oat_box": (190, 60, 128),
            "distractor_blue": (45, 90, 220),
            "distractor_green": (45, 190, 80),
            "distractor_yellow": (225, 190, 45),
            "daily_oat_box": (190, 60, 128),
            "daily_orange_can": (232, 145, 34),
            "daily_purple_cup": (155, 80, 180),
            "daily_cyan_bottle": (27, 172, 185),
        }
    )
    textured_daily_clutter_enabled: bool = True
    daily_object_catalog: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "daily_oat_box": {
                "semantic_category": "food_carton",
                "shape": "box",
                "half_extents_m": (0.028, 0.018, 0.052),
                "position_m": (-0.015, -0.205, 0.052),
                "yaw_rad": -0.28,
                "mass_kg": 0.10,
                "texture": "oat_box",
            },
            "daily_orange_can": {
                "semantic_category": "beverage_can",
                "shape": "cylinder",
                "radius_m": 0.024,
                "half_height_m": 0.055,
                "position_m": (0.405, -0.105, 0.055),
                "yaw_rad": 0.18,
                "mass_kg": 0.12,
                "texture": "orange_can",
            },
            "daily_purple_cup": {
                "semantic_category": "cup",
                "shape": "cylinder",
                "radius_m": 0.032,
                "half_height_m": 0.043,
                "position_m": (0.445, 0.075, 0.043),
                "yaw_rad": -0.12,
                "mass_kg": 0.11,
                "texture": "purple_cup",
            },
            "daily_cyan_bottle": {
                "semantic_category": "water_bottle",
                "shape": "cylinder",
                "radius_m": 0.022,
                "half_height_m": 0.070,
                "position_m": (-0.080, 0.245, 0.070),
                "yaw_rad": 0.10,
                "mass_kg": 0.14,
                "texture": "cyan_bottle",
            },
        }
    )
    # A--D remain the default cycle so existing fixed-seed datasets do not
    # silently change. E is an explicit textured-daily-target task.
    default_scene_cycle: Tuple[str, ...] = ("A", "B", "C", "D")
    scene_types: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])

    def to_dict(self) -> dict:
        return asdict(self)
