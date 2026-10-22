"""Configuration for the simple skill facade; no device or model imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Legacy xArm fallback only; calibrated FR3 observations supply their own intrinsics.
FALLBACK_INTRINSICS = {
    "fx": 909.59,
    "fy": 909.80,
    "cx": 658.97,
    "cy": 355.42,
}


@dataclass
class SimpleRobotConfig:
    """Skill execution parameters, including legacy defaults and Gate policy."""

    # Observation and motion.
    use_sam_for_grasp: bool = False
    execute: bool = False
    xarm_ip: str = "192.168.1.237"
    handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml"
    home_pose_mmrad: List[float] = field(default_factory=lambda: [350, 0, 450, 3.14, 0.0, -1.57])
    move_speed: float = 50
    move_acc: float = 500
    observe_z_mm: float = 450.0
    observe_wait_s: float = 0.8
    observe_scan_y_offsets_mm: List[float] = field(default_factory=lambda: [0.0, -150.0, 150.0])

    # Container observation and placement.
    container_refine_observation: bool = True
    container_refine_observe_z_mm: float = 450.0
    container_refine_wait_s: float = 0.8
    place_clearance_mm: float = 250.0
    place_approach_clearance_mm: float = 120.0
    place_lift_after_mm: float = 120.0
    place_slot_spacing_mm: float = 35.0
    enable_place_slot_offsets: bool = False

    # Grasp selection and execution.
    grasp_min_score: float = 0.50
    grasp_box_threshold: float = 0.22
    grasp_text_threshold: float = 0.18
    grasp_top_k: int = 50
    grasp_selection_mode: str = "best_good"
    grasp_random_select_top_k: int = 5
    grasp_random_seed: Optional[int] = None
    grasp_pose_options: Dict[str, float] = field(default_factory=dict)
    bbox_filter_margin_px: int = 0
    bbox_filter_inner_margin_ratio: float = 0.05
    collision_detection: bool = True
    verify_ocr: bool = True
    ocr_verify_strict_mode: bool = False
    ocr_verify_fuzzy_threshold: float = 0.72
    gripper_close_repeat_count: int = 2
    gripper_close_repeat_interval_s: float = 0.25
    gripper_close_wait_s: float = 1.5
    visualize_selected_grasp: bool = False
    visualize_bbox_filtered_grasps: bool = False
    reobserve_on_grasp_failure: bool = True
    reobserve_z_mm: float = 450.0
    reobserve_wait_s: float = 0.8
    reobserve_depth_window_px: int = 25

    # Gate decisions and evidence logging.
    enable_nesyro_gate: bool = True
    nesyro_gate_threshold: float = 0.7
    nesyro_gate_max_probe: int = 3
    nesyro_gate_trace_path: str = "runs/current_plan/gate_trace.jsonl"
    # Phase-1 unified gate config. gate_mode: legacy / learned / shadow.
    gate_mode: str = "legacy"
    gate_experience_log_path: str = "logs/gate_experience/gate_experience.jsonl"
    learned_gate_checkpoint_path: str = ""
    enable_vlm_occlusion_gate: bool = False
    vlm_occlusion_base_url: str = ""
    vlm_occlusion_api_key: str = ""
    vlm_occlusion_model: str = ""
    vlm_occlusion_timeout: int = 120
    vlm_occlusion_max_tokens: int = 500
    vlm_occlusion_temperature: float = 0.0

    # Occlusion recovery.
    enable_occlusion_recovery: bool = False
    occlusion_recovery_max_attempts: int = 1
    occlusion_recovery_capture_dir: str = "runs/current_capture"
    occlusion_buffer_pose_mmrad: List[float] = field(
        default_factory=lambda: [280.0, -260.0, 170.0, 3.14, 0.0, -1.57]
    )
    occlusion_buffer_approach_clearance_mm: float = 120.0
    occlusion_blocker_bbox_shrink_ratio: float = 0.08
    occlusion_blocker_search_expand_ratio: float = 1.8
    occlusion_blocker_min_area_ratio: float = 0.50
    occlusion_blocker_depth_closer_margin_m: float = 0.01
    occlusion_blocker_detector_prompts: List[str] = field(
        default_factory=lambda: [
            "medicine_box",
            "snack",
            "box",
            "package",
            "object on top of another object",
        ]
    )
    occlusion_blocker_detector_threshold: float = 0.20
    occlusion_blocker_detector_text_threshold: float = 0.16
    occlusion_blocker_detector_max_objects: int = 12
    occlusion_allow_unlocalized_target: bool = False
    occlusion_blocker_max_image_area_ratio: float = 0.50
    occlusion_blocker_max_target_area_ratio: float = 6.0
    occlusion_unlocalized_vlm_min_confidence: float = 0.85
    occlusion_unlocalized_detector_min_confidence: float = 0.25
    occlusion_min_depth_valid_ratio: float = 0.35
    occlusion_foreground_depth_band_m: float = 0.018
    occlusion_min_foreground_support_ratio: float = 0.15
    occlusion_active_view_enabled: bool = True
    occlusion_active_view_z_mm: float = 550.0
    occlusion_active_view_y_offsets_mm: List[float] = field(
        default_factory=lambda: [0.0, -120.0, 120.0]
    )
    occlusion_active_view_wait_s: float = 0.8
    enable_geometry_occlusion: bool = False
    geometry_occlusion_buffer_poses_mmrad: List[List[float]] = field(default_factory=list)
