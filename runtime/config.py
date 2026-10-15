"""Configuration shared by task orchestration and execution.

Importing this module does not load models or hardware backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SimpleAgentSystemConfig:
    """Task, planning and execution settings; hardware YAML is applied by the CLI.

    Defaults and field order retain the existing constructor contract.
    """

    # Task planning, artifacts and execution lifecycle.
    goal: str
    project_root: str = "."
    use_vdm: bool = False
    use_llm_planner: bool = False
    scene_text: str = ""
    plan_json: str = "runs/current_plan/plan_graph.json"
    plan_md: str = "runs/current_plan/plan_graph.md"
    replan_json: str = "runs/current_plan/replan_graph.json"
    replan_md: str = "runs/current_plan/replan_graph.md"
    generated_code: str = "runs/generated_code/current_plan_executor.py"
    failure_report_path: str = "runs/current_plan/failure_report.json"
    simple_memory_path: str = "runs/current_plan/simple_execution_memory.json"
    execute_robot: bool = False
    run_executor: bool = True
    enable_replan: bool = True
    max_replan_attempts: int = 1
    resume_from_failure: bool = True
    debug_output: bool = False
    verbose_events: bool = False
    show_container_observation: bool = True
    # Gate decisions and optional occlusion recovery.
    enable_nesyro_gate: bool = True
    nesyro_gate_threshold: float = 0.7
    nesyro_gate_max_probe: int = 3
    nesyro_gate_trace_path: str = "runs/current_plan/gate_trace.jsonl"
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
    enable_occlusion_recovery: bool = False
    occlusion_recovery_max_attempts: int = 1
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

    # Camera capture and legacy xArm motion defaults.
    global_serial: str = "213522251050"
    wrist_serial: str = "243222070053"
    capture_dir: str = "runs/current_capture"
    camera_warmup_s: float = 0.2
    xarm_ip: str = "192.168.1.237"
    handeye_config: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml"
    move_speed: float = 50
    move_acc: float = 500
    visualize_selected_grasp: bool = False
    visualize_bbox_filtered_grasps: bool = False
    bbox_filter_margin_px: int = 0
    bbox_filter_inner_margin_ratio: float = 0.05
    grasp_selection_mode: str = "best_good"
    grasp_random_select_top_k: int = 5
    grasp_random_seed: Optional[int] = None
    grasp_pose_options: Dict[str, float] = field(default_factory=dict)
    collision_detection: bool = True
    place_slot_spacing_mm: float = 35.0
    disable_place_slot_offsets: bool = True

    # Perception backends.
    grounding_dino_model: str = "/home/wpy/models/grounding-dino-tiny"
    detector_device: str = "cuda"
    anygrasp_dir: str = "/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection"
    anygrasp_ckpt: str = "/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"

    # Legacy ROS gripper settings.
    ros_ws: str = "/home/wpy/gt200_ws"
    grip_service: str = "/grip"
    grip_timeout: float = 5.0
    grip_open_value: int = 1
    grip_close_value: int = 0
    gripper_open_repeat_count: int = 2
    gripper_open_repeat_interval_s: float = 0.25
    gripper_close_repeat_count: int = 2
    gripper_close_repeat_interval_s: float = 0.25
    gripper_close_wait_s: float = 1.5
    ignore_grip_error: bool = True
    disable_grip: bool = False

    # Post-placement visual verification.
    verify_place_vdm: bool = False
    place_vdm_stop_on_failure: bool = True
    place_vdm_min_success_confidence: float = 0.55
    place_vdm_min_visible_success_confidence: float = 0.35
    place_vdm_min_failure_confidence: float = 0.75
    vdm_base_url: str = ""
    vdm_api_key: str = ""
    vdm_model: str = ""
    vdm_timeout: int = 120
    vdm_temperature: float = 0.0
    vdm_max_tokens: int = 500

    # Planning and fallback language models.
    fallback_vlm_base_url: str = ""
    fallback_vlm_api_key: str = ""
    fallback_vlm_model: str = ""
    planner_llm_base_url: str = ""
    planner_llm_api_key: str = ""
    planner_llm_model: str = ""
    vlm_timeout: int = 120

    # Recovery, FR3 profile and task-scoped target bindings.
    recovery_mode: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    # None retains the legacy path for callers that explicitly use it.
    hardware_config_path: Optional[str] = None
    arm_speed_scaling: Optional[float] = None
    use_sam_for_grasp: bool = False
    target_name_bindings: Dict[str, str] = field(default_factory=dict)
