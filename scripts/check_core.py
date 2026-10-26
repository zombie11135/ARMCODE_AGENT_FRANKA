"""Run the portable regression suite; never collect device experiment scripts."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Add a test here only after verifying it without models, local profiles or devices.
TESTS = (
    "test_release_export.py",
    "test_runtime_structure.py",
    "test_simple_actions_components.py",
    "test_planner_action_integrity.py",
    "test_multi_target_planning.py",
    "test_same_session_replanning.py",
    "test_gate.py",
    "test_gate_runtime_integration.py",
    "test_gate_experience_logging.py",
    "test_gate_replanning_constraints.py",
    "test_object_names.py",
    "test_gate_occlusion.py",
    "test_scene_memory_builder.py",
    "test_vlm_occlusion_gate.py",
    "test_clear_occlusion_to_buffer.py",
    "test_legacy_gate_equivalence.py",
    "test_camera_startup.py",
    "test_observation_memory.py",
    "test_perception_pipeline.py",
    "test_task_perception_pipeline.py",
    "test_reference_recognition.py",
    "test_observe_gallery.py",
    "test_reference_capture.py",
    "test_wrist_raw_capture.py",
    "test_occlusion_camera_only.py",
    "test_occlusion_multiview.py",
    "test_occlusion_observation.py",
    "test_occlusion_proposal_refinement.py",
    "test_occlusion_real_fusion.py",
    "test_occlusion_relations.py",
    "test_occlusion_robot_pair.py",
    "test_occlusion_primary_aux.py",
    "test_geometry_occlusion_agent.py",
    "test_occlusion_recovery_planner.py",
    "test_aruco_pose.py",
    "test_eye_to_hand.py",
    "test_fr3_hardware.py",
    "test_agent_home_command.py",
    "test_grasp_yaw_preference.py",
    "test_gripper_process.py",
)


def main():
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q", *[f"tests/{name}" for name in TESTS]],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
