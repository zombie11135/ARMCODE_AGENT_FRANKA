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
)


def main():
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q", *[f"tests/{name}" for name in TESTS]],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
