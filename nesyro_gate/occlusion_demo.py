"""Standalone occlusion stress demo for NeSyRo Gate."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from nesyro_gate import (
    GateDecision,
    ObjectState,
    SceneMemory,
    SkillName,
    Step,
    TaskSpec,
    build_scene_memory_from_simple_robot,
    build_task_spec_for_pick_place,
    nesyro_gate,
)


TARGET = "布洛芬"
PLATE = "blue tray"


def _base_object(occlusion_level: float = 0.1) -> ObjectState:
    """Create a high semantic-confidence medicine box ObjectState."""

    return ObjectState(
        object_id=TARGET,
        name=TARGET,
        object_type="medicine_box",
        visible=True,
        detector_confidence=0.96,
        ocr_text=TARGET,
        ocr_match_score=1.0,
        vlm_match_score=0.95,
        reachable=True,
        graspable=True,
        occlusion_level=occlusion_level,
    )


def _task_spec() -> TaskSpec:
    """Create the pick-place task spec used by all occlusion cases."""

    return TaskSpec(
        target_object_name=TARGET,
        target_plate_name=PLATE,
        allowed_skills=[SkillName.OBSERVE, SkillName.GRASP, SkillName.PLACE],
        target_object_id=TARGET,
        target_plate_id=PLATE,
    )


def _medicine_observation(mask_area: float | None = None) -> dict[str, Any]:
    """Build a cached medicine observation similar to SimpleRobotActions."""

    selected = {
        "object_id": "obj_1",
        "bbox": [20, 20, 100, 100],
        "confidence": 0.96,
        "ocr_text": TARGET,
        "ocr_texts": [TARGET],
    }
    if mask_area is not None:
        selected["mask_area"] = mask_area

    return {
        "kind": "medicine",
        "target": TARGET,
        "selected_instance": selected,
        "grasp_input": {
            "object_id": "obj_1",
            "bbox": [20, 20, 100, 100],
            "depth_path": "runs/current_capture/wrist_depth.png",
            "ocr_text": TARGET,
        },
        "result": {
            "selection_result": {
                "score": 0.96,
                "ranked_candidates": [
                    {
                        "object_id": "obj_1",
                        "score": 0.96,
                        "raw_text_score": 1.0,
                        "detection_confidence": 0.96,
                    }
                ],
            },
            "ocr_verification": {"confidence": 1.0},
        },
    }


def _make_depth_observation_if_possible() -> dict[str, Any] | None:
    """Create a depth-foreground occlusion observation when numpy is available."""

    try:
        import numpy as np
    except Exception:
        return None

    tmpdir = tempfile.TemporaryDirectory()
    depth = np.full((140, 140), 0.60, dtype=float)
    depth[50:85, 50:85] = 0.53
    depth_path = Path(tmpdir.name) / "occlusion_depth.npy"
    np.save(depth_path, depth)

    observation = _medicine_observation()
    observation["_tmpdir"] = tmpdir
    observation["grasp_input"]["depth_npy_path"] = str(depth_path)
    observation["grasp_input"]["depth_scale"] = 1000.0
    return observation


def _run_case(case_name: str, scene: SceneMemory, source: str) -> None:
    """Run gate and print the occlusion-focused result."""

    before = deepcopy(scene.objects[TARGET])
    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": TARGET}),
        scene,
        _task_spec(),
    )
    after = scene.objects[TARGET]

    print(f"\n{case_name}")
    print(f"source: {source}")
    print(f"decision: {result.decision.value}")
    print(f"confidence: {result.confidence:.3f}  csc: {result.csc:.3f}  lc: {result.lc:.3f}")
    print(f"failed_conditions: {result.failed_conditions}")
    print(f"suggested_constraint: {result.suggested_constraint}")
    print(
        "object_before: "
        f"occlusion={before.occlusion_level:.3f}, graspable={before.graspable}"
    )
    print(
        "object_after:  "
        f"occlusion={after.occlusion_level:.3f}, graspable={after.graspable}"
    )


def run_demo() -> None:
    """Run a set of occlusion-specific gate cases."""

    direct_cases = [
        ("Case 1: no occlusion, OCR high", 0.10),
        ("Case 2: mild occlusion below hard threshold", 0.30),
        ("Case 3: explicit occlusion above threshold", 0.45),
        ("Case 4: severe explicit occlusion", 0.80),
    ]

    for case_name, occlusion_level in direct_cases:
        scene = SceneMemory(objects={TARGET: _base_object(occlusion_level)})
        _run_case(case_name, scene, source="direct ObjectState.occlusion_level")

    mask_cases = [
        ("Case 5: mask mostly visible", 5600.0),
        ("Case 6: mask visible area too small, OCR still high", 2200.0),
    ]
    for case_name, mask_area in mask_cases:
        scene = build_scene_memory_from_simple_robot(
            observations={TARGET: _medicine_observation(mask_area=mask_area)}
        )
        _run_case(case_name, scene, source=f"mask_area={mask_area}")

    depth_observation = _make_depth_observation_if_possible()
    if depth_observation is None:
        print("\nCase 7: depth foreground occlusion")
        print("skipped: numpy is not available in this Python environment")
    else:
        try:
            scene = build_scene_memory_from_simple_robot(
                observations={TARGET: depth_observation}
            )
            _run_case(
                "Case 7: depth foreground occlusion, OCR still high",
                scene,
                source="synthetic depth_npy_path foreground cluster",
            )
        finally:
            depth_observation["_tmpdir"].cleanup()

    print("\nExpected behavior:")
    print("- Cases below occlusion threshold should EXECUTE.")
    print("- Cases above occlusion threshold should REPLAN with target_occluded.")
    print("- After target_occluded, SceneMemory should mark graspable=False.")


if __name__ == "__main__":
    run_demo()
