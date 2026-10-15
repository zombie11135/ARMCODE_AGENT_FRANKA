"""Runnable demo for the rule-based NeSyRo Gate prototype."""

from __future__ import annotations

from copy import deepcopy

from nesyro_gate.gate import nesyro_gate
from nesyro_gate.types import ObjectState, SceneMemory, SkillName, Step, TaskSpec


def _base_objects() -> dict[str, ObjectState]:
    """Create a baseline tabletop medicine-box scene."""

    return {
        "box_1": ObjectState(
            object_id="box_1",
            name="ibuprofen box",
            object_type="medicine_box",
            visible=True,
            detector_confidence=0.95,
            ocr_text="ibuprofen",
            ocr_match_score=0.95,
            vlm_match_score=0.9,
            reachable=True,
            graspable=True,
            occlusion_level=0.1,
        ),
        "plate_blue": ObjectState(
            object_id="plate_blue",
            name="blue plate",
            object_type="plate",
            visible=True,
            detector_confidence=0.95,
            ocr_text=None,
            ocr_match_score=0.8,
            vlm_match_score=0.9,
            reachable=True,
            graspable=False,
            occlusion_level=0.0,
            free_space=True,
            placeable=True,
        ),
    }


def _task_spec() -> TaskSpec:
    """Create the demo task specification."""

    return TaskSpec(
        target_object_name="ibuprofen box",
        target_plate_name="blue plate",
        allowed_skills=[SkillName.OBSERVE, SkillName.GRASP, SkillName.PLACE],
        target_object_id="box_1",
        target_plate_id="plate_blue",
    )


def _print_result(case_name: str, result) -> None:
    """Print a compact, readable gate result."""

    print(f"\n{case_name}")
    print(f"decision: {result.decision.value}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"csc: {result.csc:.3f}")
    print(f"lc: {result.lc:.3f}")
    print(f"reason: {result.reason}")
    print(f"failed_conditions: {result.failed_conditions}")
    print(f"suggested_probe: {result.suggested_probe}")
    print(f"suggested_constraint: {result.suggested_constraint}")


def run_demo() -> None:
    """Run five representative gate decisions."""

    task_spec = _task_spec()

    cases: list[tuple[str, Step, SceneMemory]] = []

    scene = SceneMemory(objects=deepcopy(_base_objects()))
    cases.append(("Case 1: high-confidence grasp", Step(SkillName.GRASP, {"object": "box_1"}), scene))

    scene = SceneMemory(objects=deepcopy(_base_objects()))
    scene.objects["box_1"].visible = False
    cases.append(("Case 2: target not visible", Step(SkillName.GRASP, {"object": "box_1"}), scene))

    scene = SceneMemory(objects=deepcopy(_base_objects()))
    scene.objects["box_1"].ocr_match_score = 0.15
    scene.objects["box_1"].vlm_match_score = 0.2
    scene.objects["box_1"].detector_confidence = 0.6
    cases.append(("Case 3: low OCR/VLM confidence", Step(SkillName.GRASP, {"object": "box_1"}), scene))

    scene = SceneMemory(objects=deepcopy(_base_objects()))
    scene.objects["box_1"].occlusion_level = 0.8
    cases.append(("Case 4: target severely occluded", Step(SkillName.GRASP, {"object": "box_1"}), scene))

    scene = SceneMemory(objects=deepcopy(_base_objects()))
    cases.append((
        "Case 5: place object not in gripper",
        Step(SkillName.PLACE, {"object": "box_1", "plate": "plate_blue"}),
        scene,
    ))

    for case_name, step, scene_memory in cases:
        result = nesyro_gate(step, scene_memory, task_spec)
        _print_result(case_name, result)


if __name__ == "__main__":
    run_demo()
