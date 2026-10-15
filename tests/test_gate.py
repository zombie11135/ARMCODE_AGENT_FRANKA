"""Tests for the rule-based NeSyRo Gate prototype."""

from __future__ import annotations

from copy import deepcopy

from nesyro_gate.gate import nesyro_gate
from nesyro_gate.types import GateDecision, ObjectState, SceneMemory, SkillName, Step, TaskSpec


def base_objects() -> dict[str, ObjectState]:
    """Return a reusable high-confidence medicine-box scene."""

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


def task_spec() -> TaskSpec:
    """Return the ibuprofen-to-blue-plate task spec."""

    return TaskSpec(
        target_object_name="ibuprofen box",
        target_plate_name="blue plate",
        allowed_skills=[SkillName.OBSERVE, SkillName.GRASP, SkillName.PLACE],
        target_object_id="box_1",
        target_plate_id="plate_blue",
    )


def test_high_confidence_grasp_executes() -> None:
    scene = SceneMemory(objects=deepcopy(base_objects()))

    result = nesyro_gate(Step(SkillName.GRASP, {"object": "box_1"}), scene, task_spec())

    assert result.decision == GateDecision.EXECUTE
    assert result.confidence >= 0.7
    assert result.failed_conditions == []


def test_invisible_target_requires_probe() -> None:
    scene = SceneMemory(objects=deepcopy(base_objects()))
    scene.objects["box_1"].visible = False

    result = nesyro_gate(Step(SkillName.GRASP, {"object": "box_1"}), scene, task_spec())

    assert result.decision == GateDecision.PROBE_REQUIRED
    assert result.suggested_probe == "observe_target_region"
    assert "target_not_visible" in result.failed_conditions


def test_low_ocr_vlm_confidence_requires_probe() -> None:
    scene = SceneMemory(objects=deepcopy(base_objects()))
    scene.objects["box_1"].ocr_match_score = 0.15
    scene.objects["box_1"].vlm_match_score = 0.2
    scene.objects["box_1"].detector_confidence = 0.6

    result = nesyro_gate(Step(SkillName.GRASP, {"object": "box_1"}), scene, task_spec())

    assert result.decision == GateDecision.PROBE_REQUIRED
    assert result.suggested_probe in {"observe_closeup_target", "observe_closeup_for_ocr"}
    assert "target_low_confidence" in result.failed_conditions


def test_severe_occlusion_replans_with_constraint() -> None:
    scene = SceneMemory(objects=deepcopy(base_objects()))
    scene.objects["box_1"].occlusion_level = 0.8

    result = nesyro_gate(Step(SkillName.GRASP, {"object": "box_1"}), scene, task_spec())

    assert result.decision == GateDecision.REPLAN
    assert "target_occluded" in result.failed_conditions
    assert result.suggested_constraint is not None
    assert "target_occluded(box_1)" in result.suggested_constraint


def test_place_object_not_in_gripper_replans_with_constraint() -> None:
    scene = SceneMemory(objects=deepcopy(base_objects()))

    result = nesyro_gate(
        Step(SkillName.PLACE, {"object": "box_1", "plate": "plate_blue"}),
        scene,
        task_spec(),
    )

    assert result.decision == GateDecision.REPLAN
    assert "object_not_in_gripper" in result.failed_conditions
    assert result.suggested_constraint is not None
    assert "object_not_in_gripper(box_1)" in result.suggested_constraint
