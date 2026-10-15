"""Occlusion-focused tests for NeSyRo Gate."""

from __future__ import annotations

from nesyro_gate import (
    GateDecision,
    ObjectState,
    SceneMemory,
    SkillName,
    Step,
    TaskSpec,
    build_scene_memory_from_simple_robot,
    nesyro_gate,
)


def _task() -> TaskSpec:
    return TaskSpec(
        target_object_name="布洛芬",
        target_plate_name="blue tray",
        allowed_skills=[SkillName.OBSERVE, SkillName.GRASP, SkillName.PLACE],
        target_object_id="布洛芬",
        target_plate_id="blue tray",
    )


def _object(occlusion_level: float) -> ObjectState:
    return ObjectState(
        object_id="布洛芬",
        name="布洛芬",
        object_type="medicine_box",
        visible=True,
        detector_confidence=0.96,
        ocr_text="布洛芬",
        ocr_match_score=1.0,
        vlm_match_score=0.95,
        reachable=True,
        graspable=True,
        occlusion_level=occlusion_level,
    )


def test_explicit_occlusion_overrides_high_ocr_confidence() -> None:
    scene = SceneMemory(objects={"布洛芬": _object(0.45)})

    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        scene,
        _task(),
    )

    assert result.decision == GateDecision.REPLAN
    assert "target_occluded" in result.failed_conditions
    assert scene.objects["布洛芬"].graspable is False


def test_low_occlusion_high_confidence_executes() -> None:
    scene = SceneMemory(objects={"布洛芬": _object(0.2)})

    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        scene,
        _task(),
    )

    assert result.decision == GateDecision.EXECUTE
    assert scene.objects["布洛芬"].graspable is True


def test_mask_area_occlusion_blocks_high_ocr_confidence() -> None:
    observations = {
        "布洛芬": {
            "kind": "medicine",
            "target": "布洛芬",
            "selected_instance": {
                "object_id": "obj_1",
                "bbox": [20, 20, 100, 100],
                "confidence": 0.96,
                "ocr_text": "布洛芬",
                "ocr_texts": ["布洛芬"],
                "mask_area": 2200,
            },
            "grasp_input": {
                "object_id": "obj_1",
                "bbox": [20, 20, 100, 100],
                "ocr_text": "布洛芬",
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
    }
    scene = build_scene_memory_from_simple_robot(observations)

    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        scene,
        _task(),
    )

    assert scene.objects["布洛芬"].occlusion_level > 0.35
    assert result.decision == GateDecision.REPLAN
    assert "target_occluded" in result.failed_conditions
