"""Tests for VLM occlusion results feeding NeSyRo Gate."""

from __future__ import annotations

from nesyro_gate import (
    GateDecision,
    SkillName,
    Step,
    build_scene_memory_from_simple_robot,
    build_task_spec_for_pick_place,
    nesyro_gate,
)


def test_vlm_occlusion_blocks_grasp_even_when_text_visible() -> None:
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
            },
            "grasp_input": {
                "object_id": "obj_1",
                "bbox": [20, 20, 100, 100],
                "depth_path": "runs/current_capture/wrist_depth.png",
                "ocr_text": "布洛芬",
            },
            "vlm_occlusion": {
                "target_visible": True,
                "target_partially_visible": True,
                "text_visible": True,
                "target_occluded": True,
                "occlusion_level": 0.72,
                "blocking_object_visible": True,
                "blocking_object_description": "another object lies on top of the medicine box",
                "grasp_safe": False,
                "confidence": 0.86,
                "recommendation": "replan",
                "evidence": {
                    "target_evidence": "the medicine text is visible",
                    "occlusion_evidence": "the box body is covered by another object",
                    "grasp_risk": "top surface is not clear for grasp",
                },
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
    task = build_task_spec_for_pick_place("布洛芬", "blue tray")

    assert scene.objects["布洛芬"].occlusion_level == 0.72
    assert scene.objects["布洛芬"].graspable is False

    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        scene,
        task,
    )

    assert result.decision == GateDecision.REPLAN
    assert "target_occluded" in result.failed_conditions
