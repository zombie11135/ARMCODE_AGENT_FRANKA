"""Tests for SceneMemory adapters."""

from __future__ import annotations

from nesyro_gate import (
    GateDecision,
    SkillName,
    Step,
    build_scene_memory_from_simple_robot,
    build_task_spec_for_pick_place,
    nesyro_gate,
)


def test_build_scene_memory_from_simple_robot_caches() -> None:
    observations = {
        "蒙脱石散": {
            "kind": "medicine",
            "target": "蒙脱石散",
            "selected_instance": {
                "object_id": "obj_1",
                "bbox": [10, 20, 110, 120],
                "confidence": 0.9,
                "ocr_text": "蒙脱石散",
                "ocr_texts": ["蒙脱石散"],
            },
            "grasp_input": {
                "object_id": "obj_1",
                "bbox": [10, 20, 110, 120],
                "depth_path": "offline-test-fixture://unavailable-depth.png",
                "ocr_text": "蒙脱石散",
            },
            "result": {
                "selection_result": {
                    "score": 0.92,
                    "ranked_candidates": [
                        {
                            "object_id": "obj_1",
                            "score": 0.92,
                            "raw_text_score": 1.0,
                            "detection_confidence": 0.9,
                        }
                    ],
                },
                "ocr_verification": {"confidence": 1.0},
            },
        },
        "blue tray": {
            "kind": "container",
            "target": "blue tray",
            "container_key": "tray__blue",
            "place_plan": {
                "selected_detection": {
                    "object_id": "container_1",
                    "bbox": [200, 200, 420, 420],
                    "confidence": 0.86,
                    "color_score": 0.72,
                    "phrase": "blue tray",
                },
                "center_base_xyz": [0.35, 0.1, 0.05],
                "place_pose_mmrad": [350, 100, 180, 3.14, 0.0, -1.57],
            },
        },
    }

    scene = build_scene_memory_from_simple_robot(observations)
    task = build_task_spec_for_pick_place("蒙脱石散", "blue tray")

    assert "蒙脱石散" in scene.objects
    assert "blue tray" in scene.objects
    assert scene.objects["蒙脱石散"].visible is True
    assert scene.objects["blue tray"].placeable is True

    grasp_result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "蒙脱石散"}),
        scene,
        task,
    )
    assert grasp_result.decision == GateDecision.EXECUTE

    scene.gripper_holding = "蒙脱石散"
    place_result = nesyro_gate(
        Step(SkillName.PLACE, {"object": "蒙脱石散", "plate": "blue tray"}),
        scene,
        task,
    )
    assert place_result.decision == GateDecision.EXECUTE


def test_visible_area_occlusion_blocks_high_confidence_grasp_and_updates_memory() -> None:
    observations = {
        "布洛芬": {
            "kind": "medicine",
            "target": "布洛芬",
            "selected_instance": {
                "object_id": "obj_1",
                "bbox": [20, 20, 60, 60],
                "confidence": 0.95,
                "ocr_text": "布洛芬",
                "ocr_texts": ["布洛芬"],
                "mask_area": 500,
            },
            "grasp_input": {
                "object_id": "obj_1",
                "bbox": [20, 20, 60, 60],
                "depth_path": "offline-test-fixture://unavailable-depth.png",
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
                            "detection_confidence": 0.95,
                        }
                    ],
                },
                "ocr_verification": {"confidence": 1.0},
            },
        }
    }

    scene = build_scene_memory_from_simple_robot(observations)
    task = build_task_spec_for_pick_place("布洛芬", "blue tray")

    assert scene.objects["布洛芬"].occlusion_level > 0.35

    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        scene,
        task,
    )

    assert result.decision == GateDecision.REPLAN
    assert "target_occluded" in result.failed_conditions
    assert scene.objects["布洛芬"].graspable is False
    assert "target_occluded(布洛芬)" in str(result.suggested_constraint)


def test_failed_observation_becomes_fully_occluded_scene_object() -> None:
    observations = {
        "布洛芬": {
            "kind": "medicine",
            "target": "布洛芬",
            "observation_failed": True,
            "error": "未找到稳定匹配的药盒。最高分=0.423, min_score=0.500",
            "result": {"target": "布洛芬"},
        }
    }

    scene = build_scene_memory_from_simple_robot(observations)
    task = build_task_spec_for_pick_place("布洛芬", "blue tray")

    obj = scene.objects["布洛芬"]
    assert obj.visible is False
    assert obj.reachable is False
    assert obj.graspable is False
    assert obj.occlusion_level == 1.0
    assert obj.detector_confidence == 0.423

    result = nesyro_gate(
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        scene,
        task,
        max_probe=0,
    )

    assert result.decision == GateDecision.REPLAN
    assert "target_not_visible" in result.failed_conditions
    assert "target_occluded" in result.failed_conditions
