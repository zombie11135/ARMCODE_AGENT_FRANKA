"""Phase-1 legacy equivalence tests.

LegacyNeSyRoGate / UnifiedNeSyRoGate(mode="legacy") must produce exactly the
same decisions, scores, failure conditions and SceneMemory side effects as
the original nesyro_gate() function for identical inputs.
"""

from __future__ import annotations

from copy import deepcopy

from nesyro_gate import (
    GateDecision,
    GateResult,
    LegacyNeSyRoGate,
    ProbeAction,
    SceneMemory,
    SkillName,
    Step,
    TaskSpec,
    UnifiedNeSyRoGate,
    build_scene_memory_from_simple_robot,
    build_task_spec_for_pick_place,
    legacy_action_for,
    nesyro_gate,
)

EXPECTED_ACTION = {
    GateDecision.EXECUTE: ProbeAction.EXECUTE,
    GateDecision.PROBE_REQUIRED: ProbeAction.VISUAL_PROBE,
    GateDecision.LOCAL_REPAIR: ProbeAction.REPLAN,
    GateDecision.REPLAN: ProbeAction.REPLAN,
    GateDecision.ABORT: ProbeAction.REPLAN,
}


def _task() -> TaskSpec:
    return build_task_spec_for_pick_place("布洛芬", "blue tray")


def _high_conf_scene() -> SceneMemory:
    return SceneMemory(
        objects={
            "布洛芬": _medicine(occlusion_level=0.1),
            "blue tray": _plate(),
        },
    )


def _medicine(
    occlusion_level: float = 0.1,
    visible: bool = True,
    reachable: bool = True,
    graspable: bool = True,
    ocr: float = 0.95,
    vlm: float = 0.90,
    detector: float = 0.95,
) -> object:
    from nesyro_gate import ObjectState

    return ObjectState(
        object_id="布洛芬",
        name="布洛芬",
        object_type="medicine_box",
        visible=visible,
        detector_confidence=detector,
        ocr_text="布洛芬",
        ocr_match_score=ocr,
        vlm_match_score=vlm,
        reachable=reachable,
        graspable=graspable,
        occlusion_level=occlusion_level,
    )


def _plate() -> object:
    from nesyro_gate import ObjectState

    return ObjectState(
        object_id="blue tray",
        name="blue tray",
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
    )


CASES = [
    ("high_confidence_grasp", Step(SkillName.GRASP, {"object": "布洛芬"}), _high_conf_scene),
    (
        "invisible_target",
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        lambda: _scene_with(visible=False),
    ),
    (
        "low_confidence_target",
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        lambda: _scene_with(ocr=0.15, vlm=0.2, detector=0.6),
    ),
    (
        "severely_occluded_target",
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        lambda: _scene_with(occlusion_level=0.8),
    ),
    (
        "not_reachable_target",
        Step(SkillName.GRASP, {"object": "布洛芬"}),
        lambda: _scene_with(reachable=False),
    ),
    (
        "place_not_in_gripper",
        Step(SkillName.PLACE, {"object": "布洛芬", "plate": "blue tray"}),
        _high_conf_scene,
    ),
    (
        "place_in_gripper",
        Step(SkillName.PLACE, {"object": "布洛芬", "plate": "blue tray"}),
        lambda: SceneMemory(
            objects={"布洛芬": _medicine(), "blue tray": _plate()},
            gripper_holding="布洛芬",
        ),
    ),
    (
        "observe_step",
        Step(SkillName.OBSERVE, {"query": "布洛芬"}),
        _high_conf_scene,
    ),
]


def _scene_with(**overrides) -> SceneMemory:
    scene = _high_conf_scene()
    for field, value in overrides.items():
        setattr(scene.objects["布洛芬"], field, value)
    return scene


def _compare_results(original: GateResult, unified, label: str) -> None:
    legacy = unified.legacy
    assert legacy.decision == original.decision, f"{label}: decision {legacy.decision} != {original.decision}"
    assert legacy.confidence == original.confidence, f"{label}: confidence"
    assert legacy.csc == original.csc, f"{label}: csc"
    assert legacy.lc == original.lc, f"{label}: lc"
    assert legacy.reason == original.reason, f"{label}: reason"
    assert legacy.failed_conditions == original.failed_conditions, f"{label}: failed_conditions"
    assert legacy.suggested_probe == original.suggested_probe, f"{label}: suggested_probe"
    assert legacy.suggested_constraint == original.suggested_constraint, f"{label}: suggested_constraint"


def _compare_scenes(original: SceneMemory, wrapped: SceneMemory, label: str) -> None:
    assert original.probe_counts == wrapped.probe_counts, f"{label}: probe_counts"
    assert original.gripper_holding == wrapped.gripper_holding, f"{label}: gripper_holding"
    assert set(original.objects) == set(wrapped.objects), f"{label}: object ids"
    for object_id in original.objects:
        assert original.objects[object_id] == wrapped.objects[object_id], f"{label}: object {object_id}"


def test_legacy_wrapper_matches_original_gate_on_all_cases() -> None:
    for label, step, scene_builder in CASES:
        original_scene = scene_builder()
        wrapped_scene = scene_builder()
        original = nesyro_gate(deepcopy(step), original_scene, _task())
        # LegacyNeSyRoGate.decide returns the raw unchanged GateResult
        wrapped = LegacyNeSyRoGate().decide(deepcopy(step), wrapped_scene, _task())
        assert isinstance(wrapped, GateResult)
        assert wrapped == original, f"{label}: wrapper result differs"
        _compare_scenes(original_scene, wrapped_scene, label)


def test_unified_legacy_mode_matches_original_gate_on_all_cases() -> None:
    for label, step, scene_builder in CASES:
        original_scene = scene_builder()
        unified_scene = scene_builder()
        original = nesyro_gate(deepcopy(step), original_scene, _task())
        unified = UnifiedNeSyRoGate(mode="legacy").decide(deepcopy(step), unified_scene, _task())
        _compare_results(original, unified, label)
        _compare_scenes(original_scene, unified_scene, label)
        # unified view consistent with the original decision
        assert unified.action == EXPECTED_ACTION[original.decision]
        assert unified.fell_back is False
        assert unified.effective_mode == "legacy"
        assert unified.learned is None


def test_legacy_decision_mapping() -> None:
    for decision, action in EXPECTED_ACTION.items():
        assert legacy_action_for(decision) == action


def test_probe_counting_side_effects_match_across_repeated_calls() -> None:
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    original_scene = _scene_with(visible=False)
    wrapped_scene = _scene_with(visible=False)

    decisions_original = []
    decisions_wrapped = []
    for _ in range(4):
        original = nesyro_gate(deepcopy(step), original_scene, _task())
        wrapped = UnifiedNeSyRoGate(mode="legacy").decide(deepcopy(step), wrapped_scene, _task())
        decisions_original.append(original.decision)
        decisions_wrapped.append(wrapped.legacy.decision)
        _compare_results(original, wrapped, "repeated_probe_calls")
        _compare_scenes(original_scene, wrapped_scene, "repeated_probe_calls")

    # max_probe=3: three PROBE_REQUIRED then the budget flips to REPLAN
    assert decisions_original == decisions_wrapped
    assert decisions_original[:3] == [GateDecision.PROBE_REQUIRED] * 3
    assert decisions_original[3] == GateDecision.REPLAN
    assert decisions_wrapped[3] == GateDecision.REPLAN


def test_unified_legacy_mode_preserves_threshold_and_max_probe() -> None:
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    original_scene = _scene_with(ocr=0.5, vlm=0.5, detector=0.5)
    wrapped_scene = _scene_with(ocr=0.5, vlm=0.5, detector=0.5)

    original = nesyro_gate(deepcopy(step), original_scene, _task(), threshold=0.99, max_probe=7)
    unified = UnifiedNeSyRoGate(mode="legacy", threshold=0.99, max_probe=7).decide(
        deepcopy(step), wrapped_scene, _task()
    )
    _compare_results(original, unified, "custom_threshold")


def test_equivalence_through_scene_memory_builder_observation() -> None:
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
                "depth_path": "runs/current_capture/wrist_depth.png",
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
        }
    }
    task = build_task_spec_for_pick_place("蒙脱石散", "blue tray")
    step = Step(SkillName.GRASP, {"object": "蒙脱石散"})

    original_scene = build_scene_memory_from_simple_robot(observations)
    wrapped_scene = build_scene_memory_from_simple_robot(observations)
    original = nesyro_gate(deepcopy(step), original_scene, task)
    unified = UnifiedNeSyRoGate(mode="legacy").decide(deepcopy(step), wrapped_scene, task)
    _compare_results(original, unified, "builder_observation")
    _compare_scenes(original_scene, wrapped_scene, "builder_observation")
