"""Phase-1 learnable gate tests: smoke / fallback / shadow.

The MLP checkpoint is created on the fly with random weights (no training);
these tests only verify the interface contract.
"""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

from nesyro_gate import (
    GateDecision,
    GateInput,
    LearnableNeSyRoGate,
    ProbeAction,
    SceneMemory,
    SkillName,
    Step,
    TaskSpec,
    UnifiedNeSyRoGate,
    build_task_spec_for_pick_place,
    nesyro_gate,
)
from nesyro_gate.learnable_gate import ACTION_ORDER, GATE_FEATURE_NAMES, build_mlp


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _save_random_checkpoint() -> str:
    import torch

    tmpdir = Path(tempfile.mkdtemp(prefix="gate_ckpt_"))
    model = build_mlp(len(GATE_FEATURE_NAMES))
    path = tmpdir / "learnable_gate.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


def _task() -> TaskSpec:
    return build_task_spec_for_pick_place("布洛芬", "blue tray")


def _medicine(occlusion_level: float = 0.1):
    from nesyro_gate import ObjectState

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


def _grasp_scene(occlusion_level: float = 0.1) -> SceneMemory:
    return SceneMemory(objects={"布洛芬": _medicine(occlusion_level)})


def test_learned_gate_smoke_outputs_4way_probabilities() -> None:
    if not _torch_available():
        print("SKIP test_learned_gate_smoke: torch unavailable")
        return

    gate = LearnableNeSyRoGate(checkpoint_path=_save_random_checkpoint())
    assert gate.available is True
    assert gate.load_error is None

    gate_input = GateInput(
        current_step={"op": "grasp", "target": "布洛芬"},
        target_confidence=0.9,
        target_visibility=1.0,
        occlusion_score=0.1,
        visual_probe_count=1,
        manipulation_probe_count=0,
        previous_action_result=1.0,
    )
    output = gate.predict(gate_input)

    assert set(output.action_scores.keys()) == {
        ProbeAction.EXECUTE.value,
        ProbeAction.VISUAL_PROBE.value,
        ProbeAction.MANIPULATION_PROBE.value,
        ProbeAction.REPLAN.value,
    }
    total = sum(output.action_scores.values())
    assert abs(total - 1.0) < 1e-5
    assert 0.0 <= output.confidence <= 1.0
    assert output.action in ACTION_ORDER
    assert abs(output.action_scores[output.action.value] - output.confidence) < 1e-6

    # empty / default input also produces a valid distribution
    empty_output = gate.predict(None)
    assert abs(sum(empty_output.action_scores.values()) - 1.0) < 1e-5
    # deterministic for the same input (eval mode, no dropout)
    repeat_output = gate.predict(gate_input)
    assert repeat_output.action_scores == output.action_scores


def test_learned_gate_unavailable_without_checkpoint() -> None:
    gate = LearnableNeSyRoGate(checkpoint_path="no/such/checkpoint.pt")
    assert gate.available is False
    assert gate.load_error == "no checkpoint file"
    try:
        gate.predict(GateInput())
        raise AssertionError("predict should raise when unavailable")
    except RuntimeError:
        pass


def test_learned_mode_falls_back_to_legacy_when_checkpoint_missing() -> None:
    unified = UnifiedNeSyRoGate(
        mode="learned",
        checkpoint_path="no/such/checkpoint.pt",
    )
    scene = _grasp_scene()
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    result = unified.decide(deepcopy(step), scene, _task())

    assert result.fell_back is True
    assert result.effective_mode == "learned_fallback_legacy"
    assert result.learned is None

    # decisions match the original legacy gate exactly
    original_scene = _grasp_scene()
    original = nesyro_gate(deepcopy(step), original_scene, _task())
    assert result.legacy.decision == original.decision
    assert result.action == ProbeAction.EXECUTE
    assert result.legacy.decision == GateDecision.EXECUTE


def test_shadow_mode_records_learned_but_executes_legacy() -> None:
    if not _torch_available():
        print("SKIP test_shadow_mode: torch unavailable")
        return

    checkpoint = _save_random_checkpoint()
    unified = UnifiedNeSyRoGate(mode="shadow", checkpoint_path=checkpoint)
    assert unified.learnable_available is True

    # legacy says REPLAN (occluded); learned may disagree but must not win
    scene = _grasp_scene(occlusion_level=0.8)
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    result = unified.decide(deepcopy(step), scene, _task())

    assert result.gate_mode == "shadow"
    assert result.effective_mode == "shadow"
    assert result.fell_back is False
    assert result.learned is not None
    assert set(result.learned.action_scores.keys()) == {
        ProbeAction.EXECUTE.value,
        ProbeAction.VISUAL_PROBE.value,
        ProbeAction.MANIPULATION_PROBE.value,
        ProbeAction.REPLAN.value,
    }
    assert result.legacy.decision == GateDecision.REPLAN
    assert result.action == ProbeAction.REPLAN  # legacy drives robot in shadow


def test_shadow_mode_without_checkpoint_still_executes_legacy() -> None:
    unified = UnifiedNeSyRoGate(mode="shadow", checkpoint_path="")
    scene = _grasp_scene()
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    result = unified.decide(deepcopy(step), scene, _task())

    assert result.gate_mode == "shadow"
    assert result.learned is None
    assert result.fell_back is True
    assert result.legacy.decision == GateDecision.EXECUTE
    assert result.action == ProbeAction.EXECUTE


def test_learned_mode_with_checkpoint_uses_learned_action() -> None:
    if not _torch_available():
        print("SKIP test_learned_mode_with_checkpoint: torch unavailable")
        return

    checkpoint = _save_random_checkpoint()
    unified = UnifiedNeSyRoGate(mode="learned", checkpoint_path=checkpoint)
    scene = _grasp_scene()
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    result = unified.decide(deepcopy(step), scene, _task())

    assert result.gate_mode == "learned"
    assert result.effective_mode == "learned"
    assert result.fell_back is False
    assert result.learned is not None
    assert result.action == result.learned.action
    # legacy always recorded for comparison
    assert result.legacy.decision == GateDecision.EXECUTE


def test_gate_input_enrichment_fills_csc_and_lc_from_legacy() -> None:
    unified = UnifiedNeSyRoGate(mode="learned", checkpoint_path="")
    scene = _grasp_scene()
    step = Step(SkillName.GRASP, {"object": "布洛芬"})
    result = unified.decide(deepcopy(step), scene, _task())

    gate_input = result.gate_input
    assert gate_input.target_confidence == result.legacy.csc
    assert gate_input.prerequisite_state["lc"] == result.legacy.lc
    assert gate_input.prerequisite_state["failed_conditions"] == []


def test_shadow_mode_at_runtime_executes_legacy_and_logs_learned() -> None:
    """Runtime-level shadow guarantee.

    With an occluded target the legacy gate returns REPLAN; the learned gate
    (random weights) may disagree, but in shadow mode the robot must follow
    the legacy decision while the learned output is recorded in the
    experience log.
    """

    if not _torch_available():
        print("SKIP test_shadow_mode_at_runtime: torch unavailable")
        return

    import json
    import tempfile
    from pathlib import Path

    from core.skill_base import SkillResult
    from skill_lib.simple_actions import SimpleAction, SimpleRobotActions, SimpleRobotConfig

    tmpdir = Path(tempfile.mkdtemp(prefix="gate_shadow_rt_"))
    log_path = tmpdir / "gate_experience.jsonl"
    checkpoint = _save_random_checkpoint()
    target = "西瓜霜"

    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            nesyro_gate_trace_path="",
            gate_mode="shadow",
            learned_gate_checkpoint_path=checkpoint,
            gate_experience_log_path=str(log_path),
            nesyro_gate_max_probe=0,
        ),
    )
    robot.observations[target] = {
        "kind": "medicine",
        "target": target,
        "selected_instance": {
            "object_id": "obj_1",
            "bbox": [10, 20, 110, 120],
            "confidence": 0.92,
            "ocr_text": target,
            "ocr_texts": [target],
        },
        "grasp_input": {
            "object_id": "obj_1",
            "bbox": [10, 20, 110, 120],
            "depth_path": "offline-test-fixture://unavailable-depth.png",
            "ocr_text": target,
        },
        "vlm_occlusion": {
            "target_visible": True,
            "target_occluded": True,
            "occlusion_level": 0.72,
            "blocking_object_visible": True,
            "grasp_safe": False,
            "confidence": 0.9,
        },
        "result": {
            "selection_result": {
                "score": 0.94,
                "ranked_candidates": [
                    {
                        "object_id": "obj_1",
                        "score": 0.94,
                        "raw_text_score": 1.0,
                        "detection_confidence": 0.92,
                    }
                ],
            },
            "ocr_verification": {"confidence": 1.0},
        },
    }

    result = robot.gate_before_action(SimpleAction(op="grasp", target=target))
    # legacy blocks the occluded grasp -> runtime failure, learned must not win
    assert result.success is False
    assert result.data["failure_type"] == "nesyro_gate_replan"
    assert result.data["gate"]["decision"] == "REPLAN"
    assert result.data["gate"]["gate_mode"] == "shadow"
    assert result.data["gate"]["learned"] is not None

    lines = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 1
    decision = lines[0]
    assert decision["event"] == "gate_decision"
    assert decision["gate_mode"] == "shadow"
    assert decision["legacy_decision"] == "REPLAN"
    assert decision["learned_decision"] in {
        "EXECUTE",
        "VISUAL_PROBE",
        "MANIPULATION_PROBE",
        "REPLAN",
    }
    assert decision["selected_action"] == "REPLAN"  # robot follows legacy
    robot.log_gate_outcome(primitive_success=False, execution_failure_type="nesyro_gate_replan")
