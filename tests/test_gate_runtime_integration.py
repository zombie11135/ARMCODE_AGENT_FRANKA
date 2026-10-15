"""Tests for NeSyRo Gate integration in simple robot actions."""

from __future__ import annotations

import json

from core.skill_base import SkillResult
from skill_lib.simple_actions import SimpleAction, SimpleRobotActions, SimpleRobotConfig


def _medicine_observation(target: str):
    return {
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
            "depth_path": "runs/current_capture/wrist_depth.png",
            "ocr_text": target,
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


def _occluded_medicine_observation(target: str):
    observation = _medicine_observation(target)
    observation["vlm_occlusion"] = {
        "target_visible": True,
        "target_partially_visible": True,
        "target_occluded": True,
        "occlusion_level": 0.65,
        "blocking_object_visible": True,
        "blocking_object_description": "another medicine box lies on top",
        "grasp_safe": False,
        "confidence": 0.95,
        "recommendation": "replan",
        "evidence": {},
    }
    return observation


def test_gate_before_grasp_runs_probe_when_observation_missing() -> None:
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(nesyro_gate_trace_path="", gate_experience_log_path=""),
    )

    def fake_observe(target: str) -> SkillResult:
        robot.observations[target] = _medicine_observation(target)
        return SkillResult(success=True, data={"observed": target})

    robot.observe = fake_observe

    result = robot.gate_before_action(SimpleAction(op="grasp", target="蒙脱石散"))

    assert result.success is True
    assert "蒙脱石散" in robot.observations
    assert result.data["gate"]["decision"] == "EXECUTE"


def test_gate_blocks_place_when_gripper_empty() -> None:
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(nesyro_gate_trace_path="", gate_experience_log_path=""),
    )

    result = robot.gate_before_action(SimpleAction(op="place", target="blue tray"))

    assert result.success is False
    assert result.data["failure_type"] == "nesyro_gate_empty_gripper"


def test_enabled_vlm_occlusion_gate_requires_api_config() -> None:
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            nesyro_gate_trace_path="", gate_experience_log_path="",
            enable_vlm_occlusion_gate=True,
        ),
    )
    robot.observations["布洛芬"] = _medicine_observation("布洛芬")

    result = robot.gate_before_action(SimpleAction(op="grasp", target="布洛芬"))

    assert result.success is False
    assert result.data["failure_type"] == "nesyro_gate_vlm_config_missing"


def test_gate_clears_occlusion_reobserves_and_then_executes() -> None:
    target = "西瓜霜"
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            execute=True,
            nesyro_gate_trace_path="", gate_experience_log_path="",
            nesyro_gate_max_probe=0,
            enable_occlusion_recovery=True,
            occlusion_recovery_max_attempts=1,
        ),
    )
    robot.observations[target] = _occluded_medicine_observation(target)

    def fake_recovery(recovery_target: str) -> SkillResult:
        robot.occlusion_recovery_counts[recovery_target] = 1
        return SkillResult(success=True, data={"blocker": {"object_id": "blocker"}})

    def fake_observe(observe_target: str) -> SkillResult:
        robot.observations[observe_target] = _medicine_observation(observe_target)
        return SkillResult(success=True, data={"observed": observe_target})

    robot._run_occlusion_recovery = fake_recovery
    robot.observe = fake_observe

    result = robot.gate_before_action(SimpleAction(op="grasp", target=target))

    assert result.success is True
    assert result.data["gate"]["decision"] == "EXECUTE"
    assert len(result.data["gate_attempts"]) == 2


def test_gate_keeps_replan_when_occlusion_recovery_fails() -> None:
    target = "西瓜霜"
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            execute=True,
            nesyro_gate_trace_path="", gate_experience_log_path="",
            nesyro_gate_max_probe=0,
            enable_occlusion_recovery=True,
            occlusion_recovery_max_attempts=1,
        ),
    )
    robot.observations[target] = _occluded_medicine_observation(target)

    def fake_recovery(recovery_target: str) -> SkillResult:
        robot.occlusion_recovery_counts[recovery_target] = 1
        return SkillResult(
            success=False,
            data={"failure_type": "occlusion_recovery_blocker_not_found"},
            error="blocker not found",
        )

    robot._run_occlusion_recovery = fake_recovery

    result = robot.gate_before_action(SimpleAction(op="grasp", target=target))

    assert result.success is False
    assert result.data["failure_type"] == "nesyro_gate_replan"
    assert result.data["occlusion_recovery_attempts"][0]["success"] is False
    assert "target_occluded" in result.data["failed_conditions"]


def test_occlusion_recovery_changes_view_after_foreground_depth_failure() -> None:
    target = "西瓜霜"
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            execute=True,
            enable_vlm_occlusion_gate=True,
            enable_occlusion_recovery=True,
            occlusion_active_view_enabled=True,
            occlusion_active_view_y_offsets_mm=[0.0],
            occlusion_active_view_wait_s=0.0,
        ),
    )
    robot.observations[target] = _occluded_medicine_observation(target)
    clear_results = [
        SkillResult(
            success=False,
            data={"failure_type": "occlusion_recovery_foreground_depth_invalid"},
            error="foreground support too low",
        ),
        SkillResult(success=True, data={"blocker": {"bbox": [20, 20, 80, 80]}}),
    ]

    robot._invoke_clear_occlusion_skill = lambda _: clear_results.pop(0)
    robot._move_xarm = lambda _: True

    def fake_observe(_: str) -> SkillResult:
        robot.observations[target] = _occluded_medicine_observation(target)
        return SkillResult(success=True, data={"observed": target})

    robot._observe_medicine_once = fake_observe
    robot._run_vlm_occlusion_gate_if_enabled = lambda _: SkillResult(success=True)

    result = robot._run_occlusion_recovery(target)

    assert result.success is True
    assert len(result.data["active_view_attempts"]) == 2
    json.dumps(result.data)
