"""Phase-1 gate experience logging tests.

Verify: JSONL lines are written with the required schema, outcome records
reference the decision record, and logging failures never block the gate /
robot main flow.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.skill_base import SkillResult
from nesyro_gate import GateExperienceLogger, decision_record, outcome_record
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
            "depth_path": "offline-test-fixture://unavailable-depth.png",
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


def _read_lines(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_append_writes_jsonl_lines() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="gate_log_"))
    logger = GateExperienceLogger(path=str(tmpdir / "gate_experience.jsonl"))

    record = decision_record(
        decision_id=logger.new_record_id(),
        timestamp="2026-09-03T10:00:00",
        episode_id=logger.episode_id,
        gate_mode="legacy",
        effective_mode="legacy",
        fell_back=False,
        current_step={"op": "grasp", "target": "布洛芬"},
        gate_input={},
        selected_action="EXECUTE",
        action_scores={"EXECUTE": 1.0},
        confidence=0.9,
        legacy_decision="EXECUTE",
        learned_decision=None,
        reason="passed",
        trace_reference="runs/current_plan/gate_trace.jsonl",
    )
    assert logger.append(record) is True
    assert logger.append(record) is True

    lines = _read_lines(tmpdir / "gate_experience.jsonl")
    assert len(lines) == 2
    assert lines[0]["event"] == "gate_decision"
    assert lines[0]["gate_mode"] == "legacy"
    assert lines[0]["selected_action"] == "EXECUTE"
    assert lines[0]["legacy_decision"] == "EXECUTE"


def test_append_never_raises_on_bad_path() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="gate_log_bad_"))
    blocker = tmpdir / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    # parent of the log path is a regular file -> mkdir/open must fail
    logger = GateExperienceLogger(path=str(blocker / "gate_experience.jsonl"))

    result = logger.append({"event": "gate_decision"})
    assert result is False  # swallowed failure, no exception


def test_gate_before_action_writes_decision_and_outcome_records() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="gate_log_rt_"))
    log_path = tmpdir / "gate_experience.jsonl"
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            nesyro_gate_trace_path="",
            gate_experience_log_path=str(log_path),
        ),
    )

    def fake_observe(target: str) -> SkillResult:
        robot.observations[target] = _medicine_observation(target)
        return SkillResult(success=True, data={"observed": target})

    robot.observe = fake_observe

    result = robot.gate_before_action(
        SimpleAction(op="grasp", target="蒙脱石散"),
        previous_action_result=None,
        step_index=0,
        total_steps=2,
    )
    assert result.success is True

    outcome_written = robot.log_gate_outcome(
        primitive_success=True,
        skill_success=True,
        execution_failure_type=None,
        task_progress_before=0.0,
        task_progress_after=0.5,
    )
    assert outcome_written is True

    lines = _read_lines(log_path)
    assert len(lines) == 2

    decision, outcome = lines
    assert decision["event"] == "gate_decision"
    assert decision["gate_mode"] == "legacy"
    assert decision["effective_mode"] == "legacy"
    assert decision["selected_action"] == "EXECUTE"
    assert decision["legacy_decision"] == "EXECUTE"
    assert decision["learned_decision"] is None
    assert decision["current_step"] == {"op": "grasp", "target": "蒙脱石散"}
    features = decision["gate_input_features"]
    # CSC/LC enriched from the legacy result and recorded
    assert features["target_confidence"] is not None
    assert features["prerequisite_state"]["lc"] is not None
    assert features["task_stage"]["step_index"] == 0
    assert features["task_stage"]["total_steps"] == 2

    assert outcome["event"] == "gate_outcome"
    assert outcome["decision_id"] == decision["decision_id"]
    assert outcome["episode_id"] == decision["episode_id"]
    assert outcome["primitive_success"] is True
    assert outcome["skill_success"] is True
    assert outcome["task_progress_before"] == 0.0
    assert outcome["task_progress_after"] == 0.5


def test_logging_failure_does_not_block_gate_flow() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="gate_log_block_"))
    blocker = tmpdir / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    robot = SimpleRobotActions(
        context=None,
        config=SimpleRobotConfig(
            nesyro_gate_trace_path="",
            gate_experience_log_path=str(blocker / "gate_experience.jsonl"),
        ),
    )

    def fake_observe(target: str) -> SkillResult:
        robot.observations[target] = _medicine_observation(target)
        return SkillResult(success=True, data={"observed": target})

    robot.observe = fake_observe

    # gate flow must still complete normally despite the unwritable log path
    result = robot.gate_before_action(SimpleAction(op="grasp", target="蒙脱石散"))
    assert result.success is True
    assert result.data["gate"]["decision"] == "EXECUTE"

    robot.log_gate_outcome(primitive_success=True)
    assert robot._last_gate_decision_id is not None


def test_outcome_record_schema() -> None:
    record = outcome_record(
        decision_id="gate-1",
        timestamp="2026-09-03T10:00:00",
        episode_id="ep",
        primitive_success=True,
        skill_success=True,
        task_progress_before=0.0,
        task_progress_after=1.0,
        execution_failure_type=None,
        visual_change=None,
        trace_reference=None,
    )
    assert record["event"] == "gate_outcome"
    assert record["decision_id"] == "gate-1"
    assert record["visual_change"] is None
