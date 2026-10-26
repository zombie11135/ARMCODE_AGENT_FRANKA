"""Cross-module state and failure behavior of the simple action facade.

Compound skills and devices are replaced; these tests never load models or move
hardware. They exercise the facade so delegation and caller hooks are included.
"""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from core.context import RuntimeContext
from core.skill_base import SkillResult
from skill_lib.simple_actions import (
    SimpleRobotActions, SimpleRobotConfig, grasp, observe, place,
)


@pytest.fixture
def place_plan():
    return {
        "center_base_xyz": [0.3, 0.1, 0.2],
        "approach_pose_mmrad": [300.0, 100.0, 350.0, 3.14, 0.0, 0.0],
        "place_pose_mmrad": [300.0, 100.0, 250.0, 3.14, 0.0, 0.0],
        "lift_pose_mmrad": [300.0, 100.0, 400.0, 3.14, 0.0, 0.0],
    }


def test_restored_caches_are_shared_by_observation_placement_and_gate_probe(monkeypatch, place_plan):
    from skill_lib.combound.place_to_container import PlaceToContainerSkill

    plan_skill = Mock(return_value=SkillResult(True, data=place_plan))
    monkeypatch.setattr(PlaceToContainerSkill, "run", plan_skill)
    arm = Mock()
    robot = SimpleRobotActions(
        RuntimeContext(xarm=arm),
        SimpleRobotConfig(enable_place_slot_offsets=True, place_slot_spacing_mm=25),
    )
    # External memory restoration replaces these dictionaries on the facade.
    observations, plans, counts = {}, {}, {}
    robot.observations, robot.place_plans, robot.place_counts = observations, plans, counts

    assert robot.observe_container("pink tray").success
    assert observations["pink tray"]["place_plan"] is plans["tray__pink"]
    assert robot.observe_container("pink tray").success
    plan_skill.assert_called_once()
    assert plan_skill.call_args.kwargs["execute"] is False

    first = robot.place("pink tray")
    second = robot.place("pink tray")
    assert first.success and second.success
    assert first.data["dry_run"] and second.data["dry_run"]
    assert counts["tray__pink"] == 2
    assert first.data["place_plan"]["place_pose_mmrad"][1] == 100
    assert second.data["place_plan"]["place_pose_mmrad"][1] == 75
    assert plans["tray__pink"]["place_pose_mmrad"][1] == 100
    arm.set_position.assert_not_called()

    # A Gate probe reuses a fixed container without observing or moving again.
    robot.observe = Mock(return_value=SkillResult(True, data={"refreshed": True}))
    result = robot._run_gate_probe(
        place("pink tray"), current_holding="box", suggested_probe="observe_container", reason="test"
    )
    assert result.success
    assert "tray__pink" in plans
    assert "pink tray" in observations
    assert result.data['container_plan_reused'] is True
    robot.observe.assert_not_called()
    # Explicit invalidation remains available, including aliases of this tray.
    robot._invalidate_container_cache('pink tray')
    assert 'tray__pink' not in plans and 'pink tray' not in observations


@pytest.mark.parametrize("failure", [None, "approach", "place", "gripper", "lift"])
def test_placement_stops_at_the_first_failed_motion_or_release(place_plan, failure):
    events = []
    labels = {350.0: "approach", 250.0: "place", 400.0: "lift"}

    def move(**kwargs):
        label = labels[kwargs["z"]]
        events.append(label)
        return 1 if failure == label else 0

    def release():
        events.append("gripper")
        return failure != "gripper"

    arm = Mock()
    arm.set_position.side_effect = move
    robot = SimpleRobotActions(
        RuntimeContext(xarm=arm), SimpleRobotConfig(execute=True), gripper_open=release
    )
    robot.place_plans["tray__pink"] = place_plan
    result = robot.place("pink tray")
    sequence = ["approach", "place", "gripper", "lift"]
    assert events == (sequence if failure is None else sequence[:sequence.index(failure) + 1])
    assert result.success is (failure is None)
    if failure is not None:
        expected = "gripper open failed" if failure == "gripper" else f"move {failure} pose failed"
        assert result.error == expected


@pytest.mark.parametrize("grasp_success", [True, False])
def test_grasp_updates_the_observation_cache_only_on_success(monkeypatch, grasp_success):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

    observation = {"selected_instance": {"object_id": "old"}, "grasp_input": {"bbox": [1, 2, 3, 4]}}
    grasp_data = {"selected_instance": {"object_id": "new"}, "grasp_input": {"bbox": [2, 3, 4, 5]}}
    run_skill = Mock(side_effect=[
        SkillResult(True, data=observation),
        SkillResult(grasp_success, data=grasp_data, error=None if grasp_success else "no candidate"),
    ])
    monkeypatch.setattr(FindAndGraspMedicineSkill, "run", run_skill)
    opened, closed = Mock(), Mock()
    robot = SimpleRobotActions(
        RuntimeContext(), SimpleRobotConfig(), gripper_open=opened, gripper_close=closed
    )
    restored = {}
    robot.observations = restored
    assert robot._observe_medicine_once("box").success
    cached = restored["box"]
    result = robot.grasp("box")
    assert result.success is grasp_success
    assert robot.observations is restored
    if grasp_success:
        assert restored["box"]["grasp_input"] is grasp_data["grasp_input"]
    else:
        assert restored["box"] is cached
        assert result.error == "no candidate"
    observe_args, grasp_args = [call.kwargs for call in run_skill.call_args_list]
    assert observe_args["plan_grasp"] is False
    assert observe_args["execute"] is False
    assert grasp_args["plan_grasp"] is True
    assert grasp_args["execute"] is False
    assert grasp_args["gripper_open"] is opened
    assert grasp_args["gripper_close"] is closed
    opened.assert_not_called()
    closed.assert_not_called()


def test_action_sequence_preserves_hooks_and_stops_after_failed_grasp(monkeypatch, place_plan):
    from skill_lib.combound.place_to_container import PlaceToContainerSkill

    monkeypatch.setattr(PlaceToContainerSkill, "run", Mock(return_value=SkillResult(True, place_plan)))
    events = []
    config = SimpleRobotConfig(
        enable_nesyro_gate=False, observe_wait_s=0,
        nesyro_gate_trace_path="", gate_experience_log_path="",
    )
    robot = SimpleRobotActions(RuntimeContext(), config)

    def observe_once(target):
        events.append(("observe", target))
        robot.observations[target] = {"selected_instance": {"object_id": "box"}, "grasp_input": {}}
        return SkillResult(True)

    def fail_grasp(target):
        events.append(("grasp", target))
        return SkillResult(False, error="grasp failed")

    robot._observe_medicine_once = observe_once
    robot.grasp = fail_grasp
    robot.place = Mock()
    result = robot.run([observe("药盒"), grasp("药盒"), place("pink tray")])
    assert not result.success
    assert result.error == "grasp failed"
    assert events == [("observe", "药盒"), ("grasp", "药盒")]
    assert result.data["failed_action"] == {"op": "grasp", "target": "药盒"}
    assert "tray__pink" in robot.place_plans  # Container pre-observation still ran first.
    robot.place.assert_not_called()


def test_next_run_localizes_container_again_instead_of_reusing_previous_task(monkeypatch, place_plan):
    from copy import deepcopy
    from skill_lib.combound.place_to_container import PlaceToContainerSkill
    from runtime.observation_memory import StaleObservationError
    plan_skill = Mock(side_effect=lambda **kwargs: SkillResult(True, deepcopy(place_plan)))
    monkeypatch.setattr(PlaceToContainerSkill, 'run', plan_skill)
    robot = SimpleRobotActions(RuntimeContext(), SimpleRobotConfig(
        enable_nesyro_gate=False, nesyro_gate_trace_path='', gate_experience_log_path=''))
    robot.home = Mock(return_value=SkillResult(True))
    robot.grasp = Mock(return_value=SkillResult(True))
    actions = [grasp('medicine'), place('blue tray')]
    assert robot.run(actions).success
    previous = deepcopy(robot.place_plans['tray__blue'])
    assert robot.place_counts['tray__blue'] == 1
    assert robot.run(actions).success
    assert plan_skill.call_count == 2
    assert robot.place_counts['tray__blue'] == 1
    assert previous['task_container']['task_id'] != robot.place_plans['tray__blue']['task_container']['task_id']
    robot._move_xarm = Mock(side_effect=AssertionError('Old task must not move'))
    with pytest.raises(StaleObservationError, match='not registered for this task'):
        robot._execute_cached_place(previous)
    robot._move_xarm.assert_not_called()


def test_gate_trace_serializes_nested_values_after_extraction(tmp_path):
    trace = tmp_path / "gate.jsonl"
    robot = SimpleRobotActions(None, SimpleRobotConfig(nesyro_gate_trace_path=str(trace)))
    robot._append_gate_trace({
        "image": tmp_path / "color.png",
        "nested": [{"points": np.array([[1.0, 2.0, 3.0]])}],
        "tuple": ("a", "b"),
    })
    assert json.loads(trace.read_text()) == {
        "image": str(tmp_path / "color.png"),
        "nested": [{"points": [[1.0, 2.0, 3.0]]}],
        "tuple": ["a", "b"],
    }


def test_importing_facade_does_not_load_skill_model_or_device_backends():
    code = """
import sys
class BlockBackends:
    def find_spec(self, fullname, path=None, target=None):
        roots = {'hardware', 'nesyro_gate', 'torch', 'cv2', 'requests', 'serial', 'pyrealsense2', 'xarm'}
        skill_packages = ('skill_lib.combound', 'skill_lib.manipulation', 'skill_lib.evaluation')
        if fullname.split('.')[0] in roots or fullname.startswith(skill_packages):
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig, grasp
robot = SimpleRobotActions(None, SimpleRobotConfig(enable_nesyro_gate=False))
assert robot.gate_before_action(grasp('box')).success
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
