import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.simple_execution_runtime import SimpleExecutionRuntime
from core.actions import grasp, place
from core.context import RuntimeContext
from core.skill_base import SkillResult
from hardware.fr3 import MotionError
from hardware.robotiq import GripperError
from runtime.config import SimpleAgentSystemConfig


def runtime(tmp_path, **kwargs):
    return SimpleExecutionRuntime(SimpleAgentSystemConfig(goal='test',
        failure_report_path=str(tmp_path/'failure.json'),
        simple_memory_path=str(tmp_path/'memory.json'), **kwargs))


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_planner_retry_is_bounded_and_does_not_multiply_outer_budget(tmp_path, budget):
    task = runtime(tmp_path, grasp_planning_retry_attempts=budget)
    robot = SimpleNamespace(observations={'box': {'old': True}},
        grasp=Mock(side_effect=lambda _: SkillResult(False, {'failure_type': 'grasp_pose_not_found'}, error='no pose')))
    result = task._grasp_with_cache_retry(robot, None, 'box')
    assert robot.grasp.call_count == budget + 1
    assert result.data['automatic_replan'] is False
    assert len(result.data['grasp_planning_recovery']) == budget + 1
    if budget:
        assert not robot.observations


@pytest.mark.parametrize('error', [MotionError({'error': 'no IK'}), GripperError('serial lost')])
def test_hardware_errors_never_retry(tmp_path, error):
    task = runtime(tmp_path)
    robot = SimpleNamespace(observations={}, grasp=Mock(side_effect=error))
    with pytest.raises(type(error)):
        task._grasp_with_cache_retry(robot, None, 'box')
    assert robot.grasp.call_count == 1


def test_retry_success_preserves_recovery_history(tmp_path):
    task = runtime(tmp_path)
    robot = SimpleNamespace(observations={}, grasp=Mock(side_effect=[
        SkillResult(False, {'failure_type': 'grasp_pose_not_found'}), SkillResult(True, {})]))
    result = task._grasp_with_cache_retry(robot, None, 'box')
    assert result.success
    assert [row['success'] for row in result.data['grasp_planning_recovery']] == [False, True]


def test_hardware_report_keeps_phase_and_completed_prefix(tmp_path):
    task = runtime(tmp_path)
    task._save_memory = Mock()
    task._execution_checkpoint = dict(action=place('tray'), index=1,
                                     completed=[grasp('box').__dict__], holding='box')
    ctx = RuntimeContext()
    ctx.manipulation_progress = dict(stage='retreat', released=True, holding_state='empty')
    task._record_hardware_failure(MotionError({'error': 'stopped', 'host_log_path': '/log'}), ctx, Mock())
    report = json.loads((tmp_path/'failure.json').read_text())
    assert not report['automatic_replan']
    assert report['completed_actions'] == [grasp('box').__dict__]
    assert report['evidence']['holding_state'] == 'empty'
    assert report['evidence']['hardware_result']['host_log_path'] == '/log'
    assert report['failed_action_index'] == 1


def test_outer_runtime_reports_hardware_failure_and_finishes_cleanup(tmp_path, monkeypatch):
    task = runtime(tmp_path)
    camera, ctx = Mock(), RuntimeContext()
    monkeypatch.setattr('hardware.runtime.require_task_geometry', lambda _: None)
    monkeypatch.setattr('runtime.task_perception_pipeline.start_task_perception_pipeline', lambda *a: None)
    task._build_execution_context = Mock(return_value=(ctx, camera))
    task._resolve_action_targets = Mock(side_effect=lambda c, a: a)
    task._build_robot = Mock(return_value=Mock())
    task._load_memory = Mock();task._save_memory = Mock()
    task._capture_task_home_view = Mock(return_value={})
    task._preobserve_containers = Mock(return_value=True)
    task._run_actions = Mock(side_effect=MotionError({'error': 'no collision-free IK'}))
    assert task.run([grasp('box'), place('tray')]) is False
    assert json.loads((tmp_path/'failure.json').read_text())['failure_type'] == 'motion_error'


def test_failed_place_never_homes_or_requests_automatic_replan(tmp_path):
    task = runtime(tmp_path)
    robot = Mock()
    robot.gate_before_action.return_value = SkillResult(True, {})
    task._grasp_with_cache_retry = Mock(return_value=SkillResult(True, {}))
    task._place_and_verify = Mock(return_value=SkillResult(False, {}, error='release failed'))
    task._home_after_place = Mock()
    task._print_action_bbox_summary = Mock();task._log_gate_outcome_for_action = Mock();task._save_memory = Mock()
    assert not task._run_actions(robot, RuntimeContext(), Mock(), [grasp('box'), place('tray')])
    task._home_after_place.assert_not_called()
    assert not json.loads((tmp_path/'failure.json').read_text())['automatic_replan']


def test_stack_refresh_precedes_grasp_and_rejects_unknown_support(tmp_path):
    task = runtime(tmp_path, execute_robot=True)
    robot = Mock()
    robot.context.arm.config.place_height_mode = 'stack_rgbd'
    robot.observe_container.return_value = SkillResult(False, {}, error='patch not flat')
    actions = [grasp('box'), place('blue tray')]
    result = task._refresh_stack_surface(robot, actions, 0)
    assert not result.success and result.data['failure_type'] == 'stack_surface_unverified'
    robot._invalidate_container_cache.assert_called_once_with('blue tray')
    robot.observe_container.assert_called_once_with('blue tray')
    robot.home.assert_not_called()
