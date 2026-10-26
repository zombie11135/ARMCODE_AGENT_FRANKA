"""Task repair with fake skills only; never constructs a hardware session."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.plan_graph import PlanGraph, PlanNode
from agents.simple_execution_runtime import SimpleExecutionRuntime
from core.actions import grasp, place
from core.context import RuntimeContext
from core.skill_base import SkillResult
from runtime.config import SimpleAgentSystemConfig
from runtime.observation_memory import ObservationMemory
from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig


@pytest.fixture
def task(tmp_path, monkeypatch):
    cfg = SimpleAgentSystemConfig(
        goal='将西瓜霜放到蓝色盘子中，将抗病毒口服液放到粉色盘子中',
        enable_geometry_occlusion=True, project_root=str(tmp_path),
        plan_json=str(tmp_path/'plan.json'), replan_json=str(tmp_path/'replan.json'),
        replan_md=str(tmp_path/'replan.md'), generated_code=str(tmp_path/'actions.py'),
        failure_report_path=str(tmp_path/'failure.json'), simple_memory_path=str(tmp_path/'memory.json'))
    actions = [grasp('西瓜霜'), place('blue tray'), grasp('抗病毒口服液'), place('pink tray')]
    graph = PlanGraph(goal=cfg.goal)
    for index, action in enumerate(actions):
        graph.add_node(PlanNode(f'n{index}', action.target, action.op, action.target,
                               skill=action.op, args={'target': action.target}))
        if index:
            graph.add_edge(f'n{index-1}', f'n{index}')
    graph.save_json(cfg.plan_json)
    camera = SimpleNamespace(observation_memory=ObservationMemory())
    context = RuntimeContext(camera=camera)
    robot = SimpleRobotActions(context, SimpleRobotConfig(
        enable_geometry_occlusion=True, gate_experience_log_path='', nesyro_gate_trace_path=''))
    robot.occlusion_recovery_counts['西瓜霜'] = 1
    robot.geometry_buffer_slots_used = 1
    robot.geometry_recovery_state['history'].append({'event': 'previous_clear'})
    robot.place_plans['tray__blue'] = {'fixed_for_task': True}
    robot.place_plans['tray__pink'] = {'fixed_for_task': True}
    robot.place_counts.update(tray__blue=0, tray__pink=0)
    events = []
    blocked = []
    def gate(action, **kwargs):
        if action.op == 'grasp' and action.target == '抗病毒口服液':
            blocked.append(action.target)
            if len(blocked) == 1:
                return SkillResult(False, dict(failure_type='nesyro_gate_replan',
                    failed_conditions=['target_low_confidence']), error='new observation required')
        return SkillResult(True)
    robot.gate_before_action = Mock(side_effect=gate)
    robot.grasp = Mock(side_effect=lambda target: (events.append(('grasp', target)) or SkillResult(True)))
    robot.observe = Mock(side_effect=lambda target: (events.append(('observe', target)) or SkillResult(True)))
    def place_skill(target):
        events.append(('place', target))
        robot.place_counts['tray__'+target.split()[0]] += 1
        return SkillResult(True)
    robot.place = Mock(side_effect=place_skill)
    runtime = SimpleExecutionRuntime(cfg)
    monkeypatch.setattr(runtime, '_log_gate_outcome_for_action', lambda **kwargs: None)
    monkeypatch.setattr(runtime, '_print_action_bbox_summary', lambda *args: None)
    monkeypatch.setattr(runtime, '_place_and_verify', lambda **kwargs: robot.place(kwargs['action'].target))
    return SimpleNamespace(cfg=cfg, runtime=runtime, robot=robot, context=context, camera=camera,
                           actions=actions, events=events, graph=graph)


def test_repair_resumes_second_target_without_resetting_task_state(task):
    slots = task.robot.place_plans.copy()
    history = task.robot.geometry_recovery_state
    assert task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.events == [('grasp', '西瓜霜'), ('place', 'blue tray'),
                           ('observe', '抗病毒口服液'), ('grasp', '抗病毒口服液'), ('place', 'pink tray')]
    assert task.runtime.replan_attempts == 1
    assert task.robot.occlusion_recovery_counts == {'西瓜霜': 1}
    assert task.robot.geometry_buffer_slots_used == 1
    assert task.robot.geometry_recovery_state is history
    assert history['history'] == [{'event': 'previous_clear'}]
    assert task.robot.place_plans == slots
    assert all(task.robot.place_plans[key] is value for key, value in slots.items())
    assert task.robot.place_counts == {'tray__blue': 1, 'tray__pink': 1}
    assert task.camera.observation_memory.epoch == 1
    assert task.camera.observation_memory.scene_epoch == 0
    memory = json.loads(open(task.cfg.simple_memory_path).read())
    assert len(memory['completed_actions']) == 5
    repaired = PlanGraph.load_json(task.cfg.replan_json)
    assert repaired.metadata['same_session_repair']['buffer_slots_used'] == 1
    assert any('target_low_confidence' in constraint for constraint in repaired.constraints)


def test_repair_budget_stops_repeated_gate_failure(task):
    original = task.robot.gate_before_action.side_effect
    def blocked(action, **kwargs):
        if action.op == 'grasp' and action.target == '抗病毒口服液':
            return SkillResult(False, dict(failure_type='nesyro_gate_replan'), error='still blocked')
        return original(action, **kwargs)
    task.robot.gate_before_action.side_effect = blocked
    assert not task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.runtime.replan_attempts == 1
    assert task.events.count(('grasp', '西瓜霜')) == 1
    assert ('grasp', '抗病毒口服液') not in task.events
    report = json.loads(open(task.cfg.failure_report_path).read())
    assert report['same_session_replan_attempts'] == 1
    assert not report['automatic_replan']
    assert task.robot.geometry_buffer_slots_used == 1


@pytest.mark.parametrize('setting,value', [('enable_replan', False), ('max_replan_attempts', 0),
                                          ('resume_from_failure', False)])
def test_explicit_replan_controls_remain_effective(task, setting, value):
    setattr(task.cfg, setting, value)
    assert not task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.runtime.replan_attempts == 0
    task.robot.observe.assert_not_called()


@pytest.mark.parametrize('kind', ['geometry_occlusion_budget_exhausted', 'geometry_occlusion_clear_failed',
                                 'geometry_gripper_not_empty', 'gripper_fault'])
def test_geometry_and_hardware_stops_cannot_reset_budgets_through_replanning(task, kind):
    task.robot.gate_before_action = Mock(return_value=SkillResult(
        False, dict(failure_type=kind, automatic_replan=False), error='stop'))
    assert not task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.runtime.replan_attempts == 0
    task.robot.grasp.assert_not_called()
    assert task.robot.geometry_buffer_slots_used == 1


def test_started_grasp_failure_is_not_replayed(task):
    task.robot.gate_before_action = Mock(return_value=SkillResult(True))
    task.robot.grasp = Mock(return_value=SkillResult(
        False, dict(failure_type='gripper_fault'), error='motion may have started'))
    assert not task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.runtime.replan_attempts == 0
    assert task.robot.grasp.call_count == 1


@pytest.mark.parametrize('evidence', [
    {'automatic_replan': False},
    {'occlusion_recovery_attempts': [{'success': False, 'error': 'gripper fault during clear'}]},
])
def test_gate_replan_cannot_restart_a_manual_stop_or_legacy_manipulation_attempt(task, evidence):
    task.robot.gate_before_action = Mock(return_value=SkillResult(
        False, dict(failure_type='nesyro_gate_replan', **evidence), error='stop'))
    assert not task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.runtime.replan_attempts == 0
    task.robot.grasp.assert_not_called()


def test_place_gate_failure_while_holding_is_not_replayed(task):
    def gate(action, **kwargs):
        if action.op == 'place':
            assert kwargs['current_holding'] == '西瓜霜'
            return SkillResult(False, dict(failure_type='nesyro_gate_replan'), error='holding')
        return SkillResult(True)
    task.robot.gate_before_action = Mock(side_effect=gate)
    assert not task.runtime._run_actions(task.robot, task.context, task.camera, task.actions)
    assert task.runtime.replan_attempts == 0
    task.robot.grasp.assert_called_once_with('西瓜霜')
    task.robot.place.assert_not_called()


def test_registered_object_ids_resume_after_the_completed_first_pair(task):
    target_ids = {'西瓜霜': 'ref_a09c4c218d325a46971f', '抗病毒口服液': 'ref_2a194e828c2b526e8f7b'}
    for node in task.graph.nodes:
        if node.skill == 'grasp':
            node.args['target'] = target_ids[node.args['target']]
    task.graph.save_json(task.cfg.plan_json)
    actions = [grasp(target_ids['西瓜霜']), place('blue tray'),
               grasp(target_ids['抗病毒口服液']), place('pink tray')]
    blocked = []
    def gate(action, **kwargs):
        if action.op == 'grasp' and action.target == target_ids['抗病毒口服液']:
            blocked.append(action.target)
            if len(blocked) == 1:
                return SkillResult(False, dict(failure_type='nesyro_gate_replan'), error='refresh')
        return SkillResult(True)
    task.robot.gate_before_action = Mock(side_effect=gate)
    assert task.runtime._run_actions(task.robot, task.context, task.camera, actions)
    assert task.events == [('grasp', target_ids['西瓜霜']), ('place', 'blue tray'),
        ('observe', target_ids['抗病毒口服液']), ('grasp', target_ids['抗病毒口服液']), ('place', 'pink tray')]


def test_agent_result_includes_repairs_performed_inside_execution(task, monkeypatch):
    import agents.simple_agent_system as system
    import hardware.runtime
    monkeypatch.setattr(hardware.runtime, 'require_task_geometry', lambda config: None)
    fake_runtime = SimpleNamespace(run=Mock(return_value=True), replan_attempts=1,
                                  replan_paths=dict(replan_json=task.cfg.replan_json, replan_md=task.cfg.replan_md))
    monkeypatch.setattr(system, 'SimpleExecutionRuntime', lambda config: fake_runtime)
    agent = system.SimpleArmCodeAgentSystem(task.cfg)
    monkeypatch.setattr(agent, '_planning_stage', lambda: task.graph)
    monkeypatch.setattr(agent, '_coding_stage', lambda *args, **kwargs: task.actions)
    result = agent._run_task()
    assert result.success
    assert result.replan_attempts == 1
    assert result.replan_json == task.cfg.replan_json
    fake_runtime.run.assert_called_once_with(task.actions)
