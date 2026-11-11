import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.hybrid_execution import HybridTaskRunner
from agents.plan_graph import PlanGraph, PlanNode
from agents.simple_execution_runtime import SimpleExecutionRuntime
from core.actions import grasp, place
from core.context import RuntimeContext
from core.skill_base import SkillResult
from hardware.fr3 import MotionError
from runtime.config import SimpleAgentSystemConfig
from runtime.task_state import recovery_decision
from runtime.timing import task_timing


def program(obj, tray):
    return f'def policy():\n    if observe({obj!r}):\n        pick_place({obj!r}, {tray!r})\n'


@pytest.fixture
def rig(tmp_path, monkeypatch):
    actions = [grasp('a'), place('tray'), grasp('b'), place('tray')]
    config = SimpleAgentSystemConfig(goal='two boxes', policy_backend='hybrid',
        cap_policy_source=program('a', 'tray'), plan_json=str(tmp_path/'plan.json'),
        generated_code=str(tmp_path/'program.py'), replan_json=str(tmp_path/'replan.json'),
        failure_report_path=str(tmp_path/'failure.json'), simple_memory_path=str(tmp_path/'memory.json'))
    graph = PlanGraph(goal=config.goal)
    for index, action in enumerate(actions):
        graph.add_node(PlanNode(str(index), action.target, action.op, action.target,
                               skill=action.op, args={'target':action.target}))
        if index:
            graph.add_edge(str(index-1),str(index))
    graph.save_json(config.plan_json)
    runtime = SimpleExecutionRuntime(config)
    context = RuntimeContext(camera=None)
    runtime._active_context = context
    robot = SimpleNamespace(context=context, observations={}, observe=Mock(return_value=SkillResult(True, {})))
    runtime._run_actions = Mock(return_value=True)
    runner = HybridTaskRunner(runtime, robot, context, None, actions, actions, Mock())
    generated = []
    def generate(goal, actions, output, **kwargs):
        generated.append(dict(target=actions[0].target, thread=threading.current_thread().name, **kwargs))
        return program(actions[0].target, actions[1].target)
    monkeypatch.setattr(runner.generator, 'generate', generate)
    return SimpleNamespace(runtime=runtime, context=context, robot=robot, runner=runner, generated=generated, root=tmp_path)


def test_pipeline_generates_next_node_while_first_node_executes(rig):
    entered = threading.Event()
    original = rig.runner.generator.generate
    def generate(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)
    rig.runner.generator.generate = generate
    def execute(*args, **kwargs):
        assert entered.wait(2)
        return True
    rig.runtime._run_actions.side_effect = execute
    assert rig.runner.run()
    assert rig.generated[0]['thread'].startswith('policy-prefetch')
    state = json.loads((rig.root/'task_state.json').read_text())
    assert [n['status'] for n in state['nodes'].values()] == ['completed','completed']
    assert len(rig.runner.completed) == 4
    assert any(e['event']=='prefetch_used' for e in rig.runner.events)


def test_repair_only_failed_node_keeps_completed_prefix_and_cancels_stale_code(rig):
    calls = []
    def execute(robot, context, camera, actions, *, completed_actions, start_index):
        target = actions[start_index].target
        calls.append((target, len(completed_actions)))
        if calls.count(('a',0)) == 1 and target == 'a':
            rig.runtime._write_failure_report(grasp('a'), 'a', 'grasp_pose_not_found', 'no candidate',
                                             completed_actions=completed_actions)
            return False
        return True
    rig.runtime._run_actions.side_effect = execute
    assert rig.runner.run()
    assert calls == [('a',0),('a',0),('b',2)]
    assert rig.runtime.task_state.budgets['node_0:repairs'] == 1
    assert rig.runtime.task_state.revision == 1
    assert any(e['event']=='prefetch_invalidated' for e in rig.runner.events)
    repair = next(g for g in rig.generated if g.get('feedback'))
    assert repair['feedback']['failure_type'] == 'grasp_pose_not_found'
    assert repair['constraints']
    graph = PlanGraph.load_json(rig.runtime.config.replan_json)
    assert graph.get_node('0').constraints


def test_failure_on_second_node_does_not_replay_first(rig):
    calls=[]
    def execute(robot, context, camera, actions, *, completed_actions, start_index):
        target=actions[start_index].target;calls.append(target)
        if target=='b' and calls.count('b')==1:
            rig.runtime._write_failure_report(grasp('b'),'b','grasp_pose_not_found','no pose',completed_actions=completed_actions)
            return False
        return True
    rig.runtime._run_actions.side_effect=execute
    assert rig.runner.run()
    assert calls == ['a','b','b']


def test_motion_error_is_not_retried_or_converted_to_policy_failure(rig):
    rig.runtime._run_actions.side_effect = MotionError({'error':'controller failed'})
    with pytest.raises(MotionError):
        rig.runner.run()
    assert rig.runtime._run_actions.call_count == 1
    assert rig.runtime.task_state.budgets.get('node_0:repairs',0) == 0


def test_release_then_retreat_failure_never_regrasp(rig):
    def execute(*args, **kwargs):
        rig.context.manipulation_progress = dict(stage='retreat', released=True, holding_state='empty', motion_started=True)
        rig.runtime._write_failure_report(place('tray'),'a','place_failed','retreat failed',
                                         completed_actions=[grasp('a').__dict__])
        return False
    rig.runtime._run_actions.side_effect=execute
    assert not rig.runner.run()
    assert rig.runtime._run_actions.call_count == 1
    assert rig.runtime.last_failure_report['recovery']['route']=='stop'


def test_one_recovery_owner_limits_total_attempts(rig):
    rig.runtime.config.enable_policy_pipeline=False
    def execute(*args, **kwargs):
        rig.runtime._write_failure_report(grasp('a'),'a','grasp_pose_not_found','no pose')
        return False
    rig.runtime._run_actions.side_effect=execute
    assert not rig.runner.run()
    assert rig.runtime._run_actions.call_count==2
    robot=SimpleNamespace(observations={}, grasp=Mock(return_value=SkillResult(False, {'failure_type':'grasp_pose_not_found'})))
    rig.runtime.config.grasp_planning_retry_attempts=2
    rig.runtime._grasp_with_cache_retry(robot,None,'a')
    assert robot.grasp.call_count==1


def test_pipeline_off_uses_same_actions_without_background_generation(rig):
    rig.runtime.config.enable_policy_pipeline=False
    assert rig.runner.run()
    assert all(g['thread']=='MainThread' for g in rig.generated)
    assert not any(e['event'].startswith('prefetch') for e in rig.runner.events)


@pytest.mark.parametrize('kind,progress,action,route',[
    ('grasp_pose_not_found',{},'grasp','refresh'),
    ('nesyro_gate_replan',{},'grasp','graph_repair'),
    ('policy_failed',{},'policy','code_repair'),
    ('geometry_occlusion_budget_exhausted',{},'grasp','stop'),
    ('motion_error',{},'grasp','stop'),
    ('grasp_pose_not_found',{'holding_state':'unknown'},'grasp','stop'),
    ('grasp_pose_not_found',{'motion_started':True},'grasp','stop'),
])
def test_recovery_routes(kind,progress,action,route):
    assert recovery_decision(kind,progress,action)['route']==route


def test_policy_fault_after_completed_pair_does_not_replay_physical_work(rig):
    rig.runtime.config.cap_policy_source = '''def policy():
    pick_place("a", "tray")
    observe("unbound")
'''
    assert not rig.runner.run()
    assert rig.runtime._run_actions.call_count == 1
    assert rig.runtime.last_failure_report['failure_type'] == 'policy_failed_after_completion'
    assert rig.runtime.last_failure_report['recovery']['route'] == 'stop'
    assert len(rig.runtime.last_failure_report['completed_actions']) == 2


def test_existing_clear_budget_and_buffer_occupancy_survive_node_repair(rig):
    rig.robot.occlusion_recovery_counts={'a':1}
    rig.robot.geometry_buffer_slots_used=1
    rig.robot.geometry_recovery_state={'observation_counts':{'a':1}}
    assert rig.runner.run()
    state=json.loads((rig.root/'task_state.json').read_text())
    assert state['skill_budgets']['clear_counts']=={'a':1}
    assert state['skill_budgets']['buffer_slots_used']==1
    assert state['skill_budgets']['geometry_observations']=={'a':1}


def test_information_plan_waits_without_code_repair_or_task_completion(rig):
    def pending(*args, **kwargs):
        rig.runtime._write_failure_report(grasp('a'), 'a', 'geometry_information_plan_ready',
            'Plan ready; waiting for new scene evidence',
            evidence={'recovery_pending': True, 'automatic_replan': False})
        return False
    rig.runtime._run_actions.side_effect = pending
    assert not rig.runner.run()
    state = json.loads((rig.root/'task_state.json').read_text())
    assert state['nodes']['0']['status'] == 'awaiting_information'
    assert state['completed_actions'] == []
    assert 'node_0:repairs' not in state['budgets']
    assert any(e['event'] == 'information_pending' for e in rig.runner.events)


def test_running_invalidated_generation_is_accounted_and_never_consumed(rig):
    entered, release = threading.Event(), threading.Event()
    original = rig.runner.generator.generate
    def generate(*args, **kwargs):
        if threading.current_thread().name.startswith('policy-prefetch'):
            entered.set()
            assert release.wait(3)
        return original(*args, **kwargs)
    rig.runner.generator.generate = generate
    def stop(*args, **kwargs):
        assert entered.wait(3)
        rig.runtime._write_failure_report(grasp('a'), 'a', 'geometry_occlusion_budget_exhausted', 'stop')
        release.set()
        return False
    rig.runtime._run_actions.side_effect = stop
    with task_timing('stop', rig.root/'timing.json'):
        assert not rig.runner.run()
    final = rig.runner.events[-1]
    job, = final['prefetch_jobs']
    assert job['outcome'] == 'unused_task_ended'
    assert job['duration_s'] > 0 and job['wasted_generation_s'] == job['duration_s']
    assert job['queue_wait_s'] >= 0
    assert not any(e['event'] == 'prefetch_used' for e in rig.runner.events)
    spans = json.loads((rig.root/'timing.json').read_text())['spans']
    assert any(s['name'] == '等待后台代码线程退出' for s in spans)
    assert any(s['name'] == '子任务代码生成' and s['background'] for s in spans)


def test_failed_prefetch_falls_back_once_and_records_wait(rig):
    from runtime.cap_policy import PolicyError
    original = rig.runner.generator.generate
    def generate(*args, **kwargs):
        if threading.current_thread().name.startswith('policy-prefetch'):
            raise PolicyError('injected invalid code')
        return original(*args, **kwargs)
    rig.runner.generator.generate = generate
    assert rig.runner.run()
    job, = rig.runner.events[-1]['prefetch_jobs']
    assert job['outcome'] == 'failed'
    assert job['error_type'] == 'PolicyError' and job['join_wait_s'] >= 0
    assert rig.runtime._run_actions.call_count == 2


def test_repaired_attempt_prefetches_new_revision_while_discarding_running_old_job(rig):
    entered, release, restarted = threading.Event(), threading.Event(), threading.Event()
    original = rig.runner.generator.generate
    def generate(goal, actions, path, **kwargs):
        if threading.current_thread().name.startswith('policy-prefetch'):
            if path.name.endswith('_r0.py'):
                entered.set()
                assert release.wait(3)
            elif path.name.endswith('_r1.py'):
                restarted.set()
        elif kwargs.get('feedback'):
            release.set()
        return original(goal, actions, path, **kwargs)
    rig.runner.generator.generate = generate
    calls = []
    def execute(robot, context, camera, actions, *, completed_actions, start_index):
        calls.append((actions[start_index].target, start_index))
        if len(calls) == 1:
            assert entered.wait(3)
            rig.runtime._write_failure_report(grasp('a'), 'a', 'grasp_pose_not_found', 'no candidate')
            return False
        if len(calls) == 2:
            assert restarted.wait(3), 'Repaired execution should overlap next-node generation'
        return True
    rig.runtime._run_actions.side_effect = execute
    try:
        assert rig.runner.run()
    finally:
        release.set()
    assert calls == [('a', 0), ('a', 0), ('b', 2)]
    jobs = rig.runner.events[-1]['prefetch_jobs']
    assert [(j['revision'], j['outcome']) for j in jobs] == [(0, 'invalidated'), (1, 'used')]
    assert jobs[0]['cancelled'] is False
    assert jobs[0]['work_after_invalidation_s'] > 0
    used = [e for e in rig.runner.events if e['event'] == 'prefetch_used']
    assert [(e['node'], e['revision']) for e in used] == [(1, 1)]
