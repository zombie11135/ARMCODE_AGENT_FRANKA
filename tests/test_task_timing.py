"""Timing must preserve failures and distinguish overlapping task work."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from runtime.timing import task_timing, timed_stage, timed_task, timing_stage


def test_nested_durations_are_inclusive_and_history_survives_next_task(tmp_path, monkeypatch):
    from runtime import timing
    clock = [0.]
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    report = tmp_path/'task_timing.json'
    with task_timing('first', report):
        with timing_stage('outer'):
            clock[0] = 1.
            with timing_stage('inner'):
                clock[0] = 5.
            clock[0] = 6.
        clock[0] = 8.
    first = json.loads(report.read_text())
    outer, inner = first['spans']
    assert first['duration_s'] == 8.
    assert (outer['duration_s'], inner['duration_s']) == (6., 4.)
    assert inner['parent_id'] == outer['id']
    assert outer['parent_id'] is None
    assert sum(row['duration_s'] for row in first['summary']) > first['duration_s']
    assert 'overlap' in first['note']
    with task_timing('second', report):
        pass
    second = json.loads(report.read_text())
    assert second['run_id'] != first['run_id'] and second['spans'] == []
    assert json.loads(Path(first['history_path']).read_text()) == first


def test_background_thread_shares_task_but_keeps_its_own_span_parent(tmp_path):
    entered, release = threading.Event(), threading.Event()
    report = tmp_path/'task_timing.json'

    @timed_stage('background')
    def work():
        entered.set()
        assert release.wait(3)
        with timing_stage('background child'):
            return True

    with task_timing('parallel task', report):
        with timing_stage('execution'):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(copy_context().run, work)
                try:
                    assert entered.wait(3)
                    with timing_stage('foreground'):
                        pass
                finally:
                    release.set()
                assert future.result(timeout=3)
    data = json.loads(report.read_text())
    rows = {row['name']: row for row in data['spans']}
    assert rows['foreground']['parent_id'] == rows['execution']['id']
    assert rows['background']['parent_id'] == rows['execution']['id']
    assert rows['background child']['parent_id'] == rows['background']['id']
    assert rows['background']['background'] and rows['background child']['background']
    assert not rows['foreground']['background']
    assert rows['background']['start_s'] < rows['foreground']['start_s']
    assert len({row['id'] for row in rows.values()}) == len(rows)


@pytest.mark.parametrize('result', [False, {'success': False}, SimpleNamespace(success=False)])
def test_false_results_mark_task_and_stage_failed_without_changing_return(tmp_path, result):
    class Task:
        config = SimpleNamespace(goal='failure', plan_json=str(tmp_path/'plan.json'), execute_robot=False)

        @timed_task
        @timed_stage('skill')
        def run(self):
            return result

    assert Task().run() is result
    data = json.loads((tmp_path/'task_timing.json').read_text())
    assert data['status'] == data['spans'][0]['status'] == 'failed'


@pytest.mark.parametrize('error', [RuntimeError('original failure'), KeyboardInterrupt()])
def test_exception_propagates_with_timing_report_and_no_context_leak(tmp_path, error):
    report = tmp_path/'task_timing.json'
    with pytest.raises(type(error)) as raised:
        with task_timing('failure', report):
            with timing_stage('motion'):
                raise error
    assert raised.value is error
    data = json.loads(report.read_text())
    assert data['status'] == ('interrupted' if isinstance(error, KeyboardInterrupt) else 'error')
    assert data['spans'][0]['status'] == 'error'
    assert data['spans'][0]['attributes']['error_type'] == type(error).__name__
    original = report.read_bytes()
    with timing_stage('outside task'):
        pass
    assert report.read_bytes() == original


def test_timing_write_error_does_not_replace_a_skill_result(tmp_path, capsys):
    blocked = tmp_path/'not-a-directory'
    blocked.write_text('file')
    with task_timing('diagnostic only', blocked/'report.json'):
        pass
    assert '报告写入失败' in capsys.readouterr().out


def test_agent_planning_and_coding_share_one_report_without_executing(tmp_path, monkeypatch):
    from agents.simple_agent_system import SimpleArmCodeAgentSystem
    from core.context import RuntimeContext
    from runtime.config import SimpleAgentSystemConfig
    config = SimpleAgentSystemConfig(goal='将西瓜霜放到蓝色盘子中', run_executor=False,
        project_root=str(tmp_path), plan_json=str(tmp_path/'plan.json'),
        plan_md=str(tmp_path/'plan.md'), generated_code=str(tmp_path/'actions.py'))
    system = SimpleArmCodeAgentSystem(config)
    monkeypatch.setattr(system, '_build_planner_context', lambda: RuntimeContext())
    monkeypatch.setattr(system, '_executing_stage', lambda actions: pytest.fail('No execution authorized'))
    assert system.run().success
    report = json.loads((tmp_path/'task_timing.json').read_text())
    assert report['status'] == 'ok'
    assert {row['name'] for row in report['spans']} == {'任务规划', '动作代码生成'}
    assert report['metadata']['execute_robot'] is False
    assert len(list((tmp_path/'timing').glob('*.json'))) == 1


def test_inference_cache_hit_is_visible_in_stage_attributes(tmp_path, capsys):
    @timed_stage('candidate service')
    def candidates():
        return {'success': True, 'inference_cache_hit': True}
    report = tmp_path/'task_timing.json'
    with task_timing('cache', report):
        assert candidates()['inference_cache_hit']
    data = json.loads(report.read_text())
    assert data['spans'][0]['attributes']['inference_cache_hit']
    assert '[缓存命中]' in capsys.readouterr().out
