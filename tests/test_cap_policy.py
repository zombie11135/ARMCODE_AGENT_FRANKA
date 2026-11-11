import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.cap_code_generator import CapCodeGenerator
from agents.cap_execution import run_policy
from core.actions import grasp, place
from core.context import RuntimeContext
from core.skill_base import SkillResult
from runtime.cap_policy import PolicyError, PolicyInterpreter, validate_policy


SOURCE = '''def policy():
    if visible("box"):
        pick_place("box", "tray")
def visible(target):
    for attempt in range(2):
        if observe(target):
            return True
    return False
'''


def test_feedback_loop_and_helper_change_actions_from_observations():
    for observed, expected in [([False, True], 1), ([False, False], 0)]:
        see = Mock(side_effect=observed);pick = Mock(return_value=True)
        PolicyInterpreter(SOURCE, dict(observe=see, pick_place=pick)).run()
        assert see.call_count == 2
        assert pick.call_count == expected


@pytest.mark.parametrize('body', [
    'import os', 'while True: pass', 'open("/tmp/file")',
    'observe.__globals__', '__import__("os")', 'x = [0] * 999',
    'exec("pick_place()")', 'raise Exception()', 'x = observe("box")[0]',
    'for x in range(9999): observe("box")',
])
def test_noncontract_code_is_rejected_without_skill_calls(body):
    source = 'def policy():\n    ' + body + '\n'
    api = dict(observe=Mock(), pick_place=Mock())
    with pytest.raises(PolicyError):
        PolicyInterpreter(source, api).run()
    api['observe'].assert_not_called();api['pick_place'].assert_not_called()


def test_recursion_and_nested_finite_loops_have_global_budgets():
    with pytest.raises(PolicyError, match='depth'):
        PolicyInterpreter('def policy():\n    policy()\n', {}).run()
    source = 'def policy():\n    for x in range(16):\n        for y in range(16):\n            observe("box")\n'
    observer = Mock()
    with pytest.raises(PolicyError, match='budget'):
        PolicyInterpreter(source, {'observe': observer}, max_api_calls=3).run()
    assert observer.call_count == 3


def test_hierarchical_generation_synthesizes_missing_helper_and_records_prompts(tmp_path):
    client = Mock()
    client.chat.side_effect = [SOURCE.split('def visible')[0], 'def visible(target):\n    return observe(target)']
    destination = tmp_path/'policy.py'
    result = CapCodeGenerator(client).generate('move box to tray', [grasp('box'), place('tray')], destination)
    validate_policy(result)
    assert client.chat.call_count == 2
    report = json.loads(destination.with_suffix('.cap.json').read_text())
    assert report['validated'] and len(report['generations']) == 2
    assert report['source'] == destination.read_text()


@pytest.mark.parametrize('source', [
    'def policy():\n    pass',
    'def policy():\n    pick_place("wrong", "tray")',
    'def policy():\n    pick_place("box", "tray")\n    pick_place("box", "tray")',
])
def test_generation_rejects_skipped_unbound_or_duplicate_work(tmp_path, source):
    client = Mock();client.chat.return_value = source
    with pytest.raises(PolicyError):
        CapCodeGenerator(client).generate('goal', [grasp('box'), place('tray')], tmp_path/'policy.py')
    assert not json.loads((tmp_path/'policy.cap.json').read_text())['validated']


def test_runtime_binds_names_preserves_feedback_and_stops_failed_manipulation(tmp_path):
    for ok in (True, False):
        task = SimpleNamespace(config=SimpleNamespace(cap_policy_source=SOURCE, plan_json=str(tmp_path/'plan.json')),
                               _run_actions=Mock(return_value=ok), _write_failure_report=Mock())
        robot = SimpleNamespace(observe=Mock(return_value=SkillResult(True, {})))
        assert run_policy(task, robot, RuntimeContext(), None,
                          [grasp('box'), place('tray')], [grasp('ref_123'), place('tray')]) is ok
        robot.observe.assert_called_once_with('ref_123')
        actions = task._run_actions.call_args.args[3]
        assert actions == [grasp('ref_123'), place('tray')]
        task._write_failure_report.assert_not_called()  # Keep original hardware/skill report on failure.
        trace = json.loads((tmp_path/'cap_trace.json').read_text())
        assert trace['success'] is ok


def test_runtime_failed_observation_never_claims_task_success(tmp_path):
    task = SimpleNamespace(config=SimpleNamespace(cap_policy_source=SOURCE, plan_json=str(tmp_path/'plan.json')),
                           _run_actions=Mock(), _write_failure_report=Mock())
    robot = SimpleNamespace(observe=Mock(return_value=SkillResult(False, {}, error='not seen')))
    assert not run_policy(task, robot, RuntimeContext(), None,
                          [grasp('box'), place('tray')], [grasp('box'), place('tray')])
    task._run_actions.assert_not_called()
    task._write_failure_report.assert_called_once()
