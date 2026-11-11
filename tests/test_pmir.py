"""PMIR analysis and scheduling correctness without simulator/device packages."""
from functools import partial
import json
from types import SimpleNamespace
import threading

import pytest

from runtime.pmir import Graph, Node, SkillRegistry, compile_program
from runtime.pmir.scheduler import Scheduler


SOURCE = 'def policy():\n    pick_place("a", "tray")\n    pick_place("b", "tray")\n'


def ancestors(graph, node):
    result, todo = set(), list(graph.parents(node))
    while todo:
        parent = todo.pop()
        if parent not in result:
            result.add(parent)
            todo.extend(graph.parents(parent))
    return result


def test_auto_expansion_finds_preparation_overlap_and_keeps_physical_order():
    graph = compile_program(SOURCE)
    assert {n.kind for n in graph.nodes} == {'CAPTURE', 'PERCEIVE', 'GRASP_PLAN', 'MOTION_PLAN', 'MOVE', 'GRIPPER', 'VERIFY'}
    final_a = next(n for n in graph.nodes if n.target == 'a' and n.phase == 'settle')
    capture_b = next(n for n in graph.nodes if n.target == 'b' and n.kind == 'CAPTURE')
    move_b = next(n for n in graph.nodes if n.target == 'b' and n.phase == 'raise')
    assert final_a.id not in ancestors(graph, capture_b.id)
    assert final_a.id in ancestors(graph, move_b.id)
    assert capture_b.gate == 'visible:b'
    assert all(e.kind != 'resource' for e in graph.edges)


def test_same_target_write_prevents_reusing_before_move_pose():
    graph = compile_program(SOURCE.replace('"b"', '"a"'))
    captures = [n for n in graph.nodes if n.kind == 'CAPTURE']
    first_final = next(n for n in graph.nodes if n.phase == 'settle')
    assert first_final.id in ancestors(graph, captures[1].id)


def test_explicit_observe_preserves_source_position_and_reuses_its_output():
    graph = compile_program('pick_place("a", "tray")\nobserve("b")\npick_place("b", "tray")')
    final_a = next(n for n in graph.nodes if n.target == 'a' and n.phase == 'settle')
    capture_b, = [n for n in graph.nodes if n.target == 'b' and n.kind == 'CAPTURE']
    assert final_a.id in ancestors(graph, capture_b.id)
    assert capture_b.gate is None


def test_blackbox_extension_is_conservative_and_preserves_all_arguments():
    registry = SkillRegistry()
    registry.register_blackbox('new_skill', 3)
    graph = compile_program('pick_place("a", "tray")\nnew_skill("drawer", "handle", "axis")\npick_place("b", "tray")', registry)
    opaque = next(n for n in graph.nodes if n.kind == 'BLACKBOX')
    assert opaque.arguments == ('drawer', 'handle', 'axis')
    before = [n for n in graph.nodes if int(n.id[1:]) < int(opaque.id[1:])]
    after = [n for n in graph.nodes if int(n.id[1:]) > int(opaque.id[1:])]
    assert all(n.id in ancestors(graph, opaque.id) for n in before)
    assert all(opaque.id in ancestors(graph, n.id) for n in after)


@pytest.mark.parametrize('source', [
    'import os', 'robot.move("a")', 'unknown("a")', 'while True: observe("a")',
    'observe(__import__("os"))', 'pick_place("a", goal="tray")',
    'for x in range(100000): observe(x)', 'def policy(a): observe(a)',
    'if True: observe("a")', 'def policy():\n    if observe("a"):\n        return True\n    pick_place("b", "tray")',
])
def test_unsupported_syntax_rejected_before_execution(source):
    with pytest.raises((ValueError, SyntaxError)):
        compile_program(source)


def test_literal_loop_is_unrolled_with_local_binding():
    graph = compile_program('def policy():\n    tray="tray"\n    for obj in ["a", "b"]:\n        pick_place(obj,tray)')
    assert [n.target for n in graph.nodes if n.phase == 'settle'] == ['a', 'b']


def test_branch_join_waits_for_observation_only_branch_leaves():
    graph = compile_program('if observe("a"):\n    observe("b")\nelse:\n    observe("c")\nobserve("d")')
    final = next(n for n in graph.nodes if n.kind == 'CAPTURE' and n.target == 'd')
    leaves = [n for n in graph.nodes if n.kind == 'PERCEIVE' and n.target in ('b', 'c')]
    assert all(n.id in ancestors(graph, final.id) for n in leaves)


class FakeAdapter:
    def __init__(self):
        self.backend = SimpleNamespace(sim_time_s=0., step_count=0, halted=False)
        self.completed, self.visible = [], {'a': True, 'b': False}
        self.idle_steps = 0
        self.started = threading.Event()
        self.owner = threading.get_ident()

    def probe(self):
        pass

    def valid(self, node, value):
        return True

    def physical(self, node, values):
        assert threading.get_ident() == self.owner
        self.backend.step_count += 1
        yield
        if node.phase == 'lift':
            self.visible['b'] = True
            self.emit('event_released', gate='visible:b', producer=node.id)
        if node.phase == 'transport':
            assert self.started.wait(2), 'CPU job should overlap active physical generator'
        self.completed.append(node.id)
        return {'success': True}

    def immediate(self, node, values):
        assert threading.get_ident() == self.owner
        return {'success': node.target != 'missing'}

    def worker_job(self, node, values):
        def work(started, owner):
            assert threading.get_ident() != owner
            started.set()
            return {'success': True}
        return partial(work, self.started, self.owner)

    def after_node(self, node):
        pass

    def idle_tick(self):
        self.started.wait(.001)


def test_confirmed_event_releases_background_work_before_skill_completes(tmp_path):
    graph = Graph(nodes=[
        Node('lift', 'MOVE', 'lift', 'a', resources=('robot',), owner=True, physical=True),
        Node('transfer', 'MOVE', 'transport', 'a', resources=('robot',), owner=True, physical=True),
        Node('perceive', 'PERCEIVE', 'rgbd', 'b', resources=('cpu',), gate='visible:b'),
    ])
    graph.link('lift', 'transfer', 'control', 'Phase order')
    adapter = FakeAdapter()
    scheduler = Scheduler(graph, adapter, tmp_path)
    result = scheduler.run()
    assert result['success'], result['error']
    assert any(e.source == 'lift' and e.target == 'perceive' and e.kind == 'event' for e in graph.edges)
    events = scheduler.events
    release = next(e['sequence'] for e in events if e['event'] == 'event_released')
    start = next(e['sequence'] for e in events if e['event'] == 'start' and e['node'] == 'perceive')
    finish = next(e['sequence'] for e in events if e['event'] == 'commit' and e['node'] == 'transfer')
    assert release < start < finish


def test_stale_result_stops_before_dependent_physical_work(tmp_path):
    graph = Graph(nodes=[Node('read', 'CAPTURE', 'snapshot', 'a', owner=True),
                         Node('move', 'MOVE', 'raise', 'a', inputs=('read',), owner=True, physical=True)])
    graph.link('read', 'move', 'data', 'Pose')
    adapter = FakeAdapter()
    adapter.valid = lambda node, value: False
    result = Scheduler(graph, adapter, tmp_path).run()
    assert not result['success'] and 'Stale result' in result['error']
    assert adapter.backend.step_count == 0 and adapter.completed == []


@pytest.mark.parametrize('mode', ['serial', 'naive', 'dynamic'])
def test_untaken_branch_is_skipped_even_if_visibility_gate_is_false(tmp_path, mode):
    graph = compile_program('if observe("missing"):\n    pick_place("b", "tray")\nelse:\n    observe("a")')
    adapter = FakeAdapter()
    # No simulator is needed: condition perception returns false for missing.
    adapter.worker_job = lambda node, values: partial(dict, success=node.target != 'missing')
    result = Scheduler(graph, adapter, tmp_path, mode=mode).run()
    assert result['success'], result['error']
    assert not adapter.completed
    assert any(value == 'skipped' for value in result['node_states'].values())
    assert any(e['event'] == 'commit' and e.get('target') == 'a' for e in json.loads(
        '['+','.join((tmp_path/'events.jsonl').read_text().splitlines())+']'))


def test_exclusive_foreground_costs_partition_nested_work():
    import time
    from runtime.pmir.scheduler import ForegroundCosts
    ledger = ForegroundCosts(time.perf_counter())
    with ledger.scope('outer'):
        with ledger.scope('inner'):
            sum(range(100))
    totals, wall = ledger.finish()
    assert sum(totals.values()) == pytest.approx(wall, abs=1e-10)
    assert totals['outer'] > 0 and totals['inner'] > 0
    assert all(left[1] == right[0] for left, right in zip(ledger.intervals, ledger.intervals[1:]))
    original = len(ledger.intervals)
    with ledger.scope('outside_task'):
        pass
    assert len(ledger.intervals) == original


def test_naive_reacquires_hidden_snapshot_once_without_early_physical_commit(tmp_path):
    adapter = FakeAdapter()
    adapter.immediate = lambda node, values: dict(success=True, visible=adapter.visible.get(node.target, False))
    def job(node, values):
        success = values[node.inputs[0]].get('visible', True) if node.kind == 'PERCEIVE' else True
        adapter.started.set()
        return partial(dict, success=success)
    adapter.worker_job = job
    scheduler = Scheduler(compile_program(SOURCE), adapter, tmp_path, mode='naive')
    result = scheduler.run()
    assert result['success'], result['error']
    invalid = [e for e in scheduler.events if e['event'] == 'invalidated']
    assert len(invalid) == 1 and invalid[0]['target'] == 'b'
    attempts = [s for s in result['spans'] if s['target'] == 'b' and s['kind'] == 'PERCEIVE']
    assert [s['consumed'] for s in attempts] == [False, True]
    release = next(e for e in scheduler.events if e['event'] == 'event_released')
    retry = next(e for e in scheduler.events if e['event'] == 'start' and e['kind'] == 'CAPTURE' and e['attempt'] == 2)
    physical = next(e for e in scheduler.events if e['event'] == 'start' and e['kind'] == 'MOVE' and e['target'] == 'b')
    assert release['sequence'] < retry['sequence'] < physical['sequence']


def test_naive_does_not_loop_on_failed_confirmed_reacquisition(tmp_path):
    adapter = FakeAdapter()
    adapter.started.set()
    adapter.worker_job = lambda node, values: partial(dict, success=node.target != 'b')
    result = Scheduler(compile_program(SOURCE), adapter, tmp_path, mode='naive').run()
    assert not result['success'] and 'reacquisition failed' in result['error']
    assert len([s for s in result['spans'] if s['target'] == 'b' and s['kind'] == 'CAPTURE']) == 2
