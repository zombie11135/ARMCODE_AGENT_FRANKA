"""Actual contact physics tests; run only in the existing simulation venv."""
import importlib.util
import json

import pytest

if importlib.util.find_spec('robosuite') is None:
    pytest.skip('Requires .venv-robosuite', allow_module_level=True)

from simulation.pmir.experiment import run_trial


@pytest.mark.parametrize('scenario', ['independent', 'occluded'])
def test_paired_physics_preserves_initial_state_and_goal_while_releasing_real_events(tmp_path, scenario):
    serial = run_trial(tmp_path/'serial', scenario=scenario, mode='serial')
    dynamic = run_trial(tmp_path/'dynamic', scenario=scenario, mode='dynamic', compute_delay_s=.03)
    naive = run_trial(tmp_path/'naive', scenario=scenario, mode='naive', compute_delay_s=.03)
    # Delay here is an explicit concurrency test stimulus, not a speed comparison.
    assert serial['success'] and dynamic['success'] and naive['success'], naive['error']
    assert naive['initial_qpos_sha256'] == serial['initial_qpos_sha256']
    assert naive['physical_sequence'] == serial['physical_sequence']
    assert naive['control_steps'] == serial['control_steps']
    assert naive['invalidated_attempts'] == (1 if scenario == 'occluded' else 0)
    assert dynamic['invalidated_attempts'] == 0
    for result in (serial, naive, dynamic):
        assert sum(result['foreground_cost_s'].values()) == pytest.approx(result['wall_duration_s'], abs=1e-9)
        assert result['exposed_critical_compute_s'] <= result['arm_idle_compute_wall_s']
        assert result['compute_actual_physics_overlap_s'] <= result['compute_physical_residency_overlap_wall_s']
        assert result['perception_without_injected_wait_s'] > 0

    assert serial['initial_qpos_sha256'] == dynamic['initial_qpos_sha256']
    assert serial['completed'] == dynamic['completed'] == [('red_cube', 'red_goal'), ('blue_cube', 'blue_goal')]
    assert all(v > .06 for v in dynamic['lift_height_m'].values())
    assert all(v['complete'] for v in dynamic['final_task']['objects'].values())
    assert serial['compute_physical_residency_overlap_wall_s'] == 0
    if scenario == 'occluded':
        assert dynamic['initial_pixels']['blue_cube'] == 0
        events = [json.loads(s) for s in (tmp_path/'dynamic/events.jsonl').read_text().splitlines()]
        release = next(e for e in events if e['event'] == 'event_released' and e.get('target') == 'blue_cube')
        capture = next(e for e in events if e['event'] == 'start' and e.get('kind') == 'CAPTURE' and e.get('target') == 'blue_cube')
        red_end = next(e for e in events if e['event'] == 'commit' and e.get('phase') == 'settle' and e.get('target') == 'red_cube')
        assert release['visible_pixels'] >= 25 and release['stable']
        assert release['sequence'] < capture['sequence'] < red_end['sequence']


def test_snapshot_has_no_object_truth_and_worker_cannot_mutate_rgbd(tmp_path):
    from simulation.pmir.adapter import Adapter
    adapter = Adapter(tmp_path)
    try:
        snapshot = adapter.snapshot()
        assert 'objects' not in snapshot and 'robot' not in snapshot
        assert not snapshot['camera']['rgb'].flags.writeable
        with pytest.raises(ValueError):
            snapshot['camera']['rgb'][0, 0] = 0
    finally:
        adapter.close()


def test_critical_wait_clips_worker_finish_and_does_not_equal_net_saving():
    from simulation.pmir.experiment import cost_metrics
    span = dict(node='p', kind='PERCEIVE', attempt=1, start_s=0., duration_s=.5,
                cpu_s=.01, consumed=True, components={'injected_wait': [.01, .49]})
    profile = dict(intervals=[(0., .25, 'physics_step')], wait_windows=[
        dict(start_s=.25, end_s=.75, jobs=[dict(node='p', attempt=1, reason='data_dependency')])])
    metrics = cost_metrics({'spans': [span]}, profile)
    assert metrics['exposed_critical_compute_s'] == .25
    assert metrics['compute_during_wait_s'] == .25  # Excludes .25 s of post-completion polling.
    assert metrics['compute_actual_physics_overlap_s'] == .25
    assert metrics['perception_without_injected_wait_s'] == pytest.approx(.02)
    assert 'saved_wall_s' not in metrics  # Requires a matched serial counterfactual.
