"""Explicit simulation tests; skip in environments without simulation packages."""
from dataclasses import replace
import importlib.util
import json

import numpy as np
import pytest

if importlib.util.find_spec('robosuite') is None or importlib.util.find_spec('mujoco') is None:
    pytest.skip('Use the independent .venv-robosuite environment', allow_module_level=True)

from runtime.cap_policy import PolicyError, PolicyStopped
from simulation.robosuite_backend import RobosuiteBackend, SimConfig
from simulation.robosuite_backend.policy import run_policy


@pytest.fixture
def backend(tmp_path):
    with RobosuiteBackend(log_path=tmp_path / 'events.jsonl') as sim:
        yield sim


def test_observation_rgbd_calibration_and_snapshot(backend):
    step = backend.step_count
    first = backend.observe('red_cube')
    assert first.success
    camera = first.data['camera']
    assert camera['rgb'].shape == (240, 320, 3)
    assert camera['rgb'].dtype == np.uint8
    assert camera['depth_m'].shape == (240, 320)
    assert camera['depth_m'].dtype == np.float32
    assert np.isfinite(camera['depth_m']).all() and (camera['depth_m'] > 0).all()
    assert len(first.data['robot']['joint_positions_rad']) == 7
    assert first.data['robot']['frame'] == 'sim_world'
    assert not first.data['task']['complete']
    # Project each cube's top centre into RGB-D: catches vertical flips, camera
    # convention errors, depth-buffer output, and RGB/BGR channel swaps.
    world_to_optical = np.linalg.inv(camera['T_world_from_camera_optical'])
    for name, channel in [('red_cube', 0), ('blue_cube', 2)]:
        point = np.array([*first.data['objects'][name]['position_m'], 1.0])
        point[2] += backend.config.cube_half_size
        optical = (world_to_optical @ point)[:3]
        pixel = camera['K'] @ optical
        u, v = np.rint(pixel[:2] / pixel[2]).astype(int)
        assert abs(float(camera['depth_m'][v, u]) - optical[2]) < 0.015
        assert camera['rgb'][v, u, channel] == max(camera['rgb'][v, u])
    camera['rgb'][:] = 0
    assert backend.observe().data['camera']['rgb'].any()
    assert backend.step_count == step  # Observation must not advance physics.


def test_reset_replays_seed_and_new_seed_changes_layout(backend):
    first = backend.observe().data
    backend.reset()
    second = backend.observe().data
    assert first['objects'] == second['objects']
    assert first['robot'] == second['robot']
    np.testing.assert_array_equal(first['camera']['rgb'], second['camera']['rgb'])
    backend.reset(seed=8)
    assert backend.object_states()['red_cube']['position_m'] != first['objects']['red_cube']['position_m']
    assert not backend.task_status()['complete']


def test_two_object_policy_uses_contact_physics_and_logs(backend, tmp_path, monkeypatch):
    def no_teleport(*args, **kwargs):
        pytest.fail('Skills must not assign object joint positions')

    monkeypatch.setattr(backend.env.sim.data, 'set_joint_qpos', no_teleport)
    results = run_policy(backend)
    assert len(results) == 2
    assert all(result.success and result.data['lift_height_m'] > 0.06 for result in results)
    assert all(result.data['placement']['released'] and result.data['placement']['on_table'] for result in results)
    assert backend.task_status()['complete']
    for result in results:
        assert result.data['sim_duration_s'] == pytest.approx(result.data['control_steps'] / backend.config.control_freq)
        assert result.data['wall_duration_s'] > 0
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [e['sequence'] for e in events] == list(range(len(events)))
    assert len([e for e in events if e['event'] == 'skill_result' and e['success']]) == 2
    assert all(e['simulated'] and 'wall_elapsed_s' in e and 'sim_time_s' in e for e in events)


def test_unknown_names_rejected_without_motion(backend):
    before = backend.step_count
    assert not backend.observe('absent').success
    assert not backend.pick_place('red_cube', 'hardware_tray').success
    assert not backend.pick_place('absent', 'red_goal').success
    assert backend.step_count == before


def test_timeout_stops_and_requires_reset(backend):
    backend.config = replace(backend.config, phase_timeout_steps=1)
    failed = backend.pick_place('red_cube', 'red_goal')
    assert not failed.success and failed.data['phase'] == 'raise'
    assert failed.data['control_steps'] == 1
    before = backend.step_count
    assert backend.pick_place('blue_cube', 'blue_goal').error == 'reset_required_after_failure'
    assert backend.step_count == before
    assert not backend.task_status()['complete']
    backend.reset()
    assert not backend.halted


def test_completion_rejects_airborne_wrong_goal_and_unstable_states(backend):
    sim = backend.env.sim
    joint = backend.env.objects['red_cube'].joints[0]
    # Test fixtures may place objects directly; the production skill never does.
    for position, velocity in [([0.13, -0.12, 1.0], [0] * 6),
                               ([0.13, 0.12, 0.82], [0] * 6),
                               ([0.13, -0.12, 0.82], [1, 0, 0, 0, 0, 0])]:
        sim.data.set_joint_qpos(joint, [*position, 1, 0, 0, 0])
        sim.data.set_joint_qvel(joint, velocity)
        sim.forward()
        backend._stable['red_cube'] = 100
        assert not backend.task_status()['objects']['red_cube']['complete']
    backend._stable['red_cube'] = 0
    assert not backend.task_status()['complete']


def test_policy_binding_rejects_reordering_and_incomplete_work(backend):
    before = backend.step_count
    with pytest.raises(PolicyError, match='reordered'):
        run_policy(backend, 'def policy():\n    pick_place("blue_cube", "blue_goal")')
    with pytest.raises(PolicyStopped, match='before both'):
        run_policy(backend, 'def policy():\n    return True')
    assert backend.step_count == before


def test_disturbance_hook_has_repeatable_separate_rng_and_event(tmp_path):
    samples = []

    def hook(env, *, step, sim_time_s, rng):
        if step == 0:
            samples.append(float(rng.uniform()))
            return dict(kind='test_hook', sample=samples[-1], sim_time_s=sim_time_s)

    with RobosuiteBackend(log_path=tmp_path / 'hook.jsonl', disturbance_hook=hook) as sim:
        first = sim.object_states()
        sim.reset()
        assert sim.object_states() == first
    assert len(samples) == 2 and samples[0] == samples[1]
    events = [json.loads(line) for line in (tmp_path / 'hook.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event'] == 'disturbance']) == 2


def test_event_log_refuses_overwrite(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text('previous experiment\n')
    with pytest.raises(FileExistsError):
        RobosuiteBackend(log_path=path)
    assert path.read_text() == 'previous experiment\n'
