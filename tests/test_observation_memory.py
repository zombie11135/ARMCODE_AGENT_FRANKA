"""Pose-based scheduling tests. No streams, models or robot commands are started."""
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from runtime.observation_memory import ObservationMemory, ObservationCachePolicy, pose_distance, fresh_camera
from runtime.camera import TaskCachedCamera


POSE = [.4, .1, .5, math.pi, 0., 0.]


def test_pose_metric_handles_wraparound_and_equivalent_euler_angles():
    assert pose_distance([0, 0, 0, 0, 0, math.pi], [0, 0, 0, 0, 0, -math.pi])[1] == 0
    assert pose_distance([0, 0, 0, math.pi, 0, math.pi], [0, 0, 0, 0, math.pi, 0])[1] == 0
    assert pose_distance(POSE, [.41, .1, .5, math.pi, 0, 0])[0] == pytest.approx(.01)
    with pytest.raises(ValueError):
        pose_distance([0]*5, POSE)


@pytest.mark.parametrize('field,value', [('enabled', 1), ('position_radius_m', 0),
    ('max_age_s', float('nan')), ('orientation_radius_rad', 4), ('max_entries', True)])
def test_invalid_policy_is_rejected(field, value):
    with pytest.raises(ValueError):
        ObservationCachePolicy(**{field: value})


def test_multiple_viewpoints_scope_ttl_deep_copy_and_scene_epoch():
    now = [100.]
    memory = ObservationMemory(clock=lambda: now[0])
    value = {'bbox': [1, 2, 3, 4]}
    memory.put('view', value, POSE, 100, 'camera-calibration-A')
    value['bbox'][0] = -1
    assert memory.get('view', POSE, 'camera-calibration-A')['bbox'][0] == 1
    hit = memory.get('view', POSE, 'camera-calibration-A')
    hit['bbox'][0] = -2
    assert memory.get('view', POSE, 'camera-calibration-A')['bbox'][0] == 1
    far = [.6, *POSE[1:]]
    assert memory.get('view', far, 'camera-calibration-A') is None
    memory.put('view', {'bbox': [9]}, far, 100, 'camera-calibration-A')
    assert memory.get('view', POSE, 'camera-calibration-A')['bbox'][0] == 1
    rotated = [*POSE[:5], .2]
    assert memory.get('view', rotated, 'camera-calibration-A') is None
    assert memory.get('view', POSE, 'camera-calibration-B') is None
    now[0] = 129
    assert memory.get('view', POSE, 'camera-calibration-A') is not None
    now[0] = 131  # Hits must not extend lifetime.
    assert memory.get('view', POSE, 'camera-calibration-A') is None
    memory.put('view', {}, POSE, 100, 'camera-calibration-A')  # Slow inference, expired source.
    assert not memory.entries
    memory.put('view', {}, POSE, 131, 'camera-calibration-A')
    epoch = memory.epoch
    memory.invalidate('object_moved')
    memory.put('view', {}, POSE, 131, 'camera-calibration-A', epoch)  # In-flight stale result.
    assert memory.get('view', POSE, 'camera-calibration-A') is None


def test_bounded_lru_disabled_and_explicit_refresh():
    memory = ObservationMemory(ObservationCachePolicy(max_entries=2), clock=lambda: 100)
    for key in ('a', 'b'):
        memory.put(key, {}, POSE, 100, 'scope')
    assert memory.get('a', POSE, 'scope') == {}
    memory.put('c', {}, POSE, 100, 'scope')
    assert memory.get('b', POSE, 'scope') is None
    camera = SimpleNamespace(observation_memory=memory)
    with fresh_camera(camera):
        memory.put('a', {}, POSE, 100, 'scope')
        assert memory.get('a', POSE, 'scope') is None
    assert memory.get('a', POSE, 'scope') == {}
    disabled = ObservationMemory(ObservationCachePolicy(enabled=False), clock=lambda: 100)
    disabled.put('a', {}, POSE, 100, 'scope')
    assert disabled.get('a', POSE, 'scope') is None


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    from hardware import agent_camera as module
    from hardware.config import HardwareConfig, CameraSettings
    from hardware.interfaces import Pose6D
    now, pose, calibration = [100.], POSE.copy(), {'version': 1}
    transforms = SimpleNamespace(cameras={'wrist': {'optical_frame': 'wrist_optical'}},
                                 summary=lambda: calibration.copy(), _check_capture=Mock())
    monkeypatch.setattr(module, 'load_transforms', lambda config: transforms)
    monkeypatch.setattr(module.time, 'time', lambda: now[0])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    arm = Mock()
    arm.get_pose.side_effect = lambda: Pose6D(*pose)
    arm.move_pose.side_effect = AssertionError('Robot motion forbidden')
    device = Mock()
    device.save_dir = str(tmp_path)
    number = [0]
    def frame():
        number[0] += 1
        return {'camera_name': 'wrist', 'device_serial': '349622072103',
                'color_frame_number': number[0], 'timestamp': now[0],
                'color': np.full((3, 4, 3), number[0], np.uint8),
                'depth': np.ones((3, 4), np.uint16),
                'points_map': np.ones((3, 4, 3), np.float32),
                'depth_aligned_to_color': True, 'color_intrinsics': {'coeffs': [0.]*5}}
    device.capture_wrist.side_effect = frame
    device.capture_global.side_effect = AssertionError('Fixed camera forbidden')
    config = HardwareConfig(cameras=CameraSettings(mode='wrist'))
    camera = module.CalibratedCamera(device, config, arm=arm)
    yield SimpleNamespace(camera=camera, device=device, arm=arm, pose=pose,
                          now=now, calibration=calibration, config=config)
    arm.move_pose.assert_not_called()
    device.capture_global.assert_not_called()
    device.start.assert_not_called()


def test_camera_reuses_original_geometry_and_avoids_legacy_home_cache(calibrated):
    f = calibrated
    first = f.camera.capture_wrist()
    f.pose[0] += .01
    cached = TaskCachedCamera(f.camera)
    cached.set_task_captures(wrist_capture={'color_path': 'unsafe_old_home_frame'})
    cached.enable_cache()
    second = cached.capture_wrist()
    assert f.device.capture_wrist.call_count == 3  # Fresh frame+2, only on first miss.
    assert second['color_path'] == first['color_path']
    assert second['capture_geometry'] == first['capture_geometry']
    assert second['capture_geometry']['before'][0] == .4
    assert second['observation_cache']['hit']
    fallback = cached.capture_global()
    assert fallback['camera_name'] == 'wrist' and fallback['requested_view'] == 'global'
    assert fallback['color_path'] == first['color_path']


def test_camera_distant_return_rotation_age_calibration_and_refresh(calibrated):
    f = calibrated
    first = f.camera.capture_wrist()
    f.pose[0] += .1
    assert f.camera.capture_wrist()['color_path'] != first['color_path']
    f.pose[:] = POSE
    assert f.camera.capture_wrist()['color_path'] == first['color_path']
    f.pose[5] += .2
    assert f.camera.capture_wrist()['color_path'] != first['color_path']
    f.pose[:] = POSE
    f.calibration['version'] = 2
    assert f.camera.capture_wrist()['color_path'] != first['color_path']
    assert f.device.capture_wrist.call_count == 12
    f.now[0] += 31
    old = f.camera.capture_wrist()['color_path']
    with fresh_camera(f.camera, 'operator_refresh'):
        assert f.camera.capture_wrist()['color_path'] != old
    assert f.device.capture_wrist.call_count == 18


def make_robot(f):
    from core.context import RuntimeContext
    from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig
    return SimpleRobotActions(RuntimeContext(camera=TaskCachedCamera(f.camera)),
                              SimpleRobotConfig(observe_wait_s=0))


def fake_observer(robot):
    from core.skill_base import SkillResult
    def observe(**kwargs):
        capture = robot.context.camera.capture_wrist()
        return SkillResult(True, data={'selected_instance': {'object_id': kwargs.get('query', 'box')},
            'grasp_input': {'capture_geometry': capture['capture_geometry'],
                            'color_path': capture['color_path'], 'capture_timestamp': capture['timestamp']}})
    return observe


def test_skill_reuse_skips_camera_perception_and_observation_motion(calibrated, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    robot = make_robot(calibrated)
    run = Mock(side_effect=fake_observer(robot))
    monkeypatch.setattr(FindAndGraspMedicineSkill, 'run', run)
    first = robot._observe_medicine_once('tube')
    calibrated.pose[0] += .01
    robot._move_to_cached_medicine_observe_pose = Mock(side_effect=AssertionError('Unnecessary motion'))
    second = robot._observe_medicine_action('tube')
    assert first.success and second.success
    assert second.data == first.data
    assert run.call_count == 1 and calibrated.device.capture_wrist.call_count == 3
    robot.config.verify_ocr = not robot.config.verify_ocr
    assert robot._observe_medicine_once('tube').success
    assert run.call_count == 2  # Config changes re-run recognition, may still reuse image.
    assert calibrated.device.capture_wrist.call_count == 3
    with fresh_camera(robot.context.camera, 'gate_probe'):
        assert robot._observe_medicine_once('tube').success
    assert run.call_count == 3 and calibrated.device.capture_wrist.call_count == 6
    calibrated.now[0] += 31
    assert robot._observe_medicine_once('tube').success
    assert run.call_count == 4 and calibrated.device.capture_wrist.call_count == 9


def test_failed_or_raising_observation_is_not_reused(calibrated, monkeypatch):
    from core.skill_base import SkillResult
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    robot = make_robot(calibrated)
    observed = fake_observer(robot)
    calls = [0]
    def run(**kwargs):
        calls[0] += 1
        result = observed(**kwargs)
        if calls[0] == 1:
            return SkillResult(False, error='occluded')
        if calls[0] == 2:
            raise RuntimeError('detector failed')
        return result
    monkeypatch.setattr(FindAndGraspMedicineSkill, 'run', Mock(side_effect=run))
    assert not robot._observe_medicine_once('tube').success
    with pytest.raises(RuntimeError, match='detector failed'):
        robot._observe_medicine_once('tube')
    assert robot._observe_medicine_once('tube').success
    assert calibrated.device.capture_wrist.call_count == 9


@pytest.mark.parametrize('outcome', ['success', 'failure', 'exception'])
def test_grasp_attempt_invalidates_even_partial_failure(calibrated, monkeypatch, outcome):
    from core.skill_base import SkillResult
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    robot = make_robot(calibrated)
    robot.config.execute = True
    # The entire execution skill is replaced: no robot/gripper call is made.
    monkeypatch.setattr(FindAndGraspMedicineSkill, 'run', Mock(side_effect=fake_observer(robot)))
    assert robot._observe_medicine_once('tube').success
    source = calibrated.camera.capture_wrist()['color_path']
    result = RuntimeError('partial execution') if outcome == 'exception' else SkillResult(outcome == 'success')
    run = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(FindAndGraspMedicineSkill, 'run', run)
    if outcome == 'exception':
        with pytest.raises(RuntimeError):
            robot.grasp('tube')
    else:
        robot.grasp('tube')
    assert calibrated.camera.capture_wrist()['color_path'] != source
    assert not any(entry[0][0] == 'skill' for entry in calibrated.camera.observation_memory.entries.values())


def test_wrist_verification_forces_one_fresh_view(calibrated):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    source = calibrated.camera.capture_wrist()['color_path']
    pair = SimpleExecutionRuntime._capture_verify_pair(calibrated.camera)
    assert pair['global_is_wrist_fallback']
    assert pair['global'] is pair['wrist']
    assert pair['wrist']['color_path'] != source
    assert calibrated.device.capture_wrist.call_count == 6


def test_container_skill_hit_and_probe_recompute_old_place_plan(calibrated, monkeypatch):
    from core.skill_base import SkillResult
    from skill_lib.combound.place_to_container import PlaceToContainerSkill
    robot = make_robot(calibrated)
    def observe(**kwargs):
        source = robot.context.camera.capture_wrist()
        return SkillResult(True, data={'image_path': source['color_path'], 'center_base_xyz': [.3, .2, .1]})
    run = Mock(side_effect=observe)
    monkeypatch.setattr(PlaceToContainerSkill, 'run', run)
    first = robot.observe_container('pink tray')
    calibrated.pose[0] += .01
    robot._move_to_cached_container_observe_pose = Mock(return_value=SkillResult(True))
    assert robot._observe_container_action('pink tray').data == first.data
    robot._move_to_cached_container_observe_pose.assert_not_called()
    assert run.call_count == 1
    with fresh_camera(robot.context.camera, 'probe'):
        second = robot._observe_container_action('pink tray')
    assert second.data['image_path'] != first.data['image_path']
    assert run.call_count == 2
    assert robot.place_plans['tray__pink'] == second.data


@pytest.mark.parametrize('action', ['place', 'clear'])
@pytest.mark.parametrize('outcome', ['success', 'failure', 'exception'])
def test_other_manipulations_invalidate_all_observation_entries(calibrated, monkeypatch, action, outcome):
    from core.skill_base import SkillResult
    from skill_lib.manipulation.clear_occlusion_to_buffer import ClearOcclusionToBufferSkill
    robot = make_robot(calibrated)
    robot.config.execute = True
    source = calibrated.camera.capture_wrist()['color_path']
    if action == 'place':
        robot.place_plans['tray__pink'] = {key: [300, 100, 200, math.pi, 0, 0] for key in
            ('approach_pose_mmrad', 'place_pose_mmrad', 'lift_pose_mmrad')}
        data = {'success': outcome == 'success'}
        robot._execute_cached_place = (Mock(side_effect=RuntimeError('partial execution')) if outcome == 'exception'
                                       else Mock(return_value=data))
        invoke = lambda: robot.place('pink tray')
    else:
        robot.observations['tube'] = {'kind': 'medicine'}
        result = SkillResult(outcome == 'success')
        run = (Mock(side_effect=RuntimeError('partial execution')) if outcome == 'exception'
               else Mock(return_value=result))
        monkeypatch.setattr(ClearOcclusionToBufferSkill, 'run', run)
        invoke = lambda: robot._invoke_clear_occlusion_skill('tube')
    if outcome == 'exception':
        with pytest.raises(RuntimeError, match='partial execution'):
            invoke()
    else:
        assert invoke().success is (outcome == 'success')
    assert calibrated.camera.capture_wrist()['color_path'] != source


def test_wrist_only_verifier_sends_two_truthfully_labelled_images(monkeypatch):
    from skill_lib.evaluation import verify_medicine_placed_vdm as module
    from core.context import RuntimeContext
    client = Mock()
    client.describe_images.return_value = '{"success":true,"confidence":0.9}'
    monkeypatch.setattr(module, 'OpenAIChatGPTVLMClient', Mock(return_value=client))
    module.VerifyMedicinePlacedVDMSkill(RuntimeContext()).run(
        medicine_query='tube', container_name='tray',
        before_global_color_path='before.png', before_wrist_color_path='before.png',
        after_global_color_path='after.png', after_wrist_color_path='after.png',
        base_url='unused', api_key='unused', model_name='unused', wrist_only=True)
    call = client.describe_images.call_args.kwargs
    assert call['image_paths'] == ['before.png', 'after.png']
    assert len(call['image_labels']) == 2 and all('wrist' in label for label in call['image_labels'])
    assert '- before_global:' not in call['prompt']


def test_late_inference_cannot_publish_during_manipulation(calibrated, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    robot = make_robot(calibrated)
    captured, release = threading.Event(), threading.Event()
    observe = fake_observer(robot)
    def slow(**kwargs):
        result = observe(**kwargs)
        captured.set()
        assert release.wait(2)
        return result
    monkeypatch.setattr(FindAndGraspMedicineSkill, 'run', Mock(side_effect=slow))
    memory = calibrated.camera.observation_memory
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(robot._observe_medicine_once, 'tube')
        try:
            assert captured.wait(2)
            robot.observations['old'] = {'kind': 'medicine'}
            robot.place_plans['old'] = {'pose': 'obsolete'}
            with memory.mutation('mock_grasp'):
                assert not robot.observations and not robot.place_plans
                release.set()
                result = future.result(timeout=2)
                assert not result.success and result.data['failure_type'] == 'stale_observation'
                assert not robot.observations and not memory.entries
        finally:
            release.set()
    assert memory.epoch == 2
    assert not robot.observations and not robot.place_plans


def test_commit_and_invalidation_cannot_interleave():
    memory = ObservationMemory()
    started, release, invalidating, invalidated = (threading.Event() for _ in range(4))
    state = {}
    def discard(reason): state.clear()
    memory.on_invalidate(discard)
    def publish():
        started.set()
        assert release.wait(2)
        state['observation'] = 'source_epoch_0'
    def invalidate():
        invalidating.set()
        memory.invalidate('scene_change', discard_scene=True)
        invalidated.set()
    with ThreadPoolExecutor(2) as pool:
        commit = pool.submit(memory.commit, 0, publish)
        try:
            assert started.wait(2)
            invalidation = pool.submit(invalidate)
            assert invalidating.wait(2)
            assert not invalidated.wait(.05)
        finally:
            release.set()
        assert commit.result(timeout=2)
        invalidation.result(timeout=2)
    assert state == {}
    assert not memory.commit(0, lambda: state.update(observation='late'))


def test_capture_waits_until_physical_action_finishes(calibrated):
    f = calibrated
    requested, complete = threading.Event(), threading.Event()
    def capture():
        requested.set()
        result = f.camera.capture_wrist()
        complete.set()
        return result
    with ThreadPoolExecutor(1) as pool:
        with f.camera.observation_memory.mutation('mock_place'):
            future = pool.submit(capture)
            assert requested.wait(2)
            assert not complete.wait(.05)
            f.arm.get_pose.assert_not_called()
        result = future.result(timeout=2)
    assert result['observation_cache']['epoch'] == 2


def test_last_capture_is_owned_by_each_inference_thread(calibrated):
    f = calibrated
    captured, release = threading.Event(), threading.Event()
    def observe_a():
        first = f.camera.capture_wrist()
        captured.set()
        assert release.wait(2)
        return first['color_path'], f.camera.last_wrist_capture['color_path']
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(observe_a)
        try:
            assert captured.wait(2)
            f.pose[0] += .1
            other = f.camera.capture_wrist()
        finally:
            release.set()
        first, last = future.result(timeout=2)
    assert first == last and first != other['color_path']


def test_force_refresh_bypass_is_local_to_its_request():
    memory = ObservationMemory(clock=lambda: 100)
    with memory.fresh():
        memory.put('frame', {'id': 1}, POSE, 100, 'scope')
        assert memory.get('frame', POSE, 'scope') is None
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(memory.get, 'frame', POSE, 'scope').result(timeout=2) == {'id': 1}


def test_physical_boundary_rejects_stale_grasp_plan():
    from runtime.observation_memory import scene_mutation, StaleObservationError
    memory = ObservationMemory()
    context = SimpleNamespace(camera=SimpleNamespace(observation_memory=memory))
    execution = Mock()
    @scene_mutation('mock_grasp')
    def execute(context, plan): execution()
    plan = {'grasp_input': {'observation_provenance': {'epoch': 0}}}
    memory.invalidate('external_change')
    with pytest.raises(StaleObservationError):
        execute(context, plan)
    execution.assert_not_called()


@pytest.mark.parametrize('outcome', ['success', 'exception'])
def test_direct_execution_boundary_always_invalidates(outcome):
    from runtime.observation_memory import scene_mutation
    memory = ObservationMemory(clock=lambda: 100)
    context = SimpleNamespace(camera=SimpleNamespace(observation_memory=memory))
    memory.put('old', {}, POSE, 100, 'scope')
    @scene_mutation('direct_skill')
    def execute(context):
        assert not memory.commit(memory.epoch, lambda: pytest.fail('Publish during motion'))
        memory.put('during', {}, POSE, 100, 'scope')
        if outcome == 'exception':
            raise RuntimeError('partial motion')
    if outcome == 'exception':
        with pytest.raises(RuntimeError, match='partial motion'):
            execute(context)
    else:
        execute(context)
    assert memory.epoch == 2 and not memory.entries
    assert memory.commit(2, lambda: None)


def test_stale_negative_result_does_not_erase_new_scene_results(calibrated, monkeypatch):
    from core.skill_base import SkillResult
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    robot = make_robot(calibrated)
    captured, release = threading.Event(), threading.Event()
    def slow_failure(**kwargs):
        robot.context.camera.capture_wrist()
        captured.set()
        assert release.wait(2)
        return SkillResult(False, error='target not found in old image')
    monkeypatch.setattr(FindAndGraspMedicineSkill, 'run', Mock(side_effect=slow_failure))
    memory = calibrated.camera.observation_memory
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(robot._observe_medicine_once, 'old_target')
        try:
            assert captured.wait(2)
            with memory.mutation('mock_change'):
                pass
            new = calibrated.camera.capture_wrist()
            robot.observations['new_target'] = {'kind': 'medicine'}
        finally:
            release.set()
        result = future.result(timeout=2)
    assert not result.success and result.data['failure_type'] == 'stale_observation'
    assert 'new_target' in robot.observations
    assert calibrated.camera.capture_wrist()['color_path'] == new['color_path']
    assert calibrated.device.capture_wrist.call_count == 6


def test_capture_invalidated_mid_acquisition_is_not_returned(calibrated, monkeypatch):
    from runtime.observation_memory import StaleObservationError
    camera = calibrated.camera
    freeze = camera._freeze
    def interrupted(capture):
        result = freeze(capture)
        camera.invalidate_observations('external_change')
        return result
    monkeypatch.setattr(camera, '_freeze', interrupted)
    with pytest.raises(StaleObservationError, match='acquiring RGB-D'):
        camera.capture_wrist()
    assert camera.last_wrist_capture is None
    assert not camera.observation_memory.entries


def test_base_frame_place_plan_survives_new_image_but_not_scene_change():
    from runtime.observation_memory import scene_mutation, StaleObservationError
    memory = ObservationMemory()
    context = SimpleNamespace(camera=SimpleNamespace(observation_memory=memory))
    calls = []
    @scene_mutation('place')
    def execute(context, plan): calls.append('executed')
    plan = {'observation_provenance': {'epoch': 0, 'scene_epoch': 0}}
    with memory.fresh('before_place_verification'):
        pass
    execute(context, plan)
    assert calls == ['executed']
    with pytest.raises(StaleObservationError, match='place planning'):
        execute(context, plan)
    assert calls == ['executed']
