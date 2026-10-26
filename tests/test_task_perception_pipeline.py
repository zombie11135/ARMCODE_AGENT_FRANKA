"""Pipeline overlap and cache correctness with saved arrays and fake devices."""
from copy import deepcopy
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from core.actions import grasp, place
from core.context import RuntimeContext
from runtime.task_perception_pipeline import start_task_perception_pipeline
from skill_lib.actions.geometry_occlusion import prepare_grasp
from test_geometry_occlusion_agent import rig
from test_grasp_yaw_preference import scene


def launch(rig, captured=None, enabled=True):
    return start_task_perception_pipeline(rig.robot, [grasp('target'), place('blue tray')],
        {'wrist': captured or rig.camera.capture_wrist(), 'global': None},
        SimpleNamespace(enable_perception_pipeline=enabled,
                        nesyro_gate_trace_path=str(rig.root/'gate.jsonl')))


@pytest.mark.parametrize('occluded', [False, True])
def test_saved_frame_analysis_overlaps_container_movement_and_owner_publishes(rig, monkeypatch, occluded):
    rig.camera.stage = 0 if occluded else 1
    owner = threading.get_ident()
    started, moving = threading.Event(), threading.Event()
    predict = rig.service.backend.predict
    def background_predict(*args, **kwargs):
        assert threading.get_ident() != owner
        started.set()
        assert moving.wait(3), 'Inference must overlap container observation'
        return predict(*args, **kwargs)
    monkeypatch.setattr(rig.service.backend, 'predict', background_predict)
    capture = rig.camera.capture_wrist()
    home = list(rig.camera.pose)
    take_picture = rig.camera.capture_wrist
    def owner_capture():
        assert threading.get_ident() == owner, 'Worker must never acquire another image'
        return take_picture()
    monkeypatch.setattr(rig.camera, 'capture_wrist', owner_capture)
    emitted_on = []
    rig.robot.context.event_callback = lambda event: emitted_on.append(threading.get_ident())
    pipeline = launch(rig, capture)
    try:
        assert started.wait(3)
        assert 'target' not in rig.robot.observations
        rig.camera.pose[0] += .15
        moving.set()
        assert rig.robot.preobserve_containers([place('blue tray')]).success
        rig.camera.pose = home
        assert pipeline.publish()
        assert rig.camera.count == 2  # Home snapshot plus foreground container capture.
        monkeypatch.setattr(rig.service.backend, 'predict', predict)
        assert prepare_grasp(rig.robot, 'target').success
        assert rig.camera.count == 2 + int(occluded)  # Only clear requires a new frame.
        assert [e[0] for e in rig.events].count('clear') == int(occluded)
        assert [e[0] for e in rig.events].count('recognize') == 1 + int(occluded)
        assert emitted_on and set(emitted_on) == {owner}
        report = json.loads(pipeline.trace_path.read_text())
        assert report['status'] == 'published' and report['join_wait_s'] >= 0
    finally:
        moving.set()
        pipeline.close()


@pytest.mark.parametrize('change', ['scene', 'refresh', 'pose', 'future_timestamp'])
def test_pipeline_discards_stale_results_without_publishing_or_executing(rig, change):
    capture = rig.camera.capture_wrist()
    if change == 'future_timestamp':
        capture['timestamp'] = time.time()+60
    pipeline = launch(rig, capture)
    try:
        assert pipeline.future.result(timeout=5).success
        if change == 'scene':
            rig.camera.observation_memory.invalidate('grasp', discard_scene=True)
        elif change == 'refresh':
            rig.camera.observation_memory.invalidate('fresh_capture')
        elif change == 'pose':
            rig.camera.pose[0] += .1
        assert not pipeline.publish()
        assert 'target' not in rig.robot.observations
        assert not rig.plans
        assert rig.robot.geometry_buffer_slots_used == 0
        assert rig.robot.occlusion_recovery_counts == {}
        assert not rig.robot._pipeline_frame_bindings
    finally:
        pipeline.close()


def test_long_startup_computation_uses_scene_validity_without_refreshing_capture_time(rig):
    rig.camera.stage = 1
    capture = rig.camera.capture_wrist()
    capture['timestamp'] = time.time()-60
    timestamp = capture['timestamp']
    assert rig.camera.observation_memory.policy.max_age_s == 30
    pipeline = launch(rig, capture)
    try:
        assert pipeline.publish()
        assert rig.robot.observe('target').success
        assert prepare_grasp(rig.robot, 'target').success
        assert rig.camera.count == 1
        assert rig.robot.observations['target']['result']['geometry_capture']['timestamp'] == timestamp
        assert rig.robot.grasp('target').success
        assert rig.camera.count == 1
        assert not rig.robot._pipeline_frame_bindings
    finally:
        pipeline.close()


def test_disabled_pipeline_does_not_freeze_or_infer(rig, monkeypatch):
    frozen = Mock(side_effect=AssertionError('Disabled pipeline must not start work'))
    monkeypatch.setattr(rig.service, 'freeze', frozen)
    assert launch(rig, enabled=False) is None
    frozen.assert_not_called()


def test_worker_failure_leaves_normal_observation_available(rig, monkeypatch):
    original = rig.service.backend.predict
    monkeypatch.setattr(rig.service.backend, 'predict', Mock(side_effect=RuntimeError('model failed')))
    pipeline = launch(rig)
    try:
        assert not pipeline.publish()
        assert 'target' not in rig.robot.observations
    finally:
        pipeline.close()
    monkeypatch.setattr(rig.service.backend, 'predict', original)
    assert prepare_grasp(rig.robot, 'target').success


@pytest.mark.parametrize('occluded', [False, True])
def test_prefetch_uses_original_capture_mask_and_never_owns_devices(rig, occluded):
    from skill_lib.manipulation.clear_occlusion_to_buffer import _shrink_bbox
    rig.camera.stage = 0 if occluded else 1
    service = Mock()
    service.plan_grasp.return_value = dict(run_dir='mock-run')
    rig.robot.context.anygrasp = service
    rig.robot.context.transforms = object()
    pipeline = launch(rig)
    try:
        result = pipeline.future.result(timeout=5)
        assert result.success and pipeline.record['candidate_prefetch'] == 'ready'
        assert pipeline.context.camera is pipeline.context.arm is pipeline.context.gripper is None
        assert pipeline.context.xarm is pipeline.context.robot is None
        kwargs = service.plan_grasp.call_args.kwargs
        graph = result.data['geometry_occlusion']
        selected = next(n for n in graph['primary']['relation_graph']['nodes']
                        if n['node_id'] == pipeline.record['candidate_node_id'])
        expected_box = (_shrink_bbox(selected['bbox_xyxy'], rig.robot.config.occlusion_blocker_bbox_shrink_ratio)
                        if occluded else selected['bbox_xyxy'])
        assert kwargs['bbox'] == expected_box
        assert kwargs['mask_npy_path'] == selected['visible_mask_path']
        assert kwargs['capture_geometry'] == result.data['geometry_capture']['capture_geometry']
        assert kwargs['retain_all_candidates'] and kwargs['collision_detection']
        assert not rig.plans and rig.robot.geometry_buffer_slots_used == 0
    finally:
        pipeline.close()


def test_freeze_copies_sdk_cloud_before_foreground_capture_can_replace_source(rig):
    original = rig.camera.capture_wrist()
    frozen, root = rig.service.freeze(original)
    assert frozen['points_path'] != original['points_path']
    np.save(original['points_path'], np.ones((60, 80, 3), np.float32))
    assert not np.load(frozen['points_path']).any()


def test_full_runtime_launches_pipeline_before_containers_and_consumes_before_actions(rig, monkeypatch):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    from runtime.config import SimpleAgentSystemConfig
    import hardware.runtime
    rig.camera.stage = 1
    rig.camera.start = Mock()
    config = SimpleAgentSystemConfig(goal='offline pipeline', execute_robot=True,
        disable_grip=True, camera_warmup_s=0, enable_geometry_occlusion=True,
        nesyro_gate_trace_path=str(rig.root/'gate.jsonl'),
        simple_memory_path=str(rig.root/'memory.json'))
    runtime = SimpleExecutionRuntime(config)
    monkeypatch.setattr(hardware.runtime, 'require_task_geometry', lambda config: None)
    monkeypatch.setattr(runtime, '_build_execution_context', lambda: (rig.robot.context, rig.camera))
    monkeypatch.setattr(runtime, '_build_robot', lambda context: rig.robot)
    monkeypatch.setattr(runtime, '_init_arm', lambda ip: Mock())
    monkeypatch.setattr(runtime, '_load_memory', lambda robot: None)
    def actions(robot, context, camera_service, actions):
        assert robot.observations['target']['result']['geometry_capture']['timestamp']
        assert [e[0] for e in rig.events].count('recognize') == 1
        assert rig.camera.count == 2
        return robot.run(actions).success
    monkeypatch.setattr(runtime, '_run_actions', actions)
    assert runtime.run([grasp('target'), place('blue tray')])
    report = json.loads((rig.root/'perception_pipeline.json').read_text())
    assert report['status'] == 'published'
    assert rig.camera.count == 2


@pytest.fixture
def candidate_service(scene, monkeypatch):
    from services import gsnet_service as module
    root = scene.root
    (root/'config').mkdir()
    (root/'config/workcell.json').write_text(json.dumps(scene.workcell))
    config = SimpleNamespace(arm=SimpleNamespace(workspace=str(root)),
                             perception=SimpleNamespace(gsnet_root=str(root/'mock_model')))
    monkeypatch.setattr(module, 'load_config', lambda path: config)
    monkeypatch.setattr(module, 'load_transforms', lambda config: scene.transforms)
    monkeypatch.setattr(module, 'PROJECT_ROOT', root)
    monkeypatch.chdir(root)
    service = module.GSNetService('mock_profile_not_read')
    cloud = np.zeros((20, 20, 3), np.float32)
    cloud[:, :, 2] = .5
    np.save(root/'points.npy', cloud)
    cv2.imwrite(str(root/'color.png'), np.zeros((20, 20, 3), np.uint8))
    np.save(root/'mask.npy', np.ones((20, 20), bool))
    candidate = scene.candidate()
    row = [candidate['score'], candidate['width'], candidate['height'], candidate['depth'],
           *np.asarray(candidate['rotation_matrix']).ravel(), *candidate['translation'], -1]
    def infer(request_path, log):
        request = json.loads(request_path.read_text())
        output = scene.root/request['output_dir']
        output.mkdir()
        np.save(output/'filtered_candidates.npy', np.asarray([row]))
    worker = Mock(side_effect=infer)
    monkeypatch.setattr(service, '_run_inference', worker)
    options = dict(color_path=str(root/'color.png'), points_path=str(root/'points.npy'),
        mask_npy_path=str(root/'mask.npy'), bbox=[4,4,16,16], bbox_filter_inner_margin_ratio=.05,
        intrinsics=scene.source['depth_intrinsics'], capture_geometry=scene.source['capture_geometry'],
        retain_all_candidates=True, scene_token=['fixture', 1], output_json_path=str(root/'result.json'))
    return SimpleNamespace(service=service, worker=worker, options=options, scene=scene)


def test_prefetched_candidates_reused_by_real_skill_with_final_yaw_selection(candidate_service):
    from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
    c = candidate_service
    first = c.service.plan_grasp(**c.options)
    source = dict(c.scene.source, color_path=c.options['color_path'], points_path=c.options['points_path'],
        mask_npy_path=c.options['mask_npy_path'], bbox=c.options['bbox'],
        observation_provenance=dict(scope='fixture', scene_epoch=1))
    result = PlanGraspWithAnyGraspSkill(RuntimeContext(anygrasp=c.service, transforms=c.scene.transforms)).run(
        grasp_input=source, bbox_filter_inner_margin_ratio=.05)
    assert result.success, result.error
    assert result.data['inference_cache_hit']
    assert abs(result.data['best_grasp_base_mrad'][5]+1.57) < .002
    assert result.data['selected']['jaw_symmetry_flipped']
    assert not first.get('selected')
    assert c.worker.call_count == 1
    third = c.service.plan_grasp(**c.options)
    assert third['candidate_grasps_camera'] == first['candidate_grasps_camera']
    assert 'selected' not in third


@pytest.mark.parametrize('change', ['rgb', 'points', 'mask', 'roi', 'intrinsics', 'collision', 'scene', 'pose', 'table'])
def test_candidate_cache_cannot_cross_changed_inputs(candidate_service, change):
    c = candidate_service
    if change == 'collision':
        # Match AnyGrasp's collision option in both warm and consume requests.
        model_request = c.service._model_request
        c.service._model_request = lambda **kwargs: dict(model_request(**kwargs),
            collision_detection=kwargs.get('collision_detection', True))
    c.service.plan_grasp(**c.options)
    updated = deepcopy(c.options)
    if change == 'rgb':
        cv2.imwrite(updated['color_path'], np.ones((20,20,3), np.uint8))
    elif change in ('points', 'mask'):
        path = updated['points_path' if change == 'points' else 'mask_npy_path']
        array = np.load(path)
        array[0] = .6 if change == 'points' else False
        np.save(path, array)
    elif change == 'roi':
        updated['bbox'] = [3,3,17,17]
    elif change == 'intrinsics':
        updated['intrinsics']['fx'] += 10
    elif change == 'collision':
        updated['collision_detection'] = False
    elif change == 'scene':
        updated['scene_token'] = ['fixture', 2]
    elif change == 'pose':
        updated['capture_geometry']['before'][0] += .1
        updated['capture_geometry']['after'][0] += .1
    else:
        workcell = dict(c.scene.workcell, table_z_m=.01)
        (c.scene.root/'config/workcell.json').write_text(json.dumps(workcell))
    assert not c.service.plan_grasp(**updated)['inference_cache_hit']
    assert c.worker.call_count == 2


def test_candidate_cache_is_bounded_and_does_not_cache_inference_errors(candidate_service):
    c = candidate_service
    c.worker.side_effect, original = RuntimeError('model failed'), c.worker.side_effect
    with pytest.raises(RuntimeError, match='model failed'):
        c.service.plan_grasp(**c.options)
    assert not c.service._candidate_cache
    c.worker.side_effect = original
    for margin in range(5):
        c.service.plan_grasp(**dict(c.options, bbox_filter_margin_px=margin))
    assert len(c.service._candidate_cache) == 4
    assert not c.service.plan_grasp(**c.options)['inference_cache_hit']
