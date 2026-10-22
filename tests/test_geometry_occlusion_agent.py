"""Full agent skill loop with synthetic RGB-D, mock models and no hardware I/O."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from core.actions import observe, grasp, place
from core.context import RuntimeContext
from core.skill_base import SkillResult
from runtime.observation_memory import ObservationMemory, scene_mutation
from services.wrist_occlusion_service import WristOcclusionService, choose_blocker, associate_target
from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig
from skill_lib.actions.geometry_occlusion import prepare_grasp, validate_prepared


class ShapeBackend:
    checkpoint = Path('mock_shapes_not_a_model')
    confidence = .35

    def predict(self, image, prompts, **kwargs):
        x, y = image[:, :, 0], image[:, :, 1]
        blocked = image[0, 0, 2] == 0
        back = (x >= 10) & (x < 60) & (y >= 10) & (y < 50)
        front = (x >= 25) & (x < 45) & (y >= 20) & (y < 55)
        masks = [back & ~front, front] if blocked else [back]
        rows = []
        for index, mask in enumerate(masks):
            yy, xx = np.nonzero(mask)
            if len(xx):
                rows.append(dict(source_id=f'shape_{index}', mask=mask, confidence=.95-index*.05,
                                 bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                                 prompt='box'))
        return rows, dict(proposals_per_prompt={'box': len(rows)})


class Camera:
    active_camera_names = ('wrist',)
    observation_scope = 'test-calibration-and-models'

    def __init__(self, root):
        self.root = root
        self.stage = 0
        self.count = 0
        self.pose = [.4, 0., .7, 3.14, 0., -1.57]
        self.observation_memory = ObservationMemory()
        self.last_wrist_capture = None

    def observation_pose(self):
        return list(self.pose)

    def capture_wrist(self):
        self.count += 1
        folder = self.root/f'frame_{self.count}'
        folder.mkdir()
        y, x = np.indices((60, 80))
        color = np.dstack((x, y, np.full_like(x, self.stage))).astype(np.uint8)
        depth = np.full((60, 80), 700, np.uint16)
        depth[(x >= 10) & (x < 60) & (y >= 10) & (y < 50)] = 500
        if self.stage == 0:
            depth[(x >= 25) & (x < 45) & (y >= 20) & (y < 55)] = 400
        cv2.imwrite(str(folder/'color.png'), color)
        cv2.imwrite(str(folder/'depth.png'), depth)
        np.save(folder/'points.npy', np.zeros((60, 80, 3), np.float32))
        k = dict(width=80, height=60, fx=100., fy=100., ppx=40., ppy=30., coeffs=[0.]*5)
        capture = dict(color_path=str(folder/'color.png'), depth_path=str(folder/'depth.png'),
            points_path=str(folder/'points.npy'), depth_scale=.001, depth_unit_m_per_raw_unit=.001,
            depth_aligned_to_color=True, color_intrinsics=k, depth_intrinsics=k, timestamp=time.time(),
            camera_name='wrist', device_serial='wrist_serial',
            capture_geometry=dict(before=self.pose.copy(), after=self.pose.copy(), serial='wrist_serial',
                                  optical_frame='wrist_optical', calibration={'fixture': True}),
            observation_cache=dict(epoch=self.observation_memory.epoch,
                scene_epoch=self.observation_memory.scene_epoch, scope=self.observation_scope))
        self.last_wrist_capture = capture
        return capture


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    from skill_lib.manipulation.clear_occlusion_to_buffer import ClearOcclusionToBufferSkill
    from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
    from skill_lib.combound.place_to_container import PlaceToContainerSkill
    camera = Camera(tmp_path)
    service = WristOcclusionService('unused', output_dir=tmp_path/'occlusion', backend=ShapeBackend())
    context = RuntimeContext(camera=camera, occlusion=service)
    cfg = SimpleRobotConfig(enable_geometry_occlusion=True, enable_nesyro_gate=False,
        enable_occlusion_recovery=True, execute=True, observe_wait_s=0., occlusion_recovery_max_attempts=2,
        geometry_occlusion_buffer_poses_mmrad=[[300., -200., 250., 3.14, 0., -1.57]],
        occlusion_recovery_capture_dir=str(tmp_path/'clear'), nesyro_gate_trace_path=str(tmp_path/'gate.jsonl'),
        gate_experience_log_path=str(tmp_path/'experience.jsonl'))
    events, plans = [], []
    robot = SimpleRobotActions(context, cfg, gripper_open=lambda: True, gripper_close=lambda: True)
    def recognize(skill, **kwargs):
        events.append(('recognize', camera.stage, camera.observation_memory.epoch))
        return SkillResult(True, dict(selected_instance=dict(object_id='target_identity',
            bbox=[10, 10, 60, 50], confidence=.99, ocr_text='target'),
            selection_result=dict(score=.99, ranked_candidates=[]), image_path=kwargs['image_path']))
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_recognize_image', recognize)
    def plan(skill, grasp_input, **kwargs):
        plans.append(deepcopy(grasp_input))
        events.append(('plan', grasp_input.get('query'), camera.observation_memory.epoch))
        return SkillResult(True, dict(grasp_input=grasp_input,
            approach_pose_mmrad=[400, 0, 500, 3.14, 0, 0],
            best_grasp_xarm_mmrad=[400, 0, 400, 3.14, 0, 0], lift_pose_mmrad=[400, 0, 550, 3.14, 0, 0]))
    monkeypatch.setattr(PlanGraspWithAnyGraspSkill, 'run', plan)
    @scene_mutation('mock_clear')
    def execute_clear(skill, plan, **kwargs):
        events.append(('clear', plan['grasp_input']['object_id'], camera.observation_memory.epoch))
        camera.stage = 1
        return dict(success=True)
    monkeypatch.setattr(ClearOcclusionToBufferSkill, '_execute_to_buffer', execute_clear)
    monkeypatch.setattr(ClearOcclusionToBufferSkill, '_detect_blocker_candidates',
                        lambda *a, **k: pytest.fail('Graph-selected blocker must not be redetected'))
    @scene_mutation('mock_target_grasp')
    def execute_target(skill, plan, **kwargs):
        events.append(('grasp_target', camera.stage, camera.observation_memory.epoch))
        assert camera.stage == 1
        return dict(success=True)
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_execute_grasp', execute_target)
    def container(skill, **kwargs):
        camera.capture_wrist()
        events.append(('observe_container', camera.stage, camera.observation_memory.epoch))
        return SkillResult(True, dict(center_base_xyz=[.3, .2, 0.],
            selected_detection=dict(confidence=.99, color_score=.99, bbox=[0, 0, 80, 60], label='tray'),
            approach_pose_mmrad=[300, 200, 400, 3.14, 0, 0],
            place_pose_mmrad=[300, 200, 250, 3.14, 0, 0], lift_pose_mmrad=[300, 200, 400, 3.14, 0, 0]))
    monkeypatch.setattr(PlaceToContainerSkill, 'run', container)
    def move(pose):
        events.append(('move', pose, camera.observation_memory.epoch))
        camera.pose = [v/1000 for v in pose[:3]]+pose[3:]
        return True
    monkeypatch.setattr(robot, '_move_xarm', move)
    monkeypatch.setattr(robot, 'home', lambda: SkillResult(True))
    monkeypatch.setattr(robot, 'log_gate_outcome', lambda **kwargs: None)
    return SimpleNamespace(robot=robot, camera=camera, service=service, events=events, plans=plans,
                           recognize=recognize, root=tmp_path)


@pytest.mark.parametrize('legacy_gate', [False, True])
def test_agent_observe_clear_reobserve_grasp_and_relocalize_container(rig, legacy_gate):
    rig.robot.config.enable_nesyro_gate = legacy_gate
    result = rig.robot.run([observe('target'), grasp('target'), place('blue tray')])
    assert result.success, (result.error, result.data)
    names = [e[0] for e in rig.events]
    assert names.count('clear') == 1 and names.count('grasp_target') == 1
    clear = names.index('clear')
    target = names.index('grasp_target')
    assert clear < names.index('recognize', clear) < target
    assert names.count('observe_container') == 2  # Pre-observed plan invalidated by clear/grasp.
    assert rig.robot.occlusion_recovery_counts == {'target': 1}
    assert rig.robot.geometry_buffer_slots_used == 1
    assert rig.plans[0]['mask_npy_path'] != rig.plans[-1]['mask_npy_path']
    assert rig.plans[0]['observation_provenance']['epoch'] < rig.plans[-1]['observation_provenance']['epoch']
    assert not rig.robot.observations and not rig.robot.place_plans
    assert [e['event'] for e in rig.robot.geometry_occlusion_events].count('reobserve') == 1


def test_main_execution_runtime_runs_the_same_closed_loop(rig, monkeypatch):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    runtime = SimpleExecutionRuntime(SimpleNamespace())
    monkeypatch.setattr(runtime, '_save_memory', lambda *a, **k: None)
    monkeypatch.setattr(runtime, '_print_action_bbox_summary', lambda *a, **k: None)
    monkeypatch.setattr(runtime, '_log_gate_outcome_for_action', lambda *a, **k: None)
    monkeypatch.setattr(runtime, '_place_and_verify', lambda **kwargs: rig.robot.place(kwargs['action'].target))
    failure = Mock()
    monkeypatch.setattr(runtime, '_write_failure_report', failure)
    rig.robot.config.enable_nesyro_gate = True
    success = runtime._run_actions(rig.robot, rig.robot.context, rig.camera,
                                   [observe('target'), grasp('target'), place('blue tray')])
    assert success, failure.call_args
    failure.assert_not_called()
    assert any(e[0] == 'clear' for e in rig.events)
    assert any(e[0] == 'grasp_target' for e in rig.events)


def test_cli_flags_reach_agent_and_skill_configuration_without_starting_services(monkeypatch, tmp_path):
    import sys
    from app import cli
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    import hardware.runtime
    def apply(args):
        args.simple_actions = True
    monkeypatch.setattr(hardware.runtime, 'apply_cli_profile', apply)
    configs = []
    def run(args, goal, executor_args):
        config = cli.build_config(args, goal, executor_args)
        configs.append(config)
        config.hardware_config_path = None
        robot = SimpleExecutionRuntime(config)._build_robot(RuntimeContext())
        assert robot.config.enable_geometry_occlusion
        assert len(robot.config.geometry_occlusion_buffer_poses_mmrad) == 2
        assert not robot.config.execute
        return True
    monkeypatch.setattr(cli, 'run_goal', run)
    monkeypatch.setattr(sys, 'argv', ['app', '--goal', 'offline test', '--enable_geometry_occlusion',
        '--geometry_occlusion_checkpoint', str(tmp_path/'sam3.pt'), '--disable_geometry_occlusion_auxiliary',
        '--geometry_occlusion_buffer_pose_mmrad', '300', '-200', '250', '3.14', '0', '-1.57',
        '--geometry_occlusion_buffer_pose_mmrad', '350', '-200', '250', '3.14', '0', '-1.57'])
    cli.main()
    assert configs[0].enable_geometry_occlusion and not configs[0].geometry_occlusion_auxiliary


def test_manual_stop_failure_prevents_top_level_automatic_replanning(tmp_path, monkeypatch):
    from agents.simple_agent_system import SimpleArmCodeAgentSystem
    from runtime.config import SimpleAgentSystemConfig
    config = SimpleAgentSystemConfig(goal='offline', failure_report_path=str(tmp_path/'failure.json'))
    Path(config.failure_report_path).write_text(json.dumps(dict(automatic_replan=False)))
    agent = SimpleArmCodeAgentSystem(config)
    monkeypatch.setattr(agent, '_planning_stage', lambda: object())
    monkeypatch.setattr(agent, '_coding_stage', lambda *a, **k: [])
    monkeypatch.setattr(agent, '_executing_stage', lambda actions: False)
    monkeypatch.setattr(agent, '_replanning_stage', lambda *a: pytest.fail('must not reset clearing budget'))
    result = agent._run_task()
    assert not result.success and result.replan_attempts == 0


@pytest.mark.parametrize('geometry_enabled', [True, False])
def test_runtime_failure_report_preserves_graph_task_budget_across_replanning(tmp_path, geometry_enabled):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    config = SimpleNamespace(goal='offline', failure_report_path=str(tmp_path/'failure.json'),
                             enable_geometry_occlusion=geometry_enabled)
    runtime = SimpleExecutionRuntime(config)
    report = runtime._write_failure_report(grasp('target'), 'target', 'ordinary_gate_failure', 'stopped')
    assert report['automatic_replan'] is not geometry_enabled


def test_fixed_auxiliary_opens_rgbd_without_mutating_the_hardware_profile(monkeypatch, tmp_path):
    import hardware.runtime as wiring
    import hardware.agent_camera as camera_module
    import services.realsense_dual_camera_service as camera_service
    import services.runtime_service_cache as cache
    from hardware.config import HardwareConfig, CameraSettings
    profile = HardwareConfig(cameras=CameraSettings(mode='wrist'))
    monkeypatch.setattr(wiring, 'load_config', lambda path: profile)
    monkeypatch.setattr(cache, '_SERVICE_CACHE', {})
    device = Mock()
    factory = Mock(return_value=device)
    monkeypatch.setattr(camera_service, 'RealSenseDualCameraService', factory)
    monkeypatch.setattr(camera_module, 'CalibratedCamera', lambda device, config: SimpleNamespace(config=config))
    camera, params = wiring.camera_service('mock.yaml', str(tmp_path), fixed_auxiliary=True)
    assert profile.cameras.mode == 'wrist'
    assert camera.config.cameras.mode == 'dual'
    assert factory.call_args.kwargs['global_config'].enable_depth
    assert params['fixed_auxiliary']
    device.start.assert_not_called()


def test_model_service_is_lazy_and_reused_without_opening_cameras(monkeypatch, tmp_path):
    from runtime.config import SimpleAgentSystemConfig
    from runtime.services import build_occlusion_service
    import services.runtime_service_cache as cache
    monkeypatch.setattr(cache, '_SERVICE_CACHE', {})
    config = SimpleAgentSystemConfig(goal='offline', enable_geometry_occlusion=True,
        geometry_occlusion_checkpoint=str(tmp_path/'intentionally_not_loaded.pt'))
    service = build_occlusion_service(config)
    assert service.backend is None
    assert build_occlusion_service(config) is service


def test_holding_state_survives_image_invalidation_without_fake_visual_geometry():
    from nesyro_gate.scene_memory_builder import build_scene_memory_from_simple_robot
    scene = build_scene_memory_from_simple_robot({}, gripper_holding='completed_grasp_target')
    node = scene.get_object('completed_grasp_target')
    assert scene.in_gripper(node.object_id)
    assert node.identity_source == 'execution_state'
    assert not node.visible and not node.reachable and not node.graspable
    assert node.detector_confidence == 0


def test_dry_run_plans_blocker_but_never_assumes_scene_changed(rig):
    rig.robot.config.execute = False
    result = rig.robot.grasp('target')
    assert not result.success
    assert result.data['failure_type'] == 'geometry_occlusion_dry_run_clear_plan'
    assert len(rig.plans) == 1
    assert not any(e[0] in ('clear', 'grasp_target', 'move') for e in rig.events)
    assert rig.camera.stage == 0 and rig.robot.geometry_buffer_slots_used == 0
    assert not result.data['automatic_replan']


def test_no_buffer_or_no_recovery_never_starts_blocker_planning(rig):
    rig.robot.config.geometry_occlusion_buffer_poses_mmrad = []
    result = prepare_grasp(rig.robot, 'target')
    assert not result.success and result.data['failure_type'] == 'geometry_occlusion_buffer_missing'
    assert not rig.plans
    rig.robot.config.enable_occlusion_recovery = False
    result = prepare_grasp(rig.robot, 'target')
    assert result.data['failure_type'] == 'geometry_occlusion_clear_required'


def test_target_missing_remains_unknown_and_search_is_explicit(rig, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_recognize_image',
                        lambda *a, **k: SkillResult(False, error='not identified'))
    result = prepare_grasp(rig.robot, 'hidden target')
    assert not result.success and not rig.plans
    assert result.data['geometry_decision']['status'] == 'target_not_identified'
    rig.robot.config.execute = False
    rig.robot.config.occlusion_allow_unlocalized_target = True
    result = prepare_grasp(rig.robot, 'hidden target')
    assert result.data['failure_type'] == 'geometry_occlusion_dry_run_clear_plan'
    assert rig.plans[0]['query'] == 'occluding_object'
    assert any(e.get('status') == 'exploratory_clear_candidate' for e in rig.robot.geometry_occlusion_events)


def test_clear_failure_stops_before_reobserve_or_target_grasp(rig, monkeypatch):
    from skill_lib.manipulation.clear_occlusion_to_buffer import ClearOcclusionToBufferSkill
    @scene_mutation('mock_partial_clear')
    def fail(skill, plan, **kwargs):
        rig.events.append(('partial_clear',))
        return dict(success=False, error='release failed')
    monkeypatch.setattr(ClearOcclusionToBufferSkill, '_execute_to_buffer', fail)
    result = prepare_grasp(rig.robot, 'target')
    assert result.data['failure_type'] == 'geometry_occlusion_clear_failed'
    assert not result.data['automatic_replan']
    assert rig.robot.geometry_buffer_slots_used == 1
    assert not any(e[0] == 'grasp_target' for e in rig.events)
    assert not rig.robot.observations


def test_prepared_grasp_uses_the_analyzed_frame_and_rejects_epoch_or_file_change(rig):
    rig.camera.stage = 1
    assert rig.robot._observe_medicine_once('target').success
    prepared = rig.robot.observations['target']['result']
    assert validate_prepared(rig.robot.context, prepared, 'target')
    rig.camera.observation_memory.invalidate('manual rearrangement', discard_scene=True)
    with pytest.raises(ValueError, match='Scene changed'):
        validate_prepared(rig.robot.context, prepared, 'target')
    assert not rig.robot.observations
    Path(prepared['geometry_capture']['color_path']).write_bytes(b'overwritten capture')
    with pytest.raises(ValueError, match='source changed'):
        validate_prepared(rig.robot.context, prepared, 'target')


def test_in_memory_relation_edit_cannot_bypass_saved_graph_evidence(rig):
    rig.robot._observe_medicine_once('target')
    prepared = rig.robot.observations['target']['result']
    graph = prepared['geometry_occlusion']
    graph['primary']['possible_pairs'].clear()
    with pytest.raises(ValueError, match='differs from the saved'):
        validate_prepared(rig.robot.context, prepared, 'target')


def test_unrelated_auxiliary_graph_does_not_enter_primary_selection(rig):
    rig.robot._observe_medicine_once('target')
    graph = rig.robot.observations['target']['result']['geometry_occlusion']
    original = choose_blocker(graph)
    graph['auxiliary'] = {'possible_pairs': [{'source': 'aux_high_score', 'target': 'other'}]}
    assert choose_blocker(graph) == original
    nodes = graph['primary']['relation_graph']['nodes']
    assert associate_target(nodes, None)['node_id'] is None
    graph['target_association'] = dict(status='target_association_ambiguous', node_id=None)
    assert choose_blocker(graph, allow_search=True) == dict(status='target_association_ambiguous', blocker=None)


def test_graph_root_selection_resolves_chains_and_rejects_cycles(rig):
    rig.robot._observe_medicine_once('target')
    graph = deepcopy(rig.robot.observations['target']['result']['geometry_occlusion'])
    target = graph['target_association']['node_id']
    front = choose_blocker(graph)['blocker']
    upper = dict(front, node_id='upper')
    graph['primary']['relation_graph']['nodes'].append(upper)
    graph['primary']['possible_pairs'].append(dict(source='upper', target=front['node_id']))
    assert choose_blocker(graph)['blocker']['node_id'] == 'upper'
    graph['primary']['possible_pairs'].append(dict(source=target, target=front['node_id']))
    assert choose_blocker(graph)['status'] == 'ambiguous_or_cyclic_relations'


def test_occupied_buffer_slots_are_not_reused_when_occlusion_persists(rig, monkeypatch):
    from skill_lib.manipulation.clear_occlusion_to_buffer import ClearOcclusionToBufferSkill
    @scene_mutation('mock_clear_no_reveal')
    def clear(skill, plan, **kwargs):
        return dict(success=True)
    monkeypatch.setattr(ClearOcclusionToBufferSkill, '_execute_to_buffer', clear)
    result = prepare_grasp(rig.robot, 'target')
    assert result.data['failure_type'] == 'geometry_occlusion_buffer_missing'
    assert rig.robot.geometry_buffer_slots_used == 1 and len(rig.plans) == 1
    assert not any(e[0] == 'grasp_target' for e in rig.events)


def test_geometry_guard_also_runs_when_legacy_gate_is_disabled(rig):
    rig.robot.config.enable_occlusion_recovery = False
    result = rig.robot.gate_before_action(grasp('target'))
    assert not result.success and result.data['failure_type'] == 'geometry_occlusion_clear_required'
    assert not rig.plans


def test_late_graph_result_is_not_published_after_scene_change(rig, monkeypatch):
    analyze = rig.service.analyze
    def interrupted(*args, **kwargs):
        graph = analyze(*args, **kwargs)
        rig.camera.observation_memory.invalidate('moved while inference running', discard_scene=True)
        rig.robot.observations['new_scene'] = dict(kind='medicine')
        return graph
    monkeypatch.setattr(rig.service, 'analyze', interrupted)
    result = rig.robot._observe_medicine_once('target')
    assert not result.success and result.data['failure_type'] == 'stale_observation'
    assert 'target' not in rig.robot.observations and 'new_scene' in rig.robot.observations
