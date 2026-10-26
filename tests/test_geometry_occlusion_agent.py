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
    target_right = 60
    blocker_bottom = 55

    def predict(self, image, prompts, **kwargs):
        x, y = image[:, :, 0], image[:, :, 1]
        blocked = image[0, 0, 2] == 0
        back = (x >= 10) & (x < self.target_right) & (y >= 10) & (y < 50)
        front = (x >= 25) & (x < 45) & (y >= 20) & (y < self.blocker_bottom)
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
    real_recognize = FindAndGraspMedicineSkill._recognize_image
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
                           recognize=recognize, real_recognize=real_recognize, root=tmp_path)


@pytest.mark.parametrize('registered', [True, False])
def test_mobileclip2_identity_drives_geometry_agent_without_ocr(rig, monkeypatch, registered):
    """Exercise the real identity/graph bridge; encoder and all actuators are mocks."""
    from PIL import Image
    from services.object_reference_service import ObjectReferenceService, ReferenceSettings
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

    class Encoder:
        def encode(self, images, batch_size=16):
            features = np.zeros((len(images), 512), np.float32)
            features[:, 0] = 1.
            return features

    checkpoint = rig.root/'mock_mobileclip2.pt'
    checkpoint.write_bytes(b'not-a-real-checkpoint')
    references = ObjectReferenceService(ReferenceSettings(
        gallery_dir=str(rig.root/'gallery'), checkpoint=str(checkpoint), device='cpu',
        log_dir=str(rig.root/'recognition_logs'),
        encoder_model='mobileclip2_b', recognition_policy='reference_only',
        min_similarity=.8, min_class_margin=.02), encoder=Encoder())
    reference = rig.root/'reference.png'
    Image.new('RGB', (50, 40), 'red').save(reference)
    references.gallery.register(name='target' if registered else 'another medicine',
        image_path=reference, category='medicine_box', detector_prompt='medicine_box')
    context = rig.robot.context
    context.object_references = references
    context.detector = Mock()
    context.detector.detect.return_value = {'detections': [
        dict(object_id='detected_box', bbox=[10, 10, 60, 50], confidence=.99)]}
    context.ocr = Mock()
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_recognize_image', rig.real_recognize)
    rig.robot.config.enable_nesyro_gate = True
    # An unidentified target can retain geometry, but must not authorize exploration here.
    rig.robot.config.occlusion_allow_unlocalized_target = False

    observed = rig.robot._observe_medicine_once('target')
    assert observed.success, observed.error
    assert observed.data['target_identified'] is registered
    assert observed.data['geometry_occlusion']['primary']['relation_graph']['nodes']
    if registered:
        identity = observed.data['identity_evidence']
        assert identity['encoder_model'] == 'mobileclip2_b'
        assert identity['source'] == 'reference_gallery' and identity['accepted']
        assert observed.data['grasp_input']['color_path'] == observed.data['geometry_capture']['color_path']
        result = rig.robot.run([grasp('target'), place('blue tray')])
        assert result.success, (result.error, result.data)
        events = [e[0] for e in rig.events]
        assert events.count('clear') == events.count('grasp_target') == 1
        assert context.detector.detect.call_count == 2  # A fresh identity after clearing.
        assert [e['event'] for e in rig.robot.geometry_occlusion_events].count('reobserve') == 1
    else:
        assert observed.data['selected_instance'] is None
        result = prepare_grasp(rig.robot, 'target')
        assert not result.success
        assert not rig.plans
        assert not any(e[0] in ('clear', 'grasp_target') for e in rig.events)
        context.detector.detect.assert_not_called()
    assert context.ocr.mock_calls == []


@pytest.mark.parametrize('legacy_gate', [False, True])
def test_agent_observe_clear_reobserve_grasp_and_reuse_fixed_container(rig, legacy_gate):
    rig.robot.config.enable_nesyro_gate = legacy_gate
    result = rig.robot.run([observe('target'), grasp('target'), place('blue tray')])
    assert result.success, (result.error, result.data)
    names = [e[0] for e in rig.events]
    assert names.count('clear') == 1 and names.count('grasp_target') == 1
    clear = names.index('clear')
    target = names.index('grasp_target')
    assert clear < names.index('recognize', clear) < target
    assert names.count('observe_container') == 1  # Fixed container survives clear/grasp.
    assert rig.robot.occlusion_recovery_counts == {'target': 1}
    assert rig.robot.geometry_buffer_slots_used == 1
    assert rig.plans[0]['mask_npy_path'] != rig.plans[-1]['mask_npy_path']
    assert rig.plans[0]['observation_provenance']['epoch'] < rig.plans[-1]['observation_provenance']['epoch']
    assert set(rig.robot.observations) == {'blue tray'}
    assert set(rig.robot.place_plans) == {'tray__blue'}
    assert rig.robot.place_counts['tray__blue'] == 1
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


def test_validated_unblocked_geometry_is_not_vetoed_by_legacy_bbox_depth_spread(rig, monkeypatch):
    import nesyro_gate.scene_memory_builder as builder
    rig.camera.stage = 1
    rig.robot.config.enable_nesyro_gate = True
    monkeypatch.setattr(builder, '_estimate_depth_occlusion_level', lambda **kwargs: .59)
    assert prepare_grasp(rig.robot, 'target').success
    assert not rig.robot.build_scene_memory().objects['target'].graspable
    result = rig.robot.gate_before_action(grasp('target'))
    assert result.success
    assert result.data['gate']['decision'] == 'EXECUTE'
    assert result.data['gate']['lc'] == 1.
    assert result.data['gate']['occlusion_evidence_source'] == 'validated_wrist_relation_graph'


def test_actual_graph_occlusion_still_blocks_when_clearing_is_disabled(rig):
    rig.robot.config.enable_nesyro_gate = True
    rig.robot.config.enable_occlusion_recovery = False
    result = rig.robot.gate_before_action(grasp('target'))
    assert not result.success
    assert result.data['failure_type'] == 'geometry_occlusion_clear_required'


def test_yaw_pose_options_reach_both_blocker_and_target_planning(rig, monkeypatch):
    from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
    original = PlanGraspWithAnyGraspSkill.run
    seen = []
    options = dict(target_rz=-1.57, lam_rz=.25, enable_pose_fix=True)
    rig.robot.config.grasp_pose_options = options
    def plan(*args, **kwargs):
        seen.append({key: kwargs.get(key) for key in options})
        return original(*args, **kwargs)
    monkeypatch.setattr(PlanGraspWithAnyGraspSkill, 'run', plan)
    assert rig.robot.run([grasp('target'), place('blue tray')]).success
    assert seen == [options, options]


def test_validated_geometry_does_not_override_an_explicit_vlm_occlusion_verdict(rig, monkeypatch):
    rig.camera.stage = 1
    rig.robot.config.enable_nesyro_gate = True
    rig.robot.config.enable_occlusion_recovery = False
    def verdict(target):
        rig.robot.observations[target]['vlm_occlusion'] = dict(
            target_occluded=True, grasp_safe=False, occlusion_level=.7)
        return SkillResult(True)
    monkeypatch.setattr(rig.robot, '_run_vlm_occlusion_gate_if_enabled', verdict)
    result = rig.robot.gate_before_action(grasp('target'))
    assert not result.success
    assert 'target_occluded' in result.data['failed_conditions']


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
        '--disable_perception_pipeline',
        '--geometry_occlusion_checkpoint', str(tmp_path/'sam3.pt'), '--disable_geometry_occlusion_auxiliary',
        '--geometry_occlusion_buffer_pose_mmrad', '300', '-200', '250', '3.14', '0', '-1.57',
        '--geometry_occlusion_buffer_pose_mmrad', '350', '-200', '250', '3.14', '0', '-1.57'])
    cli.main()
    assert configs[0].enable_geometry_occlusion and not configs[0].geometry_occlusion_auxiliary
    assert not configs[0].enable_perception_pipeline


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


@pytest.mark.parametrize('configured_home', [None, [.35, 0., .45, 3.14, 0., -1.57]])
def test_xyz_buffer_uses_actual_startup_flange_angles_and_preserves_mm(monkeypatch, configured_home):
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    from runtime.config import SimpleAgentSystemConfig
    from hardware.interfaces import Pose6D
    import hardware.config
    monkeypatch.setattr(hardware.config, 'load_config',
        lambda path: SimpleNamespace(arm=SimpleNamespace(home_pose=configured_home)))
    arm = Mock()
    arm.get_pose.return_value = Pose6D(.4, .02, .5, -3.05, .03, -1.35)
    explicit = [300., -200., 250., 3.14, 0., -1.57]
    config = SimpleAgentSystemConfig(goal='offline test', hardware_config_path='mock.yaml',
        geometry_occlusion_buffer_poses_mmrad=[explicit],
        geometry_occlusion_buffer_positions_mm=[[450., -300., 200.]])
    robot = SimpleExecutionRuntime(config)._build_robot(RuntimeContext(arm=arm))
    assert robot.config.geometry_occlusion_buffer_poses_mmrad == [
        explicit, [450., -300., 200., -3.05, .03, -1.35]]
    assert config.geometry_occlusion_buffer_poses_mmrad == [explicit]
    arm.get_pose.assert_called_once_with()
    arm.move_pose.assert_not_called()
    assert robot.config.home_pose_mmrad == pytest.approx(
        [v*1000 for v in configured_home[:3]]+configured_home[3:]
        if configured_home else [400., 20., 500., -3.05, .03, -1.35])


def test_xyz_buffer_cli_reaches_full_agent_configuration(monkeypatch):
    import sys
    from app import cli
    import hardware.runtime
    monkeypatch.setattr(hardware.runtime, 'apply_cli_profile',
                        lambda args: setattr(args, 'simple_actions', True))
    configs = []
    def run(args, goal, executor_args):
        configs.append(cli.build_config(args, goal, executor_args))
        return True
    monkeypatch.setattr(cli, 'run_goal', run)
    monkeypatch.setattr(sys, 'argv', ['app', '--goal', '西瓜霜放到蓝色盘子',
        '--geometry_occlusion_buffer_xyz_mm', '450', '-300', '200',
        '--enable_geometry_occlusion', '--occlusion_allow_unlocalized_target', '--use_llm_planner',
        '--enable_llm_occlusion_recovery', '--geometry_occlusion_observation_budget', '3',
        '--geometry_occlusion_recovery_view_pose_mmrad', '400', '-100', '650', '3.14', '0', '-1.57'])
    cli.main()
    assert configs[0].geometry_occlusion_buffer_positions_mm == [[450., -300., 200.]]
    assert configs[0].enable_geometry_occlusion and configs[0].occlusion_allow_unlocalized_target
    assert configs[0].use_llm_planner
    assert configs[0].enable_llm_occlusion_recovery
    assert configs[0].geometry_occlusion_observation_budget == 3
    assert configs[0].geometry_occlusion_recovery_view_poses_mmrad == [[400., -100., 650., 3.14, 0., -1.57]]


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


class MenuPlanner:
    """Choose only from the actual runtime menu; does not call any endpoint."""
    def __init__(self, prefer_view=None):
        self.snapshots = []
        self.prefer_view = prefer_view

    def decide(self, snapshot):
        self.snapshots.append(deepcopy(snapshot))
        actions = snapshot['allowed_actions']
        choice = next((a for a in actions if a['action'] == 'clear'), None)
        if choice is None and self.prefer_view:
            choice = next((a for a in actions if a.get('view_id') == self.prefer_view), None)
        choice = choice or next((a for a in actions if a['action'] == 'observe'), actions[0])
        return dict(snapshot_id=snapshot['snapshot_id'], action_id=choice['action_id'],
                    reason='依据当前快照选择可执行的下一步')


@pytest.mark.parametrize('observation_budget', [0, 2])
def test_full_agent_llm_recovery_still_clears_grasps_and_places(rig, observation_budget):
    planner = MenuPlanner()
    rig.robot.context.occlusion_recovery_planner = planner
    rig.robot.config.enable_llm_occlusion_recovery = True
    rig.robot.config.geometry_occlusion_observation_budget = observation_budget
    rig.robot.config.enable_nesyro_gate = True
    result = rig.robot.run([observe('target'), grasp('target'), place('blue tray')])
    assert result.success, (result.error, result.data)
    names = [e[0] for e in rig.events]
    assert names.count('clear') == names.count('grasp_target') == 1
    assert names.count('observe_container') == 1
    assert len(planner.snapshots) == 1


@pytest.mark.parametrize('llm_enabled', [False, True])
def test_partial_blocker_depth_reaches_original_clear_skill_and_finishes_task(rig, monkeypatch, llm_enabled):
    original_capture = rig.camera.capture_wrist
    def partial_capture():
        capture = original_capture()
        if rig.camera.stage == 0:
            depth = cv2.imread(capture['depth_path'], cv2.IMREAD_UNCHANGED)
            y, x = np.indices(depth.shape)
            front = (x >= 25) & (x < 45) & (y >= 20) & (y < 55)
            depth[front & ((x % 10) < 3)] = 0  # 70% measured depth, no fabricated fill.
            cv2.imwrite(capture['depth_path'], depth)
        return capture
    monkeypatch.setattr(rig.camera, 'capture_wrist', partial_capture)
    rig.robot.config.enable_nesyro_gate = True
    rig.robot.config.enable_llm_occlusion_recovery = llm_enabled
    planner = MenuPlanner()
    rig.robot.context.occlusion_recovery_planner = planner
    # Graph-based clearing must not inherit the legacy ROI percentage cutoff.
    rig.robot.config.occlusion_min_depth_valid_ratio = .99
    result = rig.robot.run([grasp('target'), place('blue tray')])
    assert result.success, (result.error, result.data)
    names = [e[0] for e in rig.events]
    assert names.count('clear') == names.count('grasp_target') == 1
    assert names.count('observe_container') == 1
    if llm_enabled:
        node = next(n for n in planner.snapshots[0]['nodes'] if n['geometry_eligible'])
        assert node['depth_coverage'] == pytest.approx(.7)
        assert planner.snapshots[0]['geometry_requirements']['depth_coverage_filter_enabled'] is False


@pytest.mark.parametrize('llm_enabled', [False, True])
@pytest.mark.parametrize('occluded', [False, True])
def test_border_target_and_blocker_reach_planning_without_redundant_observation(rig, llm_enabled, occluded):
    rig.service.backend.target_right = 80  # Target touches the right image edge.
    rig.service.backend.blocker_bottom = 60  # Blocker touches the bottom edge.
    rig.camera.stage = 0 if occluded else 1
    rig.robot.config.enable_nesyro_gate = True
    rig.robot.config.enable_llm_occlusion_recovery = llm_enabled
    planner = MenuPlanner()
    rig.robot.context.occlusion_recovery_planner = planner
    result = rig.robot.run([grasp('target'), place('blue tray')])
    assert result.success, (result.error, result.data)
    names = [e[0] for e in rig.events]
    assert names.count('clear') == int(occluded)
    assert names.count('grasp_target') == 1
    assert names.count('recognize') == 1 + int(occluded)
    assert len(rig.plans) == 1 + int(occluded)
    assert np.load(rig.plans[-1]['mask_npy_path'])[:, -1].any()
    if occluded:
        assert np.load(rig.plans[0]['mask_npy_path'])[-1].any()
    assert not any(item['event'] == 'recovery_observation'
                   for item in rig.robot.geometry_recovery_state['history'])
    assert len(planner.snapshots) == int(llm_enabled and occluded)
    if planner.snapshots:
        blocker = next(n for n in planner.snapshots[0]['nodes'] if n['geometry_eligible'])
        assert blocker['quality_issues'] == []
        assert 'truncated_at_image_border' in blocker['quality_diagnostics']


def test_llm_recovers_inside_same_task_without_resetting_clear_or_buffer_budget(rig, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    from agents.simple_execution_runtime import SimpleExecutionRuntime
    planner = MenuPlanner()
    rig.robot.context.occlusion_recovery_planner = planner
    rig.robot.config.enable_llm_occlusion_recovery = True
    rig.robot.config.occlusion_recovery_max_attempts = 1
    rig.robot.config.occlusion_allow_unlocalized_target = True
    def recognize(skill, **kwargs):
        if rig.camera.count < 3:
            return SkillResult(False, error='target not confirmed yet')
        return rig.recognize(skill, **kwargs)
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_recognize_image', recognize)
    result = prepare_grasp(rig.robot, 'target')
    assert result.success, (result.error, result.data)
    assert len(planner.snapshots) == 2
    after_clear = planner.snapshots[1]
    assert after_clear['clear_count'] == after_clear['clear_limit'] == 1
    assert after_clear['buffer_slots'][0]['occupied']
    assert not any(a['action'] == 'clear' for a in after_clear['allowed_actions'])
    assert rig.robot.occlusion_recovery_counts == {'target': 1}
    assert rig.robot.geometry_buffer_slots_used == 1
    assert rig.robot.geometry_recovery_state['observation_counts'] == {'target': 1}
    assert [e[0] for e in rig.events].count('clear') == 1
    assert rig.camera.count == 3
    assert validate_prepared(rig.robot.context, rig.robot.observations['target']['result'], 'target')
    memory = rig.root/'task_memory.json'
    runtime = SimpleExecutionRuntime(SimpleNamespace(goal='offline', simple_memory_path=str(memory)))
    runtime._save_memory(rig.robot, [])
    saved = json.loads(memory.read_text())['geometry_recovery']
    assert saved['clear_counts'] == {'target': 1} and saved['buffer_slots_used'] == 1
    assert saved['state']['observation_counts'] == {'target': 1}


def test_llm_observation_budget_ends_in_pause_without_repeat_clear(rig, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    planner = MenuPlanner()
    rig.robot.context.occlusion_recovery_planner = planner
    rig.robot.config.enable_llm_occlusion_recovery = True
    rig.robot.config.occlusion_allow_unlocalized_target = True
    rig.robot.config.geometry_occlusion_observation_budget = 1
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_recognize_image',
                        lambda *args, **kwargs: SkillResult(False, error='still unknown'))
    result = prepare_grasp(rig.robot, 'target')
    assert not result.success and result.data['failure_type'] == 'geometry_occlusion_recovery_paused'
    assert result.data['recovery_pending'] and not result.data['automatic_replan']
    assert len(planner.snapshots) == 3  # Clear, one fresh observation, pause.
    assert planner.snapshots[-1]['allowed_actions'] == [dict(action_id='pause', action='pause')]
    assert rig.camera.count == 3 and rig.robot.geometry_buffer_slots_used == 1
    assert [e[0] for e in rig.events].count('clear') == 1


def test_llm_can_choose_only_an_explicit_registered_recovery_view(rig, monkeypatch):
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    rig.camera.stage = 1
    rig.robot.config.enable_llm_occlusion_recovery = True
    pose = [400., -100., 650., 3.14, 0., -1.57]
    rig.robot.config.geometry_occlusion_recovery_view_poses_mmrad = [pose]
    rig.robot.context.occlusion_recovery_planner = MenuPlanner(prefer_view='view_1')
    def recognize(skill, **kwargs):
        if rig.camera.count == 1: return SkillResult(False, error='need another view')
        return rig.recognize(skill, **kwargs)
    monkeypatch.setattr(FindAndGraspMedicineSkill, '_recognize_image', recognize)
    result = prepare_grasp(rig.robot, 'target')
    assert result.success, (result.error, result.data)
    assert [('move', pose, 0)] == [e for e in rig.events if e[0] == 'move']
    assert rig.robot.geometry_recovery_state['observation_counts'] == {'target': 1}


def test_llm_fabricated_action_or_stale_clear_never_starts_motion(rig):
    rig.robot.config.enable_llm_occlusion_recovery = True
    planner = Mock()
    planner.decide.side_effect = lambda s: dict(snapshot_id=s['snapshot_id'],
        action_id='clear:invented-node', reason='invalid')
    rig.robot.context.occlusion_recovery_planner = planner
    result = prepare_grasp(rig.robot, 'target')
    assert result.data['failure_type'] == 'geometry_occlusion_recovery_planner_failed'
    assert not rig.plans and not rig.robot.geometry_buffer_slots_used
    def stale(s):
        rig.camera.observation_memory.invalidate('scene changed during model call', discard_scene=True)
        action = next(a for a in s['allowed_actions'] if a['action'] == 'clear')
        return dict(snapshot_id=s['snapshot_id'], action_id=action['action_id'], reason='old scene')
    planner.decide.side_effect = stale
    result = prepare_grasp(rig.robot, 'target')
    assert result.data['failure_type'] == 'geometry_occlusion_recovery_stale'
    assert not rig.plans and not rig.robot.geometry_buffer_slots_used


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


def test_clear_communication_error_preserves_first_fault_in_report(rig, monkeypatch):
    from hardware.robotiq import GripperError
    from skill_lib.manipulation.clear_occlusion_to_buffer import ClearOcclusionToBufferSkill
    diagnostics = dict(communication_mode='process', worker_pid=12345,
        first_fault=dict(message='Status update gap exceeded', status_gap_s=.71,
                         state_timeout_s=.5, status_transaction_s=.01))
    gripper = Mock()
    gripper.session_info.return_value = diagnostics
    rig.robot.context.gripper = gripper
    def fail(skill, plan, **kwargs):
        raise GripperError('Status update gap exceeded')
    monkeypatch.setattr(ClearOcclusionToBufferSkill, '_execute_to_buffer', fail)
    result = prepare_grasp(rig.robot, 'target')
    assert result.data['failure_type'] == 'geometry_occlusion_clear_failed'
    clear = result.data['clear_result']
    assert clear['failure_type'] == 'occlusion_recovery_error'
    assert clear['gripper_session'] == diagnostics
    assert not result.data['automatic_replan']
    assert not any(e[0] in ('grasp_target', 'clear') for e in rig.events)


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
