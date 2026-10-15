"""Agent interfaces exercised with saved/synthetic observations and mocked devices."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.context import RuntimeContext
from hardware.agent_geometry import capture_pose, execute_grasp, grasp_plan, point_to_base
from hardware.config import DEFAULT_CONFIG, load_config
from hardware.fr3 import FrankaMoveItArm, MotionError
from hardware.interfaces import Pose6D
from hardware.robotiq import GripperError, GripperStatus, Robotiq2F85
from hardware.runtime import build_hardware
from hardware.transforms import load_transforms, pose_matrix
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.transforms = load_transforms(self.config)
        self.pose = Pose6D(.4, 0, .5, np.pi, 0, -np.pi/2)
        self.record = {'before': self.pose.values, 'after': self.pose.values,
            'serial': self.config.cameras.wrist_serial,
            'optical_frame': self.transforms.cameras['wrist']['optical_frame'],
            'calibration': self.transforms.summary()}
        self.source = {'capture_geometry': self.record, 'points_path': '/saved/points.npy',
            'color_path': '/saved/color.png', 'bbox': [3, 3, 14, 14],
            'depth_intrinsics': {'fx': 10, 'fy': 10, 'cx': 10, 'cy': 10}}
        self.tcp = pose_matrix(Pose6D(.3, .02, .25, np.pi, 0, -np.pi/2))
        camera_grasp = (np.linalg.inv(self.transforms.base_T_camera('wrist', self.pose)) @
                        self.tcp @ np.linalg.inv(self.transforms.grasp_T_tcp))
        candidate = {'index': 0, 'score': .8, 'translation': camera_grasp[:3, 3].tolist(),
                     'rotation_matrix': camera_grasp[:3, :3].tolist()}
        self.workcell = json.loads((Path(self.config.arm.workspace)/'config/workcell.json').read_text())
        self.result = {'success': True, 'candidate_grasps_camera': [candidate], 'workcell': self.workcell}
        self.plan = grasp_plan(self.source, self.result, self.transforms, self.workcell)

    def test_transform_matches_standalone_without_legacy_corrections(self):
        flange = pose_matrix(Pose6D(*self.plan['best_grasp_base_mrad']))
        np.testing.assert_allclose(flange @ self.transforms.flange_T_tcp, self.tcp, atol=1e-12)
        pre = self.plan['geometry']['targets']['pregrasp']['flange_xyz_rpy']
        np.testing.assert_allclose(np.array(pre[:3]) - flange[:3, 3], [0, 0, .08], atol=1e-12)

    def test_cached_image_uses_capture_pose_and_checks_calibration(self):
        expected = self.transforms.base_T_camera('wrist', self.pose)[:3, 3]
        np.testing.assert_allclose(point_to_base(self.source, [0, 0, 0], self.transforms), expected)
        bad = copy.deepcopy(self.source)
        bad['capture_geometry']['calibration']['T_flange_tcp'][0][3] = .123
        with self.assertRaisesRegex(ValueError, 'Calibration changed'):
            capture_pose(bad, self.transforms)

    def test_missing_moving_or_wrong_camera_capture_is_rejected(self):
        with self.assertRaises(ValueError):
            capture_pose({}, self.transforms)
        for field, value in [('serial', 'other'), ('after', [.5, 0, .5, np.pi, 0, -np.pi/2])]:
            bad = copy.deepcopy(self.source)
            bad['capture_geometry'][field] = value
            with self.assertRaises(ValueError):
                capture_pose(bad, self.transforms)

    def test_skill_passes_bbox_mask_and_geometry_to_gsnet(self):
        service = Mock()
        service.plan_grasp.return_value = self.result
        ctx = RuntimeContext(anygrasp=service, transforms=self.transforms)
        source = {**self.source, 'mask_npy_path': '/saved/mask.npy'}
        with patch.object(PlanGraspWithAnyGraspSkill, '_get_current_xarm_pose_mrad',
                          side_effect=AssertionError('Do not use current FK for saved image')):
            result = PlanGraspWithAnyGraspSkill(ctx).run(grasp_input=source)
        self.assertTrue(result.success, result.error)
        args = service.plan_grasp.call_args.kwargs
        self.assertEqual(args['bbox'], source['bbox'])
        self.assertEqual(args['mask_npy_path'], source['mask_npy_path'])
        self.assertEqual(args['capture_geometry'], self.record)
        service.save_geometry.assert_called_once()

    def test_contact_then_lift_from_actual_pose(self):
        arm, grip = Mock(), Mock(spec=Robotiq2F85)
        grip.assert_healthy.return_value = GripperStatus(True, True, 3, 2, 0, 255, 120, 20)
        arm.motion_guard = lambda: None
        arm.get_pose.return_value = Pose6D(.301, .019, .399, np.pi, 0, -np.pi/2)
        ctx = RuntimeContext(arm=arm, gripper=grip)
        opened, closed = Mock(return_value=True), Mock(return_value=True)
        result = execute_grasp(ctx, self.plan, opened, closed)
        self.assertTrue(result['success'])
        self.assertEqual(arm.move_pose.call_count, 3)
        np.testing.assert_allclose(arm.move_pose.call_args.args[0].values,
                                  [.301, .019, .449, np.pi, 0, -np.pi/2])
        closed.assert_called_once()

    def test_motion_or_close_failure_never_lifts_or_retries(self):
        arm, grip = Mock(), Mock()
        arm.motion_guard = None
        ctx = RuntimeContext(arm=arm, gripper=grip)
        closed = Mock(return_value=True)
        arm.move_pose.side_effect = MotionError({'error': 'endpoint failed'})
        with self.assertRaises(MotionError):
            execute_grasp(ctx, self.plan, lambda: True, closed)
        closed.assert_not_called()
        arm.reset_mock(); arm.move_pose.side_effect = None
        with self.assertRaises(GripperError):
            execute_grasp(ctx, self.plan, lambda: True, lambda: False)
        self.assertEqual(arm.move_pose.call_count, 2)
        arm.get_pose.assert_not_called()

    def test_agent_close_without_contact_lifts_and_transport_keeps_health_guard(self):
        from agents.simple_agent_system import SimpleAgentSystemConfig
        from agents.simple_execution_runtime import SimpleExecutionRuntime
        arm, grip = Mock(), Mock(spec=Robotiq2F85)
        arm.get_pose.return_value = Pose6D(.301, .019, .399, np.pi, 0, -np.pi/2)
        arm.motion_guard = None
        empty = GripperStatus(True, True, 3, 3, 0, 255, 240, 10)
        grip.open.return_value = GripperStatus(True, True, 3, 3, 0, 0, 3, 0)
        grip.close.return_value = empty
        grip.assert_healthy.return_value = empty
        context = RuntimeContext(arm=arm, gripper=grip, transforms=self.transforms)
        cfg = SimpleAgentSystemConfig(goal='offline no-contact test', execute_robot=True,
                                      hardware_config_path=str(DEFAULT_CONFIG))
        robot = SimpleExecutionRuntime(cfg)._build_robot(context)
        def move(pose):
            arm.motion_guard()
            return {'ok': True}
        arm.move_pose.side_effect = move
        result = execute_grasp(context, self.plan, robot.gripper_open, robot.gripper_close)
        self.assertTrue(result['success'])
        self.assertTrue(result['gripper_close_completed'])
        self.assertFalse(result['contact_required'])
        self.assertFalse(result['contact_detected'])
        self.assertEqual(arm.move_pose.call_count, 3)
        self.assertAlmostEqual(arm.move_pose.call_args.args[0].z, .449)
        grip.close.assert_called_once()
        arm.move_pose(self.pose)  # Subsequent transport also accepts no-contact status.
        grip.assert_healthy.side_effect = GripperError('communication fault')
        with self.assertRaisesRegex(GripperError, 'communication fault'):
            arm.move_pose(self.pose)

    def test_post_close_fault_prevents_lift_even_when_contact_is_not_required(self):
        arm, grip = Mock(), Mock(spec=Robotiq2F85)
        arm.motion_guard = None
        grip.assert_healthy.side_effect = GripperError('gripper session fault')
        context = RuntimeContext(arm=arm, gripper=grip)
        with self.assertRaisesRegex(GripperError, 'session fault'):
            execute_grasp(context, self.plan, lambda: True, lambda: True)
        self.assertEqual(arm.move_pose.call_count, 2)
        arm.get_pose.assert_not_called()

    def test_speed_override_and_advisory_match_tested_workflow(self):
        hw = build_hardware(DEFAULT_CONFIG, speed_scaling=.2, agent=True)
        self.assertEqual(hw.arm.config.speed_scaling, .2)
        self.assertEqual(hw.arm.config.singularity_screening, 'advisory')
        self.assertFalse(hw.arm.execution_enabled)
        for value in (0, -1, float('nan'), float('inf'), 1.01, True):
            with self.assertRaises(ValueError):
                hw.arm.set_speed_scaling(value)
        with self.assertRaises(RuntimeError):
            hw.arm.move_pose(self.pose)

    def test_closing_current_cutoff_aborts_before_lift(self):
        from hardware.robotiq import GripperCurrentLimitError
        arm, grip = Mock(), Mock()
        arm.motion_guard = None
        context = RuntimeContext(arm=arm, gripper=grip)
        close = Mock(side_effect=GripperCurrentLimitError('test current cutoff'))
        with self.assertRaises(GripperCurrentLimitError):
            execute_grasp(context, self.plan, lambda: True, close)
        self.assertEqual(arm.move_pose.call_count, 2)
        arm.get_pose.assert_not_called()
        close.assert_called_once()

    def test_unconfigured_home_uses_current_fr3_flange(self):
        from agents.simple_agent_system import SimpleAgentSystemConfig
        from agents.simple_execution_runtime import SimpleExecutionRuntime
        arm = Mock(); arm.get_pose.return_value = self.pose
        cfg = SimpleAgentSystemConfig(goal='test', hardware_config_path=str(DEFAULT_CONFIG))
        robot = SimpleExecutionRuntime(cfg)._build_robot(RuntimeContext(arm=arm, transforms=self.transforms))
        np.testing.assert_allclose(robot.config.home_pose_mmrad[:3], np.array(self.pose.values[:3])*1000)
        self.assertIsNone(robot.gripper_close)

    def test_place_release_uses_fixed_flange_height_after_tcp_conversion(self):
        from skill_lib.combound.place_to_container import PlaceToContainerSkill
        from skill_lib.simple_actions import SimpleRobotActions

        def estimate(center, rpy, transforms, arm_config):
            context = RuntimeContext(transforms=transforms, arm=SimpleNamespace(config=arm_config))
            skill = PlaceToContainerSkill(context)
            skill._capture = Mock(return_value={'color_path': '/unused/color.png',
                'depth_path': '/unused/depth.png', 'depth_intrinsics': self.source['depth_intrinsics']})
            skill._detect_container = Mock(return_value=[{'bbox': [2, 2, 18, 18]}])
            skill._load_depth = Mock(return_value=np.full((20, 20), .5))
            skill._camera_point_to_base = Mock(return_value=list(center))
            with patch('hardware.agent_geometry.point_to_base', return_value=list(center)):
                return skill._estimate_place_plan_once(camera_name='wrist', container_name='tray',
                    container_color='pink', target_prompts=None, detection_threshold=.25,
                    text_threshold=.2, max_detections_per_prompt=10, merge_iou=.6,
                    detection_index=0, center_offset_ratio=(0., 0.), depth_window_px=5,
                    fallback_intrinsics=None, current_ee_pose=self.pose.values,
                    handeye_rot=np.eye(3), handeye_trans=np.zeros(3), target_rpy=rpy,
                    place_clearance_mm=250., approach_clearance_mm=120.,
                    lift_after_place_mm=120., caption='offline place geometry')

        self.assertEqual(self.config.arm.place_flange_z_m, .3)
        for center_z, rpy in [(.01, (np.pi, 0., -1.57)), (.07, (3., .15, -1.57))]:
            with self.subTest(center_z=center_z, rpy=rpy):
                center = [.3, .1, center_z]
                plan = estimate(center, rpy, self.transforms, self.config.arm)
                self.assertEqual(plan['place_height_mode'], 'absolute_flange_z')
                self.assertEqual(plan['place_flange_z_mm'], 300.)
                self.assertEqual([plan[k][2] for k in
                    ('approach_pose_mmrad', 'place_pose_mmrad', 'lift_pose_mmrad')], [420., 300., 420.])
                values = plan['place_pose_mmrad']
                actual_tcp = pose_matrix(Pose6D(*[v / 1000 for v in values[:3]], *values[3:])) @ self.transforms.flange_T_tcp
                np.testing.assert_allclose(actual_tcp[:2, 3], center[:2], atol=1e-12)
                np.testing.assert_allclose(actual_tcp[:3, :3], pose_matrix(Pose6D(0, 0, 0, *rpy))[:3, :3])

                events = []
                robot = SimpleRobotActions(RuntimeContext(),
                    gripper_open=lambda: events.append('open') or True)
                robot._move_xarm = lambda pose: events.append(pose[2]) or True
                self.assertTrue(robot._execute_cached_place(plan)['success'])
                self.assertEqual(events, [420., 300., 'open', 420.])

        relative = estimate([.3, .1, .01], (np.pi, 0., 0.), self.transforms,
                            replace(self.config.arm, place_flange_z_m=None))
        self.assertAlmostEqual(relative['place_pose_mmrad'][2], 420.)
        self.assertEqual(relative['place_height_mode'], 'container_relative')
        legacy = estimate([.3, .1, .01], (np.pi, 0., 0.), None, self.config.arm)
        self.assertAlmostEqual(legacy['place_pose_mmrad'][2], 260.)

    def test_sdk_cloud_keeps_obstacles_outside_target_mask(self):
        import cv2
        from services.gsnet_service import GSNetService
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            xyz = np.zeros((20, 20, 3), dtype=np.float32); xyz[:, :, 2] = .5
            np.save(root/'points.npy', xyz)
            cv2.imwrite(str(root/'color.png'), np.zeros((20, 20, 3), np.uint8))
            mask = np.zeros((20, 20), bool); mask[4:12, 4:12] = True
            np.save(root/'mask.npy', mask)
            candidate = self.result['candidate_grasps_camera'][0]
            saved = {'score': .8, 'width_m': .05, 'depth_m': .02, 'height_m': .02,
                     'translation_m': candidate['translation'], 'rotation_matrix': candidate['rotation_matrix']}
            def worker(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                self.assertIn(b'element vertex 400', (Path(request['capture_dir'])/'scene.ply').read_bytes())
                np.testing.assert_equal(np.load(request['target_mask_path']), mask)
                out = Path(request['output_dir']); out.mkdir()
                (out/'grasps.json').write_text(json.dumps({'grasps': [saved]}))
            with patch('services.gsnet_service.PROJECT_ROOT', root), patch('services.gsnet_service.subprocess.run', side_effect=worker):
                result = GSNetService(DEFAULT_CONFIG).plan_grasp(color_path=root/'color.png',
                    points_path=root/'points.npy', intrinsics=self.source['depth_intrinsics'],
                    capture_geometry=self.record, mask_npy_path=root/'mask.npy', bbox=[3, 3, 14, 14],
                    output_json_path=root/'result.json')
            self.assertTrue(result['success'])
            self.assertEqual(result['num_candidates'], 1)

    def test_camera_waits_for_fresh_frame_and_freezes_images_with_pose(self):
        import cv2
        from hardware.agent_camera import CalibratedCamera
        with tempfile.TemporaryDirectory() as folder:
            camera = Mock(); camera.save_dir = folder
            frames = [{'device_serial': self.config.cameras.wrist_serial,
                'color_frame_number': number, 'timestamp': 101,
                'color': np.full((3, 4, 3), number, np.uint8),
                'depth': np.ones((3, 4), np.uint16),
                'points_map': np.ones((3, 4, 3), np.float32),
                'depth_aligned_to_color': True, 'color_intrinsics': {'coeffs': [0.] * 5}}
                for number in (10, 11, 12)]
            camera.capture_wrist.side_effect = frames
            arm = Mock(); arm.get_pose.return_value = self.pose
            wrapped = CalibratedCamera(camera, self.config, arm=arm)
            with patch('hardware.agent_camera.time.time', return_value=100), patch('hardware.agent_camera.time.sleep'):
                result = wrapped.capture_wrist()
            self.assertEqual(camera.capture_wrist.call_count, 3)
            self.assertEqual(arm.get_pose.call_count, 2)
            self.assertEqual(result['capture_geometry']['before'], self.pose.values)
            self.assertEqual(int(cv2.imread(result['color_path'])[0, 0, 0]), 12)
            self.assertTrue(Path(result['points_path']).is_file())
            frames[-1]['color'][:] = 99
            self.assertEqual(int(cv2.imread(result['color_path'])[0, 0, 0]), 12)

    def test_camera_rejects_motion_between_frame_and_pose_samples(self):
        from hardware.agent_camera import CalibratedCamera
        camera = Mock()
        camera.capture_wrist.side_effect = [{'color_frame_number': 1},
            {'color_frame_number': 3, 'timestamp': 101}]
        arm = Mock()
        arm.get_pose.side_effect = [self.pose, Pose6D(.41, 0, .5, np.pi, 0, -np.pi/2)]
        wrapped = CalibratedCamera(camera, self.config, arm=arm)
        with patch('hardware.agent_camera.time.time', return_value=100), patch('hardware.agent_camera.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Robot moved'):
                wrapped.capture_wrist()


if __name__ == '__main__':
    unittest.main()
