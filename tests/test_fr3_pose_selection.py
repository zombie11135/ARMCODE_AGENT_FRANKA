"""Pose selection and saved-artifact contracts; no hardware or model inference."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.context import RuntimeContext
from hardware.agent_geometry import grasp_plan, GraspSelectionError, validate_pose_selection
from hardware.config import load_config, DEFAULT_CONFIG
from hardware.interfaces import Pose6D
from hardware.transforms import load_transforms, pose_matrix
from services.gsnet_service import GSNetService
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill


class PoseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.transforms = load_transforms(self.cfg)
        self.capture = Pose6D(.4, 0, .6, math.pi, 0, -1.57)
        self.source = {'capture_geometry': {'before': self.capture.values, 'after': self.capture.values,
            'serial': self.cfg.cameras.wrist_serial,
            'optical_frame': self.transforms.cameras['wrist']['optical_frame'],
            'calibration': self.transforms.summary()}, 'points_path': '/unused/points.npy',
            'color_path': '/unused/color.png', 'depth_intrinsics': {'fx': 10, 'fy': 10, 'cx': 10, 'cy': 10}}
        self.workcell = json.loads((Path(self.cfg.arm.workspace)/'config/workcell.json').read_text())

    def candidate(self, index, score, tilt=0., z=.45, pitch=0., yaw=-1.57):
        flange = pose_matrix(Pose6D(.3, 0, z, math.pi + tilt, pitch, yaw))
        camera_grasp = (np.linalg.inv(self.transforms.base_T_camera('wrist', self.capture)) @ flange @
                        self.transforms.flange_T_tcp @ np.linalg.inv(self.transforms.grasp_T_tcp))
        return dict(index=index, score=score, width=.05, height=.02, depth=.02,
                    translation=camera_grasp[:3, 3].tolist(), rotation_matrix=camera_grasp[:3, :3].tolist())

    def result(self, candidates):
        return dict(success=True, candidate_grasps_camera=candidates, workcell=self.workcell)

    def plan(self, candidates, **options):
        return grasp_plan(self.source, self.result(candidates), self.transforms, self.workcell, **options)

    def test_lower_raw_score_can_win_on_base_flange_tilt(self):
        tilted = self.candidate(0, .95, tilt=.4)
        flat = self.candidate(1, .8)
        plan = self.plan([tilted, flat])
        self.assertEqual(plan['best_grasp_camera']['index'], 1)
        self.assertAlmostEqual(plan['pose_selection']['candidates'][0]['effective_score'], .55)
        np.testing.assert_allclose(plan['selected']['rotation_matrix'], flat['rotation_matrix'])
        self.assertEqual(plan['num_candidates'], 1)

    def test_hard_limits_reject_higher_score_and_report_no_survivors(self):
        steep = self.candidate(0, 10., tilt=math.radians(80))
        flat = self.candidate(1, .1)
        self.assertEqual(self.plan([steep, flat])['best_grasp_camera']['index'], 1)
        with self.assertRaises(GraspSelectionError) as caught:
            self.plan([steep])
        self.assertEqual(caught.exception.report['eligible_candidates'], 0)
        with self.assertRaises(ValueError):
            self.plan([steep], fallback_to_raw_when_no_good=True)

    def test_roll_wrap_and_disabled_pose_fix_do_not_rotate_candidate(self):
        candidate = self.candidate(0, .8, tilt=.01, yaw=.8)
        plan = self.plan([candidate], max_rx_tilt_deg=1., enable_pose_fix=False, lam_rz=0.)
        self.assertAlmostEqual(plan['pose_selection']['chosen_candidate']['tilt_rx_rad'], .01)
        np.testing.assert_allclose(plan['selected']['rotation_matrix'], candidate['rotation_matrix'])
        self.assertAlmostEqual(plan['best_grasp_base_mrad'][5], .8)

    def test_pitch_penalty_and_limit_are_independent_of_roll(self):
        tilted = self.candidate(0, .95, pitch=.3)
        flat = self.candidate(1, .8)
        self.assertEqual(self.plan([tilted, flat], lam_rx=0., lam_ry=1.)['best_grasp_camera']['index'], 1)
        self.assertEqual(self.plan([tilted, flat], lam_ry=0., max_ry_tilt_deg=10.)['best_grasp_camera']['index'], 1)

    def test_random_pool_uses_effective_score_and_seed(self):
        candidates = [self.candidate(0, .99, tilt=.5), self.candidate(1, .8), self.candidate(2, .7)]
        seen = set()
        for seed in range(8):
            options = dict(grasp_selection_mode='random_top_k_good', random_select_top_k=2, random_seed=seed)
            a, b = self.plan(candidates, **options), self.plan(candidates, **options)
            self.assertEqual(a['pose_selection']['choose_pool_indices'], [1, 2])
            self.assertEqual(a['selected'], b['selected'])
            seen.add(a['best_grasp_camera']['index'])
        self.assertEqual(seen, {1, 2})

    def test_table_invalid_best_does_not_hide_valid_candidate(self):
        self.workcell['grasp_table_clearance_check'] = True
        plan = self.plan([self.candidate(0, .99, z=.02), self.candidate(1, .5)])
        self.assertEqual(plan['best_grasp_camera']['index'], 1)
        self.assertFalse(plan['pose_selection']['candidates'][0]['tool_above_table'])

    def test_disabled_table_filter_still_applies_tilt_constraints(self):
        self.workcell['grasp_table_clearance_check'] = False
        plan = self.plan([self.candidate(0, 10., tilt=math.radians(80)),
                          self.candidate(1, .99, z=.02), self.candidate(2, .5)])
        self.assertEqual(plan['best_grasp_camera']['index'], 1)
        self.assertFalse(plan['pose_selection']['table_clearance_check_enabled'])
        self.assertFalse(plan['pose_selection']['candidates'][0]['accepted'])
        self.assertFalse(plan['pose_selection']['candidates'][1]['tool_above_table'])
        self.assertTrue(plan['pose_selection']['candidates'][1]['accepted'])

    def test_skill_requests_pool_and_forwards_selection_options(self):
        service = Mock()
        service.plan_grasp.return_value = self.result([self.candidate(0, .99, tilt=.5), self.candidate(1, .8)])
        context = RuntimeContext(anygrasp=service, transforms=self.transforms)
        result = PlanGraspWithAnyGraspSkill(context).run(grasp_input=self.source,
            grasp_selection_mode='best_good', lam_rx=2., visualize_selected_grasp=True)
        self.assertTrue(result.success, result.error)
        self.assertTrue(service.plan_grasp.call_args.kwargs['retain_all_candidates'])
        self.assertEqual(result.data['best_grasp_camera']['index'], 1)
        self.assertEqual(result.data['pose_selection']['options']['lam_rx'], 2.)
        service.save_geometry.assert_called_once()
        service.visualize_last_grasps_by_indices.assert_called_once_with([0])

    def test_service_keeps_full_pool_and_saves_only_final_choice(self):
        import cv2
        from hardware.gsnet_o3d import load_scene
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.zeros((20, 20, 3), dtype=np.float32); points[:, :, 2] = .5
            np.save(root/'points.npy', points)
            cv2.imwrite(str(root/'color.png'), np.zeros((20, 20, 3), np.uint8))
            candidates = [self.candidate(i, .99, tilt=.5) for i in range(60)] + [self.candidate(60, .8)]
            rows = np.asarray([[c['score'], c['width'], c['height'], c['depth'],
                *np.asarray(c['rotation_matrix']).ravel(), *c['translation'], -1] for c in candidates])
            def worker(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                self.assertTrue(request['export_filtered_candidates'])
                out = Path(request['output_dir']); out.mkdir()
                np.save(out/'filtered_candidates.npy', rows)
            service = GSNetService(DEFAULT_CONFIG)
            with patch('services.gsnet_service.PROJECT_ROOT', root), patch('services.gsnet_service.subprocess.run', side_effect=worker):
                result = service.plan_grasp(color_path=root/'color.png', points_path=root/'points.npy',
                    intrinsics=self.source['depth_intrinsics'], capture_geometry=self.source['capture_geometry'],
                    retain_all_candidates=True, output_json_path=root/'result.json')
            self.assertEqual(result['num_candidates'], 61)
            plan = grasp_plan(self.source, result, self.transforms, self.workcell)
            service.save_geometry(plan)
            self.assertEqual(plan['selected']['source_candidate_index'], 60)
            self.assertEqual(np.load(service.last_run/'gsnet/grasps.npy').shape, (1,17))
            public = json.loads((root/'result.json').read_text())
            self.assertEqual(public['best_grasp_camera']['index'], 60)
            load_scene(service.last_run)  # Actual saved viewer geometry consistency, no GUI/hardware.

    def test_invalid_options_fail_early(self):
        for options in ({'lam_rx': -1}, {'target_rx': float('nan')}, {'max_rx_tilt_deg': 181},
                        {'random_select_top_k': 0}, {'grasp_selection_mode': 'invalid'}):
            with self.assertRaises(ValueError):
                validate_pose_selection(**options)

    def test_cli_options_reach_runtime_robot_without_device_access(self):
        import sys
        from skill_lab import run_agent_system as cli
        from agents.simple_execution_runtime import SimpleExecutionRuntime
        captured = []
        def run_goal(args, *, goal, executor_args):
            config = cli.build_config(args, goal, executor_args)
            arm = Mock(); arm.get_pose.return_value = self.capture
            robot = SimpleExecutionRuntime(config)._build_robot(RuntimeContext(arm=arm, transforms=self.transforms))
            captured.append(robot.config)
            return True
        argv = ['agent', '--goal', 'offline test', '--grasp-selection-mode', 'random_top_k_good',
                '--grasp-random-select-top-k', '3', '--grasp-random-seed', '42',
                '--grasp-max-rx-tilt-deg', '35', '--grasp-lam-ry', '2.5']
        with patch.object(sys, 'argv', argv), patch.object(cli, 'run_goal', side_effect=run_goal):
            cli.main()
        self.assertEqual(captured[0].grasp_selection_mode, 'random_top_k_good')
        self.assertEqual(captured[0].grasp_random_select_top_k, 3)
        self.assertEqual(captured[0].grasp_random_seed, 42)
        self.assertEqual(captured[0].grasp_pose_options['max_rx_tilt_deg'], 35.)
        self.assertEqual(captured[0].grasp_pose_options['lam_ry'], 2.5)


if __name__ == '__main__':
    unittest.main()
