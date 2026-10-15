"""Single-grasp selection and geometry regression; no hardware/model access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hardware.config import load_config
from hardware.interfaces import Pose6D
from hardware.transforms import load_transforms, pose_matrix
from hardware.gsnet_worker import roi_mask, select_top_one
from hardware.gsnet_single import (build_targets, envelope_min_z, plan, pose_difference,
                                    validate_capture, validate_roi)


class SingleGraspTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.transforms = load_transforms(self.cfg)
        self.workcell = {'tool_length_m': .19, 'tool_max_width_m': .10, 'tool_padding_m': .005,
                         'table_z_m': 0., 'table_clearance_m': .01}

    def test_selects_exactly_one_maximum_and_does_not_modify_input(self):
        rows = np.zeros((4, 17)); rows[:, 0] = [.3, .9, .2, .9]
        before = rows.copy()
        selected = select_top_one(rows)
        self.assertEqual(selected.shape, (1, 17))
        self.assertEqual(selected[0, 0], .9)
        selected[0, 1] = 5
        np.testing.assert_array_equal(rows, before)

    def test_empty_or_bad_predictions_never_synthesize_target(self):
        for rows in (np.zeros((0, 17)), np.zeros((2, 6)), np.full((1, 17), np.nan)):
            with self.subTest(shape=rows.shape), self.assertRaises((RuntimeError, ValueError)):
                select_top_one(rows)

    def test_roi_rejects_behind_camera_and_right_edge(self):
        intr = dict(fx=100, fy=100, ppx=50, ppy=50)
        points = [[0, 0, 1], [0, 0, -1], [.5, 0, 1], [0, .5, 1], [.1, .1, 1]]
        np.testing.assert_array_equal(roi_mask(points, intr, [0, 0, 100, 100]),
                                      [True, False, False, False, True])

    def test_roi_bounds(self):
        meta = {'color_intrinsics': {'width': 640, 'height': 480}}
        validate_roi([0, 0, 640, 480], meta)
        for roi in ([-1, 0, 10, 20], [10, 0, 10, 20], [0, 0, 641, 480]):
            with self.assertRaises(ValueError):
                validate_roi(roi, meta)

    def test_pose_comparison_handles_rpy_wrap(self):
        a = Pose6D(.4, 0, .5, np.pi, 0, 0)
        b = Pose6D(.4, 0, .5, -np.pi, 0, 0)
        self.assertLess(pose_difference(a, b)['rotation_rad'], 1e-12)

    def test_pregrasp_retreats_along_approach_and_tool_offset_applied_once(self):
        capture = Pose6D(.4, 0, .5, np.pi, 0, -np.pi/2)
        camera = np.eye(4); camera[:3, 3] = [.02, .03, .30]
        out = build_targets(camera, capture, self.transforms, self.cfg.cameras.wrist_serial,
                            'wrist_color_optical_frame', self.workcell)
        base_grasp = np.asarray(out['base_T_grasp'])
        target = pose_matrix(Pose6D(*out['targets']['grasp']['flange_xyz_rpy']))
        pre = pose_matrix(Pose6D(*out['targets']['pregrasp']['flange_xyz_rpy']))
        np.testing.assert_allclose(target @ self.transforms.flange_T_tcp,
                                   base_grasp @ self.transforms.grasp_T_tcp, atol=1e-12)
        np.testing.assert_allclose(pre[:3, 3] - target[:3, 3], -.08 * base_grasp[:3, 0])
        np.testing.assert_allclose(pre[:3, :3], target[:3, :3], atol=1e-12)

    def test_envelope_uses_full_19cm_tool_plus_padding(self):
        pose = Pose6D(.4, 0, .20, np.pi, 0, 0)
        self.assertAlmostEqual(envelope_min_z(pose, self.workcell), .005)
        self.assertLess(envelope_min_z(pose, self.workcell), self.workcell['table_clearance_m'])

    def make_capture(self, directory):
        root = Path(directory); (root / 'capture').mkdir()
        sample = {'pose': [.4, 0, .5, 0, 0, 0],
                  'result': {'stack': 'real', 'operation': 'pose', 'ok': True, 'cleanup_ok': True}}
        record = {'before': sample, 'after': json.loads(json.dumps(sample)),
                  'calibration': self.transforms.summary()}
        meta = {'device': {'serial_number': self.cfg.cameras.wrist_serial},
                'frame': f'realsense_{self.cfg.cameras.wrist_serial}_color_optical',
                'deprojection': 'rs.pointcloud.calculate on aligned depth'}
        (root / 'capture_pose.json').write_text(json.dumps(record))
        (root / 'capture/capture.json').write_text(json.dumps(meta))
        return root, record, meta

    def test_capture_rejects_robot_motion_and_fake_fk(self):
        with tempfile.TemporaryDirectory() as directory:
            root, record, _ = self.make_capture(directory)
            validate_capture(root, self.cfg)
            record['after']['pose'][0] += .002
            (root / 'capture_pose.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'moved'):
                validate_capture(root, self.cfg)
            record['after']['pose'][0] -= .002
            record['before']['result']['stack'] = 'fake'
            (root / 'capture_pose.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'REAL'):
                validate_capture(root, self.cfg)

    def test_capture_rejects_wrong_camera_or_depth_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, meta = self.make_capture(directory)
            meta['device']['serial_number'] = 'other'
            (root / 'capture/capture.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'serial'):
                validate_capture(root, self.cfg)
            meta['device']['serial_number'] = self.cfg.cameras.wrist_serial
            meta['frame'] = 'depth_optical_frame'
            (root / 'capture/capture.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'COLOR'):
                validate_capture(root, self.cfg)

    def test_plan_never_enables_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'targets.json').write_text('{}')
            target = {'targets': {'pregrasp': {'above_table_margin': True,
                                              'flange_xyz_rpy': [.4, 0, .5, 0, 0, 0]}}}
            with patch('hardware.gsnet_single.targets', return_value=target), \
                    patch('hardware.gsnet_single.FrankaMoveItArm') as constructor:
                arm = constructor.return_value
                arm.plan_pose.return_value = {'ok': True}
                report = plan(root, self.cfg, 'pregrasp')
                self.assertFalse(constructor.call_args.kwargs['execution_enabled'])
                self.assertFalse(report['motion_command_sent'])
                arm.move_pose.assert_not_called()
                arm.plan_pose.assert_called_once()
                arm.close.assert_called_once()

    def test_table_violation_does_not_call_robot_or_select_another_grasp(self):
        target = {'targets': {'grasp': {'above_table_margin': False}}}
        with patch('hardware.gsnet_single.targets', return_value=target), \
                patch('hardware.gsnet_single.FrankaMoveItArm') as constructor:
            with self.assertRaisesRegex(RuntimeError, 'no alternate grasp'):
                plan(Path('unused'), self.cfg, 'grasp')
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
