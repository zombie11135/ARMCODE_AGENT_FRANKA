"""Selection provenance and operator execution boundaries; all hardware mocked."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hardware.config import load_config
from hardware.fr3 import FrankaMoveItArm
from hardware.interfaces import Pose6D
from hardware.gsnet_ranked import plan, move


class RankedGraspTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_plan_outputs_one_original_grasp_and_never_executes(self):
        rows = np.zeros((2, 17))
        rows[:, 0] = [.8, .6]
        rows[:, 1] = .07
        rows[:, 4:13] = np.eye(3).ravel()
        rows[:, 13:16] = [[.01, .02, .3], [.02, .03, .4]]
        meta = {'device': {'serial_number': 'test'}, 'frame': 'test_color'}
        transforms = Mock(cameras={'wrist': {'optical_frame': 'test_color'}})
        transforms.base_T_camera.return_value = np.eye(4)
        transforms.summary.return_value = {'test': 'calibration'}
        target = {'targets': {'grasp': {'above_table_margin': True,
                                      'flange_xyz_rpy': [.3, 0, .4, 3.14, 0, 0]}}}
        robot = {'ok': True, 'grasp_selection': {'candidate_index': 1,
            'goal_joints': list(range(7)), 'motion_command_sent': False}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'result'
            def infer(*args, **kwargs):
                (root / 'gsnet').mkdir()
                np.save(root / 'gsnet/candidates.npy', rows)
            with patch('hardware.gsnet_ranked.validate_capture', return_value=(None, meta, transforms, None)), \
                    patch('hardware.gsnet_ranked.validate_roi'), \
                    patch('hardware.gsnet_ranked.build_targets', return_value=target) as geometry, \
                    patch('hardware.gsnet_ranked.subprocess.run', side_effect=infer), \
                    patch('hardware.gsnet_ranked.render'), \
                    patch('hardware.gsnet_ranked.FrankaMoveItArm') as constructor:
                arm = constructor.return_value
                arm.plan_grasps.return_value = robot
                report = plan(Path(tmp), root, self.cfg, Path(tmp), [0, 0, 100, 100])
                self.assertFalse(constructor.call_args.kwargs['execution_enabled'])
                self.assertFalse(report['motion_command_sent'])
                self.assertFalse(report['jaw_symmetry_flipped'])
                self.assertEqual(len(arm.plan_grasps.call_args.args[0]), 2)
                for call, row in zip(geometry.call_args_list, rows):
                    np.testing.assert_array_equal(np.asarray(call.args[0])[:3, :3], row[4:13].reshape(3, 3))
                np.testing.assert_array_equal(np.load(root / 'gsnet/grasps.npy'), rows[1:2])
                self.assertEqual(len(json.loads((root / 'gsnet/grasps.json').read_text())['grasps']), 1)
                arm.move_pose.assert_not_called()
                arm.move_joint_goal.assert_not_called()

    def test_move_requires_execution_and_forwards_exact_selected_joint_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / 'workspace'; (workspace / 'config').mkdir(parents=True)
            for name in ('workcell', 'singularity'):
                (workspace / 'config' / (name + '.json')).write_text('{}')
            cfg = replace(self.cfg, arm=replace(self.cfg.arm, workspace=str(workspace)))
            report = {'jaw_symmetry_flipped': False, 'calibration': {}, 'policy': {}, 'workcell': {},
                'robot': {'ok': True, 'grasp_selection': {'motion_command_sent': False}},
                'flange_xyz_rpy': [.3, 0, .4, 3.14, 0, 0], 'goal_joints': [.1] * 7}
            (root / 'selection.json').write_text(json.dumps(report))
            with patch('hardware.gsnet_ranked.FrankaMoveItArm') as constructor, \
                    patch('hardware.gsnet_ranked.load_transforms') as transforms:
                transforms.return_value.summary.return_value = {}
                with self.assertRaisesRegex(ValueError, '--execute'):
                    move(root, cfg, False)
                constructor.assert_not_called()
                move(root, cfg, True)
                self.assertTrue(constructor.call_args.kwargs['execution_enabled'])
                arm = constructor.return_value
                arm.move_joint_goal.assert_called_once_with(Pose6D(*report['flange_xyz_rpy']), report['goal_joints'])
                arm.move_pose.assert_not_called()
                report['jaw_symmetry_flipped'] = True
                (root / 'selection.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'original GSNet orientation'):
                    move(root, cfg, True)
                self.assertEqual(constructor.call_count, 1)

    def test_pinned_motion_cannot_bypass_host_execution_guard(self):
        with patch('subprocess.Popen') as process:
            arm = FrankaMoveItArm(self.cfg.arm, execution_enabled=False)
            with self.assertRaisesRegex(RuntimeError, 'execution_enabled'):
                arm.move_joint_goal(Pose6D(.3, 0, .4, 3.14, 0, 0), [.1] * 7)
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
