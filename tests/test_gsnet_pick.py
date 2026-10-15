"""End-to-end workflow ordering/failure boundaries with all devices mocked."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hardware.config import load_config
from hardware.gsnet_pick import run, contact_guard
from hardware.interfaces import Pose6D
from hardware.robotiq import GripperStatus, GripperError, Robotiq2F85


def status(obj=3, active=True):
    return GripperStatus(active, True, 3 if active else 0, obj, 0, 255, 140, 100)


class PickTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = self.folder / 'gsnet'
        (self.model / 'checkpoints').mkdir(parents=True)
        (self.model / 'realsense_capture.py').touch()
        (self.model / 'checkpoints/graspness_realsense.tar').touch()
        self.cfg = load_config()
        self.arm = Mock(last_result=None)
        self.arm.plan_pose.return_value = {'ok': True, 'cleanup_ok': True}
        self.arm.move_pose.return_value = {'ok': True, 'cleanup_ok': True}
        self.actual = Pose6D(.302, .014, .3, 3.12, .01, -1.5)
        self.arm.get_pose.return_value = self.actual
        self.gripper = Mock(spec=Robotiq2F85)
        self.gripper.start_session.return_value = status()
        self.gripper.activate.return_value = status()
        self.gripper.open.return_value = status()
        self.gripper.close.return_value = status(2)
        self.gripper.assert_healthy.return_value = status(2)
        self.gripper.session_info.return_value = {'active': False}
        self.gripper.status.return_value = status(2)
        self.geometry = {'selected': {'score': .8, 'width_m': .07},
            'targets': {'pregrasp': {'above_table_margin': True,
                                     'flange_xyz_rpy': [.3, 0, .4, 3.14, 0, -1.57]},
                        'grasp': {'above_table_margin': True,
                                  'flange_xyz_rpy': [.3, 0, .3, 3.14, 0, -1.57]}},
            'workcell': {'tool_length_m': .19, 'tool_max_width_m': .10, 'tool_padding_m': .005,
                         'table_z_m': 0., 'table_clearance_m': .01}}
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.capture = self.stack.enter_context(patch('hardware.gsnet_pick.capture'))
        self.capture.return_value = None
        self.infer = self.stack.enter_context(patch('hardware.gsnet_pick.infer', return_value=None))
        self.stack.enter_context(patch('hardware.gsnet_pick.targets', return_value=self.geometry))
        self.arm_ctor = self.stack.enter_context(patch('hardware.gsnet_pick.FrankaMoveItArm', return_value=self.arm))
        self.grip_ctor = self.stack.enter_context(patch('hardware.gsnet_pick.Robotiq2F85', return_value=self.gripper))

    def run_pick(self, **options):
        return run(self.folder / 'result', self.cfg, self.model, hold=False, **options)

    def test_preview_has_no_actuation_and_uses_fresh_capture(self):
        result = self.run_pick()
        self.assertTrue(result['ok'])
        self.assertFalse(result['arm_motion_attempted'])
        self.grip_ctor.assert_not_called()
        self.arm.move_pose.assert_not_called()
        self.capture.assert_called_once()
        self.infer.assert_called_once()
        self.arm.plan_pose.assert_called_once()
        self.assertFalse(self.arm_ctor.call_args.kwargs['execution_enabled'])
        self.assertEqual(self.arm_ctor.call_args.args[0].singularity_screening, 'advisory')
        self.assertEqual(self.cfg.arm.singularity_screening, 'endpoint')
        self.assertEqual(result['roi_xyxy'], [0, 0, 640, 480])

    def test_contact_then_lift_five_cm_from_actual_pose_without_orientation_flip(self):
        result = self.run_pick(execute=True)
        self.assertTrue(result['ok'])
        self.assertTrue(result['contact_detected'])
        self.assertTrue(result['lift_completed'])
        stages = [s['stage'] for s in result['steps']]
        self.assertLess(stages.index('move_grasp'), stages.index('close_gripper'))
        self.assertLess(stages.index('close_gripper'), stages.index('lift'))
        goals = [call.args[0] for call in self.arm.move_pose.call_args_list]
        self.assertEqual(goals[1].values, self.geometry['targets']['grasp']['flange_xyz_rpy'])
        self.assertAlmostEqual(goals[2].z - self.actual.z, .05)
        self.assertEqual(goals[2].values[3:], self.actual.values[3:])
        self.assertEqual(goals[2].values[:2], self.actual.values[:2])
        self.gripper.activate.assert_not_called()
        self.gripper.open.assert_called_once()
        self.gripper.close.assert_called_once()
        self.gripper.disconnect.assert_called_once()
        saved = json.loads((self.folder / 'result/run.json').read_text())
        self.assertTrue(saved['lift_completed'])

    def test_no_contact_never_lifts_or_retries(self):
        self.gripper.close.return_value = status(3)
        result = self.run_pick(execute=True)
        self.assertFalse(result['ok'])
        self.assertFalse(result['lift_completed'])
        self.assertEqual(self.arm.move_pose.call_count, 2)
        self.arm.get_pose.assert_not_called()
        self.gripper.close.assert_called_once()
        self.gripper.open.assert_called_once()  # No automatic release/retry.

    def test_arm_failure_and_interrupt_never_close(self):
        self.arm.move_pose.side_effect = [dict(ok=True, cleanup_ok=True), KeyboardInterrupt()]
        result = self.run_pick(execute=True)
        self.assertFalse(result['ok'])
        self.assertEqual(result['stage'], 'move_grasp')
        self.gripper.close.assert_not_called()
        self.assertEqual(self.arm.move_pose.call_count, 2)
        self.gripper.disconnect.assert_called_once()

    def test_failed_arm_response_never_advances(self):
        self.arm.move_pose.return_value = {'ok': False, 'cleanup_ok': False}
        result = self.run_pick(execute=True)
        self.assertFalse(result['ok'])
        self.arm.move_pose.assert_called_once()
        self.gripper.close.assert_not_called()

    def test_opening_contact_stops_before_arm_motion(self):
        self.gripper.open.return_value = status(1)
        result = self.run_pick(execute=True)
        self.assertFalse(result['ok'])
        self.arm.move_pose.assert_not_called()
        self.gripper.close.assert_not_called()

    def test_bad_geometry_stops_before_any_actuation(self):
        self.geometry['targets']['grasp']['above_table_margin'] = False
        result = self.run_pick(execute=True)
        self.assertFalse(result['ok'])
        self.arm.move_pose.assert_not_called()
        self.gripper.open.assert_not_called()
        self.gripper.activate.assert_not_called()

    def test_lift_zero_stops_at_contact(self):
        result = self.run_pick(execute=True, lift_m=0)
        self.assertTrue(result['ok'])
        self.assertEqual(self.arm.move_pose.call_count, 2)
        self.arm.get_pose.assert_not_called()

    def test_inactive_gripper_activation_precedes_open(self):
        self.gripper.assert_healthy.return_value = status(active=False)
        result = self.run_pick(execute=True, lift_m=0)
        self.assertTrue(result['ok'])
        stages = [s['stage'] for s in result['steps']]
        self.assertLess(stages.index('activate_gripper'), stages.index('open_gripper'))
        self.gripper.activate.assert_called_once()

    def test_capture_error_does_not_activate_or_move(self):
        self.capture.side_effect = RuntimeError('camera unavailable')
        result = self.run_pick(execute=True)
        self.assertFalse(result['ok'])
        self.infer.assert_not_called()
        self.arm.move_pose.assert_not_called()
        self.gripper.activate.assert_not_called()
        self.gripper.disconnect.assert_called_once()

    def test_contact_guard_rejects_slip_and_session_fault(self):
        self.gripper.assert_healthy.return_value = status(3)
        with self.assertRaisesRegex(GripperError, 'contact'):
            contact_guard(self.gripper)
        self.gripper.assert_healthy.side_effect = GripperError('communication fault')
        with self.assertRaisesRegex(GripperError, 'communication'):
            contact_guard(self.gripper)

    def test_manual_release_in_same_session_and_quit_does_not_close_again(self):
        lines = iter(['status', 'open', 'quit'])
        result = run(self.folder / 'result', self.cfg, self.model, execute=True,
                     hold=True, input_fn=lambda _: next(lines))
        self.assertTrue(result['ok'])
        self.assertEqual(self.gripper.open.call_count, 2)  # initial open + explicit release
        self.gripper.close.assert_called_once()
        self.gripper.start_session.assert_called_once()
        self.gripper.disconnect.assert_called_once()

    def test_visualization_is_before_actuation_and_reopens_with_execution_feedback(self):
        def show(root, continuation):
            if continuation == 'execute':
                self.arm.move_pose.assert_not_called()
                self.gripper.open.assert_not_called()
                self.gripper.activate.assert_not_called()
            else:
                self.assertEqual(continuation, 'done')  # This test disables the hold console.
                saved = json.loads((root / 'run.json').read_text())
                self.assertTrue(saved['lift_completed'])
            return {'viewer': 'Open3D'}
        with patch('hardware.gsnet_pick.preview_o3d', side_effect=show) as viewer:
            result = self.run_pick(execute=True, visualize=True)
        self.assertTrue(result['ok'])
        self.assertEqual(viewer.call_count, 2)

    def test_failed_preview_never_starts_actuation(self):
        with patch('hardware.gsnet_pick.preview_o3d', side_effect=RuntimeError('no display')):
            result = self.run_pick(execute=True, visualize=True)
        self.assertFalse(result['ok'])
        self.arm.move_pose.assert_not_called()
        self.gripper.open.assert_not_called()
        self.gripper.close.assert_not_called()

    def test_post_motion_preview_failure_preserves_completed_grasp(self):
        with patch('hardware.gsnet_pick.preview_o3d', side_effect=[{}, RuntimeError('no display')]):
            result = self.run_pick(execute=True, visualize=True)
        self.assertTrue(result['ok'])
        self.assertTrue(result['lift_completed'])
        self.assertIn('o3d_result_error', result)


if __name__ == '__main__':
    unittest.main()
