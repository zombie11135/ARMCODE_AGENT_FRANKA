"""Direction mapping checks; no actual robot/serial/camera access."""
from contextlib import redirect_stdout
import io
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from hardware.cli import main
from hardware.grasp_alignment import axes_report, rotation_grasp_to_tcp
from hardware.config import load_config
from hardware.interfaces import Pose6D
from hardware.transforms import load_transforms


class AlignmentTests(unittest.TestCase):
    def test_jaw_x_and_y_have_distinct_known_rotations(self):
        np.testing.assert_allclose(rotation_grasp_to_tcp([1, 0, 0]),
                                   [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(rotation_grasp_to_tcp([0, 1, 0]),
                                   [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    def test_arbitrary_mounting_angle_maps_axes_and_preserves_handedness(self):
        for theta in (0, .4, math.pi/2, math.pi, -math.pi/2):
            jaw = np.array([math.cos(theta), math.sin(theta), 0])
            transform = rotation_grasp_to_tcp(jaw)
            # With a canonical model grasp R_base_grasp=I, TCP forward must
            # follow base +X and the physical jaw must follow base +Y.
            np.testing.assert_allclose(transform @ [0, 0, 1], [1, 0, 0], atol=1e-12)
            np.testing.assert_allclose(transform @ jaw, [0, 1, 0], atol=1e-12)
            np.testing.assert_allclose(transform.T @ transform, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(transform), 1)

    def test_invalid_jaw_directions_rejected(self):
        for direction in ([0, 0, 0], [1, 0, .1], [float('nan'), 0, 0], [1, 0]):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                rotation_grasp_to_tcp(direction)

    def test_report_changes_base_directions_with_pose(self):
        result = axes_report(Pose6D(.4, .1, .5, math.pi, 0, -math.pi/2), load_transforms(load_config()))
        np.testing.assert_allclose(result['tcp_position_base_m'], [.4, .1, .34], atol=1e-12)
        np.testing.assert_allclose(result['candidates']['jaw_parallel_to_flange_X']['jaw_direction_base'],
                                   [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(result['candidates']['jaw_parallel_to_flange_Y']['jaw_direction_base'],
                                   [-1, 0, 0], atol=1e-12)
        self.assertFalse(result['configuration_written'])

    def test_offline_cli_never_reads_robot(self):
        with patch('hardware.cli.FrankaMoveItArm', side_effect=AssertionError('offline only')), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['calibration', 'axes', '--flange-pose', '0', '0', '0', '0', '0', '0']), 0)
        result=json.loads(output.getvalue())
        self.assertFalse(result['hardware_accessed'])
        self.assertFalse(result['motion_command_sent'])

    def test_current_pose_cli_constructs_read_only_adapter(self):
        with patch('hardware.cli.FrankaMoveItArm') as constructor, redirect_stdout(io.StringIO()) as output:
            robot=constructor.return_value
            robot.get_pose.return_value=Pose6D(.4, 0, .5, 0, 0, 0)
            self.assertEqual(main(['calibration','axes','--current-pose',
                                   '--closing-axis-base','0','1','0']),0)
        self.assertFalse(constructor.call_args.kwargs['execution_enabled'])
        robot.get_pose.assert_called_once_with()
        robot.close.assert_called_once_with()
        robot.move_pose.assert_not_called()
        robot.plan_pose.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result['pose_source'], 'stack_read_only')
        np.testing.assert_allclose(result['observed_base_line']['projected_direction_flange'], [0, 1, 0])

    def test_base_y_at_common_downward_pose_is_flange_x_line(self):
        result = axes_report(Pose6D(.3, 0, .5, math.pi, 0, -math.pi/2), load_transforms(load_config()),
                             closing_axis_base=[0, 1, 0])['observed_base_line']
        np.testing.assert_allclose(result['projected_direction_flange'], [-1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(result['line_angle_to_flange_X_deg'], 0)
        self.assertAlmostEqual(result['line_angle_to_flange_Y_deg'], 90)
        self.assertAlmostEqual(result['deviation_from_flange_xy_plane_deg'], 0)
        candidates = result['sign_ambiguous_candidates']
        np.testing.assert_allclose(candidates[0]['jaw_direction_base'], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(candidates[1]['jaw_direction_base'], [0, -1, 0], atol=1e-12)

    def test_approximate_observation_reports_projection_error(self):
        angle = .2
        result = axes_report(Pose6D(0, 0, 0, angle, 0, 0), load_transforms(load_config()),
                             closing_axis_base=[0, 1, 0])['observed_base_line']
        self.assertAlmostEqual(result['deviation_from_flange_xy_plane_deg'], math.degrees(angle))
        np.testing.assert_allclose(result['projected_direction_flange'], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(result['direction_flange_unprojected'], [0, math.cos(angle), -math.sin(angle)])

    def test_parallel_nonfinite_and_ambiguous_inputs_rejected(self):
        transforms = load_transforms(load_config())
        for vector in ([0, 0, 0], [0, 0, 1], [0, float('inf'), 0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                axes_report(Pose6D(0, 0, 0, 0, 0, 0), transforms, closing_axis_base=vector)
        with self.assertRaisesRegex(ValueError, 'not both'):
            axes_report(Pose6D(0, 0, 0, 0, 0, 0), transforms, [1, 0, 0], closing_axis_base=[0, 1, 0])


if __name__ == '__main__':
    unittest.main()
