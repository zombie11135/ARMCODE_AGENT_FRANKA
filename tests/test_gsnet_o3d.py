"""Saved-frame alignment and Open3D geometry checks; no windows or hardware."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hardware.config import load_config
from hardware.gsnet_single import build_targets
from hardware.gsnet_o3d import load_scene, transform_points, build_geometry, view
from hardware.interfaces import Pose6D
from hardware.transforms import load_transforms, pose_matrix


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.observation = self.root / 'observation'
        (self.observation / 'capture').mkdir(parents=True)
        self.transforms = load_transforms(load_config())
        self.capture_pose = Pose6D(.4, 0, .5, np.pi, 0, -np.pi/2)
        self.camera_grasp = np.eye(4); self.camera_grasp[:3, 3] = [.02, .03, .3]
        self.cell = {'tool_length_m': .19, 'tool_max_width_m': .10, 'tool_padding_m': .005,
                     'table_z_m': 0., 'table_clearance_m': .01}
        self.camera = self.transforms.cameras['wrist']
        self.geometry = build_targets(self.camera_grasp, self.capture_pose, self.transforms,
                                      self.camera['serial'], self.camera['optical_frame'], self.cell)
        self.geometry.update(calibration=self.transforms.summary(), workcell=self.cell,
            selected={'score': .7, 'width_m': .07, 'depth_m': .03, 'height_m': .02,
                      'T_input_grasp': self.camera_grasp.tolist()})
        self.write('targets.json', self.geometry)
        self.record = {'before': {'pose': self.capture_pose.values}, 'calibration': self.transforms.summary()}
        self.write('capture_pose.json', self.record)
        self.meta = {'device': {'serial_number': self.camera['serial']},
                     'frame': f"realsense_{self.camera['serial']}_color_optical"}
        self.write('capture/capture.json', self.meta)
        (self.observation / 'capture/scene.ply').write_text(
            'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n'
            'property float z\nend_header\n0 0 .3\n.01 0 .3\n0 .01 .3\n')

    def write(self, relative, value):
        (self.observation / relative).write_text(json.dumps(value))

    def test_capture_frame_and_tcp_offset_are_applied_exactly_once(self):
        scene = load_scene(self.root)
        base_camera = self.transforms.base_T_camera('wrist', self.capture_pose)
        np.testing.assert_allclose(scene['base_grasp'], base_camera @ self.camera_grasp)
        np.testing.assert_allclose(scene['base_tcp'], scene['base_flange'] @ self.transforms.flange_T_tcp)
        np.testing.assert_allclose(scene['base_tcp'], scene['base_grasp'] @ self.transforms.grasp_T_tcp)
        np.testing.assert_allclose(transform_points([[0, 0, 0]], scene['base_grasp'])[0],
                                   scene['base_grasp'][:3, 3])

    def test_changed_target_orientation_is_rejected(self):
        self.geometry['targets']['grasp']['flange_xyz_rpy'][5] += np.pi
        self.write('targets.json', self.geometry)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            load_scene(self.root)

    def test_wrong_camera_or_changed_snapshot_rejected(self):
        self.meta['device']['serial_number'] = 'wrong'
        self.write('capture/capture.json', self.meta)
        with self.assertRaisesRegex(ValueError, 'COLOR frame'):
            load_scene(self.root)
        self.record['calibration'] = {}
        self.write('capture_pose.json', self.record)
        with self.assertRaisesRegex(ValueError, 'snapshots differ'):
            load_scene(self.root)

    def test_feedback_comes_from_recorded_actual_endpoint_not_requested_pose(self):
        pose = self.geometry['targets']['grasp']['flange_xyz_rpy'].copy(); pose[0] += .001
        workflow = {'steps': [{'stage': 'move_grasp', 'result': {'ok': True,
            'endpoint': {'actual_model_pose_xyz_rpy': pose, 'position_error_m': .001}}}]}
        (self.root / 'run.json').write_text(json.dumps(workflow))
        scene = load_scene(self.root)
        self.assertEqual(scene['actual'][0]['pose'], pose)
        groups = build_geometry(scene)
        self.assertEqual(len(groups['actual']), 2)
        self.assertEqual(len(groups['cloud'][0].points), 3)
        self.assertTrue(groups['grasp'][0].has_triangles())

    def test_external_log_for_other_target_is_rejected(self):
        path = self.root / 'robot.jsonl'
        path.write_text(json.dumps({'event': 'plan', 'summary': {
            'target_pose_xyz_rpy': [.5, 0, .5, 0, 0, 0]}}) + '\n' +
            json.dumps({'event': 'endpoint', 'actual_model_pose_xyz_rpy': [.5, 0, .5, 0, 0, 0]}))
        with self.assertRaisesRegex(ValueError, 'different grasp'):
            load_scene(self.root, path)

    def test_offline_manifest_and_original_input_unchanged(self):
        before = (self.observation / 'targets.json').read_bytes()
        with redirect_stdout(io.StringIO()):
            summary = view(self.root, check_only=True)
        self.assertTrue(summary['no_hardware_access'])
        self.assertFalse(summary['guide_line_is_trajectory'])
        self.assertTrue((self.root / 'o3d/scene.json').is_file())
        self.assertEqual((self.observation / 'targets.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
