"""Backend routing and geometry contracts with mocked inference, no devices."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hardware.agent_geometry import grasp_plan, GraspSelectionError
from hardware.anygrasp_worker import filter_candidates
from hardware.config import DEFAULT_CONFIG, load_config
from hardware.interfaces import Pose6D
from hardware.transforms import load_transforms, pose_matrix
from services.anygrasp_fr3_service import AnyGraspFR3Service


class AnyGraspFR3Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.transforms = load_transforms(self.cfg)
        self.pose = Pose6D(.4, 0, .5, np.pi, 0, 0)
        self.record = dict(before=self.pose.values, after=self.pose.values,
                           serial=self.cfg.cameras.wrist_serial,
                           optical_frame=self.transforms.cameras['wrist']['optical_frame'],
                           calibration=self.transforms.summary())
        self.workcell = json.loads((Path(self.cfg.arm.workspace) / 'config/workcell.json').read_text())

    def test_sdk_outputs_keep_target_membership_and_height_checks(self):
        row = [.5, .05, .03, .02, *np.eye(3).ravel(), 0., 0., .2, -1.]
        rows = np.tile(row, (6, 1))
        rows[1, 1] = .09
        rows[2, 0] = 0
        rows[3, 13] = .5
        rows[4, 13] = .02
        rows[5, 15] = .004
        meta = {'color_intrinsics': dict(fx=10, fy=10, ppx=10, ppy=10, width=20, height=20)}
        request = dict(max_gripper_width=.085, roi=[0, 0, 20, 20],
                       base_T_camera=np.eye(4).tolist(), table_z_m=0)
        mask = np.zeros((20, 20), bool); mask[10, 10] = True
        filtered, targets, counts = filter_candidates(rows, meta, request, mask)
        self.assertEqual(list(counts.values()), [6, 4, 3, 2, 1])
        np.testing.assert_array_equal(filtered, rows[:1])
        self.assertEqual(targets.shape, (2, 17))
        request['grasp_table_clearance_check'] = False
        unfiltered, targets, counts = filter_candidates(rows, meta, request, mask)
        np.testing.assert_array_equal(unfiltered, rows[[0, 5]])
        self.assertEqual(list(counts.values()), [6, 4, 3, 2, 2])
        request['roi'] = [0, 0, 1, 1]
        empty, _, _ = filter_candidates(rows, meta, request, mask)
        self.assertEqual(empty.shape, (0, 17))

    def test_five_mm_clearance_accepts_seven_mm_but_ten_mm_rejects(self):
        self.workcell['grasp_table_clearance_check'] = True
        # Place the downward tool's padded tip exactly 7 mm above the table.
        flange_z = self.workcell['tool_length_m'] + self.workcell['tool_padding_m'] + .007
        flange = pose_matrix(Pose6D(.3, 0, flange_z, np.pi, 0, 0))
        camera_grasp = (np.linalg.inv(self.transforms.base_T_camera('wrist', self.pose)) @ flange @
                        self.transforms.flange_T_tcp @ np.linalg.inv(self.transforms.grasp_T_tcp))
        result = dict(backend='anygrasp', candidate_grasps_camera=[dict(index=0, score=.5,
            rotation_matrix=camera_grasp[:3, :3].tolist(), translation=camera_grasp[:3, 3].tolist())])
        plan = grasp_plan({'capture_geometry': self.record}, result, self.transforms,
                          {**self.workcell, 'table_clearance_m': .005})
        self.assertEqual(plan['backend'], 'anygrasp')
        self.assertIn('AnyGrasp', plan['choose_reason'])
        with self.assertRaises(GraspSelectionError):
            grasp_plan({'capture_geometry': self.record}, result, self.transforms,
                       {**self.workcell, 'table_clearance_m': .010})

    def test_factory_selects_calibrated_service_without_loading_sdk(self):
        from services.grasp_service import build_grasp_service
        from services.runtime_service_cache import clear_service_cache
        clear_service_cache()
        self.addCleanup(clear_service_cache)
        config = SimpleNamespace(hardware_config_path=str(DEFAULT_CONFIG))
        with patch('services.anygrasp_fr3_service.AnyGraspFR3Service') as constructor:
            self.assertIs(build_grasp_service(config), constructor.return_value)
            constructor.assert_called_once_with(str(DEFAULT_CONFIG))

    def test_service_child_interpreter_and_saved_result_match_selected_pose(self):
        import cv2
        from hardware.gsnet_o3d import load_scene
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.zeros((20, 20, 3), np.float32); points[:, :, 2] = .5
            np.save(root / 'points.npy', points)
            mask = np.ones((20, 20), bool); np.save(root / 'mask.npy', mask)
            cv2.imwrite(str(root / 'color.png'), np.zeros((20, 20, 3), np.uint8))
            flange = pose_matrix(Pose6D(.3, 0, .3, np.pi, 0, 0))
            matrix = (np.linalg.inv(self.transforms.base_T_camera('wrist', self.pose)) @ flange @
                      self.transforms.flange_T_tcp @ np.linalg.inv(self.transforms.grasp_T_tcp))
            rows = np.asarray([[.5, .05, .03, .02, *matrix[:3, :3].ravel(), *matrix[:3, 3], -1.]])
            def worker(command, **kwargs):
                self.assertEqual(command[:4], [self.cfg.perception.anygrasp_python, '-u', '-m', 'hardware.anygrasp_worker'])
                self.assertIn('compat_libs', kwargs['env']['LD_LIBRARY_PATH'])
                request = json.loads(Path(command[-1]).read_text())
                self.assertTrue(request['collision_detection'])
                self.assertFalse(request['grasp_table_clearance_check'])
                self.assertTrue(request['export_filtered_candidates'])
                np.testing.assert_array_equal(np.load(request['target_mask_path']), mask)
                self.assertIn(b'element vertex 400', (Path(request['capture_dir']) / 'scene.ply').read_bytes())
                output = Path(request['output_dir']); output.mkdir()
                np.save(output / 'filtered_candidates.npy', rows)
            service = AnyGraspFR3Service(DEFAULT_CONFIG)
            with patch('services.gsnet_service.PROJECT_ROOT', root), \
                    patch('services.anygrasp_fr3_service.subprocess.run', side_effect=worker):
                result = service.plan_grasp(color_path=root/'color.png', points_path=root/'points.npy',
                    intrinsics=dict(fx=10, fy=10, cx=10, cy=10), capture_geometry=self.record,
                    mask_npy_path=root/'mask.npy', retain_all_candidates=True,
                    output_json_path=root/'result.json')
            plan = grasp_plan({'capture_geometry': self.record}, result, self.transforms, self.workcell)
            service.save_geometry(plan)
            self.assertEqual(plan['backend'], 'anygrasp')
            saved = np.load(service.last_run/'anygrasp/grasps.npy')
            np.testing.assert_array_equal(saved, rows)
            self.assertEqual(json.loads((root/'result.json').read_text())['backend'], 'anygrasp')
            load_scene(service.last_run)


if __name__ == '__main__':
    unittest.main()
