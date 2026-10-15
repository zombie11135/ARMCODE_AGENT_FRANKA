"""Geometry regressions; all values synthetic except the archived calibration."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import yaml

from hardware.cli import main
from hardware.config import load_config
from hardware.interfaces import Pose6D
from hardware.runtime import build_hardware, require_task_geometry
from hardware.transforms import CalibratedTransforms, load_transforms, matrix_pose, pose_matrix, rigid_matrix


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "config/calibration/fr3_handeye.yaml"
SERIAL = "349622072103"
FRAME = "wrist_color_optical_frame"


class TransformTests(unittest.TestCase):
    def setUp(self):
        for name in ("serial.Serial", "subprocess.Popen"):
            guard = patch(name, side_effect=AssertionError("Offline test attempted device/process access"))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = yaml.safe_load(CALIBRATION.read_text())

    def write_calibration(self):
        path = self.directory / "calibration.yaml"
        path.write_text(yaml.safe_dump(self.data))
        return path

    def load_synthetic(self, *, tool=False):
        self.data["cameras"]["wrist"]["extrinsic"]["matrix"] = [
            [0, -1, 0, .1], [1, 0, 0, .2], [0, 0, 1, .3], [0, 0, 0, 1]]
        self.data["tool"]["T_flange_tcp"] = None
        self.data["grasp_model"]["T_grasp_tcp"] = None
        if tool:
            self.data["tool"]["T_flange_tcp"] = [
                [0, -1, 0, .1], [1, 0, 0, 0], [0, 0, 1, .2], [0, 0, 0, 1]]
            self.data["grasp_model"]["T_grasp_tcp"] = [
                [0, -1, 0, .1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return CalibratedTransforms(self.write_calibration())

    def test_archived_matrix_bound_without_inversion(self):
        source = json.loads((ROOT / "config/calibration/fr3_handeye_result.json").read_text())
        transform = load_transforms(load_config())
        np.testing.assert_array_equal(transform.cameras["wrist"]["matrix"], source["gripper_T_camera"])
        self.assertEqual(transform.cameras["wrist"]["serial"], SERIAL)
        hardware = build_hardware()
        try:
            self.assertIsInstance(hardware.transforms, CalibratedTransforms)
        finally:
            hardware.disconnect()

    def test_rotated_camera_and_flange_point_known_answer(self):
        transform = self.load_synthetic()
        point = transform.camera_point_to_base([1, 0, 0], "wrist",
                    Pose6D(1, 2, 3, 0, 0, math.pi/2), device_serial=SERIAL, optical_frame=FRAME)
        np.testing.assert_allclose(point, [-.2, 2.1, 3.3], atol=1e-12)

    def test_tool_offset_removed_in_flange_axes(self):
        transform = self.load_synthetic(tool=True)
        tcp = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        flange = transform.tcp_to_flange_pose(tcp)
        np.testing.assert_allclose(flange.values, [.9, 2, 2.8, 0, 0, 0], atol=1e-12)

    def test_full_grasp_chain_with_noncommuting_offsets(self):
        transform = self.load_synthetic(tool=True)
        grasp = np.eye(4)
        grasp[0, 3] = 1
        flange = transform.camera_grasp_to_flange_pose(grasp, "wrist", Pose6D(0, 0, 0, 0, 0, 0),
                            device_serial=SERIAL, optical_frame=FRAME)
        np.testing.assert_allclose(flange.values, [.1, 1.2, .1, 0, 0, math.pi/2], atol=1e-12)

    def test_pending_tcp_allows_points_but_not_grasp_targets(self):
        transform = self.load_synthetic()
        with self.assertRaisesRegex(RuntimeError, "T_flange_tcp"):
            transform.tcp_to_flange_pose(np.eye(4))
        with self.assertRaisesRegex(RuntimeError, "T_grasp_tcp"):
            transform.camera_grasp_to_flange_pose(np.eye(4), "wrist", Pose6D(0, 0, 0, 0, 0, 0),
                            device_serial=SERIAL, optical_frame=FRAME)

    def test_camera_and_optical_frame_mismatch(self):
        transform = self.load_synthetic()
        for serial, frame in (("346522072096", FRAME), (SERIAL, "wrist_depth_optical_frame")):
            with self.subTest(serial=serial, frame=frame), self.assertRaisesRegex(ValueError, "mismatch"):
                transform.camera_point_to_base([0, 0, 1], "wrist", Pose6D(0, 0, 0, 0, 0, 0),
                            device_serial=serial, optical_frame=frame)
        with self.assertRaisesRegex(RuntimeError, "no hand-eye"):
            transform.base_T_camera("global", Pose6D(0, 0, 0, 0, 0, 0))

    def test_profile_cannot_silently_retarget_camera(self):
        config = load_config()
        cameras = replace(config.cameras, wrist_serial="different-camera")
        with self.assertRaisesRegex(ValueError, "serials differ"):
            load_transforms(replace(config, cameras=cameras))

    def test_invalid_rigid_matrices(self):
        reflected = np.eye(4); reflected[0, 0] = -1
        bottom = np.eye(4); bottom[3, 0] = .1
        scaled = np.eye(4); scaled[0, 0] = 2
        invalid = np.eye(4); invalid[0, 3] = float("nan")
        for matrix in (reflected, bottom, scaled, invalid, np.eye(3)):
            with self.subTest(matrix=matrix.tolist()), self.assertRaises(ValueError):
                rigid_matrix(matrix, "test")

    def test_nonfinite_point_and_missing_capture_pose(self):
        transform = self.load_synthetic()
        with self.assertRaises(ValueError):
            transform.camera_point_to_base([0, 0, float("inf")], "wrist", Pose6D(0, 0, 0, 0, 0, 0),
                            device_serial=SERIAL, optical_frame=FRAME)
        with self.assertRaises(TypeError):
            transform.base_T_camera("wrist", None)

    def test_pending_frames_and_wrong_units_rejected(self):
        for key, value in (("status", "pending_frame_binding"), ("translation_unit", "mm"),
                           ("matrix_convention", "inverse")):
            self.data = yaml.safe_load(CALIBRATION.read_text())
            self.data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CalibratedTransforms(self.write_calibration())
        self.data = yaml.safe_load(CALIBRATION.read_text())
        self.data["cameras"]["wrist"]["extrinsic"]["parent_frame"] = "franka_EE_at_calibration"
        with self.assertRaisesRegex(ValueError, "fr3_link8"):
            CalibratedTransforms(self.write_calibration())

    def test_rpy_roundtrip_preserves_rotation_not_angle_branch(self):
        for angles in ((3.3, -.1, -1.57), (.2, 1.1, 2.4)):
            matrix = pose_matrix(Pose6D(.1, -.2, .3, *angles))
            np.testing.assert_allclose(pose_matrix(matrix_pose(matrix)), matrix, atol=1e-12)

    def test_relative_calibration_path_independent_of_working_directory(self):
        self.write_calibration()
        path = self.directory / "hardware.yaml"
        path.write_text("calibration_file: calibration.yaml\n")
        self.assertEqual(load_config(path).calibration_file, str(self.directory / "calibration.yaml"))

    def test_configured_tcp_enables_migrated_task_geometry(self):
        self.load_synthetic(tool=True)
        path = self.directory / "hardware.yaml"
        path.write_text("calibration_file: calibration.yaml\n")
        require_task_geometry(SimpleNamespace(hardware_config_path=str(path)))

    def test_cli_inspect_and_point_are_offline(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["calibration", "inspect"]), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["handeye_loaded"])
        self.assertTrue(result["tcp_configured"])
        self.assertTrue(result["grasp_alignment_configured"])
        self.assertEqual(result["grasp_alignment_status"], "nominal_preview")
        self.assertFalse(result["grasp_alignment_verified"])
        self.assertFalse(result["full_task_enabled"])
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["calibration", "point", "0", "0", "0",
                                  "--flange-pose", "0", "0", "0", "0", "0", "0",
                                  "--serial", SERIAL, "--frame", FRAME]), 0)
        result = json.loads(output.getvalue())
        np.testing.assert_allclose(result["point"], [.031286928370626625, -.07270577837570529, .019019428957029558])
        self.assertFalse(result["hardware_accessed"])

    def test_16cm_tcp_offset_rotates_with_flange(self):
        transform = load_transforms(load_config())
        # Here the flange Z axis points along base +X. A 16 cm tool requires
        # subtracting 16 cm in base X, not subtracting from base Z.
        target_tcp = [[0, 0, 1, .5], [0, 1, 0, .2], [-1, 0, 0, .3], [0, 0, 0, 1]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            flange = transform.tcp_to_flange_pose(target_tcp)
        np.testing.assert_allclose(flange.values[:3], [.34, .2, .3], atol=1e-12)

    def test_nominal_grasp_alignment_and_single_tool_offset(self):
        transform = load_transforms(load_config())
        capture = Pose6D(.394, .0084, .4713, -3.05, .0086, -1.3538)
        # Synthetic target with approach +baseX and jaw line +baseY.
        base_grasp = np.eye(4)
        base_grasp[:3, 3] = [.4, 0, .35]
        camera_grasp = np.linalg.inv(transform.base_T_camera('wrist', capture)) @ base_grasp
        flange = transform.camera_grasp_to_flange_pose(camera_grasp, 'wrist', capture,
                            device_serial=SERIAL, optical_frame=FRAME)
        matrix = pose_matrix(flange)
        np.testing.assert_allclose(matrix[:3, 3], [.24, 0, .35], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 2], [1, 0, 0], atol=1e-12)  # approach = flange Z
        np.testing.assert_allclose(matrix[:3, 0], [0, 1, 0], atol=1e-12)  # jaws = flange X
        np.testing.assert_allclose((matrix @ transform.flange_T_tcp)[:3, 3], [.4, 0, .35], atol=1e-12)
        require_task_geometry(SimpleNamespace(hardware_config_path=str(ROOT/'config/hardware/fr3.yaml')))


if __name__ == "__main__":
    unittest.main()
