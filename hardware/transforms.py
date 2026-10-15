"""Offline FR3 geometry. Never opens devices, queries poses, or sends motion.

The caller supplies the flange pose at capture, not the pose at execution.
Column vectors, metres; Pose6D uses fixed-axis XYZ RPY (Rz @ Ry @ Rx).
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .interfaces import PendingTransforms, Pose6D


def rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: expected a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{name}: invalid rigid transform (rotation/bottom row)")
    return matrix.copy()


def pose_matrix(pose):
    if not isinstance(pose, Pose6D):
        raise TypeError("Expected the fr3_link8 Pose6D at image capture")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", pose.values[3:]).as_matrix()
    matrix[:3, 3] = pose.values[:3]
    return matrix


def matrix_pose(value):
    matrix = rigid_matrix(value, "base_T_flange")
    angles = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    return Pose6D(*matrix[:3, 3].tolist(), *angles.tolist())


class CalibratedTransforms:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        data = yaml.safe_load(self.path.read_text())
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Expected calibration schema_version: 1")
        if (data.get("translation_unit") != "m"
                or data.get("matrix_convention") != "p_parent = T_parent_child @ p_child"):
            raise ValueError("Calibration requires metres and parent <- child column-vector convention")
        if data.get("status") not in ("handeye_ready_tcp_pending", "handeye_ready_alignment_pending",
                                      "geometry_preview_ready", "ready"):
            raise ValueError("Calibration is still pending frame binding")
        self.cameras = {}
        for name, camera in data.get("cameras", {}).items():
            extrinsic = camera.get("extrinsic", {})
            if extrinsic.get("matrix") is None:
                continue  # An uncalibrated global camera is not an identity transform.
            if (camera.get("mounting") != "eye_in_hand"
                    or extrinsic.get("parent_frame") != "fr3_link8"):
                raise ValueError(f"{name}: expected an eye-in-hand fr3_link8 extrinsic")
            if (not isinstance(camera.get("serial"), str) or not camera["serial"]
                    or not isinstance(extrinsic.get("child_frame"), str) or not extrinsic["child_frame"]):
                raise ValueError(f"{name}: serial and optical frame are required")
            if camera.get("image_stream") != "color" or camera.get("depth_aligned_to") != "color":
                raise ValueError(f"{name}: expected color optical coordinates and depth aligned to color")
            self.cameras[name] = dict(serial=camera["serial"], optical_frame=extrinsic["child_frame"],
                                      matrix=rigid_matrix(extrinsic["matrix"], f"{name}.extrinsic"))
        if not self.cameras:
            raise ValueError("No calibrated camera found")
        tool = data.get("tool", {})
        if tool.get("flange_frame") != "fr3_link8":
            raise ValueError("tool.flange_frame must be fr3_link8")
        self.flange_T_tcp = self._optional_matrix(tool.get("T_flange_tcp"), "T_flange_tcp")
        self.tcp_translation_is_approximate = tool.get("translation_is_approximate", False)
        model = data.get("grasp_model", {})
        self.grasp_T_tcp = self._optional_matrix(model.get("T_grasp_tcp"), "T_grasp_tcp")
        self.alignment_status = model.get("alignment_status", "unspecified")
        self.alignment_verified = model.get("alignment_verified", False) is True

    @staticmethod
    def _optional_matrix(value, name):
        return None if value is None else rigid_matrix(value, name)

    def _camera(self, name):
        if name not in self.cameras:
            raise RuntimeError(f"Camera {name!r} has no hand-eye calibration")
        return self.cameras[name]

    def _check_capture(self, camera, device_serial, optical_frame):
        cfg = self._camera(camera)
        if device_serial != cfg["serial"]:
            raise ValueError(f"Camera serial mismatch: expected {cfg['serial']}, got {device_serial}")
        if optical_frame != cfg["optical_frame"]:
            raise ValueError(f"Camera optical frame mismatch: expected {cfg['optical_frame']}")

    def base_T_camera(self, camera, current_flange):
        """current_flange MUST be the pose at image capture; no live lookup."""
        return pose_matrix(current_flange) @ self._camera(camera)["matrix"]

    def camera_point_to_base(self, point, camera, current_flange, *, device_serial, optical_frame):
        self._check_capture(camera, device_serial, optical_frame)
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Camera point must contain three finite metre values")
        transform = self.base_T_camera(camera, current_flange)
        return (transform[:3, :3] @ point + transform[:3, 3]).tolist()

    def require_ready(self):
        """Check matrix availability for computation, not real-world accuracy or execution readiness."""
        missing = [name for name, matrix in (("T_flange_tcp", self.flange_T_tcp),
                                            ("T_grasp_tcp", self.grasp_T_tcp)) if matrix is None]
        if missing:
            raise RuntimeError("手眼标定已加载；抓取还缺少 " + ", ".join(missing)
                               + "。请配置实际抓取 TCP 及模型坐标轴与夹爪开合方向的对应关系。")

    def tcp_to_flange_pose(self, base_tcp):
        if self.flange_T_tcp is None:
            raise RuntimeError("T_flange_tcp 尚未配置")
        return matrix_pose(rigid_matrix(base_tcp, "base_T_tcp") @ np.linalg.inv(self.flange_T_tcp))

    def camera_grasp_to_flange_pose(self, camera_grasp, camera, current_flange,
                                    *, device_serial, optical_frame):
        self._check_capture(camera, device_serial, optical_frame)
        self.require_ready()
        base_tcp = (self.base_T_camera(camera, current_flange)
                    @ rigid_matrix(camera_grasp, "camera_T_grasp") @ self.grasp_T_tcp)
        return self.tcp_to_flange_pose(base_tcp)

    def summary(self):
        return dict(calibration_file=str(self.path), handeye_loaded=True,
                    cameras={name: dict(serial=cfg["serial"], optical_frame=cfg["optical_frame"],
                                        T_flange_camera=cfg["matrix"].tolist())
                             for name, cfg in self.cameras.items()},
                    tcp_configured=self.flange_T_tcp is not None,
                    T_flange_tcp=None if self.flange_T_tcp is None else self.flange_T_tcp.tolist(),
                    tcp_translation_is_approximate=self.tcp_translation_is_approximate,
                    grasp_alignment_configured=self.grasp_T_tcp is not None,
                    T_grasp_tcp=None if self.grasp_T_tcp is None else self.grasp_T_tcp.tolist(),
                    grasp_alignment_status=self.alignment_status,
                    grasp_alignment_verified=self.alignment_verified,
                    full_task_enabled=False)


def load_transforms(config):
    if config.calibration_file is None:
        return PendingTransforms()
    transforms = CalibratedTransforms(config.calibration_file)
    for role, camera in transforms.cameras.items():
        if camera["serial"] != getattr(config.cameras, f"{role}_serial", None):
            raise ValueError(f"{role}: hardware profile and calibration serials differ")
    return transforms
