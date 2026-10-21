"""Camera deployment geometry, independent of simulator and hardware services."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


ARM_BASE_BODY = "robot0_link0"
LEFT_CAMERA = "left_oblique"
CAMERA_NAMES = ("inspection", "overhead", "overview", LEFT_CAMERA, "robot0_eye_in_hand")


def camera_quaternion(eye, target):
    """MuJoCo camera: +X right, +Y up, looking down local -Z; wxyz quaternion."""
    z = np.asarray(eye, float) - np.asarray(target, float)
    distance = np.linalg.norm(z)
    if not np.isfinite(distance) or distance < 1e-8:
        raise ValueError("camera position and look-at point must be finite and distinct")
    z /= distance
    up = np.array([0.0, 0.0, 1.0])
    if abs(z @ up) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
    return q[[3, 0, 1, 2]]


@dataclass(frozen=True)
class LeftCameraDeployment:
    # User-specified optical-center position, relative to the arm mounting base.
    position_base_m: tuple = (-0.25, 0.45, 0.89)
    lookat_base_m: tuple | None = None
    fovy_deg: float = 60.0

    def __post_init__(self):
        for name in ("position_base_m", "lookat_base_m"):
            value = getattr(self, name)
            if value is None and name == "lookat_base_m":
                continue
            vector = np.asarray(value, float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must contain three finite values in metres")
        if not np.isfinite(self.fovy_deg) or not 1 < self.fovy_deg < 179:
            raise ValueError("fovy_deg must be between 1 and 179 degrees")

    def resolve(self, world_from_base, work_area_world):
        """Freeze the external camera in world space after robot initialization."""
        transform = np.asarray(world_from_base, float)
        rotation, origin = transform[:3, :3], transform[:3, 3]
        eye = origin + rotation @ np.asarray(self.position_base_m)
        target = (np.asarray(work_area_world, float) if self.lookat_base_m is None
                  else origin + rotation @ np.asarray(self.lookat_base_m))
        quaternion = camera_quaternion(eye, target)
        return {
            "camera_id": LEFT_CAMERA,
            "mounting": "fixed_in_world_after_reset",
            "reference_frame": "arm_base",
            "reference_body": ARM_BASE_BODY,
            "position_base_m": list(self.position_base_m),
            "lookat_base_m": (rotation.T @ (target - origin)).tolist(),
            "lookat_source": ("work_area_center_assumption" if self.lookat_base_m is None
                              else "explicit_base_frame_point"),
            "fovy_deg": self.fovy_deg,
            "intrinsics_source": "configured_pinhole_fovy_not_measured_real_camera_calibration",
            "position_world_m": eye.tolist(),
            "lookat_world_m": target.tolist(),
            "quaternion_world_wxyz": quaternion.tolist(),
            "T_world_from_arm_base_at_setup": transform.tolist(),
        }
