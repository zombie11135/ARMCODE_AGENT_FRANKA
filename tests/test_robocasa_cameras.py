"""Deployment-frame checks independent of RoboCasa assets and OpenGL."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robocasa_occlusion.cameras import LeftCameraDeployment, camera_quaternion


def test_camera_position_uses_rotated_arm_base_and_optical_axis_faces_target():
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    base[:3, 3] = [2, -1, .7]
    setup = LeftCameraDeployment().resolve(base, [3, 0, .95])
    np.testing.assert_allclose(setup["position_world_m"], [1.55, -1.25, 1.59])
    q = np.asarray(setup["quaternion_world_wxyz"])
    optical_rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() @ np.diag([1, -1, -1])
    direction = np.array([3, 0, .95]) - setup["position_world_m"]
    np.testing.assert_allclose(optical_rotation[:, 2], direction / np.linalg.norm(direction))
    assert setup["reference_body"] == "robot0_link0"


def test_explicit_lookat_is_also_relative_to_arm_base():
    base = np.eye(4)
    base[:3, 3] = [1, 2, .7]
    setup = LeftCameraDeployment(lookat_base_m=(.4, 0, .2), fovy_deg=55).resolve(base, [99, 99, 99])
    np.testing.assert_allclose(setup["lookat_world_m"], [1.4, 2, .9])
    assert setup["lookat_source"] == "explicit_base_frame_point"
    assert setup["fovy_deg"] == 55


def test_invalid_deployment_and_degenerate_lookat_are_rejected():
    with pytest.raises(ValueError, match="three finite"):
        LeftCameraDeployment(position_base_m=(0, np.nan, 0))
    with pytest.raises(ValueError, match="degrees"):
        LeftCameraDeployment(fovy_deg=180)
    with pytest.raises(ValueError, match="distinct"):
        camera_quaternion([0, 0, 0], [0, 0, 0])
