import numpy as np

from anyplace_real.geometry import (
    pose_mrad_to_transform,
    rgbd_points,
    rotation_checks,
    transform_points,
)


def test_rgbd_backprojection_uses_explicit_meter_multiplier():
    depth = np.array([[1000]], dtype=np.uint16)
    color = np.zeros((1, 1, 3), dtype=np.uint8)
    points, _, valid = rgbd_points(
        depth,
        color,
        np.ones((1, 1), dtype=bool),
        {"fx": 2.0, "fy": 2.0, "cx": 0.0, "cy": 0.0},
        0.001,
        0.1,
        2.0,
    )
    assert valid[0, 0]
    assert np.allclose(points, [[0.0, 0.0, 1.0]])


def test_pose_transform_roundtrip_and_rotation_convention():
    transform = pose_mrad_to_transform([0.1, -0.2, 0.3, 0.2, -0.1, 0.4])
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.0, 0.2]])
    restored = transform_points(transform_points(points, transform), np.linalg.inv(transform))
    assert np.allclose(restored, points, atol=1e-12)
    checks = rotation_checks(transform)
    assert abs(checks["determinant"] - 1.0) < 1e-12
    assert checks["orthogonality_frobenius_error"] < 1e-12

