from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def normalized_intrinsics(value: Dict[str, Any]) -> Dict[str, float]:
    cx = value.get("cx", value.get("ppx"))
    cy = value.get("cy", value.get("ppy"))
    required = {"fx": value.get("fx"), "fy": value.get("fy"), "cx": cx, "cy": cy}
    if any(item is None for item in required.values()):
        raise ValueError("intrinsics require fx, fy and cx/cy (or ppx/ppy)")
    result = {name: float(item) for name, item in required.items()}
    if result["fx"] <= 0 or result["fy"] <= 0:
        raise ValueError("intrinsic focal lengths must be positive")
    return result


def polygon_mask(shape: Tuple[int, int], points: Iterable[Iterable[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.asarray(list(points), dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("manual polygon must contain at least three [x, y] points")
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def load_mask(spec: Dict[str, Any], shape: Tuple[int, int], project_root: Path) -> np.ndarray:
    kind = spec.get("kind")
    if kind == "npy":
        mask = np.load(str(resolve_path(project_root, spec["path"])))
        mask = np.asarray(mask).astype(bool)
    elif kind == "image":
        mask = cv2.imread(str(resolve_path(project_root, spec["path"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(spec["path"])
        mask = mask > 0
    elif kind == "polygon":
        mask = polygon_mask(shape, spec["points_xy"])
    else:
        raise ValueError("mask kind must be one of: npy, image, polygon")
    if mask.shape != shape:
        raise ValueError("mask shape {} does not match depth shape {}".format(mask.shape, shape))
    return mask


def rgbd_points(
    depth_raw: np.ndarray,
    color_bgr: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict[str, float],
    depth_unit_m_per_raw_unit: float,
    min_depth_m: float,
    max_depth_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_raw.ndim != 2:
        raise ValueError("depth image must be single-channel")
    if color_bgr.shape[:2] != depth_raw.shape:
        raise ValueError("aligned RGB and depth must have identical image dimensions")
    if not (0 < depth_unit_m_per_raw_unit < 1):
        raise ValueError("depth_unit_m_per_raw_unit must be an explicit meter multiplier")

    depth_m = depth_raw.astype(np.float64) * float(depth_unit_m_per_raw_unit)
    valid = mask & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    rows, cols = np.nonzero(valid)
    z = depth_m[rows, cols]
    x = (cols.astype(np.float64) - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (rows.astype(np.float64) - intrinsics["cy"]) * z / intrinsics["fy"]
    points = np.column_stack((x, y, z))
    colors_rgb = color_bgr[rows, cols, ::-1].astype(np.float64) / 255.0
    return points, colors_rgb, valid


def deterministic_subsample(
    points: np.ndarray, colors: Optional[np.ndarray], max_points: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices], None if colors is None else colors[indices]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    return points @ transform[:3, :3].T + transform[:3, 3]


def euler_zyx_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def pose_mrad_to_transform(pose: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pose), dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("pose must be [x_m, y_m, z_m, roll, pitch, yaw]")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = euler_zyx_matrix(values[3], values[4], values[5])
    result[:3, 3] = values[:3]
    return result


def pose_mmrad_to_transform(pose: Iterable[float]) -> np.ndarray:
    """Build ``T_base_frame`` from xArm-style mm/radian XYZ-RPY values."""
    values = np.asarray(list(pose), dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("pose must be [x_mm, y_mm, z_mm, roll, pitch, yaw]")
    values = values.copy()
    values[:3] /= 1000.0
    return pose_mrad_to_transform(values)


def fit_pivot_tcp(flange_transforms: Iterable[np.ndarray]) -> Dict[str, Any]:
    """Fit flange-to-tip translation from poses sharing one physical pivot.

    For each taught pose ``i``, ``R_i p_flange_tcp + t_i = p_base_pivot``.
    The least-squares residual is a direct, interpretable repeatability check.
    This determines the TCP origin only; it cannot determine TCP orientation.
    """
    transforms = [np.asarray(value, dtype=np.float64) for value in flange_transforms]
    if len(transforms) < 4:
        raise ValueError("pivot calibration requires at least four flange poses")
    if any(value.shape != (4, 4) for value in transforms):
        raise ValueError("every flange transform must be 4x4")
    system = []
    target = []
    for transform in transforms:
        system.append(np.hstack((transform[:3, :3], -np.eye(3))))
        target.append(-transform[:3, 3])
    matrix = np.vstack(system)
    vector = np.concatenate(target)
    solution, _, rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    tcp = solution[:3]
    pivot = solution[3:]
    predicted = np.asarray(
        [transform[:3, :3] @ tcp + transform[:3, 3] for transform in transforms]
    )
    errors = np.linalg.norm(predicted - pivot[None, :], axis=1)
    relative_angles = []
    for left_index, left in enumerate(transforms):
        for right in transforms[left_index + 1 :]:
            relative = left[:3, :3].T @ right[:3, :3]
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            relative_angles.append(float(np.degrees(np.arccos(cosine))))
    return {
        "tcp_translation_flange_m": tcp,
        "pivot_point_base_m": pivot,
        "predicted_tip_points_base_m": predicted,
        "residuals_m": errors,
        "rms_residual_m": float(np.sqrt(np.mean(errors**2))),
        "max_residual_m": float(np.max(errors)),
        "linear_system_rank": int(rank),
        "linear_system_condition": float(
            np.inf if singular_values[-1] <= 0 else singular_values[0] / singular_values[-1]
        ),
        "maximum_pairwise_orientation_change_deg": max(relative_angles, default=0.0),
    }


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size_m: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Deterministically average points (and colours) in metric voxels."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if len(points) == 0:
        return points.copy(), None if colors is None else np.asarray(colors).copy()
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    reduced = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    reduced_colors = None
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.shape != points.shape:
            raise ValueError("colors must have the same Nx3 shape as points")
        reduced_colors = np.column_stack(
            [np.bincount(inverse, weights=colors[:, axis]) / counts for axis in range(3)]
        )
    return reduced, reduced_colors


def fit_tabletop_cuboid(
    observed_points_base: np.ndarray,
    dimensions_m: Iterable[float],
    support_plane_z_m: float,
) -> np.ndarray:
    """Estimate a gravity-aligned medicine-box frame from observed surfaces.

    ``dimensions_m`` is ``[long_side, short_side, height]``. The z origin is
    fixed by the measured support plane, while PCA supplies only the tabletop
    yaw. The sign of the long axis remains visually ambiguous and must be
    resolved from package artwork when orientation matters.
    """
    points = np.asarray(observed_points_base, dtype=np.float64)
    dimensions = np.asarray(list(dimensions_m), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        raise ValueError("at least 20 observed Nx3 points are required")
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("dimensions_m must contain three positive values")
    if dimensions[0] < dimensions[1]:
        raise ValueError("dimensions_m must be ordered [long_side, short_side, height]")
    xy = points[:, :2]
    center_xy = np.median(xy, axis=0)
    covariance = np.cov((xy - center_xy).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if long_axis[0] < 0 or (abs(long_axis[0]) < 1e-12 and long_axis[1] < 0):
        long_axis = -long_axis
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    rotation = np.array(
        [
            [long_axis[0], short_axis[0], 0.0],
            [long_axis[1], short_axis[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local_xy = (xy - center_xy) @ rotation[:2, :2]
    robust_center_local = 0.5 * (
        np.percentile(local_xy, 2.0, axis=0) + np.percentile(local_xy, 98.0, axis=0)
    )
    center_xy = center_xy + robust_center_local @ rotation[:2, :2].T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [center_xy[0], center_xy[1], float(support_plane_z_m) + dimensions[2] / 2.0]
    return transform


def sample_cuboid_surface(
    dimensions_m: Iterable[float],
    transform: np.ndarray,
    spacing_m: float = 0.003,
) -> np.ndarray:
    """Sample all six faces of a measured cuboid collision proxy."""
    dimensions = np.asarray(list(dimensions_m), dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("dimensions_m must contain three positive values")
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    half = dimensions / 2.0
    faces = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grids = [
            np.linspace(-half[axis], half[axis], max(2, int(np.ceil(dimensions[axis] / spacing_m)) + 1))
            for axis in free_axes
        ]
        first, second = np.meshgrid(grids[0], grids[1], indexing="ij")
        for sign in (-1.0, 1.0):
            face = np.zeros((first.size, 3), dtype=np.float64)
            face[:, fixed_axis] = sign * half[fixed_axis]
            face[:, free_axes[0]] = first.ravel()
            face[:, free_axes[1]] = second.ravel()
            faces.append(face)
    local = np.vstack(faces)
    return transform_points(local, transform)


def rotation_checks(transform: np.ndarray) -> Dict[str, float]:
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    return {
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_frobenius_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
    }


def cloud_stats(points: np.ndarray) -> Dict[str, Any]:
    if len(points) == 0:
        return {"point_count": 0, "centroid_m": None, "extent_m": None}
    return {
        "point_count": int(len(points)),
        "centroid_m": np.mean(points, axis=0).tolist(),
        "extent_m": np.ptp(points, axis=0).tolist(),
        "min_m": np.min(points, axis=0).tolist(),
        "max_m": np.max(points, axis=0).tolist(),
    }


def write_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    has_color = colors is not None
    if has_color:
        colors_u8 = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
        if colors_u8.shape != points.shape:
            raise ValueError("colors must be Nx3")
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write("element vertex {}\n".format(len(points)))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        if has_color:
            for point, color in zip(points, colors_u8):
                stream.write("{:.9g} {:.9g} {:.9g} {} {} {}\n".format(*point, *color))
        else:
            for point in points:
                stream.write("{:.9g} {:.9g} {:.9g}\n".format(*point))


def save_mask_review(
    path: Path,
    color_bgr: np.ndarray,
    target_mask: np.ndarray,
    base_mask: np.ndarray,
) -> None:
    overlay = color_bgr.astype(np.float32).copy()
    target_color = np.array([20, 220, 20], dtype=np.float32)
    base_color = np.array([220, 80, 20], dtype=np.float32)
    overlay[target_mask] = 0.55 * overlay[target_mask] + 0.45 * target_color
    overlay[base_mask] = 0.55 * overlay[base_mask] + 0.45 * base_color
    cv2.imwrite(str(path), np.clip(overlay, 0, 255).astype(np.uint8))


def save_cloud_overlay(
    path: Path,
    base_points: np.ndarray,
    target_points: np.ndarray,
    transformed_points: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_show = 10000
    base_points, _ = deterministic_subsample(base_points, None, max_show)
    target_points, _ = deterministic_subsample(target_points, None, max_show)
    transformed_points, _ = deterministic_subsample(transformed_points, None, max_show)
    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    axis = figure.add_subplot(121, projection="3d")
    axis.scatter(base_points[:, 0], base_points[:, 1], base_points[:, 2], s=0.5, c="#e45756", label="base_obj/support")
    axis.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], s=1.0, c="#4c78a8", label="target before")
    axis.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], s=1.0, c="#54a24b", label="target after")
    axis.set_xlabel("camera x [m]")
    axis.set_ylabel("camera y [m]")
    axis.set_zlabel("camera z [m]")
    axis.legend(loc="best")
    axis.set_title("AnyPlace transform in input camera frame")

    axis2 = figure.add_subplot(122)
    axis2.scatter(base_points[:, 0], base_points[:, 1], s=0.5, c="#e45756")
    axis2.scatter(target_points[:, 0], target_points[:, 1], s=1.0, c="#4c78a8")
    axis2.scatter(transformed_points[:, 0], transformed_points[:, 1], s=1.0, c="#54a24b")
    axis2.set_xlabel("camera x [m]")
    axis2.set_ylabel("camera y [m]")
    axis2.set_aspect("equal", adjustable="box")
    axis2.set_title("Camera x-y projection")
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
