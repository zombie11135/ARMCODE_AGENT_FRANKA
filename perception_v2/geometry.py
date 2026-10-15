from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import yaml


def as_binary_mask(mask: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    result = np.asarray(mask)
    if result.ndim == 3:
        result = result[..., 0]
    result = result > 0
    if shape is not None and result.shape != shape:
        result = cv2.resize(
            result.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return result


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = as_binary_mask(left)
    right = as_binary_mask(right, left.shape)
    union = np.count_nonzero(left | right)
    return 0.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def load_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError("failed to load depth image: {}".format(path))
    depth = np.asarray(depth, dtype=np.float32)
    scale = float(depth_scale)
    if scale <= 0:
        raise ValueError("depth_scale must be positive")
    # Existing project convention accepts either meters/raw-unit (<1) or a divisor (>=1).
    return depth * scale if scale < 1.0 else depth / scale


def clean_visible_surface_mask(
    raw_mask: np.ndarray,
    depth_m: np.ndarray,
    min_depth_m: float = 0.10,
    max_depth_m: float = 1.50,
    minimum_band_m: float = 0.018,
    mad_multiplier: float = 4.0,
    close_kernel_px: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = as_binary_mask(raw_mask, depth_m.shape)
    valid_depth = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    raw_valid = raw & valid_depth
    raw_area = int(np.count_nonzero(raw))
    valid_count = int(np.count_nonzero(raw_valid))
    metrics: Dict[str, Any] = {
        "raw_mask_pixels": raw_area,
        "valid_depth_pixels": valid_count,
        "valid_depth_ratio": 0.0 if raw_area == 0 else float(valid_count / raw_area),
        "depth_median_m": None,
        "depth_mad_m": None,
        "depth_band_m": None,
        "largest_component_ratio": 0.0,
        "clean_mask_pixels": 0,
    }
    if valid_count == 0:
        return np.zeros_like(raw), metrics

    depths = depth_m[raw_valid]
    median = float(np.median(depths))
    mad = float(np.median(np.abs(depths - median)))
    # A robust band suppresses table/background leaks without assuming a fronto-parallel box.
    band = max(float(minimum_band_m), float(mad_multiplier) * 1.4826 * mad)
    band = min(band, 0.08)
    depth_consistent = raw_valid & (np.abs(depth_m - median) <= band)

    if close_kernel_px > 1:
        kernel_size = int(close_kernel_px) | 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        depth_consistent = cv2.morphologyEx(
            depth_consistent.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        depth_consistent &= raw_valid

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        depth_consistent.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        clean = depth_consistent
        largest = int(np.count_nonzero(clean))
    else:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean = labels == component
        largest = int(stats[component, cv2.CC_STAT_AREA])

    metrics.update(
        {
            "depth_median_m": median,
            "depth_mad_m": mad,
            "depth_band_m": band,
            "largest_component_ratio": 0.0 if valid_count == 0 else float(largest / valid_count),
            "clean_mask_pixels": int(np.count_nonzero(clean)),
        }
    )
    return clean, metrics


def make_grasp_mask(visible_mask: np.ndarray, erosion_px: int = 2) -> np.ndarray:
    visible = as_binary_mask(visible_mask)
    if erosion_px <= 0:
        return visible.copy()
    size = 2 * int(erosion_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    eroded = cv2.erode(visible.astype(np.uint8), kernel).astype(bool)
    # Do not erase small but otherwise valid objects completely.
    return eroded if np.count_nonzero(eroded) >= 64 else visible.copy()


def masked_point_cloud(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    mask = as_binary_mask(mask, depth_m.shape)
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)
    z = depth_m[rows, cols].astype(np.float64)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    colors = color_bgr[rows, cols, ::-1].astype(np.float64) / 255.0
    return points, colors


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
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_zyx_matrix(values[3], values[4], values[5])
    transform[:3, 3] = values[:3]
    return transform


def load_camera_to_ee(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rotation = np.asarray(config["rotation"], dtype=np.float64)
    translation = np.asarray(config["translation"], dtype=np.float64).reshape(3)
    if config.get("unit", "meter") != "meter":
        raise ValueError("only meter hand-eye calibration is supported")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    frame_type = config.get("frame_type", "camera_to_ee")
    if frame_type in {"camera_to_ee", "cam2ee", "T_cam2ee"}:
        return transform
    if frame_type in {"ee_to_camera", "ee2cam", "T_ee2cam"}:
        return np.linalg.inv(transform)
    raise ValueError("unknown hand-eye frame_type: {}".format(frame_type))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def cloud_stats(points: np.ndarray) -> Dict[str, Any]:
    points = np.asarray(points)
    if len(points) == 0:
        return {"point_count": 0, "centroid_m": None, "extent_m": None}
    return {
        "point_count": int(len(points)),
        "centroid_m": np.mean(points, axis=0).tolist(),
        "extent_m": np.ptp(points, axis=0).tolist(),
        "min_m": np.min(points, axis=0).tolist(),
        "max_m": np.max(points, axis=0).tolist(),
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    colors_u8 = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write("element vertex {}\n".format(len(points)))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors_u8):
            stream.write("{:.9g} {:.9g} {:.9g} {} {} {}\n".format(*point, *color))


def build_instance_geometry(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    raw_mask: np.ndarray,
    intrinsics: Dict[str, float],
    camera_to_base: Optional[np.ndarray] = None,
    erosion_px: int = 2,
) -> Dict[str, Any]:
    raw = as_binary_mask(raw_mask, depth_m.shape)
    visible, quality = clean_visible_surface_mask(raw, depth_m)
    grasp = make_grasp_mask(visible, erosion_px=erosion_px)
    points_camera, colors = masked_point_cloud(color_bgr, depth_m, grasp, intrinsics)
    points_base = None
    if camera_to_base is not None:
        points_base = transform_points(points_camera, camera_to_base)
    quality["grasp_mask_pixels"] = int(np.count_nonzero(grasp))
    quality["point_count"] = int(len(points_camera))
    return {
        "semantic_mask_raw": raw,
        "visible_surface_mask": visible,
        "grasp_mask": grasp,
        "points_camera": points_camera,
        "points_base": points_base,
        "colors": colors,
        "quality": quality,
        "camera_cloud_stats": cloud_stats(points_camera),
        "base_cloud_stats": None if points_base is None else cloud_stats(points_base),
    }
