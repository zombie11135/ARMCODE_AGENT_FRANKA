"""RGB-D-only perception and policy memory.

The renderer's object/segmentation ids are intentionally ignored here.  They
remain available only in evaluator state inside ``environment.py``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "target_confidence": {"source": "RGB color identity + visible pixel support", "unit": "ratio"},
    "target_visibility": {"source": "RGB target-color pixel support", "unit": "ratio"},
    "occlusion_score": {"source": "visible silhouette deficit versus known box geometry", "unit": "ratio"},
    "pose_confidence": {"source": "valid RGB-D support and point dispersion", "unit": "ratio"},
    "blocker_confidence": {"source": "RGB color identity + visible pixel support", "unit": "ratio"},
    "blocker_graspability": {"source": "RGB-D pose validity and workspace bounds", "unit": "boolean"},
    "blocker_target_relation": {"source": "2-D observed bbox overlap/search-region proximity", "unit": "ratio"},
    "previous_action_result": {"source": "executed primitive feedback", "unit": "{-1,0,1}"},
    "visual_probe_count": {"source": "policy counter", "unit": "count"},
    "manipulation_probe_count": {"source": "policy counter", "unit": "count"},
    "task_progress": {"source": "policy task-node state", "unit": "ratio"},
}


@dataclass
class Estimate:
    object_id: str
    color_class: str
    position_m: Optional[Tuple[float, float, float]]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    visible_pixels: int
    visibility: float
    identity_confidence: float
    pose_confidence: Optional[float]
    graspable: Optional[bool]
    observation_index: int
    camera_index: int
    stale_pose: bool = False
    stale_occlusion: bool = False
    field_validity: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyMemory:
    """Everything the high-level policy is allowed to inspect."""

    objects: Dict[str, Estimate] = field(default_factory=dict)
    observations: list = field(default_factory=list)
    visited_viewpoints: list = field(default_factory=list)
    gripper_holding: Optional[str] = None
    buffer_occupied: bool = False
    task_node: str = "grasp_target"
    last_action_result: Optional[float] = None

    def clone(self) -> "PolicyMemory":
        return deepcopy(self)

    def invalidate_after_change(self, *, pose: bool, occlusion: bool) -> None:
        for estimate in self.objects.values():
            if pose:
                estimate.stale_pose = True
                estimate.field_validity["position_m"] = False
            if occlusion:
                estimate.stale_occlusion = True
                estimate.field_validity["visibility"] = False

    def to_dict(self) -> dict:
        return {
            "objects": {key: value.to_dict() for key, value in self.objects.items()},
            "observations": deepcopy(self.observations),
            "visited_viewpoints": list(self.visited_viewpoints),
            "gripper_holding": self.gripper_holding,
            "buffer_occupied": self.buffer_occupied,
            "task_node": self.task_node,
            "last_action_result": self.last_action_result,
        }


def _mask_for(color_class: str, rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.astype(np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    if color_class == "target_red":
        return (red > 90) & (red > 1.80 * green) & (red > 1.80 * blue)
    if color_class == "distractor_blue":
        return (blue > 70) & (blue > 1.30 * red) & (blue > 1.18 * green)
    if color_class == "distractor_green":
        return (green > 65) & (green > 1.28 * red) & (green > 1.18 * blue)
    if color_class == "distractor_yellow":
        return (red > 90) & (green > 80) & (red < 1.35 * green) & (green < 1.20 * red) & (red > 2.0 * blue) & (green > 1.8 * blue)
    if color_class in {"daily_oat_box", "target_daily_oat_box"}:
        base_magenta = (red > 105) & (blue > 70) & (green < 0.62 * red) & (red > 1.22 * blue)
        package_pink = (red > 175) & (green > 105) & (blue > 115) & (red > 1.28 * green) & (red > 1.16 * blue)
        return base_magenta | package_pink
    if color_class == "daily_orange_can":
        return (red > 120) & (green > 70) & (red > 1.30 * green) & (blue < 0.68 * green)
    if color_class == "daily_purple_cup":
        return (red > 95) & (blue > 110) & (green < 0.68 * np.minimum(red, blue)) & (red > 0.72 * blue) & (red < 1.25 * blue)
    if color_class == "daily_cyan_bottle":
        return (green > 90) & (blue > 100) & (np.abs(green - blue) < 55) & (red < 0.55 * np.minimum(green, blue))
    raise KeyError(color_class)


def _world_points(mask: np.ndarray, depth_buffer: np.ndarray, view: Iterable[float], projection: Iterable[float]) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float64)
    # PyBullet matrices are OpenGL column-major.  Unproject through inv(P*V).
    view_matrix = np.asarray(view, dtype=np.float64).reshape((4, 4), order="F")
    projection_matrix = np.asarray(projection, dtype=np.float64).reshape((4, 4), order="F")
    inverse = np.linalg.inv(projection_matrix @ view_matrix)
    height, width = depth_buffer.shape
    x_ndc = 2.0 * (cols.astype(np.float64) + 0.5) / width - 1.0
    y_ndc = 1.0 - 2.0 * (rows.astype(np.float64) + 0.5) / height
    z_ndc = 2.0 * depth_buffer[rows, cols].astype(np.float64) - 1.0
    clip = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)], axis=0)
    world_h = inverse @ clip
    world = (world_h[:3] / world_h[3:4]).T
    finite = np.all(np.isfinite(world), axis=1)
    return world[finite]


def perceive_rgbd(
    rgb: np.ndarray,
    depth_buffer: np.ndarray,
    view_matrix: Iterable[float],
    projection_matrix: Iterable[float],
    camera_index: int,
    observation_index: int,
    known_half_heights: Dict[str, float],
) -> Dict[str, Estimate]:
    """Return object estimates from actual RGB-D pixels, never render ids."""

    estimates: Dict[str, Estimate] = {}
    for color_class, half_height in known_half_heights.items():
        mask = _mask_for(color_class, rgb)
        count = int(mask.sum())
        if count < 8:
            continue
        rows, cols = np.nonzero(mask)
        points = _world_points(mask, depth_buffer, view_matrix, projection_matrix)
        points = points[(points[:, 2] > -0.02) & (points[:, 2] < 0.35)]
        if len(points) < 6:
            continue
        # The visible colored surface is normally the top surface; known simple
        # geometry converts it to the box center without body-id access.
        top = float(np.percentile(points[:, 2], 90))
        top_points = points[points[:, 2] >= top - 0.004]
        if len(top_points) < 6:
            top_points = points
        center = (
            float(np.median(top_points[:, 0])),
            # TinyRenderer pixel-center calibration for this fixed camera model.
            float(np.median(top_points[:, 1])) + 0.002,
            max(float(half_height), top - float(half_height)),
        )
        dispersion = float(np.linalg.norm(np.std(top_points[:, :2], axis=0)))
        pose_confidence = float(np.clip((count - 6) / 90.0, 0.0, 1.0) * np.clip(0.07 / max(dispersion, 1e-4), 0.0, 1.0))
        visibility = float(np.clip(count / 180.0, 0.0, 1.0))
        identity_confidence = float(np.clip(0.45 + count / 260.0, 0.0, 0.98))
        object_id = color_class
        estimates[object_id] = Estimate(
            object_id=object_id,
            color_class=color_class,
            position_m=center,
            bbox_xyxy=(int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())),
            visible_pixels=count,
            visibility=visibility,
            identity_confidence=identity_confidence,
            pose_confidence=pose_confidence,
            graspable=bool(pose_confidence >= 0.25 and -0.05 <= center[0] <= 0.48 and abs(center[1]) <= 0.22),
            observation_index=observation_index,
            camera_index=camera_index,
            stale_pose=False,
            stale_occlusion=False,
            field_validity={
                "position_m": True,
                "bbox_xyxy": True,
                "visibility": True,
                "identity_confidence": True,
                "pose_confidence": True,
                "graspable": True,
            },
        )
    return estimates
