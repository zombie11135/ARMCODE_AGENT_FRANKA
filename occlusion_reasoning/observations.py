"""Build perception evidence; do not invent occlusion edges or target probabilities.

Coordinate convention: optical camera X right, Y down, Z forward, metres.
Masks represent predicted visible pixels, not amodal silhouettes. Surface
centroids/extents are measurements of observed points, not object poses/sizes.
Detection IDs are frame-local; cross-frame association is a separate algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class InstanceEvidence:
    source_id: str
    mask: np.ndarray
    bbox_xyxy: list[float]
    detection_score: float
    category: str | None = None
    identity_scores: dict[str, float] | None = None


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera pose must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not (np.allclose(transform[3], [0, 0, 0, 1])
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("camera pose must be a rigid transform")
    return transform


def _surface_stats(points: np.ndarray) -> dict:
    if not len(points):
        return {"point_count": 0, "centroid_m": None, "aabb_min_m": None,
                "aabb_max_m": None, "extent_m": None}
    return {"point_count": len(points), "centroid_m": points.mean(axis=0).tolist(),
            "aabb_min_m": points.min(axis=0).tolist(),
            "aabb_max_m": points.max(axis=0).tolist(),
            "extent_m": np.ptp(points, axis=0).tolist()}


def build_observation(
    *,
    depth_m: np.ndarray,
    intrinsics: dict,
    instances: list[InstanceEvidence],
    frame_id: str,
    query: str,
    alignment_status: str,
    timestamp_s: float | None = None,
    camera_id: str | None = None,
    capture_group_id: str | None = None,
    timestamp_domain: str | None = None,
    reference_frame: str | None = None,
    T_reference_from_camera_optical: np.ndarray | None = None,
    duplicate_iou: float = 0.90,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Return JSON-compatible evidence and arrays for an offline packet.

    alignment_status is either recorded_aligned or assumed_aligned. An assumption
    must be explicit in the adapter; matching image dimensions does not prove
    registration. All finite positive depths are retained, without filtering a
    narrow depth band, selecting a connected component, or eroding thin edges.

    Highly overlapping masks are grouped as a *duplicate hypothesis*. Original
    proposals remain available. This is not proof of physical object identity.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or min(depth.shape) == 0:
        raise ValueError("depth must be an HxW optical-Z array")
    if not frame_id.strip() or not query.strip():
        raise ValueError("frame_id and task query are required")
    if alignment_status not in {"recorded_aligned", "assumed_aligned"}:
        raise ValueError("RGB-depth alignment must be recorded or explicitly assumed")
    if not 0 < duplicate_iou <= 1:
        raise ValueError("duplicate_iou must be in (0, 1]")
    intrinsic = {key: _finite(intrinsics[key], key) for key in ("fx", "fy", "cx", "cy")}
    if min(intrinsic["fx"], intrinsic["fy"]) <= 0:
        raise ValueError("focal lengths must be positive")
    transform = None
    if T_reference_from_camera_optical is not None:
        if not reference_frame:
            raise ValueError("reference_frame is required with a camera pose")
        transform = _validate_transform(T_reference_from_camera_optical)
    elif reference_frame is not None:
        raise ValueError("reference_frame requires a camera pose")
    timestamp = None if timestamp_s is None else _finite(timestamp_s, "timestamp_s")
    for name, value in (("camera_id", camera_id), ("capture_group_id", capture_group_id),
                        ("timestamp_domain", timestamp_domain)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string when supplied")

    depth_valid = np.isfinite(depth) & (depth > 0)
    arrays = {"depth_m": depth.copy(), "depth_valid": depth_valid,
              "depth_missing": ~depth_valid}
    validated = []
    seen_ids = set()
    for item in instances:
        if not item.source_id or item.source_id in seen_ids:
            raise ValueError("source instance IDs must be nonempty and unique within a frame")
        seen_ids.add(item.source_id)
        raw = np.asarray(item.mask)
        if raw.shape != depth.shape or not np.isfinite(raw).all():
            raise ValueError(f"invalid mask shape or values for {item.source_id}")
        mask = raw > 0
        box = [_finite(x, "bbox") for x in item.bbox_xyxy]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("bbox must be a nonempty XYXY rectangle")
        score = _finite(item.detection_score, "detection_score")
        if not 0 <= score <= 1:
            raise ValueError("detection_score must be in [0, 1]")
        identity_scores = {str(k): _finite(v, "identity_score")
                           for k, v in (item.identity_scores or {}).items()}
        validated.append((item, mask, box, score, identity_scores))

    # Complete-link grouping avoids merging a chain of only partly overlapping masks.
    groups: list[list[int]] = []
    for index in sorted(range(len(validated)), key=lambda i: (-validated[i][3], i)):
        mask = validated[index][1]
        def overlaps(other):
            other_mask = validated[other][1]
            union = np.count_nonzero(mask | other_mask)
            return union > 0 and np.count_nonzero(mask & other_mask) / union >= duplicate_iou
        group = next((g for g in groups if all(overlaps(other) for other in g)), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)

    all_proposal_union = np.zeros(depth.shape, bool)
    proposal_rows = []
    for index, (item, mask, box, score, identity_scores) in enumerate(validated):
        key = f"proposal_{index:03d}_raw_mask"
        arrays[key] = mask
        all_proposal_union |= mask
        proposal_rows.append({
            "source_id": item.source_id, "raw_mask_key": key,
            "bbox_xyxy": box, "detection_score": score, "category": item.category,
            "identity_scores": identity_scores,
            "identity_score_kind": "uncalibrated_similarity",
        })
    arrays["unassigned_valid_depth"] = depth_valid & ~all_proposal_union
    # Unassigned pixels include the table/background as well as missed objects.
    mask_count = np.zeros(depth.shape, dtype=np.int32)
    for group in groups:
        mask_count += validated[group[0]][1]
    arrays["overlapping_node_masks"] = mask_count > 1

    nodes, candidates = [], []
    for index, group in enumerate(groups):
        node_id = f"{frame_id}/instance_{index:03d}"
        representative = group[0]
        item, raw, box, score, identity_scores = validated[representative]
        support = raw & depth_valid
        rows, cols = np.nonzero(support)
        z = depth[rows, cols].astype(float)
        points = np.column_stack(((cols - intrinsic["cx"]) * z / intrinsic["fx"],
                                  (rows - intrinsic["cy"]) * z / intrinsic["fy"], z))
        support_key = f"instance_{index:03d}_depth_supported_mask"
        cloud_key = f"instance_{index:03d}_observed_points_camera"
        arrays[support_key], arrays[cloud_key] = support, points
        reference_cloud_key = None
        if transform is not None:
            reference_cloud_key = f"instance_{index:03d}_observed_points_reference"
            arrays[reference_cloud_key] = points @ transform[:3, :3].T + transform[:3, 3]
        raw_pixels = int(raw.sum())
        nodes.append({
            "node_id": node_id, "track_id": None, "kind": "observed_instance_hypothesis",
            "source_ids": [validated[i][0].source_id for i in group],
            "representative_proposal_index": representative,
            "grouping": "high_mask_overlap_hypothesis" if len(group) > 1 else "single_proposal",
            "bbox_xyxy": box, "category": item.category, "detection_score": score,
            "identity_status": "unverified",
            "raw_mask_key": proposal_rows[representative]["raw_mask_key"],
            "depth_supported_mask_key": support_key,
            "observed_points_camera_key": cloud_key,
            "observed_points_reference_key": reference_cloud_key,
            "observed_surface_camera": _surface_stats(points),
            "geometry_scope": "predicted_visible_surface_only_not_full_object_pose_or_extent",
            "quality": {
                "depth_filter": "finite_positive_only_no_uncertainty_filter",
                "raw_mask_pixels": raw_pixels, "depth_supported_pixels": int(support.sum()),
                "depth_missing_pixels": int((raw & ~depth_valid).sum()),
                "depth_coverage_ratio": float(support.sum() / raw_pixels) if raw_pixels else 0.0,
                "overlap_with_other_nodes_pixels": int((raw & (mask_count > 1)).sum()),
                "empty_mask": raw_pixels == 0,
            },
        })
        candidates.append({
            "kind": "observed_instance", "node_id": node_id, "probability": None,
            "target_similarity": identity_scores.get(query),
            "score_kind": "uncalibrated_similarity", "identity_confirmed": False,
        })
    candidates.extend([
        {"kind": "unobserved_or_missed", "probability": None, "spatial_region": None},
        {"kind": "target_absent", "probability": None},
    ])
    missing = ["workspace_bounds_and_support_surfaces", "cross_frame_instance_association",
               "occlusion_relation_inference", "unknown_space_mapping",
               "target_prior_and_observation_likelihood", "action_history"]
    if transform is None:
        missing.append("capture_time_camera_pose_in_shared_reference_frame")
    if timestamp is None:
        missing.append("capture_timestamp")
    if any(v is None for v in (camera_id, capture_group_id, timestamp_domain)):
        missing.append("camera_id_capture_group_and_shared_timestamp_domain_for_multiview")
    packet = {
        "schema": "occlusion_observation_v1",
        "stage": "perception_evidence_only",
        "frame": {
            "frame_id": frame_id, "timestamp_s": timestamp,
            "camera_id": camera_id, "capture_group_id": capture_group_id,
            "timestamp_domain": timestamp_domain,
            "width": depth.shape[1], "height": depth.shape[0],
            "intrinsics": intrinsic, "depth_units": "metres",
            "depth_definition": "optical_Z", "axes": "X_right_Y_down_Z_forward",
            "projection_model": "pinhole_requires_rectified_or_negligible_distortion",
            "rgb_depth_alignment": alignment_status, "reference_frame": reference_frame,
            "T_reference_from_camera_optical": None if transform is None else transform.tolist(),
        },
        "task": {"query": query, "target_dimensions_prior_m": None},
        "proposals": proposal_rows,
        "relation_graph": {"status": "edges_not_inferred", "nodes": nodes, "edges": []},
        "target_candidates": {"status": "probabilities_not_estimated", "hypotheses": candidates},
        "frame_masks": {
            "depth_valid": "depth_valid", "depth_missing": "depth_missing",
            "unassigned_valid_depth": "unassigned_valid_depth",
            "unassigned_semantics": "includes_background_and_missed_objects_not_a_hidden_region",
            "overlapping_node_masks": "overlapping_node_masks",
        },
        "duplicate_iou_threshold": duplicate_iou,
        "missing_for_full_reasoning": missing,
    }
    return packet, arrays
