"""Single-frame boundary evidence from visible masks and optical-Z depth.

This is a geometric baseline, not an amodal segmentation model. A local depth
order does not prove occlusion. ``possible_occludes`` additionally requires a
small image gap and intrusion into a convex completion of the farther mask.
That completion is an explicit prior appropriate to box-like silhouettes; a
concave object can violate it. Scores are heuristic, never probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import warnings

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RelationConfig:
    neighbor_radius_px: float = 6.0
    contact_distance_px: float = 2.0
    depth_patch_radius_px: int = 2
    min_depth_delta_m: float = 0.008
    min_valid_pairs: int = 8
    min_valid_fraction: float = 0.5
    min_order_fraction: float = 0.8
    max_mask_overlap_fraction: float = 0.05
    min_completion_overlap_pixels: int = 8
    min_completion_overlap_fraction: float = 0.1
    convex_completion_prior: bool = True

    def validate(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float, bool)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.contact_distance_px <= self.neighbor_radius_px:
            raise ValueError("contact distance must be positive and <= neighbor radius")
        if not self.min_depth_delta_m > 0:
            raise ValueError("depth threshold must be positive")
        if type(self.depth_patch_radius_px) is not int or not 0 <= self.depth_patch_radius_px <= 10:
            raise ValueError("depth patch radius must be an integer from 0 to 10")
        for name in ("min_valid_pairs", "min_completion_overlap_pixels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.convex_completion_prior) is not bool:
            raise ValueError("convex_completion_prior must be a boolean")
        for name in ("min_valid_fraction", "max_mask_overlap_fraction",
                     "min_completion_overlap_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not .5 < self.min_order_fraction <= 1:
            raise ValueError("min_order_fraction must exceed one half")


def _boundary(mask):
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return np.argwhere(mask & ~interior)  # row, col


def _hull(mask, boundary):
    hull_mask = np.zeros(mask.shape, np.uint8)
    if len(boundary) >= 3:
        hull = cv2.convexHull(boundary[:, ::-1].astype(np.int32))
        cv2.fillConvexPoly(hull_mask, hull, 1)
    return (hull_mask > 0) | mask


def _patch_depth(boundary, mask, depth, valid, radius):
    """Only measured pixels of this unambiguously assigned instance can vote."""
    samples = []
    rows, cols = boundary.T
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = rows + dy, cols + dx
            inside = (y >= 0) & (x >= 0) & (y < depth.shape[0]) & (x < depth.shape[1])
            y, x = np.clip(y, 0, depth.shape[0] - 1), np.clip(x, 0, depth.shape[1] - 1)
            samples.append(np.where(inside & mask[y, x] & valid[y, x], depth[y, x], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN patches stay unknown
        return np.nanmedian(np.asarray(samples), axis=0)


def _nearest_pairs(left, right, radius):
    """Match facing contours without jumping across the back object's thin rim.

    Use mutual nearest *distances*, rather than equal nearest-point indices, so
    pixel-grid ties do not arbitrarily discard an otherwise valid interface.
    """
    pairs, nearest_distances = [], []
    for first, second, reverse in ((left, right, False), (right, left, True)):
        distance, index = cKDTree(second).query(first, distance_upper_bound=radius)
        nearest_distances.append(distance)
        active = np.flatnonzero(np.isfinite(distance))
        columns = (index[active], active) if reverse else (active, index[active])
        pairs.append(np.column_stack(columns))
    candidates = np.unique(np.vstack(pairs).astype(int), axis=0)
    distance = np.linalg.norm(left[candidates[:, 0]] - right[candidates[:, 1]], axis=1)
    mutual = ((distance <= nearest_distances[0][candidates[:, 0]] + 1e-6)
              & (distance <= nearest_distances[1][candidates[:, 1]] + 1e-6))
    return candidates[mutual]


def _unblocked_pairs(a, b, other_mask, radius):
    """Do not bridge across a third detected instance at a narrow junction."""
    clear = np.ones(len(a), bool)
    for weight in np.linspace(0, 1, int(np.ceil(radius)) + 2):
        xy = np.rint(a * (1 - weight) + b * weight).astype(int)
        clear &= ~other_mask[xy[:, 0], xy[:, 1]]
    return clear


def _pair_evidence(index, a, b, mask_count, config):
    raw_overlap = int((a["mask"] & b["mask"]).sum())
    overlap_fraction = raw_overlap / max(1, min(int(a["mask"].sum()), int(b["mask"].sum())))
    result = {
        "pair_id": f"pair_{index:03d}", "a": a["id"], "b": b["id"],
        "status": "no_near_boundary_evidence",
        "mask_overlap": {"pixels": raw_overlap, "fraction_of_smaller_mask": overlap_fraction},
        "boundary": {"min_distance_px": None, "near_pair_count": 0,
                     "third_instance_blocked_pairs": 0, "contact_fraction": 0.0,
                     "matching": "mutual_nearest_boundary_distance"},
        "depth": {"order": "unknown", "valid_pair_count": 0, "valid_fraction": 0.0,
                  "median_b_minus_a_m": None, "a_closer_fraction": None,
                  "b_closer_fraction": None, "tied_fraction": None},
        "completion": None, "samples": [], "edges": [],
    }
    if not len(a["boundary"]) or not len(b["boundary"]):
        result["status"] = "empty_visible_mask"
        return result
    result["boundary"]["min_distance_px"] = float(
        np.min(cKDTree(b["boundary"]).query(a["boundary"])[0]))
    if overlap_fraction > config.max_mask_overlap_fraction:
        result["status"] = "mask_assignment_conflict"
        return result
    pairs = _nearest_pairs(a["boundary"], b["boundary"], config.neighbor_radius_px)
    if not len(pairs):
        return result
    ya, yb = a["boundary"][pairs[:, 0]], b["boundary"][pairs[:, 1]]
    others = mask_count - a["mask"].astype(int) - b["mask"].astype(int) > 0
    clear = _unblocked_pairs(ya, yb, others, config.neighbor_radius_px)
    result["boundary"]["third_instance_blocked_pairs"] = int((~clear).sum())
    pairs, ya, yb = pairs[clear], ya[clear], yb[clear]
    if not len(pairs):
        result["status"] = "boundary_blocked_by_third_instance"
        return result
    distance = np.linalg.norm(ya - yb, axis=1)
    za, zb = a["depth"][pairs[:, 0]], b["depth"][pairs[:, 1]]
    valid = np.isfinite(za) & np.isfinite(zb)
    count = int(valid.sum())
    result["boundary"].update(
        near_pair_count=len(pairs), median_distance_px=float(np.median(distance)),
        contact_fraction=float(np.mean(distance <= config.contact_distance_px)),
    )
    result["depth"].update(valid_pair_count=count, valid_fraction=float(valid.mean()))
    # Diagnostics include missing-depth endpoints explicitly, not fabricated depths.
    for k in np.linspace(0, len(pairs) - 1, min(48, len(pairs)), dtype=int):
        result["samples"].append({
            "a_xy": ya[k, ::-1].tolist(), "b_xy": yb[k, ::-1].tolist(),
            "a_depth_m": float(za[k]) if np.isfinite(za[k]) else None,
            "b_depth_m": float(zb[k]) if np.isfinite(zb[k]) else None,
        })
    if count:
        delta = zb[valid] - za[valid]
        positive = float(np.mean(delta > config.min_depth_delta_m))
        negative = float(np.mean(delta < -config.min_depth_delta_m))
        result["depth"].update(
            median_b_minus_a_m=float(np.median(delta)),
            mad_b_minus_a_m=float(np.median(np.abs(delta - np.median(delta)))),
            a_closer_fraction=positive, b_closer_fraction=negative,
            tied_fraction=float(np.mean(np.abs(delta) <= config.min_depth_delta_m)),
        )
    if count < config.min_valid_pairs or valid.mean() < config.min_valid_fraction:
        result["status"] = "insufficient_local_depth"
        return result
    if max(positive, negative) < config.min_order_fraction:
        result["status"] = "no_resolved_depth_step" if result["depth"]["tied_fraction"] >= config.min_order_fraction else "mixed_local_depth_order"
        return result

    front, back = (a, b) if positive > negative else (b, a)
    result["depth"]["order"] = "a_closer" if positive > negative else "b_closer"
    depth_score = float(max(positive, negative) * valid.mean())
    common = {
        "source": front["id"], "target": back["id"], "pair_id": result["pair_id"],
        "evidence_score": depth_score, "score_kind": "uncalibrated_heuristic",
        "calibrated_probability": None, "confirmed": False,
    }
    result["edges"].append({**common, "relation": "locally_in_front_of",
                            "scope": "sampled_boundary_only"})
    result["status"] = "local_depth_order_only"
    if config.convex_completion_prior:
        missing = back["hull"] & ~back["mask"]
        exclusive_front = front["mask"] & (mask_count == 1)
        completion_overlap = int((missing & exclusive_front).sum())
        completion_fraction = completion_overlap / max(1, int(missing.sum()))
        result["completion"] = {
            "model": "convex_projected_silhouette_prior", "back_instance": back["id"],
            "missing_silhouette_pixels": int(missing.sum()),
            "front_overlap_pixels": completion_overlap,
            "front_overlap_fraction": completion_fraction,
            "observed_amodal_mask": False,
        }
        if (completion_overlap >= config.min_completion_overlap_pixels
                and completion_fraction >= config.min_completion_overlap_fraction
                and result["boundary"]["contact_fraction"] >= config.min_order_fraction):
            result["status"] = "possible_occlusion"
            result["edges"].append({
                **common, "relation": "possible_occludes", "scope": "current_camera",
                "evidence_score": float(depth_score * result["boundary"]["contact_fraction"]),
                "assumption": "farther_object_has_approximately_convex_projected_silhouette",
            })
    return result


def extract_relations(packet: dict, arrays, config: RelationConfig | None = None) -> dict:
    """Use only frame geometry and raw masks; identities and task labels are unused."""
    config = config or RelationConfig()
    config.validate()
    frame = packet["frame"]
    if packet.get("schema") != "occlusion_observation_v1":
        raise ValueError("unsupported observation schema")
    if frame.get("depth_units") != "metres" or frame.get("depth_definition") != "optical_Z":
        raise ValueError("relations require optical-Z depth in metres")
    if frame.get("rgb_depth_alignment") not in {"recorded_aligned", "assumed_aligned"}:
        raise ValueError("aligned RGB-D is required")
    depth = np.asarray(arrays["depth_m"], dtype=float)
    shape = (frame["height"], frame["width"])
    if depth.shape != shape:
        raise ValueError("frame/depth dimensions differ")
    depth_valid = np.asarray(arrays["depth_valid"], bool)
    if depth_valid.shape != shape:
        raise ValueError("depth validity dimensions differ")
    valid = depth_valid & np.isfinite(depth) & (depth > 0)
    nodes, ids = [], set()
    for node in packet["relation_graph"]["nodes"]:
        if node["node_id"] in ids:
            raise ValueError("duplicate node ID")
        ids.add(node["node_id"])
        raw = np.asarray(arrays[node["raw_mask_key"]])
        if raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError("mask dimensions/values invalid")
        mask = raw > 0
        boundary = _boundary(mask)
        nodes.append({"id": node["node_id"], "mask": mask, "boundary": boundary,
                      "hull": _hull(mask, boundary) if config.convex_completion_prior else None})
    count = np.zeros(shape, np.int32)
    for node in nodes:
        count += node["mask"]
    for node in nodes:
        node["depth"] = _patch_depth(node["boundary"], node["mask"], depth,
                                     valid & (count == 1), config.depth_patch_radius_px) if len(node["boundary"]) else np.array([])
    pairs = [_pair_evidence(i, a, b, count, config)
             for i, (a, b) in enumerate(combinations(nodes, 2))]
    edges = [dict(edge, frame_id=frame["frame_id"]) for pair in pairs for edge in pair["edges"]]
    return {
        "schema": "single_frame_relations_v1",
        "frame_id": frame["frame_id"], "alignment_status": frame["rgb_depth_alignment"],
        "config": asdict(config), "pairs": pairs,
        "relation_graph": {
            "status": "single_frame_evidence_and_hypotheses",
            "node_ids": [node["id"] for node in nodes], "edges": edges,
        },
        "limitations": [
            "scores_are_not_probabilities",
            "local_depth_order_does_not_prove_occlusion",
            "convex_completion_is_a_prior_not_observed_hidden_geometry",
            "no_fully_hidden_object_discovery_or_amodal_ground_truth",
            "no_support_relation_or_grasp_feasibility_inference",
            "pixel_thresholds_depend_on_resolution",
        ],
    }
