"""Two synchronized RGB-D views -> anonymous, view-conditioned occlusion evidence.

Only visible masks, measured depth and camera calibration are read. No semantic
scores, cached object clouds, simulator object poses or hidden silhouettes enter
inference. This is an evidence extractor, not a target recognizer or grasp policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .observations import _validate_transform


@dataclass(frozen=True)
class MultiViewConfig:
    association_depth_tolerance_m: float = 0.004
    occlusion_depth_margin_m: float = 0.008
    boundary_margin_px: int = 1
    source_neighbor_depth_tolerance_m: float = 0.008
    min_source_depth_neighbors: int = 2
    min_association_pixels: int = 8
    min_association_fraction: float = 0.10
    association_best_ratio: float = 2.0
    min_occlusion_pixels: int = 8
    min_occlusion_fraction: float = 0.05
    max_timestamp_delta_s: float = 0.05
    min_consistency_check_pixels: int = 64
    max_free_space_violation_fraction: float = 0.10

    def __post_init__(self):
        for name in ("boundary_margin_px", "min_association_pixels", "min_occlusion_pixels",
                     "min_source_depth_neighbors", "min_consistency_check_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "boundary_margin_px" else 1):
                raise ValueError(f"{name} must be a valid integer")
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.association_depth_tolerance_m < self.occlusion_depth_margin_m:
            raise ValueError("require 0 < association tolerance < occlusion margin")
        if self.source_neighbor_depth_tolerance_m <= 0 or self.min_source_depth_neighbors > 8:
            raise ValueError("invalid source-neighbor depth support")
        if self.max_timestamp_delta_s < 0 or self.association_best_ratio <= 1:
            raise ValueError("invalid synchronization or association margin")
        for name in ("min_association_fraction", "min_occlusion_fraction",
                     "max_free_space_violation_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


def _prepare(packet, arrays, config):
    frame = packet["frame"]
    depth = np.asarray(arrays["depth_m"])
    if depth.shape != (frame["height"], frame["width"]):
        raise ValueError("depth shape does not match frame")
    if (frame["depth_units"] != "metres" or frame["depth_definition"] != "optical_Z"
            or frame["axes"] != "X_right_Y_down_Z_forward"):
        raise ValueError("metric optical-Z depth in OpenCV axes is required")
    if frame["rgb_depth_alignment"] != "recorded_aligned":
        raise ValueError("recorded RGB/depth alignment is required")
    if not frame.get("reference_frame") or frame.get("T_reference_from_camera_optical") is None:
        raise ValueError("camera pose and reference frame are required")
    transform = _validate_transform(frame["T_reference_from_camera_optical"])
    intrinsic = np.array([frame["intrinsics"][k] for k in ("fx", "fy", "cx", "cy")], float)
    if not np.isfinite(intrinsic).all() or min(intrinsic[:2]) <= 0:
        raise ValueError("invalid camera intrinsics")
    nodes = packet["relation_graph"]["nodes"]
    ids = [n["node_id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate node IDs")
    masks = []
    counts = np.zeros(depth.shape, np.int32)
    for node in nodes:
        raw = np.asarray(arrays[node["raw_mask_key"]])
        if raw.shape != depth.shape or not np.isfinite(raw).all():
            raise ValueError("invalid visible instance mask")
        mask = raw > 0
        masks.append(mask)
        counts += mask
    owner = np.full(depth.shape, -1, np.int32)  # -1 unassigned; -2 conflict/boundary
    owner[counts > 1] = -2
    points, evidence_points = [], []
    fx, fy, cx, cy = intrinsic
    radius = config.boundary_margin_px
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    for index, mask in enumerate(masks):
        inner = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool) if radius else mask
        inner &= counts == 1
        owner[mask & (counts == 1)] = -2
        owner[inner] = index
        y, x = np.nonzero(inner & np.isfinite(depth) & (depth > 0))
        z = depth[y, x].astype(float)
        points.append(np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z)))
        # Do not erase a narrow visible rim merely because it has no eroded
        # interior. Keep actual measured samples with same-instance, similar-
        # depth neighbors; neither interpolate nor fill missing measurements.
        supported = _depth_supported_pixels(
            mask & (counts == 1), depth, config.source_neighbor_depth_tolerance_m,
            config.min_source_depth_neighbors)
        y, x = np.nonzero(supported)
        z = depth[y, x].astype(float)
        evidence_points.append(np.column_stack(((x-cx)*z/fx, (y-cy)*z/fy, z)))
    return dict(frame=frame, depth=depth, owner=owner, points=points, nodes=nodes,
                evidence_points=evidence_points, transform=transform, intrinsic=intrinsic)


def _depth_supported_pixels(mask, depth, tolerance, min_neighbors):
    valid = mask & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    output = np.zeros(mask.shape, bool)
    if not len(y):
        return output
    y0, y1, x0, x1 = y.min(), y.max()+1, x.min(), x.max()+1
    local = valid[y0:y1, x0:x1]
    measured = depth[y0:y1, x0:x1]
    padded_valid = np.pad(local, 1, constant_values=False)
    padded_depth = np.pad(np.where(local, measured, 0), 1, constant_values=0)
    height, width = local.shape
    support = np.zeros(local.shape, np.uint8)
    current = np.where(local, measured, 0)
    for dy in range(3):
        for dx in range(3):
            if dy == dx == 1:
                continue
            neighbor = padded_depth[dy:dy+height, dx:dx+width]
            support += (padded_valid[dy:dy+height, dx:dx+width]
                        & (np.abs(neighbor-current) <= tolerance))
    output[y0:y1, x0:x1] = local & (support >= min_neighbors)
    return output


def _project(source, destination, points, config):
    """Nearest-pixel projection, with a per-instance destination Z buffer."""
    relative = np.linalg.inv(destination["transform"]) @ source["transform"]
    xyz = points @ relative[:3, :3].T + relative[:3, 3]
    positive = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz = xyz[positive]
    fx, fy, cx, cy = destination["intrinsic"]
    height, width = destination["depth"].shape
    uv = np.column_stack((xyz[:, 0] * fx / xyz[:, 2] + cx,
                          xyz[:, 1] * fy / xyz[:, 2] + cy))
    inside = ((uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
              & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5))
    pixels = np.floor(uv[inside] + 0.5).astype(np.int64)
    z = xyz[inside, 2]
    flat = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((z, flat))
    keep = order[np.r_[True, np.diff(flat[order]) != 0]] if len(order) else order
    pixels, z = pixels[keep], z[keep]
    x, y = pixels.T
    measured = destination["depth"][y, x]
    valid = np.isfinite(measured) & (measured > 0)
    owner = destination["owner"][y, x]
    delta = z - measured
    same = valid & (np.abs(delta) <= config.association_depth_tolerance_m)
    behind = valid & (delta > config.occlusion_depth_margin_m)
    front = valid & (delta < -config.occlusion_depth_margin_m)
    stats = {
        "source_valid_surface_pixels": len(points),
        "behind_camera_points": int((~positive).sum()),
        "out_of_frame_points": int((~inside).sum()),
        "duplicate_destination_pixels_removed": int(len(flat) - len(keep)),
        "unique_projected_pixels": len(keep),
        "missing_destination_depth_pixels": int((~valid).sum()),
        "mask_conflict_or_boundary_pixels": int((valid & (owner == -2)).sum()),
        "unassigned_depth_pixels": int((valid & (owner == -1)).sum()),
        "same_surface_pixels": int(same.sum()),
        "same_surface_without_instance_pixels": int((same & (owner == -1)).sum()),
        "behind_measured_surface_pixels": int(behind.sum()),
        "in_front_inconsistent_pixels": int(front.sum()),
        "uncertainty_band_pixels": int((valid & ~same & ~behind & ~front).sum()),
    }
    return dict(pixels=pixels, owner=owner, delta=delta, same=same, behind=behind,
                stats=stats)


def extract_multiview(observations, config=None):
    """Infer conservative evidence from exactly two contemporaneous view packets.

    `observations` is [(packet, arrays), (packet, arrays)]. IDs are only local
    labels. Entity IDs are observation-group-local, not persistent object tracks.
    A view-local split of one physical object into two masks remains a perception
    failure: distinct-instance witnesses assume each mask is a distinct object.
    """
    config = config or MultiViewConfig()
    if len(observations) != 2:
        raise ValueError("this version requires exactly two views")
    views = [_prepare(p, a, config) for p, a in observations]
    a, b = [v["frame"] for v in views]
    for field in ("capture_group_id", "timestamp_domain", "reference_frame"):
        if not a.get(field) or a[field] != b.get(field):
            raise ValueError(f"views require the same nonempty {field}")
    for field in ("camera_id", "frame_id"):
        if not a.get(field) or not b.get(field) or a[field] == b[field]:
            raise ValueError(f"views require distinct nonempty {field}")
    timestamps = [f.get("timestamp_s") for f in (a, b)]
    if any(t is None or not np.isfinite(t) for t in timestamps):
        raise ValueError("acquisition timestamps are required")
    if abs(timestamps[0] - timestamps[1]) > config.max_timestamp_delta_s:
        raise ValueError("acquisition timestamps are too far apart")
    all_ids = [n["node_id"] for v in views for n in v["nodes"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("node IDs must be unique across views")
    association_projections = [
        [_project(views[s], views[1 - s], points, config) for points in views[s]["points"]]
        for s in range(2)
    ]
    projections = [
        [_project(views[s], views[1-s], points, config) for points in views[s]["evidence_points"]]
        for s in range(2)
    ]
    # A surface observed by A cannot lie in the free space in front of B's
    # measured first hit, beyond the allowed depth uncertainty. Test this across
    # all interior instance samples before trusting ANY identity correspondence.
    # This is a rejection check, not a certificate of calibration accuracy.
    consistency_checks = []
    for s in range(2):
        tested = violations = 0
        for projection in association_projections[s]:
            valid = (projection["owner"] >= 0) & np.isfinite(projection["delta"])
            # Missing depth is not free space, even if delta itself is finite.
            x, y = projection["pixels"].T
            measured = views[1-s]["depth"][y, x]
            valid &= np.isfinite(measured) & (measured > 0)
            tested += int(valid.sum())
            violations += int((valid & (projection["delta"] < -config.occlusion_depth_margin_m)).sum())
        fraction = violations / max(1, tested)
        consistency_checks.append({
            "source_camera": views[s]["frame"]["camera_id"],
            "destination_camera": views[1-s]["frame"]["camera_id"],
            "tested_interior_pixels": tested, "free_space_violations": violations,
            "violation_fraction": fraction,
            "rejected": tested >= config.min_consistency_check_pixels
                        and fraction > config.max_free_space_violation_fraction,
        })
    geometry_rejected = any(c["rejected"] for c in consistency_checks)
    count_matrices, fraction_matrices = [], []
    for s in range(2):
        counts = np.zeros((len(views[s]["nodes"]), len(views[1-s]["nodes"])), int)
        fractions = np.zeros_like(counts, float)
        for i, projection in enumerate(association_projections[s]):
            owners = projection["owner"][projection["same"] & (projection["owner"] >= 0)]
            counts[i] = np.bincount(owners, minlength=counts.shape[1])
            fractions[i] = counts[i] / max(1, projection["stats"]["unique_projected_pixels"])
        count_matrices.append(counts)
        fraction_matrices.append(fractions)
    # Require enough shared surface in BOTH directions. A one-way accidental
    # overlap, centroid proximity or common text/category never merges identities.
    scores = np.minimum(fraction_matrices[0], fraction_matrices[1].T)
    eligible = ((count_matrices[0] >= config.min_association_pixels)
                & (count_matrices[1].T >= config.min_association_pixels)
                & (scores >= config.min_association_fraction))
    accepted = []
    associations = []
    for i, j in zip(*np.nonzero(eligible)):
        score = scores[i, j]
        row_other = np.delete(scores[i], j)
        col_other = np.delete(scores[:, j], i)
        competitor = max(row_other.max(initial=0), col_other.max(initial=0))
        unambiguous = score > competitor * config.association_best_ratio
        row = {
            "nodes": [views[0]["nodes"][i]["node_id"], views[1]["nodes"][j]["node_id"]],
            "status": ("rejected_inconsistent_cross_view_geometry" if geometry_rejected else
                       ("accepted" if unambiguous else "ambiguous")),
            "shared_surface_pixels": [int(count_matrices[0][i, j]), int(count_matrices[1][j, i])],
            "bidirectional_support_fraction": float(score),
            "strongest_competitor_fraction": float(competitor),
        }
        associations.append(row)
        if unambiguous and not geometry_rejected:
            accepted.append((int(i), int(j)))
    entities, node_to_entity = [], {}
    for i, j in accepted:
        members = [views[0]["nodes"][i]["node_id"], views[1]["nodes"][j]["node_id"]]
        entity = f"entity_{len(entities):03d}"
        entities.append({"entity_id": entity, "members": members, "scope": "current_capture_group"})
        node_to_entity.update({node: entity for node in members})
    for node in all_ids:
        if node not in node_to_entity:
            entity = f"entity_{len(entities):03d}"
            node_to_entity[node] = entity
            entities.append({"entity_id": entity, "members": [node], "scope": "current_capture_group"})
    entity_members = {e["entity_id"]: set(e["members"]) for e in entities}
    edges, unresolved, projection_rows = [], [], []
    for s in range(2):
        source_ids = {n["node_id"] for n in views[s]["nodes"]}
        for i, projection in enumerate(projections[s]):
            source_node = views[s]["nodes"][i]["node_id"]
            back_entity = node_to_entity[source_node]
            projection_rows.append({
                "source_node": source_node, "source_camera": views[s]["frame"]["camera_id"],
                "destination_camera": views[1-s]["frame"]["camera_id"],
                **projection["stats"],
                "source_sample_policy": "measured_visible_pixels_with_local_depth_support_including_thin_rims",
                "association_source_interior_pixels": len(views[s]["points"][i]),
            })
            for j, dest_node in enumerate(views[1-s]["nodes"]):
                if geometry_rejected:
                    break
                selection = projection["behind"] & (projection["owner"] == j)
                count = int(selection.sum())
                fraction = count / max(1, projection["stats"]["unique_projected_pixels"])
                if count < config.min_occlusion_pixels or fraction < config.min_occlusion_fraction:
                    continue
                front_entity = node_to_entity[dest_node["node_id"]]
                # Same physical object may hide its own back/side surfaces.
                if front_entity == back_entity:
                    continue
                # Distinctness needs two separate masks in at least one view,
                # linked across cameras by an accepted shared-surface match.
                front_in_source = entity_members[front_entity] & source_ids
                back_in_destination = entity_members[back_entity] - source_ids
                distinct = bool(front_in_source or back_in_destination)
                deltas = projection["delta"][selection]
                pixel_list = projection["pixels"][selection]
                stride = max(1, int(np.ceil(len(pixel_list) / 256)))
                row = {
                    "source": front_entity, "target": back_entity,
                    "relation": "occludes_observed_surface" if distinct else "unresolved_front_surface",
                    "view_id": views[1-s]["frame"]["camera_id"],
                    "source_view_id": views[s]["frame"]["camera_id"],
                    "foreground_node": dest_node["node_id"], "reprojected_node": source_node,
                    "unique_support_pixels": count,
                    "fraction_of_projected_observed_surface": float(fraction),
                    "depth_gap_median_m": float(np.median(deltas)),
                    "depth_gap_p10_m": float(np.quantile(deltas, 0.1)),
                    "distinct_instance_witness_nodes": sorted(front_in_source | back_in_destination),
                    "evidence": "reprojected_visible_surface_behind_measured_distinct_instance",
                    "scope": "only_source_observed_surface_in_destination_view",
                    "calibrated_probability": None,
                    "support_pixels_sample_xy": pixel_list[::stride].tolist(),
                }
                if distinct:
                    edges.append(row)
                else:
                    row["reason"] = "cross_view_identity_unresolved_could_be_self_occlusion"
                    unresolved.append(row)
    return {
        "schema": "multiview_occlusion_v1", "config": asdict(config),
        "capture_group_id": a["capture_group_id"],
        "geometry_health": {
            "status": "rejected_inconsistent_depth" if geometry_rejected else "no_large_contradiction_detected",
            "cross_view_inference_enabled": not geometry_rejected,
            "checks": consistency_checks,
            "meaning": "rejection_test_only_not_proof_of_calibration_or_scene_synchronization",
        },
        "views": [{"camera_id": v["frame"]["camera_id"], "frame_id": v["frame"]["frame_id"],
                   "observed_instances": len(v["nodes"])} for v in views],
        "associations": associations, "entities": entities, "node_to_entity": node_to_entity,
        "relation_graph": {"edges": edges, "scope": "current_capture_group_and_named_view"},
        "unresolved_front_surfaces": unresolved, "projections": projection_rows,
        "unobserved_hypothesis": {
            "status": "not_localized", "probability": None,
            "reason": "objects_hidden_in_both_views_or_missed_by_segmentation_are_not_invented",
        },
        "limitations": [
            "No full-object occlusion percentage, hidden geometry, target identity or grasp feasibility.",
            "Visible instance masks must distinguish physical objects; split/merged masks can corrupt evidence.",
            "Unmatched nodes may still be the same object; unresolved evidence is not an occlusion edge.",
            "Thresholds are sensor-dependent margins, not calibrated confidence probabilities.",
            "Depth agreement and association need overlap, alignment, calibrated poses and a static synchronized scene.",
            "No persistent tracking across manipulations; rebuild after each action.",
        ],
    }
