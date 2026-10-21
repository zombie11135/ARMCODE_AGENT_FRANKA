"""Optional planar/convex patch continuation prior for box-like objects.

A fitted visible surface is continued ONLY inside the convex hull of its own
inlier pixels. This can suggest a hidden patch, not prove that patch exists.
Unlike a whole-object median-depth comparison, depth order is evaluated along
each candidate camera ray. Curved/concave objects violate this prior.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .multiview import MultiViewConfig, _prepare


@dataclass(frozen=True)
class PlanarConfig:
    plane_tolerance_m: float = 0.002
    depth_margin_m: float = 0.008
    min_plane_points: int = 40
    min_plane_fraction: float = 0.12
    min_hidden_pixels: int = 12
    min_hidden_fraction: float = 0.05
    max_planes: int = 3
    ransac_trials: int = 80
    max_fit_points: int = 2000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("min_plane_points", "min_hidden_pixels", "max_planes", "ransac_trials", "max_fit_points"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if not 0 < self.min_plane_fraction <= 1 or not 0 < self.min_hidden_fraction <= 1:
            raise ValueError("fractions must be in (0, 1]")
        if self.plane_tolerance_m >= self.depth_margin_m or self.max_fit_points < 3:
            raise ValueError("plane tolerance must be smaller than occlusion depth margin")


def _planes(points, config):
    rng = np.random.default_rng(0)
    remaining = np.arange(len(points))
    for _ in range(config.max_planes):
        if len(remaining) < config.min_plane_points:
            break
        subset = remaining[np.linspace(0, len(remaining)-1, min(len(remaining), config.max_fit_points), dtype=int)]
        fit = points[subset]
        best = None
        best_count = 0
        for _ in range(config.ransac_trials):
            sample = fit[rng.choice(len(fit), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            offset = -sample[0] @ normal
            inlier = np.abs(fit @ normal + offset) <= config.plane_tolerance_m
            if inlier.sum() > best_count:
                best_count, best = int(inlier.sum()), inlier
        if best is None or best_count < 3:
            break
        center = fit[best].mean(axis=0)
        _, singular, vh = np.linalg.svd(fit[best] - center, full_matrices=False)
        if singular[1] < 1e-5:  # no reliable 2D surface support
            break
        normal, offset = vh[-1], -vh[-1] @ center
        indices = remaining[np.abs(points[remaining] @ normal + offset) <= config.plane_tolerance_m]
        if len(indices) < config.min_plane_points or len(indices) / len(points) < config.min_plane_fraction:
            break
        yield normal, float(offset), indices
        remaining = remaining[~np.isin(remaining, indices)]


def extract_planar_hypotheses(packet, arrays, config=None):
    config = config or PlanarConfig()
    view = _prepare(packet, arrays, MultiViewConfig())
    fx, fy, cx, cy = view["intrinsic"]
    edges, patches = [], []
    for node_index, points in enumerate(view["evidence_points"]):
        for plane_index, (normal, offset, indices) in enumerate(_planes(points, config)):
            inliers = points[indices]
            pixels = np.rint(np.column_stack((inliers[:, 0] * fx / inliers[:, 2] + cx,
                                             inliers[:, 1] * fy / inliers[:, 2] + cy))).astype(np.int32)
            hull = np.zeros(view["depth"].shape, np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(pixels).reshape(-1, 2), 1)
            source_mask = np.asarray(arrays[view["nodes"][node_index]["raw_mask_key"]], bool)
            missing = hull.astype(bool) & ~source_mask
            y, x = np.nonzero(missing & (view["owner"] >= 0))
            rays = np.column_stack(((x - cx) / fx, (y - cy) / fy, np.ones(len(x))))
            denominator = rays @ normal
            predicted_z = np.divide(-offset, denominator, out=np.full(len(x), np.nan),
                                    where=np.abs(denominator) > 1e-8)
            measured = view["depth"][y, x]
            valid = np.isfinite(measured) & (measured > 0) & np.isfinite(predicted_z) & (predicted_z > 0)
            hidden = valid & (predicted_z - measured > config.depth_margin_m)
            patches.append({
                "node_id": view["nodes"][node_index]["node_id"], "plane_index": plane_index,
                "inlier_points": len(indices), "hull_pixels": int(hull.sum()),
                "plane_normal_camera": normal.tolist(), "plane_offset_m": offset,
                "median_residual_m": float(np.median(np.abs(inliers @ normal + offset))),
            })
            for front_index in range(len(view["nodes"])):
                if front_index == node_index:
                    continue
                selected = hidden & (view["owner"][y, x] == front_index)
                count = int(selected.sum())
                fraction = count / max(1, int(hull.sum()))
                if count < config.min_hidden_pixels or fraction < config.min_hidden_fraction:
                    continue
                xy = np.column_stack((x[selected], y[selected]))
                stride = max(1, int(np.ceil(len(xy) / 256)))
                edges.append({
                    "source": view["nodes"][front_index]["node_id"],
                    "target": view["nodes"][node_index]["node_id"],
                    "relation": "possible_occludes",
                    "evidence": "visible_planar_patch_convex_continuation_prior",
                    "view_id": view["frame"]["camera_id"],
                    "plane_index": plane_index, "unique_support_pixels": count,
                    "fraction_of_completed_patch": float(fraction),
                    "depth_gap_median_m": float(np.median((predicted_z-measured)[selected])),
                    "support_pixels_sample_xy": xy[::stride].tolist(),
                    "confirmed": False, "calibrated_probability": None,
                    "scope": "hypothesized_surface_only_inside_visible_patch_convex_hull",
                })
    return {"schema": "planar_continuation_v1", "config": asdict(config),
            "patches": patches, "edges": edges}
