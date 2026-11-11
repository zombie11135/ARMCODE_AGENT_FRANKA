"""Conservative RGB-D support patch estimate; no hardware or model calls."""
import math
import numpy as np


def estimate_support(depth_m, intrinsics, base_T_camera, bbox, center_xy, *,
                     footprint_m=.08, max_relief_m=.008, min_coverage=.85):
    depth = np.asarray(depth_m, dtype=float)
    transform = np.asarray(base_T_camera, dtype=float)
    if depth.ndim != 2 or transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Invalid calibrated depth geometry')
    if (not math.isfinite(footprint_m) or footprint_m <= 0 or
            not math.isfinite(max_relief_m) or max_relief_m <= 0):
        raise ValueError('Invalid support footprint or relief limit')
    h, w = depth.shape
    x1, y1, x2, y2 = map(float, bbox)
    # Reject points on the detected container rim/border.
    mx, my = (x2-x1)*.1, (y2-y1)*.1
    x1, x2 = max(0, math.ceil(x1+mx)), min(w, math.floor(x2-mx))
    y1, y2 = max(0, math.ceil(y1+my)), min(h, math.floor(y2-my))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Container interior is empty')
    yy, xx = np.mgrid[y1:y2, x1:x2]
    z = depth[y1:y2, x1:x2]
    valid = np.isfinite(z) & (z > .05) & (z < 2.)
    if valid.sum() < 100:
        raise ValueError('Insufficient valid support depth')
    z = z[valid]
    camera = np.column_stack(((xx[valid]-intrinsics['cx'])*z/intrinsics['fx'],
                              (yy[valid]-intrinsics['cy'])*z/intrinsics['fy'], z))
    points = camera @ transform[:3, :3].T + transform[:3, 3]
    xy = np.asarray(center_xy, dtype=float)
    local = points[:, :2] - xy
    keep = (np.abs(local) <= footprint_m/2).all(axis=1)
    points, local = points[keep], local[keep]
    if len(points) < 100:
        raise ValueError('Insufficient depth over the placement footprint')
    # Grid coverage prevents a visible sliver or a gap from being called a surface.
    cells = np.clip(((local / footprint_m + .5)*8).astype(int), 0, 7)
    counts = np.bincount(cells[:, 0]*8+cells[:, 1], minlength=64)
    coverage = float(np.count_nonzero(counts >= 3)/64)
    if coverage < min_coverage:
        raise ValueError(f'Support footprint coverage {coverage:.1%} < {min_coverage:.1%}')
    low, high = np.quantile(points[:, 2], [.05, .95])
    relief = float(high-low)
    if relief > max_relief_m:
        raise ValueError(f'Stack support is not flat: relief {relief:.4f} m > {max_relief_m:.4f} m')
    return dict(support_z_m=float(high), relief_m=relief, coverage=coverage,
                footprint_m=footprint_m, point_count=len(points),
                method='calibrated_rgbd_patch_p95', center_xy_m=xy.tolist())
