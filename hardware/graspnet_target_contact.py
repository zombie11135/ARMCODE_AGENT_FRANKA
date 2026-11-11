"""Observed target cross-section checks in the GraspNet frame (metres).

X is approach, Y is jaw closure, Z is finger height. This supplements the
upstream surface collision test; it does not certify unseen contact geometry.
"""
import numpy as np

FINGER_COLLISION_THRESHOLD = .03
TARGET_VOXEL_M = .002
JAW_CLEARANCE_M = .002
MIN_SECTION_POINTS = 12
SECTION_QUANTILES = (.02, .98)


def mask_membership(points, mask, intrinsics):
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    indices = np.flatnonzero(valid)
    xy = points[indices, :2] / points[indices, 2, None]
    uv = np.rint(xy * [intrinsics['fx'], intrinsics['fy']] +
                 [intrinsics['ppx'], intrinsics['ppy']]).astype(int)
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) &
              (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0]))
    result = np.zeros(len(points), dtype=bool)
    result[indices[inside]] = mask[uv[inside, 1], uv[inside, 0]]
    return result


def target_surface(points, mask, intrinsics, base_T_camera, table_z_m):
    """Target-only observed surface, above table, one sample per 2 mm voxel."""
    target = np.asarray(points)[mask_membership(points, mask, intrinsics)]
    transform = np.asarray(base_T_camera)
    height = (target @ transform[:3, :3].T + transform[:3, 3])[:, 2]
    target = target[height > table_z_m + .008]
    _, indices = np.unique(np.floor(target / TARGET_VOXEL_M).astype(np.int64),
                           axis=0, return_index=True)
    return target[indices]


def jaw_clearance(row, target, *, finger_length_m):
    """Require the observed local target span to fit BETWEEN the open fingers.

    Use only the finger's X/Z contact band, not the object's whole bounding box.
    This detects a finger placed inside the target even when the visible surface
    has holes. Quantiles tolerate isolated depth noise; no hull is extrapolated.
    """
    local = (target - row[13:16]) @ row[4:13].reshape(3, 3)
    section = local[(local[:, 0] > row[3] - finger_length_m) &
                    (local[:, 0] < row[3]) & (np.abs(local[:, 2]) < row[2] / 2)]
    result = {'section_points': len(section), 'accepted': False, 'reasons': []}
    if len(section) < MIN_SECTION_POINTS:
        result['reasons'].append('insufficient_target_section')
        return result
    low, high = np.quantile(section[:, 1], SECTION_QUANTILES)
    left, right = low + row[1] / 2, row[1] / 2 - high
    result.update(target_y_bounds_m=[float(low), float(high)],
                  left_clearance_m=float(left), right_clearance_m=float(right),
                  required_opening_at_current_center_m=float(2 * max(-low, high) + 2 * JAW_CLEARANCE_M))
    if min(left, right) < JAW_CLEARANCE_M:
        result['reasons'].append('target_not_between_open_fingers')
    if not low < 0 < high:
        result['reasons'].append('target_not_straddling_grasp_center')
    result['accepted'] = not result['reasons']
    return result


def filter_contacts(rows, part_ious, *, target=None, finger_length_m=.06):
    """Return independent finger and target masks plus per-candidate evidence."""
    finger_ok = (part_ious[1] <= FINGER_COLLISION_THRESHOLD) & (part_ious[2] <= FINGER_COLLISION_THRESHOLD)
    target_ok = np.ones(len(rows), dtype=bool)
    diagnostics = []
    for i, row in enumerate(rows):
        evidence = {'pre_contact_index': i, 'global_collision_iou': float(part_ious[0][i]),
                    'left_finger_iou': float(part_ious[1][i]),
                    'right_finger_iou': float(part_ious[2][i]),
                    'finger_collision_passed': bool(finger_ok[i])}
        if target is not None:
            evidence['target_section'] = jaw_clearance(row, target, finger_length_m=finger_length_m)
            target_ok[i] = evidence['target_section']['accepted']
        evidence['accepted'] = bool(finger_ok[i] and target_ok[i])
        diagnostics.append(evidence)
    return finger_ok, target_ok, diagnostics
