"""Shared GraspNet-format filtering; model implementation is supplied by the worker."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hardware.graspnet_target_contact import (
    FINGER_COLLISION_THRESHOLD, JAW_CLEARANCE_M, MIN_SECTION_POINTS,
    SECTION_QUANTILES, TARGET_VOXEL_M, filter_contacts, mask_membership, target_surface,
)


def checked_gripper_width(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= .085:
        raise ValueError('Gripper width must be finite and in (0, 0.085] m for Robotiq 2F-85')
    return float(value)


def roi_mask(translations, intrinsics, roi):
    points = np.asarray(translations)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    z = np.where(valid, points[:, 2], 1.)
    u = intrinsics['fx'] * points[:, 0] / z + intrinsics['ppx']
    v = intrinsics['fy'] * points[:, 1] / z + intrinsics['ppy']
    x0, y0, x1, y1 = roi
    return valid & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)


def select_top_one(rows):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise ValueError('Expected finite Nx17 GraspNet-format candidates')
    if len(rows) == 0:
        raise RuntimeError('No valid grasp remains; no fallback pose is generated')
    return rows[np.argmax(rows[:, 0]):np.argmax(rows[:, 0]) + 1].copy()


def run(request, model, *, render=None):
    from graspnetAPI import GraspGroup
    import torch

    capture, output = Path(request['capture_dir']), Path(request['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((capture / 'capture.json').read_text())
    # ROI is for target membership only: the entire visible cloud is retained
    # for collision checks, including the table and points outside the ROI.
    max_width = checked_gripper_width(request.get('max_gripper_width', .085))
    args = SimpleNamespace(cloud=capture / 'scene.ply', checkpoint=Path(request['checkpoint']),
                           num_point=int(request.get('num_point', 15000)), voxel_size=.005, seed=0, max_width=max_width)
    if meta['workspace']['roi_xyxy'] is not None:
        raise ValueError('Capture must keep the full scene for collision checks')
    if np.any(np.asarray(meta['color_intrinsics']['coeffs']) != 0):
        raise ValueError('ROI projection requires zero-distortion color intrinsics')
    points, colors, source = model.read_cloud(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw, timing = model.infer(args, points)
    np.save(output / 'grasps_raw.npy', raw)
    group = GraspGroup(raw)
    counts = {'raw': len(group)}
    group = group[(group.scores > 0) & (group.widths > 0) & (group.widths <= args.max_width)]
    counts['after_width_score'] = len(group)
    group = group[roi_mask(group.translations, meta['color_intrinsics'], request['roi'])]
    counts['after_box_roi'] = len(group)
    target_mask = None
    if request.get('target_mask_path'):
        target_mask = np.load(request['target_mask_path'], allow_pickle=False)
        if target_mask.shape != (meta['color_intrinsics']['height'], meta['color_intrinsics']['width']):
            raise ValueError('SAM mask and camera image dimensions differ')
        group = group[mask_membership(group.translations, target_mask, meta['color_intrinsics'])]
        counts['after_target_sam_mask'] = len(group)

    base_camera = np.asarray(request['base_T_camera'])
    heights = (group.translations @ base_camera[:3, :3].T + base_camera[:3, 3])[:, 2]
    if request.get('grasp_table_clearance_check', True):
        group = group[heights > request['table_z_m'] + .008]
    counts['after_above_table'] = len(group)
    rejected = {'collision': 0, 'empty': 0, 'both': 0}
    part_ious = np.empty((5, 0), dtype=float)
    finger_length_m = .06
    if len(group):
        detector = model.ModelFreeCollisionDetector(points, voxel_size=.005)
        finger_length_m = detector.finger_length
        keep, part_batches = [], []
        for offset in range(0, len(group), 32):
            collision, empty, parts = detector.detect(group[offset:offset + 32], approach_dist=.08,
                                               collision_thresh=.01, return_empty_grasp=True,
                                               empty_thresh=.01, return_ious=True)
            part_batches.append(np.asarray(parts))
            rejected['collision'] += int(np.count_nonzero(collision))
            rejected['empty'] += int(np.count_nonzero(empty))
            rejected['both'] += int(np.count_nonzero(collision & empty))
            keep.extend((~collision & ~empty).tolist())
        keep = np.asarray(keep, dtype=bool)
        part_ious = np.concatenate(part_batches, axis=1)[:, keep]
        group = group[keep]
    counts['after_full_scene_collision_and_empty'] = len(group)
    # Keep the old-stage pool for diagnosing candidates rejected by the new checks.
    group.save_npy(str(output / 'pre_contact_candidates.npy'))
    target = None if target_mask is None else target_surface(
        points, target_mask, meta['color_intrinsics'], base_camera, request['table_z_m'])
    finger_ok, target_ok, contact_diagnostics = filter_contacts(
        group.grasp_group_array, part_ious, target=target, finger_length_m=finger_length_m)
    counts['after_per_finger_collision'] = int(np.count_nonzero(finger_ok))
    group = group[finger_ok & target_ok]
    counts['after_target_jaw_clearance'] = len(group)
    contact_report = {
        'version': 1, 'finger_collision_threshold': FINGER_COLLISION_THRESHOLD,
        'target_check_enabled': target is not None,
        'target_points': len(target) if target is not None else None,
        'target_voxel_m': TARGET_VOXEL_M, 'jaw_clearance_m': JAW_CLEARANCE_M,
        'min_section_points': MIN_SECTION_POINTS, 'section_quantiles': SECTION_QUANTILES,
        'finger_length_m': finger_length_m, 'candidates': contact_diagnostics,
    }
    (output / 'contact_filter.json').write_text(json.dumps(contact_report, indent=2))
    if request.get('export_filtered_candidates'):
        # Do not truncate or NMS by raw score before the base-frame tilt ranking.
        group.save_npy(str(output / 'filtered_candidates.npy'))
    if request.get('retain_candidates'):
        # Internal planning pool; the public grasp result still contains one.
        pool = group.nms().sort_by_score() if len(group) else group
        pool = pool[:int(request.get('candidate_limit', 32))]
        pool.save_npy(str(output / 'candidates.npy'))
        counts['internal_planning_candidates'] = len(pool)
    report = {'input': source, 'frame': meta['frame'], 'roi_xyxy': request['roi'],
              'counts': counts, 'input_points': len(points), 'collision_breakdown': rejected,
              'backend': request.get('backend', 'gsnet'),
              'selection': ('pending calibrated pose ranking' if request.get('export_filtered_candidates')
                            else 'highest score AFTER filters'),
              'max_width_m': args.max_width, 'collision_voxel_m': .005,
              'contact_filter': 'contact_filter.json', 'contact_filter_version': 1,
              'collision_threshold': .01, 'empty_threshold': .01, 'approach_check_m': .08,
              'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'collision_model': 'GraspNet approximate parallel fingers; full FR3/tool checked separately by MoveIt',
              'motion_command_sent': False, **timing}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    if request.get('export_filtered_candidates'):
        print(json.dumps({'counts': counts, **timing}), flush=True)
        return
    if request.get('retain_candidates') and request.get('defer_preview'):
        print(json.dumps({'counts': counts, **timing}), flush=True)
        return  # No public grasp until the robot screening has selected one.
    # NMS cannot change the maximum, so it is unnecessary when retaining one.
    selected = GraspGroup(select_top_one(group.grasp_group_array))
    selected.save_npy(str(output / 'grasps.npy'))
    grasp = selected[0]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = grasp.rotation_matrix, grasp.translation
    record = {'rank': 1, 'score': float(grasp.score), 'width_m': float(grasp.width),
              'height_m': float(grasp.height), 'depth_m': float(grasp.depth),
              'translation_m': grasp.translation.tolist(), 'rotation_matrix': grasp.rotation_matrix.tolist(),
              'T_input_grasp': transform.tolist()}
    (output / 'grasps.json').write_text(json.dumps({'frame': meta['frame'],
        'pose_definition': 'Raw GraspNet frame: X approach, Y jaw line; not flange/TCP',
        'grasps': [record]}, indent=2))
    if not request.get('defer_preview') and render is not None:
        model.save_preview(output / 'preview.html', points, colors, selected, 1)
        render(capture, output, top_k=1)
    print(json.dumps({'counts': counts, 'selected': record, **timing}, indent=2), flush=True)
