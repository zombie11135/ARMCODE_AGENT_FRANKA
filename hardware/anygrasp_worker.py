"""Licensed AnyGrasp inference on saved full-scene points; no device access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .gsnet_worker import roi_mask, select_top_one


def filter_candidates(rows, meta, request, mask=None):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise ValueError('Expected finite Nx17 AnyGrasp candidates')
    intr = meta['color_intrinsics']
    if np.any(np.asarray(intr.get('coeffs', [0.] * 5))):
        raise ValueError('Mask projection requires zero-distortion color intrinsics')
    counts = {'sdk_output_after_internal_filters': len(rows)}
    rows = rows[(rows[:, 0] > 0) & (rows[:, 1] > 0) & (rows[:, 1] <= request['max_gripper_width'])]
    counts['after_width_score'] = len(rows)
    rows = rows[roi_mask(rows[:, 13:16], intr, request['roi'])]
    counts['after_box_roi'] = len(rows)
    if mask is not None:
        if mask.shape != (intr['height'], intr['width']) or not np.any(mask):
            raise ValueError('Target SAM mask must match the image and be nonempty')
        xyz = rows[:, 13:16]
        u = np.rint(intr['fx'] * xyz[:, 0] / xyz[:, 2] + intr['ppx']).astype(int)
        v = np.rint(intr['fy'] * xyz[:, 1] / xyz[:, 2] + intr['ppy']).astype(int)
        inside = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        keep = np.zeros(len(rows), dtype=bool)
        keep[inside] = mask[v[inside], u[inside]].astype(bool)
        rows = rows[keep]
        counts['after_target_sam_mask'] = len(rows)
    target_rows = rows.copy()
    base_camera = np.asarray(request['base_T_camera'])
    heights = (rows[:, 13:16] @ base_camera[:3, :3].T + base_camera[:3, 3])[:, 2]
    if request.get('grasp_table_clearance_check', True):
        rows = rows[heights > request['table_z_m'] + .008]
    counts['after_above_table'] = len(rows)
    return rows, target_rows, counts


def run(request):
    import open3d as o3d
    sdk, capture, output = [Path(request[key]).resolve() for key in ('sdk_dir', 'capture_dir', 'output_dir')]
    checkpoint = Path(request['checkpoint']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((capture / 'capture.json').read_text())
    if meta['workspace']['roi_xyxy'] is not None:
        raise ValueError('Inference must retain the full scene for collision checks')
    cloud = o3d.io.read_point_cloud(str(capture / 'scene.ply'))
    points, colors = np.asarray(cloud.points).astype(np.float32), np.asarray(cloud.colors).astype(np.float32)
    if (len(points) < 200 or colors.shape != points.shape or not np.isfinite(points).all()
            or not np.isfinite(colors).all()):
        raise ValueError('Invalid saved colored point cloud')
    mask = (np.load(request['target_mask_path'], allow_pickle=False)
            if request.get('target_mask_path') else None)
    sys.path.insert(0, str(sdk))
    from gsnet import AnyGrasp
    import torch
    from services.anygrasp_service import pushd
    np.random.seed(0)
    torch.manual_seed(0)
    options = SimpleNamespace(checkpoint_path=str(checkpoint), max_gripper_width=request['max_gripper_width'],
                              gripper_height=request['gripper_height'],
                              top_down_grasp=request['top_down_grasp'], debug=False)
    started = time.perf_counter()
    with pushd(str(sdk)):
        net = AnyGrasp(options)
        net.load_net()
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        lims = [float(v) for pair in zip(points.min(0)-.001, points.max(0)+.001) for v in pair]
        group, _ = net.get_grasp(points, colors, lims=lims,
                                apply_object_mask=request['apply_object_mask'],
                                dense_grasp=request['dense_grasp'],
                                collision_detection=request['collision_detection'])
        torch.cuda.synchronize()
        inferred = time.perf_counter()
    raw = np.asarray(group.grasp_group_array)
    rows, target_rows, counts = filter_candidates(raw, meta, request, mask)
    np.save(output / 'sdk_candidates.npy', raw)
    np.save(output / 'target_candidates.npy', target_rows)
    np.save(output / 'filtered_candidates.npy', rows)
    report = dict(backend='anygrasp', counts=counts, input_points=len(points),
                  table_clearance_check_enabled=request.get('grasp_table_clearance_check', True),
                  inference_cloud='full scene; bbox/SAM filters output grasp centers',
                  sdk_collision_detection=request['collision_detection'],
                  sdk_apply_object_mask=request['apply_object_mask'], nms_applied=False,
                  frame=meta['frame'], checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  model_load_s=loaded-started, inference_s=inferred-loaded,
                  hardware_accessed=False, motion_command_sent=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    if not request.get('export_filtered_candidates'):
        chosen = select_top_one(rows)
        np.save(output / 'grasps.npy', chosen)
        row = chosen[0]
        record = dict(score=float(row[0]), width_m=float(row[1]), height_m=float(row[2]),
                      depth_m=float(row[3]), rotation_matrix=row[4:13].reshape(3, 3).tolist(),
                      translation_m=row[13:16].tolist())
        (output / 'grasps.json').write_text(json.dumps({'frame': meta['frame'], 'grasps': [record]}, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()))
