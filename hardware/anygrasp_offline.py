"""Replay saved RGB-D/SAM data with the licensed SDK; no device access or motion."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .agent_geometry import grasp_plan, GraspSelectionError
from .gsnet_worker import roi_mask
from .transforms import load_transforms


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(args):
    import open3d as o3d
    from PIL import Image

    source, observation = args.gsnet_run.resolve(), args.observation.resolve()
    request = json.loads((source / 'request.json').read_text())
    meta = json.loads((source / 'capture/capture.json').read_text())
    record = json.loads((source / 'capture_pose.json').read_text())
    capture = json.loads((observation / 'capture.json').read_text())
    intr = meta['color_intrinsics']
    mask_path = Path(request['target_mask_path'])
    mask = np.load(mask_path, allow_pickle=False).astype(bool)
    organized = np.load(observation / 'points.npy', allow_pickle=False)
    rgb = np.asarray(Image.open(observation / 'color.png').convert('RGB'))
    cloud = o3d.io.read_point_cloud(str(source / 'capture/scene.ply'))
    points = np.asarray(cloud.points).astype(np.float32)
    colors = np.asarray(cloud.colors).astype(np.float32)
    valid = np.isfinite(organized).all(axis=2) & (organized[:, :, 2] > .02) & (organized[:, :, 2] < 1.5)
    if (mask.shape != valid.shape or rgb.shape != organized.shape or not mask.any()
            or not np.array_equal(organized[valid], points)
            or not np.array_equal(rgb[valid], np.rint(colors * 255).astype(np.uint8))):
        raise ValueError('Observation RGB/point cloud must match the saved GSNet scene exactly')
    if np.any(np.asarray(intr.get('coeffs', [0.] * 5))):
        raise ValueError('Target projection requires zero-distortion color intrinsics')
    cfg = load_config(args.config)
    transforms = load_transforms(cfg)
    geometry = capture['capture_geometry']
    if (geometry['calibration'] != record['calibration']
            or geometry['before'] != record['before']['pose']
            or geometry['after'] != record['after']['pose']):
        raise ValueError('Observation and GSNet capture-time geometry differ')
    workcell = json.loads((Path(cfg.arm.workspace) / 'config/workcell.json').read_text())
    request = {**request, 'grasp_table_clearance_check': workcell.get('grasp_table_clearance_check', True)}
    sdk = args.sdk.resolve()
    checkpoint = sdk / 'log/checkpoint_detection.tar'
    output = args.output.resolve() if args.output else (
        PROJECT_ROOT / 'runs/anygrasp_offline' / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    for original, name in [(observation / 'color.png', 'color.png'),
                           (observation / 'depth.png', 'depth.png'),
                           (source / 'capture/scene.ply', 'scene.ply'),
                           (mask_path, 'target_mask.npy')]:
        shutil.copy2(original, output / name)
    overlay = rgb.copy()
    overlay[mask] = (.55 * overlay[mask] + .45 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(output / 'mask_overlay.png')
    manifest = dict(source_run=str(source), observation=str(observation), sdk=str(sdk),
                    checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    observation_matches_gsnet=True,
                    input_sha256={name: sha256(output / name) for name in
                                  ('color.png', 'depth.png', 'scene.ply', 'target_mask.npy')},
                    capture_geometry=geometry, workcell=workcell, request=request,
                    input_points=len(points), mask_pixels=int(mask.sum()),
                    valid_mask_points=int((valid & mask).sum()),
                    top_down_grasp=args.top_down, max_gripper_width_m=.085,
                    inference_cloud='full scene; saved SAM mask filters grasp centers',
                    sdk_apply_object_mask=True, sdk_collision_detection=True,
                    sdk_dense_grasp=False, hardware_accessed=False, motion_command_sent=False)
    write_json(output / 'input.json', manifest)
    print('Output:', output, flush=True)

    # Licensed gsnet and open-source gsnet must run in separate processes.
    sys.path.insert(0, str(sdk))
    from gsnet import AnyGrasp
    from graspnetAPI import GraspGroup
    import torch
    from services.anygrasp_service import pushd

    np.random.seed(0)
    torch.manual_seed(0)
    options = SimpleNamespace(checkpoint_path=str(checkpoint), max_gripper_width=.085,
                              gripper_height=.03, top_down_grasp=args.top_down, debug=False)
    started = time.perf_counter()
    with pushd(str(sdk)):
        net = AnyGrasp(options)
        net.load_net()
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        # Keep all scene points for the SDK's own collision test.
        lims = [float(v) for pair in zip(points.min(0)-.001, points.max(0)+.001) for v in pair]
        group, _ = net.get_grasp(points, colors, lims=lims, apply_object_mask=True,
                                dense_grasp=False, collision_detection=True)
        torch.cuda.synchronize()
        inferred = time.perf_counter()
    rows = np.asarray(group.grasp_group_array)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise ValueError('SDK returned invalid grasp array')
    group = GraspGroup(rows.copy())
    np.save(output / 'sdk_candidates.npy', group.grasp_group_array)
    counts = {'sdk_output_after_internal_filters': len(group)}
    group = group[(group.scores > 0) & (group.widths > 0) & (group.widths <= .085)]
    counts['after_width_score'] = len(group)
    group = group[roi_mask(group.translations, intr, request['roi'])]
    counts['after_same_box_roi'] = len(group)
    xyz = group.translations
    u = np.rint(intr['fx'] * xyz[:, 0] / xyz[:, 2] + intr['ppx']).astype(int)
    v = np.rint(intr['fy'] * xyz[:, 1] / xyz[:, 2] + intr['ppy']).astype(int)
    inside = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    keep = np.zeros(len(group), dtype=bool)
    keep[inside] = mask[v[inside], u[inside]]
    group = group[keep]
    counts['after_same_sam_mask'] = len(group)
    np.save(output / 'target_candidates.npy', group.grasp_group_array)
    base_camera = np.asarray(request['base_T_camera'])
    heights = (group.translations @ base_camera[:3, :3].T + base_camera[:3, 3])[:, 2]
    if request.get('grasp_table_clearance_check', True):
        group = group[heights > request['table_z_m'] + .008]
    counts['after_same_above_table'] = len(group)
    np.save(output / 'filtered_candidates.npy', group.grasp_group_array)
    candidates = [dict(index=i, score=float(row[0]), width=float(row[1]), height=float(row[2]),
                       depth=float(row[3]), rotation_matrix=row[4:13].reshape(3, 3).tolist(),
                       translation=row[13:16].tolist()) for i, row in enumerate(group.grasp_group_array)]
    source_selection = source / 'pose_selection.json'
    selection_options = json.loads(source_selection.read_text())['options'] if source_selection.exists() else {}
    try:
        plan = grasp_plan({'capture_geometry': geometry},
                          {'candidate_grasps_camera': candidates}, transforms, workcell, **selection_options)
    except GraspSelectionError as error:
        selection = error.report
        plan = None
    else:
        selection = plan['pose_selection']
        # Offline output is a preview, not an executable cached agent plan.
        write_json(output / 'selected_preview.json', dict(backend='anygrasp',
                   selected=plan['selected'], geometry=plan['geometry'],
                   hardware_accessed=False, motion_command_sent=False, moveit_checked=False))
        np.save(output / 'selected.npy', group.grasp_group_array[[selection['selected_candidate_index']]])
    accepted = [row['candidate_index'] for row in selection['candidates'] if row['accepted']]
    np.save(output / 'eligible_candidates.npy', group.grasp_group_array[accepted])
    write_json(output / 'pose_selection.json', selection)
    counts['within_rx_ry_limits'] = sum(row['within_tilt_limits'] for row in selection['candidates'])
    counts['tool_above_table'] = sum(row['tool_above_table'] for row in selection['candidates'])
    counts['eligible'] = len(accepted)
    gsnet_counts = json.loads((source / 'gsnet/report.json').read_text())['counts']
    report = dict(backend='anygrasp', counts=counts, gsnet_reference_counts=gsnet_counts,
                  gsnet_reference_eligible=(json.loads(source_selection.read_text())['eligible_candidates']
                                           if source_selection.exists() else None),
                  model_load_s=loaded-started, inference_s=inferred-loaded,
                  total_compute_s=time.perf_counter()-started,
                  output_dir=str(output), hardware_accessed=False, motion_command_sent=False,
                  moveit_checked=False, nms_applied=False,
                  comparison_note='Same saved RGB-D, mask, ROI and FR3 pose/table gates. '
                                  'Each backend uses its own internal collision/object filtering; '
                                  'scores are not calibrated across models.')
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def view(args):
    import open3d as o3d
    from graspnetAPI import GraspGroup
    path = args.run.resolve()
    agent_run = (path / 'anygrasp').is_dir()
    data = path / 'anygrasp' if agent_run else path
    files = {'target': 'target_candidates.npy', 'eligible': 'eligible_candidates.npy',
             'sdk': 'sdk_candidates.npy', 'selected': 'grasps.npy' if agent_run else 'selected.npy'}
    if agent_run and args.stage == 'eligible':
        selection = json.loads((path / 'pose_selection.json').read_text())
        indices = [row['candidate_index'] for row in selection['candidates'] if row['accepted']]
        rows = np.load(data / 'filtered_candidates.npy', allow_pickle=False)[indices]
    else:
        rows = np.load(data / files[args.stage], allow_pickle=False)
    group = GraspGroup(rows).sort_by_score()
    if args.top_k > 0:
        group = group[:args.top_k]
    print(f'{args.stage}: displaying {len(group)} grasps; saved data only, no hardware.')
    cloud = o3d.io.read_point_cloud(str(path / ('capture/scene.ply' if agent_run else 'scene.ply')))
    o3d.visualization.draw_geometries([cloud, *group.to_open3d_geometry_list()],
                                    window_name=f'AnyGrasp / {args.stage} / {len(group)} grasps',
                                    width=1280, height=800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    infer = sub.add_parser('infer')
    infer.add_argument('--gsnet-run', type=Path, required=True)
    infer.add_argument('--observation', type=Path, required=True)
    infer.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    infer.add_argument('--sdk', type=Path, default=Path('/home/seur/franka_teleop/graspNet/anygrasp_sdk/grasp_detection'))
    infer.add_argument('--output', type=Path)
    infer.add_argument('--top-down', action='store_true')
    visual = sub.add_parser('view')
    visual.add_argument('run', type=Path)
    visual.add_argument('--stage', choices=['target', 'eligible', 'sdk', 'selected'], default='target')
    visual.add_argument('--top-k', type=int, default=0, help='0 shows all candidates in the chosen stage')
    args = parser.parse_args()
    replay(args) if args.command == 'infer' else view(args)


if __name__ == '__main__':
    main()
