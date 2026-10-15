"""GSNet -> parallel robot screening -> ONE selected grasp + audited trajectory.

`plan` is planning only. `move --execute` is an operator command which pins the
selected joint configuration, replans and audits again from the current state.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .fr3 import FrankaMoveItArm
from .gsnet_single import validate_capture, validate_roi, build_targets, write_json
from .interfaces import Pose6D
from .transforms import pose_matrix, load_transforms


def grasp_record(row):
    matrix = np.eye(4)
    matrix[:3, :3] = row[4:13].reshape(3, 3)
    matrix[:3, 3] = row[13:16]
    return {'rank': 1, 'score': float(row[0]), 'width_m': float(row[1]),
            'height_m': float(row[2]), 'depth_m': float(row[3]),
            'translation_m': matrix[:3, 3].tolist(), 'rotation_matrix': matrix[:3, :3].tolist(),
            'T_input_grasp': matrix.tolist()}


def render(root, capture, gsnet_root):
    from graspnetAPI import GraspGroup
    import open3d as o3d
    import plotly.graph_objects as go
    group = GraspGroup(str(root / 'gsnet/grasps.npy'))
    if len(group) != 1:
        raise ValueError('Only one selected grasp may be rendered')
    spec = importlib.util.spec_from_file_location('gsnet_rgb_overlay', gsnet_root / 'realsense_preview.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render(capture, root / 'gsnet', top_k=1)
    cloud = o3d.io.read_point_cloud(str(capture / 'scene.ply'))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    indices = np.random.default_rng(0).choice(len(points), min(15000, len(points)), replace=False)
    xyz, rgb = points[indices], np.rint(colors[indices]*255).astype(np.uint8)
    figure = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
        marker={'size': 1.5, 'color': [f'rgb({r},{g},{b})' for r, g, b in rgb]}, name='Scene', hoverinfo='skip'))
    mesh = group[0].to_open3d_geometry()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], color='#00b86b',
        name=f'Selected score={group[0].score:.3f}', showlegend=True))
    figure.update_layout(title='One grasp after FR3 kinematic and trajectory screening',
        scene={'aspectmode': 'data', 'xaxis_title': 'X (m)', 'yaxis_title': 'Y (m)', 'zaxis_title': 'Z (m)',
               'camera': {'up': {'x': 0, 'y': -1, 'z': 0}, 'eye': {'x': -.3, 'y': -.4, 'z': -1.7}}})
    figure.write_html(root / 'gsnet/preview.html', include_plotlyjs=True, auto_open=False)


def plan(source, root, cfg, gsnet_root, roi=None):
    begun = time.perf_counter()
    _, meta, transforms, capture_pose = validate_capture(source, cfg)
    if roi is None:
        old_request = json.loads((source / 'inference_request.json').read_text())
        roi = old_request['roi']
    validate_roi(roi, meta)
    workspace = Path(cfg.arm.workspace)
    policy = json.loads((workspace / 'config/singularity.json').read_text())
    workcell = json.loads((workspace / 'config/workcell.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    request = {'gsnet_root': str(gsnet_root),
        'checkpoint': str(gsnet_root / 'checkpoints/graspness_realsense.tar'),
        'capture_dir': str(source / 'capture'), 'output_dir': str(root / 'gsnet'),
        'roi': roi, 'base_T_camera': transforms.base_T_camera('wrist', capture_pose).tolist(),
        'table_z_m': workcell['table_z_m'], 'retain_candidates': True, 'defer_preview': True,
        'grasp_table_clearance_check': workcell.get('grasp_table_clearance_check', True),
        'candidate_limit': policy['max_grasp_candidates']}
    write_json(root / 'inference_request.json', request)
    inference_start = time.perf_counter()
    subprocess.run([sys.executable, str(Path(__file__).with_name('gsnet_worker.py')),
                    str(root / 'inference_request.json')], check=True, timeout=180)
    inference_wall = time.perf_counter()-inference_start
    rows = np.load(root / 'gsnet/candidates.npy', allow_pickle=False)
    candidates = []
    for index, row in enumerate(rows):
        record = grasp_record(row)
        geometry = build_targets(record['T_input_grasp'], capture_pose, transforms,
            meta['device']['serial_number'], transforms.cameras['wrist']['optical_frame'], workcell)
        target = geometry['targets']['grasp']
        if workcell.get('grasp_table_clearance_check', True) and not target['above_table_margin']:
            continue
        pose = Pose6D(*target['flange_xyz_rpy'])
        candidates.append({'pool_index': index, 'jaw_symmetry_flipped': False, 'score': record['score'],
                           'base_T_flange': pose_matrix(pose).tolist(), 'flange_xyz_rpy': pose.values})
    if not candidates:
        raise RuntimeError('No candidate clears the actual tool envelope above the table')
    write_json(root / 'robot_candidates.json', candidates)
    arm = FrankaMoveItArm(cfg.arm, execution_enabled=False)
    try:
        robot = arm.plan_grasps(candidates)
    except Exception as error:
        write_json(root / 'failure.json', {'error': repr(error), 'robot': arm.last_result,
                                           'motion_command_sent': False})
        raise
    finally:
        arm.close()
    selection = robot['grasp_selection']
    selected = candidates[selection['candidate_index']]
    row = rows[selected['pool_index']].copy()
    record = grasp_record(row)
    # Replace only generated public output, never the original capture/result.
    np.save(root / 'gsnet/grasps.npy', row[None])
    write_json(root / 'gsnet/grasps.json', {'frame': meta['frame'],
        'pose_definition': 'Raw GraspNet reference frame, X approach and Y jaw line', 'grasps': [record]})
    report = {'schema_version': 1, 'source_capture': str(source), 'run_dir': str(root),
        'selected': record, 'selected_pool_index': selected['pool_index'],
        'jaw_symmetry_flipped': selected['jaw_symmetry_flipped'],
        'flange_xyz_rpy': selected['flange_xyz_rpy'], 'goal_joints': selection['goal_joints'],
        'robot': robot, 'calibration': transforms.summary(), 'workcell': workcell, 'policy': policy,
        'base_frame': 'fr3_link0', 'flange_frame': 'fr3_link8', 'motion_command_sent': False,
        'inference_process_wall_s': inference_wall, 'compute_to_validated_plan_s': time.perf_counter()-begun,
        'source_scene_assumption': 'Operator confirmed box unchanged since source capture; capture-time hand-eye pose is reused.',
        'scope': 'Engineering singularity margins, table, tool envelope, robot self-collision. '
                 'Desk warning absence and actual object contact still need physical verification.'}
    write_json(root / 'selection.json', report)
    visual_start = time.perf_counter()
    try:
        render(root, source / 'capture', gsnet_root)
    except Exception as error:
        report['preview_error'] = repr(error)  # Validated trajectory remains available.
    report['preview_wall_s'] = time.perf_counter()-visual_start
    report['total_wall_s'] = time.perf_counter()-begun
    write_json(root / 'selection.json', report)
    return report


def move(root, cfg, execute):
    if not execute:
        raise ValueError('Real movement requires --execute; plan does not move the arm')
    report = json.loads((root / 'selection.json').read_text())
    if report.get('jaw_symmetry_flipped') is not False:
        raise ValueError('This workflow accepts only the original GSNet orientation')
    workspace = Path(cfg.arm.workspace)
    if report['calibration'] != load_transforms(cfg).summary():
        raise ValueError('Calibration changed since selection')
    for name, filename in [('workcell', 'workcell.json'), ('policy', 'singularity.json')]:
        if report[name] != json.loads((workspace / 'config' / filename).read_text()):
            raise ValueError(f'{name} changed since selection')
    if not report['robot']['ok'] or report['robot']['grasp_selection']['motion_command_sent'] is not False:
        raise ValueError('Expected a successful planning-only selection')
    pose = Pose6D(*report['flange_xyz_rpy'])
    arm = FrankaMoveItArm(cfg.arm, execution_enabled=True)
    try:
        return arm.move_joint_goal(pose, report['goal_joints'])
    finally:
        arm.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('capture_run', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--gsnet-root', type=Path, default=Path(sys.prefix) / 'src/gsnet')
    p.add_argument('--roi', nargs=4, type=int)
    p = sub.add_parser('move', help='Operator command: pin selected joint goal, replan and execute once')
    p.add_argument('run_dir', type=Path)
    p.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'move' and not args.execute:
        parser.error('move requires --execute')
    cfg = load_config(args.config)
    try:
        if args.command == 'plan':
            root = (args.output_dir or PROJECT_ROOT / 'runs/gsnet_ranked' /
                    datetime.now().strftime('%Y%m%d-%H%M%S-%f')).expanduser().resolve()
            report = plan(args.capture_run.expanduser().resolve(), root, cfg,
                          args.gsnet_root.expanduser().resolve(), args.roi)
            print(json.dumps({key: report[key] for key in ('run_dir', 'selected', 'goal_joints',
                'flange_xyz_rpy', 'compute_to_validated_plan_s', 'total_wall_s')}, indent=2))
            print('ROBOT_PLAN=' + json.dumps(report['robot'], allow_nan=False), flush=True)
        else:
            print(json.dumps(move(args.run_dir.expanduser().resolve(), cfg, args.execute), indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'Stopped: {error!r}; no automatic retry or recovery', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
