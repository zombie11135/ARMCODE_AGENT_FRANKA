"""Single-box GSNet capture, top-one selection and MoveIt planning.

This module never activates a controller or commands the gripper. Real motion
uses the existing hardware.cli interface, explicitly executed by the operator.
"""
import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .fr3 import FrankaMoveItArm
from .interfaces import Pose6D
from .transforms import load_transforms, pose_matrix, rigid_matrix, matrix_pose


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def pose_difference(before, after):
    a, b = pose_matrix(before), pose_matrix(after)
    return {'translation_m': float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            'rotation_rad': float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()),
            'method': 'poses bracketing stationary capture; not hardware synchronized'}


def validate_capture(root, cfg):
    record = json.loads((root / 'capture_pose.json').read_text())
    for name in ('before', 'after'):
        result = record[name]['result']
        if (result.get('stack') != 'real' or result.get('operation') != 'pose'
                or not result.get('ok') or not result.get('cleanup_ok')):
            raise ValueError('Wrist capture requires successful read-only REAL robot pose samples')
    before, after = [Pose6D(*record[name]['pose']) for name in ('before', 'after')]
    difference = pose_difference(before, after)
    if difference['translation_m'] > .001 or difference['rotation_rad'] > .01:
        raise ValueError('Robot moved during capture; take a new stationary capture')
    transforms = load_transforms(cfg)
    if record['calibration'] != transforms.summary():
        raise ValueError('Calibration changed since capture; use a new capture')
    meta = json.loads((root / 'capture/capture.json').read_text())
    serial = meta['device']['serial_number']
    if serial != cfg.cameras.wrist_serial:
        raise ValueError('Capture serial does not match the calibrated wrist camera')
    if meta['frame'] != f'realsense_{serial}_color_optical':
        raise ValueError('Expected the RealSense COLOR optical frame from the capture helper')
    if meta.get('deprojection') != 'rs.pointcloud.calculate on aligned depth':
        raise ValueError('Expected SDK deprojection of depth aligned to color')
    return record, meta, transforms, before


def validate_roi(roi, meta):
    width, height = [meta['color_intrinsics'][name] for name in ('width', 'height')]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Box ROI must be within the captured color image')


def capture(root, cfg, gsnet_root):
    if cfg.arm.stack != 'real':
        raise ValueError('Live wrist images must be paired with REAL robot poses, never fake FK')
    root.mkdir(parents=True, exist_ok=False)
    arm = FrankaMoveItArm(cfg.arm, execution_enabled=False)
    record = {'motion_command_sent': False, 'calibration': load_transforms(cfg).summary()}
    try:
        for name in ('before', 'after'):
            stamp = time.time()
            pose = arm.get_pose()
            record[name] = {'pose': pose.values, 'time_unix_s': stamp, 'result': arm.last_result}
            if name == 'before':
                subprocess.run([sys.executable, str(gsnet_root / 'realsense_capture.py'),
                    '--serial', cfg.cameras.wrist_serial, '--output-dir', str(root / 'capture'),
                    '--width', str(cfg.cameras.width), '--height', str(cfg.cameras.height),
                    '--fps', str(cfg.cameras.fps), '--warmup', '18', '--samples', '6'],
                    check=True, timeout=90)
        record['stationarity'] = pose_difference(Pose6D(*record['before']['pose']),
                                                 Pose6D(*record['after']['pose']))
    finally:
        arm.close()
        write_json(root / 'capture_pose.json', record)
    validate_capture(root, cfg)


def infer(root, cfg, gsnet_root, roi):
    _, meta, transforms, pose = validate_capture(root, cfg)
    validate_roi(roi, meta)
    workcell = json.loads((Path(cfg.arm.workspace) / 'config/workcell.json').read_text())
    request = {'gsnet_root': str(gsnet_root),
               'checkpoint': str(gsnet_root / 'checkpoints/graspness_realsense.tar'),
               'capture_dir': str(root / 'capture'), 'output_dir': str(root / 'gsnet'),
               'roi': roi, 'base_T_camera': transforms.base_T_camera('wrist', pose).tolist(),
               'table_z_m': workcell['table_z_m'],
               'grasp_table_clearance_check': workcell.get('grasp_table_clearance_check', True)}
    if (root / 'gsnet').exists():
        raise FileExistsError('This capture already has inference results; create a new run')
    write_json(root / 'inference_request.json', request)
    subprocess.run([sys.executable, str(Path(__file__).with_name('gsnet_worker.py')),
                    str(root / 'inference_request.json')], check=True, timeout=180)


def envelope_min_z(flange, workcell):
    half_width = workcell['tool_max_width_m'] / 2 + workcell['tool_padding_m']
    length = workcell['tool_length_m'] + workcell['tool_padding_m']
    corners = np.array(list(itertools.product((-half_width, half_width),
                                              (-half_width, half_width), (0., length))))
    matrix = pose_matrix(flange)
    return float(np.min((corners @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]))


def build_targets(camera_grasp, capture_pose, transforms, serial, optical_frame, workcell):
    camera_grasp = rigid_matrix(camera_grasp, 'GSNet camera_T_grasp')
    flange = transforms.camera_grasp_to_flange_pose(camera_grasp, 'wrist', capture_pose,
                       device_serial=serial, optical_frame=optical_frame)
    base_grasp = transforms.base_T_camera('wrist', capture_pose) @ camera_grasp
    base_flange = pose_matrix(flange)
    approach = base_grasp[:3, 0]
    pregrasp = base_flange.copy()
    pregrasp[:3, 3] -= .08 * approach
    result = {}
    for name, target in [('pregrasp', matrix_pose(pregrasp)), ('grasp', flange)]:
        min_z = envelope_min_z(target, workcell)
        result[name] = {'flange_xyz_rpy': target.values,
                        'tool_envelope_min_z_m': min_z,
                        'above_table_margin': min_z >= workcell['table_z_m'] + workcell['table_clearance_m']}
    return {'targets': result, 'base_T_grasp': base_grasp.tolist(),
            'grasp_reference_position_base_m': base_grasp[:3, 3].tolist(),
            'approach_direction_base': approach.tolist(),
            'jaw_line_direction_base': base_grasp[:3, 1].tolist(),
            'pregrasp_distance_m': .08}


def targets(root, cfg):
    _, meta, transforms, capture_pose = validate_capture(root, cfg)
    source = json.loads((root / 'gsnet/grasps.json').read_text())
    if source['frame'] != meta['frame'] or len(source['grasps']) != 1:
        raise ValueError('Expected exactly one grasp in the capture color optical frame')
    grasp = source['grasps'][0]
    path = Path(cfg.arm.workspace) / 'config/workcell.json'
    workcell = json.loads(path.read_text())
    if (workcell['base_frame'] != 'fr3_link0' or workcell['flange_link'] != 'fr3_link8'
            or workcell['straight_mount_confirmed'] is not True):
        raise ValueError('Expected the existing FR3 straight-mounted tool workcell')
    # These are two names for the same camera's color optical coordinates.
    # validate_capture verifies the serial, color frame and aligned deprojection.
    frame = transforms.cameras['wrist']['optical_frame']
    result = build_targets(grasp['T_input_grasp'], capture_pose, transforms,
                           meta['device']['serial_number'], frame, workcell)
    result.update({'schema_version': 1, 'run_dir': str(root), 'selected': grasp,
        'base_frame': 'fr3_link0', 'target_link': 'fr3_link8', 'units': 'metres; XYZ RPY radians',
        'camera_frame_alias': {'capture': meta['frame'], 'calibration': frame},
        'calibration': transforms.summary(), 'workcell': workcell,
        'grasps_sha256': hashlib.sha256((root / 'gsnet/grasps.json').read_bytes()).hexdigest(),
        'motion_command_sent': False, 'full_grasp_execution_verified': False,
        'reference_assumption': 'GSNet reference point is the nominal 15 cm grasp TCP. '
            'Predicted depth is recorded, not added again to the TCP. The Robotiq finger '
            'contact reference and nominal flange-X jaw alignment need this first physical check.',
        'planning_scope': 'MoveIt checks robot self-collision, table and attached full tool envelope. '
            'The image box is not inserted as a solid obstacle; object contact is not planned.'})
    write_json(root / 'targets.json', result)
    return result


def plan(root, cfg, stage):
    result = targets(root, cfg)
    target = result['targets'][stage]
    if result.get('workcell', {}).get('grasp_table_clearance_check', True) and not target['above_table_margin']:
        raise RuntimeError('Selected pose puts the tool envelope below the table clearance; '
                           'no alternate grasp is selected and no clearance is weakened')
    arm = FrankaMoveItArm(cfg.arm, execution_enabled=False)
    report = {'stage': stage, 'motion_command_sent': False,
              'note': 'Plans this target from CURRENT joints; not a chained grasp trajectory',
              'targets_sha256': hashlib.sha256((root / 'targets.json').read_bytes()).hexdigest()}
    try:
        report['result'] = arm.plan_pose(Pose6D(*target['flange_xyz_rpy']))
        report['ok'] = True
    except Exception as error:
        report.update(ok=False, error=repr(error), result=arm.last_result)
        raise
    finally:
        arm.close()
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        write_json(root / f'plan-{stage}-{stamp}.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--gsnet-root', type=Path, default=Path(sys.prefix) / 'src/gsnet')
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture', help='Read wrist camera and robot pose, infer and export ONE target')
    cap.add_argument('--output-dir', type=Path)
    cap.add_argument('--roi', nargs=4, type=int, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'))
    inf = sub.add_parser('infer', help='Offline inference on an existing capture without results')
    inf.add_argument('run_dir', type=Path)
    inf.add_argument('--roi', nargs=4, type=int, required=True)
    for name in ('targets', 'plan'):
        p = sub.add_parser(name)
        p.add_argument('run_dir', type=Path)
        if name == 'plan':
            p.add_argument('--stage', choices=('pregrasp', 'grasp'), default='pregrasp')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = ((args.output_dir or PROJECT_ROOT / 'runs/gsnet_single' / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            if args.command == 'capture' else args.run_dir).expanduser().resolve()
    try:
        if args.command == 'capture':
            validate_roi(args.roi, {'color_intrinsics': {'width': cfg.cameras.width,
                                                         'height': cfg.cameras.height}})
            capture(root, cfg, args.gsnet_root.resolve())
        if args.command in ('capture', 'infer'):
            infer(root, cfg, args.gsnet_root.resolve(), args.roi)
        result = plan(root, cfg, args.stage) if args.command == 'plan' else targets(root, cfg)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        print('RUN_DIR=' + str(root), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'Stopped: {error!r}; no motion commands sent. Run: {root}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
