"""Static dual RGB-D + passive joint/FK reads. No robot motion or gripper APIs.

Fixed: base <- global optical. Wrist: base <- capture-time flange <- wrist optical.
The saved geometry is valid for this capture group only, never a permanent pair.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from .camera_only import _choose_pair, _default_service, save_pair
from .observations import _validate_transform


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def calibration_inputs(fixed_path, wrist_path, serials):
    fixed_path, wrist_path = Path(fixed_path), Path(wrist_path)
    fixed = json.loads(fixed_path.read_text())
    wrist_file = yaml.safe_load(wrist_path.read_text())
    if (fixed.get('schema') != 'fr3_eye_to_hand_result_v1'
            or fixed.get('base_frame') != 'fr3_link0'
            or fixed.get('flange_frame') != 'fr3_link8'
            or fixed.get('camera_frame') != 'global_color_optical_frame'
            or fixed.get('translation_unit') != 'm'
            or fixed.get('matrix_convention') != 'p_parent = T_parent_child @ p_child'):
        raise ValueError('Fixed calibration must describe metric fr3_link0 <- global_color_optical_frame')
    if (wrist_file.get('schema_version') != 1 or wrist_file.get('translation_unit') != 'm'
            or wrist_file.get('matrix_convention') != 'p_parent = T_parent_child @ p_child'):
        raise ValueError('Invalid wrist calibration convention')
    wrist = wrist_file['cameras']['wrist']
    extrinsic = wrist['extrinsic']
    if (wrist.get('mounting') != 'eye_in_hand' or extrinsic.get('parent_frame') != 'fr3_link8'
            or extrinsic.get('child_frame') != 'wrist_color_optical_frame'
            or wrist.get('image_stream') != 'color' or wrist.get('depth_aligned_to') != 'color'):
        raise ValueError('Wrist extrinsic must be flange <- wrist color optical, with aligned depth')
    if (set(serials) != {'global', 'wrist'} or not all(isinstance(s, str) and s for s in serials.values())
            or len(set(serials.values())) != 2
            or fixed.get('camera_serial') != serials['global'] or wrist.get('serial') != serials['wrist']):
        raise ValueError('Fixed/wrist calibration serials do not match the two devices')
    return {
        'fixed': {'path': str(fixed_path.resolve()), 'sha256': digest(fixed_path),
                  'status': fixed.get('status'), 'base_T_camera': _validate_transform(fixed['base_T_camera']).tolist()},
        'wrist': {'path': str(wrist_path.resolve()), 'sha256': digest(wrist_path),
                  'status': wrist_file.get('status'),
                  'flange_T_camera': _validate_transform(extrinsic['matrix']).tolist()},
    }


def stationary_joint_window(states, begin, end, image_time):
    window = [s for s in states if begin-.15 <= s['received_at'] <= end+.15]
    stamps = np.array([s['received_at'] for s in window], float)
    if (len(window) < 5 or not np.isfinite(stamps).all()
            or stamps[0] > begin or stamps[-1] < end
            or np.any(np.diff(stamps) <= 0) or np.max(np.diff(stamps)) > .2):
        raise ValueError('Fresh robot feedback must continuously bracket the RGB-D acquisition')
    q, dq = (np.asarray([s[key] for s in window], float) for key in ('q', 'dq'))
    if q.shape != (len(window), 7) or dq.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError('Invalid finite seven-joint feedback')
    span, speed = float(np.ptp(q, axis=0).max()), float(np.abs(dq).max())
    if span > .002 or speed > .01:
        raise ValueError('Robot moved during capture; wait stationary and capture again')
    chosen = window[int(np.argmin(np.abs(stamps-image_time)))]
    delta = abs(chosen['received_at']-image_time)
    if delta > .1:
        raise ValueError('Wrist image and joint feedback receive times differ by more than 100 ms')
    return chosen, {'max_joint_span_rad': span, 'max_joint_speed_rad_s': speed,
                    'joint_samples': len(window), 'image_state_receive_delta_s': delta,
                    'bracket_begin_monotonic_s': begin, 'bracket_end_monotonic_s': end}


def validate_geometry(geometry):
    if (geometry.get('schema') != 'robot_base_capture_geometry_v1'
            or geometry.get('reference_frame') != 'fr3_link0'
            or geometry.get('flange_frame') != 'fr3_link8'
            or geometry.get('translation_unit') != 'm'
            or geometry.get('matrix_convention') != 'p_parent = T_parent_child @ p_child'
            or geometry.get('validity') != 'this_stationary_capture_group_only'
            or not geometry.get('capture_group_id')):
        raise ValueError('Invalid capture-bound base-frame geometry')
    serials = geometry['camera_serials']
    max_skew = geometry['max_frame_receive_skew_s']
    if not np.isfinite(max_skew) or max_skew <= 0:
        raise ValueError('Finite positive frame receive skew limit required')
    timestamps = [geometry['frames'][c]['capture_timestamp_s'] for c in ('global', 'wrist')]
    if not np.isfinite(timestamps).all() or abs(timestamps[0]-timestamps[1]) > max_skew:
        raise ValueError('Captured frame timestamps exceed their recorded pairing limit')
    if (set(serials) != {'global', 'wrist'} or len(set(serials.values())) != 2
            or not all(isinstance(s, str) and s for s in serials.values())):
        raise ValueError('Two explicitly bound camera serials required')
    joint = geometry['robot_state']
    selected, _ = stationary_joint_window(joint['samples'], joint['begin_monotonic_s'],
                                          joint['end_monotonic_s'], joint['wrist_receive_monotonic_s'])
    if selected != joint['selected_joint_state']:
        raise ValueError('Selected capture-time joint state changed')
    before, after = (_validate_transform(joint[key]) for key in ('base_T_flange_before', 'base_T_flange_after'))
    shift = np.linalg.norm(before[:3, 3]-after[:3, 3])
    angle = np.arccos(np.clip((np.trace(before[:3, :3].T@after[:3, :3])-1)/2, -1, 1))
    if shift > .001 or angle > .01:
        raise ValueError('Capture-time FK endpoints are not stationary')
    fixed = _validate_transform(geometry['calibrations']['fixed']['base_T_camera'])
    wrist = (_validate_transform(joint['base_T_flange_at_wrist_capture'])
             @ _validate_transform(geometry['calibrations']['wrist']['flange_T_camera']))
    for camera, expected in [('global', fixed), ('wrist', wrist)]:
        frame = geometry['frames'][camera]
        if frame['device_serial'] != serials[camera]:
            raise ValueError('Capture geometry serial mismatch')
        if not np.allclose(_validate_transform(frame['base_T_camera']), expected, atol=1e-8, rtol=0):
            raise ValueError('Camera pose does not match its capture-time composition')
    return {'global': fixed, 'wrist': wrist}


def verify_snapshot(geometry, camera, metadata, source):
    transforms = validate_geometry(geometry)
    saved = geometry['frames'][camera]
    if (metadata['capture_group_id'] != geometry['capture_group_id']
            or metadata['device_serial'] != geometry['camera_serials'][camera]
            or metadata['capture_timestamp_s'] != saved['capture_timestamp_s']
            or metadata['color_frame_number'] != saved['color_frame_number']
            or metadata['depth_frame_number'] != saved['depth_frame_number']
            or metadata['depth_intrinsics'] != saved['depth_intrinsics']
            or metadata['timestamp_domain'] != geometry['timestamp_domain']):
        raise ValueError('Snapshot does not match this capture geometry')
    for filename, key in [('color.png', 'rgb_sha256'), ('depth_m.npy', 'depth_sha256')]:
        if digest(Path(source)/filename) != saved[key]:
            raise ValueError('Snapshot image/depth no longer matches capture geometry')
    return transforms[camera]


def bind_base_pair(observations, geometry, serials):
    transforms = validate_geometry(geometry)
    if serials != geometry['camera_serials'] or len(observations) != 2:
        raise ValueError('Camera pair does not match capture geometry')
    bound = []
    if {p['frame']['camera_id'] for p, _ in observations} != {'global', 'wrist'}:
        raise ValueError('One global and one wrist observation required')
    for packet, arrays in observations:
        packet, arrays = copy.deepcopy(packet), dict(arrays)
        frame = packet['frame']
        camera = frame['camera_id']
        saved = geometry['frames'][camera]
        if (frame['capture_group_id'] != geometry['capture_group_id']
                or frame['timestamp_s'] != saved['capture_timestamp_s']
                or frame['timestamp_domain'] != geometry['timestamp_domain']
                or packet['provenance']['rgb_sha256'] != saved['rgb_sha256']
                or packet['provenance']['depth_sha256'] != saved['depth_sha256']):
            raise ValueError('Observation belongs to a different frame/capture')
        frame['reference_frame'] = 'fr3_link0'
        frame['T_reference_from_camera_optical'] = transforms[camera].tolist()
        k, depth = frame['intrinsics'], np.asarray(arrays['depth_m'])
        expected = saved['depth_intrinsics']
        if any(k[a] != expected[b] for a, b in [('fx', 'fx'), ('fy', 'fy'), ('cx', 'ppx'), ('cy', 'ppy')]):
            raise ValueError('Observation intrinsics changed since capture')
        for node in packet['relation_graph']['nodes']:
            mask = np.asarray(arrays[node['raw_mask_key']], bool)
            y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
            z = depth[y, x]
            xyz = np.column_stack(((x-k['cx'])*z/k['fx'], (y-k['cy'])*z/k['fy'], z))
            key = node.get('observed_points_reference_key') or node['node_id']+'_base_points'
            node['observed_points_reference_key'] = key
            arrays[key] = xyz@transforms[camera][:3, :3].T+transforms[camera][:3, 3]
        packet['provenance'].update(reference_frame_scope='capture_time_robot_base',
                                     external_camera_pose_available=True,
                                     capture_robot_state_queried=True)
        bound.append((packet, arrays))
    return bound


def capture(config_path, fixed_calibration, output_dir=None, *, verification_note='',
            service_factory=None, reader_factory=None, warmup_s=2., pairing_timeout_s=3., max_skew_s=.05):
    # Imports are local: loading saved geometry never constructs a ROS/device adapter.
    from hardware.config import load_config
    from hardware.eye_to_hand_capture import RobotReader
    config = load_config(config_path)
    if config.arm.stack != 'real':
        raise ValueError('Real images require real joint feedback, never fake joint states')
    if not all(np.isfinite(v) and v > 0 for v in (warmup_s, pairing_timeout_s, max_skew_s)):
        raise ValueError('Capture timing parameters must be finite and positive')
    serials = {c: getattr(config.cameras, c+'_serial') for c in ('global', 'wrist')}
    calibrations = calibration_inputs(fixed_calibration, config.calibration_file, serials)
    if output_dir is None:
        output_dir = Path('runs/occlusion_real')/('base_pair_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime(
            '%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    specs = {c: dict(name=c, serial=serials[c], enable_depth=True, color_width=config.cameras.width,
             color_height=config.cameras.height, color_fps=config.cameras.fps,
             depth_width=config.cameras.width, depth_height=config.cameras.height,
             depth_fps=config.cameras.fps) for c in serials}
    service = (service_factory or _default_service)(specs, root/'stream_cache')
    reader = (reader_factory or RobotReader)(config.arm)
    try:
        reader.start()
        service.start()
        time.sleep(warmup_s)
        initial = reader.window(time.monotonic())[-1]
        before = _validate_transform(reader.fk(initial['q']))
        offset = time.time()-time.monotonic()
        fresh_after = time.time()
        pair, delta = _choose_pair(service, pairing_timeout_s, 1., min_timestamp_s=fresh_after)
        if delta > max_skew_s:
            raise ValueError(f'Dual frame receive skew {delta:.4f}s exceeds {max_skew_s}s')
        end = max(time.monotonic()+.1, initial['received_at']+.7)
        states = reader.window(end)
        image_time = pair['wrist']['timestamp']-offset
        selected, stability = stationary_joint_window(states, initial['received_at'], end, image_time)
        at_image = _validate_transform(reader.fk(selected['q']))
        after = _validate_transform(reader.fk(states[-1]['q']))
        if abs((time.time()-time.monotonic())-offset) > .02:
            raise ValueError('Host wall/monotonic clock relationship changed during capture')
        report = save_pair(pair, root, delta, max_skew_s)
        robot_state = dict(samples=states, selected_joint_state=selected,
            begin_monotonic_s=initial['received_at'], end_monotonic_s=end,
            wrist_receive_monotonic_s=image_time, base_T_flange_before=before.tolist(),
            base_T_flange_after=after.tolist(), base_T_flange_at_wrist_capture=at_image.tolist(),
            stability=stability, timestamp_semantics='host_receive_not_sensor_exposure')
        transforms = {'global': np.asarray(calibrations['fixed']['base_T_camera']),
                      'wrist': at_image@np.asarray(calibrations['wrist']['flange_T_camera'])}
        frames = {}
        for camera in serials:
            frame = pair[camera]
            if frame['device_serial'] != serials[camera]:
                raise ValueError('Captured camera serial differs from calibration')
            frames[camera] = dict(device_serial=serials[camera], capture_timestamp_s=frame['timestamp'],
                color_frame_number=frame['color_frame_number'], depth_frame_number=frame['depth_frame_number'],
                depth_intrinsics=frame['depth_intrinsics'], base_T_camera=transforms[camera].tolist(),
                rgb_sha256=digest(root/camera/'color.png'), depth_sha256=digest(root/camera/'depth_m.npy'))
        geometry = dict(schema='robot_base_capture_geometry_v1', capture_group_id=report['capture_group_id'],
            reference_frame='fr3_link0', flange_frame='fr3_link8', translation_unit='m',
            matrix_convention='p_parent = T_parent_child @ p_child', camera_serials=serials,
            timestamp_domain='shared_host_unix_receive', validity='this_stationary_capture_group_only',
            max_frame_receive_skew_s=max_skew_s,
            calibrations=calibrations, operator_verification_note=verification_note,
            robot_state=robot_state, frames=frames, robot_motion_commanded=False,
            robot_control_initialized=False, robot_state_queried=True, accuracy_evaluated=False)
        validate_geometry(geometry)
        _write(root/'geometry.json', geometry)
        geometry_hash = digest(root/'geometry.json')
        for camera in serials:
            path = root/camera/'grasp_input.json'
            meta = json.loads(path.read_text())
            meta.update(reference_frame='fr3_link0', T_reference_from_camera_optical=transforms[camera].tolist(),
                        camera_pose_available=True, robot_state_queried=True,
                        geometry_file='../geometry.json', geometry_sha256=geometry_hash)
            _write(path, meta)
        report.update(schema='robot_base_rgbd_capture_v1', robot_state_queried=True,
                      robot_motion_commanded=False, camera_pose_available=True,
                      geometry_file='geometry.json', geometry_sha256=geometry_hash,
                      cross_view_inference_ready=True, accuracy_evaluated=False,
                      calibration_status={c: v['status'] for c, v in calibrations.items()},
                      operator_verification_note=verification_note)
        _write(root/'capture_report.json', report)
        print(json.dumps({'capture_dir': str(root), 'robot_motion_commanded': False,
                          'calibration_status': report['calibration_status'], 'frame_skew_s': delta}), flush=True)
        return root
    except BaseException as error:
        _write(root/'capture_error.json', {'error': str(error), 'robot_motion_commanded': False,
                                          'cross_view_inference_ready': False})
        # Even partial snapshots must not be mistaken for a completed pose-bound capture.
        path = root/'capture_report.json'
        if path.exists():
            failed = json.loads(path.read_text())
            failed.update(cross_view_inference_ready=False, status='capture_failed', robot_state_queried=True)
            _write(path, failed)
        raise
    finally:
        try:
            service.stop()
        finally:
            reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/hardware/fr3.yaml')
    parser.add_argument('--fixed-calibration', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verification-note', default='', help='Operator evidence; does not change raw quality status')
    parser.add_argument('--max-skew-s', type=float, default=.05,
                        help='Static-scene host receive pairing limit; does not imply exposure synchronization')
    parser.add_argument('--analyze', action='store_true', help='After capture, run SAM3 and offline fusion')
    parser.add_argument('--checkpoint', default='/home/seur/data/wpy/models/sam3/sam3.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fusion-config', type=Path, help='Optional multiview/planar JSON; defaults stay conservative')
    args = parser.parse_args()
    root = capture(args.config, args.fixed_calibration, args.output_dir, verification_note=args.verification_note,
                   max_skew_s=args.max_skew_s)
    if args.analyze:
        try:
            subprocess.run([sys.executable, '-u', '-m', 'occlusion_reasoning.real_single_view',
                '--capture-dir', str(root), '--output-dir', str(root/'analysis'),
                '--checkpoint', args.checkpoint, '--device', args.device], check=True)
            from .real_fusion import fuse
            result = fuse(root/'analysis', root/'geometry.json', root/'fusion',
                          config=json.loads(args.fusion_config.read_text()) if args.fusion_config else None)
            print(json.dumps({'output_dir': str(root), 'geometry_health': result['geometry_health']['status'],
                'accepted_associations': sum(a['status'] == 'accepted' for a in result['associations']),
                'measured_cross_view_edges': sum(e['relation'] == 'occludes_observed_surface'
                                                for e in result['relation_graph']['edges']),
                'robot_motion_commanded': False}), flush=True)
        except BaseException as error:
            _write(root/'pipeline_error.json', {'error': str(error), 'robot_motion_commanded': False})
            raise


if __name__ == '__main__':
    main()
