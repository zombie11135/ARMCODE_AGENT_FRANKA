"""Fixed-camera ArUco pose in fr3_link0; camera only, no robot/ROS connection."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .eye_to_hand import atomic_json, detect_marker
from .eye_to_hand_capture import ColorCamera, create_detector
from .transforms import rigid_matrix


def marker_base_pose(calibration, camera_marker):
    """Compose base <- camera <- marker, never invert the supplied base_T_camera."""
    if (calibration.get('schema') != 'fr3_eye_to_hand_result_v1'
            or calibration.get('translation_unit') != 'm'
            or calibration.get('matrix_convention') != 'p_parent = T_parent_child @ p_child'
            or calibration.get('base_frame') != 'fr3_link0'
            or calibration.get('camera_frame') != 'global_color_optical_frame'):
        raise ValueError('需要米制 fr3_link0 <- global_color_optical_frame 眼在手外结果')
    pose = (rigid_matrix(calibration['base_T_camera'], 'base_T_camera')
            @ rigid_matrix(camera_marker, 'camera_T_marker'))
    rotation = Rotation.from_matrix(pose[:3, :3])
    return {'base_T_marker': pose.tolist(), 'position_base_mm': (pose[:3, 3]*1000).tolist(),
            'rpy_xyz_deg': rotation.as_euler('xyz', degrees=True).tolist(),
            'quaternion_xyzw': rotation.as_quat().tolist()}


def _now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')


def _save_images(output, image, detection, result, camera):
    annotated = image.copy()
    corners = np.asarray(detection['corners_px'], np.float32).reshape(1, 4, 2)
    cv2.aruco.drawDetectedMarkers(annotated, [corners], np.array([[result['board']['marker_id']]], np.int32))
    pose = np.asarray(detection['camera_T_marker'])
    cv2.drawFrameAxes(annotated, camera.k, camera.d, cv2.Rodrigues(pose[:3, :3])[0],
                     pose[:3, 3], result['board']['marker_length_m']/2)
    labels = [result['observed_at'],
              'fr3_link0 XYZ mm: '+', '.join(f'{x:.2f}' for x in result['position_base_mm']),
              'RPY deg: '+', '.join(f'{x:.2f}' for x in result['rpy_xyz_deg']),
              'Extrinsic: '+result['calibration_status']]
    for row, label in enumerate(labels):
        origin = (8, 23+row*23)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    # Images are explicitly the last VALID frame. latest.json may report loss.
    for name, pixels in [('last_valid_color.png', image), ('last_valid_pose.png', annotated)]:
        if not cv2.imwrite(str(output/name), pixels):
            raise IOError(f'无法保存 {output/name}')
    return annotated


def run(args):
    for name in ('interval', 'timeout', 'marker_length_mm', 'max_reprojection_px'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须为正有限数')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError('duration 必须为正有限数')
    path = Path(args.calibration).expanduser().resolve()
    raw = path.read_bytes()
    calibration = json.loads(raw)
    marker_base_pose(calibration, np.eye(4))  # Check coordinates before opening the camera.
    if not isinstance(calibration.get('camera_serial'), str) or not calibration['camera_serial']:
        raise ValueError('标定文件缺少固定相机序列号')
    detector, dictionary = create_detector(args.dictionary)
    if not 0 <= args.marker_id < len(dictionary.bytesList):
        raise ValueError('marker-id 不在所选字典中')
    intrinsic = calibration['intrinsics']
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              Path('runs/eye_to_hand')/('marker_pose_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime(
                  '%Y%m%d-%H%M%S')+'_'+uuid.uuid4().hex[:6]))
    output.mkdir(parents=True, exist_ok=False)
    session = {'schema': 'fixed_camera_marker_pose_v1', 'calibration_file': str(path),
               'calibration_sha256': hashlib.sha256(raw).hexdigest(),
               'calibration_status': calibration.get('status', 'unspecified'),
               'base_frame': calibration['base_frame'], 'camera_frame': calibration['camera_frame'],
               'camera_serial': calibration['camera_serial'],
               'board': {'dictionary': args.dictionary, 'marker_id': args.marker_id,
                         'marker_length_m': args.marker_length_mm/1000,
                         'frame': 'marker_center_x_right_y_up_z_out_of_print'},
               'translation_unit': 'm', 'display_translation_unit': 'mm',
               'matrix_convention': 'p_parent = T_parent_child @ p_child',
               'rpy_convention': 'fixed_axis_XYZ; R = Rz(yaw) @ Ry(pitch) @ Rx(roll)',
               'robot_control_initialized': False, 'robot_state_queried': False,
               'independent_accuracy_verified': False}
    camera = ColorCamera(session['camera_serial'], intrinsic['width'], intrinsic['height'],
                         intrinsic.get('fps', 6))
    print('输出目录：'+str(output.resolve()), flush=True)
    print('固定相机 '+session['camera_serial']+'；基座系 fr3_link0；码中心 XYZ(mm)、RPY(度)。', flush=True)
    if session['calibration_status'] != 'validated':
        print('外参状态 '+session['calibration_status']+'：本命令用于人工核对，未通过精度认证。', flush=True)
    code, valid_count, total = 0, 0, 0
    try:
        camera.start()
        session['live_intrinsics'] = camera.metadata
        atomic_json(output/'session.json', session)
        started, last_report = time.monotonic(), -float('inf')
        with (output/'poses.jsonl').open('a', encoding='utf-8') as stream:
            while True:
                elapsed = time.monotonic()-started
                limit = args.duration if args.continuous else args.timeout
                if limit is not None and elapsed >= limit:
                    code = 0 if valid_count else 1
                    break
                image, metadata = camera.frame()
                current = time.monotonic()
                if current-last_report < args.interval:
                    continue  # Drain frames continuously; never report a queued old frame later.
                last_report = current
                total += 1
                result = {**session, 'sample_index': total, 'observed_at': _now(), 'capture': metadata}
                annotated = image
                try:
                    detection = detect_marker(image, detector, session['board'], camera.k, camera.d,
                                              args.max_reprojection_px)
                except ValueError as error:
                    result.update(status='marker_not_observed', valid=False, error=str(error))
                    print(f"[{result['observed_at']}] 无有效位姿：{error}", flush=True)
                else:
                    result.update(status='observed', valid=True, **detection,
                                  **marker_base_pose(calibration, detection['camera_T_marker']))
                    annotated = _save_images(output, image, detection, result, camera)
                    valid_count += 1
                    xyz = ', '.join(f'{v:.2f}' for v in result['position_base_mm'])
                    rpy = ', '.join(f'{v:.2f}' for v in result['rpy_xyz_deg'])
                    print(f"[{result['observed_at']}] ID={args.marker_id} XYZ(mm)=({xyz}) "
                          f"RPY(deg)=({rpy}) reproj={result['reprojection_rms_px']:.3f}px", flush=True)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                # A missed marker replaces the live result with an invalid record, never stale XYZ.
                atomic_json(output/'latest.json', result)
                if args.display:
                    cv2.imshow('Fixed camera ArUco pose', annotated)
                    if cv2.waitKey(1) & 0xff in (27, ord('q')):
                        break
                if result['valid'] and not args.continuous:
                    break
    except KeyboardInterrupt:
        code = 130
        print('\n检测已停止。', flush=True)
    except Exception as error:
        atomic_json(output/'latest.json', {**session, 'status': 'error', 'valid': False,
                                         'observed_at': _now(), 'error': str(error)})
        raise
    finally:
        camera.close()
        if args.display:
            cv2.destroyAllWindows()
        atomic_json(output/'summary.json', {'reported_frames': total, 'valid_frames': valid_count,
                    'ended_at': _now(), 'robot_control_initialized': False,
                    'robot_state_queried': False})
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True, help='眼在手外 result.json')
    parser.add_argument('--dictionary', default='DICT_4X4_50')
    parser.add_argument('--marker-id', type=int, default=20)
    parser.add_argument('--marker-length-mm', type=float, default=95,
                        help='黑色码区外边长，不含白边；2026-10-22 实测默认 95 mm')
    parser.add_argument('--continuous', action='store_true', help='持续检测，Ctrl-C 退出')
    parser.add_argument('--interval', type=float, default=.5, help='输出最小间隔，秒')
    parser.add_argument('--duration', type=float, help='连续模式的运行时长，秒；默认不限')
    parser.add_argument('--timeout', type=float, default=15, help='单次模式等待有效位姿的时长，秒')
    parser.add_argument('--max-reprojection-px', type=float, default=1.5)
    parser.add_argument('--display', action='store_true', help='显示画面，q/Esc 退出')
    parser.add_argument('--output-dir', help='新建保存目录，必须尚不存在')
    args = parser.parse_args(argv)
    if args.duration is not None and not args.continuous:
        parser.error('--duration 与 --continuous 一起使用')
    try:
        return run(args)
    except (ValueError, KeyError, RuntimeError, OSError, cv2.error) as error:
        print(f'Stopped: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
