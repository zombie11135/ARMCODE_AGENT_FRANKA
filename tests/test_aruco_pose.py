"""Transform direction, lost-marker handling and camera ownership; no devices."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hardware import aruco_pose


def calibration():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.2, -.3, .5]).as_matrix()
    transform[:3, 3] = [.1, -.4, .9]
    return {'schema': 'fr3_eye_to_hand_result_v1', 'translation_unit': 'm',
            'matrix_convention': 'p_parent = T_parent_child @ p_child',
            'base_frame': 'fr3_link0', 'camera_frame': 'global_color_optical_frame',
            'base_T_camera': transform.tolist(), 'camera_serial': 'test-fixed-camera',
            'status': 'quality_failed', 'intrinsics': {'width': 640, 'height': 480, 'fps': 6}}


def test_base_marker_uses_forward_extrinsic_and_reports_marker_rotation():
    data = calibration()
    camera_marker = np.eye(4)
    camera_marker[:3, :3] = Rotation.from_euler('xyz', [2.8, .2, -.6]).as_matrix()
    camera_marker[:3, 3] = [.2, .1, .8]
    result = aruco_pose.marker_base_pose(data, camera_marker)
    base_camera = np.array(data['base_T_camera'])
    expected_center = base_camera[:3, :3] @ camera_marker[:3, 3] + base_camera[:3, 3]
    np.testing.assert_allclose(result['position_base_mm'], expected_center*1000)
    np.testing.assert_allclose(Rotation.from_euler('xyz', result['rpy_xyz_deg'], degrees=True).as_matrix(),
                               base_camera[:3, :3] @ camera_marker[:3, :3])
    np.testing.assert_allclose(Rotation.from_quat(result['quaternion_xyzw']).as_matrix(),
                               np.array(result['base_T_marker'])[:3, :3])


@pytest.mark.parametrize('field,value', [('base_frame', 'world'), ('translation_unit', 'mm'),
                                        ('camera_frame', 'depth_optical_frame')])
def test_wrong_coordinate_contract_rejected(field, value):
    data = calibration()
    data[field] = value
    with pytest.raises(ValueError):
        aruco_pose.marker_base_pose(data, np.eye(4))


def fake_camera(monkeypatch, images, *, failure=None):
    clock = SimpleNamespace(value=0.)
    camera = Mock()
    camera.metadata = {'width': 640, 'height': 480}
    def frame():
        clock.value += 1
        if not images:
            raise KeyboardInterrupt
        return images.pop(0), {'frame_number': int(clock.value), 'received_at': clock.value}
    camera.frame.side_effect = frame
    if failure:
        camera.start.side_effect = failure
    constructor = Mock(return_value=camera)
    monkeypatch.setattr(aruco_pose, 'ColorCamera', constructor)
    monkeypatch.setattr(aruco_pose.time, 'monotonic', lambda: clock.value)
    monkeypatch.setattr(aruco_pose, '_save_images', lambda output, image, *args: image)
    return camera, constructor


def write_calibration(tmp_path):
    path = tmp_path/'calibration.json'
    path.write_text(json.dumps(calibration()))
    return path


def test_continuous_loss_replaces_latest_pose_instead_of_reusing_old_xyz(tmp_path, monkeypatch):
    path = write_calibration(tmp_path)
    camera, constructor = fake_camera(monkeypatch, [np.zeros((2, 2)), np.ones((2, 2))])
    def detect(image, *args):
        if image.any():
            raise ValueError('没有检测到 ArUco')
        return {'camera_T_marker': np.eye(4).tolist(), 'reprojection_rms_px': .1}
    monkeypatch.setattr(aruco_pose, 'detect_marker', detect)
    # No RobotReader is permitted even though it lives in a shared camera module.
    monkeypatch.setattr('hardware.eye_to_hand_capture.RobotReader.start',
                        Mock(side_effect=AssertionError('Must not connect to ROS')))
    output = tmp_path/'out'
    assert aruco_pose.main(['--calibration', str(path), '--continuous', '--output-dir', str(output)]) == 130
    rows = [json.loads(line) for line in (output/'poses.jsonl').read_text().splitlines()]
    assert [r['valid'] for r in rows] == [True, False]
    latest = json.loads((output/'latest.json').read_text())
    assert latest['status'] == 'marker_not_observed'
    assert 'position_base_mm' not in latest and 'base_T_marker' not in latest
    assert all(not r['robot_control_initialized'] and not r['robot_state_queried'] for r in rows)
    constructor.assert_called_once_with('test-fixed-camera', 640, 480, 6)
    camera.close.assert_called_once()


def test_start_failure_records_error_and_closes_owned_camera(tmp_path, monkeypatch):
    path = write_calibration(tmp_path)
    camera, _ = fake_camera(monkeypatch, [], failure=RuntimeError('Device busy'))
    output = tmp_path/'out'
    assert aruco_pose.main(['--calibration', str(path), '--output-dir', str(output)]) == 1
    camera.close.assert_called_once()
    latest = json.loads((output/'latest.json').read_text())
    assert latest['status'] == 'error' and latest['valid'] is False
    assert 'position_base_mm' not in latest


def test_single_mode_stops_after_first_valid_capture(tmp_path, monkeypatch):
    path = write_calibration(tmp_path)
    camera, _ = fake_camera(monkeypatch, [np.zeros((2, 2)), np.zeros((2, 2))])
    monkeypatch.setattr(aruco_pose, 'detect_marker', lambda *args: {
        'camera_T_marker': np.eye(4).tolist(), 'reprojection_rms_px': .1})
    output = tmp_path/'out'
    assert aruco_pose.main(['--calibration', str(path), '--output-dir', str(output)]) == 0
    camera.frame.assert_called_once()
    assert json.loads((output/'latest.json').read_text())['valid']
    camera.close.assert_called_once()
