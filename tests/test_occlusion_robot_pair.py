"""Capture-time base binding with mock cameras and passive FK; no real devices."""
import json
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from occlusion_reasoning import robot_pair
from occlusion_reasoning.real_single_view import prepare_observation, save_view
from occlusion_reasoning.real_fusion import fuse


def matrix(xyz, rpy=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def inputs(tmp_path):
    fixed = tmp_path/'fixed.json'
    fixed.write_text(json.dumps({'schema': 'fr3_eye_to_hand_result_v1', 'base_frame': 'fr3_link0',
        'flange_frame': 'fr3_link8', 'camera_frame': 'global_color_optical_frame',
        'translation_unit': 'm', 'matrix_convention': 'p_parent = T_parent_child @ p_child',
        'camera_serial': 'g', 'status': 'quality_failed',
        'base_T_camera': matrix([.1, -.4, .9], [.2, -.3, .4]).tolist()}))
    wrist = tmp_path/'wrist.yaml'
    wrist.write_text(yaml.safe_dump({'schema_version': 1, 'status': 'geometry_preview_ready',
        'translation_unit': 'm', 'matrix_convention': 'p_parent = T_parent_child @ p_child',
        'cameras': {'wrist': {'serial': 'w', 'mounting': 'eye_in_hand', 'image_stream': 'color',
            'depth_aligned_to': 'color', 'extrinsic': {'parent_frame': 'fr3_link8',
                'child_frame': 'wrist_color_optical_frame',
                'matrix': matrix([.02, -.07, .03], [.01, .02, 3.]).tolist()}}}}))
    config = tmp_path/'hardware.yaml'
    config.write_text(yaml.safe_dump({'arm': {'stack': 'real', 'workspace': '/tmp/test_no_ros'},
        'cameras': {'global_serial': 'g', 'wrist_serial': 'w', 'mode': 'wrist',
                    'width': 12, 'height': 10, 'fps': 6}, 'calibration_file': str(wrist)}))
    return config, fixed, wrist


class Service:
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True
    def get_latest_rgbd(self, camera):
        intrinsic = dict(width=12, height=10, fx=100., fy=100., ppx=6., ppy=5., coeffs=[0.]*5)
        return dict(color=np.full((10, 12, 3), 100, np.uint8), depth=np.full((10, 12), 500, np.uint16),
            timestamp=time.time(), color_intrinsics=intrinsic, depth_intrinsics=intrinsic,
            depth_aligned_to_color=True, depth_unit_m_per_raw_unit=.001,
            device_serial='g' if camera == 'global' else 'w', color_frame_number=7, depth_frame_number=8)


class Reader:
    def __init__(self, pose=None):
        self.pose = matrix([.3, .1, .5], [3., .1, -.5]) if pose is None else pose
        self.calls = 0
    def start(self):
        self.started = True
    def window(self, after):
        if not hasattr(self, 'begin'):
            self.begin = after-.2
        return [dict(received_at=float(t), q=[.1]*7, dq=[0.]*7)
                for t in np.arange(self.begin, after+.06, .025)]
    def fk(self, joints):
        self.calls += 1
        return self.pose
    def close(self):
        self.closed = True


def captured(tmp_path, *, reader=None, service=None):
    config, fixed, wrist = inputs(tmp_path)
    service, reader = service or Service(), reader or Reader()
    root = robot_pair.capture(config, fixed, tmp_path/'capture', verification_note='operator_checked_marker',
            warmup_s=.001, service_factory=lambda *args: service, reader_factory=lambda *args: reader)
    return root, service, reader


def proposals():
    return [dict(source_id='anonymous', mask=np.ones((10, 12), bool),
                 bbox_xyxy=[0, 0, 12, 10], confidence=.9)]


def test_capture_composes_flange_wrist_keeps_raw_status_and_releases_readers(tmp_path):
    root, service, reader = captured(tmp_path)
    geometry = json.loads((root/'geometry.json').read_text())
    transforms = robot_pair.validate_geometry(geometry)
    wrist = yaml.safe_load((tmp_path/'wrist.yaml').read_text())
    expected = reader.pose@np.array(wrist['cameras']['wrist']['extrinsic']['matrix'])
    np.testing.assert_allclose(transforms['wrist'], expected, atol=1e-12)
    assert geometry['calibrations']['fixed']['status'] == 'quality_failed'
    assert geometry['operator_verification_note'] == 'operator_checked_marker'
    assert not geometry['robot_motion_commanded']
    report = json.loads((root/'capture_report.json').read_text())
    assert report['robot_state_queried'] and not report['robot_control_initialized']
    assert service.stopped and reader.closed and reader.calls == 3
    packet, arrays, _ = prepare_observation(root/'wrist', proposals())
    assert packet['frame']['reference_frame'] == 'fr3_link0'
    node = packet['relation_graph']['nodes'][0]
    xyz = arrays[node['observed_points_camera_key']]
    np.testing.assert_allclose(arrays[node['observed_points_reference_key']],
                               xyz@expected[:3, :3].T+expected[:3, 3])


def test_offline_fusion_reads_matching_capture_and_preserves_calibration_evidence(tmp_path):
    root, _, _ = captured(tmp_path)
    analysis = tmp_path/'analysis'
    for camera in ('global', 'wrist'):
        save_view(root/camera, proposals(), {}, analysis/camera)
    result = fuse(analysis, root/'geometry.json', tmp_path/'fusion')
    assert result['real_camera_fusion']['coordinate_reference'] == 'fr3_link0'
    assert result['real_camera_fusion']['capture_robot_state_queried']
    assert not result['real_camera_fusion']['robot_state_queried']  # Offline inference.
    assert result['real_camera_fusion']['calibration_status']['fixed'] == 'quality_failed'
    assert (tmp_path/'fusion/evidence.png').is_file()
    assert (tmp_path/'fusion/alignment.png').is_file()
    assert (tmp_path/'fusion/alignment_report.json').is_file()


@pytest.mark.parametrize('fault', ['group', 'image', 'intrinsic', 'geometry', 'camera_serial'])
def test_snapshot_cannot_be_rebound_to_unrelated_data(tmp_path, fault):
    root, _, _ = captured(tmp_path)
    path = root/'wrist/grasp_input.json'
    metadata = json.loads(path.read_text())
    if fault == 'group': metadata['capture_group_id'] = 'old-capture'
    if fault == 'intrinsic': metadata['depth_intrinsics']['fx'] += 1
    if fault == 'camera_serial': metadata['device_serial'] = 'wrong'
    if fault == 'image': np.save(root/'wrist/depth_m.npy', np.full((10, 12), .7))
    if fault == 'geometry':
        geometry = json.loads((root/'geometry.json').read_text())
        geometry['robot_state']['base_T_flange_at_wrist_capture'][0][3] += .1
        (root/'geometry.json').write_text(json.dumps(geometry))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        prepare_observation(root/'wrist', proposals())


def test_wrong_static_wrist_pose_composition_rejected_even_if_rigid(tmp_path):
    root, _, _ = captured(tmp_path)
    geometry = json.loads((root/'geometry.json').read_text())
    geometry['frames']['wrist']['base_T_camera'][0][3] += .1
    with pytest.raises(ValueError, match='composition'):
        robot_pair.validate_geometry(geometry)


def test_kinematic_binding_changes_after_moving_wrist_not_fixed_camera(tmp_path):
    first, _, _ = captured(tmp_path)
    a = json.loads((first/'geometry.json').read_text())
    second_dir = tmp_path/'next'
    second_dir.mkdir()
    second, _, _ = captured(second_dir, reader=Reader(matrix([.4, .1, .5], [3., .1, -.5])))
    b = json.loads((second/'geometry.json').read_text())
    np.testing.assert_allclose(a['frames']['global']['base_T_camera'], b['frames']['global']['base_T_camera'])
    assert not np.allclose(a['frames']['wrist']['base_T_camera'], b['frames']['wrist']['base_T_camera'])
    with pytest.raises(ValueError):
        robot_pair.bind_base_pair([prepare_observation(first/c, proposals())[:2] for c in ('global', 'wrist')],
                                  b, {'global': 'g', 'wrist': 'w'})


@pytest.mark.parametrize('fault', ['moving', 'gap', 'not_bracketed', 'nan'])
def test_joint_window_requires_continuous_stationary_feedback(fault):
    states = [dict(received_at=t, q=[.1]*7, dq=[0.]*7) for t in np.arange(0, 1.01, .05)]
    if fault == 'moving': states[8]['dq'][0] = .02
    if fault == 'gap': states = states[:5]+states[15:]
    if fault == 'not_bracketed': states = states[8:]
    if fault == 'nan': states[8]['q'][0] = float('nan')
    with pytest.raises(ValueError):
        robot_pair.stationary_joint_window(states, .2, .8, .5)


def test_device_busy_and_reader_failures_release_both_owned_resources(tmp_path):
    config, fixed, _ = inputs(tmp_path)
    service, reader = Service(), Reader()
    def fail(): raise RuntimeError('Device busy')
    service.start = fail
    with pytest.raises(RuntimeError, match='busy'):
        robot_pair.capture(config, fixed, tmp_path/'capture', service_factory=lambda *args: service,
                           reader_factory=lambda *args: reader)
    assert service.stopped and reader.closed
    report = json.loads((tmp_path/'capture/capture_error.json').read_text())
    assert report['cross_view_inference_ready'] is False


def test_fusion_rejects_npz_depth_tampering_even_when_original_files_unchanged(tmp_path):
    root, _, _ = captured(tmp_path)
    analysis = tmp_path/'analysis'
    for camera in ('global', 'wrist'):
        save_view(root/camera, proposals(), {}, analysis/camera)
    archive = analysis/'wrist/observation/observation_arrays.npz'
    with np.load(archive) as original:
        arrays = {key: original[key] for key in original.files}
    arrays['depth_m'] += .1
    np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError, match='depth arrays differ'):
        fuse(analysis, root/'geometry.json', tmp_path/'fusion')


def test_frame_skew_cannot_be_replaced_with_unrelated_old_image(tmp_path):
    root, _, _ = captured(tmp_path)
    geometry = json.loads((root/'geometry.json').read_text())
    geometry['frames']['wrist']['capture_timestamp_s'] += 1
    with pytest.raises(ValueError, match='pairing limit'):
        robot_pair.validate_geometry(geometry)


def test_capture_failure_does_not_start_analysis(tmp_path, monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(robot_pair.sys, 'argv', ['robot_pair', '--fixed-calibration', 'unused', '--analyze'])
    monkeypatch.setattr(robot_pair, 'capture', Mock(side_effect=RuntimeError('not stationary')))
    analysis = Mock()
    monkeypatch.setattr(robot_pair.subprocess, 'run', analysis)
    with pytest.raises(RuntimeError, match='not stationary'):
        robot_pair.main()
    analysis.assert_not_called()


def test_one_command_analyzes_only_after_camera_and_reader_are_closed(tmp_path, monkeypatch):
    root, service, reader = captured(tmp_path)
    monkeypatch.setattr(robot_pair, 'capture', lambda *args, **kwargs: root)
    monkeypatch.setattr(robot_pair.sys, 'argv', ['robot_pair', '--fixed-calibration', 'unused', '--analyze'])
    def analyze(command, check):
        assert check and service.stopped and reader.closed
        assert command[command.index('-m')+1] == 'occlusion_reasoning.real_single_view'
        for camera in ('global', 'wrist'):
            save_view(root/camera, proposals(), {}, root/'analysis'/camera)
    monkeypatch.setattr(robot_pair.subprocess, 'run', analyze)
    robot_pair.main()
    assert (root/'fusion/multiview_relations.json').is_file()
    assert (root/'fusion/alignment.png').is_file()
