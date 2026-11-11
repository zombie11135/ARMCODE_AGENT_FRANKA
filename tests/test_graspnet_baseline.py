"""Backend routing and shared filters without models, devices or CUDA."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hardware.config import load_config
from hardware.graspnet_worker_common import run
from services.graspnet_baseline_service import GraspNetBaselineService, validate_graspnet_paths


def test_baseline_profile_and_factory(monkeypatch, tmp_path):
    from services import grasp_service, graspnet_baseline_service
    from services.runtime_service_cache import clear_service_cache
    path = tmp_path / 'profile.yaml'
    path.write_text('perception:\n  grasp_backend: graspnet_baseline\n')
    assert load_config(path).perception.grasp_backend == 'graspnet_baseline'
    constructor = Mock()
    monkeypatch.setattr(graspnet_baseline_service, 'GraspNetBaselineService', constructor)
    clear_service_cache()
    try:
        assert grasp_service.build_grasp_service(SimpleNamespace(hardware_config_path=str(path))) is constructor.return_value
        constructor.assert_called_once_with(str(path))
    finally:
        clear_service_cache()
    path.write_text('perception:\n  grasp_backend: invalid\n')
    with pytest.raises(ValueError, match='grasp_backend'):
        load_config(path)


def test_baseline_requests_use_own_checkpoint_and_interpreter(monkeypatch, tmp_path):
    from services import graspnet_baseline_service as module
    p = SimpleNamespace(graspnet_root='/models/baseline', graspnet_ckpt='/weights/baseline.tar',
                        graspnet_python='/env/baseline/bin/python', graspnet_max_gripper_width_m=.08)
    service = GraspNetBaselineService.__new__(GraspNetBaselineService)
    service.config = SimpleNamespace(perception=p)
    service.last_run = tmp_path
    assert service._model_request() == dict(graspnet_root=p.graspnet_root,
        checkpoint=p.graspnet_ckpt, backend='graspnet_baseline', num_point=20000, max_gripper_width=.08)
    call = Mock()
    monkeypatch.setattr(module.subprocess, 'run', call)
    service._run_inference(tmp_path/'request.json', None)
    assert call.call_args.args[0][:4] == [p.graspnet_python, '-u', '-m', 'hardware.graspnet_baseline_worker']
    assert call.call_args.kwargs['check'] is True
    service.visualize_last_grasps_by_indices([0])
    assert call.call_args.args[0] == [p.graspnet_python, '-m', 'hardware.graspnet_baseline_worker', '--view', str(tmp_path)]
    with pytest.raises(ValueError):
        service.visualize_last_grasps_by_indices([1])


def test_missing_baseline_model_fails_before_device_start(tmp_path):
    p = SimpleNamespace(graspnet_root=str(tmp_path), graspnet_ckpt=str(tmp_path/'weights.tar'),
                        graspnet_python=sys.executable)
    with pytest.raises(FileNotFoundError, match='GraspNet-baseline'):
        validate_graspnet_paths(p)


class Group:
    def __init__(self, rows):
        self.grasp_group_array = np.asarray(rows)
    def __len__(self):
        return len(self.grasp_group_array)
    def __getitem__(self, item):
        return Group(self.grasp_group_array[item])
    @property
    def scores(self):
        return self.grasp_group_array[:, 0]
    @property
    def widths(self):
        return self.grasp_group_array[:, 1]
    @property
    def translations(self):
        return self.grasp_group_array[:, 13:16]
    def save_npy(self, path):
        np.save(path, self.grasp_group_array)


@pytest.mark.parametrize('reject_all', [False, True])
def test_shared_filter_keeps_full_collision_cloud_and_never_fills_empty_pool(monkeypatch, tmp_path, reject_all):
    monkeypatch.setitem(sys.modules, 'graspnetAPI', SimpleNamespace(GraspGroup=Group))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(manual_seed=lambda seed: None))
    capture = tmp_path/'capture'; capture.mkdir()
    intr = dict(width=4, height=4, fx=1., fy=1., ppx=1., ppy=1., coeffs=[0.]*5)
    (capture/'capture.json').write_text(json.dumps(dict(color_intrinsics=intr, workspace={'roi_xyxy': None}, frame='test')))
    checkpoint = tmp_path/'checkpoint'; checkpoint.write_bytes(b'test')
    mask = np.zeros((4, 4), bool); mask[1, 1] = True
    np.save(tmp_path/'mask.npy', mask)
    row = [.5, .05, .02, .02, *np.eye(3).ravel(), 0., 0., 1., -1.]
    raw = np.tile(row, (7, 1))
    raw[1, 1] = .09  # Wider than 2F-85.
    raw[2, 13] = 9.  # Outside image ROI.
    raw[3, 13] = 1.  # Inside ROI, outside target mask.
    raw[4, 15] = .004  # Below configured table margin.
    raw[5, 0], raw[6, 0] = .6, .7  # Collision and empty grasp.
    full_cloud = np.array([[x, y, 1. + z] for x in (-.005, 0., .005)
                           for y in (-.015, -.005, .005, .015)
                           for z in (-.004, .004)] + [[5., 5., 1.]])
    class Detector:
        finger_length = .06
        def __init__(self, points, voxel_size):
            assert points is full_cloud  # Even out-of-ROI obstacles are retained.
            assert voxel_size == .005
        def detect(self, group, **kwargs):
            assert kwargs == dict(approach_dist=.08, collision_thresh=.01,
                                  return_empty_grasp=True, empty_thresh=.01, return_ious=True)
            return ((np.ones(len(group), bool) if reject_all else group.scores == .6), group.scores == .7, np.zeros((5, len(group))))
    model = SimpleNamespace(read_cloud=lambda args: (full_cloud, full_cloud, {}),
        infer=lambda args, points: (raw, {}), ModelFreeCollisionDetector=Detector)
    request = dict(capture_dir=str(capture), output_dir=str(tmp_path/'out'), checkpoint=str(checkpoint),
        roi=[0, 0, 4, 4], target_mask_path=str(tmp_path/'mask.npy'), base_T_camera=np.eye(4).tolist(),
        table_z_m=0., export_filtered_candidates=True, backend='graspnet_baseline')
    run(request, model)
    report = json.loads((tmp_path/'out/report.json').read_text())
    assert list(report['counts'].values()) == [7, 6, 5, 4, 3, *([0 if reject_all else 1] * 3)]
    result = np.load(tmp_path/'out/filtered_candidates.npy')
    assert result.shape == (0 if reject_all else 1, 17)
    if not reject_all:
        np.testing.assert_array_equal(result, raw[:1])
    assert report['collision_breakdown'] == dict(collision=3 if reject_all else 1, empty=1, both=1 if reject_all else 0)
    np.testing.assert_array_equal(np.load(tmp_path/'out/grasps_raw.npy'), raw)


@pytest.mark.parametrize('value', [0, -.01, .086, float('nan'), float('inf'), True, '0.085'])
def test_decoder_rejects_invalid_physical_width(value):
    from hardware.graspnet_baseline_worker import decode_with_width_limit
    module = SimpleNamespace(GRASP_MAX_WIDTH=.1, pred_decode=Mock())
    with pytest.raises(ValueError):
        decode_with_width_limit(module, {}, value)
    module.pred_decode.assert_not_called()
    assert module.GRASP_MAX_WIDTH == .1


@pytest.mark.parametrize('failure', [False, True])
def test_decoder_limit_is_applied_before_decode_and_restored(failure):
    from hardware.graspnet_baseline_worker import decode_with_width_limit
    endpoints = object()
    module = SimpleNamespace(GRASP_MAX_WIDTH=.1)
    def decode(value):
        assert value is endpoints
        assert module.GRASP_MAX_WIDTH == .085
        if failure:
            raise RuntimeError('decode failed')
        return np.minimum(np.array([.04, .09, .1]), module.GRASP_MAX_WIDTH)
    module.pred_decode = decode
    if failure:
        with pytest.raises(RuntimeError, match='decode failed'):
            decode_with_width_limit(module, endpoints, .085)
    else:
        np.testing.assert_array_equal(decode_with_width_limit(module, endpoints, .085), [.04, .085, .085])
    assert module.GRASP_MAX_WIDTH == .1


@pytest.mark.parametrize('value', ['0', '-0.01', '0.1', '.nan', '.inf', 'true'])
def test_profile_rejects_invalid_baseline_width(tmp_path, value):
    p = tmp_path/'bad-width.yaml'
    p.write_text('perception:\n  graspnet_max_gripper_width_m: '+value+'\n')
    with pytest.raises(ValueError, match='graspnet_max_gripper_width_m'):
        load_config(p)
