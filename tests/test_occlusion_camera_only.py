"""Camera-only boundaries, cleanup and local-frame inference without devices."""

import builtins
import json
import time

import cv2
import numpy as np
import pytest
import yaml

from occlusion_reasoning.camera_only import capture, save_pair, _choose_pair
from occlusion_reasoning.real_single_view import prepare_observation, save_view
from occlusion_reasoning.multiview import extract_multiview


def frame(name, timestamp=None):
    intrinsic = dict(width=12, height=10, fx=100., fy=100., ppx=6., ppy=5., coeffs=[0.]*5)
    depth = np.full((10, 12), 450, np.uint16)
    depth[0, 0] = 0
    return dict(color=np.full((10, 12, 3), 100, np.uint8), depth=depth,
                timestamp=time.time() if timestamp is None else timestamp,
                color_intrinsics=intrinsic, depth_intrinsics=intrinsic,
                depth_aligned_to_color=True, depth_unit_m_per_raw_unit=.001,
                device_serial=f"serial_{name}", color_frame_number=7, depth_frame_number=8,
                timestamp_domain="independent_hardware_clock")


def save_fixture(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    now = time.time()
    save_pair({name: frame(name, now) for name in ("global", "wrist")}, root, 0, .05)
    return root


def proposal():
    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:10] = True
    return dict(source_id="anonymous", mask=mask, confidence=.9, bbox_xyxy=[2, 2, 10, 8])


def test_wrist_only_capture_never_requests_fixed_or_robot_state(tmp_path, monkeypatch):
    import occlusion_reasoning.camera_only as module
    config = tmp_path/'wrist.yaml'
    config.write_text(yaml.safe_dump({'cameras': {'mode': 'wrist',
        'wrist_serial': 'serial_wrist', 'width': 12, 'height': 10, 'fps': 6}}))
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    requested = []
    class Service:
        stopped = False
        def start(self): pass
        def stop(self): self.stopped = True
        def get_latest_rgbd(self, name):
            assert name == 'wrist'
            requested.append(name)
            return frame(name)
    service = Service()
    def factory(specs, directory):
        assert tuple(specs) == ('wrist',)
        return service
    output = capture(config, tmp_path/'output', service_factory=factory)
    report = json.loads((output/'capture_report.json').read_text())
    assert list(report['cameras']) == ['wrist']
    assert report['pairing']['method'] == 'single_camera_no_pairing'
    assert not report['robot_state_queried'] and not report['robot_control_initialized']
    assert not (output/'global').exists()
    assert service.stopped and requested == ['wrist']


def test_raw_depth_and_units_are_preserved_without_external_pose(tmp_path):
    root = save_fixture(tmp_path)
    meta = json.loads((root/"global/grasp_input.json").read_text())
    assert meta["reference_frame"] is None
    assert meta["T_reference_from_camera_optical"] is None
    assert meta["timestamp_domain"] == "shared_host_unix_receive"
    assert meta["sensor_timestamp_domain"] == "independent_hardware_clock"
    raw = cv2.imread(str(root/"global/depth.png"), cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16 and raw[1, 1] == 450
    metric = np.load(root/"global/depth_m.npy")
    assert metric[1, 1] == pytest.approx(.45)
    assert metric[0, 0] == 0
    report = json.loads((root/"capture_report.json").read_text())
    assert not report["robot_control_initialized"] and not report["robot_state_queried"]
    assert not report["pairing"]["hardware_synchronized"]
    assert not report["cross_view_inference_ready"]


def test_host_receive_skew_is_not_claimed_as_hardware_sync(tmp_path):
    report = save_pair({n: frame(n) for n in ("global", "wrist")}, tmp_path, .12, .05)
    assert not report["pairing"]["passes_host_receive_skew_screen"]
    assert not report["pairing"]["exposure_time_synchronization_verified"]


def test_capture_owns_only_camera_service_and_releases_it(tmp_path, monkeypatch):
    config = tmp_path/"hardware.yaml"
    config.write_text(yaml.safe_dump({
        "arm": {"must_not_be_constructed": True}, "gripper": {"must_not_be_constructed": True},
        "cameras": {"global_serial": "serial_global", "wrist_serial": "serial_wrist",
                    "width": 12, "height": 10, "fps": 6}}))
    original_import = builtins.__import__
    def camera_only_import(name, *args, **kwargs):
        if name.split('.')[0] in {"hardware", "agents", "runtime", "skill_lib"}:
            raise AssertionError(f"Robot/Agent import forbidden: {name}")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", camera_only_import)
    class Service:
        stopped = False
        def start(self): pass
        def stop(self): self.stopped = True
        def get_latest_rgbd(self, name): return frame(name)
    service = Service()
    def factory(specs, directory):
        assert specs["global"]["enable_depth"] and specs["wrist"]["enable_depth"]
        assert specs["global"]["serial"] == "serial_global"
        return service
    output = capture(config, tmp_path/"captured", warmup_s=.001, service_factory=factory)
    assert service.stopped
    assert (output/"wrist/grasp_input.json").is_file()
    with pytest.raises(FileExistsError):
        capture(config, output, service_factory=factory)


@pytest.mark.parametrize("fail_start", [True, False])
def test_capture_errors_release_camera_service(tmp_path, fail_start):
    config = tmp_path/"hardware.yaml"
    config.write_text(yaml.safe_dump({"cameras": {
        "global_serial": "g", "wrist_serial": "w", "width": 12, "height": 10, "fps": 6}}))
    class Service:
        stopped = False
        def start(self):
            if fail_start: raise RuntimeError("simulated startup error")
        def get_latest_rgbd(self, name): raise RuntimeError("simulated capture error")
        def stop(self): self.stopped = True
    service = Service()
    with pytest.raises(RuntimeError, match="simulated"):
        capture(config, tmp_path/"failed", warmup_s=.001, service_factory=lambda *a: service)
    assert service.stopped
    assert (tmp_path/"failed/capture_error.json").is_file()


def test_stale_frames_are_not_accepted_as_a_fresh_capture():
    class Service:
        def get_latest_rgbd(self, name): return frame(name, time.time()-10)
    with pytest.raises(RuntimeError, match="fresh"):
        _choose_pair(Service(), timeout_s=.001, max_age_s=1)


def test_local_frame_planes_do_not_fabricate_cross_camera_calibration(tmp_path):
    root = save_fixture(tmp_path)
    global_packet = prepare_observation(root/"global", [proposal()])[:2]
    wrist_packet = prepare_observation(root/"wrist", [proposal()])[:2]
    assert global_packet[0]["frame"]["reference_frame"] == "global_optical_local"
    assert wrist_packet[0]["frame"]["reference_frame"] == "wrist_optical_local"
    with pytest.raises(ValueError, match="reference_frame"):
        extract_multiview([global_packet, wrist_packet])


def test_unrectified_lens_distortion_is_rejected(tmp_path):
    root = save_fixture(tmp_path)
    path = root/"global/grasp_input.json"
    meta = json.loads(path.read_text())
    meta["depth_intrinsics"]["coeffs"][0] = .01
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="distortion"):
        prepare_observation(root/"global", [proposal()])


def test_empty_segmentation_remains_unknown_and_exports_review_artifacts(tmp_path):
    root = save_fixture(tmp_path)
    report = save_view(root/"wrist", [], {"backend": "test_empty"}, tmp_path/"analysis")
    assert report["instance_hypotheses"] == 0
    assert report["possible_occlusion_pairs"] == 0
    assert not report["accuracy_evaluated"]
    packet = json.loads((tmp_path/"analysis/observation/scene_observation.json").read_text())
    assert any(h["kind"] == "unobserved_or_missed" for h in packet["target_candidates"]["hypotheses"])
    assert (tmp_path/"analysis/evidence.png").is_file()
