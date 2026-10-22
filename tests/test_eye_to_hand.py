"""Synthetic calibration/detection, timing guards and lifecycle; no devices."""
import copy
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hardware.eye_to_hand import (detect_marker, difference, inverse, reject_duplicate,
                                  solve_dataset, stationary_sample, transform)
from hardware.eye_to_hand_capture import (ColorCamera, RobotReader,
                                         create_detector, main, opencv_intrinsics, solve_file)


def dataset(seed=41, noise=False):
    rng = np.random.default_rng(seed)
    b_c = transform(Rotation.from_euler("xyz", [2.8, .1, -.3]).as_matrix(), [.3, -.1, 1.1])
    f_m = transform(Rotation.from_euler("xyz", [.2, -.3, .1]).as_matrix(), [.04, .03, .17])
    samples = []
    for i in range(25):
        b_f = transform(Rotation.from_euler("xyz", [3., .1, -.4] + rng.normal(0, .35, 3)).as_matrix(),
                        [.35, 0, .45] + rng.uniform(-.13, .13, 3))
        c_m = inverse(b_c) @ b_f @ f_m
        if noise:
            c_m = c_m @ transform(Rotation.from_rotvec(rng.normal(0, .0008, 3)).as_matrix(),
                                  rng.normal(0, .0002, 3))
        samples.append({"id": str(i), "role": "calibration" if i < 20 else "validation",
                        "base_T_flange": b_f.tolist(), "camera_T_marker": c_m.tolist()})
    return {"schema": "fr3_eye_to_hand_samples_v1", "translation_unit": "m",
            "base_frame": "fr3_link0", "flange_frame": "fr3_link8",
            "camera_frame": "global_color_optical_frame", "camera_serial": "346522072096",
            "board": {"kind": "aruco", "marker_id": 20, "marker_length_m": .1},
            "intrinsics": {}, "samples": samples}, b_c, f_m


@pytest.mark.parametrize("seed", [4, 17, 42])
def test_recovers_eye_to_hand_direction_and_unknown_marker_mount(seed):
    data, b_c, f_m = dataset(seed)
    result = solve_dataset(data)
    assert result["status"] == "validated"
    np.testing.assert_allclose(result["base_T_camera"], b_c, atol=1e-9)
    np.testing.assert_allclose(result["camera_T_base"], inverse(b_c), atol=1e-9)
    np.testing.assert_allclose(result["flange_T_marker"], f_m, atol=1e-9)
    assert result["cross_validation"]["count"] == 5
    assert result["training"]["count"] == 20
    assert result["validation"]["count"] == 5


def test_noisy_recovery_and_independent_validation_do_not_leak_into_fit():
    data, b_c, _ = dataset(noise=True)
    result = solve_dataset(data)
    position, angle = difference(np.array(result["base_T_camera"]), b_c)
    assert position < .003 and angle < .01
    assert result["status"] == "validated"
    altered = copy.deepcopy(data)
    altered["samples"][-1]["camera_T_marker"][0][3] += .08
    failed = solve_dataset(altered)
    assert failed["status"] == "quality_failed"
    np.testing.assert_allclose(failed["base_T_camera"], result["base_T_camera"], atol=1e-12)
    assert failed["validation"]["position_max_mm"] > 70


def test_marker_slip_in_training_is_reported():
    data, _, _ = dataset()
    data["samples"][8]["camera_T_marker"][1][3] += .05
    result = solve_dataset(data)
    assert result["status"] == "quality_failed"
    assert result["training"]["position_max_mm"] > 10


@pytest.mark.parametrize("rotate", [False, True])
def test_translation_only_and_single_axis_motion_are_rejected(rotate):
    data, _, _ = dataset()
    for i, sample in enumerate(data["samples"]):
        sample["base_T_flange"] = transform(Rotation.from_euler("z", i * .07 if rotate else 0).as_matrix(),
                                            [i * .01, 0, .4]).tolist()
    with pytest.raises(ValueError, match="姿态变化不足"):
        solve_dataset(data)


def test_needs_real_validation_and_rejects_invalid_frames_or_units():
    data, _, _ = dataset()
    data["samples"] = data["samples"][:20]
    assert solve_dataset(data)["status"] == "needs_validation"
    data["translation_unit"] = "mm"
    with pytest.raises(ValueError, match="米"):
        solve_dataset(data)
    data["translation_unit"] = "m"
    data["flange_frame"] = "custom_tcp"
    with pytest.raises(ValueError, match="fr3_link8"):
        solve_dataset(data)


def test_too_few_samples_or_bad_matrix_are_rejected():
    data, _, _ = dataset()
    data["samples"] = data["samples"][:7]
    with pytest.raises(ValueError, match="至少需要 8"):
        solve_dataset(data)
    data, _, _ = dataset()
    data["samples"][0]["base_T_flange"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        solve_dataset(data)


def test_offline_solve_never_initializes_devices_and_preserves_raw(tmp_path, monkeypatch):
    data, _, _ = dataset()
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    monkeypatch.setattr(ColorCamera, "start", Mock(side_effect=AssertionError("camera accessed")))
    monkeypatch.setattr(RobotReader, "start", Mock(side_effect=AssertionError("robot accessed")))
    assert main(["solve", str(path)]) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "validated"
    assert len(result["source_sha256"]) == 64
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="不能覆盖"):
        solve_file(path, path)


def stationary_data():
    states = [{"received_at": float(t), "q": [0.] * 7, "dq": [0.] * 7}
              for t in np.arange(9.8, 11.25, .03)]
    frames = [{"received_at": float(t), "camera_T_marker": np.eye(4).tolist()}
              for t in np.linspace(10, 11, 5)]
    return states, frames


def test_stationary_capture_selects_matching_pose():
    states, frames = stationary_data()
    chosen, state, report = stationary_sample(states, frames)
    assert chosen is frames[2]
    assert abs(state["received_at"] - 10.5) < .03
    assert report["max_joint_span_rad"] == 0


@pytest.mark.parametrize("fault", ["moving", "drifting", "stale", "gap", "board_slip", "nan"])
def test_motion_stale_feedback_and_board_instability_reject_sample(fault):
    states, frames = stationary_data()
    if fault == "moving":
        states[15]["dq"][0] = .05
    elif fault == "drifting":
        states[15]["q"][0] = .005
    elif fault == "stale":
        states = states[:10]
    elif fault == "gap":
        states = states[:10] + states[25:]
    elif fault == "board_slip":
        frames[0]["camera_T_marker"][0][3] = .01
    elif fault == "nan":
        states[15]["q"][0] = float("nan")
    with pytest.raises(ValueError):
        stationary_sample(states, frames)


def test_duplicate_pose_rejected_but_distinct_orientation_allowed():
    old = [{"base_T_flange": np.eye(4).tolist()}]
    with pytest.raises(ValueError, match="太接近"):
        reject_duplicate(transform(np.eye(3), [.001, 0, 0]), old)
    reject_duplicate(transform(Rotation.from_euler("y", .2).as_matrix(), [0, 0, 0]), old)


def test_aruco_rendered_image_detects_correct_marker_pose():
    detector, dictionary = create_detector("DICT_4X4_50")
    board = {"marker_id": 20, "marker_length_m": .1}
    k = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]])
    d = np.zeros(5)
    pose = transform(Rotation.from_euler("xyz", [2.8, .2, -.2]).as_matrix(), [.01, .02, .6])
    obj = np.array([[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0]])
    corners = cv2.projectPoints(obj, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], k, d)[0].reshape(4, 2)
    marker = cv2.aruco.generateImageMarker(dictionary, 20, 300)
    mapping = cv2.getPerspectiveTransform(np.float32([[0, 0], [299, 0], [299, 299], [0, 299]]),
                                         corners.astype(np.float32))
    gray = cv2.warpPerspective(marker, mapping, (640, 480), borderValue=255)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    found = detect_marker(image, detector, board, k, d)
    distance, angle = difference(np.array(found["camera_T_marker"]), pose)
    assert distance < .006 and angle < .06
    assert found["reprojection_rms_px"] < 1.5
    with pytest.raises(ValueError, match="ID=21"):
        detect_marker(image, detector, dict(board, marker_id=21), k, d)


def test_planar_pose_ambiguity_rejected():
    k = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1.]])
    # Fake corner detector tests PnP geometry independently of image segmentation.
    obj = np.array([[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0]])
    pose = transform(Rotation.from_euler("xyz", [3.04, 0, 0]).as_matrix(), [.1, 0, 1.5])
    corners = cv2.projectPoints(obj, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], k, np.zeros(5))[0]
    detector = Mock()
    detector.detectMarkers.return_value = ([corners.reshape(1, 4, 2)], np.array([[20]]), [])
    with pytest.raises(ValueError, match="歧义"):
        detect_marker(np.zeros((480, 640, 3), np.uint8), detector,
                      {"marker_id": 20, "marker_length_m": .1}, k, np.zeros(5))


def test_distortion_never_silently_discarded():
    intr = SimpleNamespace(fx=600., fy=600., ppx=320., ppy=240., width=640, height=480,
                           coeffs=[.1, 0, 0, 0, 0], model="distortion.brown_conrady")
    _, d, metadata = opencv_intrinsics(intr)
    assert d[0] == .1 and metadata["realsense_distortion_model"] == "brown_conrady"
    intr.model = "distortion.inverse_brown_conrady"
    with pytest.raises(ValueError, match="畸变"):
        opencv_intrinsics(intr)
    intr.coeffs = [0.] * 5
    assert not opencv_intrinsics(intr)[1].any()


def test_camera_start_failure_releases_partial_pipeline(monkeypatch):
    import sys
    pipeline = Mock()
    pipeline.start.side_effect = RuntimeError("Device busy")
    fake_rs = SimpleNamespace(pipeline=lambda: pipeline, config=Mock(),
                              stream=SimpleNamespace(color=1), format=SimpleNamespace(bgr8=1))
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    camera = ColorCamera("serial", 640, 480, 6)
    with pytest.raises(RuntimeError, match="busy"):
        camera.start()
    pipeline.stop.assert_called_once()
    assert camera.pipeline is None


def test_collector_failure_always_closes_camera_and_read_only_bridge(tmp_path, monkeypatch):
    monkeypatch.setattr(RobotReader, "start", Mock())
    robot_close = Mock()
    camera_close = Mock()
    monkeypatch.setattr(RobotReader, "close", robot_close)
    monkeypatch.setattr(ColorCamera, "close", camera_close)
    monkeypatch.setattr(ColorCamera, "start", Mock(side_effect=RuntimeError("busy")))
    monkeypatch.setattr(cv2, "destroyAllWindows", Mock())
    profile = tmp_path / "hardware.json"
    profile.write_text(json.dumps({"arm": {"stack": "real", "workspace": str(tmp_path)}}))
    assert main(["collect", "--config", str(profile),
                 "--session-dir", str(tmp_path / "session")]) == 1
    robot_close.assert_called_once()
    camera_close.assert_called_once()


def test_read_only_bridge_protocol_and_owned_process_cleanup(tmp_path, monkeypatch):
    # Replace docker with a local Python fixture: exercises actual pipes/thread/EOF.
    source = '''import json, sys, time
def emit(row):
    print("EYE_TO_HAND=" + json.dumps(row), flush=True)
emit({"kind": "joints", "received_at": time.monotonic(), "q": [0]*7, "dq": [0]*7})
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "quit":
        break
    assert request["op"] == "fk" and len(request["q"]) == 7
    emit({"kind": "fk", "id": request["id"], "xyz": [.3, .1, .4], "xyzw": [0, 0, 0, 1]})
'''
    original = subprocess.Popen
    commands = []

    def local_worker(command, **kwargs):
        commands.append(command)
        return original([sys.executable, "-u", "-c", source], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", local_worker)
    reader = RobotReader(SimpleNamespace(stack="real", workspace=str(tmp_path)))
    try:
        reader.start()
        assert len(reader.states) == 1
        np.testing.assert_allclose(reader.fk([0.] * 7), transform(np.eye(3), [.3, .1, .4]))
        assert commands[0][:3] == ["docker", "compose", "-f"]
    finally:
        reader.close()
    assert reader.process.poll() == 0
    assert not reader.thread.is_alive()
    assert reader.process.stdin.closed and reader.process.stdout.closed


def test_bridge_early_failure_is_diagnostic_and_does_not_leak_process(tmp_path, monkeypatch):
    original = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kw: original(
        [sys.executable, "-u", "-c", "print('service real is not running')"], **kw))
    reader = RobotReader(SimpleNamespace(stack="real", workspace=str(tmp_path)))
    try:
        with pytest.raises(RuntimeError, match="service real is not running"):
            reader.start()
    finally:
        reader.close()
    assert reader.process.poll() is not None
    assert not reader.thread.is_alive()
