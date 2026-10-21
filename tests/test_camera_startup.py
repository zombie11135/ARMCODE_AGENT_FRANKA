"""Frame readiness and CLI cleanup without opening cameras or moving hardware."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def camera(tmp_path, monkeypatch):
    from services import realsense_dual_camera_service as module

    monkeypatch.setattr(module.rs, "pipeline", Mock(side_effect=AssertionError("Hardware forbidden")))
    service = module.RealSenseDualCameraService(
        module.CameraConfig("global", serial="global-test", enable_depth=False),
        module.CameraConfig("wrist", serial="wrist-test", enable_depth=True),
        save_dir=str(tmp_path), show_window=False, startup_timeout_s=0.3,
    )

    def start_camera(name, cfg):
        service._pipelines[name] = Mock()
        service._frames[name]["enable_depth"] = cfg.enable_depth

    monkeypatch.setattr(service, "_start_camera", start_camera)
    monkeypatch.setattr(service, "_camera_worker", lambda name: service._stop_event.wait())
    yield service
    service.stop()


def publish(service, name, *, depth=False):
    with service._locks[name]:
        service._frames[name]["color"] = np.zeros((2, 2, 3), dtype=np.uint8)
        if depth:
            service._frames[name]["depth"] = np.ones((2, 2), dtype=np.uint16)


def test_start_waits_for_both_views_and_wrist_depth(camera, monkeypatch):
    camera.startup_timeout_s = 2
    rgb_ready = threading.Event()
    allow_depth = threading.Event()
    finished = threading.Event()
    errors = []

    def worker(name):
        publish(camera, name)
        if name == "wrist":
            rgb_ready.set()
            if allow_depth.wait(1):
                publish(camera, name, depth=True)
        camera._stop_event.wait()

    def start():
        try:
            camera.start()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(camera, "_camera_worker", worker)
    launcher = threading.Thread(target=start)
    launcher.start()
    try:
        assert rgb_ready.wait(1)
        assert not finished.wait(0.05), "RGB alone must not release RGB-D startup"
        allow_depth.set()
        assert finished.wait(1)
        assert not errors
        assert camera.capture_global()["color"].shape == (2, 2, 3)
        assert camera.capture_wrist()["depth"].shape == (2, 2)
    finally:
        allow_depth.set()
        launcher.join(2)


def test_missing_frames_timeout_releases_streams_and_workers(camera, monkeypatch):
    streams = []

    def worker(name):
        streams.append(camera._pipelines[name])
        if name == "global":
            publish(camera, name)
        else:
            camera._last_frame_errors[name] = "simulated USB timeout"
        camera._stop_event.wait()

    monkeypatch.setattr(camera, "_camera_worker", worker)
    with pytest.raises(RuntimeError, match="wrist.*wrist-test.*simulated USB timeout"):
        camera.start()
    assert not camera._running
    assert not camera._threads
    assert not camera._pipelines
    for stream in streams:
        stream.stop.assert_called_once()


def test_second_camera_start_failure_releases_first(camera, monkeypatch):
    stream = Mock()

    def start_camera(name, cfg):
        if name == "wrist":
            raise RuntimeError("Device busy")
        camera._pipelines[name] = stream

    monkeypatch.setattr(camera, "_start_camera", start_camera)
    with pytest.raises(RuntimeError, match="Device busy"):
        camera.start()
    stream.stop.assert_called_once()
    assert not camera._pipelines


def test_wrist_only_starts_and_waits_for_no_fixed_camera(camera, monkeypatch):
    camera.active_camera_names = ('wrist',)
    started = []
    original = camera._start_camera
    def start(name, cfg):
        assert name == 'wrist', 'Fixed stream must not be opened'
        started.append(name)
        original(name, cfg)
    def worker(name):
        publish(camera, name, depth=True)
        camera._stop_event.wait()
    monkeypatch.setattr(camera, '_start_camera', start)
    monkeypatch.setattr(camera, '_camera_worker', worker)
    camera.start()
    assert started == ['wrist']
    assert camera.capture_wrist()['depth'].shape == (2, 2)
    with pytest.raises(ValueError, match='disabled'):
        camera.capture_global()


def test_previous_frames_cannot_satisfy_a_new_start(camera):
    publish(camera, "global")
    publish(camera, "wrist", depth=True)
    with pytest.raises(RuntimeError, match="first camera frames"):
        camera.start()
    assert camera._frames["global"]["color"] is None
    assert camera._frames["wrist"]["depth"] is None


def test_metadata_failure_also_releases_started_stream(camera, monkeypatch):
    from services import realsense_dual_camera_service as module

    stream = Mock()
    stream.start.return_value.get_device.side_effect = RuntimeError("metadata unavailable")
    monkeypatch.setattr(module.rs, "pipeline", Mock(return_value=stream))
    monkeypatch.setattr(camera, "_start_camera",
                        module.RealSenseDualCameraService._start_camera.__get__(camera))
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        camera.start()
    stream.stop.assert_called_once()


@pytest.mark.parametrize("outcome", [True, False, RuntimeError("VDM failed"), KeyboardInterrupt()])
def test_single_goal_closes_cached_services_on_every_exit(monkeypatch, outcome):
    import sys
    from app import cli
    from services import runtime_service_cache

    cleanup = Mock()
    monkeypatch.setattr(runtime_service_cache, "close_cached_services", cleanup)
    run = Mock(side_effect=outcome) if isinstance(outcome, BaseException) else Mock(return_value=outcome)
    monkeypatch.setattr(cli, "run_goal", run)
    monkeypatch.setattr(sys, "argv", ["app", "--goal", "offline", "--close_services_on_exit"])
    if isinstance(outcome, BaseException):
        with pytest.raises(type(outcome)):
            cli.main()
    elif outcome is False:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
    else:
        cli.main()
    cleanup.assert_called_once()


def test_interactive_warmup_failure_closes_services(monkeypatch):
    from app import cli
    from services import runtime_service_cache

    cleanup = Mock()
    monkeypatch.setattr(runtime_service_cache, "close_cached_services", cleanup)
    monkeypatch.setattr(cli, "warmup_camera_if_needed", Mock(side_effect=RuntimeError("warmup failed")))
    with pytest.raises(RuntimeError, match="warmup failed"):
        cli.interactive_loop(SimpleNamespace(execute_robot=False, close_services_on_exit=True), [])
    cleanup.assert_called_once()
