"""Terminal-triggered RGB capture with simulated frames and no device access."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from hardware import wrist_raw_capture as capture
from hardware.reference_capture import PreviewFrame


def frame(value, received_at):
    pixels = np.full((12, 16, 3), value, dtype=np.uint8)
    pixels[0, 0] = [255, 0, 7]
    return PreviewFrame(Image.fromarray(pixels), {}, received_at)


def test_saves_original_rgb_pixels_without_changing_gallery(tmp_path):
    gallery = tmp_path / "gallery.json"
    gallery.write_text('{"objects": []}')
    original = frame(42, 0)
    target = capture.save_raw_image(original, tmp_path / "raw_images")
    with Image.open(target) as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(original.image))
        assert saved.format == "PNG"
    assert gallery.read_text() == '{"objects": []}'
    assert set(tmp_path.iterdir()) == {gallery, target.parent}


def test_trigger_waits_for_frame_received_after_enter():
    old, new = frame(10, 1), frame(20, 3)
    worker = SimpleNamespace(running=True, latest=Mock(side_effect=[(old, None), (new, None)]))
    assert capture.wait_for_frame(worker, after=2) is new
    assert worker.latest.call_count == 2


@pytest.fixture
def worker(monkeypatch):
    class FakeWorker:
        running = False
        error = None
        sequence = 0
        thread = Mock()

        def start(self):
            self.running = True

        def latest(self):
            self.sequence += 1
            return frame(self.sequence, capture.time.monotonic()), self.error

        def stop(self):
            self.running = False

    instance = FakeWorker()
    monkeypatch.setattr(capture, "PreviewWorker", lambda source: instance)
    monkeypatch.setattr(capture, "load_config", lambda path: SimpleNamespace(
        cameras=SimpleNamespace(wrist_serial="test-wrist")))
    return instance


def test_each_enter_saves_one_new_frame_and_q_releases_camera(tmp_path, monkeypatch, worker):
    commands = iter(["", "", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    assert capture.main(["--output-dir", str(tmp_path)]) == 0
    paths = sorted(tmp_path.glob("wrist_*.png"))
    assert len(paths) == 2
    with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
        assert np.asarray(first)[1, 1, 0] < np.asarray(second)[1, 1, 0]
    assert not worker.running
    worker.thread.join.assert_called_once()


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
def test_terminal_exit_releases_camera_without_saving(tmp_path, monkeypatch, worker, interrupt):
    def read(prompt):
        raise interrupt()

    monkeypatch.setattr("builtins.input", read)
    assert capture.main(["--output-dir", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())
    assert not worker.running
    worker.thread.join.assert_called_once()


def test_camera_failure_is_reported_and_released(tmp_path, worker, capsys):
    worker.error = "USB disconnected"
    assert capture.main(["--output-dir", str(tmp_path)]) == 1
    assert "USB disconnected" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
    assert not worker.running
    worker.thread.join.assert_called_once()
