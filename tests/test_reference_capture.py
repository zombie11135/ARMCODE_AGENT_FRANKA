"""Photo registration contracts; fake camera only, GUI optionally on an explicit virtual display."""
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from hardware.config import CameraSettings
from hardware.reference_capture import ImageViewport, PreviewFrame, PreviewWorker, save_snapshot, wrist_frames
from services.object_reference_service import ReferenceGallery


@pytest.fixture
def capture(tmp_path):
    image = Image.new("RGB", (80, 60), "blue")
    image.paste((255, 0, 0), (10, 10, 40, 50))
    frame = PreviewFrame(image, {"source": "local_image", "frame_number": 42}, time.monotonic())
    gallery = ReferenceGallery(tmp_path / "gallery")
    options = dict(frame=frame, bbox=[10, 10, 40, 50], name="测试药盒", category="medicine_box",
                   detector_prompt="medicine package", viewpoint="front")
    return gallery, options


def test_snapshot_persists_exact_crop_full_frame_view_and_aliases(capture):
    gallery, args = capture
    first = save_snapshot(gallery, **args, aliases=["药盒全称"], attributes={"brand": "品牌"})
    obj = gallery.resolve("药盒全称")
    ref = obj["references"][0]
    assert ref["viewpoint"] == "front" and ref["capture"]["bbox"] == [10, 10, 40, 50]
    assert ref["capture"]["frame_number"] == 42
    with Image.open(gallery.directory / ref["path"]) as crop:
        assert crop.size == (30, 40) and crop.getpixel((0, 0)) == (255, 0, 0)
    source = gallery.directory / ref["capture"]["source_path"]
    with Image.open(source) as image:
        assert np.array_equal(image, args["frame"].image)
    assert json.loads(source.with_name("capture.json").read_text())["viewpoint"] == "front"
    second = save_snapshot(gallery, **(args | {"name": "药盒全称", "viewpoint": "side", "bbox": [40, 10, 70, 50]}))
    assert second["object_id"] == first["object_id"] and second["reference_count"] == 2
    assert [r["viewpoint"] for r in gallery.resolve("测试药盒")["references"]] == ["front", "side"]


def test_duplicate_keeps_original_view_and_does_not_leave_extra_capture(capture):
    gallery, args = capture
    save_snapshot(gallery, **args)
    before = gallery.read()
    second = save_snapshot(gallery, **(args | {"viewpoint": "side"}))
    assert not second["added"] and second["viewpoint"] == "front"
    assert gallery.read() == before
    assert len(list((gallery.directory / "captures").iterdir())) == 1


@pytest.mark.parametrize("change", [{"bbox": None}, {"bbox": [0, 0, 1, 1]}, {"bbox": [0, 0, float("nan"), 10]},
                                    {"viewpoint": "leftish"}, {"name": " "}, {"detector_prompt": ""}])
def test_invalid_registration_creates_no_files(capture, change):
    gallery, args = capture
    with pytest.raises(ValueError):
        save_snapshot(gallery, **(args | change))
    assert not gallery.directory.exists()


def test_wrong_identity_metadata_is_rejected_and_failed_capture_is_removed(capture):
    gallery, args = capture
    save_snapshot(gallery, **args, attributes={"specification": "10ml"})
    before = gallery.manifest.read_bytes()
    with pytest.raises(ValueError, match="differs"):
        save_snapshot(gallery, **args, attributes={"specification": "20ml"})
    assert gallery.manifest.read_bytes() == before
    assert len(list((gallery.directory / "captures").iterdir())) == 1


def test_letterbox_mouse_mapping_and_clipping():
    viewport = ImageViewport(640, 480, 1000, 500)
    assert viewport.to_image(0, 100) is None  # left padding
    assert viewport.to_image(-100, -100, clamp=True) == (0, 0)
    assert viewport.to_image(2000, 2000, clamp=True) == (640, 480)
    assert viewport.to_image(*viewport.to_canvas(300, 200)) == pytest.approx((300, 200))


def fake_realsense(serial):
    rs = Mock()
    rs.stream.color = "color"
    rs.format.rgb8 = "rgb8"
    rs.camera_info.serial_number = "serial_number"
    rs.pipeline.return_value.start.return_value.get_device.return_value.get_info.return_value = serial
    color = Mock()
    color.get_data.return_value = np.full((6, 8, 3), [250, 20, 10], dtype=np.uint8)
    color.get_frame_number.return_value = 21
    color.get_timestamp.return_value = 22.5
    frames = Mock()
    frames.get_color_frame.return_value = color
    rs.pipeline.return_value.try_wait_for_frames.return_value = (True, frames)
    return rs


def test_camera_is_lazy_serial_pinned_rgb_only_and_released():
    settings = CameraSettings()
    rs = fake_realsense(settings.wrist_serial)
    frames = wrist_frames(settings, threading.Event(), rs)
    rs.pipeline.assert_not_called()
    frame = next(frames)
    assert frame.image.getpixel((0, 0)) == (250, 20, 10)  # RGB, not BGR
    assert frame.metadata["device_serial"] == settings.wrist_serial
    rs.config.return_value.enable_device.assert_called_once_with(settings.wrist_serial)
    rs.config.return_value.enable_stream.assert_called_once_with("color", 640, 480, "rgb8", 6)
    frames.close()
    rs.pipeline.return_value.stop.assert_called_once()


def test_wrong_camera_never_yields_and_is_released():
    rs = fake_realsense("wrong_camera")
    with pytest.raises(RuntimeError, match="序列号不匹配"):
        next(wrist_frames(CameraSettings(), threading.Event(), rs))
    rs.pipeline.return_value.stop.assert_called_once()


def test_preview_worker_error_and_shutdown_do_not_retain_old_frame(capture):
    _, args = capture
    closed = threading.Event()
    def source(stop):
        try:
            yield args["frame"]
            raise RuntimeError("USB disconnected")
        finally:
            closed.set()
    worker = PreviewWorker(source)
    assert worker.latest() == (None, None)
    worker.start()
    worker.thread.join(2)
    assert closed.is_set() and worker.latest()[1] == "USB disconnected"
    def idle(stop):
        stop.wait(2)
        return
        yield
    worker.source = idle
    worker.start()
    assert worker.latest() == (None, None)
    worker.stop()
    worker.thread.join(2)
    assert not worker.running


def test_preview_reports_camera_release_failure_without_thread_traceback(capture):
    _, args = capture
    delivered = threading.Event()
    def source(stop):
        try:
            delivered.set()
            yield args["frame"]
            stop.wait(2)
            yield args["frame"]
        finally:
            raise RuntimeError("camera release failed")
    worker = PreviewWorker(source)
    worker.start()
    assert delivered.wait(2)
    worker.stop()
    worker.thread.join(2)
    assert not worker.running and worker.latest()[1] == "camera release failed"


@pytest.fixture
def gui(capture, monkeypatch):
    display = os.environ.get("REFERENCE_CAPTURE_TEST_DISPLAY")
    if not display:
        pytest.skip("GUI smoke test needs explicit REFERENCE_CAPTURE_TEST_DISPLAY (virtual display)")
    monkeypatch.setenv("DISPLAY", display)
    from hardware.reference_capture_ui import ReferenceCaptureApp
    import tkinter as tk
    monkeypatch.setattr("hardware.reference_capture_ui.messagebox.showerror",
                        lambda *a, **k: pytest.fail(str(a)))
    gallery, args = capture
    root = tk.Tk()
    app = ReferenceCaptureApp(root, gallery, CameraSettings(), offline_frame=args["frame"])
    root.update()
    app.draw()
    yield root, app, args
    app.close()


def drag(app, box):
    # Reverse drag verifies that XYXY is normalized on a resized preview.
    x1, y1 = app.viewport.to_canvas(box[2], box[3])
    x2, y2 = app.viewport.to_canvas(box[0], box[1])
    app.begin_box(SimpleNamespace(x=x1, y=y1))
    app.end_box(SimpleNamespace(x=x2, y=y2))


def test_gui_freezes_displayed_frame_appends_views_and_reloads_identity(gui):
    root, app, args = gui
    assert app.worker is None
    app.form["name"].set("测试药盒")
    app.form["brand"].set("品牌")
    app.capture()
    drag(app, [10, 10, 40, 50])
    assert app.bbox == pytest.approx([10, 10, 40, 50])
    app.save()
    obj = app.gallery.resolve("测试药盒")
    assert len(obj["references"]) == 1 and obj["attributes"]["brand"] == "品牌"
    app.form["view"].set("侧面")
    app.capture()
    drag(app, [40, 10, 70, 50])
    app.save()
    assert [r["viewpoint"] for r in app.gallery.resolve("测试药盒")["references"]] == ["front", "side"]
    assert "2 张" in app.object_info.get()
    app.new_object()
    assert app.form["brand"].get() == ""
    app.form["name"].set("测试药盒")
    app.load_object()
    assert app.form["brand"].get() == "品牌"
    assert len(app.history.get_children()) == 2


def test_gui_does_not_save_live_frame_and_rejects_stale_camera(gui):
    root, app, args = gui
    app.form["name"].set("测试药盒")
    app.save()
    assert not app.gallery.manifest.exists()
    app.live_frame = PreviewFrame(args["frame"].image, {"source": "realsense_rgb"}, time.monotonic() - 10)
    app.draw()
    app.capture()
    assert app.frozen is None and "过期" in app.status.get()
    app.live_frame = args["frame"]
    app.draw()
    app.capture()
    app.live_frame = PreviewFrame(Image.new("RGB", (80, 60), "green"), {}, time.monotonic())
    app.draw()
    drag(app, [10, 10, 40, 50])
    app.save()
    ref = app.gallery.resolve("测试药盒")["references"][0]
    with Image.open(app.gallery.directory / ref["path"]) as crop:
        assert crop.getpixel((0, 0)) == (255, 0, 0)
