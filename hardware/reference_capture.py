"""Reference-photo capture helpers. No robot clients or hardware access at import."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np
from PIL import Image

from services.object_reference_service import (
    REFERENCE_VIEWPOINTS, ReferenceGallery, atomic_json, crop_image, load_settings,
)

ROOT = Path(__file__).resolve().parents[1]
VIEW_LABELS = {"正面": "front", "侧面": "side", "背面": "back", "顶面": "top",
               "底面": "bottom", "斜视": "oblique", "未区分": "unspecified"}


@dataclass(frozen=True)
class PreviewFrame:
    image: Image.Image
    metadata: dict
    received_at: float


def wrist_frames(settings, stop, rs_module=None):
    """Only the configured wrist serial and RGB stream; never fall back to another device."""
    if not settings.wrist_serial:
        raise ValueError("腕部相机序列号未配置")
    if rs_module is None:
        import pyrealsense2 as rs_module
    rs = rs_module
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(settings.wrist_serial)
    config.enable_stream(rs.stream.color, settings.width, settings.height, rs.format.rgb8, settings.fps)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        actual = profile.get_device().get_info(rs.camera_info.serial_number)
        if actual != settings.wrist_serial:
            raise RuntimeError(f"相机序列号不匹配：期望 {settings.wrist_serial}，实际 {actual}")
        last_frame = time.monotonic()
        while not stop.is_set():
            ok, frames = pipeline.try_wait_for_frames(250)
            color = frames.get_color_frame() if ok else None
            if not color:
                if time.monotonic() - last_frame > 5:
                    raise RuntimeError("腕部相机超过 5 秒未返回彩色图像；请检查 USB 和设备占用")
                continue
            last_frame = time.monotonic()
            image = Image.fromarray(np.asanyarray(color.get_data()).copy(), "RGB")
            yield PreviewFrame(image, {
                "source": "realsense_rgb", "camera_role": "wrist", "device_serial": actual,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "frame_number": color.get_frame_number(), "sensor_timestamp_ms": color.get_timestamp(),
                "width": image.width, "height": image.height,
            }, last_frame)
    finally:
        if started:
            pipeline.stop()


class PreviewWorker:
    """Latest-frame mailbox: slow GUI rendering cannot accumulate stale camera frames."""
    def __init__(self, source):
        self.source = source
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.frame = None
        self.error = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running:
            return
        with self.lock:
            self.frame = None
            self.error = None
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="reference-wrist-preview", daemon=True)
        self.thread.start()

    def _run(self):
        iterator = None
        try:
            iterator = iter(self.source(self.stop_event))
            for frame in iterator:
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.frame = frame
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                try:
                    iterator.close()
                except Exception as exc:
                    with self.lock:
                        self.error = str(exc)

    def latest(self):
        with self.lock:
            return self.frame, self.error

    def stop(self):
        self.stop_event.set()


@dataclass(frozen=True)
class ImageViewport:
    """Letterboxed canvas coordinates mapped back to original image pixels."""
    image_width: int
    image_height: int
    canvas_width: int
    canvas_height: int

    @property
    def display_size(self):
        scale = min(self.canvas_width / self.image_width, self.canvas_height / self.image_height)
        return max(1, round(self.image_width * scale)), max(1, round(self.image_height * scale))

    @property
    def offset(self):
        width, height = self.display_size
        return (self.canvas_width - width) / 2, (self.canvas_height - height) / 2

    def to_image(self, x, y, *, clamp=False):
        width, height = self.display_size
        ox, oy = self.offset
        if not clamp and not (ox <= x <= ox + width and oy <= y <= oy + height):
            return None
        return (max(0, min(self.image_width, (x - ox) * self.image_width / width)),
                max(0, min(self.image_height, (y - oy) * self.image_height / height)))

    def to_canvas(self, x, y):
        width, height = self.display_size
        ox, oy = self.offset
        return ox + x * width / self.image_width, oy + y * height / self.image_height


def save_snapshot(gallery, *, frame, bbox, name, category, detector_prompt, viewpoint,
                  aliases=(), attributes=None, source_group=None):
    """Persist full RGB + crop/label metadata and append the crop to the existing gallery."""
    if bbox is None:
        raise ValueError("请先框选物品；只有整幅画面均为该物品时才使用整张图片")
    if viewpoint not in REFERENCE_VIEWPOINTS:
        raise ValueError("请选择有效的照片视角")
    if not all(str(v).strip() for v in (name, category, detector_prompt)):
        raise ValueError("物品名称、类别和检测提示词不能为空")
    crop_image(frame.image, bbox)  # validate before creating any files
    a = np.asarray(bbox, dtype=float)
    clipped = [max(0, round(a[0])), max(0, round(a[1])),
               min(frame.image.width, round(a[2])), min(frame.image.height, round(a[3]))]
    capture_id = uuid.uuid4().hex
    directory = gallery.directory / "captures" / capture_id
    source_path = directory / "color.png"
    metadata = dict(frame.metadata, capture_id=capture_id, bbox=clipped, viewpoint=viewpoint,
                    source_path=str(source_path.relative_to(gallery.directory)), requested_name=name,
                    saved_at_utc=datetime.now(timezone.utc).isoformat())
    directory.mkdir(parents=True)
    try:
        frame.image.convert("RGB").save(source_path)
        atomic_json(directory / "capture.json", metadata)
        obj = gallery.resolve(name)
        result = gallery.register(name=obj["name"] if obj else name, image_path=source_path,
            category=category, detector_prompt=detector_prompt, bbox=clipped, aliases=aliases,
            attributes=attributes, viewpoint=viewpoint, capture_metadata=metadata,
            source_group=source_group or capture_id, provenance="user_reference_capture")
    except Exception:
        shutil.rmtree(directory)
        raise
    if not result["added"]:
        # Keep the original photo label and provenance when the crop is identical.
        shutil.rmtree(directory)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="腕部相机参考图库录入（不连接机械臂/夹爪，不加载识别模型）")
    parser.add_argument("--hardware-config", default=str(ROOT / "config/hardware/fr3.yaml"))
    parser.add_argument("--config", help="参考图库配置，默认沿用硬件配置中的 reference_config")
    parser.add_argument("--gallery-dir", help="指定独立图库目录，可用于离线试用")
    parser.add_argument("--image", type=Path, help="只打开本地图片；此模式完全不访问相机")
    args = parser.parse_args(argv)
    from hardware.config import load_config
    profile = load_config(args.hardware_config)
    settings = load_settings(args.config or profile.perception.reference_config
                             or ROOT / "config/perception/references.yaml")
    gallery = ReferenceGallery(args.gallery_dir or settings.gallery_dir)
    gallery.read()  # fail clearly before opening a window if the manifest is invalid
    offline_frame = None
    if args.image:
        path = args.image.expanduser().resolve()
        with Image.open(path) as image:
            offline_frame = PreviewFrame(image.convert("RGB").copy(),
                {"source": "local_image", "original_path": str(path)}, time.monotonic())
    # Tk loads only after --help and all read-only argument validation.
    from hardware.reference_ui_style import prepare_tk_runtime
    prepare_tk_runtime()
    import tkinter as tk
    from hardware.reference_capture_ui import ReferenceCaptureApp
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(2, f"无法打开图形窗口，请在本机桌面终端运行：{exc}\n")
    worker = PreviewWorker(lambda stop: wrist_frames(profile.cameras, stop)) if not offline_frame else None
    try:
        ReferenceCaptureApp(root, gallery, profile.cameras, worker=worker, offline_frame=offline_frame)
        root.mainloop()
    finally:
        if worker:
            worker.stop()
            if worker.thread:
                worker.thread.join(timeout=5)
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
