"""Save full wrist RGB frames on terminal Enter; camera access stays in the worker."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .reference_capture import PreviewWorker, wrist_frames


DEFAULT_OUTPUT = PROJECT_ROOT / "data/object_gallery/raw_images"


def wait_for_frame(worker, *, after=None, timeout_s=8.0):
    """Wait for a new frame after the trigger, rather than saving a queued old image."""
    deadline = time.monotonic() + timeout_s
    while True:
        frame, error = worker.latest()
        if error:
            raise RuntimeError(f"腕部相机采集失败：{error}")
        if frame is not None and (after is None or frame.received_at >= after):
            return frame
        if not worker.running:
            raise RuntimeError("腕部相机采集已停止")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待腕部相机新图像超时，请检查 USB 连接和相机占用")
        time.sleep(0.01)


def save_raw_image(frame, directory):
    """Save the complete original RGB image losslessly, without touching the gallery."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"wrist_{stamp}.png"
    # Exclusive creation also protects an existing photo on a filename collision.
    with path.open("xb") as stream:
        try:
            frame.image.save(stream, format="PNG")
        except BaseException:
            path.unlink()
            raise
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="腕部相机原图拍摄：终端按 Enter 拍照，q + Enter 退出")
    parser.add_argument("--hardware-config", default=str(DEFAULT_CONFIG), help="使用其中的腕部相机配置")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="原始照片保存目录")
    parser.add_argument("--timeout-s", type=float, default=8.0, help="等待新图像的超时时间（秒）")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s 必须是有限正数")

    worker = None
    count = 0
    try:
        profile = load_config(args.hardware_config)
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        worker = PreviewWorker(lambda stop: wrist_frames(profile.cameras, stop))
        print(f"正在连接腕部相机 {profile.cameras.wrist_serial}……", flush=True)
        worker.start()
        wait_for_frame(worker, timeout_s=args.timeout_s)
        print(f"照片目录：{output}\n终端按 Enter 拍照；输入 q 后按 Enter 退出，也可按 Ctrl+C。", flush=True)
        while True:
            command = input("拍照 [Enter] / 退出 [q]：").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            if command:
                print("直接按 Enter 拍照，或输入 q 退出。", flush=True)
                continue
            frame = wait_for_frame(worker, after=time.monotonic(), timeout_s=args.timeout_s)
            path = save_raw_image(frame, output)
            count += 1
            print(f"已保存第 {count} 张：{path}", flush=True)
    except (EOFError, KeyboardInterrupt):
        print("\n结束拍摄。", flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"拍摄失败：{exc}", flush=True)
        return 1
    finally:
        if worker is not None:
            worker.stop()
            if worker.thread is not None:
                worker.thread.join(timeout=6)
    print(f"本次共保存 {count} 张照片。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
