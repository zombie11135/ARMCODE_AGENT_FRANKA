import os
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class CameraConfig:
    name: str
    serial: Optional[str] = None

    color_width: int = 1280
    color_height: int = 720
    color_fps: int = 30

    enable_depth: bool = True
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30


class RealSenseDualCameraService:
    """
    双 RealSense 相机服务。

    global camera:
        默认只开 RGB，用于 VDM / 场景描述 / 目标识别。

    wrist camera:
        默认开 RGB-D，用于 OCR / 分割 / 深度定位 / 抓取确认。
    """

    def __init__(
        self,
        global_config: CameraConfig,
        wrist_config: CameraConfig,
        save_dir: str = "runs/current_capture",
        show_window: bool = True,
        save_depth_vis: bool = True,
        startup_timeout_s: float = 10.0,
        active_cameras=("global", "wrist"),
    ):
        self.global_config = global_config
        self.wrist_config = wrist_config
        self.active_camera_names = tuple(active_cameras)
        if (not self.active_camera_names or len(set(self.active_camera_names)) != len(self.active_camera_names)
                or not set(self.active_camera_names) <= {'global', 'wrist'}):
            raise ValueError('active_cameras must contain distinct global/wrist names')
        self.save_dir = save_dir
        self.show_window = show_window
        self.save_depth_vis = save_depth_vis
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be finite and positive")
        self.startup_timeout_s = float(startup_timeout_s)

        os.makedirs(self.save_dir, exist_ok=True)

        self._stop_event = threading.Event()

        self._pipelines = {}
        self._aligners = {}
        self._profiles = {}

        self._frames = {
            "global": self._empty_frame_dict(),
            "wrist": self._empty_frame_dict(),
        }

        self._locks = {
            "global": threading.Lock(),
            "wrist": threading.Lock(),
        }

        self._threads = []
        self._running = False
        self._last_frame_errors = {}

    @staticmethod
    def _empty_frame_dict():
        return {
            "points_map": None,
            "color": None,
            "depth": None,
            "depth_vis": None,
            "timestamp": None,
            "color_intrinsics": None,
            "depth_intrinsics": None,
            "depth_scale": None,
            "depth_unit_m_per_raw_unit": None,
            "depth_aligned_to_color": False,
            "device_name": None,
            "device_serial": None,
            "frame_number": None,
            "sensor_timestamp_ms": None,
            "timestamp_domain": None,
            "color_frame_number": None,
            "depth_frame_number": None,
            "color_sensor_timestamp_ms": None,
            "depth_sensor_timestamp_ms": None,
            "rgb_depth_sensor_timestamp_delta_ms": None,
            "enable_depth": False,
        }

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            print(f"[CameraService] RealSense cameras already running: {self.active_camera_names}")
            return

        if any(t.is_alive() for t in self._threads):
            raise RuntimeError("Previous camera workers have not stopped; cannot restart")
        self._threads = []
        self._stop_event.clear()
        self._last_frame_errors.clear()
        for name in self._frames:
            with self._locks[name]:
                self._frames[name] = self._empty_frame_dict()
        try:
            for name in self.active_camera_names:
                self._start_camera(name, getattr(self, name + '_config'))
            for camera_name in self.active_camera_names:
                t = threading.Thread(
                    target=self._camera_worker, args=(camera_name,), daemon=True,
                )
                t.start()
                self._threads.append(t)

            self._wait_for_first_frames()
            if self.show_window:
                display_thread = threading.Thread(target=self._display_worker, daemon=True)
                display_thread.start()
                self._threads.append(display_thread)
            self._running = True
            print(f"[CameraService] RealSense cameras started: {self.active_camera_names}; first frames ready.")
        except BaseException:
            self.stop()
            raise

    def _wait_for_first_frames(self):
        """Opening streams does not mean their asynchronous frame buffers are ready."""
        deadline = time.monotonic() + self.startup_timeout_s
        configs = {name: getattr(self, name + '_config') for name in self.active_camera_names}
        while not self._stop_event.is_set():
            pending = []
            for name, cfg in configs.items():
                with self._locks[name]:
                    frame = self._frames[name]
                    ready = frame["color"] is not None and (
                        not cfg.enable_depth or frame["depth"] is not None)
                if not ready:
                    pending.append(f"{name}(serial={cfg.serial}, "
                                   f"streams={'RGB-D' if cfg.enable_depth else 'RGB'}, "
                                   f"last_error={self._last_frame_errors.get(name, 'none')})")
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Timed out after {self.startup_timeout_s:g}s waiting for "
                                   f"first camera frames: {', '.join(pending)}")
            self._stop_event.wait(min(0.02, remaining))
        raise RuntimeError("Camera startup interrupted before first frames were ready")

    def stop(self):
        self._stop_event.set()

        for name, pipeline in self._pipelines.items():
            try:
                pipeline.stop()
                print(f"[CameraService] Stopped {name} camera.")
            except Exception as e:
                print(f"[CameraService] Failed to stop {name}: {e}")

        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
        if self.show_window and self._threads:
            cv2.destroyAllWindows()

        self._running = False
        self._threads = [t for t in self._threads if t.is_alive()]
        self._pipelines = {}
        self._aligners = {}
        self._profiles = {}
        for name in self._frames:
            with self._locks[name]:
                self._frames[name] = self._empty_frame_dict()

    def capture_global(self) -> Dict[str, Any]:
        """
        全局相机拍照。

        只保存彩色图像：
            runs/current_capture/global_color.png
        """
        return self.capture("global")

    def capture_wrist(self) -> Dict[str, Any]:
        """
        腕部相机拍照。

        保存：
            runs/current_capture/wrist_color.png
            runs/current_capture/wrist_depth.png
            runs/current_capture/wrist_depth_vis.png
        """
        return self.capture("wrist")

    def capture(self, camera_name: str) -> Dict[str, Any]:
        if camera_name not in self.active_camera_names:
            raise ValueError(f'Camera {camera_name} is disabled')
        if camera_name not in ["global", "wrist"]:
            raise ValueError(f"Unknown camera_name: {camera_name}")

        with self._locks[camera_name]:
            color = self._frames[camera_name]["color"]
            depth = self._frames[camera_name]["depth"]
            depth_vis = self._frames[camera_name]["depth_vis"]
            timestamp = self._frames[camera_name]["timestamp"]
            color_intrinsics = self._frames[camera_name]["color_intrinsics"]
            depth_intrinsics = self._frames[camera_name]["depth_intrinsics"]
            depth_scale = self._frames[camera_name]["depth_scale"]
            depth_unit_m_per_raw_unit = self._frames[camera_name]["depth_unit_m_per_raw_unit"]
            enable_depth = self._frames[camera_name]["enable_depth"]
            metadata = {
                "device_name": self._frames[camera_name]["device_name"],
                "device_serial": self._frames[camera_name]["device_serial"],
                "frame_number": self._frames[camera_name]["frame_number"],
                "sensor_timestamp_ms": self._frames[camera_name]["sensor_timestamp_ms"],
                "timestamp_domain": self._frames[camera_name]["timestamp_domain"],
                "depth_aligned_to_color": self._frames[camera_name]["depth_aligned_to_color"],
                "color_frame_number": self._frames[camera_name]["color_frame_number"],
                "depth_frame_number": self._frames[camera_name]["depth_frame_number"],
                "color_sensor_timestamp_ms": self._frames[camera_name]["color_sensor_timestamp_ms"],
                "depth_sensor_timestamp_ms": self._frames[camera_name]["depth_sensor_timestamp_ms"],
                "rgb_depth_sensor_timestamp_delta_ms": self._frames[camera_name]["rgb_depth_sensor_timestamp_delta_ms"],
            }

            if color is None:
                raise RuntimeError(
                    f"No color frame available for {camera_name}. "
                    f"Please make sure camera service is running."
                )

            points = self._frames[camera_name].get("points_map")
            points = points.copy() if points is not None else None
            color = color.copy()
            depth = depth.copy() if depth is not None else None
            depth_vis = depth_vis.copy() if depth_vis is not None else None

        color_path = os.path.join(self.save_dir, f"{camera_name}_color.png")
        cv2.imwrite(color_path, color)

        result = {
            "points_map": points,
            "camera_name": camera_name,
            "color_path": color_path,
            "color": color,
            "timestamp": timestamp,
            "color_intrinsics": color_intrinsics,
            "enable_depth": enable_depth,
            **metadata,
        }

        if enable_depth:
            if depth is None:
                raise RuntimeError(f"No depth frame available for {camera_name}.")

            depth_path = os.path.join(self.save_dir, f"{camera_name}_depth.png")
            depth_vis_path = os.path.join(self.save_dir, f"{camera_name}_depth_vis.png")

            cv2.imwrite(depth_path, depth)

            if self.save_depth_vis and depth_vis is not None:
                cv2.imwrite(depth_vis_path, depth_vis)
            else:
                depth_vis_path = None

            result.update(
                {
                    "depth_path": depth_path,
                    "depth_vis_path": depth_vis_path,
                    "depth": depth,
                    "depth_intrinsics": depth_intrinsics,
                    "depth_scale": depth_scale,
                    # Explicit multiplier form. RealSense reports meters per raw
                    # depth unit; legacy callers may still use depth_scale as
                    # either a multiplier (<1) or divisor (>1).
                    "depth_unit_m_per_raw_unit": depth_unit_m_per_raw_unit,
                }
            )
        else:
            result.update(
                {
                    "depth_path": None,
                    "depth_vis_path": None,
                    "depth": None,
                    "depth_intrinsics": None,
                    "depth_scale": None,
                    "depth_unit_m_per_raw_unit": None,
                }
            )

        return result

    def get_latest_color(self, camera_name: str) -> np.ndarray:
        if camera_name not in ["global", "wrist"]:
            raise ValueError(f"Unknown camera_name: {camera_name}")

        with self._locks[camera_name]:
            color = self._frames[camera_name]["color"]
            if color is None:
                raise RuntimeError(f"No color frame available for {camera_name}.")
            return color.copy()

    def get_latest_rgbd(self, camera_name: str) -> Dict[str, Any]:
        if camera_name not in ["global", "wrist"]:
            raise ValueError(f"Unknown camera_name: {camera_name}")

        with self._locks[camera_name]:
            frame = self._frames[camera_name]

            if frame["color"] is None:
                raise RuntimeError(f"No color frame available for {camera_name}.")

            if not frame["enable_depth"]:
                raise RuntimeError(f"{camera_name} camera does not enable depth stream.")

            if frame["depth"] is None:
                raise RuntimeError(f"No depth frame available for {camera_name}.")

            return {
                "camera_name": camera_name,
                "color": frame["color"].copy(),
                "depth": frame["depth"].copy(),
                "timestamp": frame["timestamp"],
                "color_intrinsics": frame["color_intrinsics"],
                "depth_intrinsics": frame["depth_intrinsics"],
                "depth_scale": frame["depth_scale"],
                "depth_unit_m_per_raw_unit": frame["depth_unit_m_per_raw_unit"],
                "depth_aligned_to_color": frame["depth_aligned_to_color"],
                "device_name": frame["device_name"],
                "device_serial": frame["device_serial"],
                "frame_number": frame["frame_number"],
                "sensor_timestamp_ms": frame["sensor_timestamp_ms"],
                "timestamp_domain": frame["timestamp_domain"],
                "color_frame_number": frame["color_frame_number"],
                "depth_frame_number": frame["depth_frame_number"],
                "color_sensor_timestamp_ms": frame["color_sensor_timestamp_ms"],
                "depth_sensor_timestamp_ms": frame["depth_sensor_timestamp_ms"],
                "rgb_depth_sensor_timestamp_delta_ms": frame["rgb_depth_sensor_timestamp_delta_ms"],
            }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _start_camera(self, camera_name: str, cam_cfg: CameraConfig):
        pipeline = rs.pipeline()
        config = rs.config()

        if cam_cfg.serial is not None:
            config.enable_device(cam_cfg.serial)

        config.enable_stream(
            rs.stream.color,
            cam_cfg.color_width,
            cam_cfg.color_height,
            rs.format.bgr8,
            cam_cfg.color_fps,
        )

        if cam_cfg.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                cam_cfg.depth_width,
                cam_cfg.depth_height,
                rs.format.z16,
                cam_cfg.depth_fps,
            )

        try:
            profile = pipeline.start(config)
        except Exception as e:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to start {camera_name} camera, "
                f"serial={cam_cfg.serial}, error={e}"
            )

        # Own the started stream before metadata validation, so failures release it.
        self._pipelines[camera_name] = pipeline
        device = profile.get_device()
        active_name = device.get_info(rs.camera_info.name)
        active_serial = device.get_info(rs.camera_info.serial_number)

        if cam_cfg.serial is not None and active_serial != cam_cfg.serial:
            raise RuntimeError(
                f"{camera_name} camera serial mismatch: "
                f"expected {cam_cfg.serial}, got {active_serial}"
            )

        color_stream = profile.get_stream(rs.stream.color)
        color_intr = color_stream.as_video_stream_profile().get_intrinsics()
        color_intrinsics = self._intrinsics_to_dict(color_intr)

        depth_intrinsics = None
        depth_scale = None
        aligner = None

        if cam_cfg.enable_depth:
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
            depth_intrinsics = self._intrinsics_to_dict(depth_intr)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # 只有启用深度时才需要对齐
            aligner = rs.align(rs.stream.color)

        self._pipelines[camera_name] = pipeline
        self._aligners[camera_name] = aligner
        self._profiles[camera_name] = profile

        with self._locks[camera_name]:
            self._frames[camera_name]["color_intrinsics"] = color_intrinsics
            self._frames[camera_name]["depth_intrinsics"] = depth_intrinsics
            self._frames[camera_name]["depth_scale"] = depth_scale
            self._frames[camera_name]["depth_unit_m_per_raw_unit"] = depth_scale
            self._frames[camera_name]["depth_aligned_to_color"] = False
            self._frames[camera_name]["device_name"] = active_name
            self._frames[camera_name]["device_serial"] = active_serial
            self._frames[camera_name]["enable_depth"] = cam_cfg.enable_depth

        '''print(
            f"[CameraService] Started {camera_name} camera: "
            f"name={active_name}, serial={active_serial}, "
            f"color={cam_cfg.color_width}x{cam_cfg.color_height}@{cam_cfg.color_fps}, "
            f"depth_enabled={cam_cfg.enable_depth}"
        )'''

    def _camera_worker(self, camera_name: str):
        pipeline = self._pipelines[camera_name]
        aligner = self._aligners[camera_name]

        frame_count = 0
        error_count = 0
        last_print_time = time.time()

        while not self._stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)

                with self._locks[camera_name]:
                    enable_depth = self._frames[camera_name]["enable_depth"]

                if enable_depth:
                    # 腕部相机：RGB-D，对齐 depth 到 color
                    aligned_frames = aligner.process(frames)

                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_vis = self._make_depth_vis(depth_image)
                    cloud = rs.pointcloud().calculate(depth_frame)
                    points_map = np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(
                        depth_image.shape[0], depth_image.shape[1], 3).copy()

                    # rs.align(depth -> color) changes the depth image geometry.
                    # Its valid pinhole model is therefore the aligned depth
                    # frame profile (normally identical to the color profile),
                    # not the raw depth-stream intrinsics captured at startup.
                    aligned_depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                    aligned_depth_intrinsics = self._intrinsics_to_dict(aligned_depth_intr)
                    color_sensor_timestamp_ms = float(color_frame.get_timestamp())
                    depth_sensor_timestamp_ms = float(depth_frame.get_timestamp())

                    with self._locks[camera_name]:
                        self._frames[camera_name]["points_map"] = points_map
                        self._frames[camera_name]["color"] = color_image
                        self._frames[camera_name]["depth"] = depth_image
                        self._frames[camera_name]["depth_vis"] = depth_vis
                        self._frames[camera_name]["timestamp"] = time.time()
                        self._frames[camera_name]["depth_intrinsics"] = aligned_depth_intrinsics
                        self._frames[camera_name]["depth_aligned_to_color"] = True
                        self._frames[camera_name]["frame_number"] = int(depth_frame.get_frame_number())
                        self._frames[camera_name]["sensor_timestamp_ms"] = float(depth_frame.get_timestamp())
                        self._frames[camera_name]["timestamp_domain"] = str(depth_frame.get_frame_timestamp_domain())
                        self._frames[camera_name]["color_frame_number"] = int(color_frame.get_frame_number())
                        self._frames[camera_name]["depth_frame_number"] = int(depth_frame.get_frame_number())
                        self._frames[camera_name]["color_sensor_timestamp_ms"] = color_sensor_timestamp_ms
                        self._frames[camera_name]["depth_sensor_timestamp_ms"] = depth_sensor_timestamp_ms
                        self._frames[camera_name]["rgb_depth_sensor_timestamp_delta_ms"] = color_sensor_timestamp_ms - depth_sensor_timestamp_ms

                else:
                    # 全局相机：只取 RGB
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())

                    with self._locks[camera_name]:
                        self._frames[camera_name]["color"] = color_image
                        self._frames[camera_name]["depth"] = None
                        self._frames[camera_name]["depth_vis"] = None
                        self._frames[camera_name]["timestamp"] = time.time()
                        self._frames[camera_name]["frame_number"] = int(color_frame.get_frame_number())
                        self._frames[camera_name]["sensor_timestamp_ms"] = float(color_frame.get_timestamp())
                        self._frames[camera_name]["timestamp_domain"] = str(color_frame.get_frame_timestamp_domain())
                        self._frames[camera_name]["color_frame_number"] = int(color_frame.get_frame_number())
                        self._frames[camera_name]["color_sensor_timestamp_ms"] = float(color_frame.get_timestamp())

                frame_count += 1

                now = time.time()
                if now - last_print_time > 3.0:
                    #print(f"[CameraService] {camera_name}: frames={frame_count}")
                    last_print_time = now

            except RuntimeError as e:
                self._last_frame_errors[camera_name] = str(e)
                error_count += 1
                now = time.time()
                if now - last_print_time > 3.0:
                    print(
                        f"[CameraService] {camera_name}: runtime error, "
                        f"errors={error_count}, error={e}"
                    )
                    last_print_time = now

            except Exception as e:
                self._last_frame_errors[camera_name] = str(e)
                print(f"[CameraService] Error in {camera_name} worker: {e}")
                time.sleep(0.1)

    def _display_worker(self):
        cv2.namedWindow("Global Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                global_color = self._safe_get_color("global")
                wrist_color = self._safe_get_color("wrist")

                if global_color is not None:
                    cv2.imshow("Global Camera", global_color)

                if wrist_color is not None:
                    cv2.imshow("Wrist Camera", wrist_color)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[CameraService] q pressed, stopping camera service.")
                    self._stop_event.set()
                    break

            except Exception as e:
                print(f"[CameraService] Display error: {e}")
                time.sleep(0.1)

    def _safe_get_color(self, camera_name: str):
        with self._locks[camera_name]:
            color = self._frames[camera_name]["color"]
            if color is None:
                return None
            return color.copy()

    @staticmethod
    def _make_depth_vis(depth_image: np.ndarray) -> np.ndarray:
        depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        return depth_colormap

    @staticmethod
    def _intrinsics_to_dict(intr):
        return {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }


def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()

    result = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        result.append({"name": name, "serial": serial})

    return result


if __name__ == "__main__":
    devices = list_realsense_devices()
    print("Detected RealSense devices:")
    for d in devices:
        print(d)

    service = RealSenseDualCameraService(
        global_config=CameraConfig(
            name="global",
            serial='213522251050',
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,   # 全局相机只开 RGB
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial='243222070053',
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=True,
            depth_width=1280,
            depth_height=720,
            depth_fps=30,
        ),
        save_dir="runs/current_capture",
        show_window=True,
    )

    try:
        service.start()

        print("Camera service is running.")
        print("Press ENTER to capture both cameras.")
        print("Press Ctrl+C to exit.")

        while True:
            input()

            global_result = service.capture_global()
            wrist_result = service.capture_wrist()

            print("[Capture] Global:")
            print("  color:", global_result["color_path"])
            print("  depth:", global_result["depth_path"])

            print("[Capture] Wrist:")
            print("  color:", wrist_result["color_path"])
            print("  depth:", wrist_result["depth_path"])
            print("  depth_vis:", wrist_result["depth_vis_path"])

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        service.stop()
