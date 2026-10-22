"""Stationary wrist-only or dual RGB-D capture. No arm, gripper, Agent or ROS construction.

Camera settings are read from the hardware YAML's cameras section only. Each
enabled stream has depth. Dual frames are approximately paired by host receive
time; this does not claim hardware synchronization or known cross-camera extrinsics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yaml


def _default_service(specs, save_dir):
    # This service only owns RealSense streams; do not use runtime.services or
    # hardware.agent_camera, which construct other hardware or query robot FK.
    from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
    # Disabled role configuration is never opened by the service.
    global_spec = specs.get('global', dict(specs['wrist'], name='global'))
    return RealSenseDualCameraService(
        CameraConfig(**global_spec), CameraConfig(**specs["wrist"]),
        save_dir=str(save_dir), show_window=False, startup_timeout_s=20.,
        active_cameras=tuple(specs))


def _choose_pair(service, timeout_s, max_age_s, cameras=("global", "wrist"), *, min_timestamp_s=None):
    deadline = time.monotonic() + timeout_s
    best, best_delta = None, float("inf")
    while True:
        pair = {name: service.get_latest_rgbd(name) for name in cameras}
        now = time.time()
        timestamps = [float(f["timestamp"]) for f in pair.values()]
        if (all(0 <= now-t <= max_age_s for t in timestamps)
                and (min_timestamp_s is None or min(timestamps) >= min_timestamp_s)):
            delta = max(timestamps)-min(timestamps)
            if delta < best_delta:
                best, best_delta = pair, delta
            if delta <= .015:
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(.02)
    if best is None:
        raise RuntimeError("No pair of fresh RGB-D frames received")
    return best, best_delta


def depth_quality(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    height, width = depth_m.shape
    center = depth_m[height//4:3*height//4, width//4:3*width//4]
    values = depth_m[valid]
    center_values = center[np.isfinite(center) & (center > 0)]
    return {
        "valid_depth_fraction": float(valid.mean()),
        "valid_depth_pixels": int(valid.sum()),
        "center_half_valid_depth_fraction": float((np.isfinite(center) & (center > 0)).mean()),
        "depth_p05_p50_p95_m": np.quantile(values, [.05, .5, .95]).tolist() if len(values) else None,
        "center_half_depth_median_m": float(np.median(center_values)) if len(center_values) else None,
        "scope": "image_and_center_region_quality_not_target_visibility_or_occlusion_truth",
    }


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save {path}")


def save_pair(pair, root, delta, max_skew_s):
    root = Path(root)
    capture_group = uuid.uuid4().hex
    report = {
        "schema": "camera_only_static_capture_v1", "capture_group_id": capture_group,
        "robot_control_initialized": False, "robot_state_queried": False,
        "camera_pose_available": False, "cross_view_inference_ready": False,
        "pairing": {
            "method": "closest_available_host_receive_timestamps_for_static_scene",
            "host_receive_delta_s": float(delta), "max_host_receive_delta_s": max_skew_s,
            "passes_host_receive_skew_screen": bool(delta <= max_skew_s),
            "hardware_synchronized": False, "exposure_time_synchronization_verified": False,
            "scene_stationarity": "required_user_experiment_condition_not_measured",
        },
        "cameras": {},
    }
    if len(pair) == 1:
        report['pairing'] = {'method': 'single_camera_no_pairing',
                             'hardware_synchronized': False, 'pairing_applicable': False}
    panels = []
    for name in pair:
        frame = pair[name]
        color, raw_depth = np.asarray(frame["color"]), np.asarray(frame["depth"])
        if not frame.get("depth_aligned_to_color") or raw_depth.shape != color.shape[:2]:
            raise ValueError(f"{name}: recorded, shape-matched RGB/depth alignment required")
        scale = float(frame["depth_unit_m_per_raw_unit"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("explicit positive metres-per-raw-unit depth scale required")
        depth = raw_depth.astype(np.float32)*scale
        directory = root / name
        directory.mkdir(parents=True, exist_ok=False)
        _write_image(directory / "color.png", color)
        _write_image(directory / "depth.png", raw_depth)
        np.save(directory / "depth_m.npy", depth)
        view_depth = cv2.applyColorMap(np.uint8(np.clip(depth/2., 0, 1)*255), cv2.COLORMAP_TURBO)
        view_depth[~np.isfinite(depth) | (depth <= 0)] = 0
        _write_image(directory / "depth_vis.png", view_depth)
        metadata = {
            "camera_id": name, "capture_group_id": capture_group,
            "capture_timestamp_s": float(frame["timestamp"]),
            "timestamp_domain": "shared_host_unix_receive",
            "timestamp_semantics": "host_receive_after_alignment_not_sensor_exposure",
            "sensor_timestamp_domain": frame.get("timestamp_domain"),
            "sensor_timestamp_ms": frame.get("sensor_timestamp_ms"),
            "color_sensor_timestamp_ms": frame.get("color_sensor_timestamp_ms"),
            "depth_sensor_timestamp_ms": frame.get("depth_sensor_timestamp_ms"),
            "rgb_depth_sensor_timestamp_delta_ms": frame.get("rgb_depth_sensor_timestamp_delta_ms"),
            "color_frame_number": frame.get("color_frame_number"),
            "depth_frame_number": frame.get("depth_frame_number"),
            "device_name": frame.get("device_name"), "device_serial": frame["device_serial"],
            "color_intrinsics": frame["color_intrinsics"],
            "depth_intrinsics": frame["depth_intrinsics"],
            "depth_scale": scale, "depth_unit_m_per_raw_unit": scale,
            "depth_units": "metres", "depth_definition": "optical_Z",
            "depth_aligned_to_color": True, "image_origin": "top_left",
            "reference_frame": None, "T_reference_from_camera_optical": None,
            "camera_pose_available": False, "robot_control_initialized": False,
        }
        (directory / "grasp_input.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        quality = depth_quality(depth)
        report["cameras"][name] = {
            "snapshot_dir": name, "device_serial": frame["device_serial"],
            "image_size": [color.shape[1], color.shape[0]], "depth_quality": quality,
            "depth_aligned_to_color": True, "depth_unit_m_per_raw_unit": scale,
        }
        image = np.hstack((color, view_depth))
        image = cv2.copyMakeBorder(image, 38, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(image, f"{name} {frame['device_serial']} | color / depth (0-2 m)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
    _write_image(root / "camera_preview.png", np.vstack(panels))
    (root / "capture_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def capture(config_path, output_dir=None, *, warmup_s=2., pairing_timeout_s=2.,
            max_skew_s=.05, service_factory=None, camera_mode=None):
    if not all(np.isfinite(v) and v > 0 for v in (warmup_s, pairing_timeout_s, max_skew_s)):
        raise ValueError("capture timing parameters must be finite and positive")
    config = yaml.safe_load(Path(config_path).read_text())["cameras"]
    mode = camera_mode or config.get('mode', 'dual')
    if mode not in ('wrist', 'dual'):
        raise ValueError('camera mode must be wrist or dual')
    cameras = ('wrist',) if mode == 'wrist' else ('global', 'wrist')
    serials = [config[f"{name}_serial"] for name in cameras]
    if not all(isinstance(s, str) and s for s in serials) or len(set(serials)) != len(cameras):
        raise ValueError("distinct explicitly bound camera serial strings are required")
    if output_dir is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs/occlusion_real") / f"{stamp}_{uuid.uuid4().hex[:6]}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    specs = {
        name: dict(name=name, serial=serial, enable_depth=True,
                   color_width=int(config["width"]), color_height=int(config["height"]),
                   color_fps=int(config["fps"]), depth_width=int(config["width"]),
                   depth_height=int(config["height"]), depth_fps=int(config["fps"]))
        for name, serial in zip(cameras, serials)
    }
    service = (service_factory or _default_service)(specs, output_dir / "stream_cache")
    try:
        service.start()
        time.sleep(warmup_s)
        pair, delta = _choose_pair(service, pairing_timeout_s, max_age_s=1., cameras=cameras)
    except BaseException as error:
        (output_dir / "capture_error.json").write_text(json.dumps({
            "error": str(error), "error_type": type(error).__name__,
            "robot_control_initialized": False, "requested_camera_specs": specs,
        }, indent=2))
        raise
    finally:
        service.stop()
    report = save_pair(pair, output_dir, delta, max_skew_s)
    print(json.dumps({"output_dir": str(output_dir), "capture": report}, indent=2), flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/hardware/fr3.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument('--camera-mode', choices=['wrist', 'dual'], help='Defaults to cameras.mode in hardware YAML')
    parser.add_argument("--warmup-s", type=float, default=2.)
    parser.add_argument("--pairing-timeout-s", type=float, default=2.)
    parser.add_argument("--max-skew-s", type=float, default=.05)
    args = parser.parse_args()
    capture(args.config, args.output_dir, warmup_s=args.warmup_s,
            pairing_timeout_s=args.pairing_timeout_s, max_skew_s=args.max_skew_s,
            camera_mode=args.camera_mode)


if __name__ == "__main__":
    main()
