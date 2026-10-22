"""Operator-triggered overhead ArUco calibration. Never commands robot motion."""
import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import time
import uuid
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .eye_to_hand import (atomic_json, detect_marker, reject_duplicate, solve_dataset,
                          stationary_sample, transform)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds")


class RobotReader:
    """Owns one passive subscriber subprocess, not the ROS stack or controllers."""
    def __init__(self, config):
        self.config = config
        self.process = None
        self.thread = None
        self.condition = threading.Condition()
        self.states = deque(maxlen=1000)
        self.responses = {}
        self.diagnostics = deque(maxlen=8)
        self.finished = False

    def start(self):
        if self.config.stack not in ("real", "fake"):
            raise ValueError("Unknown stack")
        source = Path(__file__).with_name("eye_to_hand_ros_worker.py").read_text()
        entry = "/fr3-real-entrypoint.sh" if self.config.stack == "real" else "/fr3-entrypoint.sh"
        command = ["docker", "compose", "-f", str(Path(self.config.workspace) / "compose.yaml"),
                   "exec", "-T", self.config.stack, "/bin/bash", entry, "python3", "-u", "-c", source]
        self.process = subprocess.Popen(command, cwd=self.config.workspace, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, start_new_session=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 12
        with self.condition:
            while not self.states:
                self._check(deadline)
                self.condition.wait(.1)

    def _read(self):
        try:
            for line in self.process.stdout:
                with self.condition:
                    if line.startswith("EYE_TO_HAND="):
                        record = json.loads(line.split("=", 1)[1])
                        if record.get("kind") == "joints":
                            age = time.monotonic() - record["received_at"]
                            if -.1 <= age < .5:
                                self.states.append(record)
                            else:
                                self.diagnostics.append("Robot bridge timestamp is stale or clock differs")
                        elif record.get("kind") == "fk":
                            self.responses[record["id"]] = record
                    else:
                        self.diagnostics.append(line.strip())
                    self.condition.notify_all()
        except Exception as exc:
            self.diagnostics.append(str(exc))
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def _check(self, deadline):
        if self.finished or time.monotonic() > deadline:
            raise RuntimeError("无法获得新鲜机器人反馈。请先启动 real.sh；本程序不会启动控制器。 "
                               + " | ".join(self.diagnostics))

    def window(self, after):
        deadline = time.monotonic() + 3
        with self.condition:
            while not self.states or self.states[-1]["received_at"] < after:
                self._check(deadline)
                self.condition.wait(.03)
            return list(self.states)

    def fk(self, q):
        identifier = uuid.uuid4().hex
        request = {"op": "fk", "id": identifier, "q": q}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + 5
        with self.condition:
            while identifier not in self.responses:
                self._check(deadline)
                self.condition.wait(.05)
            result = self.responses.pop(identifier)
        if "error" in result:
            raise RuntimeError(result["error"])
        quaternion = np.asarray(result["xyzw"], dtype=float)
        if not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-3:
            raise ValueError("FK returned invalid quaternion")
        return transform(Rotation.from_quat(quaternion).as_matrix(), result["xyz"])

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"quit"}\n')
                self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()  # Only the owned docker exec client, never real.sh.
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        if self.thread:
            self.thread.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream and not stream.closed:
                stream.close()


def opencv_intrinsics(intrinsics):
    """Do not silently treat an unsupported nonzero distortion model as pinhole."""
    k = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1.]])
    d = np.asarray(intrinsics.coeffs, dtype=float)
    model = str(intrinsics.model).split(".")[-1]
    if not np.isfinite(k).all() or not np.isfinite(d).all() or min(intrinsics.fx, intrinsics.fy) <= 0:
        raise ValueError("相机内参无效")
    if np.any(d != 0) and model not in ("brown_conrady",):
        raise ValueError(f"当前彩色流为非零 {model} 畸变，不能直接当作 OpenCV Brown 模型。"
                         "需先做该模型的去畸变适配；没有使用错误内参继续标定。")
    return k, d, {"width": intrinsics.width, "height": intrinsics.height,
                   "camera_matrix": k.tolist(), "dist_coeffs": d.tolist(),
                   "realsense_distortion_model": model}


class ColorCamera:
    def __init__(self, serial, width, height, fps):
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self.pipeline = None
        self.last_frame = None

    def start(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            profile = self.pipeline.start(config)
            actual = profile.get_device().get_info(rs.camera_info.serial_number)
            if actual != self.serial:
                raise RuntimeError("相机序列号与请求不符")
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.k, self.d, self.metadata = opencv_intrinsics(intrinsics)
            self.metadata["fps"] = self.fps
            for _ in range(8):
                self.frame()
        except BaseException:
            self.close()
            raise

    def frame(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            frame = self.pipeline.wait_for_frames(timeout_ms=2000).get_color_frame()
            if frame and frame.get_frame_number() != self.last_frame:
                self.last_frame = frame.get_frame_number()
                received = time.monotonic()
                return np.asanyarray(frame.get_data()).copy(), {
                    "received_at": received, "frame_number": frame.get_frame_number(),
                    "device_timestamp_ms": frame.get_timestamp(),
                    "timestamp_domain": str(frame.get_frame_timestamp_domain())}
        raise TimeoutError("没有收到新的彩色图像")

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None


def create_detector(dictionary_name):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"OpenCV 不支持 ArUco 字典 {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters), dictionary


def solve_file(path, output=None):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    result = solve_dataset(data)
    result.update(created_at=now(), source_samples=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                  calibration_sample_ids=[s["id"] for s in data["samples"] if s["role"] == "calibration"],
                  validation_sample_ids=[s["id"] for s in data["samples"] if s["role"] == "validation"])
    output = Path(output).expanduser().resolve() if output else path.with_name("result.json")
    if output == path:
        raise ValueError("结果文件不能覆盖原始样本文件")
    atomic_json(output, result)
    print(f"\n结果：{output}\n状态：{result['status']}", flush=True)
    for key in ("training", "cross_validation", "validation"):
        report = result[key]
        if report:
            print(f"{key}: n={report['count']}, RMS={report['position_rms_mm']:.2f} mm / "
                  f"{report['rotation_rms_deg']:.2f} deg; max={report['position_max_mm']:.2f} mm / "
                  f"{report['rotation_max_deg']:.2f} deg")
    print("base_T_camera =\n" + np.array2string(np.array(result["base_T_camera"]), precision=7))
    if result["status"] != "validated":
        print("尚未通过独立验证：可增加不同姿态的标定样本，再用 V 采集至少 3 个独立验证样本。")
    print("未覆盖已有腕部标定或抓取配置。", flush=True)
    return result


def collect_sample(camera, robot, detector, board, samples, role, max_error):
    # Discard the queued frame; then observe a fresh ~0.7 s stationary interval.
    camera.frame()
    frames, images = [], []
    started = time.monotonic()
    while len(frames) < 5 or time.monotonic() - started < .7:
        image, metadata = camera.frame()
        record = detect_marker(image, detector, board, camera.k, camera.d, max_error)
        frames.append({**metadata, **record})
        images.append(image)
    states = robot.window(frames[-1]["received_at"] + .08)
    frame, state, stability = stationary_sample(states, frames)
    base_flange = robot.fk(state["q"])
    reject_duplicate(base_flange, samples)
    sample = {**frame, "id": uuid.uuid4().hex, "role": role, "captured_at": now(),
              "base_T_flange": base_flange.tolist(), "joint_state": state, "stability": stability}
    return sample, images[len(images) // 2]


def collect(args):
    config = load_config(args.config)
    if config.arm.stack != "real":
        raise ValueError("实机相机标定必须读取 real 状态，不能用 fake 关节状态拟合真实外参")
    if not math.isfinite(args.marker_length_mm) or not 0 < args.marker_length_mm <= 1000:
        raise ValueError("--marker-length-mm 必须为有效毫米尺寸 (0,1000]")
    if not math.isfinite(args.max_reprojection_px) or args.max_reprojection_px <= 0:
        raise ValueError("重投影阈值必须为有限正数")
    detector, dictionary = create_detector(args.dictionary)
    if not 0 <= args.marker_id < len(dictionary.bytesList):
        raise ValueError("marker ID 不在所选字典中")
    serial = args.serial or config.cameras.global_serial
    if serial == config.cameras.wrist_serial:
        raise ValueError("此程序用于固定俯视相机；指定的序列号却是腕部相机")
    board = {"kind": "aruco", "dictionary": args.dictionary, "marker_id": args.marker_id,
             "marker_length_m": args.marker_length_mm / 1000,
             "frame": "marker_center_x_right_y_up_z_out_of_print"}
    directory = Path(args.session_dir).expanduser().resolve() if args.session_dir else (
        PROJECT_ROOT / "runs/eye_to_hand" / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                           + "-" + uuid.uuid4().hex[:6]))
    directory.mkdir(parents=True, exist_ok=False)
    camera = ColorCamera(serial, config.cameras.width, config.cameras.height, config.cameras.fps)
    robot = RobotReader(config.arm)
    window = "Overhead eye-to-hand calibration (read only)"
    data = {"schema": "fr3_eye_to_hand_samples_v1", "created_at": now(), "camera_serial": serial,
            "camera_frame": "global_color_optical_frame", "base_frame": "fr3_link0",
            "flange_frame": "fr3_link8", "translation_unit": "m", "board": board,
            "robot_pose_source": "ROS joint_states + MoveIt compute_fk, never O_T_EE/TCP",
            "stack": config.arm.stack, "samples": []}
    path = directory / "samples.json"
    message = "S: sample | V: validation | C: solve | U: undo | Q: quit"
    print(f"俯视相机 {serial}；{args.dictionary} ID={args.marker_id}；边长 {args.marker_length_mm:g} mm\n"
          f"数据目录：{directory}\n"
          "请先固定夹持标定板，采样期间不要开合夹爪或改变相机位置。\n"
          "先停稳，再按 S 保存；V 保存独立验证样本；C 计算；U 撤销最后一条；Q 退出。\n"
          "建议 15–25 个标定姿态和至少 3 个独立验证姿态；绕不同轴倾斜，不要只做平移。\n"
          "本程序只读机器人状态和计算 FK，不激活控制器，不控制机械臂或夹爪。", flush=True)
    try:
        robot.start()
        camera.start()
        data["intrinsics"] = camera.metadata
        atomic_json(path, data)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1000, 750)
        while True:
            image, _ = camera.frame()
            preview = image.copy()
            valid = False
            try:
                detection = detect_marker(image, detector, board, camera.k, camera.d, args.max_reprojection_px)
                points = np.array(detection["corners_px"], np.int32)
                cv2.polylines(preview, [points], True, (0, 220, 0), 2)
                pose = np.asarray(detection["camera_T_marker"])
                cv2.drawFrameAxes(preview, camera.k, camera.d, cv2.Rodrigues(pose[:3, :3])[0],
                                  pose[:3, 3], board["marker_length_m"] * .5)
                marker_text = f"ID {args.marker_id}: OK, reprojection {detection['reprojection_rms_px']:.2f} px"
                valid = True
            except ValueError:
                marker_text = "Marker missing / small / ambiguous / unstable"
            n = sum(s["role"] == "calibration" for s in data["samples"])
            v = len(data["samples"]) - n
            panel = np.zeros((125, preview.shape[1], 3), np.uint8)
            for row, label in enumerate([f"{serial} | calibration {n} | validation {v}", marker_text,
                                          "S sample  V validation  C solve  U undo  Q quit", message]):
                cv2.putText(panel, label, (8, 25 + row * 29), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            (100, 240, 100) if valid else (80, 200, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, np.vstack((preview, panel)))
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S"), ord("v"), ord("V")):
                role = "calibration" if key in (ord("s"), ord("S")) else "validation"
                print("采样中，请保持机械臂和标定板静止…", flush=True)
                try:
                    sample, saved_image = collect_sample(camera, robot, detector, board,
                                                         data["samples"], role, args.max_reprojection_px)
                    filename = sample["id"] + ".png"
                    if not cv2.imwrite(str(directory / filename), saved_image):
                        raise RuntimeError("保存图像失败")
                    sample["image_file"] = filename
                    data["samples"].append(sample)
                    atomic_json(path, data)
                    message = f"Saved {role} #{len(data['samples'])}"
                    print(message, flush=True)
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    message = "Sample rejected; see terminal"
                    print(f"样本未保存：{exc}", flush=True)
            elif key in (ord("u"), ord("U")) and data["samples"]:
                removed = data["samples"].pop()
                data.setdefault("excluded_samples", []).append(removed)
                atomic_json(path, data)
                message = "Last sample excluded (image preserved)"
                print("已撤销最后一条样本，原图和 excluded_samples 记录仍保留。", flush=True)
            elif key in (ord("c"), ord("C")):
                try:
                    result = solve_file(path)
                    message = "Calibration: " + result["status"]
                except (ValueError, cv2.error) as exc:
                    message = "Calibration failed; see terminal"
                    print(f"标定未完成：{exc}", flush=True)
    finally:
        camera.close()
        robot.close()
        cv2.destroyAllWindows()
    print(f"采样已结束。已保存数据：{path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="FR3 固定俯视相机眼在手外标定；不会发送机械臂/夹爪运动命令")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect", help="打开俯视相机，手动静止采样和标定")
    capture.add_argument("--config", default=str(DEFAULT_CONFIG))
    capture.add_argument("--serial", help="默认硬件配置的 global_serial")
    capture.add_argument("--dictionary", default="DICT_4X4_50")
    capture.add_argument("--marker-id", type=int, default=20)
    capture.add_argument("--marker-length-mm", type=float, default=95,
                         help="黑色码的外边长，不含白边；用户于 2026-10-22 实测为 95 mm")
    capture.add_argument("--max-reprojection-px", type=float, default=1.5)
    capture.add_argument("--session-dir", help="新建数据目录，必须尚不存在")
    offline = commands.add_parser("solve", help="只读保存的样本离线计算，不连接任何硬件")
    offline.add_argument("samples")
    offline.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            return collect(args)
        result = solve_file(args.samples, args.output)
        return 2 if result["status"] == "quality_failed" else 0
    except KeyboardInterrupt:
        print("已中断，已保存样本保留。")
        return 130
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"Stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
