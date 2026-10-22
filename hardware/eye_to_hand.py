"""Offline eye-to-hand calibration: B_T_F @ F_T_M = B_T_C @ C_T_M.

Column vectors, metres. B=fr3_link0, F=fr3_link8, C=color optical,
M=ArUco marker. No camera, robot connection, or motion on import/solve.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import rigid_matrix


def inverse(t):
    t = rigid_matrix(t, "transform")
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def transform(rotation, translation):
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation).reshape(3, 3)
    out[:3, 3] = np.asarray(translation).reshape(3)
    return rigid_matrix(out, "transform")


def difference(a, b):
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def average(matrices):
    return transform(Rotation.from_matrix(np.array([t[:3, :3] for t in matrices])).mean().as_matrix(),
                     np.mean([t[:3, 3] for t in matrices], axis=0))


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def diversity(samples):
    poses = [rigid_matrix(s["base_T_flange"], "base_T_flange") for s in samples]
    vectors = []
    for i, a in enumerate(poses):
        for b in poses[i + 1:]:
            vectors.append(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec())
    vectors = np.asarray(vectors).reshape(-1, 3)
    singular = np.linalg.svd(vectors, compute_uv=False) if len(vectors) else np.zeros(3)
    max_angle = float(np.max(np.linalg.norm(vectors, axis=1))) if len(vectors) else 0.
    second_ratio = float(singular[1] / singular[0]) if len(singular) > 1 and singular[0] > 0 else 0.
    # Translation alone or rotations about a single axis do not determine the extrinsic.
    if max_angle < math.radians(15) or second_ratio < .15:
        raise ValueError("姿态变化不足：需绕至少两个不同轴转动，不能只平移或始终保持标定板水平。"
                         f" 最大相对转角={math.degrees(max_angle):.1f}°，第二旋转分量比例={second_ratio:.3f}")
    return {"max_relative_rotation_deg": math.degrees(max_angle),
            "rotation_singular_values": singular.tolist(), "second_axis_ratio": second_ratio}


def fit(samples):
    diversity(samples)
    # OpenCV eye-to-hand formulation takes F_T_B in its gripper2base argument.
    # The returned matrix is B_T_C (not its inverse).
    f_b = [inverse(s["base_T_flange"]) for s in samples]
    c_m = [rigid_matrix(s["camera_T_marker"], "camera_T_marker") for s in samples]
    rotation, translation = cv2.calibrateHandEye(
        [t[:3, :3] for t in f_b], [t[:3, 3] for t in f_b],
        [t[:3, :3] for t in c_m], [t[:3, 3] for t in c_m],
        method=cv2.CALIB_HAND_EYE_PARK)
    b_c = transform(rotation, translation)
    f_m = average([a @ b_c @ c for a, c in zip(f_b, c_m)])
    return b_c, f_m


def residuals(samples, b_c, f_m):
    rows = []
    for sample in samples:
        b_f = rigid_matrix(sample["base_T_flange"], "base_T_flange")
        c_m = rigid_matrix(sample["camera_T_marker"], "camera_T_marker")
        position, angle = difference(b_f @ f_m, b_c @ c_m)
        rows.append({"sample_id": sample["id"], "position_mm": position * 1000,
                     "rotation_deg": math.degrees(angle)})
    if not rows:
        return None
    positions = np.array([r["position_mm"] for r in rows])
    angles = np.array([r["rotation_deg"] for r in rows])
    return {"count": len(rows), "position_rms_mm": float(np.sqrt(np.mean(positions**2))),
            "position_max_mm": float(max(positions)),
            "rotation_rms_deg": float(np.sqrt(np.mean(angles**2))),
            "rotation_max_deg": float(max(angles)), "per_sample": rows}


def passes(report):
    return bool(report and report["position_rms_mm"] <= 5 and report["position_max_mm"] <= 10
                and report["rotation_rms_deg"] <= 2 and report["rotation_max_deg"] <= 3)


def solve_dataset(data):
    if data.get("schema") != "fr3_eye_to_hand_samples_v1":
        raise ValueError("不是本程序的眼在手外采样文件")
    if data.get("base_frame") != "fr3_link0" or data.get("flange_frame") != "fr3_link8":
        raise ValueError("必须使用 fr3_link0 / fr3_link8，不接受未标注的 TCP 或 O_T_EE")
    if data.get("translation_unit") != "m" or data.get("camera_frame") != "global_color_optical_frame":
        raise ValueError("必须使用米及俯视相机彩色光学坐标系")
    samples = data["samples"]
    if any(s.get("role") not in ("calibration", "validation") for s in samples):
        raise ValueError("样本 role 必须为 calibration 或 validation")
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("重复的样本 ID")
    training = [s for s in samples if s["role"] == "calibration"]
    validation = [s for s in samples if s["role"] == "validation"]
    if len(training) < 8:
        raise ValueError(f"只有 {len(training)} 个标定样本，至少需要 8 个，建议 15–25 个")
    coverage = diversity(training)
    b_c, f_m = fit(training)
    training_report = residuals(training, b_c, f_m)
    validation_report = residuals(validation, b_c, f_m)
    # A deterministic held-out subset is never used for fitting its own prediction.
    cross_report = None
    cross_error = None
    if len(training) >= 12:
        held_out = [s for i, s in enumerate(training) if i % 4 == 3]
        fit_set = [s for i, s in enumerate(training) if i % 4 != 3]
        try:
            cross_b_c, cross_f_m = fit(fit_set)
            cross_report = residuals(held_out, cross_b_c, cross_f_m)
        except (ValueError, cv2.error) as exc:
            cross_error = str(exc)
    validated = len(validation) >= 3 and passes(validation_report)
    status = "validated" if passes(training_report) and validated else "needs_validation"
    if not passes(training_report) or (validation_report and not passes(validation_report)):
        status = "quality_failed"
    if cross_error or (cross_report and not passes(cross_report)):
        status = "quality_failed"
    return {"schema": "fr3_eye_to_hand_result_v1", "status": status,
            "method": "OpenCV PARK; inverse(base_T_flange), camera_T_marker",
            "translation_unit": "m", "matrix_convention": "p_parent = T_parent_child @ p_child",
            "equation": "base_T_flange @ flange_T_marker = base_T_camera @ camera_T_marker",
            "camera_serial": data["camera_serial"], "camera_frame": data["camera_frame"],
            "base_frame": data["base_frame"], "flange_frame": data["flange_frame"],
            "board": data["board"], "intrinsics": data["intrinsics"],
            "base_T_camera": b_c.tolist(), "camera_T_base": inverse(b_c).tolist(),
            "flange_T_marker": f_m.tolist(), "pose_diversity": coverage,
            "training": training_report, "validation": validation_report,
            "cross_validation": cross_report, "cross_validation_error": cross_error,
            "quality_thresholds": {"position_rms_mm": 5, "position_max_mm": 10,
                                   "rotation_rms_deg": 2, "rotation_max_deg": 3},
            "note": "残差是样本一致性指标，不是绝对精度保证。结果未自动写入抓取配置。"
                    "相机、机器人基座位置改变后需重标定；采样中标定板不能相对法兰滑动。"}


def detect_marker(image, detector, board, k, distortion, max_reprojection_px=1.5):
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None:
        raise ValueError("没有检测到 ArUco")
    indices = np.flatnonzero(ids.reshape(-1) == board["marker_id"])
    if len(indices) != 1:
        raise ValueError(f"需要且只能有一个 ID={board['marker_id']} 的标记")
    points = np.asarray(corners[indices[0]], dtype=np.float64).reshape(4, 2)
    if np.min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) < 30:
        raise ValueError("标记太小或倾斜太大，请让标记在图像中更清晰")
    half = board["marker_length_m"] / 2
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    ok, rotations, translations, _ = cv2.solvePnPGeneric(obj, points, k, distortion,
                                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
    choices = []
    if ok:
        for rvec, tvec in zip(rotations, translations):
            r = cv2.Rodrigues(rvec)[0]
            if np.min((obj @ r.T + tvec.reshape(3))[:, 2]) <= 0:
                continue
            projected = cv2.projectPoints(obj, rvec, tvec, k, distortion)[0].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((projected - points)**2, axis=1))))
            if math.isfinite(rms):
                choices.append((rms, transform(r, tvec)))
    if not choices:
        raise ValueError("PnP 没有有效解")
    choices.sort(key=lambda v: v[0])
    error, pose = choices[0]
    if error > max_reprojection_px:
        raise ValueError(f"重投影误差 {error:.2f}px 超过 {max_reprojection_px:.2f}px")
    if len(choices) > 1:
        angle = difference(pose, choices[1][1])[1]
        if choices[1][0] - error < .2 and angle > math.radians(10):
            raise ValueError("平面标记存在姿态歧义，请稍微改变倾角或改善光照")
    return {"camera_T_marker": pose.tolist(), "corners_px": points.tolist(),
            "reprojection_rms_px": error}


def stationary_sample(states, frame_records):
    """Require continuous robot coverage around a newly captured image burst."""
    begin, end = frame_records[0]["received_at"], frame_records[-1]["received_at"]
    window = [s for s in states if begin - .2 <= s["received_at"] <= end + .2]
    if len(window) < 5 or window[0]["received_at"] > begin or window[-1]["received_at"] < end:
        raise ValueError("机器人反馈未覆盖整个取图时段，请检查 real.sh 和状态话题")
    if max(np.diff([s["received_at"] for s in window])) > .2:
        raise ValueError("机器人反馈中断，样本未保存")
    positions = np.asarray([s["q"] for s in window], dtype=float)
    velocities = np.asarray([s["dq"] for s in window], dtype=float)
    if positions.shape[1:] != (7,) or velocities.shape != positions.shape:
        raise ValueError("关节反馈维度无效")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("关节反馈包含非有限数")
    if np.max(np.abs(velocities)) > .01 or np.max(np.ptp(positions, axis=0)) > .002:
        raise ValueError("机械臂仍在移动：请完全静止后再采样")
    chosen = frame_records[len(frame_records) // 2]
    reference = rigid_matrix(chosen["camera_T_marker"], "camera_T_marker")
    for frame in frame_records:
        delta, angle = difference(reference, rigid_matrix(frame["camera_T_marker"], "camera_T_marker"))
        if delta > .003 or angle > math.radians(2):
            raise ValueError("标定板检测不稳定或仍在移动，请固定夹持并等待静止")
    state = min(window, key=lambda s: abs(s["received_at"] - chosen["received_at"]))
    if abs(state["received_at"] - chosen["received_at"]) > .1:
        raise ValueError("图像与关节反馈时差过大")
    return chosen, state, {"max_joint_speed_rad_s": float(np.max(np.abs(velocities))),
                           "max_joint_span_rad": float(np.max(np.ptp(positions, axis=0))),
                           "image_state_receipt_delta_s": abs(state["received_at"] - chosen["received_at"]),
                           "window_duration_s": end - begin, "joint_samples": len(window)}


def reject_duplicate(pose, samples):
    for old in samples:
        distance, angle = difference(pose, rigid_matrix(old["base_T_flange"], "base_T_flange"))
        if distance < .005 and angle < math.radians(3):
            raise ValueError("此姿态与已有样本太接近，请换位置或转动标定板")
