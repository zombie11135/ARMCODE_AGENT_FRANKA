# skill_lab/run_medicine_pick_loop.py

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import shutil
import argparse
import subprocess
import builtins
from pathlib import Path
from pprint import pprint
from datetime import datetime

from xarm.wrapper import XArmAPI

from core.context import RuntimeContext

from services.realsense_dual_camera_service import (
    RealSenseDualCameraService,
    CameraConfig,
)
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.sam3_bbox_segmenter_service import SAM3BBoxSegmenterService
from services.ocr_service import PaddleOCRService

from skill_lib.combound.find_obj_by_text import FindObjectByTextSkill


# ============================================================
# 输出过滤：去掉 CameraService frames=xxx 刷屏
# ============================================================

def install_print_filter():
    old_print = builtins.print

    def filtered_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.startswith("[CameraService]") and "frames=" in text:
            return
        old_print(*args, **kwargs)

    builtins.print = filtered_print


def print_event(event: dict):
    event_type = event.get("type")

    if event_type == "image":
        print(f"[Image] {event.get('content')}: {event.get('image_path')}")
    elif event_type == "text":
        print(f"[Text] {event.get('content')}")
    elif event_type == "warning":
        print(f"[Warning] {event.get('content')}")
    elif event_type == "error":
        print(f"[Error] {event.get('content')}")
    else:
        print(event)


# ============================================================
# xArm
# ============================================================

def check_code(arm, code, label=""):
    if code != 0:
        print(f"[ERROR] {label} failed, code={code}")
        print(f"        arm state={arm.state}, error={arm.error_code}, warn={arm.warn_code}")
        return False
    return True


def init_arm(ip: str, is_radian: bool = True):
    arm = XArmAPI(ip, is_radian=is_radian)
    time.sleep(0.5)

    print("[xArm] Connected:", arm.connected)
    print("[xArm] State:", arm.state)
    print("[xArm] Error code:", arm.error_code)
    print("[xArm] Warn code:", arm.warn_code)

    arm.clean_warn()
    arm.clean_error()

    check_code(arm, arm.motion_enable(enable=True), "motion_enable")
    check_code(arm, arm.set_mode(0), "set_mode")
    check_code(arm, arm.set_state(0), "set_state")

    time.sleep(0.5)
    return arm


def get_current_xarm_pose_mrad(arm):
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None or len(pose) < 6:
        raise RuntimeError(f"获取 xArm 当前位姿失败, code={code}, pose={pose}")

    x_mm, y_mm, z_mm, rx, ry, rz = pose[:6]

    return [
        float(x_mm) / 1000.0,
        float(y_mm) / 1000.0,
        float(z_mm) / 1000.0,
        float(rx),
        float(ry),
        float(rz),
    ]


def move_to_pose(
    arm,
    pose,
    speed: float = 50,
    acc: float = 500,
    wait: bool = True,
    label: str = "",
):
    if pose is None or len(pose) != 6:
        raise ValueError(f"pose 必须是 [x, y, z, rx, ry, rz]，当前为: {pose}")

    x, y, z, rx, ry, rz = [float(v) for v in pose]

    print(f"\n[xArm] Move {label}")
    print(f"       x={x:.3f}, y={y:.3f}, z={z:.3f}")
    print(f"       rx={rx:.6f}, ry={ry:.6f}, rz={rz:.6f}")

    code, current_pose = arm.get_position(is_radian=True)
    if code == 0:
        print("[xArm] Current pose:", current_pose)
    else:
        print("[xArm][WARN] Failed to get current pose")

    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=rx,
        pitch=ry,
        yaw=rz,
        speed=speed,
        mvacc=acc,
        wait=wait,
    )

    if not check_code(arm, code, f"set_position {label}"):
        return False

    print("[xArm] Move finished")
    return True

# ============================================================
# ROS gripper service
# ============================================================

def call_grip_service(
    value: int,
    ros_ws: str = "/home/wpy/gt200_ws",
    service_name: str = "/grip",
    timeout: float = 5.0,
    ignore_error: bool = True,
):
    """
    通过 ROS service 控制夹爪。

    当前已测试：
        rosservice call /grip 0  # 关夹爪
        rosservice call /grip 1  # 开夹爪

    注意：
        /grip 服务端可能会返回：
        field num2 must be an integer type

        但如果夹爪实际已经动作，展示阶段可以忽略该错误。
    """

    value = int(value)

    cmd = (
        f"source {ros_ws}/devel/setup.bash && "
        f"rosservice call {service_name} {value}"
    )

    print(f"\n[Gripper] {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

        print("[Gripper] output:")
        print(result.stdout)

        if result.returncode != 0:
            print(
                f"[Gripper][WARN] rosservice call returned non-zero code={result.returncode}"
            )

            if ignore_error:
                print("[Gripper][WARN] 忽略该错误，继续执行后续流程。")
                return True

            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"[Gripper][ERROR] rosservice call timeout: {service_name}")

        if ignore_error:
            print("[Gripper][WARN] 忽略 timeout，继续执行后续流程。")
            return True

        return False

    except Exception as e:
        print(f"[Gripper][ERROR] failed to call grip service: {e}")

        if ignore_error:
            print("[Gripper][WARN] 忽略该异常，继续执行后续流程。")
            return True

        return False

def open_gripper(args):
    if not args.enable_grip:
        print("[Gripper] enable_grip=False, skip open.")
        return True

    return call_grip_service(
        value=args.grip_open_value,
        ros_ws=args.ros_ws,
        service_name=args.grip_service,
        timeout=args.grip_timeout,
        ignore_error=args.ignore_grip_error,
    )


def close_gripper(args):
    if not args.enable_grip:
        print("[Gripper] enable_grip=False, skip close.")
        return True

    return call_grip_service(
        value=args.grip_close_value,
        ros_ws=args.ros_ws,
        service_name=args.grip_service,
        timeout=args.grip_timeout,
        ignore_error=args.ignore_grip_error,
    )


def go_home(arm, home_pose, speed=80, acc=500):
    print("\n[xArm] Going home...")
    return move_to_pose(
        arm,
        home_pose,
        speed=speed,
        acc=acc,
        wait=True,
        label="home",
    )


# ============================================================
# grasp_input 构造与 snapshot
# ============================================================

def normalize_intrinsics(intrinsics, fallback_intrinsics=None):
    if intrinsics is None:
        return fallback_intrinsics

    if isinstance(intrinsics, dict):
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        cx = intrinsics.get("cx", intrinsics.get("ppx"))
        cy = intrinsics.get("cy", intrinsics.get("ppy"))

        if fx is not None and fy is not None and cx is not None and cy is not None:
            return {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }

    return fallback_intrinsics


def existing_or_none(path):
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    return None


def build_grasp_input_from_find_result(find_data: dict, drug_name: str, fallback_intrinsics: dict):
    selected = find_data["selected_instance"]
    seg = find_data["segmentation_result"]

    color_path = (
        seg.get("color_path")
        or seg.get("image_path")
        or selected.get("image_path")
        or "runs/current_capture/wrist_color.png"
    )
    color_path = existing_or_none(color_path)
    if color_path is None:
        raise FileNotFoundError("找不到 color_path。")

    # 为了展示稳定性：优先用 png depth，不乱用可能残留的旧 npy
    depth_path = (
        seg.get("depth_path")
        or "runs/current_capture/wrist_depth.png"
    )
    depth_path = existing_or_none(depth_path)

    depth_npy_path = None
    if depth_path is None:
        depth_npy_path = existing_or_none(
            seg.get("depth_npy_path") or seg.get("depth_meter_path")
        )

    if depth_path is None and depth_npy_path is None:
        raise FileNotFoundError("找不到 depth_path/depth_npy_path。")

    mask_path = existing_or_none(selected.get("mask_path"))
    mask_npy_path = existing_or_none(selected.get("mask_npy_path"))

    if mask_path is None and mask_npy_path is None:
        raise FileNotFoundError("找不到目标 mask_path/mask_npy_path。")

    intrinsics = (
        seg.get("depth_intrinsics")
        or seg.get("color_intrinsics")
        or seg.get("camera_intrinsics")
    )
    intrinsics = normalize_intrinsics(intrinsics, fallback_intrinsics=fallback_intrinsics)

    if intrinsics is None:
        raise RuntimeError("缺少相机内参，请检查 RealSense 输出或 --fx --fy --cx --cy 参数。")

    grasp_input = {
        "object_id": selected.get("object_id"),
        "target_name": selected.get("target_name"),
        "query": drug_name,

        "color_path": color_path,
        "depth_path": depth_path,
        "depth_npy_path": depth_npy_path,

        "mask_path": mask_path,
        "mask_npy_path": mask_npy_path,

        "bbox": selected.get("bbox"),
        "crop_path": selected.get("crop_path"),
        "overlay_path": selected.get("overlay_path"),

        "depth_intrinsics": intrinsics,
        "color_intrinsics": intrinsics,
        "camera_intrinsics": intrinsics,

        # depth png 通常是 uint16 mm，因此 scale=1000
        "depth_scale": seg.get("depth_scale", 1000.0),

        "ocr_text": selected.get("ocr_text"),
        "ocr_texts": selected.get("ocr_texts"),
    }

    return grasp_input


def copy_file_to_snapshot(src, dst):
    if src is None:
        return None

    src_path = Path(src)
    if not src_path.exists():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def snapshot_grasp_input_files(grasp_input: dict, drug_name: str, root_dir="runs/grasp_snapshots"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in drug_name).strip("_")
    if not safe_name:
        safe_name = "drug"

    snapshot_dir = Path(root_dir) / f"{timestamp}_{safe_name[:32]}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    new_input = dict(grasp_input)

    file_map = {
        "color_path": "color",
        "depth_path": "depth",
        "depth_npy_path": "depth",
        "mask_path": "mask",
        "mask_npy_path": "mask",
        "crop_path": "crop",
        "overlay_path": "overlay",
    }

    for key, stem in file_map.items():
        src = grasp_input.get(key)
        if not src:
            continue

        src_path = Path(src)
        if not src_path.exists():
            print(f"[Snapshot][WARN] {key} 不存在，跳过: {src}")
            continue

        dst = snapshot_dir / f"{stem}{src_path.suffix}"
        copied = copy_file_to_snapshot(src_path, dst)
        if copied:
            new_input[key] = copied

    grasp_input_json = snapshot_dir / "grasp_input.json"
    with open(grasp_input_json, "w", encoding="utf-8") as f:
        json.dump(new_input, f, ensure_ascii=False, indent=2)

    print("\n=== Snapshot Grasp Input ===")
    print("snapshot_dir:", snapshot_dir)
    print("color_path:", new_input.get("color_path"))
    print("depth_path:", new_input.get("depth_path"))
    print("depth_npy_path:", new_input.get("depth_npy_path"))
    print("mask_path:", new_input.get("mask_path"))
    print("mask_npy_path:", new_input.get("mask_npy_path"))

    return new_input, snapshot_dir, grasp_input_json


def check_grasp_input(grasp_input):
    required_groups = [
        ("color_path",),
        ("depth_path", "depth_npy_path"),
        ("mask_path", "mask_npy_path"),
        ("depth_intrinsics", "color_intrinsics", "camera_intrinsics"),
    ]

    missing = []
    for group in required_groups:
        if not any(grasp_input.get(k) for k in group):
            missing.append(group)

    if missing:
        print("\n[Debug] grasp_input:")
        pprint(grasp_input)
        raise RuntimeError(f"grasp_input 缺少字段组: {missing}")

    print("\n=== Grasp Input Check ===")
    print("color_path:", grasp_input.get("color_path"))
    print("depth_path:", grasp_input.get("depth_path"))
    print("depth_npy_path:", grasp_input.get("depth_npy_path"))
    print("mask_path:", grasp_input.get("mask_path"))
    print("mask_npy_path:", grasp_input.get("mask_npy_path"))
    print("bbox:", grasp_input.get("bbox"))
    print("intrinsics:", grasp_input.get("depth_intrinsics") or grasp_input.get("color_intrinsics"))
    print("ocr_text:", grasp_input.get("ocr_text"))


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# AnyGrasp 子进程 worker
# ============================================================

WORKER_CODE = r'''
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import argparse
from pathlib import Path
import numpy as np

def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--grasp_input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--anygrasp_dir", required=True)
    parser.add_argument("--anygrasp_ckpt", required=True)
    parser.add_argument("--xarm_ip", required=True)
    parser.add_argument("--handeye_config", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lims_margin", type=float, default=0.03)
    parser.add_argument("--min_mask_points", type=int, default=200)
    parser.add_argument("--approach_clearance_mm", type=float, default=80.0)
    parser.add_argument("--lift_after_grasp_mm", type=float, default=120.0)
    args = parser.parse_args()

    sys.path.insert(0, args.project_root)

    from core.context import RuntimeContext
    from services.anygrasp_service import AnyGraspService
    from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill

    with open(args.grasp_input_json, "r", encoding="utf-8") as f:
        grasp_input = json.load(f)

    print("[Worker] Loading AnyGraspService...")
    anygrasp_service = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )

    try:
        context = RuntimeContext(anygrasp=anygrasp_service)
    except TypeError:
        context = RuntimeContext()
        context.anygrasp = anygrasp_service

    skill = PlanGraspWithAnyGraspSkill(context)

    result = skill.run(
        grasp_input=grasp_input,

        # 关键：已经从主进程拿到了当前末端位姿，worker 不再连接 xArm
        current_ee_pose_mrad=grasp_input.get("current_ee_pose_mrad"),
        xarm_ip=args.xarm_ip,
        handeye_config_path=args.handeye_config,

        # 和你单测可跑通的配置保持一致
        use_mask_points=True,
        min_mask_points=args.min_mask_points,
        auto_lims_from_mask=True,
        lims_margin=args.lims_margin,

        top_k=args.top_k,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,

        approach_clearance_mm=args.approach_clearance_mm,
        lift_after_grasp_mm=args.lift_after_grasp_mm,

        enable_pose_fix=True,
    )

    payload = {
        "success": bool(result.success),
        "message": result.message,
        "error": result.error,
        "data": result.data,
    }
    payload = jsonable(payload)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[Worker] success:", payload["success"])
    print("[Worker] message:", payload["message"])
    print("[Worker] error:", payload["error"])

    if not payload["success"]:
        sys.exit(2)

if __name__ == "__main__":
    main()
'''


def write_worker_script(snapshot_dir: Path):
    worker_path = snapshot_dir / "_anygrasp_worker.py"
    worker_path.write_text(WORKER_CODE, encoding="utf-8")
    return worker_path


def run_anygrasp_worker(args, snapshot_dir: Path, grasp_input_json: Path):
    worker_path = write_worker_script(snapshot_dir)
    output_json = snapshot_dir / "anygrasp_plan_result.json"

    cmd = [
        sys.executable,
        str(worker_path),
        "--project_root", args.project_root,
        "--grasp_input_json", str(grasp_input_json),
        "--output_json", str(output_json),
        "--anygrasp_dir", args.anygrasp_dir,
        "--anygrasp_ckpt", args.anygrasp_ckpt,
        "--xarm_ip", args.xarm_ip,
        "--handeye_config", args.handeye_config,
        "--top_k", str(args.top_k),
        "--lims_margin", str(args.lims_margin),
        "--min_mask_points", str(args.min_mask_points),
        "--approach_clearance_mm", str(args.approach_clearance_mm),
        "--lift_after_grasp_mm", str(args.lift_after_grasp_mm),
    ]

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONBREAKPOINT"] = "0"

    print("\n=== Running AnyGrasp Worker ===")
    print(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=args.project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print("\n=== AnyGrasp Worker Output ===")
    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"AnyGrasp worker failed, returncode={proc.returncode}")

    with open(output_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload.get("success", False):
        raise RuntimeError(f"AnyGrasp worker planning failed: {payload.get('error')}")

    return payload, output_json


# ============================================================
# Runtime
# ============================================================

def build_runtime(args):
    print("\n=== Starting Camera Service ===")

    camera_service = RealSenseDualCameraService(
        global_config=CameraConfig(
            name="global",
            serial=args.global_serial,
            color_width=args.global_width,
            color_height=args.global_height,
            color_fps=30,
            enable_depth=False,
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial=args.wrist_serial,
            color_width=args.wrist_width,
            color_height=args.wrist_height,
            color_fps=30,
            enable_depth=True,
            depth_width=args.wrist_width,
            depth_height=args.wrist_height,
            depth_fps=30,
        ),
        save_dir="runs/current_capture",
        show_window=True,
    )

    print("\n=== Loading GroundingDINO ===")
    detector_service = GroundingDINODetectorService(
        model_name_or_path=args.grounding_dino_model,
        device="cuda",
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    print("\n=== Loading SAM3 ===")
    segmenter_service = SAM3BBoxSegmenterService(
        model_dir=args.sam3_model_dir,
        checkpoint_path=None,
        device="cuda",
        confidence_threshold=0.0,
        use_autocast=True,
        enable_inst_interactivity=True,
        compile_model=False,
    )

    print("\n=== Loading OCR ===")
    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    print("\n=== Initializing xArm ===")
    arm = init_arm(args.xarm_ip, is_radian=True)

    camera_service.start()
    time.sleep(2.0)

    context = RuntimeContext(
        camera=camera_service,
        detector=detector_service,
        segmenter=segmenter_service,
        ocr=ocr_service,
        event_callback=print_event,
    )
    context.xarm = arm

    find_skill = FindObjectByTextSkill(context)

    return context, camera_service, arm, find_skill


# ============================================================
# Main
# ============================================================

def main():
    install_print_filter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[1]),
    )

    # camera
    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--global_width", type=int, default=1280)
    parser.add_argument("--global_height", type=int, default=720)
    parser.add_argument("--wrist_width", type=int, default=1280)
    parser.add_argument("--wrist_height", type=int, default=720)

    # perception
    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--sam3_model_dir", default="/home/wpy/models/sam3")

    # AnyGrasp
    parser.add_argument(
        "--anygrasp_dir",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
    )

    # xArm
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument(
        "--home_pose",
        type=float,
        nargs=6,
        default=[350, 0, 450, 3.14, 0.0, -1.57],
        metavar=("x", "y", "z", "rx", "ry", "rz"),
    )
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)

    # hand-eye
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )

    # fallback intrinsics
    parser.add_argument("--fx", type=float, default=909.59)
    parser.add_argument("--fy", type=float, default=909.80)
    parser.add_argument("--cx", type=float, default=658.97)
    parser.add_argument("--cy", type=float, default=355.42)

    # detection
    parser.add_argument("--target_name", default="medicine_box")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)

    # grasp planning
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lims_margin", type=float, default=0.03)
    parser.add_argument("--min_mask_points", type=int, default=200)
    parser.add_argument("--approach_clearance_mm", type=float, default=80.0)
    parser.add_argument("--lift_after_grasp_mm", type=float, default=120.0)

    # gripper
    parser.add_argument("--enable_grip", action="store_true", help="是否启用 ROS /grip 服务控制夹爪")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws", help="包含 /grip 服务定义的 ROS 工作空间")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_pause", type=float, default=1.0)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument(
    "--ignore_grip_error",
    action="store_true",
    default=True,
    help="忽略 /grip 服务返回错误，只要夹爪实际动作即可继续流程。",
)
    args = parser.parse_args()

    fallback_intrinsics = {
        "fx": args.fx,
        "fy": args.fy,
        "cx": args.cx,
        "cy": args.cy,
    }

    camera_service = None
    arm = None

    try:
        context, camera_service, arm, find_skill = build_runtime(args)

        print("\n=== Medicine Pick Loop Ready ===")
        print("流程：输入药品名 -> 拍照 -> 检测 -> 分割 -> OCR -> 选择目标 -> AnyGrasp子进程规划 -> xArm移动")
        print("输入 q / quit / exit 退出。")

        while True:
            drug_name = input("\n请输入药品名称: ").strip()

            if drug_name.lower() in ["q", "quit", "exit"]:
                break

            if not drug_name:
                continue

            print(f"\n========== New Task: {drug_name} ==========")

            # 1. 视觉查找目标
            find_result = find_skill.run(
                target_name=args.target_name,
                query=drug_name,
                camera_name="wrist",
                min_score=0.45,
                max_instances=10,
                expand_bbox_px=8,
                select_by="iou",
            )

            print("\n=== find_object_by_text result ===")
            print("success:", find_result.success)
            print("message:", find_result.message)
            print("error:", find_result.error)

            if not find_result.success:
                print("[FAILED] 没有找到目标，等待下一个输入。")
                continue

            selected = find_result.data["selected_instance"]

            print("\n=== Selected Object ===")
            print("object_id:", selected.get("object_id"))
            print("bbox:", selected.get("bbox"))
            print("ocr_text:", selected.get("ocr_text"))
            print("mask_path:", selected.get("mask_path"))
            print("mask_npy_path:", selected.get("mask_npy_path"))
            print("crop_path:", selected.get("crop_path"))

            # 2. 构造 grasp_input
            grasp_input = build_grasp_input_from_find_result(
                find_result.data,
                drug_name=drug_name,
                fallback_intrinsics=fallback_intrinsics,
            )
            check_grasp_input(grasp_input)

            # 3. snapshot 固化文件
            grasp_input, snapshot_dir, grasp_input_json = snapshot_grasp_input_files(
                grasp_input,
                drug_name=drug_name,
                root_dir="runs/grasp_snapshots",
            )

            # 4. 把当前 xArm 位姿写入 grasp_input
            current_ee_pose_mrad = get_current_xarm_pose_mrad(arm)
            grasp_input["current_ee_pose_mrad"] = current_ee_pose_mrad

            save_json(grasp_input_json, grasp_input)

            print("\n=== Current EE Pose m/rad ===")
            pprint(current_ee_pose_mrad)

            # 5. AnyGrasp 子进程规划
            try:
                payload, output_json = run_anygrasp_worker(
                    args=args,
                    snapshot_dir=snapshot_dir,
                    grasp_input_json=grasp_input_json,
                )
            except Exception as e:
                print(f"[FAILED] AnyGrasp worker 规划失败: {e}")
                print("snapshot_dir:", snapshot_dir)
                continue

            data = payload["data"]

            best_pose = data["best_grasp_xarm_mmrad"]
            approach_pose = data["approach_pose_mmrad"]
            lift_pose = data["lift_pose_mmrad"]

            print("\n=== Planned xArm Poses ===")
            print("best_grasp_xarm_mmrad:")
            pprint(best_pose)
            print("approach_pose_mmrad:")
            pprint(approach_pose)
            print("lift_pose_mmrad:")
            pprint(lift_pose)
            print("choose_reason:", data.get("choose_reason"))
            print("snapshot_dir:", snapshot_dir)
            print("anygrasp_output_json:", output_json)

            # 6. 用户确认移动
            print("\n操作选项：")
            print("  m: 移动到 approach pose，再下降到 grasp pose")
            print("  h: 直接归位")
            print("  s: 跳过当前目标")
            print("  q: 退出程序")

            cmd = input("请输入操作: ").strip().lower()

            if cmd == "q":
                break

            if cmd == "s":
                print("[INFO] 跳过当前目标。")
                continue

            if cmd == "h":
                go_home(arm, args.home_pose, speed=args.move_speed, acc=args.move_acc)
                continue

            if cmd != "m":
                print("[INFO] 未执行移动，等待下一个输入。")
                continue

            # 7. 执行抓取流程：open -> approach -> grasp -> close -> lift
            print("\n[Pick] Step 0: open gripper")
            if not open_gripper(args):
                print("[FAILED] 夹爪打开失败。")
                continue

            time.sleep(args.grip_pause)

            print("\n[Pick] Step 1: move to approach pose")
            ok = move_to_pose(
                arm,
                approach_pose,
                speed=args.move_speed,
                acc=args.move_acc,
                wait=True,
                label="approach",
            )
            if not ok:
                print("[FAILED] approach pose 移动失败。")
                continue

            input("\n已到达预抓取位姿。按 Enter 下降到抓取位姿，或 Ctrl+C 停止。")

            print("\n[Pick] Step 2: move to grasp pose")
            ok = move_to_pose(
                arm,
                best_pose,
                speed=args.move_speed,
                acc=args.move_acc,
                wait=True,
                label="grasp",
            )
            if not ok:
                print("[FAILED] grasp pose 移动失败。")
                continue

            print("\n[Pick] Step 3: close gripper")
            if not close_gripper(args):
                print("[FAILED] 夹爪闭合失败。")
                continue

            time.sleep(args.grip_pause)

            print("\n[Pick] Step 4: lift object")
            ok = move_to_pose(
                arm,
                lift_pose,
                speed=args.move_speed,
                acc=args.move_acc,
                wait=True,
                label="lift",
            )
            if not ok:
                print("[FAILED] lift pose 移动失败。")
                continue

            print("\n[SUCCESS] 抓取流程完成：已闭合夹爪并抬升。")

            next_cmd = input("输入 h 归位，输入 o 打开夹爪释放，其他键等待下一个物品: ").strip().lower()

            if next_cmd == "o":
                open_gripper(args)
                time.sleep(args.grip_pause)
                next_cmd = input("输入 h 归位，其他键跳过: ").strip().lower()

            if next_cmd == "h":
                go_home(arm, args.home_pose, speed=args.move_speed, acc=args.move_acc)

            print("\n[INFO] 当前目标流程结束，等待下一个药品名称。")
            if not ok:
                print("[FAILED] grasp pose 移动失败。")
                continue

            print("\n已移动到抓取位姿。当前版本暂不闭合夹爪，只验证位姿。")

            next_cmd = input("输入 h 归位，输入 l 抬升，其他键跳过: ").strip().lower()

            if next_cmd == "l":
                move_to_pose(
                    arm,
                    lift_pose,
                    speed=args.move_speed,
                    acc=args.move_acc,
                    wait=True,
                    label="lift",
                )
                next_cmd = input("输入 h 归位，其他键跳过: ").strip().lower()

            if next_cmd == "h":
                go_home(arm, args.home_pose, speed=args.move_speed, acc=args.move_acc)

            print("\n[INFO] 当前目标流程结束，等待下一个药品名称。")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, stopping...")

        if arm is not None:
            try:
                arm.set_state(4)
            except Exception:
                pass

    finally:
        if camera_service is not None:
            try:
                camera_service.stop()
            except Exception:
                pass

        if arm is not None:
            try:
                arm.disconnect()
                print("[INFO] xArm disconnected.")
            except Exception:
                pass


if __name__ == "__main__":
    main()