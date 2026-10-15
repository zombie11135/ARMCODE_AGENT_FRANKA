"""Home and observation poses through the existing robot motion adapter.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, TYPE_CHECKING

from core.skill_base import SkillResult
from skill_lib.config import FALLBACK_INTRINSICS

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


def home(robot: SimpleRobotActions) -> SkillResult:
    cfg = robot.config
    if not cfg.execute:
        return SkillResult(success=True, data={"home_pose_mmrad": cfg.home_pose_mmrad, "dry_run": True})
    ok = robot._move_xarm(cfg.home_pose_mmrad)
    return SkillResult(
        success=ok,
        data={"home_pose_mmrad": cfg.home_pose_mmrad},
        error=None if ok else "move home failed",
    )


def _move_to_scan_pose(robot: SimpleRobotActions, offset_y_mm: float) -> SkillResult:
    pose = list(robot.config.home_pose_mmrad)
    pose[1] += float(offset_y_mm)
    if not robot.config.execute:
        return SkillResult(success=True, data={"scan_pose_mmrad": pose, "dry_run": True})
    ok = robot._move_xarm(pose)
    return SkillResult(
        success=ok,
        data={"scan_pose_mmrad": pose},
        error=None if ok else f"move scan pose failed: y offset {offset_y_mm}",
    )


def _move_to_cached_container_observe_pose(
    robot: SimpleRobotActions,
    container: str,
    place_plan: Dict[str, Any],
) -> SkillResult:
    center = place_plan.get("center_base_xyz")
    if not center or len(center) < 2:
        return SkillResult(success=False, data={"container": container}, error="cached container center missing")

    pose = [
        float(center[0]) * 1000.0,
        float(center[1]) * 1000.0,
        float(robot.config.observe_z_mm),
        float(robot.config.home_pose_mmrad[3]),
        float(robot.config.home_pose_mmrad[4]),
        float(robot.config.home_pose_mmrad[5]),
    ]
    if not robot.config.execute:
        return SkillResult(success=True, data={"observe_pose_mmrad": pose, "dry_run": True})
    ok = robot._move_xarm(pose)
    if ok and robot.config.observe_wait_s > 0:
        time.sleep(float(robot.config.observe_wait_s))
    return SkillResult(
        success=ok,
        data={"observe_pose_mmrad": pose},
        error=None if ok else f"move cached container observe pose failed: {container}",
    )


def _move_to_cached_medicine_observe_pose(
    robot: SimpleRobotActions,
    obj: str,
    cached: Dict[str, Any],
) -> SkillResult:
    grasp_input = cached.get("grasp_input") or {}
    selected = cached.get("selected_instance") or {"bbox": grasp_input.get("bbox")}
    if not selected.get("bbox"):
        return SkillResult(success=False, data={"target": obj}, error="cached medicine bbox missing")

    capture = {
        "capture_geometry": grasp_input.get("capture_geometry"),
        "points_path": grasp_input.get("points_path"),
        "depth_path": grasp_input.get("depth_path"),
        "depth_scale": grasp_input.get("depth_scale", 1000.0),
        "depth_intrinsics": (
            grasp_input.get("depth_intrinsics")
            or grasp_input.get("camera_intrinsics")
            or grasp_input.get("color_intrinsics")
        ),
    }
    try:
        from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

        pose = FindAndGraspMedicineSkill(robot.context)._build_target_observe_pose(
            capture=capture,
            selected=selected,
            fallback_intrinsics=None if robot.context.transforms is not None else FALLBACK_INTRINSICS,
            xarm_ip=robot.config.xarm_ip,
            handeye_config_path=robot.config.handeye_config_path,
            observe_z_mm=robot.config.observe_z_mm,
            depth_window_px=robot.config.reobserve_depth_window_px,
        )
    except Exception as e:
        return SkillResult(success=False, data={"target": obj}, error=f"build cached observe pose failed: {e}")

    if not robot.config.execute:
        return SkillResult(success=True, data={"observe_pose_mmrad": pose, "dry_run": True})
    ok = robot._move_xarm(pose)
    if ok and robot.config.observe_wait_s > 0:
        time.sleep(float(robot.config.observe_wait_s))
    return SkillResult(
        success=ok,
        data={"observe_pose_mmrad": pose},
        error=None if ok else f"move cached medicine observe pose failed: {obj}",
    )


def _move_xarm(robot: SimpleRobotActions, pose: List[float]) -> bool:
    arm = robot.context.require("xarm")
    x, y, z, rx, ry, rz = [float(v) for v in pose]
    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=rx,
        pitch=ry,
        yaw=rz,
        speed=robot.config.move_speed,
        mvacc=robot.config.move_acc,
        wait=True,
    )
    return code == 0
