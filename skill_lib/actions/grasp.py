"""Grasp skill invocation and refresh of the shared target observation.

The robot argument is the owning SimpleRobotActions facade. State remains on
that object; calls through its methods preserve overrides and recovery hooks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.skill_base import SkillResult
from skill_lib.config import FALLBACK_INTRINSICS
from skill_lib.actions.observation_cache import invalidate_after_manipulation

if TYPE_CHECKING:
    from skill_lib.simple_actions import SimpleRobotActions


@invalidate_after_manipulation('grasp')
def grasp(robot: SimpleRobotActions, obj: str) -> SkillResult:
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill

    cfg = robot.config
    result = FindAndGraspMedicineSkill(robot.context).run(
        query=str(obj),
        camera_name="wrist",
        min_score=cfg.grasp_min_score,
        detection_threshold=cfg.grasp_box_threshold,
        text_threshold=cfg.grasp_text_threshold,
        use_sam_for_selection=cfg.use_sam_for_grasp,
        use_mask_for_grasp=cfg.use_sam_for_grasp,
        plan_grasp=True,
        execute=cfg.execute,
        top_k=cfg.grasp_top_k,
        grasp_selection_mode=cfg.grasp_selection_mode,
        random_select_top_k=cfg.grasp_random_select_top_k,
        random_seed=cfg.grasp_random_seed,
        grasp_pose_options=cfg.grasp_pose_options,
        bbox_filter_margin_px=cfg.bbox_filter_margin_px,
        bbox_filter_inner_margin_ratio=cfg.bbox_filter_inner_margin_ratio,
        collision_detection=cfg.collision_detection,
        infer_grasps_on_full_cloud=True,
        filter_grasps_by_bbox=True,
        visualize_selected_grasp=cfg.visualize_selected_grasp,
        visualize_bbox_filtered_grasps=cfg.visualize_bbox_filtered_grasps,
        move_speed=cfg.move_speed,
        move_acc=cfg.move_acc,
        gripper_open=robot.gripper_open,
        gripper_close=robot.gripper_close,
        gripper_close_repeat_count=cfg.gripper_close_repeat_count,
        gripper_close_repeat_interval_s=cfg.gripper_close_repeat_interval_s,
        gripper_close_wait_s=cfg.gripper_close_wait_s,
        verify_ocr=cfg.verify_ocr,
        ocr_verify_strict_mode=cfg.ocr_verify_strict_mode,
        ocr_verify_fuzzy_threshold=cfg.ocr_verify_fuzzy_threshold,
        reobserve_on_grasp_failure=cfg.reobserve_on_grasp_failure,
        reobserve_only_when_execute=not cfg.execute,
        reobserve_z_mm=cfg.reobserve_z_mm,
        reobserve_wait_s=cfg.reobserve_wait_s,
        reobserve_depth_window_px=cfg.reobserve_depth_window_px,
        fallback_intrinsics=None if robot.context.transforms is not None else FALLBACK_INTRINSICS,
        xarm_ip=cfg.xarm_ip,
        handeye_config_path=cfg.handeye_config_path,
    )
    if result.success:
        robot.observations[obj] = {
            "kind": "medicine",
            "target": obj,
            "selected_instance": result.data.get("selected_instance"),
            "grasp_input": result.data.get("grasp_input"),
            "result": result.data,
        }
    return result
