from typing import Any, Callable, Dict, List, Optional

from core.skill_base import BaseSkill, SkillResult
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.combound.place_to_container import PlaceToContainerSkill


class PickMedicineAndPlaceToContainerSkill(BaseSkill):
    name = "pick_medicine_and_place_to_container"
    description = "复合技能：空手定位容器，抓取指定药盒，再使用缓存的容器位姿完成放置。"

    input_schema = {
        "medicine_query": "目标药品名称，例如 布洛芬。",
        "container_name": "放置容器，例如 tray / plate / container。",
        "container_color": "可选。目标容器颜色，例如 粉色 / 蓝色 / green。",
        "execute": "是否执行真实抓取和放置。",
        "pre_grasp_pose_mmrad": "可选。容器定位后、抓取前回到的观察/安全位姿。",
    }

    output_schema = {
        "place_plan": "空手阶段得到的容器放置位姿。",
        "grasp_result": "药盒抓取结果。",
        "place_execution": "使用缓存放置位姿执行释放的结果。",
    }

    def run(
        self,
        medicine_query: str,
        container_name: str = "tray",
        container_color: Optional[str] = None,
        execute: bool = False,
        xarm_ip: str = "192.168.1.237",
        handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
        fallback_intrinsics: Optional[Dict[str, float]] = None,
        pre_grasp_pose_mmrad: Optional[List[float]] = None,
        move_speed: float = 50,
        move_acc: float = 500,
        gripper_open: Optional[Callable[[], bool]] = None,
        gripper_close: Optional[Callable[[], bool]] = None,

        # container location
        container_detection_threshold: float = 0.25,
        container_text_threshold: float = 0.20,
        container_detection_index: int = 0,
        container_depth_window_px: int = 25,
        container_refine_observation: bool = True,
        container_refine_observe_z_mm: float = 400.0,
        container_refine_wait_s: float = 0.8,
        place_clearance_mm: float = 250.0,
        place_approach_clearance_mm: float = 120.0,
        place_lift_after_mm: float = 120.0,

        # grasp
        grasp_detection_threshold: float = 0.22,
        grasp_text_threshold: float = 0.18,
        grasp_min_score: float = 0.50,
        grasp_top_k: int = 50,
        grasp_selection_mode: str = "random_top_k_good",
        grasp_random_select_top_k: int = 5,
        gripper_close_wait_s: float = 1.5,
        verify_ocr: bool = True,
        ocr_verify_strict_mode: bool = False,
        ocr_verify_fuzzy_threshold: float = 0.72,
        bbox_filter_inner_margin_ratio: float = 0.15,
        visualize_selected_grasp: bool = False,
        visualize_bbox_filtered_grasps: bool = False,
        **kwargs,
    ) -> SkillResult:
        try:
            self.context.emit_text(
                f"开始组合技能：pick '{medicine_query}' -> place to '{container_name}'"
                + (f" ({container_color})" if container_color else "")
            )

            # 1. Empty-hand container localization. If execute=True, allow the
            # refine observation move before grasping; after grasping the wrist
            # camera may be occluded by the held object.
            place_skill = PlaceToContainerSkill(self.context)
            place_result = place_skill.run(
                container_name=container_name,
                container_color=container_color,
                camera_name="wrist",
                detection_threshold=container_detection_threshold,
                text_threshold=container_text_threshold,
                detection_index=container_detection_index,
                depth_window_px=container_depth_window_px,
                xarm_ip=xarm_ip,
                handeye_config_path=handeye_config_path,
                place_clearance_mm=place_clearance_mm,
                approach_clearance_mm=place_approach_clearance_mm,
                lift_after_place_mm=place_lift_after_mm,
                refine_observation=container_refine_observation,
                refine_only_when_execute=not execute,
                refine_observe_z_mm=container_refine_observe_z_mm,
                refine_wait_s=container_refine_wait_s,
                execute=False,
                move_speed=move_speed,
                move_acc=move_acc,
                gripper_open=None,
                fallback_intrinsics=fallback_intrinsics,
            )

            if not place_result.success:
                return SkillResult(
                    success=False,
                    data={"stage": "locate_container", "place_result": place_result.data},
                    error=f"空手定位容器失败: {place_result.error}",
                )

            place_plan = place_result.data

            if execute and pre_grasp_pose_mmrad is not None:
                self.context.emit_text(f"容器定位完成，回到抓取观察位姿：{pre_grasp_pose_mmrad}")
                arm = self.context.require("xarm")
                if not self._move_xarm(arm, pre_grasp_pose_mmrad, move_speed, move_acc):
                    return SkillResult(
                        success=False,
                        data={"stage": "move_pre_grasp", "place_plan": place_plan},
                        error="移动到抓取前观察位姿失败。",
                    )

            # 2. Pick target medicine. This can execute the actual grasp.
            grasp_skill = FindAndGraspMedicineSkill(self.context)
            grasp_result = grasp_skill.run(
                query=medicine_query,
                camera_name="wrist",
                min_score=grasp_min_score,
                detection_threshold=grasp_detection_threshold,
                text_threshold=grasp_text_threshold,
                use_sam_for_selection=False,
                use_mask_for_grasp=False,
                plan_grasp=True,
                execute=execute,
                xarm_ip=xarm_ip,
                handeye_config_path=handeye_config_path,
                top_k=grasp_top_k,
                grasp_selection_mode=grasp_selection_mode,
                random_select_top_k=grasp_random_select_top_k,
                bbox_filter_inner_margin_ratio=bbox_filter_inner_margin_ratio,
                infer_grasps_on_full_cloud=True,
                filter_grasps_by_bbox=True,
                visualize_selected_grasp=visualize_selected_grasp,
                visualize_bbox_filtered_grasps=visualize_bbox_filtered_grasps,
                move_speed=move_speed,
                move_acc=move_acc,
                gripper_open=gripper_open,
                gripper_close=gripper_close,
                gripper_close_wait_s=gripper_close_wait_s,
                verify_ocr=verify_ocr,
                ocr_verify_strict_mode=ocr_verify_strict_mode,
                ocr_verify_fuzzy_threshold=ocr_verify_fuzzy_threshold,
                fallback_intrinsics=fallback_intrinsics,
            )

            if not grasp_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "pick_medicine",
                        "place_plan": place_plan,
                        "grasp_result": grasp_result.data,
                    },
                    error=f"抓取药盒失败: {grasp_result.error}",
                )

            data: Dict[str, Any] = {
                "place_plan": place_plan,
                "grasp_result": grasp_result.data,
                "place_execution": None,
            }

            # 3. Use the cached container pose from empty-hand observation.
            if execute:
                place_execution = self._execute_cached_place(
                    approach_pose=place_plan["approach_pose_mmrad"],
                    place_pose=place_plan["place_pose_mmrad"],
                    lift_pose=place_plan["lift_pose_mmrad"],
                    move_speed=move_speed,
                    move_acc=move_acc,
                    gripper_open=gripper_open,
                )
                data["place_execution"] = place_execution

                if not place_execution.get("success", False):
                    return SkillResult(
                        success=False,
                        data=data,
                        error=place_execution.get("error", "缓存放置位姿执行失败。"),
                    )

            return SkillResult(
                success=True,
                data=data,
                message=f"已完成组合流程：{medicine_query} -> {container_name}",
            )

        except Exception as e:
            error = f"pick_medicine_and_place_to_container 执行失败: {e}"
            try:
                self.context.emit_error(error)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=error)

    def _execute_cached_place(
        self,
        approach_pose: List[float],
        place_pose: List[float],
        lift_pose: List[float],
        move_speed: float,
        move_acc: float,
        gripper_open: Optional[Callable[[], bool]],
    ) -> Dict[str, Any]:
        arm = self.context.require("xarm")

        if not self._move_xarm(arm, approach_pose, move_speed, move_acc):
            return {"success": False, "error": "移动到缓存 approach pose 失败。"}

        if not self._move_xarm(arm, place_pose, move_speed, move_acc):
            return {"success": False, "error": "移动到缓存 place pose 失败。"}

        if gripper_open and not gripper_open():
            return {"success": False, "error": "放置时打开夹爪失败。"}

        if not self._move_xarm(arm, lift_pose, move_speed, move_acc):
            return {"success": False, "error": "放置后抬升失败。"}

        return {"success": True, "message": "缓存放置位姿执行完成。"}

    @staticmethod
    def _move_xarm(arm, pose: List[float], speed: float, acc: float) -> bool:
        x, y, z, rx, ry, rz = [float(v) for v in pose]
        code = arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=rx,
            pitch=ry,
            yaw=rz,
            speed=speed,
            mvacc=acc,
            wait=True,
        )
        return code == 0
