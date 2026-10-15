from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from core.skill_base import BaseSkill, SkillResult
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill


class PlaceToContainerSkill(BaseSkill):
    name = "place_to_container"
    description = "复合技能：检测盘子/容器 bbox，估计容器中心 base 坐标，并可选执行放置。"

    input_schema = {
        "container_name": "容器类别，例如 tray / plate / bowl / container。",
        "container_color": "可选。容器颜色，例如 pink / red / blue / green / 粉色 / 蓝色。",
        "camera_name": "默认 wrist。",
        "target_prompts": "可选。GroundingDINO 多 prompt 列表。",
        "execute": "是否执行机械臂放置动作。",
        "gripper_open": "可选夹爪打开函数，用于释放物体。",
    }

    output_schema = {
        "selected_detection": "选中的容器检测框。",
        "center_pixel": "容器中心像素。",
        "center_camera_xyz": "容器中心相机系坐标，单位 m。",
        "center_base_xyz": "容器中心 base 坐标，单位 m。",
        "approach_pose_mmrad": "预放置位姿。",
        "place_pose_mmrad": "放置释放位姿。",
        "lift_pose_mmrad": "释放后抬升位姿。",
    }

    DEFAULT_TARGET_PROMPTS = [
        "tray",
        "plate",
        "bowl",
        "empty container",
        "plastic container",
        "storage container",
        "transparent storage box",
        "clear plastic container",
        "open organizer box",
    ]

    def run(
        self,
        container_name: str = "tray",
        container_color: Optional[str] = None,
        camera_name: str = "wrist",
        target_prompts: Optional[List[str]] = None,
        detection_threshold: Optional[float] = 0.25,
        text_threshold: Optional[float] = 0.20,
        max_detections_per_prompt: int = 10,
        merge_iou: float = 0.60,
        detection_index: int = 0,
        center_offset_ratio: Tuple[float, float] = (0.0, 0.0),
        depth_window_px: int = 25,
        fallback_intrinsics: Optional[Dict[str, float]] = None,
        xarm_ip: str = "192.168.1.237",
        handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
        target_rpy: Tuple[float, float, float] = (3.14, 0.0, -1.57),
        place_clearance_mm: float = 250.0,
        approach_clearance_mm: float = 120.0,
        lift_after_place_mm: float = 120.0,
        refine_observation: bool = True,
        refine_only_when_execute: bool = True,
        refine_observe_z_mm: float = 400.0,
        refine_wait_s: float = 0.8,
        execute: bool = False,
        move_speed: float = 50,
        move_acc: float = 500,
        gripper_open: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            color_label = f", color={container_color}" if container_color else ""
            self.context.emit_text(f"开始放置技能：container={container_name}{color_label}")

            current_ee_pose_mrad = PlanGraspWithAnyGraspSkill(
                self.context
            )._get_current_xarm_pose_mrad(xarm_ip=xarm_ip)

            if current_ee_pose_mrad is None:
                return SkillResult(
                    success=False,
                    data={},
                    error="无法通过 xArm SDK 获取当前末端位姿。",
                )

            handeye_rot, handeye_trans = PlanGraspWithAnyGraspSkill._load_handeye_cam2ee(
                handeye_config_path,
            )

            first_plan = self._estimate_place_plan_once(
                camera_name=camera_name,
                container_name=container_name,
                container_color=container_color,
                target_prompts=target_prompts,
                detection_threshold=detection_threshold,
                text_threshold=text_threshold,
                max_detections_per_prompt=max_detections_per_prompt,
                merge_iou=merge_iou,
                detection_index=detection_index,
                center_offset_ratio=center_offset_ratio,
                depth_window_px=depth_window_px,
                fallback_intrinsics=fallback_intrinsics,
                current_ee_pose=current_ee_pose_mrad,
                handeye_rot=handeye_rot,
                handeye_trans=handeye_trans,
                target_rpy=target_rpy,
                place_clearance_mm=place_clearance_mm,
                approach_clearance_mm=approach_clearance_mm,
                lift_after_place_mm=lift_after_place_mm,
                caption="第一次放置容器检测输入图像",
            )

            data = dict(first_plan)
            data["first_observation"] = first_plan
            data["refined_observation"] = None
            data["refine_observation"] = refine_observation
            data["refine_only_when_execute"] = refine_only_when_execute

            should_refine = refine_observation and (execute or not refine_only_when_execute)

            if should_refine:
                arm = self.context.require("xarm")
                observe_pose = self._build_observe_pose(
                    center_base_xyz=first_plan["center_base_xyz"],
                    target_rpy=target_rpy,
                    observe_z_mm=refine_observe_z_mm,
                )

                self.context.emit_text(f"移动到二次观察位姿：{observe_pose}")

                if not self._move_xarm(arm, observe_pose, move_speed, move_acc):
                    return SkillResult(
                        success=False,
                        data={**data, "refine_observe_pose_mmrad": observe_pose},
                        error="移动到二次观察位姿失败。",
                    )

                if refine_wait_s > 0:
                    import time
                    time.sleep(float(refine_wait_s))

                refined_pose_mrad = PlanGraspWithAnyGraspSkill(
                    self.context
                )._get_current_xarm_pose_mrad(xarm_ip=xarm_ip)

                if refined_pose_mrad is None:
                    return SkillResult(
                        success=False,
                        data={**data, "refine_observe_pose_mmrad": observe_pose},
                        error="二次观察前无法重新获取 xArm 当前末端位姿。",
                    )

                refined_plan = self._estimate_place_plan_once(
                    camera_name=camera_name,
                    container_name=container_name,
                    container_color=container_color,
                    target_prompts=target_prompts,
                    detection_threshold=detection_threshold,
                    text_threshold=text_threshold,
                    max_detections_per_prompt=max_detections_per_prompt,
                    merge_iou=merge_iou,
                    detection_index=detection_index,
                    center_offset_ratio=center_offset_ratio,
                    depth_window_px=depth_window_px,
                    fallback_intrinsics=fallback_intrinsics,
                    current_ee_pose=refined_pose_mrad,
                    handeye_rot=handeye_rot,
                    handeye_trans=handeye_trans,
                    target_rpy=target_rpy,
                    place_clearance_mm=place_clearance_mm,
                    approach_clearance_mm=approach_clearance_mm,
                    lift_after_place_mm=lift_after_place_mm,
                    caption="第二次放置容器检测输入图像",
                )

                data = dict(refined_plan)
                data["first_observation"] = first_plan
                data["refined_observation"] = refined_plan
                data["refine_observe_pose_mmrad"] = observe_pose
                data["refine_observation"] = refine_observation
                data["refine_only_when_execute"] = refine_only_when_execute

            data.update(
                {
                "current_ee_pose_mrad": current_ee_pose_mrad,
                "handeye_rot": np.asarray(handeye_rot, dtype=float).tolist(),
                "handeye_trans": np.asarray(handeye_trans, dtype=float).reshape(3).tolist(),
                }
            )

            if execute:
                execution = self._execute_place(
                    approach_pose=data["approach_pose_mmrad"],
                    place_pose=data["place_pose_mmrad"],
                    lift_pose=data["lift_pose_mmrad"],
                    move_speed=move_speed,
                    move_acc=move_acc,
                    gripper_open=gripper_open,
                )
                data["execution"] = execution

                if not execution.get("success", False):
                    return SkillResult(
                        success=False,
                        data=data,
                        error=execution.get("error", "放置执行失败。"),
                    )

            return SkillResult(
                success=True,
                data=data,
                message=f"已生成容器 {container_name} 的放置位姿。",
            )

        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            error = f"place_to_container 执行失败: {e}"
            try:
                self.context.emit_error(error)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=error)

    def _capture(self, camera_name: str) -> Dict[str, Any]:
        camera = self.context.require("camera")
        if camera_name == "wrist":
            return camera.capture_wrist()
        if camera_name == "global":
            return camera.capture_global()
        raise ValueError(f"Unknown camera_name: {camera_name}")

    def _estimate_place_plan_once(
        self,
        camera_name: str,
        container_name: str,
        container_color: Optional[str],
        target_prompts: Optional[List[str]],
        detection_threshold: Optional[float],
        text_threshold: Optional[float],
        max_detections_per_prompt: int,
        merge_iou: float,
        detection_index: int,
        center_offset_ratio: Tuple[float, float],
        depth_window_px: int,
        fallback_intrinsics: Optional[Dict[str, float]],
        current_ee_pose: List[float],
        handeye_rot,
        handeye_trans,
        target_rpy: Tuple[float, float, float],
        place_clearance_mm: float,
        approach_clearance_mm: float,
        lift_after_place_mm: float,
        caption: str,
    ) -> Dict[str, Any]:
        capture = self._capture(camera_name)
        image_path = capture["color_path"]
        depth_path = capture.get("depth_path")

        if not depth_path:
            raise RuntimeError("放置技能需要 wrist RGB-D 图像，但当前 capture 缺少 depth_path。")

        self.context.emit_image(image_path=image_path, caption=caption)

        detections = self._detect_container(
            image_path=image_path,
            container_name=container_name,
            container_color=container_color,
            target_prompts=target_prompts,
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            max_detections_per_prompt=max_detections_per_prompt,
            merge_iou=merge_iou,
        )

        if not detections:
            raise RuntimeError(f"没有检测到放置容器：{container_name}")

        if detection_index >= len(detections):
            raise RuntimeError(
                f"detection_index={detection_index} 超出检测数量 {len(detections)}。"
            )

        selected = detections[detection_index]
        bbox = selected["bbox"]

        intrinsics = self._normalize_intrinsics(
            fallback_intrinsics
            or capture.get("depth_intrinsics")
            or capture.get("color_intrinsics")
            or capture.get("camera_intrinsics")
        )

        if intrinsics is None:
            raise RuntimeError("缺少相机内参，无法估计容器中心坐标。")

        depth = self._load_depth(
            depth_path=depth_path,
            depth_scale=capture.get("depth_scale", 1000.0),
        )

        center_pixel = self._bbox_center_pixel(
            bbox=bbox,
            offset_ratio=center_offset_ratio,
            image_width=depth.shape[1],
            image_height=depth.shape[0],
        )
        depth_estimate = self._estimate_container_depth(
            depth=depth,
            bbox=bbox,
            preferred_pixel=center_pixel,
            window_px=depth_window_px,
        )

        if depth_estimate is None:
            raise RuntimeError("容器 bbox 内没有可靠有效深度。")

        center_depth = depth_estimate["depth_m"]
        center_pixel = depth_estimate["pixel"]

        center_camera_xyz = self._pixel_to_camera_xyz(
            pixel=center_pixel,
            depth_m=center_depth,
            intrinsics=intrinsics,
        )

        center_base_xyz = self._camera_point_to_base(
            point_camera=center_camera_xyz,
            current_ee_pose=current_ee_pose,
            handeye_rot=handeye_rot,
            handeye_trans=handeye_trans,
        )
        if self.context.transforms is not None:
            from hardware.agent_geometry import point_to_base
            center_base_xyz = point_to_base(capture, center_camera_xyz, self.context.transforms)


        place_pose = [
            center_base_xyz[0] * 1000.0,
            center_base_xyz[1] * 1000.0,
            center_base_xyz[2] * 1000.0 + place_clearance_mm,
            float(target_rpy[0]),
            float(target_rpy[1]),
            float(target_rpy[2]),
        ]

        place_flange_z_m = None
        if self.context.transforms is not None:
            from hardware.agent_geometry import mmrad
            from hardware.interfaces import Pose6D
            from hardware.transforms import pose_matrix
            tcp = Pose6D(*[v / 1000 for v in place_pose[:3]], *place_pose[3:])
            place_pose = mmrad(self.context.transforms.tcp_to_flange_pose(pose_matrix(tcp)).values)
            arm_config = getattr(self.context.arm, 'config', None)
            place_flange_z_m = getattr(arm_config, 'place_flange_z_m', None)
            if place_flange_z_m is not None:
                # Apply AFTER TCP conversion: this is flange Z, not TCP Z or tray clearance.
                place_pose[2] = float(place_flange_z_m) * 1000.0

        approach_pose = list(place_pose)
        approach_pose[2] += approach_clearance_mm

        lift_pose = list(place_pose)
        lift_pose[2] += lift_after_place_mm

        return {
            "image_path": image_path,
            "depth_path": depth_path,
            "detections": detections,
            "selected_detection": selected,
            "container_color": self._canonical_color(container_color),
            "center_pixel": center_pixel,
            "center_depth_m": center_depth,
            "depth_estimation": depth_estimate,
            "center_camera_xyz": center_camera_xyz,
            "center_base_xyz": center_base_xyz,
            "target_rpy": list(target_rpy),
            "approach_pose_mmrad": approach_pose,
            "place_pose_mmrad": place_pose,
            "lift_pose_mmrad": lift_pose,
            "place_clearance_mm": place_clearance_mm,
            "place_height_mode": "absolute_flange_z" if place_flange_z_m is not None else "container_relative",
            "place_flange_z_mm": None if place_flange_z_m is None else float(place_flange_z_m) * 1000.0,
            "approach_clearance_mm": approach_clearance_mm,
            "lift_after_place_mm": lift_after_place_mm,
        }

    @staticmethod
    def _build_observe_pose(
        center_base_xyz: List[float],
        target_rpy: Tuple[float, float, float],
        observe_z_mm: float,
    ) -> List[float]:
        return [
            float(center_base_xyz[0]) * 1000.0,
            float(center_base_xyz[1]) * 1000.0,
            float(observe_z_mm),
            float(target_rpy[0]),
            float(target_rpy[1]),
            float(target_rpy[2]),
        ]

    def _detect_container(
        self,
        image_path: str,
        container_name: str,
        container_color: Optional[str],
        target_prompts: Optional[List[str]],
        detection_threshold: Optional[float],
        text_threshold: Optional[float],
        max_detections_per_prompt: int,
        merge_iou: float,
    ) -> List[Dict[str, Any]]:
        detector = self.context.require("detector")
        canonical_color = self._canonical_color(container_color)
        prompts = self._build_prompts(container_name, target_prompts)
        if canonical_color:
            color_prompts = [f"{canonical_color} {prompt}" for prompt in prompts]
            prompts = self._dedupe_prompts(color_prompts + prompts)
        fused: List[Dict[str, Any]] = []

        for prompt in prompts:
            self.context.emit_text(f"GroundingDINO 检测放置容器 prompt={prompt}")
            result = detector.detect(
                image_path=image_path,
                target_name=prompt,
                threshold=detection_threshold,
                text_threshold=text_threshold,
                max_objects=max_detections_per_prompt,
                enable_filter=True,
            )

            for det in result.get("detections", []):
                det = dict(det)
                det["source_prompt"] = prompt
                self._merge_detection(fused, det, merge_iou=merge_iou)

        if canonical_color:
            self._annotate_color_scores(
                image_path=image_path,
                detections=fused,
                canonical_color=canonical_color,
            )

        fused.sort(
            key=lambda d: (
                float(d.get("color_score", 0.0) or 0.0) if canonical_color else 0.0,
                len(d.get("prompt_hits", [])),
                float(d.get("confidence", 0.0) or 0.0),
                self._bbox_area(d.get("bbox")),
            ),
            reverse=True,
        )

        if canonical_color and fused:
            best_color_score = float(fused[0].get("color_score", 0.0) or 0.0)
            if best_color_score < 0.04:
                self.context.emit_warning(
                    f"未找到颜色明显匹配 {container_color} 的容器候选，"
                    "将退化为 GroundingDINO 容器检测排序。"
                )

        for idx, det in enumerate(fused):
            det["object_id"] = f"container_{idx + 1}"

        return fused

    @classmethod
    def _build_prompts(
        cls,
        container_name: str,
        target_prompts: Optional[List[str]],
    ) -> List[str]:
        prompts = list(target_prompts or [])
        if not prompts:
            name = str(container_name).strip()
            if name in ["tray", "盘子", "托盘"]:
                prompts = ["tray", "plate", "shallow tray", "empty plate"]
            elif name in ["plate", "碟子"]:
                prompts = ["plate", "dish", "empty plate"]
            elif name in ["bowl", "碗"]:
                prompts = ["bowl", "empty bowl"]
            elif name in ["container", "容器", "盒子", "收纳盒", "整理盒", "透明盒", "半透明盒"]:
                prompts = [
                    "empty container",
                    "plastic container",
                    "storage container",
                    "open box container",
                    "transparent storage box",
                    "translucent storage box",
                    "clear plastic container",
                    "clear plastic storage bin",
                    "open plastic storage box",
                    "empty organizer box",
                    "open organizer box",
                    "empty storage bin",
                ]
            else:
                prompts = [name]

        clean = []
        for prompt in prompts:
            prompt = str(prompt).strip()
            if prompt and prompt not in clean:
                clean.append(prompt)
        return clean

    @staticmethod
    def _dedupe_prompts(prompts: List[str]) -> List[str]:
        clean = []
        for prompt in prompts:
            prompt = str(prompt).strip()
            if prompt and prompt not in clean:
                clean.append(prompt)
        return clean

    @classmethod
    def _canonical_color(cls, color: Optional[str]) -> Optional[str]:
        text = str(color or "").strip().lower()
        if not text:
            return None
        if any(token in text for token in ["transparent", "translucent", "clear", "透明", "半透明"]):
            return None

        aliases = {
            "pink": ["pink", "magenta", "rose", "粉", "粉色", "粉红", "粉红色"],
            "red": ["red", "红", "红色"],
            "blue": ["blue", "蓝", "蓝色"],
            "green": ["green", "绿", "绿色"],
            "yellow": ["yellow", "黄", "黄色"],
            "orange": ["orange", "橙", "橙色"],
            "purple": ["purple", "violet", "紫", "紫色"],
            "white": ["white", "白", "白色"],
            "black": ["black", "黑", "黑色"],
            "gray": ["gray", "grey", "灰", "灰色"],
            "brown": ["brown", "棕", "棕色", "褐", "褐色"],
        }
        for canonical, names in aliases.items():
            if any(name in text for name in names):
                return canonical
        return text

    @classmethod
    def _annotate_color_scores(
        cls,
        image_path: str,
        detections: List[Dict[str, Any]],
        canonical_color: str,
    ) -> None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return

        for det in detections:
            score = cls._bbox_color_score(image, det.get("bbox"), canonical_color)
            det["color_name"] = canonical_color
            det["color_score"] = float(score)

    @classmethod
    def _bbox_color_score(cls, image_bgr: np.ndarray, bbox, canonical_color: str) -> float:
        if not bbox:
            return 0.0

        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        # Use the central region to reduce interference from table edges and
        # nearby medicine boxes when the detector bbox is slightly oversized.
        ch, cw = crop.shape[:2]
        mx = int(cw * 0.12)
        my = int(ch * 0.12)
        if cw - 2 * mx > 8 and ch - 2 * my > 8:
            crop = crop[my : ch - my, mx : cw - mx]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        if canonical_color == "pink":
            mask = (
                (((h_chan >= 145) & (h_chan <= 179)) | ((h_chan >= 0) & (h_chan <= 8)))
                & (s_chan >= 35)
                & (v_chan >= 90)
            )
        elif canonical_color == "red":
            mask = (
                (((h_chan >= 0) & (h_chan <= 10)) | ((h_chan >= 170) & (h_chan <= 179)))
                & (s_chan >= 45)
                & (v_chan >= 70)
            )
        elif canonical_color == "blue":
            mask = (h_chan >= 90) & (h_chan <= 130) & (s_chan >= 45) & (v_chan >= 50)
        elif canonical_color == "green":
            mask = (h_chan >= 35) & (h_chan <= 85) & (s_chan >= 35) & (v_chan >= 50)
        elif canonical_color == "yellow":
            mask = (h_chan >= 18) & (h_chan <= 35) & (s_chan >= 45) & (v_chan >= 80)
        elif canonical_color == "orange":
            mask = (h_chan >= 8) & (h_chan <= 24) & (s_chan >= 55) & (v_chan >= 80)
        elif canonical_color == "purple":
            mask = (h_chan >= 125) & (h_chan <= 155) & (s_chan >= 35) & (v_chan >= 60)
        elif canonical_color == "white":
            mask = (s_chan <= 45) & (v_chan >= 150)
        elif canonical_color == "black":
            mask = v_chan <= 70
        elif canonical_color == "gray":
            mask = (s_chan <= 45) & (v_chan >= 70) & (v_chan <= 190)
        elif canonical_color == "brown":
            mask = (h_chan >= 5) & (h_chan <= 25) & (s_chan >= 45) & (v_chan >= 35) & (v_chan <= 180)
        else:
            return 0.0

        valid = np.isfinite(v_chan)
        if not np.any(valid):
            return 0.0
        return float(np.count_nonzero(mask & valid) / np.count_nonzero(valid))

    @classmethod
    def _merge_detection(
        cls,
        fused: List[Dict[str, Any]],
        det: Dict[str, Any],
        merge_iou: float,
    ) -> None:
        for old in fused:
            if cls._bbox_iou(old["bbox"], det["bbox"]) < merge_iou:
                continue

            old_conf = float(old.get("confidence", 0.0) or 0.0)
            new_conf = float(det.get("confidence", 0.0) or 0.0)
            if new_conf > old_conf:
                old["bbox"] = det["bbox"]
                old["confidence"] = new_conf
                old["phrase"] = det.get("phrase")
                old["source_prompt"] = det.get("source_prompt")

            hits = old.setdefault("prompt_hits", [])
            prompt = det.get("source_prompt")
            if prompt and prompt not in hits:
                hits.append(prompt)
            return

        new_det = dict(det)
        prompt = det.get("source_prompt")
        new_det["prompt_hits"] = [prompt] if prompt else []
        fused.append(new_det)

    @staticmethod
    def _load_depth(depth_path: str, depth_scale: float) -> np.ndarray:
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)

        depth_raw = depth_raw.astype(np.float32)
        depth_scale = float(depth_scale)

        if 0 < depth_scale < 1:
            return depth_raw * depth_scale

        return depth_raw / depth_scale

    @staticmethod
    def _bbox_center_pixel(
        bbox: List[int],
        offset_ratio: Tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> List[int]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) * 0.5 + (x2 - x1) * float(offset_ratio[0])
        cy = (y1 + y2) * 0.5 + (y2 - y1) * float(offset_ratio[1])
        u = int(round(max(0, min(cx, image_width - 1))))
        v = int(round(max(0, min(cy, image_height - 1))))
        return [u, v]

    @staticmethod
    def _median_depth_near_pixel(
        depth: np.ndarray,
        pixel: List[int],
        window_px: int,
    ) -> Optional[float]:
        u, v = [int(x) for x in pixel]
        half = max(1, int(window_px) // 2)
        h, w = depth.shape[:2]
        x1 = max(0, u - half)
        x2 = min(w, u + half + 1)
        y1 = max(0, v - half)
        y2 = min(h, v + half + 1)

        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    @classmethod
    def _estimate_container_depth(
        cls,
        depth: np.ndarray,
        bbox: List[int],
        preferred_pixel: List[int],
        window_px: int,
    ) -> Optional[Dict[str, Any]]:
        center_depth = cls._median_depth_near_pixel(
            depth=depth,
            pixel=preferred_pixel,
            window_px=window_px,
        )
        if center_depth is not None:
            return {
                "depth_m": float(center_depth),
                "pixel": [int(preferred_pixel[0]), int(preferred_pixel[1])],
                "method": "center_window",
                "num_valid": None,
            }

        grid_estimate = cls._estimate_depth_from_bbox_grid(
            depth=depth,
            bbox=bbox,
            window_px=window_px,
        )
        if grid_estimate is not None:
            return grid_estimate

        return cls._estimate_depth_from_bbox_valid_pixels(depth=depth, bbox=bbox)

    @classmethod
    def _estimate_depth_from_bbox_grid(
        cls,
        depth: np.ndarray,
        bbox: List[int],
        window_px: int,
        grid_size: int = 5,
        inner_margin_ratio: float = 0.18,
    ) -> Optional[Dict[str, Any]]:
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = cls._clip_bbox(bbox, image_width=w, image_height=h)
        if x2 <= x1 or y2 <= y1:
            return None

        width = x2 - x1
        height = y2 - y1
        gx1 = x1 + width * inner_margin_ratio
        gx2 = x2 - width * inner_margin_ratio
        gy1 = y1 + height * inner_margin_ratio
        gy2 = y2 - height * inner_margin_ratio
        if gx2 <= gx1 or gy2 <= gy1:
            gx1, gx2, gy1, gy2 = float(x1), float(x2), float(y1), float(y2)

        samples = []
        for u in np.linspace(gx1, gx2, grid_size):
            for v in np.linspace(gy1, gy2, grid_size):
                pixel = [int(round(u)), int(round(v))]
                value = cls._median_depth_near_pixel(
                    depth=depth,
                    pixel=pixel,
                    window_px=window_px,
                )
                if value is not None:
                    samples.append({"pixel": pixel, "depth_m": float(value)})

        if not samples:
            return None

        depths = np.asarray([sample["depth_m"] for sample in samples], dtype=np.float32)
        median_depth = float(np.median(depths))
        best = min(samples, key=lambda sample: abs(sample["depth_m"] - median_depth))
        return {
            "depth_m": float(best["depth_m"]),
            "pixel": [int(best["pixel"][0]), int(best["pixel"][1])],
            "method": "bbox_grid_window",
            "num_valid": len(samples),
            "median_depth_m": median_depth,
        }

    @classmethod
    def _estimate_depth_from_bbox_valid_pixels(
        cls,
        depth: np.ndarray,
        bbox: List[int],
        inner_margin_ratio: float = 0.12,
        max_std_m: float = 0.12,
    ) -> Optional[Dict[str, Any]]:
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = cls._clip_bbox(bbox, image_width=w, image_height=h)
        if x2 <= x1 or y2 <= y1:
            return None

        width = x2 - x1
        height = y2 - y1
        mx = int(round(width * inner_margin_ratio))
        my = int(round(height * inner_margin_ratio))
        ix1 = min(max(x1 + mx, 0), w - 1)
        ix2 = min(max(x2 - mx, ix1 + 1), w)
        iy1 = min(max(y1 + my, 0), h - 1)
        iy2 = min(max(y2 - my, iy1 + 1), h)

        patch = depth[iy1:iy2, ix1:ix2]
        valid_mask = np.isfinite(patch) & (patch > 0.02) & (patch < 1.5)
        valid = patch[valid_mask]
        if len(valid) == 0:
            return None

        median_depth = float(np.median(valid))
        robust = valid[np.abs(valid - median_depth) <= max_std_m]
        if len(robust) > 0:
            median_depth = float(np.median(robust))

        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            pixel = [int(round((ix1 + ix2) * 0.5)), int(round((iy1 + iy2) * 0.5))]
        else:
            values = patch[valid_mask]
            best_index = int(np.argmin(np.abs(values - median_depth)))
            pixel = [int(ix1 + xs[best_index]), int(iy1 + ys[best_index])]

        return {
            "depth_m": median_depth,
            "pixel": pixel,
            "method": "bbox_valid_pixels",
            "num_valid": int(len(valid)),
            "median_depth_m": median_depth,
        }

    @staticmethod
    def _clip_bbox(bbox, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(x1, image_width - 1))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height - 1))
        y2 = max(0, min(y2, image_height))
        return x1, y1, x2, y2

    @staticmethod
    def _pixel_to_camera_xyz(
        pixel: List[int],
        depth_m: float,
        intrinsics: Dict[str, float],
    ) -> List[float]:
        u, v = [float(x) for x in pixel]
        z = float(depth_m)
        x = (u - float(intrinsics["cx"])) / float(intrinsics["fx"]) * z
        y = (v - float(intrinsics["cy"])) / float(intrinsics["fy"]) * z
        return [float(x), float(y), float(z)]

    @staticmethod
    def _camera_point_to_base(
        point_camera: List[float],
        current_ee_pose: List[float],
        handeye_rot,
        handeye_trans,
    ) -> List[float]:
        point_camera = np.asarray(point_camera, dtype=float).reshape(3)
        handeye_rot = np.asarray(handeye_rot, dtype=float)
        handeye_trans = np.asarray(handeye_trans, dtype=float).reshape(3)

        x_ee, y_ee, z_ee, rx_ee, ry_ee, rz_ee = current_ee_pose
        r_ee2base = R.from_euler(
            "ZYX",
            [rz_ee, ry_ee, rx_ee],
            degrees=False,
        ).as_matrix()

        point_ee = handeye_rot @ point_camera + handeye_trans
        point_base = r_ee2base @ point_ee + np.asarray([x_ee, y_ee, z_ee], dtype=float)

        return [float(x) for x in point_base.tolist()]

    def _execute_place(
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
            return {"success": False, "error": "移动到 approach pose 失败。"}

        if not self._move_xarm(arm, place_pose, move_speed, move_acc):
            return {"success": False, "error": "移动到 place pose 失败。"}

        if gripper_open and not gripper_open():
            return {"success": False, "error": "夹爪打开释放失败。"}

        if not self._move_xarm(arm, lift_pose, move_speed, move_acc):
            return {"success": False, "error": "释放后抬升失败。"}

        return {"success": True, "message": "放置执行完成。"}

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

    @staticmethod
    def _normalize_intrinsics(intrinsics):
        if intrinsics is None:
            return None
        if isinstance(intrinsics, dict):
            fx = intrinsics.get("fx")
            fy = intrinsics.get("fy")
            cx = intrinsics.get("cx", intrinsics.get("ppx"))
            cy = intrinsics.get("cy", intrinsics.get("ppy"))
            if fx is not None and fy is not None and cx is not None and cy is not None:
                return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
        return intrinsics

    @staticmethod
    def _bbox_area(bbox) -> float:
        if not bbox:
            return 0.0
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    @classmethod
    def _bbox_iou(cls, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union = cls._bbox_area(box_a) + cls._bbox_area(box_b) - inter
        return 0.0 if union <= 0 else float(inter / union)
