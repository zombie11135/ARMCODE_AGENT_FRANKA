from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
import shutil
import time

from core.skill_base import BaseSkill, SkillResult
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.combound.place_to_container import PlaceToContainerSkill
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill


class InspectMedicineCandidatesSkill(BaseSkill):
    name = "inspect_medicine_candidates"
    description = "保守检测药盒候选，必要时对高置信候选近距离 OCR，输出可抓取的 confirmed medicine targets。"

    input_schema = {
        "camera_name": "默认 wrist。",
        "execute_observation": "是否真实移动机械臂到候选上方近距离观察。",
        "observe_z_mm": "近距离观察固定高度，默认 400mm。",
        "target_prompts": "可选。用于粗检测药盒类似物的 GroundingDINO prompts。",
    }

    output_schema = {
        "confirmed_targets": "已通过 OCR 确认的药品目标，包含 medicine_query / bbox / ocr_text。",
        "unknown_candidates": "检测到了但无法确认药名或不是药盒的候选。",
        "all_candidates": "完整候选及近距离观察明细。",
    }

    DEFAULT_TARGET_PROMPTS = [
        "medicine_box",
        "rectangular medicine carton",
        "pharmaceutical paper box",
        "medicine package with printed text",
        "drug package box",
    ]

    KNOWN_MEDICINES = [
        ("布洛芬缓释胶囊", ["布洛芬缓释胶囊", "布洛芬", "IBUPROFEN", "洛芬", "芬必得", "今来芬布得"]),
        ("抗病毒口服液", ["抗病毒口服液", "抗病毒", "口服液"]),
        ("蒙脱石散", ["蒙脱石散", "蒙脱石"]),
        ("西瓜霜", ["西瓜霜"]),
        ("999", ["999", "三九"]),
    ]

    MEDICINE_TEXT_HINTS = [
        "胶囊",
        "口服液",
        "颗粒",
        "片",
        "散",
        "丸",
        "药",
        "OTC",
        "准字",
        "症",
        "痛",
        "IBUPROFEN",
    ]

    def run(
        self,
        camera_name: str = "wrist",
        target_prompts: Optional[List[str]] = None,
        detection_threshold: Optional[float] = 0.26,
        text_threshold: Optional[float] = 0.22,
        max_detections_per_prompt: int = 8,
        max_candidates: int = 5,
        merge_iou: float = 0.62,
        initial_ocr_padding_px: int = 24,
        close_ocr_padding_px: int = 20,
        close_detection_threshold: Optional[float] = 0.20,
        close_text_threshold: Optional[float] = 0.16,
        execute_observation: bool = False,
        observe_z_mm: float = 400.0,
        observe_wait_s: float = 0.8,
        target_rpy: Tuple[float, float, float] = (3.14, 0.0, -1.57),
        depth_window_px: int = 25,
        fallback_intrinsics: Optional[Dict[str, float]] = None,
        xarm_ip: str = "192.168.1.237",
        handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
        move_speed: float = 50,
        move_acc: float = 500,
        allow_unknown_text_targets: bool = False,
        min_text_chars: int = 2,
        deduplicate_confirmed: bool = True,
        duplicate_iou_threshold: float = 0.35,
        duplicate_containment_threshold: float = 0.80,
        min_prompt_hits: int = 2,
        min_candidate_area_px: float = 12000.0,
        max_candidate_area_ratio: float = 0.55,
        edge_margin_px: int = 8,
        max_close_observations: int = 2,
        **kwargs,
    ) -> SkillResult:
        try:
            self.context.emit_text("开始候选药盒近距离确认。")

            finder = FindAndGraspMedicineSkill(self.context)
            capture = finder._capture(camera_name)
            capture = self._snapshot_capture(
                capture=capture,
                output_prefix="runs/current_capture/inspect_medicine_initial",
            )
            image_path = capture["color_path"]
            self.context.emit_image(image_path=image_path, caption="候选药盒粗检测输入图像")

            prompts = self._build_prompts(target_prompts)
            detections = finder._detect_with_prompt_fusion(
                image_path=image_path,
                prompts=prompts,
                threshold=detection_threshold,
                text_threshold=text_threshold,
                max_detections_per_prompt=max_detections_per_prompt,
                merge_iou=merge_iou,
            )
            detections, rejected_detections = self._filter_candidate_detections(
                detections=detections,
                image_path=image_path,
                min_prompt_hits=min_prompt_hits,
                min_candidate_area_px=min_candidate_area_px,
                max_candidate_area_ratio=max_candidate_area_ratio,
                edge_margin_px=edge_margin_px,
                max_candidates=max_candidates,
            )

            if not detections:
                return SkillResult(
                    success=False,
                    data={
                        "image_path": image_path,
                        "prompts": prompts,
                        "rejected_detections": rejected_detections,
                    },
                    error="没有检测到药盒类似候选。",
                )

            self.context.emit_text(f"检测到 {len(detections)} 个药盒类似候选。")

            current_ee_pose = None
            handeye_rot = None
            handeye_trans = None
            depth = None
            intrinsics = None

            if execute_observation:
                current_ee_pose = PlanGraspWithAnyGraspSkill(
                    self.context
                )._get_current_xarm_pose_mrad(xarm_ip=xarm_ip)
                if current_ee_pose is None:
                    return SkillResult(
                        success=False,
                        data={"image_path": image_path, "detections": detections},
                        error="无法通过 xArm SDK 获取当前末端位姿，不能执行近距离观察。",
                    )

                handeye_rot, handeye_trans = PlanGraspWithAnyGraspSkill._load_handeye_cam2ee(
                    handeye_config_path
                )

                depth_path = capture.get("depth_path")
                if not depth_path:
                    return SkillResult(
                        success=False,
                        data={"image_path": image_path, "detections": detections},
                        error="近距离观察需要 wrist depth，但当前 capture 缺少 depth_path。",
                    )

                depth = PlaceToContainerSkill._load_depth(
                    depth_path=depth_path,
                    depth_scale=capture.get("depth_scale", 1000.0),
                )
                intrinsics = PlaceToContainerSkill._normalize_intrinsics(
                    fallback_intrinsics
                    or capture.get("depth_intrinsics")
                    or capture.get("color_intrinsics")
                    or capture.get("camera_intrinsics")
                )
                if intrinsics is None:
                    return SkillResult(
                        success=False,
                        data={"image_path": image_path, "detections": detections},
                        error="缺少相机内参，无法将候选 bbox 转为机械臂观察位姿。",
                    )

            all_candidates = []
            confirmed_targets = []
            unknown_candidates = []
            close_observations_used = 0

            for idx, det in enumerate(detections):
                allow_close_observation = (
                    execute_observation
                    and close_observations_used < max_close_observations
                )
                candidate = self._inspect_one_candidate(
                    index=idx,
                    detection=det,
                    initial_capture=capture,
                    initial_image_path=image_path,
                    finder=finder,
                    prompts=prompts,
                    initial_ocr_padding_px=initial_ocr_padding_px,
                    close_ocr_padding_px=close_ocr_padding_px,
                    close_detection_threshold=close_detection_threshold,
                    close_text_threshold=close_text_threshold,
                    max_detections_per_prompt=max_detections_per_prompt,
                    merge_iou=merge_iou,
                    execute_observation=allow_close_observation,
                    observe_z_mm=observe_z_mm,
                    observe_wait_s=observe_wait_s,
                    target_rpy=target_rpy,
                    depth_window_px=depth_window_px,
                    depth=depth,
                    intrinsics=intrinsics,
                    current_ee_pose=current_ee_pose,
                    handeye_rot=handeye_rot,
                    handeye_trans=handeye_trans,
                    move_speed=move_speed,
                    move_acc=move_acc,
                    allow_unknown_text_targets=allow_unknown_text_targets,
                    min_text_chars=min_text_chars,
                )
                if allow_close_observation:
                    close_observations_used += 1

                all_candidates.append(candidate)
                if candidate.get("confirmed"):
                    confirmed_targets.append(candidate)
                else:
                    unknown_candidates.append(candidate)

            duplicate_groups = []
            if deduplicate_confirmed:
                confirmed_targets, duplicate_groups = self._dedupe_confirmed_targets(
                    confirmed_targets=confirmed_targets,
                    iou_threshold=duplicate_iou_threshold,
                    containment_threshold=duplicate_containment_threshold,
                )

            self.context.emit_text(
                "候选确认完成："
                f"confirmed={len(confirmed_targets)}, "
                f"unknown={len(unknown_candidates)}, "
                f"duplicates={len(duplicate_groups)}"
            )

            return SkillResult(
                success=bool(confirmed_targets),
                data={
                    "image_path": image_path,
                    "prompts": prompts,
                    "detections": detections,
                    "rejected_detections": rejected_detections,
                    "confirmed_targets": confirmed_targets,
                    "unknown_candidates": unknown_candidates,
                    "all_candidates": all_candidates,
                    "duplicate_groups": duplicate_groups,
                    "execute_observation": execute_observation,
                    "close_observations_used": close_observations_used,
                },
                message=f"已确认 {len(confirmed_targets)} 个可抓取药品目标。",
                error=None if confirmed_targets else "没有确认到可抓取药品目标。",
            )

        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            error = f"inspect_medicine_candidates 执行失败: {e}"
            try:
                self.context.emit_error(error)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=error)

    def _inspect_one_candidate(
        self,
        index: int,
        detection: Dict[str, Any],
        initial_capture: Dict[str, Any],
        initial_image_path: str,
        finder: FindAndGraspMedicineSkill,
        prompts: List[str],
        initial_ocr_padding_px: int,
        close_ocr_padding_px: int,
        close_detection_threshold: Optional[float],
        close_text_threshold: Optional[float],
        max_detections_per_prompt: int,
        merge_iou: float,
        execute_observation: bool,
        observe_z_mm: float,
        observe_wait_s: float,
        target_rpy: Tuple[float, float, float],
        depth_window_px: int,
        depth,
        intrinsics,
        current_ee_pose,
        handeye_rot,
        handeye_trans,
        move_speed: float,
        move_acc: float,
        allow_unknown_text_targets: bool,
        min_text_chars: int,
    ) -> Dict[str, Any]:
        object_id = detection.get("object_id", f"candidate_{index + 1}")
        bbox = detection.get("bbox")
        candidate = {
            "object_id": object_id,
            "bbox": bbox,
            "detection": detection,
            "initial_image_path": initial_image_path,
            "confirmed": False,
            "medicine_query": None,
            "ocr_text": "",
            "ocr_texts": [],
            "close_observation": None,
        }

        self.context.emit_text(f"检查候选 {object_id}: bbox={bbox}")

        initial_ocr = self._ocr_region(
            image_path=initial_image_path,
            bbox=bbox,
            output_prefix=f"runs/current_capture/inspect_medicine_{index + 1:02d}_initial",
            padding=initial_ocr_padding_px,
        )
        candidate["initial_ocr"] = initial_ocr

        texts = list(initial_ocr.get("texts", []))
        ocr_sources = ["initial"]

        if execute_observation:
            observe_result = self._move_and_capture_close_view(
                candidate=candidate,
                initial_capture=initial_capture,
                bbox=bbox,
                depth=depth,
                intrinsics=intrinsics,
                current_ee_pose=current_ee_pose,
                handeye_rot=handeye_rot,
                handeye_trans=handeye_trans,
                target_rpy=target_rpy,
                observe_z_mm=observe_z_mm,
                observe_wait_s=observe_wait_s,
                depth_window_px=depth_window_px,
                move_speed=move_speed,
                move_acc=move_acc,
            )
            candidate["close_observation"] = observe_result

            if observe_result.get("success"):
                close_capture = observe_result["capture"]
                close_image_path = close_capture["color_path"]
                self.context.emit_image(
                    image_path=close_image_path,
                    caption=f"候选 {object_id} 近距离观察图像",
                )

                close_ocr = self._ocr_close_view(
                    image_path=close_image_path,
                    finder=finder,
                    prompts=prompts,
                    close_detection_threshold=close_detection_threshold,
                    close_text_threshold=close_text_threshold,
                    max_detections_per_prompt=max_detections_per_prompt,
                    merge_iou=merge_iou,
                    padding=close_ocr_padding_px,
                    index=index,
                )
                candidate["close_ocr"] = close_ocr
                texts.extend(close_ocr.get("texts", []))
                ocr_sources.append("close")
            else:
                candidate.setdefault("warnings", []).append(observe_result.get("error"))

        texts = self._dedupe_texts(texts)
        candidate["ocr_texts"] = texts
        candidate["ocr_text"] = " ".join(texts)
        candidate["ocr_sources"] = ocr_sources

        medicine_query, reason = self._extract_medicine_query(
            texts=texts,
            allow_unknown_text_targets=allow_unknown_text_targets,
            min_text_chars=min_text_chars,
        )
        candidate["medicine_query"] = medicine_query
        candidate["confirm_reason"] = reason
        candidate["confirmed"] = medicine_query is not None

        if candidate["confirmed"]:
            self.context.emit_text(
                f"候选 {object_id} 确认为药品目标：{medicine_query} ({reason})"
            )
        else:
            self.context.emit_warning(
                f"候选 {object_id} 未确认药名：{candidate['ocr_text'] or '<empty>'}"
            )

        return candidate

    @classmethod
    def _dedupe_confirmed_targets(
        cls,
        confirmed_targets: List[Dict[str, Any]],
        iou_threshold: float,
        containment_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        duplicate_groups: List[Dict[str, Any]] = []

        for target in confirmed_targets:
            duplicate_index = None
            for idx, old in enumerate(kept):
                if cls._is_duplicate_target(
                    old=old,
                    new=target,
                    iou_threshold=iou_threshold,
                    containment_threshold=containment_threshold,
                ):
                    duplicate_index = idx
                    break

            if duplicate_index is None:
                kept.append(target)
                continue

            old = kept[duplicate_index]
            best = cls._choose_better_duplicate(old, target)
            removed = target if best is old else old
            kept[duplicate_index] = best

            duplicate_groups.append(
                {
                    "medicine_query": best.get("medicine_query"),
                    "kept_object_id": best.get("object_id"),
                    "removed_object_id": removed.get("object_id"),
                    "kept_bbox": best.get("bbox"),
                    "removed_bbox": removed.get("bbox"),
                    "reason": cls._duplicate_reason(old, target),
                }
            )

        return kept, duplicate_groups

    @classmethod
    def _is_duplicate_target(
        cls,
        old: Dict[str, Any],
        new: Dict[str, Any],
        iou_threshold: float,
        containment_threshold: float,
    ) -> bool:
        if old.get("medicine_query") != new.get("medicine_query"):
            return False

        old_bbox = old.get("bbox")
        new_bbox = new.get("bbox")
        if not old_bbox or not new_bbox:
            return False

        iou = cls._bbox_iou(old_bbox, new_bbox)
        containment = max(
            cls._bbox_containment(old_bbox, new_bbox),
            cls._bbox_containment(new_bbox, old_bbox),
        )
        return iou >= iou_threshold or containment >= containment_threshold

    @classmethod
    def _choose_better_duplicate(
        cls,
        old: Dict[str, Any],
        new: Dict[str, Any],
    ) -> Dict[str, Any]:
        old_score = cls._target_quality_score(old)
        new_score = cls._target_quality_score(new)
        return new if new_score > old_score else old

    @classmethod
    def _target_quality_score(cls, target: Dict[str, Any]) -> float:
        bbox_area = cls._bbox_area(target.get("bbox"))
        text_len = len(str(target.get("ocr_text") or ""))
        close_bonus = 1.0 if target.get("close_observation") else 0.0
        alias_bonus = 1.0 if str(target.get("confirm_reason", "")).startswith("matched_alias") else 0.0
        # Prefer full-object boxes over tiny text-region boxes, but still keep
        # a useful OCR signal in the ranking.
        return 0.65 * min(bbox_area / 120000.0, 1.0) + 0.25 * min(text_len / 40.0, 1.0) + 0.05 * close_bonus + 0.05 * alias_bonus

    @classmethod
    def _duplicate_reason(cls, old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, float]:
        old_bbox = old.get("bbox")
        new_bbox = new.get("bbox")
        return {
            "iou": cls._bbox_iou(old_bbox, new_bbox),
            "old_contains_new": cls._bbox_containment(old_bbox, new_bbox),
            "new_contains_old": cls._bbox_containment(new_bbox, old_bbox),
        }

    def _move_and_capture_close_view(
        self,
        candidate: Dict[str, Any],
        initial_capture: Dict[str, Any],
        bbox: List[int],
        depth,
        intrinsics,
        current_ee_pose,
        handeye_rot,
        handeye_trans,
        target_rpy: Tuple[float, float, float],
        observe_z_mm: float,
        observe_wait_s: float,
        depth_window_px: int,
        move_speed: float,
        move_acc: float,
    ) -> Dict[str, Any]:
        if bbox is None:
            return {"success": False, "error": "候选缺少 bbox，无法近距离观察。"}

        center_pixel = PlaceToContainerSkill._bbox_center_pixel(
            bbox=bbox,
            offset_ratio=(0.0, 0.0),
            image_width=depth.shape[1],
            image_height=depth.shape[0],
        )
        center_depth = PlaceToContainerSkill._median_depth_near_pixel(
            depth=depth,
            pixel=center_pixel,
            window_px=depth_window_px,
        )
        if center_depth is None:
            return {"success": False, "error": "候选中心附近没有有效深度。"}

        center_camera_xyz = PlaceToContainerSkill._pixel_to_camera_xyz(
            pixel=center_pixel,
            depth_m=center_depth,
            intrinsics=intrinsics,
        )
        center_base_xyz = PlaceToContainerSkill._camera_point_to_base(
            point_camera=center_camera_xyz,
            current_ee_pose=current_ee_pose,
            handeye_rot=handeye_rot,
            handeye_trans=handeye_trans,
        )
        if self.context.transforms is not None:
            from hardware.agent_geometry import point_to_base
            center_base_xyz = point_to_base(initial_capture, center_camera_xyz, self.context.transforms)

        observe_pose = [
            center_base_xyz[0] * 1000.0,
            center_base_xyz[1] * 1000.0,
            float(observe_z_mm),
            float(target_rpy[0]),
            float(target_rpy[1]),
            float(target_rpy[2]),
        ]

        self.context.emit_text(f"移动到候选近距离观察位姿：{observe_pose}")
        arm = self.context.require("xarm")
        if not PlaceToContainerSkill._move_xarm(arm, observe_pose, move_speed, move_acc):
            return {
                "success": False,
                "error": "移动到候选近距离观察位姿失败。",
                "observe_pose_mmrad": observe_pose,
            }

        if observe_wait_s > 0:
            time.sleep(float(observe_wait_s))

        camera = self.context.require("camera")
        close_capture = camera.capture_wrist()
        return {
            "success": True,
            "center_pixel": center_pixel,
            "center_depth_m": center_depth,
            "center_camera_xyz": center_camera_xyz,
            "center_base_xyz": center_base_xyz,
            "observe_pose_mmrad": observe_pose,
            "capture": close_capture,
        }

    def _ocr_close_view(
        self,
        image_path: str,
        finder: FindAndGraspMedicineSkill,
        prompts: List[str],
        close_detection_threshold: Optional[float],
        close_text_threshold: Optional[float],
        max_detections_per_prompt: int,
        merge_iou: float,
        padding: int,
        index: int,
    ) -> Dict[str, Any]:
        texts = []
        items = []
        confidences = []
        vis_paths = []
        crop_paths = []

        try:
            full = self.context.require("ocr").read_image(
                image_path=image_path,
                output_prefix=f"runs/current_capture/inspect_medicine_{index + 1:02d}_close_full",
                min_confidence=0.0,
            )
            texts.extend(full.get("texts", []))
            items.extend(full.get("items", []))
            confidences.extend(full.get("confidences", []))
            if full.get("vis_path"):
                vis_paths.append(full["vis_path"])
        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            return {"texts": [], "error": f"近距离整图 OCR 失败: {e}"}

        try:
            close_detections = finder._detect_with_prompt_fusion(
                image_path=image_path,
                prompts=prompts,
                threshold=close_detection_threshold,
                text_threshold=close_text_threshold,
                max_detections_per_prompt=max_detections_per_prompt,
                merge_iou=merge_iou,
            )
            selected = self._select_center_detection(close_detections, image_path=image_path)
            if selected is not None:
                region = self._ocr_region(
                    image_path=image_path,
                    bbox=selected["bbox"],
                    output_prefix=f"runs/current_capture/inspect_medicine_{index + 1:02d}_close_region",
                    padding=padding,
                )
                texts.extend(region.get("texts", []))
                items.extend(region.get("items", []))
                confidences.extend(region.get("confidences", []))
                vis_paths.extend(region.get("vis_paths", []))
                if region.get("crop_path"):
                    crop_paths.append(region["crop_path"])
        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            return {
                "texts": self._dedupe_texts(texts),
                "items": items,
                "confidences": confidences,
                "vis_paths": vis_paths,
                "crop_paths": crop_paths,
                "warning": f"近距离 bbox OCR 失败: {e}",
            }

        return {
            "texts": self._dedupe_texts(texts),
            "text": " ".join(self._dedupe_texts(texts)),
            "items": items,
            "confidences": confidences,
            "avg_confidence": self._avg(confidences),
            "vis_paths": vis_paths,
            "crop_paths": crop_paths,
        }

    def _ocr_region(
        self,
        image_path: str,
        bbox: List[int],
        output_prefix: str,
        padding: int,
    ) -> Dict[str, Any]:
        try:
            result = self.context.require("ocr").read_region(
                image_path=image_path,
                bbox=bbox,
                output_prefix=output_prefix,
                padding=padding,
                min_confidence=0.0,
            )
            texts = self._dedupe_texts(result.get("texts", []))
            if result.get("vis_path"):
                self.context.emit_image(
                    image_path=result["vis_path"],
                    caption=f"OCR: {output_prefix}",
                )
            return {
                "texts": texts,
                "text": " ".join(texts),
                "items": result.get("items", []),
                "confidences": result.get("confidences", []),
                "avg_confidence": result.get("avg_confidence", 0.0),
                "vis_paths": [result["vis_path"]] if result.get("vis_path") else [],
                "crop_path": result.get("crop_path"),
            }
        except Exception as e:
            from hardware.fr3 import MotionError
            from hardware.robotiq import GripperError
            if isinstance(e, (MotionError, GripperError)):
                raise
            return {"texts": [], "text": "", "error": str(e)}

    @staticmethod
    def _build_prompts(target_prompts: Optional[List[str]]) -> List[str]:
        prompts = list(target_prompts or InspectMedicineCandidatesSkill.DEFAULT_TARGET_PROMPTS)
        clean = []
        for prompt in prompts:
            prompt = str(prompt).strip()
            if prompt and prompt not in clean:
                clean.append(prompt)
        return clean

    @classmethod
    def _filter_candidate_detections(
        cls,
        detections: List[Dict[str, Any]],
        image_path: str,
        min_prompt_hits: int,
        min_candidate_area_px: float,
        max_candidate_area_ratio: float,
        edge_margin_px: int,
        max_candidates: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            image_width = 1280
            image_height = 720
        else:
            image_height, image_width = image.shape[:2]

        image_area = float(image_width * image_height)
        accepted = []
        rejected = []

        for det in detections:
            reason = cls._reject_detection_reason(
                det=det,
                image_width=image_width,
                image_height=image_height,
                image_area=image_area,
                min_prompt_hits=min_prompt_hits,
                min_candidate_area_px=min_candidate_area_px,
                max_candidate_area_ratio=max_candidate_area_ratio,
                edge_margin_px=edge_margin_px,
            )
            if reason:
                item = dict(det)
                item["reject_reason"] = reason
                rejected.append(item)
            else:
                accepted.append(det)

        return accepted[:max_candidates], rejected

    @classmethod
    def _reject_detection_reason(
        cls,
        det: Dict[str, Any],
        image_width: int,
        image_height: int,
        image_area: float,
        min_prompt_hits: int,
        min_candidate_area_px: float,
        max_candidate_area_ratio: float,
        edge_margin_px: int,
    ) -> Optional[str]:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            return "missing_bbox"

        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = cls._bbox_area(bbox)
        prompt_hits = len(det.get("prompt_hits", []))

        if prompt_hits < int(min_prompt_hits):
            return f"low_prompt_hits:{prompt_hits}"

        if area < float(min_candidate_area_px):
            return f"too_small:{area:.0f}"

        if image_area > 0 and area / image_area > float(max_candidate_area_ratio):
            return f"too_large:{area / image_area:.2f}"

        if (
            x1 <= edge_margin_px
            or y1 <= edge_margin_px
            or x2 >= image_width - edge_margin_px
            or y2 >= image_height - edge_margin_px
        ):
            return "touches_image_edge"

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect = max(w / h, h / w)
        if aspect > 4.0:
            return f"extreme_aspect:{aspect:.2f}"

        return None

    @staticmethod
    def _snapshot_capture(capture: Dict[str, Any], output_prefix: str) -> Dict[str, Any]:
        stable = dict(capture)
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        copy_keys = [
            ("color_path", f"{output_prefix}_color.png"),
            ("depth_path", f"{output_prefix}_depth.png"),
            ("depth_npy_path", f"{output_prefix}_depth.npy"),
            ("depth_meter_path", f"{output_prefix}_depth_meter.npy"),
        ]

        for key, dst in copy_keys:
            src = stable.get(key)
            if not src:
                continue
            src_path = Path(src)
            if not src_path.exists():
                continue
            dst_path = Path(dst)
            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
            stable[key] = str(dst_path)

        return stable

    @classmethod
    def _extract_medicine_query(
        cls,
        texts: List[str],
        allow_unknown_text_targets: bool,
        min_text_chars: int,
    ) -> Tuple[Optional[str], str]:
        joined = " ".join([str(t) for t in texts if str(t).strip()])
        joined_upper = joined.upper()

        for canonical, aliases in cls.KNOWN_MEDICINES:
            for alias in aliases:
                if alias.upper() in joined_upper:
                    return canonical, f"matched_alias:{alias}"

        for text in texts:
            clean = cls._clean_ocr_text(text)
            if len(clean) < min_text_chars:
                continue
            if cls._looks_like_medicine_text(clean):
                return clean, "medicine_text_hint"

        if allow_unknown_text_targets:
            candidates = [cls._clean_ocr_text(t) for t in texts]
            candidates = [t for t in candidates if len(t) >= min_text_chars]
            candidates.sort(key=len, reverse=True)
            if candidates:
                return candidates[0], "unknown_text_allowed"

        return None, "no_reliable_medicine_text"

    @classmethod
    def _looks_like_medicine_text(cls, text: str) -> bool:
        upper = text.upper()
        return any(hint.upper() in upper for hint in cls.MEDICINE_TEXT_HINTS)

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
        return text

    @staticmethod
    def _dedupe_texts(texts: List[str]) -> List[str]:
        clean = []
        for text in texts:
            text = str(text).strip()
            if text and text not in clean:
                clean.append(text)
        return clean

    @staticmethod
    def _avg(values: List[float]) -> float:
        values = [float(v) for v in values]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _select_center_detection(detections: List[Dict[str, Any]], image_path: str):
        if not detections:
            return None

        import cv2

        image = cv2.imread(image_path)
        if image is None:
            return detections[0]

        h, w = image.shape[:2]
        cx = w * 0.5
        cy = h * 0.5

        def score(det):
            x1, y1, x2, y2 = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
            bx = (x1 + x2) * 0.5
            by = (y1 + y2) * 0.5
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            return (dist, -area)

        return sorted(detections, key=score)[0]

    @staticmethod
    def _bbox_area(bbox) -> float:
        if not bbox:
            return 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @classmethod
    def _bbox_iou(cls, box_a, box_b) -> float:
        if not box_a or not box_b:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        union = cls._bbox_area(box_a) + cls._bbox_area(box_b) - inter
        return 0.0 if union <= 0 else float(inter / union)

    @classmethod
    def _bbox_containment(cls, outer, inner) -> float:
        if not outer or not inner:
            return 0.0
        ox1, oy1, ox2, oy2 = [float(v) for v in outer]
        ix1, iy1, ix2, iy2 = [float(v) for v in inner]
        inter_x1 = max(ox1, ix1)
        inter_y1 = max(oy1, iy1)
        inter_x2 = min(ox2, ix2)
        inter_y2 = min(oy2, iy2)
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        inner_area = cls._bbox_area(inner)
        return 0.0 if inner_area <= 0 else float(inter / inner_area)
