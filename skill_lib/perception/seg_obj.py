# skill_lib/perception/seg_obj.py

from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2

from core.skill_base import BaseSkill, SkillResult


class SegmentObjectSkill(BaseSkill):
    name = "segment_object"
    description = "使用 GroundingDINO 的 bbox 作为 SAM3 box prompt，对一个或多个目标物体进行精细分割。"

    input_schema = {
        "image_path": "可选。输入 RGB 图像路径。若不提供，则默认使用腕部相机拍摄。",
        "bbox": "可选。单个 bbox，格式 [x1, y1, x2, y2]。",
        "target_name": "可选。若未提供 bbox，则先用 GroundingDINO 检测该目标。",
        "camera_name": "默认 wrist。",
        "detection_index": "当 segment_all=False 时，选择第几个检测框，默认 0。",
        "segment_all": "是否对所有检测到的 bbox 都做 SAM3 分割，默认 True。",
        "expand_bbox_px": "bbox 外扩像素，默认 5。",
    }

    output_schema = {
        "image_path": "分割输入图像路径",
        "target_name": "目标类别",
        "detections": "GroundingDINO 检测结果",
        "instances": "SAM3 分割后的实例列表",
        "summary_path": "所有 mask 的汇总可视化图",
    }

    def run(
        self,
        image_path: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        target_name: Optional[str] = None,
        camera_name: str = "wrist",
        detection_index: int = 0,
        segment_all: bool = True,
        output_prefix: str = "runs/current_capture/segment_object",
        expand_bbox_px: int = 5,
        select_by: str = "iou",
        max_instances: Optional[int] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            segmenter = self.context.require("segmenter")

            # 1. 如果没有传 image_path，则拍一张图
            if image_path is None:
                camera = self.context.require("camera")

                if camera_name == "wrist":
                    capture_result = camera.capture_wrist()
                elif camera_name == "global":
                    capture_result = camera.capture_global()
                else:
                    raise ValueError(f"Unknown camera_name: {camera_name}")

                image_path = capture_result["color_path"]

                self.context.emit_image(
                    image_path=image_path,
                    caption=f"{camera_name} 相机 SAM3 分割输入图像",
                )

            detections = []
            selected_detections = []

            # 2. 如果直接传入 bbox，则只分割这个 bbox
            if bbox is not None:
                selected_detections = [
                    {
                        "object_id": "manual_obj_1",
                        "name": target_name or "manual_target",
                        "bbox": bbox,
                        "confidence": None,
                        "phrase": "manual_bbox",
                        "image_path": image_path,
                    }
                ]

            # 3. 如果没有 bbox，则先用 GroundingDINO 检测
            else:
                if target_name is None:
                    raise ValueError("bbox 和 target_name 至少需要提供一个。")

                detector = self.context.require("detector")

                self.context.emit_text(
                    f"未提供 bbox，正在使用 GroundingDINO 检测目标：{target_name}"
                )

                detect_result = detector.detect(
                    image_path=image_path,
                    target_name=target_name,
                )

                detections = detect_result.get("detections", [])

                if len(detections) == 0:
                    return SkillResult(
                        success=False,
                        data={
                            "image_path": image_path,
                            "target_name": target_name,
                            "detections": [],
                            "instances": [],
                        },
                        error=f"GroundingDINO 没有检测到目标：{target_name}",
                    )

                detections = sorted(
                    detections,
                    key=lambda x: x.get("confidence", 0.0),
                    reverse=True,
                )

                if segment_all:
                    selected_detections = detections
                else:
                    if detection_index >= len(detections):
                        raise IndexError(
                            f"detection_index={detection_index} 超出检测结果数量 {len(detections)}"
                        )
                    selected_detections = [detections[detection_index]]

            if max_instances is not None:
                selected_detections = selected_detections[:max_instances]

            self.context.emit_text(
                f"准备对 {len(selected_detections)} 个 bbox 进行 SAM3 分割。"
            )

            # 4. 对每个 bbox 逐个调用 SAM3
            instances = []

            base_prefix = Path(output_prefix)

            for idx, det in enumerate(selected_detections):
                obj_id = det.get("object_id", f"obj_{idx + 1}")
                det_bbox = det["bbox"]

                instance_prefix = str(
                    base_prefix.parent / f"{base_prefix.name}_{idx + 1:02d}_{obj_id}"
                )

                self.context.emit_text(
                    f"正在分割 {obj_id}: bbox={det_bbox}"
                )

                try:
                    segment_result = segmenter.segment_by_bbox(
                        image_path=image_path,
                        bbox=det_bbox,
                        output_prefix=instance_prefix,
                        expand_bbox_px=expand_bbox_px,
                        select_by=select_by,
                        **kwargs,
                    )

                    instance = {
                        "object_id": obj_id,
                        "name": det.get("name", target_name),
                        "target_name": target_name,
                        "detection": det,
                        "bbox": segment_result.get("bbox", det_bbox),
                        "confidence": det.get("confidence"),
                        "phrase": det.get("phrase"),
                        "mask_path": segment_result.get("mask_path"),
                        "mask_npy_path": segment_result.get("mask_npy_path"),
                        "overlay_path": segment_result.get("overlay_path"),
                        "masked_image_path": segment_result.get("masked_image_path"),
                        "crop_path": segment_result.get("crop_path"),
                        "mask_area": segment_result.get("mask_area"),
                        "selected_score": segment_result.get("selected_score"),
                        "selected_box": segment_result.get("selected_box"),
                        "sam3_box_cxcywh_norm": segment_result.get("sam3_box_cxcywh_norm"),
                        "segmenter": segment_result.get("segmenter"),
                        "success": True,
                        "error": None,
                    }

                    instances.append(instance)

                except Exception as e:
                    instance = {
                        "object_id": obj_id,
                        "name": det.get("name", target_name),
                        "target_name": target_name,
                        "detection": det,
                        "bbox": det_bbox,
                        "success": False,
                        "error": str(e),
                    }

                    instances.append(instance)

                    self.context.emit_error(
                        f"{obj_id} 分割失败: {e}"
                    )

            # 5. 保存汇总图
            summary_path = self._draw_summary(
                image_path=image_path,
                instances=instances,
                output_path=str(base_prefix.parent / f"{base_prefix.name}_summary.png"),
            )

            if summary_path:
                self.context.emit_image(
                    image_path=summary_path,
                    caption="SAM3 多目标分割汇总图",
                )

            # 6. 输出前几个实例的 overlay，避免事件太多
            for inst in instances[:5]:
                if inst.get("success") and inst.get("overlay_path"):
                    self.context.emit_image(
                        image_path=inst["overlay_path"],
                        caption=f"SAM3 分割结果：{inst['object_id']}",
                    )

            success_instances = [x for x in instances if x.get("success")]

            data = {
                "image_path": image_path,
                "target_name": target_name,
                "detections": detections,
                "segment_all": segment_all,
                "instances": instances,
                "summary_path": summary_path,
                "num_detections": len(detections),
                "num_selected": len(selected_detections),
                "num_success": len(success_instances),
            }

            # 兼容旧版本：如果只有一个实例，也把顶层字段填上
            if len(success_instances) > 0:
                first = success_instances[0]
                data.update(
                    {
                        "detection_used": first.get("detection"),
                        "bbox": first.get("bbox"),
                        "mask_path": first.get("mask_path"),
                        "mask_npy_path": first.get("mask_npy_path"),
                        "overlay_path": first.get("overlay_path"),
                        "masked_image_path": first.get("masked_image_path"),
                        "crop_path": first.get("crop_path"),
                        "mask_area": first.get("mask_area"),
                        "selected_score": first.get("selected_score"),
                        "selected_box": first.get("selected_box"),
                        "sam3_box_cxcywh_norm": first.get("sam3_box_cxcywh_norm"),
                        "segmenter": first.get("segmenter"),
                    }
                )

            return SkillResult(
                success=len(success_instances) > 0,
                data=data,
                message=f"SAM3 分割完成：检测到 {len(detections)} 个 bbox，成功分割 {len(success_instances)} 个。",
                error=None if len(success_instances) > 0 else "所有 bbox 分割均失败。",
            )

        except Exception as e:
            error = f"SAM3 bbox prompt 多目标分割失败: {e}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={},
                error=error,
            )

    @staticmethod
    def _draw_summary(
        image_path: str,
        instances: List[Dict[str, Any]],
        output_path: str,
    ) -> Optional[str]:
        image = cv2.imread(image_path)

        if image is None:
            return None

        for inst in instances:
            if not inst.get("success"):
                continue

            bbox = inst.get("bbox")
            mask_path = inst.get("mask_path")
            obj_id = inst.get("object_id", "")
            name = inst.get("name", "")
            conf = inst.get("confidence")

            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]

            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    color_layer = image.copy()
                    color_layer[mask > 0] = (0, 255, 0)
                    image = cv2.addWeighted(image, 0.75, color_layer, 0.25, 0)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            if conf is None:
                label = f"{obj_id}:{name}"
            else:
                label = f"{obj_id}:{name} {conf:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

        return output_path