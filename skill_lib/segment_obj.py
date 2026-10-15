# skill_lib/perception/segment_object.py

from pathlib import Path
from typing import Optional, List

from core.skill_base import BaseSkill, SkillResult


class SegmentObjectSkill(BaseSkill):
    name = "segment_object"
    description = "根据 bbox 对目标物体进行图像分割，输出 mask，用于后续抓取、点云提取和三维定位。"

    input_schema = {
        "image_path": "可选。输入 RGB 图像路径。若不提供，则默认使用腕部相机拍摄。",
        "bbox": "可选。目标 bbox，格式 [x1, y1, x2, y2]。",
        "target_name": "可选。若未提供 bbox，则用 detector 检测该目标。",
        "camera_name": "默认 wrist。",
        "detection_index": "当检测到多个目标时，选择第几个，默认 0。",
    }

    output_schema = {
        "image_path": "分割输入图像路径",
        "bbox": "用于分割的 bbox",
        "mask_path": "目标 mask 图像路径",
        "overlay_path": "mask 叠加可视化图路径",
        "masked_image_path": "只保留目标区域的图像路径",
        "crop_path": "目标 crop 图像路径",
        "mask_area": "mask 面积，像素数量",
    }

    def run(
        self,
        image_path: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        target_name: Optional[str] = None,
        camera_name: str = "wrist",
        detection_index: int = 0,
        output_prefix: str = "runs/current_capture/segment_object",
        **kwargs,
    ) -> SkillResult:
        try:
            segmenter = self.context.require("segmenter")

            # 1. 如果没有 image_path，则拍腕部图像
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
                    caption=f"{camera_name} 相机分割输入图像",
                )

            # 2. 如果没有 bbox，则调用 detector
            detection_used = None

            if bbox is None:
                if target_name is None:
                    raise ValueError("bbox 和 target_name 至少需要提供一个。")

                detector = self.context.require("detector")

                self.context.emit_text(
                    f"未提供 bbox，正在先检测目标：{target_name}"
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
                        },
                        error=f"没有检测到目标：{target_name}",
                    )

                detections = sorted(
                    detections,
                    key=lambda x: x.get("confidence", 0.0),
                    reverse=True,
                )

                if detection_index >= len(detections):
                    raise IndexError(
                        f"detection_index={detection_index} 超出检测结果数量 {len(detections)}"
                    )

                detection_used = detections[detection_index]
                bbox = detection_used["bbox"]

                self.context.emit_text(
                    f"选择检测目标 {detection_used.get('object_id')}，bbox={bbox}"
                )

            # 3. 调用分割服务
            self.context.emit_text(
                f"正在分割目标，image={image_path}, bbox={bbox}"
            )

            segment_result = segmenter.segment_by_bbox(
                image_path=image_path,
                bbox=bbox,
                output_prefix=output_prefix,
                **kwargs,
            )

            overlay_path = segment_result.get("overlay_path")
            mask_path = segment_result.get("mask_path")
            crop_path = segment_result.get("crop_path")
            masked_image_path = segment_result.get("masked_image_path")

            if overlay_path:
                self.context.emit_image(
                    image_path=overlay_path,
                    caption="目标分割 overlay 可视化结果",
                )

            if mask_path:
                self.context.emit_image(
                    image_path=mask_path,
                    caption="目标分割 mask",
                )

            if crop_path:
                self.context.emit_image(
                    image_path=crop_path,
                    caption="目标 crop 图像",
                )

            data = {
                "image_path": image_path,
                "bbox": bbox,
                "target_name": target_name,
                "detection_used": detection_used,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "masked_image_path": masked_image_path,
                "crop_path": crop_path,
                "mask_area": segment_result.get("mask_area"),
                "segmenter": segment_result.get("segmenter"),
            }

            return SkillResult(
                success=True,
                data=data,
                message="目标分割成功。",
            )

        except Exception as e:
            error = f"目标分割失败: {e}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={},
                error=error,
            )