# skill_lib/perception/ocr_region.py

from typing import Optional, List

from core.skill_base import BaseSkill, SkillResult


class OCRRegionSkill(BaseSkill):
    name = "ocr_region"
    description = "对指定图像或 bbox 区域进行 OCR，用于读取药盒、零食包装上的文字。"

    input_schema = {
        "image_path": "输入图像路径。可以是 crop_path，也可以是完整 wrist_color.png。",
        "bbox": "可选。若提供，则先裁剪 bbox 区域再 OCR。",
        "camera_name": "若 image_path 不提供，则默认使用 wrist 相机拍摄。",
        "padding": "bbox 裁剪时外扩像素。",
        "min_confidence": "OCR 最低置信度过滤。",
    }

    output_schema = {
        "texts": "识别出的文本列表",
        "text": "拼接后的文本",
        "confidences": "每个文本的置信度",
        "avg_confidence": "平均置信度",
        "items": "详细 OCR 结果",
        "crop_path": "如果使用 bbox，则返回裁剪图路径",
        "vis_path": "OCR 可视化图路径",
    }

    def run(
        self,
        image_path: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        camera_name: str = "wrist",
        padding: int = 5,
        min_confidence: float = 0.0,
        output_prefix: str = "runs/current_capture/ocr_region",
        **kwargs,
    ) -> SkillResult:
        try:
            ocr = self.context.require("ocr")

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
                    caption=f"{camera_name} 相机 OCR 输入图像",
                )

            self.context.emit_text(
                f"正在进行 OCR，image={image_path}, bbox={bbox}"
            )

            if bbox is not None:
                result = ocr.read_region(
                    image_path=image_path,
                    bbox=bbox,
                    output_prefix=output_prefix,
                    padding=padding,
                    min_confidence=min_confidence,
                )
            else:
                result = ocr.read_image(
                    image_path=image_path,
                    output_prefix=output_prefix,
                    min_confidence=min_confidence,
                )

            vis_path = result.get("vis_path")
            crop_path = result.get("crop_path")

            if crop_path:
                self.context.emit_image(
                    image_path=crop_path,
                    caption="OCR 裁剪区域",
                )

            if vis_path:
                self.context.emit_image(
                    image_path=vis_path,
                    caption="OCR 识别结果可视化",
                )

            text = result.get("text", "")
            self.context.emit_text(f"OCR 识别文本：{text}")

            return SkillResult(
                success=True,
                data=result,
                message="OCR 识别完成。",
            )

        except Exception as e:
            error = f"OCR 识别失败: {e}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={},
                error=error,
            )