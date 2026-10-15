# skill_lib/perception/detect_object.py

from pathlib import Path
from typing import Optional

import cv2

from core.skill_base import BaseSkill, SkillResult


class DetectObjectSkill(BaseSkill):
    name = "detect_object"
    description = "使用 GroundingDINO 在腕部相机图像中检测水果、药盒、零食等抓取目标 bbox。"

    input_schema = {
        "target_name": "目标类别，例如 medicine_box / fruit / apple / orange / snack / tray",
        "image_path": "可选。若不提供，则默认调用腕部相机拍摄 RGB 图像。",
        "camera_name": "默认 wrist，也可以是 global。",
        "threshold": "可选。检测框阈值。",
        "text_threshold": "可选。文本匹配阈值。",
    }

    output_schema = {
        "image_path": "检测输入图像路径",
        "detections": "检测结果列表，每个包含 object_id, name, bbox, confidence, phrase",
        "vis_path": "bbox 可视化图像路径",
    }

    def run(
        self,
        target_name: str = "medicine_box",
        image_path: Optional[str] = None,
        camera_name: str = "wrist",
        threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            detector = self.context.require("detector")

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
                    caption=f"{camera_name} 相机检测输入图像",
                )

            self.context.emit_text(
                f"正在检测目标：{target_name}，图像：{image_path}"
            )

            detect_result = detector.detect(
                image_path=image_path,
                target_name=target_name,
                threshold=threshold,
                text_threshold=text_threshold,
            )

            detections = detect_result["detections"]

            vis_path = self._draw_detections(
                image_path=image_path,
                detections=detections,
                output_path="runs/current_capture/detect_object_vis.png",
            )

            self.context.emit_image(
                image_path=vis_path,
                caption=f"GroundingDINO 检测结果：{target_name}",
            )

            self.context.emit_text(
                f"目标检测完成：{target_name}，检测到 {len(detections)} 个候选目标。"
            )

            return SkillResult(
                success=True,
                data={
                    "image_path": image_path,
                    "target_name": target_name,
                    "detections": detections,
                    "vis_path": vis_path,
                    "text_prompt": detect_result.get("text_prompt", ""),
                    "threshold": detect_result.get("threshold"),
                    "text_threshold": detect_result.get("text_threshold"),
                },
                message=f"检测完成，共找到 {len(detections)} 个候选目标。",
            )

        except Exception as e:
            error = f"目标检测失败: {e}"

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
    def _draw_detections(
        image_path: str,
        detections,
        output_path: str,
    ) -> str:
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det.get("name", "")
            conf = det.get("confidence", 0.0)
            phrase = det.get("phrase", "")
            obj_id = det.get("object_id", "")

            cv2.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            label = f"{obj_id}:{name} {conf:.2f}"

            if phrase:
                label += f" [{phrase}]"

            cv2.putText(
                image,
                label,
                (int(x1), max(25, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

        return output_path