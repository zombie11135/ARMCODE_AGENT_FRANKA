# skill_lib/perception/ocr_instances.py

from typing import List, Dict, Any

from core.skill_base import BaseSkill, SkillResult


class OCRInstancesSkill(BaseSkill):
    name = "ocr_instances"
    description = "对 SAM3 分割得到的多个实例 crop 逐个 OCR，用于药盒/零食文字确认。"

    input_schema = {
        "instances": "segment_object 输出的 instances 列表，每个实例应包含 crop_path。",
        "min_confidence": "最低 OCR 置信度。",
    }

    output_schema = {
        "instances": "加入 OCR 结果后的实例列表。",
    }

    def run(
        self,
        instances: List[Dict[str, Any]],
        min_confidence: float = 0.0,
        output_prefix: str = "runs/current_capture/ocr_instances",
        **kwargs,
    ) -> SkillResult:
        try:
            ocr = self.context.require("ocr")

            new_instances = []

            for idx, inst in enumerate(instances):
                inst = dict(inst)

                if not inst.get("success", True):
                    inst["ocr_success"] = False
                    inst["ocr_error"] = "instance segmentation failed"
                    new_instances.append(inst)
                    continue

                crop_path = inst.get("crop_path")

                if not crop_path:
                    inst["ocr_success"] = False
                    inst["ocr_error"] = "missing crop_path"
                    new_instances.append(inst)
                    continue

                self.context.emit_text(
                    f"正在对实例 {inst.get('object_id', idx)} 做 OCR: {crop_path}"
                )

                try:
                    ocr_result = ocr.read_image(
                        image_path=crop_path,
                        output_prefix=f"{output_prefix}_{idx + 1:02d}",
                        min_confidence=min_confidence,
                    )

                    inst["ocr_success"] = True
                    inst["ocr_texts"] = ocr_result.get("texts", [])
                    inst["ocr_text"] = ocr_result.get("text", "")
                    inst["ocr_confidences"] = ocr_result.get("confidences", [])
                    inst["ocr_avg_confidence"] = ocr_result.get("avg_confidence", 0.0)
                    inst["ocr_items"] = ocr_result.get("items", [])
                    inst["ocr_vis_path"] = ocr_result.get("vis_path")

                    if inst["ocr_vis_path"]:
                        self.context.emit_image(
                            image_path=inst["ocr_vis_path"],
                            caption=f"OCR 实例 {inst.get('object_id', idx)} 可视化",
                        )

                    self.context.emit_text(
                        f"实例 {inst.get('object_id', idx)} OCR 文本：{inst['ocr_text']}"
                    )

                except Exception as e:
                    inst["ocr_success"] = False
                    inst["ocr_error"] = str(e)

                new_instances.append(inst)

            return SkillResult(
                success=True,
                data={
                    "instances": new_instances,
                    "num_instances": len(new_instances),
                },
                message=f"OCR instances 完成，共处理 {len(new_instances)} 个实例。",
            )

        except Exception as e:
            error = f"OCR instances 失败: {e}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={},
                error=error,
            )