# skill_lib/compound/find_object_by_text.py

from typing import Optional, List

from core.skill_base import BaseSkill, SkillResult

from skill_lib.perception.seg_obj import SegmentObjectSkill
from skill_lib.perception.ocr_instances import OCRInstancesSkill
from skill_lib.combound.select_obj_by_text import SelectObjectByTextSkill


class FindObjectByTextSkill(BaseSkill):
    name = "find_object_by_text"
    description = "复合技能：分割指定类别的所有物体，对每个实例 OCR，并根据文本选择目标。"

    input_schema = {
        "target_name": "目标类别，例如 medicine_box / snack。",
        "query": "需要匹配的文字，例如 布洛芬 / 西瓜霜 / 999。",
        "camera_name": "默认 wrist。",
        "aliases": "可选。query 的别名列表。",
        "min_score": "文本选择最低分数。",
    }

    output_schema = {
        "selected_instance": "最终选中的目标实例。",
        "segmentation_result": "分割结果。",
        "ocr_result": "OCR 结果。",
        "selection_result": "文本选择结果。",
    }

    def run(
        self,
        target_name: str,
        query: str,
        camera_name: str = "wrist",
        aliases: Optional[List[str]] = None,
        min_score: float = 0.45,
        max_instances: int = 10,
        expand_bbox_px: int = 8,
        select_by: str = "iou",
        **kwargs,
    ) -> SkillResult:
        try:
            self.context.emit_text(
                f"开始复合技能 find_object_by_text: target={target_name}, query={query}"
            )

            # 1. GroundingDINO + SAM3 分割所有候选实例
            segment_skill = SegmentObjectSkill(self.context)

            seg_result = segment_skill.run(
                target_name=target_name,
                camera_name=camera_name,
                segment_all=True,
                max_instances=max_instances,
                output_prefix="runs/current_capture/find_object_segment",
                expand_bbox_px=expand_bbox_px,
                select_by=select_by,
            )

            if not seg_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "segment_object",
                        "segmentation_result": seg_result.data,
                    },
                    error=f"分割候选目标失败: {seg_result.error}",
                )

            instances = seg_result.data.get("instances", [])

            if len(instances) == 0:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "segment_object",
                        "segmentation_result": seg_result.data,
                    },
                    error=f"没有分割出任何候选目标: {target_name}",
                )

            # 2. 对每个实例 OCR
            ocr_skill = OCRInstancesSkill(self.context)

            ocr_result = ocr_skill.run(
                instances=instances,
                min_confidence=0.0,
                output_prefix="runs/current_capture/find_object_ocr",
            )

            if not ocr_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "ocr_instances",
                        "segmentation_result": seg_result.data,
                        "ocr_result": ocr_result.data,
                    },
                    error=f"OCR 候选目标失败: {ocr_result.error}",
                )

            ocr_instances = ocr_result.data.get("instances", [])

            # 3. 根据文字选择目标
            select_skill = SelectObjectByTextSkill(self.context)

            select_result = select_skill.run(
                instances=ocr_instances,
                query=query,
                aliases=aliases,
                min_score=min_score,
            )

            if not select_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "select_object_by_text",
                        "segmentation_result": seg_result.data,
                        "ocr_result": ocr_result.data,
                        "selection_result": select_result.data,
                    },
                    error=f"没有找到文字匹配目标: {select_result.error}",
                )

            selected_instance = select_result.data["selected_instance"]

            self.context.emit_text(
                f"复合技能完成：已找到目标 {query}，object_id={selected_instance.get('object_id')}"
            )

            return SkillResult(
                success=True,
                data={
                    "selected_instance": selected_instance,
                    "segmentation_result": seg_result.data,
                    "ocr_result": ocr_result.data,
                    "selection_result": select_result.data,
                },
                message=f"成功找到文本匹配目标：{query}",
            )

        except Exception as e:
            error = f"find_object_by_text 执行失败: {e}"

            try:
                self.context.emit_error(error)
            except Exception:
                pass

            return SkillResult(
                success=False,
                data={},
                error=error,
            )