from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from core.skill_base import BaseSkill, SkillResult
from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient


GRASP_VDM_JSON_SCHEMA: Dict[str, Any] = {
    "name": "grasp_verification",
    "schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "medicine_query": {"type": "string"},
            "success": {"type": "boolean"},
            "failure_type": {
                "type": "string",
                "enum": [
                    "success",
                    "still_on_table",
                    "held_unclear",
                    "dropped",
                    "moved_out_of_view",
                    "occluded",
                    "uncertain",
                ],
            },
            "object_visible_before": {"type": "boolean"},
            "object_visible_after": {"type": "boolean"},
            "held_by_gripper": {"type": "boolean"},
            "dropped": {"type": "boolean"},
            "moved_or_lost": {"type": "boolean"},
            "change_summary": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {
                "type": "object",
                "properties": {
                    "before_global": {"type": "string"},
                    "before_wrist": {"type": "string"},
                    "after_global": {"type": "string"},
                    "after_wrist": {"type": "string"},
                    "table_evidence": {"type": "string"},
                    "gripper_evidence": {"type": "string"},
                    "negative_evidence": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "success_score": {"type": "number"},
            "recommendation": {
                "type": "string",
                "enum": ["proceed", "retry_grasp", "reobserve", "relocalize"],
            },
        },
        "required": [
            "task",
            "medicine_query",
            "success",
            "failure_type",
            "object_visible_before",
            "object_visible_after",
            "held_by_gripper",
            "dropped",
            "moved_or_lost",
            "change_summary",
            "confidence",
            "evidence",
            "success_score",
            "recommendation",
        ],
        "additionalProperties": False,
    },
}

GRASP_VDM_SYSTEM_PROMPT = """You are a strict robot manipulation grasp verifier for a medicine sorting task.

You must compare the BEFORE and AFTER observations from two cameras:
- global camera: overview of the tabletop
- wrist camera: close view near the gripper

Your job is to decide whether the medicine box grasp succeeded.

Image order:
- Each image is preceded by a text label.
- If a labeled 2x2 comparison sheet is available, inspect it first.
- Then inspect the original before_global, before_wrist, after_global, and after_wrist images.

You must be conservative:
- If evidence is insufficient, mark failure_type as "uncertain".
- Do not guess.
- Do not mark success just because the target is not obvious after grasp.
- Success requires positive evidence that the target object is held by or moving with the gripper.
- Do not output markdown.
- Do not output any text outside valid JSON.
- The JSON must strictly follow the provided schema.

Decision rules:
- Use the before images to locate the target medicine/package on the table.
- Use the after images to check whether the same target is no longer on the table.
- Use the after_wrist image to check whether the gripper is holding a rectangular medicine package or the target object.
- success = true only when:
  1. the target package is clearly no longer at its original tabletop location, AND
  2. the gripper clearly holds the target package or a matching medicine package, AND
  3. there is no clear evidence that it was dropped.
- If the target still appears at its original tabletop location, success = false and failure_type = "still_on_table".
- If a package is in the gripper but target identity is unclear, set success = false and failure_type = "held_unclear".
- If the table after image is unclear but wrist image clearly shows the object held by the gripper, use held_by_gripper=true, success_score around 0.65-0.80.
- If after images show the object fell, slipped, or is away from both table target location and gripper, failure_type = "dropped" or "moved_out_of_view".
- Confidence must reflect visual certainty. Use confidence < 0.5 for unclear images.
- success_score is a numeric visual evidence score from 0.0 to 1.0.
"""

GRASP_VDM_USER_PROMPT = """Task: verify whether the medicine grasp succeeded.

Target medicine: {medicine_query}

Please inspect the labeled comparison sheet first, then use the original images for detail.

Return a strict JSON object matching the schema.

Important:
- Compare before vs after.
- Use both global and wrist views together.
- Be conservative.
- If you cannot verify success, mark failure.
- Put concrete visual evidence into evidence.table_evidence and evidence.gripper_evidence.
- If the gripper holds only a corner/edge and the object may slip, use held_unclear unless the package is clearly lifted and stable.
"""


class VerifyMedicineGraspVDMSkill(BaseSkill):
    name = "verify_medicine_grasp_vdm"
    description = "基于抓取前后 global + wrist 四图，调用 VLM 进行视觉差分与抓取结果判别。"

    def run(
        self,
        medicine_query: str,
        before_global_color_path: str,
        before_wrist_color_path: str,
        after_global_color_path: str,
        after_wrist_color_path: str,
        base_url: str,
        api_key: str,
        model_name: str,
        prompt_template: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        execute: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        min_success_confidence: float = 0.60,
        min_success_score: float = 0.60,
        use_contact_sheet: bool = False,
        contact_sheet_path: Optional[str] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            client = OpenAIChatGPTVLMClient(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout,
            )

            user_prompt = (prompt_template or GRASP_VDM_USER_PROMPT).format(
                medicine_query=medicine_query
            )

            if response_format is None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": GRASP_VDM_JSON_SCHEMA,
                }

            image_paths = [
                before_global_color_path,
                before_wrist_color_path,
                after_global_color_path,
                after_wrist_color_path,
            ]
            if use_contact_sheet:
                sheet_path = contact_sheet_path or self._default_contact_sheet_path(
                    after_wrist_color_path
                )
                contact_sheet = self._build_contact_sheet(
                    before_global_color_path=before_global_color_path,
                    before_wrist_color_path=before_wrist_color_path,
                    after_global_color_path=after_global_color_path,
                    after_wrist_color_path=after_wrist_color_path,
                    output_path=sheet_path,
                )
                if contact_sheet:
                    image_paths = [contact_sheet, *image_paths]

            image_labels = self._image_labels(has_contact_sheet=len(image_paths) == 5)

            raw_text = client.describe_images(
                image_paths=image_paths,
                prompt=GRASP_VDM_SYSTEM_PROMPT + "\n\n" + user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                image_labels=image_labels,
            )

            parsed = self._parse_json_or_raise(raw_text)
            parsed = self._normalize_verdict(
                parsed,
                min_success_confidence=min_success_confidence,
                min_success_score=min_success_score,
            )
            success = bool(parsed.get("success", False))
            return SkillResult(success=success, data=parsed, error=None)

        except Exception as e:
            err = f"verify_medicine_grasp_vdm failed: {e}"
            try:
                self.context.emit_error(err)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=err)

    @staticmethod
    def _image_labels(has_contact_sheet: bool):
        labels = [
            "before_global: overview camera before grasp, target should be on the table.",
            "before_wrist: wrist camera before grasp, close view before grasp.",
            "after_global: overview camera after grasp/lift, check whether target remains on table.",
            "after_wrist: wrist camera after grasp/lift, check whether gripper holds target.",
        ]
        if has_contact_sheet:
            return [
                "labeled 2x2 contact sheet: BEFORE_GLOBAL, BEFORE_WRIST, AFTER_GLOBAL, AFTER_WRIST.",
                *labels,
            ]
        return labels

    @staticmethod
    def _default_contact_sheet_path(after_wrist_color_path: str) -> str:
        path = Path(after_wrist_color_path)
        return str(path.with_name(f"{path.stem}_contact_sheet.png"))

    def _build_contact_sheet(
        self,
        before_global_color_path: str,
        before_wrist_color_path: str,
        after_global_color_path: str,
        after_wrist_color_path: str,
        output_path: str,
    ) -> Optional[str]:
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageOps

            items = [
                ("BEFORE_GLOBAL", before_global_color_path),
                ("BEFORE_WRIST", before_wrist_color_path),
                ("AFTER_GLOBAL", after_global_color_path),
                ("AFTER_WRIST", after_wrist_color_path),
            ]
            cell_w, cell_h = 640, 360
            label_h = 42
            margin = 8
            sheet = Image.new(
                "RGB",
                (cell_w * 2 + margin * 3, (cell_h + label_h) * 2 + margin * 3),
                "white",
            )
            draw = ImageDraw.Draw(sheet)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            except Exception:
                font = ImageFont.load_default()

            for index, (label, image_path) in enumerate(items):
                row = index // 2
                col = index % 2
                x = margin + col * (cell_w + margin)
                y = margin + row * (cell_h + label_h + margin)
                img = Image.open(image_path).convert("RGB")
                img = ImageOps.contain(img, (cell_w, cell_h))
                canvas = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
                paste_x = (cell_w - img.width) // 2
                paste_y = (cell_h - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))
                draw.rectangle([x, y, x + cell_w, y + label_h], fill=(30, 30, 30))
                draw.text((x + 12, y + 8), label, fill="white", font=font)
                sheet.paste(canvas, (x, y + label_h))
                draw.rectangle(
                    [x, y, x + cell_w, y + label_h + cell_h],
                    outline=(30, 30, 30),
                    width=2,
                )

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(output)
            return str(output)
        except ImportError:
            return None
        except Exception as e:
            try:
                self.context.emit_warning(f"contact sheet build failed: {e}")
            except Exception:
                pass
            return None

    @staticmethod
    def _normalize_verdict(
        parsed: Dict[str, Any],
        min_success_confidence: float,
        min_success_score: float,
    ) -> Dict[str, Any]:
        confidence = VerifyMedicineGraspVDMSkill._safe_float(parsed.get("confidence"), 0.0)
        success_score = VerifyMedicineGraspVDMSkill._safe_float(parsed.get("success_score"), confidence)
        held_by_gripper = bool(parsed.get("held_by_gripper", False))
        dropped = bool(parsed.get("dropped", False))
        failure_type = str(parsed.get("failure_type") or "uncertain")

        if parsed.get("success", False):
            if (
                confidence < min_success_confidence
                or success_score < min_success_score
                or not held_by_gripper
                or dropped
            ):
                parsed["success"] = False
                parsed["failure_type"] = "held_unclear" if held_by_gripper and not dropped else "uncertain"
                parsed["recommendation"] = "retry_grasp"
                parsed["change_summary"] = (
                    str(parsed.get("change_summary") or "")
                    + " Normalized to failure because positive grasp evidence was insufficient."
                ).strip()
            else:
                parsed["failure_type"] = "success"
                parsed["recommendation"] = "proceed"
        else:
            if failure_type == "success":
                parsed["failure_type"] = "uncertain"
            parsed.setdefault("recommendation", "retry_grasp")

        parsed["confidence"] = confidence
        parsed["success_score"] = success_score
        return parsed

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _parse_json_or_raise(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError(f"Model output is not valid JSON: {text}")
