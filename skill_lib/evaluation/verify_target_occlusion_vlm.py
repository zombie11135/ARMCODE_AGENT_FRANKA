from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.skill_base import BaseSkill, SkillResult
from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient


TARGET_OCCLUSION_VLM_JSON_SCHEMA: Dict[str, Any] = {
    "name": "target_occlusion_verification",
    "schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "target_object_name": {"type": "string"},
            "target_visible": {"type": "boolean"},
            "target_partially_visible": {"type": "boolean"},
            "text_visible": {"type": "boolean"},
            "target_occluded": {"type": "boolean"},
            "occlusion_level": {"type": "number"},
            "blocking_object_visible": {"type": "boolean"},
            "blocking_object_description": {"type": "string"},
            "grasp_safe": {"type": "boolean"},
            "confidence": {"type": "number"},
            "evidence": {
                "type": "object",
                "properties": {
                    "target_evidence": {"type": "string"},
                    "occlusion_evidence": {"type": "string"},
                    "grasp_risk": {"type": "string"},
                    "negative_evidence": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "recommendation": {
                "type": "string",
                "enum": ["proceed", "reobserve", "replan"],
            },
        },
        "required": [
            "task",
            "target_object_name",
            "target_visible",
            "target_partially_visible",
            "text_visible",
            "target_occluded",
            "occlusion_level",
            "blocking_object_visible",
            "blocking_object_description",
            "grasp_safe",
            "confidence",
            "evidence",
            "recommendation",
        ],
        "additionalProperties": False,
    },
}


TARGET_OCCLUSION_VLM_PROMPT = """You are a strict robot pre-grasp safety gate for a tabletop medicine-box grasp task.

Your job is to decide whether the target medicine box is safe to grasp now.

Target medicine/package: {target_object_name}
Target bbox if available: {bbox_text}

The bbox is only a noisy detector/OCR candidate for the visible target region.
It is NOT proof that the whole physical medicine box is visible. Inspect the
entire image around the bbox and reason about the full rectangular package.

Important decision rules:
- The robot may perform a top-down or near top-down grasp on the target package.
- If another object is lying on top of, pressing on, covering, or strongly overlapping the target package, set target_occluded=true and grasp_safe=false.
- If only the printed text is visible but the body of the package is covered or under another object, this is NOT safe to grasp. Set target_occluded=true.
- If the target text/logo is visible but only part of the medicine box body is visible, set target_partially_visible=true, target_occluded=true, and grasp_safe=false.
- If the bbox tightly surrounds only the exposed part of the target while the rest of the package appears hidden behind another object, this is meaningful occlusion. Do NOT call it fully visible just because the bbox contents are clear.
- If the full outline, full top surface, or expected rectangular body boundary of the medicine box is interrupted, hidden, or overlapped by another object, set target_occluded=true.
- If the target is not visible or cannot be localized, treat it as fully occluded/unknown for grasp safety: target_visible=false, target_occluded=true, occlusion_level close to 1.0, grasp_safe=false.
- If a small edge/corner is blocked but most of the top surface and graspable body are clear, occlusion_level may be low and grasp_safe may be true.
- Be conservative. Do not allow grasp just because OCR text is readable.
- When uncertain whether the visible region is the complete package or only an exposed fragment, choose reobserve/replan rather than proceed.
- If a blocking object is visible, describe it in blocking_object_description with visual cues such as color, relative position, and object type. Do not output pixel coordinates.
- Return strict JSON only. Do not output markdown.

Occlusion level guide:
- 0.0 to 0.2: clear or tiny edge occlusion.
- 0.2 to 0.35: mild occlusion, usually still graspable.
- 0.35 to 0.7: meaningful occlusion or object overlap, unsafe for grasp.
- 0.7 to 1.0: severe or complete occlusion, unsafe for grasp.
"""


class VerifyTargetOcclusionVLMSkill(BaseSkill):
    name = "verify_target_occlusion_vlm"
    description = "使用 VLM 判断目标药盒是否被其他物体遮挡或压住，作为抓取前 Gate 输入。"

    def run(
        self,
        target_object_name: str,
        image_path: str,
        base_url: str,
        api_key: str,
        model_name: str,
        bbox: Optional[List[float]] = None,
        annotated_image_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout: int = 120,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            if not image_path:
                return SkillResult(
                    success=False,
                    data={},
                    error="verify_target_occlusion_vlm missing image_path",
                )

            image_paths = [image_path]
            image_labels = ["wrist/global observation image for pre-grasp occlusion check."]

            annotated = self._build_annotated_image(
                image_path=image_path,
                bbox=bbox,
                output_path=annotated_image_path,
            )
            if annotated:
                image_paths = [annotated, image_path]
                image_labels = [
                    "annotated image: red rectangle marks target bbox when available.",
                    "original image.",
                ]

            client = OpenAIChatGPTVLMClient(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout,
            )
            if response_format is None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": TARGET_OCCLUSION_VLM_JSON_SCHEMA,
                }

            raw_text = client.describe_images(
                image_paths=image_paths,
                image_labels=image_labels,
                prompt=TARGET_OCCLUSION_VLM_PROMPT.format(
                    target_object_name=target_object_name,
                    bbox_text=list(bbox) if bbox else "not available",
                ),
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            parsed = self._parse_json_or_raise(raw_text)
            parsed = self._normalize_verdict(parsed, target_object_name)
            return SkillResult(success=True, data=parsed)
        except Exception as e:
            err = f"verify_target_occlusion_vlm failed: {e}"
            try:
                self.context.emit_error(err)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=err)

    @staticmethod
    def _normalize_verdict(parsed: Dict[str, Any], target_object_name: str) -> Dict[str, Any]:
        parsed["target_object_name"] = str(parsed.get("target_object_name") or target_object_name)
        parsed["confidence"] = VerifyTargetOcclusionVLMSkill._clamp01(parsed.get("confidence"))
        parsed["occlusion_level"] = VerifyTargetOcclusionVLMSkill._clamp01(parsed.get("occlusion_level"))

        target_visible = bool(parsed.get("target_visible"))
        target_partially_visible = bool(parsed.get("target_partially_visible"))
        target_occluded = bool(parsed.get("target_occluded"))
        grasp_safe = bool(parsed.get("grasp_safe"))
        occlusion_level = float(parsed["occlusion_level"])

        if not target_visible:
            target_occluded = True
            grasp_safe = False
            occlusion_level = max(occlusion_level, 0.9)
        if target_partially_visible:
            target_occluded = True
            grasp_safe = False
            occlusion_level = max(occlusion_level, 0.45)
        if target_occluded or occlusion_level > 0.35:
            grasp_safe = False
            target_occluded = True
        if not grasp_safe and parsed.get("recommendation") == "proceed":
            parsed["recommendation"] = "replan" if target_occluded else "reobserve"
        if grasp_safe and not target_occluded:
            parsed["recommendation"] = "proceed"

        parsed["target_visible"] = target_visible
        parsed["target_partially_visible"] = target_partially_visible
        parsed["target_occluded"] = target_occluded
        parsed["grasp_safe"] = grasp_safe
        parsed["occlusion_level"] = occlusion_level
        parsed.setdefault("evidence", {})
        return parsed

    @staticmethod
    def _build_annotated_image(
        image_path: str,
        bbox: Optional[List[float]],
        output_path: Optional[str],
    ) -> Optional[str]:
        if not bbox or len(bbox) != 4:
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return None

        try:
            source = Path(image_path)
            output = Path(output_path) if output_path else source.with_name(
                f"{source.stem}_vlm_occlusion_bbox.png"
            )
            image = Image.open(source).convert("RGB")
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
            except Exception:
                font = ImageFont.load_default()
            label = "TARGET BBOX"
            draw.rectangle([x1, max(0, y1 - 38), x1 + 220, y1], fill=(255, 0, 0))
            draw.text((x1 + 6, max(0, y1 - 34)), label, fill=(255, 255, 255), font=font)
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            return str(output)
        except Exception:
            return None

    @staticmethod
    def _parse_json_or_raise(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError(f"Model output is not valid JSON: {text}")

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        return max(0.0, min(1.0, numeric))
