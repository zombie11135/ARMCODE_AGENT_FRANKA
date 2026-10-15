from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from core.skill_base import BaseSkill, SkillResult
from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient


PLACED_VDM_JSON_SCHEMA: Dict[str, Any] = {
    "name": "medicine_placed_verification",
    "schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "medicine_query": {"type": "string"},
            "container_name": {"type": "string"},
            "container_color": {"type": "string"},
            "success": {"type": "boolean"},
            "failure_type": {
                "type": "string",
                "enum": [
                    "success",
                    "not_in_container",
                    "container_occluded",
                    "wrong_object",
                    "uncertain",
                ],
            },
            "target_visible_in_container": {"type": "boolean"},
            "target_visible_elsewhere": {"type": "boolean"},
            "container_visible": {"type": "boolean"},
            "confidence": {"type": "number"},
            "evidence": {
                "type": "object",
                "properties": {
                    "before_global": {"type": "string"},
                    "before_wrist": {"type": "string"},
                    "after_global": {"type": "string"},
                    "after_wrist": {"type": "string"},
                    "container_evidence": {"type": "string"},
                    "negative_evidence": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "recommendation": {
                "type": "string",
                "enum": ["proceed", "retry_place", "retry_grasp", "reobserve"],
            },
        },
        "required": [
            "task",
            "medicine_query",
            "container_name",
            "success",
            "failure_type",
            "target_visible_in_container",
            "target_visible_elsewhere",
            "container_visible",
            "confidence",
            "evidence",
            "recommendation",
        ],
        "additionalProperties": False,
    },
}


PLACED_VDM_PROMPT = """You are a strict robot task verifier.

Task: decide whether the target medicine/package was placed inside the target container.

Image order:
- Each image is preceded by a text label.
- before_global: scene before placement attempt
- before_wrist: wrist view before placement attempt
- after_global: scene after placement attempt
- after_wrist: wrist view after placement attempt

Decision rules:
- The target container is identified by both object type and color when color is provided.
- success=true if the target medicine/package, a matching medicine box, or the newly placed package-like object is visible inside the target container after placement.
- The package may be partially occluded by the container rim or by other already placed objects; partial visibility inside the target container is enough for success.
- Do not require the medicine text to be fully readable after placement if the before images and placement context identify the target.
- Use failure_type="not_in_container" only when the target is clearly visible outside the target container after placement.
- If the after images are ambiguous, success=false and failure_type="uncertain"; do not guess a hard failure.
- Return strict JSON only.

Target medicine: {medicine_query}
Target container: {container_name}
Target container color: {container_color}
"""


class VerifyMedicinePlacedVDMSkill(BaseSkill):
    name = "verify_medicine_placed_vdm"
    description = "基于放置前后 global + wrist 四图，验证目标药盒是否进入容器。"

    def run(
        self,
        medicine_query: str,
        container_name: str,
        before_global_color_path: str,
        before_wrist_color_path: str,
        after_global_color_path: str,
        after_wrist_color_path: str,
        base_url: str,
        api_key: str,
        model_name: str,
        container_color: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout: int = 120,
        response_format: Optional[Dict[str, Any]] = None,
        min_success_confidence: float = 0.55,
        min_visible_success_confidence: float = 0.35,
        **kwargs,
    ) -> SkillResult:
        try:
            client = OpenAIChatGPTVLMClient(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout,
            )
            if response_format is None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": PLACED_VDM_JSON_SCHEMA,
                }

            raw_text = client.describe_images(
                image_paths=[
                    before_global_color_path,
                    before_wrist_color_path,
                    after_global_color_path,
                    after_wrist_color_path,
                ],
                image_labels=[
                    "before_global: scene before placement attempt.",
                    "before_wrist: wrist view before placement attempt.",
                    "after_global: scene after placement attempt, inspect container contents.",
                    "after_wrist: wrist view after placement attempt, inspect container contents.",
                ],
                prompt=PLACED_VDM_PROMPT.format(
                    medicine_query=medicine_query,
                    container_name=container_name,
                    container_color=container_color or "unspecified",
                ),
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            parsed = self._parse_json_or_raise(raw_text)
            parsed["container_color"] = str(container_color or "")
            confidence = self._safe_float(parsed.get("confidence"), 0.0)
            if parsed.get("target_visible_in_container") and confidence >= min_visible_success_confidence:
                parsed["success"] = True
                parsed["failure_type"] = "success"
                parsed["recommendation"] = "proceed"
            if parsed.get("success") and confidence < min_success_confidence:
                parsed["success"] = False
                parsed["failure_type"] = "uncertain"
                parsed["recommendation"] = "reobserve"
            parsed["confidence"] = confidence
            return SkillResult(success=bool(parsed.get("success")), data=parsed)
        except Exception as e:
            err = f"verify_medicine_placed_vdm failed: {e}"
            try:
                self.context.emit_error(err)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=err)

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
