from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from core.skill_base import BaseSkill, SkillResult


OCR_VERIFY_RESULT_SCHEMA: Dict[str, Any] = {
    "task": "verify_medicine_ocr",
    "medicine_query": "",
    "strict_mode": False,
    "success": True,
    "failure_type": "success",
    "recognized_texts": [],
    "matched_candidates": [],
    "best_match": None,
    "confidence": 0.0,
    "evidence": {
        "raw_ocr_text": "",
        "selected_region": None,
        "vis_path": "",
    },
    "recommendation": "proceed",
}


class VerifyMedicineOCRSkill(BaseSkill):
    name = "verify_medicine_ocr"
    description = "基于 OCR 结果判别药盒识别是否成功。默认模糊匹配，可切换严格模式。"

    def run(
        self,
        medicine_query: str,
        camera_name: str = "wrist",
        bbox: Optional[List[int]] = None,
        image_path: Optional[str] = None,
        texts: Optional[List[str]] = None,
        raw_text: Optional[str] = None,
        strict_mode: bool = False,
        min_confidence: float = 0.0,
        fuzzy_threshold: float = 0.72,
        alias_map: Optional[Dict[str, List[str]]] = None,
        execute: bool = False,
        ocr_output_prefix: str = "runs/current_capture/ocr_verify",
        **kwargs,
    ) -> SkillResult:
        try:
            vis_path = ""
            selected_region = bbox

            if texts is None:
                if image_path is None:
                    camera = self.context.require("camera")
                    capture = self._capture(camera, camera_name)
                    image_path = capture["color_path"]

                ocr = self.context.require("ocr")
                if bbox is None:
                    ocr_result = ocr.read_image(
                        image_path=image_path,
                        output_prefix=ocr_output_prefix,
                        min_confidence=min_confidence,
                    )
                else:
                    ocr_result = ocr.read_region(
                        image_path=image_path,
                        bbox=bbox,
                        output_prefix=ocr_output_prefix,
                        min_confidence=min_confidence,
                    )

                texts = ocr_result.get("texts", [])
                raw_text = ocr_result.get("text", "")
                vis_path = ocr_result.get("vis_path", "")
            else:
                texts = list(texts)
                raw_text = raw_text if raw_text is not None else " ".join(texts)

            match_result = self._match_medicine_text(
                query=medicine_query,
                texts=texts,
                strict_mode=strict_mode,
                fuzzy_threshold=fuzzy_threshold,
                alias_map=alias_map,
            )

            success = bool(match_result["success"])
            data = {
                "task": "verify_medicine_ocr",
                "medicine_query": medicine_query,
                "strict_mode": strict_mode,
                "success": success,
                "failure_type": match_result["failure_type"],
                "recognized_texts": texts,
                "matched_candidates": match_result["matched_candidates"],
                "best_match": match_result["best_match"],
                "confidence": match_result["confidence"],
                "evidence": {
                    "raw_ocr_text": raw_text or "",
                    "selected_region": selected_region,
                    "vis_path": vis_path,
                },
                "recommendation": match_result["recommendation"],
            }

            return SkillResult(success=success, data=data, error=None)

        except Exception as e:
            err = f"verify_medicine_ocr failed: {e}"
            try:
                self.context.emit_error(err)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=err)

    @staticmethod
    def _capture(camera, camera_name: str) -> Dict[str, Any]:
        if hasattr(camera, "capture"):
            return camera.capture(camera_name)
        if camera_name == "wrist":
            return camera.capture_wrist()
        if camera_name == "global":
            return camera.capture_global()
        raise ValueError(f"Unknown camera_name: {camera_name}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(
            ch.lower()
            for ch in text.strip()
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
        )

    def _match_medicine_text(
        self,
        query: str,
        texts: List[str],
        strict_mode: bool,
        fuzzy_threshold: float,
        alias_map: Optional[Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        candidates = [query]
        if alias_map and query in alias_map:
            candidates.extend(alias_map[query])
        candidates.extend(self._default_aliases(query))

        normalized_candidates = []
        for candidate in candidates:
            normalized = self._normalize_text(candidate)
            if normalized and normalized not in normalized_candidates:
                normalized_candidates.append(normalized)

        matched_candidates: List[Dict[str, Any]] = []
        best_match = None
        best_score = 0.0

        if not texts:
            return {
                "success": False,
                "failure_type": "no_text",
                "matched_candidates": [],
                "best_match": None,
                "confidence": 0.0,
                "recommendation": "retry_ocr",
            }

        for text in texts:
            nt = self._normalize_text(text)
            score = 0.0
            match_type = "none"

            for cand in normalized_candidates:
                if strict_mode:
                    if nt == cand and cand:
                        score = 1.0
                        match_type = "exact"
                        break
                else:
                    if cand and (cand in nt or nt in cand):
                        s = 1.0
                        current_match_type = "contains"
                    else:
                        s = SequenceMatcher(None, nt, cand).ratio()
                        current_match_type = "fuzzy"
                    if s > score:
                        score = s
                        match_type = current_match_type

            if score >= fuzzy_threshold or (strict_mode and score == 1.0):
                matched_candidates.append(
                    {
                        "candidate": text,
                        "match_type": match_type,
                        "score": round(score, 4),
                    }
                )

            if score > best_score:
                best_score = score
                best_match = {
                    "candidate": text,
                    "match_type": match_type,
                    "score": round(score, 4),
                }

        if best_score >= fuzzy_threshold:
            return {
                "success": True,
                "failure_type": "success",
                "matched_candidates": matched_candidates,
                "best_match": best_match,
                "confidence": round(best_score, 4),
                "recommendation": "proceed",
            }

        if len(texts) == 1:
            return {
                "success": False,
                "failure_type": "wrong_text",
                "matched_candidates": matched_candidates,
                "best_match": best_match,
                "confidence": round(best_score, 4),
                "recommendation": "reobserve",
            }

        return {
            "success": False,
            "failure_type": "ambiguous",
            "matched_candidates": matched_candidates,
            "best_match": best_match,
            "confidence": round(best_score, 4),
            "recommendation": "refine_bbox",
        }

    @staticmethod
    def _default_aliases(query: str) -> List[str]:
        text = str(query or "").strip().lower()
        aliases = {
            "布洛芬": ["布洛芬缓释胶囊", "ibuprofen", "芬必得", "今来芬布得"],
            "抗病毒": ["抗病毒口服液", "999抗病毒口服液"],
            "抗病毒口服液": ["抗病毒", "999抗病毒口服液"],
            "蒙脱石散": ["mengtuoshi", "mengtuoshi san"],
            "西瓜霜": ["xiguashuang", "suai xiguashuang"],
        }
        result = []
        for key, values in aliases.items():
            if key.lower() in text or text in key.lower():
                result.extend(values)
        return result
