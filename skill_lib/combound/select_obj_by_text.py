# skill_lib/compound/select_object_by_text.py

import re
import difflib
from typing import Any, Dict, List, Optional

from core.skill_base import BaseSkill, SkillResult


DEFAULT_TEXT_ALIASES = {
    "布洛芬": [
        "布洛芬",
        "芬必得",
        "ibuprofen",
        "IBUPROFEN",
        "ibuprofe",
        "buprofen",
    ],
    "西瓜霜": [
        "西瓜霜",
        "桂林西瓜霜",
        "西瓜",
        "瓜霜",
    ],
    "999": [
        "999",
        "三九",
        "抗病毒口服液",
        "抗病毒",
        "口服液",
    ],
    "抗病毒": [
        "抗病毒",
        "抗病毒口服液",
        "999",
        "口服液",
    ],
    "抗病毒口服液": [
        "抗病毒口服液",
        "抗病毒",
        "口服液",
        "999",
        "三九",
    ],
    "蒙脱石散": [
        "蒙脱石散",
        "蒙脱石",
        "思密达",
        "smecta",
    ],
}


class SelectObjectByTextSkill(BaseSkill):
    name = "select_object_by_text"
    description = "根据 OCR 文本从多个候选实例中选择目标物体，例如选择写有“布洛芬”的药盒。"

    input_schema = {
        "instances": "ocr_instances 输出的实例列表，每个实例应包含 ocr_text / ocr_texts。",
        "query": "需要匹配的目标文本，例如 布洛芬 / 西瓜霜 / 999。",
        "aliases": "可选。query 的别名列表。",
        "min_score": "最低匹配分数，低于该分数则认为未找到目标。",
    }

    output_schema = {
        "selected_instance": "被选中的实例。",
        "selected_index": "被选中实例在 instances 中的索引。",
        "score": "匹配分数。",
        "matched_keyword": "匹配到的关键词。",
        "ranked_candidates": "所有候选实例的排序结果。",
    }

    def run(
        self,
        instances: List[Dict[str, Any]],
        query: str,
        aliases: Optional[List[str]] = None,
        min_score: float = 0.45,
        **kwargs,
    ) -> SkillResult:
        try:
            if not instances:
                return SkillResult(
                    success=False,
                    data={
                        "instances": [],
                        "query": query,
                    },
                    error="没有可选择的实例 instances。",
                )

            keywords = self._build_keywords(query, aliases)

            ranked_candidates = []

            for idx, inst in enumerate(instances):
                text = self._collect_instance_text(inst)
                score, matched_keyword, match_type = self._score_text(
                    text=text,
                    keywords=keywords,
                )

                ocr_conf = float(inst.get("ocr_avg_confidence", 0.0) or 0.0)

                # OCR 置信度只作为轻微加权，避免置信度低但关键词明确时被错误过滤
                final_score = 0.85 * score + 0.15 * min(max(ocr_conf, 0.0), 1.0)

                candidate = {
                    "index": idx,
                    "object_id": inst.get("object_id"),
                    "bbox": inst.get("bbox"),
                    "crop_path": inst.get("crop_path"),
                    "ocr_text": text,
                    "ocr_avg_confidence": ocr_conf,
                    "score": final_score,
                    "raw_text_score": score,
                    "matched_keyword": matched_keyword,
                    "match_type": match_type,
                    "instance": inst,
                }

                ranked_candidates.append(candidate)

            ranked_candidates = sorted(
                ranked_candidates,
                key=lambda x: x["score"],
                reverse=True,
            )

            best = ranked_candidates[0]

            if best["score"] < min_score:
                return SkillResult(
                    success=False,
                    data={
                        "query": query,
                        "keywords": keywords,
                        "ranked_candidates": ranked_candidates,
                    },
                    error=(
                        f"没有找到匹配 query='{query}' 的实例。"
                        f"最高分={best['score']:.3f}, "
                        f"OCR文本={best['ocr_text']}"
                    ),
                )

            selected_instance = best["instance"]

            self.context.emit_text(
                f"已根据文本选择目标：query={query}, "
                f"object_id={best.get('object_id')}, "
                f"score={best['score']:.3f}, "
                f"matched={best.get('matched_keyword')}"
            )

            if selected_instance.get("crop_path"):
                self.context.emit_image(
                    image_path=selected_instance["crop_path"],
                    caption=f"根据文本选择的目标 crop：{query}",
                )

            if selected_instance.get("overlay_path"):
                self.context.emit_image(
                    image_path=selected_instance["overlay_path"],
                    caption=f"根据文本选择的目标分割结果：{query}",
                )

            return SkillResult(
                success=True,
                data={
                    "query": query,
                    "keywords": keywords,
                    "selected_instance": selected_instance,
                    "selected_index": best["index"],
                    "score": best["score"],
                    "matched_keyword": best["matched_keyword"],
                    "match_type": best["match_type"],
                    "ranked_candidates": ranked_candidates,
                },
                message=f"成功选择匹配文本 '{query}' 的目标实例。",
            )

        except Exception as e:
            error = f"文本目标选择失败: {e}"

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
    def _normalize_text(text: str) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.lower()
        text = text.replace("〇", "0").replace("Ｏ", "0").replace("o", "0")
        text = text.replace("Ⅰ", "1").replace("ｌ", "1").replace("l", "1")
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[，。、“”‘’：:；;（）()【】\[\]{}<>《》,.\-_/\\|]", "", text)

        return text

    @classmethod
    def _build_keywords(
        cls,
        query: str,
        aliases: Optional[List[str]],
    ) -> List[str]:
        keywords = [query]

        if aliases:
            keywords.extend(aliases)

        if query in DEFAULT_TEXT_ALIASES:
            keywords.extend(DEFAULT_TEXT_ALIASES[query])

        # 去重，同时保留顺序
        clean = []
        for k in keywords:
            if k is None:
                continue
            k = str(k).strip()
            if k and k not in clean:
                clean.append(k)

        return clean

    @staticmethod
    def _collect_instance_text(inst: Dict[str, Any]) -> str:
        parts = []

        if inst.get("ocr_text"):
            parts.append(str(inst["ocr_text"]))

        if inst.get("ocr_texts"):
            parts.extend([str(x) for x in inst["ocr_texts"]])

        # 有时 OCR 结果在 ocr_items 里面
        for item in inst.get("ocr_items", []) or []:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))

        return " ".join(parts).strip()

    @classmethod
    def _score_text(
        cls,
        text: str,
        keywords: List[str],
    ):
        norm_text = cls._normalize_text(text)

        if not norm_text:
            return 0.0, None, "empty_text"

        best_score = 0.0
        best_keyword = None
        best_type = "none"

        for keyword in keywords:
            norm_kw = cls._normalize_text(keyword)

            if not norm_kw:
                continue

            # 1. 直接包含，最高优先级
            if norm_kw in norm_text:
                return 1.0, keyword, "substring"

            # 2. 反向包含，例如 OCR 只识别出 query 的一部分
            if norm_text in norm_kw and len(norm_text) >= 2:
                score = 0.75
                if score > best_score:
                    best_score = score
                    best_keyword = keyword
                    best_type = "reverse_substring"

            # 3. 模糊匹配
            ratio = difflib.SequenceMatcher(None, norm_kw, norm_text).ratio()

            if ratio > best_score:
                best_score = ratio
                best_keyword = keyword
                best_type = "fuzzy"

            # 4. 局部窗口模糊匹配：OCR 常把药盒上的整段说明拼在一起，
            # 直接拿 keyword 和全文比会被长文本稀释。
            if len(norm_text) > len(norm_kw) and len(norm_kw) >= 2:
                win = len(norm_kw)
                for start in range(0, max(1, len(norm_text) - win + 1)):
                    chunk = norm_text[start:start + win]
                    local_ratio = difflib.SequenceMatcher(
                        None,
                        norm_kw,
                        chunk,
                    ).ratio()
                    score = 0.92 * local_ratio
                    if score > best_score:
                        best_score = score
                        best_keyword = keyword
                        best_type = "local_fuzzy"

        return best_score, best_keyword, best_type
