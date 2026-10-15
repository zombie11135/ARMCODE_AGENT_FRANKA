"""Small reusable visual skills and deterministic scheduling over one saved frame."""
from __future__ import annotations

import copy
from pathlib import Path
import time
import uuid

from core.skill_base import BaseSkill, SkillResult
from services.object_reference_service import atomic_json, image_digest


class RegisterObjectReferenceSkill(BaseSkill):
    name = "register_object_reference"
    description = "使用用户明确标注的保存图片登记物体；不自动把模型预测加入图库。"
    input_schema = {"name": "物体名称", "image_path": "保存图片", "bbox": "可选 XYXY 像素框",
                    "category": "物体类别", "detector_prompt": "英文候选检测描述", "aliases": "可选别名"}
    output_schema = {"reference_count": "该物体的参考图数量"}

    def run(self, **kwargs):
        service = self.context.require("object_references")
        return SkillResult(success=True, data=service.gallery.register(**kwargs))


class MatchObjectReferenceSkill(BaseSkill):
    name = "match_object_reference"
    description = "批量编码候选区域，与本地图片特征库匹配；返回独立的相似度与类别分差。"
    input_schema = {"image_path": "保存图片", "query": "登记名称/别名", "candidates": "含 bbox 的候选列表"}
    output_schema = {"accepted": "是否满足相似度及分差", "ranked_candidates": "候选匹配证据"}

    def run(self, **kwargs):
        result = self.context.require("object_references").match(**kwargs)
        return SkillResult(success=result["accepted"], data=result,
                           error=None if result["accepted"] else result["reason"])


class RecognizeObjectSkill(BaseSkill):
    name = "recognize_object"
    description = "参考图片识别快速分支：一次检测、批量匹配、仅分割选中目标。歧义返回给原 OCR 流程处理。"
    input_schema = {"image_path": "同一帧保存 RGB", "query": "登记名称/别名", "segment": "是否需要 SAM3 掩码"}
    output_schema = {"selected_instance": "选中的当前帧实例", "identity_evidence": "身份匹配依据",
                     "recognition_schedule": "调用计数、耗时与缓存命中"}

    def run(self, *, image_path, query, segment=False, detection_threshold=0.22,
            text_threshold=0.18, max_instances=12, expand_bbox_px=12,
            candidates=None, exclude_groups=(), **kwargs):
        started = time.perf_counter()
        service = getattr(self.context, "object_references", None)
        if service is None:
            return SkillResult(False, {"stage": "reference_disabled"}, error="Reference matching disabled")
        target = service.resolve(query)
        if not target or not target.get("references"):
            return SkillResult(False, {"stage": "unregistered_target"}, error="Target is not registered")
        run_dir = Path(service.settings.log_dir) / uuid.uuid4().hex
        report = {"query": query, "image_path": str(Path(image_path).resolve()),
                  "hardware_accessed": False, "calls": {"detector": 0, "reference_match": 0, "sam3": 0, "ocr": 0, "vlm": 0},
                  "timing_s": {}, "source": "reference_gallery"}
        try:
            digest = image_digest(image_path)
            key = (digest, target["detector_prompt"], detection_threshold, text_threshold, max_instances)
            detector_started = time.perf_counter()
            if candidates is None:
                # Cache only the same image content and detector parameters, never a pose/trajectory.
                cache = getattr(service, "_detection_cache", None)
                if cache is None:
                    from collections import OrderedDict
                    cache = service._detection_cache = OrderedDict()
                report["detection_cache_hit"] = key in cache
                if key not in cache:
                    result = self.context.require("detector").detect(image_path=image_path,
                        target_name=target["detector_prompt"], threshold=detection_threshold,
                        text_threshold=text_threshold, max_objects=max_instances, enable_filter=True)
                    cache[key] = copy.deepcopy(result.get("detections", []))
                    report["calls"]["detector"] += 1
                    while len(cache) > service.settings.cache_frames:
                        cache.popitem(last=False)
                candidates = copy.deepcopy(cache[key])
                cache.move_to_end(key)
            candidates = copy.deepcopy(candidates[:max_instances])
            for i, det in enumerate(candidates):
                det.setdefault("object_id", f"obj_{i+1}")
                det["image_path"] = image_path
            report["timing_s"]["detect"] = time.perf_counter() - detector_started
            report["candidate_count"] = len(candidates)
            matching = MatchObjectReferenceSkill(self.context).run(image_path=image_path,
                candidates=candidates, query=query, exclude_groups=exclude_groups)
            report["calls"]["reference_match"] += 1
            report["match"] = matching.data
            report["timing_s"]["match"] = matching.data.get("elapsed_s", 0)
            if not matching.success:
                return SkillResult(False, {"stage": "reference_uncertain", "reference_attempt": report},
                                   error="参考图片匹配不明确，保留原识别流程")
            match = matching.data
            selected = copy.deepcopy(candidates[match["selected_index"]])
            selected.update(success=True, image_path=image_path, segmenter="bbox_only",
                            target_name=target["category"], mask_path=None, mask_npy_path=None)
            if segment:
                from skill_lib.perception.seg_obj import SegmentObjectSkill
                segment_started = time.perf_counter()
                report["calls"]["sam3"] += 1
                result = SegmentObjectSkill(self.context).run(image_path=image_path,
                    bbox=selected["bbox"], target_name=target["category"], segment_all=False,
                    output_prefix=str(run_dir / "selected"), expand_bbox_px=expand_bbox_px, select_by="iou")
                report["timing_s"]["segment"] = time.perf_counter() - segment_started
                valid = [inst for inst in result.data.get("instances", []) if inst.get("success")]
                if not result.success or not valid:
                    return SkillResult(False, {"stage": "reference_segment", "reference_attempt": report},
                                       error=f"选中目标分割失败: {result.error}")
                selected.update(valid[0])
                selected.update(object_id=candidates[match["selected_index"]]["object_id"],
                                confidence=candidates[match["selected_index"]].get("confidence"))
            evidence = {k: v for k, v in match.items() if k != "ranked_candidates"}
            selected["identity_evidence"] = evidence
            report["accepted"] = True
            report["selected_bbox"] = selected["bbox"]
            selection = dict(selected_instance=selected, selected_index=match["selected_index"],
                             score=match["score"], matched_keyword=match["identity"],
                             match_type="reference_gallery", ranked_candidates=match["ranked_candidates"],
                             identity_evidence=evidence)
            self.context.emit_text(f"[recognition] {match['identity']}: 参考图匹配 {match['score']:.3f}, "
                                   f"类别分差 {match['class_margin']:.3f}；跳过 OCR")
            return SkillResult(True, dict(image_path=image_path, prompts=[target["detector_prompt"]],
                detections=candidates, instances=[selected], selected_instance=selected,
                selection_result=selection, identity_evidence=evidence,
                ocr_verification=None, ranked_candidates=match["ranked_candidates"],
                recognition_schedule=report))
        except Exception as error:
            # These skills have no motion clients. Model/file errors take the original OCR fallback.
            report["error"] = f"{type(error).__name__}: {error}"
            self.context.emit_warning(f"[recognition] 参考图分支不可用，回退原流程：{report['error']}")
            return SkillResult(False, {"stage": "reference_error", "reference_attempt": report}, error=report["error"])
        finally:
            report["timing_s"]["total"] = time.perf_counter() - started
            report["report_path"] = str(run_dir / "recognition.json")
            atomic_json(run_dir / "recognition.json", report)
