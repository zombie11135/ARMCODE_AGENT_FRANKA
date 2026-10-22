"""Small reusable visual skills and deterministic scheduling over one saved frame."""
from __future__ import annotations

import copy
from pathlib import Path
import time
import uuid

from core.skill_base import BaseSkill, SkillResult
from services.object_reference_service import atomic_json, image_digest, crop_image
from services.detection_prompts import single_detection_prompt


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
    description = "一次检测、批量图库匹配、仅按需分割选中目标；歧义明确返回失败，由调用方决定恢复策略。"
    input_schema = {"image_path": "同一帧保存 RGB", "query": "登记名称/别名", "segment": "是否需要 SAM3 掩码"}
    output_schema = {"selected_instance": "选中的当前帧实例", "identity_evidence": "身份匹配依据",
                     "recognition_schedule": "调用计数、耗时与缓存命中"}

    def run(self, *, image_path, query, segment=False, detection_threshold=0.22,
            text_threshold=0.18, max_instances=12, expand_bbox_px=12,
            candidates=None, exclude_groups=(), target_prompts=None, **kwargs):
        started = time.perf_counter()
        service = getattr(self.context, "object_references", None)
        if service is None:
            return SkillResult(False, {"stage": "reference_disabled"}, error="Reference matching disabled")
        target = service.resolve(query)
        if not target or not target.get("references"):
            return SkillResult(False, {"stage": "unregistered_target"}, error="Target is not registered")
        prompt = (service.settings.medicine_detector_prompt if target["category"] == "medicine_box"
                  else target["detector_prompt"])
        if target_prompts:
            # Keep custom prompt-list inputs usable, but send all descriptions in
            # one request. Default medicine_box keeps the detector's geometry rules.
            supplied = list(dict.fromkeys(str(p).strip() for p in target_prompts if str(p).strip()))
            prompt = supplied[0] if len(supplied) == 1 else single_detection_prompt(supplied)
        run_dir = Path(service.settings.log_dir) / uuid.uuid4().hex
        report = {"query": query, "image_path": str(Path(image_path).resolve()),
                  "hardware_accessed": False, "calls": {"detector": 0, "reference_match": 0, "sam3": 0, "ocr": 0, "vlm": 0},
                  "timing_s": {}, "source": "reference_gallery",
                  "encoder_model": service.settings.encoder_model, "detector_prompt": prompt,
                  "recognition_policy": service.settings.recognition_policy}
        data = dict(image_path=image_path, prompts=[prompt], detections=[], instances=[],
                    selected_instance=None, selection_result={}, ocr_verification=None,
                    ranked_candidates=[], identity_evidence=None, recognition_schedule=report)
        try:
            digest = image_digest(image_path)
            key = (digest, id(self.context.detector), prompt, detection_threshold, text_threshold, max_instances)
            detector_started = time.perf_counter()
            if candidates is None:
                # Cache only the same image content and detector parameters, never a pose/trajectory.
                cache = getattr(service, "_detection_cache", None)
                if cache is None:
                    from collections import OrderedDict
                    cache = service._detection_cache = OrderedDict()
                report["detection_cache_hit"] = key in cache
                if key not in cache:
                    report["calls"]["detector"] += 1
                    result = self.context.require("detector").detect(image_path=image_path,
                        target_name=prompt, threshold=detection_threshold,
                        text_threshold=text_threshold, max_objects=max_instances, enable_filter=True)
                    cache[key] = copy.deepcopy(result.get("detections", []))
                    while len(cache) > service.settings.cache_frames:
                        cache.popitem(last=False)
                candidates = copy.deepcopy(cache[key])
                cache.move_to_end(key)
            candidates = copy.deepcopy(candidates[:max_instances])
            for i, det in enumerate(candidates):
                det.setdefault("object_id", f"obj_{i+1}")
                det["image_path"] = image_path
            data["detections"] = candidates
            instances = []
            for det in candidates:
                inst = dict(det, detection=copy.deepcopy(det), success=True,
                            target_name=target["category"], segmenter="bbox_only",
                            mask_path=None, mask_npy_path=None, mask_area=None,
                            crop_path=None, overlay_path=None, source_prompt=prompt, prompt_hits=[prompt],
                            ocr_success=False, ocr_text="", ocr_texts=[], ocr_confidences=[],
                            ocr_avg_confidence=0.0, ocr_items=[], ocr_vis_paths=[])
                instances.append(inst)
            data["instances"] = instances
            report["timing_s"]["detect"] = time.perf_counter() - detector_started
            report["candidate_count"] = len(candidates)
            report["calls"]["reference_match"] += 1
            matching = MatchObjectReferenceSkill(self.context).run(image_path=image_path,
                candidates=candidates, query=query, exclude_groups=exclude_groups)
            report["match"] = matching.data
            report["timing_s"]["match"] = matching.data.get("elapsed_s", 0)
            data["ranked_candidates"] = matching.data.get("ranked_candidates", [])
            for row in data["ranked_candidates"]:
                row.update(instance=instances[row["index"]], crop_path=None, ocr_text="",
                           ocr_avg_confidence=0.0, raw_text_score=0.0, match_type="reference_gallery",
                           matched_keyword=matching.data.get("identity") if row["accepted"] else None)
            evidence = {k: v for k, v in matching.data.items() if k != "ranked_candidates"}
            data["identity_evidence"] = evidence
            if not matching.success:
                report["accepted"] = False
                data.update(stage="reference_uncertain", reference_attempt=report,
                            failure_type="reference_uncertain")
                return SkillResult(False, data, error="参考图片匹配未通过相似度或类别分差门槛，未确认目标")
            match = matching.data
            selected = instances[match["selected_index"]]
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
                    data.update(stage="reference_segment", reference_attempt=report)
                    return SkillResult(False, data,
                                       error=f"选中目标分割失败: {result.error}")
                selected.update(valid[0])
                selected.update(object_id=candidates[match["selected_index"]]["object_id"],
                                confidence=candidates[match["selected_index"]].get("confidence"))
            if not selected.get("crop_path"):
                from PIL import Image
                run_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(image_path) as image:
                    crop_image(image, selected["bbox"]).save(run_dir / "selected_crop.png")
                selected["crop_path"] = str(run_dir / "selected_crop.png")
            selected["identity_evidence"] = evidence
            for row in data["ranked_candidates"]:
                row["crop_path"] = row["instance"].get("crop_path")
            report["accepted"] = True
            report["selected_bbox"] = selected["bbox"]
            selection = dict(query=query, keywords=[], selected_instance=selected, selected_index=match["selected_index"],
                             score=match["score"], matched_keyword=match["identity"],
                             match_type="reference_gallery", ranked_candidates=match["ranked_candidates"],
                             identity_evidence=evidence)
            self.context.emit_text(f"[recognition] {match['identity']}: 参考图匹配 {match['score']:.3f}, "
                                   f"类别分差 {match['class_margin']:.3f}；跳过 OCR")
            data.update(selected_instance=selected, selection_result=selection)
            return SkillResult(True, data)
        except Exception as error:
            # No motion clients. The configured caller policy decides any legacy fallback.
            report["error"] = f"{type(error).__name__}: {error}"
            report["accepted"] = False
            self.context.emit_warning(f"[recognition] 参考图识别不可用：{report['error']}")
            data.update(stage="reference_error", failure_type="reference_error", reference_attempt=report,
                        selected_instance=None)
            return SkillResult(False, data, error=report["error"])
        finally:
            report["timing_s"]["total"] = time.perf_counter() - started
            report["report_path"] = str(run_dir / "recognition.json")
            try:
                atomic_json(run_dir / "recognition.json", report)
            except OSError as error:
                report["log_error"] = str(error)
                self.context.emit_warning(f"[recognition] 无法保存调用记录：{error}")
