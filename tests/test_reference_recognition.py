"""Offline contracts: identity ambiguity, cache lifetime, fallback and gate evidence."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

from core.context import RuntimeContext
from core.skill_base import SkillResult
from services.object_reference_service import ObjectReferenceService, ReferenceSettings
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.perception.reference_recognition import RecognizeObjectSkill
from nesyro_gate.scene_memory_builder import medicine_observation_to_object_state
from nesyro_gate import SceneMemory, SkillName, Step, TaskSpec
from nesyro_gate.gate import nesyro_gate


class ColorEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, images, batch_size=16):
        self.calls += 1
        return np.asarray([np.asarray(im).mean(axis=(0, 1)) + .01 for im in images])


@pytest.fixture
def setup(tmp_path):
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"test encoder")
    settings = ReferenceSettings(gallery_dir=str(tmp_path / "gallery"), checkpoint=str(checkpoint),
        device="cpu", log_dir=str(tmp_path / "logs"), min_similarity=.90, min_class_margin=.10)
    encoder = ColorEncoder()
    service = ObjectReferenceService(settings, encoder=encoder)
    for name, color in [("red", (255, 0, 0)), ("blue", (0, 0, 255))]:
        path = tmp_path / f"{name}.png"; Image.new("RGB", (20, 20), color).save(path)
        service.gallery.register(name=name, image_path=path, category="cube", detector_prompt="cube",
                                 source_group=f"reference_{name}", aliases=[f"{name} cube"])
    scene = tmp_path / "scene.png"
    im = Image.new("RGB", (40, 20), (255, 0, 0)); im.paste((0, 0, 255), (20, 0, 40, 20)); im.save(scene)
    candidates = [dict(object_id="a", bbox=[0, 0, 20, 20], confidence=.95),
                  dict(object_id="b", bbox=[20, 0, 40, 20], confidence=.95)]
    return service, encoder, scene, candidates


def test_reference_choice_uses_identity_not_detector_score(setup):
    service, encoder, image, candidates = setup
    candidates[0]["confidence"] = .1
    result = service.match(image_path=image, candidates=candidates, query="red cube")
    assert result["accepted"] and result["selected_index"] == 0
    assert result["identity"] == "red"
    assert result["score_kind"] == "cosine_similarity_not_probability"
    assert encoder.calls == 2  # gallery batch + candidate batch


def test_unknown_identity_does_not_load_model(setup):
    service, encoder, image, candidates = setup
    assert not service.match(image_path=image, candidates=candidates, query="unknown")["accepted"]
    assert encoder.calls == 0


def test_ambiguous_and_out_of_gallery_appearance_rejected(setup):
    service, _, image, candidates = setup
    Image.new("RGB", (40, 20), (255, 0, 255)).save(image)
    assert not service.match(image_path=image, candidates=candidates, query="red")["accepted"]
    Image.new("RGB", (40, 20), (0, 255, 0)).save(image)
    assert not service.match(image_path=image, candidates=candidates, query="red")["accepted"]


def test_identical_references_with_different_names_rejected_by_margin(setup):
    service, _, image, candidates = setup
    service.gallery.register(name="other red", image_path=image.parent / "red.png", category="cube", detector_prompt="cube")
    result = service.match(image_path=image, candidates=candidates, query="red")
    assert not result["accepted"]


def test_cache_reuses_only_same_image_and_boxes(setup):
    service, encoder, image, candidates = setup
    service.match(image_path=image, candidates=candidates, query="red")
    before = encoder.calls
    cached = service.match(image_path=image, candidates=candidates, query="blue")
    assert encoder.calls == before and cached["frame_features_cache_hit"]
    # Same filename with changed content must not use the previous frame.
    Image.new("RGB", (40, 20), (0, 0, 255)).save(image)
    result = service.match(image_path=image, candidates=candidates, query="red")
    assert encoder.calls == before + 1 and not result["accepted"]


def test_capture_excluded_from_own_reference_evaluation(setup):
    service, _, image, candidates = setup
    result = service.match(image_path=image, candidates=candidates, query="red", exclude_groups=["reference_red"])
    assert not result["accepted"]
    result = service.match(image_path=image.parent / "red.png", candidates=[candidates[0]], query="red")
    assert not result["accepted"]  # exact reference source hash is excluded too


def test_gallery_change_rebuilds_cache_and_duplicate_import_is_idempotent(setup):
    service, encoder, image, candidates = setup
    first = service.match(image_path=image, candidates=candidates, query="red")
    duplicate = service.gallery.register(name="red", image_path=image.parent / "red.png", category="cube", detector_prompt="cube")
    assert duplicate["reference_count"] == 1
    new = image.parent / "green.png"; Image.new("RGB", (20, 20), (0, 255, 0)).save(new)
    service.gallery.register(name="green", image_path=new, category="cube", detector_prompt="cube")
    result = service.match(image_path=image, candidates=candidates, query="red")
    assert first["gallery_revision"] != result["gallery_revision"]
    assert result["gallery_cache"] == "built"


def test_gallery_alias_conflicts_are_rejected_without_modifying_manifest(setup):
    service, _, image, _ = setup
    before = service.gallery.manifest.read_bytes()
    with pytest.raises(ValueError, match="already assigned"):
        service.gallery.register(name="new", image_path=image, category="cube", detector_prompt="cube", aliases=["red cube"])
    assert service.gallery.manifest.read_bytes() == before


def test_registration_merges_case_variants_without_corrupting_catalog(setup):
    service, _, image, _ = setup
    service.gallery.register(name="RED", image_path=image, category="cube", detector_prompt="cube",
                             aliases=["Red", "RED CUBE", "rouge", "ROUGE"])
    obj = service.resolve("red")
    assert obj["name"] == "red" and obj["aliases"] == ["red cube", "rouge"]
    assert len(service.catalog()) == 2


def test_fast_route_only_segments_selected_object_and_skips_ocr(setup):
    service, _, image, candidates = setup
    detector, segmenter, ocr = Mock(), Mock(), Mock()
    detector.detect.return_value = {"detections": candidates}
    segmenter.segment_by_bbox.return_value = {"bbox": candidates[0]["bbox"], "mask_npy_path": "mask.npy"}
    ctx = RuntimeContext(object_references=service, detector=detector, segmenter=segmenter, ocr=ocr)
    result = RecognizeObjectSkill(ctx).run(image_path=str(image), query="red", segment=True)
    assert result.success, result.error
    detector.detect.assert_called_once(); segmenter.segment_by_bbox.assert_called_once()
    ocr.read_image.assert_not_called(); ocr.read_region.assert_not_called()
    assert result.data["ocr_verification"] is None
    assert result.data["recognition_schedule"]["calls"]["sam3"] == 1
    repeat = RecognizeObjectSkill(ctx).run(image_path=str(image), query="blue", segment=False)
    assert repeat.success and detector.detect.call_count == 1


def test_gallery_match_is_used_by_actual_grasp_skill_without_motion(setup):
    service, _, image, candidates = setup
    depth = image.parent / "depth.png"; Image.fromarray(np.full((20, 40), 500, dtype=np.uint16)).save(depth)
    camera, detector, ocr = Mock(), Mock(), Mock()
    detector.detect.return_value = {"detections": candidates}
    camera.capture_wrist.return_value = dict(color_path=str(image), depth_path=str(depth),
        depth_scale=1000, color_intrinsics=dict(fx=100, fy=100, cx=20, cy=10))
    ctx = RuntimeContext(camera=camera, detector=detector, ocr=ocr, object_references=service)
    response = FindAndGraspMedicineSkill(ctx).run(query="red", plan_grasp=False, execute=False)
    assert response.success, response.error
    assert response.data["grasp_input"]["color_path"] == str(image)
    assert response.data["identity_evidence"]["accepted"]
    ocr.read_region.assert_not_called()


def test_uncertain_gallery_falls_back_and_explicit_strict_ocr_is_retained(setup):
    service, _, image, _ = setup
    finder = FindAndGraspMedicineSkill(RuntimeContext(object_references=service))
    old = SkillResult(True, {"selected_instance": {"ocr_text": "red"}})
    with patch.object(finder, "_recognize_legacy", return_value=old) as legacy:
        with patch.object(RecognizeObjectSkill, "run", return_value=SkillResult(False, {"stage": "reference_uncertain"})) as fast:
            result = finder._recognize_image(image_path=str(image), query="red")
            assert result.success and legacy.call_count == 1 and fast.call_count == 1
            finder._recognize_image(image_path=str(image), query="red", ocr_verify_strict_mode=True)
            assert legacy.call_count == 2 and fast.call_count == 1


def reference_observation(setup):
    service, _, image, candidates = setup
    ctx = RuntimeContext(object_references=service, detector=Mock())
    ctx.detector.detect.return_value = {"detections": candidates}
    result = RecognizeObjectSkill(ctx).run(image_path=str(image), query="red")
    assert result.success
    return dict(kind="medicine", result=result.data, selected_instance=result.data["selected_instance"],
                grasp_input={"bbox": candidates[0]["bbox"], "depth_path": "depth.png"})


def decide(observation):
    state = medicine_observation_to_object_state("red", observation)
    scene = SceneMemory(objects={"red": state})
    result = nesyro_gate(Step(SkillName.GRASP, {"object": "red"}), scene,
                         TaskSpec("red", "tray", [SkillName.GRASP]), threshold=.7)
    return state, result


def test_gate_accepts_reference_evidence_without_faking_ocr(setup):
    state, result = decide(reference_observation(setup))
    assert state.ocr_match_score == 0 and state.vlm_match_score == 0
    assert state.identity_verified and state.object_type == "object"
    assert result.decision.value == "EXECUTE"


@pytest.mark.parametrize("change", ["wrong_query", "rejected_match", "missing_depth", "occluded"])
def test_reference_identity_does_not_bypass_other_gate_requirements(setup, change):
    observation = reference_observation(setup)
    identity = observation["result"]["identity_evidence"]
    if change == "wrong_query": identity["query"] = "blue"
    elif change == "rejected_match": identity["accepted"] = False
    elif change == "missing_depth": observation["grasp_input"].pop("depth_path")
    else: observation["vlm_occlusion"] = {"grasp_safe": False, "occlusion_level": .8}
    state, result = decide(observation)
    assert result.decision.value != "EXECUTE"


def test_registered_objects_are_offered_to_planner_without_model_calls(setup):
    from agents.planning_agent import PlanningAgent
    service, encoder, _, _ = setup
    prompt = PlanningAgent(RuntimeContext(object_references=service))._build_llm_prompt("抓取 red", "", [])
    assert '"name": "red"' in prompt and "不证明物体当前在场" in prompt
    assert encoder.calls == 0


def test_new_skills_preserve_existing_registry_entries():
    from skill_lib.perception import PERCEPTION_SKILLS
    from skill_lib.combound import COMPOUND_SKILLS
    names = {c.name for c in [*PERCEPTION_SKILLS, *COMPOUND_SKILLS]}
    assert {"recognize_object", "match_object_reference", "register_object_reference", "find_and_grasp_medicine", "select_object_by_text"} <= names
