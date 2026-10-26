"""Adapters that build SceneMemory from the existing robot skill outputs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from nesyro_gate.types import ObjectState, SceneMemory, SkillName, TaskSpec


def build_scene_memory_from_simple_robot(
    observations: dict[str, dict[str, Any]],
    place_plans: dict[str, dict[str, Any]] | None = None,
    place_counts: dict[str, int] | None = None,
    gripper_holding: str | None = None,
    probe_counts: dict[str, int] | None = None,
    searched_regions: set[str] | None = None,
    geometry_prepared_target: str | None = None,
) -> SceneMemory:
    """Build SceneMemory from SimpleRobotActions runtime caches.

    The adapter is intentionally conservative and schema-tolerant. Medicine
    observations come from FindAndGraspMedicineSkill result data; container
    observations come from PlaceToContainerSkill place plans.
    """

    objects: dict[str, ObjectState] = {}

    for target, observation in (observations or {}).items():
        kind = str(observation.get("kind") or "").lower()
        if kind == "medicine":
            state = medicine_observation_to_object_state(
                target=target,
                observation=observation,
                in_gripper=(gripper_holding == target),
                geometry_prepared=(target == geometry_prepared_target),
            )
            objects[state.object_id] = state
        elif kind == "container":
            state = container_observation_to_object_state(
                target=target,
                observation=observation,
                place_count=(place_counts or {}).get(observation.get("container_key", ""), 0),
            )
            objects[state.object_id] = state

    for key, plan in (place_plans or {}).items():
        if any(_same_container_plan(obj, plan) for obj in objects.values()):
            continue
        state = container_plan_to_object_state(
            object_id=key,
            name=key,
            plan=plan,
            place_count=(place_counts or {}).get(key, 0),
        )
        objects[state.object_id] = state

    if gripper_holding and gripper_holding not in objects:
        # Execution progress survives image invalidation. This node records the
        # executor's holding state, without restoring an old bbox, pose or score.
        objects[gripper_holding] = ObjectState(
            object_id=gripper_holding, name=gripper_holding, object_type='object',
            visible=False, detector_confidence=0., ocr_text=None, ocr_match_score=0.,
            vlm_match_score=0., reachable=False, graspable=False, occlusion_level=0.,
            in_gripper=True, free_space=False, placeable=False, identity_source='execution_state')

    return SceneMemory(
        objects=objects,
        probe_counts=dict(probe_counts or {}),
        searched_regions=set(searched_regions or set()),
        gripper_holding=gripper_holding,
    )


def medicine_observation_to_object_state(
    target: str,
    observation: dict[str, Any],
    in_gripper: bool = False,
    geometry_prepared: bool = False,
) -> ObjectState:
    """Convert one cached medicine observation into an ObjectState."""

    if observation.get("observation_failed"):
        return failed_medicine_observation_to_object_state(
            target=target,
            observation=observation,
            in_gripper=in_gripper,
        )

    result = _as_dict(observation.get("result"))
    selected = _as_dict(observation.get("selected_instance") or result.get("selected_instance"))
    grasp_input = _as_dict(observation.get("grasp_input") or result.get("grasp_input"))
    selection = _as_dict(result.get("selection_result"))
    ocr_verification = _as_dict(result.get("ocr_verification"))
    grasp_plan = _as_dict(result.get("grasp_plan"))
    identity = _as_dict(result.get("identity_evidence") or selected.get("identity_evidence"))
    reference_verified = bool(identity.get("source") == "reference_gallery"
                              and identity.get("accepted") is True
                              and identity.get("query") == target
                              and identity.get("gallery_revision") and identity.get("image_sha256")
                              and _first_float(identity.get("score"), -1) >= _first_float(identity.get("min_similarity"), 2)
                              and _first_float(identity.get("class_margin"), -1) >= _first_float(identity.get("min_class_margin"), 2))
    vlm_occlusion = _as_dict(observation.get("vlm_occlusion") or result.get("vlm_occlusion"))

    selected_id = str(
        selected.get("object_id")
        or grasp_input.get("object_id")
        or target
    )
    object_id = str(target or selected_id)

    best_candidate = _find_ranked_candidate(selection, selected_id)
    detector_confidence = _first_float(
        selected.get("confidence"),
        selected.get("detection_confidence"),
        _as_dict(selected.get("detection")).get("confidence"),
        best_candidate.get("detection_confidence"),
        best_candidate.get("confidence"),
        0.0,
    )
    raw_text_score = _first_float(
        best_candidate.get("raw_text_score"),
        selection.get("raw_text_score"),
        None,
    )
    ocr_match_score = _first_float(
        ocr_verification.get("confidence"),
        raw_text_score,
        selection.get("score"),
        0.0,
    )
    vlm_match_score = _first_float(
        selection.get("score"),
        best_candidate.get("score"),
        detector_confidence,
        0.0,
    )
    if identity.get("source") == "reference_gallery":
        # Do not disguise a visual similarity score as OCR or VLM evidence.
        ocr_match_score = _first_float(ocr_verification.get("confidence"), 0.0)
        vlm_match_score = 0.0

    bbox = selected.get("bbox") or grasp_input.get("bbox")
    has_depth = bool(grasp_input.get("depth_path") or grasp_input.get("depth_npy_path"))
    has_grasp_pose = bool(
        grasp_plan.get("best_grasp_xarm_mmrad")
        or grasp_plan.get("grasp_pose")
        or grasp_plan.get("chosen_grasp")
    )

    visible = bool(bbox or selected)
    reachable = bool(visible and (has_depth or has_grasp_pose))
    occlusion_level = _estimate_occlusion_level(
        selected, grasp_input, vlm_occlusion, use_heuristics=not geometry_prepared)
    graspable = bool(visible and (bbox or has_grasp_pose))
    if vlm_occlusion and not bool(vlm_occlusion.get("grasp_safe", True)):
        graspable = False
    if occlusion_level > 0.35:
        graspable = False

    return ObjectState(
        object_id=object_id,
        name=str(identity.get('identity') or _as_dict(result.get('target_identity')).get('name')
                 or target or selected.get("name") or grasp_input.get("query") or object_id),
        object_type="object" if reference_verified and identity.get("category") != "medicine_box" else "medicine_box",
        visible=visible,
        detector_confidence=clamp01(detector_confidence),
        ocr_text=_first_text(
            selected.get("ocr_text"),
            grasp_input.get("ocr_text"),
            _as_dict(ocr_verification.get("evidence")).get("raw_ocr_text"),
        ),
        ocr_match_score=clamp01(ocr_match_score),
        vlm_match_score=clamp01(vlm_match_score),
        reachable=reachable,
        graspable=graspable,
        occlusion_level=occlusion_level,
        in_gripper=in_gripper,
        free_space=True,
        placeable=False,
        identity_source="reference_gallery" if reference_verified else None,
        identity_match_score=clamp01(identity.get("score", 0)) if reference_verified else 0.0,
        identity_verified=reference_verified,
    )


def failed_medicine_observation_to_object_state(
    target: str,
    observation: dict[str, Any],
    in_gripper: bool = False,
) -> ObjectState:
    """Convert a failed medicine observe attempt into a conservative state."""

    score = _first_float(
        observation.get("best_score"),
        _extract_best_score_from_text(observation.get("error")),
        0.0,
    )
    return ObjectState(
        object_id=str(target),
        name=str(target),
        object_type="medicine_box",
        visible=False,
        detector_confidence=clamp01(score),
        ocr_text=None,
        ocr_match_score=clamp01(score),
        vlm_match_score=clamp01(score),
        reachable=False,
        graspable=False,
        occlusion_level=1.0,
        in_gripper=in_gripper,
        free_space=True,
        placeable=False,
    )


def container_observation_to_object_state(
    target: str,
    observation: dict[str, Any],
    place_count: int = 0,
) -> ObjectState:
    """Convert one cached container observation into an ObjectState."""

    plan = _as_dict(observation.get("place_plan"))
    return container_plan_to_object_state(
        object_id=str(target),
        name=str(target),
        plan=plan,
        place_count=place_count,
    )


def container_plan_to_object_state(
    object_id: str,
    name: str,
    plan: dict[str, Any],
    place_count: int = 0,
) -> ObjectState:
    """Convert a PlaceToContainerSkill plan into an ObjectState."""

    selected = _as_dict(plan.get("selected_detection"))
    detector_confidence = _first_float(selected.get("confidence"), 0.0)
    color_score = _first_float(selected.get("color_score"), None)
    has_pose = bool(plan.get("place_pose_mmrad") or plan.get("center_base_xyz"))
    object_type = _infer_container_type(name, selected)
    free_space = bool(has_pose and place_count < 3)

    semantic_score = color_score if color_score is not None else detector_confidence
    if object_type in {"plate", "container"}:
        semantic_score = max(float(semantic_score or 0.0), 0.6)
    binding = _as_dict(plan.get('task_container'))
    fixed_for_task = bool(binding.get('fixed_for_task') is True and binding.get('task_id')
                          and binding.get('coordinate_frame') == 'base' and has_pose)

    return ObjectState(
        object_id=str(object_id),
        name=str(name),
        object_type=object_type,
        visible=bool(selected or has_pose),
        detector_confidence=clamp01(detector_confidence),
        ocr_text=None,
        ocr_match_score=clamp01(semantic_score),
        vlm_match_score=clamp01(detector_confidence or semantic_score),
        reachable=has_pose,
        graspable=False,
        occlusion_level=0.0,
        in_gripper=False,
        free_space=free_space,
        placeable=has_pose,
        fixed_for_task=fixed_for_task,
    )


def build_task_spec_for_pick_place(
    target_object_name: str,
    target_plate_name: str,
    target_object_id: str | None = None,
    target_plate_id: str | None = None,
) -> TaskSpec:
    """Create a TaskSpec for the current observe-grasp-place skill set."""

    return TaskSpec(
        target_object_name=target_object_name,
        target_plate_name=target_plate_name,
        allowed_skills=[SkillName.OBSERVE, SkillName.GRASP, SkillName.PLACE],
        target_object_id=target_object_id or target_object_name,
        target_plate_id=target_plate_id or target_plate_name,
    )


def clamp01(value: float | int | None) -> float:
    """Clamp a possibly missing numeric value into [0, 1]."""

    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_best_score_from_text(text: Any) -> float | None:
    if text is None:
        return None
    matches = re.findall(r"最高分\s*=\s*([0-9]+(?:\.[0-9]+)?)", str(text))
    if not matches:
        return None
    return max(float(value) for value in matches)


def _find_ranked_candidate(
    selection_result: dict[str, Any],
    selected_id: str,
) -> dict[str, Any]:
    for candidate in selection_result.get("ranked_candidates", []) or []:
        if str(candidate.get("object_id") or "") == selected_id:
            return candidate
    candidates = selection_result.get("ranked_candidates", []) or []
    return candidates[0] if candidates and isinstance(candidates[0], dict) else {}


def _estimate_occlusion_level(
    selected: dict[str, Any],
    grasp_input: dict[str, Any] | None = None,
    vlm_occlusion: dict[str, Any] | None = None,
    *, use_heuristics: bool = True,
) -> float:
    vlm_occlusion = _as_dict(vlm_occlusion)
    if vlm_occlusion:
        vlm_level = _first_float(vlm_occlusion.get("occlusion_level"), None)
        if vlm_level is not None:
            if vlm_occlusion.get("target_occluded") or not vlm_occlusion.get("grasp_safe", True):
                return clamp01(max(vlm_level, 0.36))
            return clamp01(vlm_level)

    if selected.get("occlusion_level") is not None:
        return clamp01(selected.get("occlusion_level"))
    if not selected.get("bbox"):
        return 1.0
    if not use_heuristics:
        # The current grasp preflight has already validated this target's
        # same-frame relation graph. Bbox depth spread / mask occupancy are
        # not independent evidence of occlusion and must not veto that result.
        # Explicit selected/VLM occlusion verdicts above remain in effect.
        return 0.0

    grasp_input = _as_dict(grasp_input)
    evidence_scores: list[float] = []

    mask_score = _first_float(selected.get("mask_score"), None)
    if mask_score is not None:
        evidence_scores.append(clamp01(1.0 - mask_score))

    bbox_area = _bbox_area(selected.get("bbox"))
    mask_area = _first_float(selected.get("mask_area"), None)
    if bbox_area > 0 and mask_area is not None:
        visible_ratio = clamp01(mask_area / bbox_area)
        evidence_scores.append(clamp01(1.0 - visible_ratio))

    depth_score = _estimate_depth_occlusion_level(
        bbox=selected.get("bbox") or grasp_input.get("bbox"),
        depth_npy_path=grasp_input.get("depth_npy_path"),
        depth_path=grasp_input.get("depth_path"),
        depth_scale=_first_float(grasp_input.get("depth_scale"), 1000.0) or 1000.0,
    )
    if depth_score is not None:
        evidence_scores.append(depth_score)

    if evidence_scores:
        return clamp01(max(evidence_scores))

    return 0.1


def _estimate_depth_occlusion_level(
    bbox: Any,
    depth_npy_path: Any,
    depth_path: Any,
    depth_scale: float = 1000.0,
    closer_margin_m: float = 0.025,
    min_valid_pixels: int = 80,
) -> float | None:
    """Estimate target occlusion from closer foreground depth inside a bbox.

    In wrist RGB-D observations, an object lying on top of a medicine box is
    usually closer to the camera than the box surface. This function treats a
    substantial cluster of closer pixels inside the target bbox as occlusion.
    """

    depth = _load_depth_meters(
        depth_npy_path=depth_npy_path,
        depth_path=depth_path,
        depth_scale=depth_scale,
    )
    if depth is None or not bbox or len(bbox) != 4:
        return None

    try:
        import numpy as np
    except Exception:
        return None

    h, w = depth.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, image_width=w, image_height=h)
    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth[y1:y2, x1:x2]
    valid_mask = np.isfinite(patch) & (patch > 0.02) & (patch < 2.0)
    valid = patch[valid_mask]
    if valid.size < min_valid_pixels:
        return None

    surface_depth = float(np.median(valid))
    closer_mask = valid_mask & (patch < surface_depth - float(closer_margin_m))
    closer_ratio = float(np.count_nonzero(closer_mask) / np.count_nonzero(valid_mask))

    ph, pw = patch.shape[:2]
    cy1 = max(0, int(round(ph * 0.25)))
    cy2 = min(ph, int(round(ph * 0.75)))
    cx1 = max(0, int(round(pw * 0.25)))
    cx2 = min(pw, int(round(pw * 0.75)))
    central_valid = valid_mask[cy1:cy2, cx1:cx2]
    central_closer = closer_mask[cy1:cy2, cx1:cx2]
    central_ratio = 0.0
    if np.count_nonzero(central_valid) > 0:
        central_ratio = float(np.count_nonzero(central_closer) / np.count_nonzero(central_valid))

    depth_spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))

    ratio_score = closer_ratio / 0.22
    central_score = central_ratio / 0.16
    spread_score = max(0.0, (depth_spread - 0.04) / 0.10)

    return clamp01(max(ratio_score, central_score, spread_score))


def _load_depth_meters(
    depth_npy_path: Any,
    depth_path: Any,
    depth_scale: float,
):
    try:
        import numpy as np
    except Exception:
        return None

    path = _first_existing_path(depth_npy_path)
    if path is not None:
        try:
            depth = np.load(str(path))
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            depth = depth.astype(float)
            if np.nanmedian(depth) > 10.0:
                depth = depth / float(depth_scale or 1000.0)
            return depth
        except Exception:
            return None

    path = _first_existing_path(depth_path)
    if path is None:
        return None

    try:
        import cv2
    except Exception:
        return None

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(float)
    if np.nanmedian(depth) > 10.0:
        depth = depth / float(depth_scale or 1000.0)
    return depth


def _first_existing_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _bbox_area(bbox: Any) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _clip_bbox(bbox: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(image_width, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))
    return x1, y1, x2, y2


def _infer_container_type(name: str, selected: dict[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in [
            name,
            selected.get("phrase"),
            selected.get("source_prompt"),
            selected.get("target_name"),
        ]
    ).lower()
    if any(token in text for token in ["plate", "tray", "盘", "托盘", "盘子", "dish"]):
        return "plate"
    if any(token in text for token in ["bowl", "container", "box", "碗", "容器", "盒"]):
        return "container"
    return "container"


def _same_container_plan(obj: ObjectState, plan: dict[str, Any]) -> bool:
    if obj.object_type not in {"plate", "container"}:
        return False
    return bool(plan.get("center_base_xyz") and obj.placeable)
