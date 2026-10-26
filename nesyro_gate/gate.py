"""Core rule-based NeSyRo Gate decision logic."""

from __future__ import annotations

from nesyro_gate.rules import (
    compute_csc,
    compute_lc,
    convert_failure_to_constraint,
    is_probe_solvable,
    suggest_probe,
    verify_step,
    semantic_identity_score,
    has_fixed_container_plan,
)
from nesyro_gate.types import GateDecision, GateResult, SceneMemory, SkillName, Step, TaskSpec


def _step_key(step: Step) -> str:
    """Build a stable key for probe accounting."""

    if step.step_id:
        return step.step_id

    args = ",".join(f"{key}={step.args[key]}" for key in sorted(step.args))
    return f"{step.name.value}({args})"


def _append_confidence_failures(
    step: Step,
    scene_memory: SceneMemory,
    threshold: float,
    failed_conditions: list[str],
) -> list[str]:
    """Add semantic low-confidence failures used to select observation probes."""

    merged = list(failed_conditions)

    def add(condition: str) -> None:
        if condition not in merged:
            merged.append(condition)

    if step.name == SkillName.GRASP:
        obj = scene_memory.get_object(step.args["object"])
        if obj is None:
            return merged
        semantic_score = semantic_identity_score(obj)
        has_reference_identity = obj.identity_source == "reference_gallery" and obj.identity_verified
        if not has_reference_identity and obj.ocr_match_score < threshold:
            add("ocr_low_confidence")
        if semantic_score < threshold:
            add("target_low_confidence")

    elif step.name == SkillName.PLACE:
        plate = scene_memory.get_object(step.args["plate"])
        if plate is None:
            return merged
        if has_fixed_container_plan(plate):
            return merged
        csc = (
            0.5 * plate.ocr_match_score
            + 0.3 * plate.vlm_match_score
            + 0.2 * plate.detector_confidence
        )
        if plate.ocr_match_score < threshold:
            add("ocr_low_confidence")
        if csc < threshold:
            add("plate_low_confidence")

    return merged


def _update_scene_memory_from_failures(
    step: Step,
    scene_memory: SceneMemory,
    failed_conditions: list[str],
) -> None:
    """Write conservative gate conclusions back into SceneMemory."""

    object_id = step.args.get("object")
    plate_id = step.args.get("plate")

    obj = scene_memory.get_object(object_id) if object_id else None
    plate = scene_memory.get_object(plate_id) if plate_id else None

    if obj is not None:
        if "target_not_visible" in failed_conditions:
            obj.visible = False
        if "target_not_reachable" in failed_conditions:
            obj.reachable = False
        if "target_not_graspable" in failed_conditions:
            obj.graspable = False
        if "target_occluded" in failed_conditions:
            obj.occlusion_level = max(float(obj.occlusion_level), 0.36)
            obj.graspable = False
        if "object_not_in_gripper" in failed_conditions:
            obj.in_gripper = False
            if scene_memory.gripper_holding == object_id:
                scene_memory.gripper_holding = None

    if plate is not None:
        if "plate_not_visible" in failed_conditions:
            plate.visible = False
        if "plate_no_free_space" in failed_conditions:
            plate.free_space = False
        if "plate_not_placeable" in failed_conditions:
            plate.placeable = False


def nesyro_gate(
    step: Step,
    scene_memory: SceneMemory,
    task_spec: TaskSpec,
    threshold: float = 0.7,
    max_probe: int = 3,
) -> GateResult:
    """Decide whether to execute, probe, repair, or replan before a step."""

    step_key = _step_key(step)
    verified, verify_reason, verification_failures = verify_step(
        step=step,
        task_spec=task_spec,
        scene_memory=scene_memory,
    )

    if not verified:
        _update_scene_memory_from_failures(step, scene_memory, verification_failures)
        return GateResult(
            decision=GateDecision.LOCAL_REPAIR,
            confidence=0.0,
            csc=0.0,
            lc=0.0,
            reason=verify_reason,
            failed_conditions=verification_failures,
            suggested_constraint=convert_failure_to_constraint(verification_failures, step),
        )

    if step.name == SkillName.OBSERVE and scene_memory.get_probe_count(step_key) >= max_probe:
        failed_conditions = ["repeated_observation_no_new_information"]
        return GateResult(
            decision=GateDecision.REPLAN,
            confidence=0.0,
            csc=1.0,
            lc=0.0,
            reason="repeated observation produced no new information",
            failed_conditions=failed_conditions,
            suggested_constraint=convert_failure_to_constraint(failed_conditions, step),
        )

    csc = compute_csc(step, scene_memory, task_spec)
    lc, lc_failures = compute_lc(step, scene_memory)
    confidence = csc * lc

    if confidence >= threshold:
        return GateResult(
            decision=GateDecision.EXECUTE,
            confidence=confidence,
            csc=csc,
            lc=lc,
            reason="NeSyConf passed threshold",
            failed_conditions=[],
        )

    failed_conditions = _append_confidence_failures(
        step=step,
        scene_memory=scene_memory,
        threshold=threshold,
        failed_conditions=lc_failures,
    )
    _update_scene_memory_from_failures(step, scene_memory, failed_conditions)

    if is_probe_solvable(failed_conditions):
        probe_count = scene_memory.get_probe_count(step_key)
        if probe_count < max_probe:
            suggested = suggest_probe(step, failed_conditions, scene_memory)
            scene_memory.increment_probe_count(step_key)
            return GateResult(
                decision=GateDecision.PROBE_REQUIRED,
                confidence=confidence,
                csc=csc,
                lc=lc,
                reason="low NeSyConf; request safe observation probe",
                failed_conditions=failed_conditions,
                suggested_probe=suggested,
            )

    return GateResult(
        decision=GateDecision.REPLAN,
        confidence=confidence,
        csc=csc,
        lc=lc,
        reason="low NeSyConf cannot be solved by another probe",
        failed_conditions=failed_conditions,
        suggested_constraint=convert_failure_to_constraint(failed_conditions, step),
    )
