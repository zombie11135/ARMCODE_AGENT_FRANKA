"""Rule functions for the rule-based NeSyRo Gate prototype."""

from __future__ import annotations

from nesyro_gate.types import SceneMemory, SkillName, Step, TaskSpec

GRASP_OCCLUSION_THRESHOLD = 0.35


def has_fixed_container_plan(obj):
    return bool(obj and obj.fixed_for_task and obj.placeable
                and obj.object_type in {'plate', 'container'})


def semantic_identity_score(obj):
    """Reference evidence is separate from OCR; legacy observations keep old weights."""
    if obj.identity_source == "reference_gallery" and obj.identity_verified:
        return clamp01(0.8 * obj.identity_match_score + 0.2 * obj.detector_confidence)
    return clamp01(0.5 * obj.ocr_match_score + 0.3 * obj.vlm_match_score + 0.2 * obj.detector_confidence)


def clamp01(value: float) -> float:
    """Clamp a numeric confidence value into the inclusive [0, 1] range."""

    return max(0.0, min(1.0, float(value)))


def verify_step(
    step: Step,
    task_spec: TaskSpec,
    scene_memory: SceneMemory,
) -> tuple[bool, str, list[str]]:
    """Verify that a step is structurally valid for the task and scene."""

    failed_conditions: list[str] = []

    if step.name not in task_spec.allowed_skills:
        failed_conditions.append("invalid_skill")

    if step.name == SkillName.OBSERVE:
        if "region" not in step.args and "query" not in step.args:
            failed_conditions.append("missing_argument")

    elif step.name == SkillName.GRASP:
        object_id = step.args.get("object")
        if object_id is None:
            failed_conditions.append("missing_argument")
        else:
            obj = scene_memory.get_object(object_id)
            if obj is None:
                failed_conditions.append("object_not_found")
            elif task_spec.target_object_id is not None:
                if object_id != task_spec.target_object_id:
                    failed_conditions.append("wrong_target_object")
            elif obj.object_type not in {"medicine_box", "object"}:
                failed_conditions.append("wrong_target_object")

    elif step.name == SkillName.PLACE:
        object_id = step.args.get("object")
        plate_id = step.args.get("plate")
        if object_id is None or plate_id is None:
            failed_conditions.append("missing_argument")
        else:
            if scene_memory.get_object(object_id) is None:
                failed_conditions.append("object_not_found")
            if scene_memory.get_object(plate_id) is None:
                failed_conditions.append("plate_not_found")
            if (
                task_spec.target_plate_id is not None
                and plate_id != task_spec.target_plate_id
            ):
                failed_conditions.append("wrong_target_plate")

    else:
        failed_conditions.append("invalid_skill")

    if failed_conditions:
        return False, "step verification failed", failed_conditions

    return True, "step verification passed", []


def compute_csc(step: Step, scene_memory: SceneMemory, task_spec: TaskSpec) -> float:
    """Compute semantic/common-sense confidence without external models."""

    del task_spec

    if step.name == SkillName.OBSERVE:
        return 1.0

    if step.name == SkillName.GRASP:
        obj = scene_memory.get_object(step.args["object"])
        if obj is None:
            return 0.0
        return semantic_identity_score(obj)

    if step.name == SkillName.PLACE:
        plate = scene_memory.get_object(step.args["plate"])
        if plate is None:
            return 0.0
        if has_fixed_container_plan(plate):
            # Completed task-local localization, not a probability estimate.
            return 1.0
        score = (
            0.5 * plate.ocr_match_score
            + 0.3 * plate.vlm_match_score
            + 0.2 * plate.detector_confidence
        )
        if plate.object_type in {"plate", "container"}:
            score = max(score, 0.6)
        return clamp01(score)

    return 0.0


def compute_lc(step: Step, scene_memory: SceneMemory) -> tuple[float, list[str]]:
    """Compute symbolic logic confidence using simple precondition rules."""

    if step.name == SkillName.OBSERVE:
        return 1.0, []

    failed_conditions: list[str] = []

    if step.name == SkillName.GRASP:
        object_id = step.args["object"]

        if not scene_memory.is_visible(object_id):
            failed_conditions.append("target_not_visible")
        if not scene_memory.is_reachable(object_id):
            failed_conditions.append("target_not_reachable")
        if not scene_memory.is_graspable(object_id):
            failed_conditions.append("target_not_graspable")
        if scene_memory.is_severely_occluded(
            object_id,
            threshold=GRASP_OCCLUSION_THRESHOLD,
        ):
            failed_conditions.append("target_occluded")

        if not failed_conditions:
            return 1.0, []
        if "target_not_visible" in failed_conditions or "target_occluded" in failed_conditions:
            return 0.0, failed_conditions
        return 0.5, failed_conditions

    if step.name == SkillName.PLACE:
        object_id = step.args["object"]
        plate_id = step.args["plate"]

        if not scene_memory.in_gripper(object_id):
            failed_conditions.append("object_not_in_gripper")
        if not scene_memory.is_visible(plate_id):
            failed_conditions.append("plate_not_visible")
        if not scene_memory.plate_has_free_space(plate_id):
            failed_conditions.append("plate_no_free_space")

        plate = scene_memory.get_object(plate_id)
        if plate is None or not plate.placeable:
            failed_conditions.append("plate_not_placeable")

        if not failed_conditions:
            return 1.0, []
        if (
            "object_not_in_gripper" in failed_conditions
            or "plate_not_visible" in failed_conditions
        ):
            return 0.0, failed_conditions
        return 0.5, failed_conditions

    return 0.0, ["invalid_skill"]


def is_probe_solvable(failed_conditions: list[str]) -> bool:
    """Return whether all failures can reasonably be solved by observation."""

    probe_solvable = {
        "target_not_visible",
        "target_low_confidence",
        "plate_not_visible",
        "plate_low_confidence",
        "ocr_low_confidence",
    }
    return bool(failed_conditions) and all(
        condition in probe_solvable for condition in failed_conditions
    )


def suggest_probe(
    step: Step,
    failed_conditions: list[str],
    scene_memory: SceneMemory,
) -> str:
    """Suggest a safe probe/observe action for the current failures."""

    del step, scene_memory

    if "target_not_visible" in failed_conditions:
        return "observe_target_region"
    if "target_low_confidence" in failed_conditions:
        return "observe_closeup_target"
    if "ocr_low_confidence" in failed_conditions:
        return "observe_closeup_for_ocr"
    if "plate_not_visible" in failed_conditions:
        return "observe_plate_region"
    if "plate_low_confidence" in failed_conditions:
        return "observe_closeup_plate"
    return "observe_global_table"


def convert_failure_to_constraint(failed_conditions: list[str], step: Step) -> str:
    """Convert failure labels into simple symbolic constraint strings."""

    object_id = step.args.get("object", "unknown_object")
    plate_id = step.args.get("plate", "unknown_plate")
    constraints: list[str] = []

    for condition in failed_conditions:
        if condition == "target_occluded":
            constraints.append(f"target_occluded({object_id})")
        elif condition == "target_not_reachable":
            constraints.append(f"not_reachable({object_id})")
        elif condition == "target_not_graspable":
            constraints.append(f"not_graspable({object_id})")
        elif condition == "object_not_in_gripper":
            constraints.append(f"object_not_in_gripper({object_id})")
        elif condition == "plate_no_free_space":
            constraints.append(f"plate_no_free_space({plate_id})")
        elif condition == "plate_not_placeable":
            constraints.append(f"plate_not_placeable({plate_id})")
        elif condition == "wrong_target_object":
            constraints.append(f"wrong_target_object({object_id})")
        elif condition == "wrong_target_plate":
            constraints.append(f"wrong_target_plate({plate_id})")
        elif condition == "object_not_found":
            constraints.append(f"object_not_found({object_id})")
        elif condition == "plate_not_found":
            constraints.append(f"plate_not_found({plate_id})")
        elif condition == "invalid_skill":
            constraints.append(f"invalid_skill({step.name})")
        elif condition == "missing_argument":
            constraints.append(f"missing_argument({step.name})")
        else:
            constraints.append(condition)

    return " and ".join(constraints)
