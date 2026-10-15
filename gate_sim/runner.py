"""Shared policy runners and validation routines for the phase-1 simulator."""

from __future__ import annotations

import json
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pybullet
from PIL import Image, ImageDraw

from gate_sim.environment import GateAction, GateOccludedPickPlaceEnv


METHODS = ("direct", "observe_first", "observe_then_manipulate", "legacy")


def _run_execute_to_completion(env: GateOccludedPickPlaceEnv, records: List[dict], max_actions: int) -> None:
    while len(records) < max_actions and env.action_mask()[GateAction.EXECUTE.value]:
        record = env.step_gate(GateAction.EXECUTE)
        records.append(record)
        if record["task_success"] or not record["primitive_success"]:
            return


def _attempt_execute_to_completion(env: GateOccludedPickPlaceEnv, records: List[dict], max_actions: int) -> None:
    """Try the task even when masked, preserving unknown outcome semantics."""
    if env.action_mask()[GateAction.EXECUTE.value]:
        _run_execute_to_completion(env, records, max_actions)
    elif len(records) < max_actions:
        records.append(env.step_gate(GateAction.EXECUTE))


def _scene_memory_for_legacy(env: GateOccludedPickPlaceEnv):
    """Conservative adapter; missing semantics remain zero, never invented."""
    from nesyro_gate import ObjectState, SceneMemory

    target_id = env.target_key
    target_estimate = env.memory.objects.get(target_id)
    target_current = bool(target_estimate and not target_estimate.stale_pose)
    objects = {
        target_id: ObjectState(
            object_id=target_id, name="task target", object_type="box",
            visible=target_current,
            detector_confidence=target_estimate.identity_confidence if target_current else 0.0,
            ocr_text=None, ocr_match_score=0.0, vlm_match_score=0.0,
            reachable=bool(target_current and target_estimate.graspable),
            graspable=bool(target_current and target_estimate.graspable and target_estimate.visibility >= 0.90),
            occlusion_level=(1.0 - target_estimate.visibility) if target_current else 1.0,
            in_gripper=env.memory.gripper_holding == target_id,
        ),
        "goal_container": ObjectState(
            object_id="goal_container", name="goal container", object_type="plate",
            visible=True, detector_confidence=1.0, ocr_text=None,
            ocr_match_score=1.0, vlm_match_score=1.0,
            reachable=True, graspable=False, occlusion_level=0.0,
            free_space=True, placeable=True,
        ),
    }
    return SceneMemory(objects=objects, gripper_holding=env.memory.gripper_holding)


def _legacy_action(env: GateOccludedPickPlaceEnv) -> GateAction:
    from nesyro_gate import GateInput, SkillName, Step, TaskSpec, UnifiedNeSyRoGate

    target_id = env.target_key
    if env.memory.task_node == "place_target":
        step = Step(SkillName.PLACE, {"object": target_id, "plate": "goal_container"}, step_id="sim-place")
    else:
        step = Step(SkillName.GRASP, {"object": target_id}, step_id="sim-grasp")
    task = TaskSpec(
        target_object_name="red target box", target_plate_name="goal container",
        allowed_skills=[SkillName.OBSERVE, SkillName.GRASP, SkillName.PLACE],
        target_object_id=target_id, target_plate_id="goal_container",
    )
    raw = env.build_gate_input_dict()
    gate_input = GateInput(
        current_step={"name": step.name.value, "skill": step.name.value, "step_id": step.step_id},
        target_confidence=raw["raw_values"][0],
        target_visibility=raw["raw_values"][1],
        occlusion_score=raw["raw_values"][2],
        pose_confidence=raw["raw_values"][3],
        previous_action_result=env.memory.last_action_result,
        visual_probe_count=env.counters["visual_probes"],
        manipulation_probe_count=env.counters["manipulation_probes"],
        task_progress=raw["raw_values"][11],
        extra={"sim_adapter_missing_semantics_mapped_to_zero_for_legacy": ["ocr_match_score", "vlm_match_score"]},
    )
    result = UnifiedNeSyRoGate(mode="legacy").decide(step, _scene_memory_for_legacy(env), task, gate_input)
    return GateAction(result.action.value)


def run_method(env: GateOccludedPickPlaceEnv, method: str, max_actions: int = 6) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    records: List[dict] = []
    if method == "direct":
        _attempt_execute_to_completion(env, records, max_actions)
    elif method == "observe_first":
        if env.action_mask()[GateAction.VISUAL_PROBE.value]:
            records.append(env.step_gate(GateAction.VISUAL_PROBE))
        _attempt_execute_to_completion(env, records, max_actions)
    elif method == "observe_then_manipulate":
        if env.action_mask()[GateAction.VISUAL_PROBE.value]:
            records.append(env.step_gate(GateAction.VISUAL_PROBE))
        if not env.action_mask()[GateAction.EXECUTE.value] and env.action_mask()[GateAction.MANIPULATION_PROBE.value] and len(records) < max_actions:
            records.append(env.step_gate(GateAction.MANIPULATION_PROBE))
        _attempt_execute_to_completion(env, records, max_actions)
    else:
        while len(records) < max_actions:
            action = _legacy_action(env)
            if not env.action_mask().get(action.value, False):
                records.append(env.step_gate(action))
                break
            record = env.step_gate(action)
            records.append(record)
            if record["task_success"] or not record["primitive_success"]:
                break
    evaluator = env.get_evaluator_state()
    return {
        "method": method,
        "episode_id": env.episode_id,
        "seed": env.seed,
        "scene_type": env.scenario_type,
        "task_success": any(record["task_success"] for record in records),
        "actions": records,
        "counts": deepcopy(env.counters),
        "total_sim_time_seconds": env.sim_time_seconds,
        "total_compute_time_seconds": env.compute_time_seconds,
        "evaluator_final": evaluator,
    }


def run_demos(output_dir: str) -> dict:
    specs = [
        ("direct", 100, "A"),
        ("observe_first", 101, "B"),
        ("observe_then_manipulate", 102, "C"),
    ]
    results = []
    for method, seed, scene in specs:
        with GateOccludedPickPlaceEnv(output_dir=output_dir, record_video=True) as env:
            env.reset(seed, scene)
            result = run_method(env, method)
            result["video"] = env.save_video(f"demo_{method}")
            results.append(result)
    report = {"kind": "demo_report", "results": results}
    _write_report(output_dir, "demo_report.json", report)
    return report


def render_scene_gallery(output_dir: str, overview_width: int = 1280, overview_height: int = 720) -> dict:
    """Render one untouched RGB-D observation for every scene type."""

    rows = []
    gallery_frames = []
    scene_config_version = None
    for index, scene in enumerate("ABCD"):
        with GateOccludedPickPlaceEnv(output_dir=output_dir) as env:
            scene_config_version = env.config.scene_config_version
            observation = env.reset(900 + index, scene)
            policy = observation["policy_observation"]
            gallery_frames.append(policy["rgb"].copy())
            evaluator = env.get_evaluator_state()
            overview_path = Path(output_dir) / "overview_scene_{}_{}x{}.png".format(
                scene, overview_width, overview_height
            )
            Image.fromarray(env.render_overview(overview_width, overview_height)).save(overview_path)
            rows.append(
                {
                    "scene_type": scene,
                    "seed": 900 + index,
                    "rgb": policy["artifact_refs"]["rgb"],
                    "depth": policy["artifact_refs"]["depth"],
                    "overview_rgb": str(overview_path),
                    "overview_is_policy_observation": False,
                    "detected_objects": sorted(policy["current_estimates"]),
                    "physical_objects": sorted(evaluator["object_states"]),
                }
            )
    frame_height, frame_width = gallery_frames[0].shape[:2]
    gallery = Image.new("RGB", (2 * frame_width, 2 * frame_height), (245, 245, 245))
    draw = ImageDraw.Draw(gallery)
    for index, (scene, frame) in enumerate(zip("ABCD", gallery_frames)):
        x = (index % 2) * frame_width
        y = (index // 2) * frame_height
        gallery.paste(Image.fromarray(frame), (x, y))
        draw.rectangle((x + 4, y + 4, x + 28, y + 20), fill=(255, 255, 255))
        draw.text((x + 10, y + 6), scene, fill=(15, 15, 15))
    gallery_path = Path(output_dir) / "textured_scene_gallery.png"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(gallery_path)
    report = {
        "kind": "textured_scene_gallery",
        "scene_config_version": scene_config_version,
        "gallery_rgb": str(gallery_path),
        "overview_resolution": [overview_width, overview_height],
        "rows": rows,
    }
    _write_report(output_dir, "textured_scene_gallery.json", report)
    return report


def record_manipulation_probe_video(
    output_dir: str,
    width: int = 1280,
    height: int = 720,
    seed: int = 902,
) -> dict:
    """Record the physical blocker-to-buffer macro action from enriched scene C."""

    with GateOccludedPickPlaceEnv(
        output_dir=output_dir,
        record_video=True,
        video_view="overview",
        video_resolution=(width, height),
    ) as env:
        env.reset(seed, "C")
        selected_before = env._select_manipulation_candidate()
        result = env.step_gate(GateAction.MANIPULATION_PROBE)
        video = env.save_video("scene_C_blocker_to_buffer_{}x{}".format(width, height))
        report = {
            "kind": "physical_manipulation_probe_video",
            "simulation_only": True,
            "scene_type": "C",
            "seed": seed,
            "resolution": [width, height],
            "selected_candidate_from_policy_memory": selected_before,
            "primitive_success": result["primitive_success"],
            "failure_reason": result["failure_reason"],
            "video": video,
            "action": result,
        }
    _write_report(output_dir, "manipulation_video_report.json", report)
    return report


def record_gate_action_videos(
    output_dir: str,
    width: int = 1280,
    height: int = 720,
    seed: int = 1004,
) -> dict:
    """Record one honest presentation video for every high-level Gate action."""

    videos = {}

    # VISUAL_PROBE: physical camera-pose motion and a genuinely new RGB-D frame.
    with GateOccludedPickPlaceEnv(
        output_dir=output_dir, record_video=True, video_view="overview",
        video_resolution=(width, height),
    ) as env:
        env.reset(seed, "E")
        result = env.step_gate(GateAction.VISUAL_PROBE)
        path = env.save_video(
            "gate_VISUAL_PROBE",
            ["Gate output: VISUAL_PROBE", "Physical view motion; new RGB-D observation"],
        )
        videos[GateAction.VISUAL_PROBE.value] = {"video": path, "results": [result]}

    # MANIPULATION_PROBE: move the observed blue blocker to the cyan buffer.
    with GateOccludedPickPlaceEnv(
        output_dir=output_dir, record_video=True, video_view="overview",
        video_resolution=(width, height),
    ) as env:
        env.reset(seed, "E")
        result = env.step_gate(GateAction.MANIPULATION_PROBE)
        path = env.save_video(
            "gate_MANIPULATION_PROBE",
            ["Gate output: MANIPULATION_PROBE", "Blue blocker -> cyan buffer; then re-observe"],
        )
        videos[GateAction.MANIPULATION_PROBE.value] = {"video": path, "results": [result]}

    # EXECUTE: establish the valid post-probe state without recording it, then
    # record the two task-node executions (grasp target, place in goal).
    with GateOccludedPickPlaceEnv(
        output_dir=output_dir, record_video=False, video_view="overview",
        video_resolution=(width, height),
    ) as env:
        env.reset(seed, "E")
        precondition = env.step_gate(GateAction.MANIPULATION_PROBE)
        if not precondition["primitive_success"]:
            raise RuntimeError("could not establish EXECUTE demo precondition")
        env.record_video = True
        env.begin_video_segment()
        results = []
        while env.action_mask()[GateAction.EXECUTE.value] and len(results) < 2:
            results.append(env.step_gate(GateAction.EXECUTE))
        path = env.save_video(
            "gate_EXECUTE",
            ["Gate output: EXECUTE", "Textured daily target -> matching goal container"],
        )
        videos[GateAction.EXECUTE.value] = {"video": path, "results": results}

    # REPLAN is intentionally masked in this phase. Its video is static and
    # explicitly states that no motion/effect occurred.
    with GateOccludedPickPlaceEnv(
        output_dir=output_dir, record_video=True, video_view="overview",
        video_resolution=(width, height),
    ) as env:
        env.reset(seed, "E")
        result = env.step_gate(GateAction.REPLAN)
        path = env.save_video(
            "gate_REPLAN_unavailable",
            ["Gate output: REPLAN (masked/unavailable)", "No planner attached; no motion and no free reset"],
            minimum_duration_seconds=3.0,
        )
        videos[GateAction.REPLAN.value] = {"video": path, "results": [result]}

    # Full task sequence, separately retained for intuitive dataset review.
    with GateOccludedPickPlaceEnv(
        output_dir=output_dir, record_video=True, video_view="overview",
        video_resolution=(width, height),
    ) as env:
        env.reset(seed, "E")
        sequence = [env.step_gate(GateAction.MANIPULATION_PROBE)]
        while env.action_mask()[GateAction.EXECUTE.value] and len(sequence) < 3:
            sequence.append(env.step_gate(GateAction.EXECUTE))
        full_path = env.save_video(
            "full_daily_target_occlusion_task",
            ["Full task: blocker -> buffer; daily target -> goal", "Physical contacts, release and RGB-D re-observation"],
        )
        full_task = {
            "video": full_path,
            "results": sequence,
            "task_success": any(row["task_success"] for row in sequence),
            "evaluator_final": env.get_evaluator_state(),
        }

    report = {
        "kind": "gate_action_video_suite",
        "simulation_only": True,
        "scene_type": "E",
        "seed": seed,
        "target_object_id": GateOccludedPickPlaceEnv.DAILY_TARGET_KEY,
        "resolution": [width, height],
        "gate_outputs": videos,
        "full_task": full_task,
        "replan_semantics": "masked and side-effect free because no simulator replanner is attached",
    }
    _write_report(output_dir, "gate_action_video_report.json", report)
    return report


def run_comparison(output_dir: str, seeds: Iterable[int]) -> dict:
    seeds = list(seeds)
    if len(seeds) > 20:
        raise ValueError("phase-1 development comparison is capped at 20 seeds")
    attempts = []
    for seed in seeds:
        scene = "ABCD"[seed % 4]
        for method in METHODS:
            with GateOccludedPickPlaceEnv(output_dir=output_dir) as env:
                env.reset(seed, scene)
                attempts.append(run_method(env, method))
    summary = {}
    for method in METHODS:
        rows = [row for row in attempts if row["method"] == method]
        summary[method] = {
            "attempts": len(rows),
            "successes": sum(bool(row["task_success"]) for row in rows),
            "success_rate": sum(bool(row["task_success"]) for row in rows) / len(rows) if rows else None,
            "mean_observations": float(np.mean([row["counts"]["observations"] for row in rows])) if rows else None,
            "mean_manipulation_probes": float(np.mean([row["counts"]["manipulation_probes"] for row in rows])) if rows else None,
        }
    report = {"kind": "development_comparison", "fixed_seeds": seeds, "all_attempts_included": True, "summary": summary, "attempts": attempts}
    _write_report(output_dir, "comparison_report.json", report)
    return report


def _result_signature(result: dict) -> dict:
    return {
        "action": result["action"],
        "available": result["available"],
        "primitive_success": result["primitive_success"],
        "task_success": result["task_success"],
        "failure_reason": result["failure_reason"],
        "probe_effect": result["probe_effect"],
    }


def run_restore_validation(output_dir: str) -> dict:
    cases = []
    with GateOccludedPickPlaceEnv(output_dir=output_dir) as env:
        env.reset(700, "B")
        snapshot = env.save_snapshot()
        first = env.step_gate(GateAction.VISUAL_PROBE)
        first_state = env.physical_state_vector()
        first_rgb = env.get_gate_observation()["policy_observation"]["rgb"]
        env.restore_snapshot(snapshot, branch_id="replay")
        second = env.step_gate(GateAction.VISUAL_PROBE)
        second_state = env.physical_state_vector()
        second_rgb = env.get_gate_observation()["policy_observation"]["rgb"]
        cases.append(_restore_case("visual_probe", first, second, first_state, second_state, first_rgb, second_rgb, env))

    with GateOccludedPickPlaceEnv(output_dir=output_dir) as env:
        env.reset(701, "C")
        env.step_gate(GateAction.VISUAL_PROBE)
        snapshot = env.save_snapshot()
        first = env.step_gate(GateAction.MANIPULATION_PROBE)
        first_state = env.physical_state_vector()
        first_rgb = env.get_gate_observation()["policy_observation"]["rgb"]
        env.restore_snapshot(snapshot, branch_id="replay")
        second = env.step_gate(GateAction.MANIPULATION_PROBE)
        second_state = env.physical_state_vector()
        second_rgb = env.get_gate_observation()["policy_observation"]["rgb"]
        cases.append(_restore_case("manipulation_probe", first, second, first_state, second_state, first_rgb, second_rgb, env))

    ready = all(case["passed"] for case in cases)
    branches = run_branch_validation(output_dir) if ready else {"skipped": True, "reason": "restore validation failed"}
    report = {"kind": "restore_validation", "fork_evaluation_ready": ready, "tolerances": {"max_state_abs_error": 1e-4, "max_rgb_abs_error": 0}, "cases": cases, "branch_test": branches}
    _write_report(output_dir, "restore_report.json", report)
    return report


def _restore_case(name: str, first: dict, second: dict, state_a: np.ndarray, state_b: np.ndarray, rgb_a: np.ndarray, rgb_b: np.ndarray, env: GateOccludedPickPlaceEnv) -> dict:
    state_error = float(np.max(np.abs(state_a - state_b)))
    rgb_error = int(np.max(np.abs(rgb_a.astype(np.int16) - rgb_b.astype(np.int16))))
    consistent = _result_signature(first) == _result_signature(second)
    return {
        "name": name,
        "state_max_abs_error": state_error,
        "rgb_max_abs_error": rgb_error,
        "result_consistent": consistent,
        "counter_state_after_replay": deepcopy(env.counters),
        "passed": bool(state_error <= 1e-4 and rgb_error == 0 and consistent),
    }


def run_branch_validation(output_dir: str) -> dict:
    with GateOccludedPickPlaceEnv(output_dir=output_dir) as env:
        env.reset(702, "C")
        snapshot = env.save_snapshot()
        baseline_memory = json.dumps(snapshot.memory.to_dict(), sort_keys=True)
        branches = []
        for branch_id, first_action in (("visual_first", GateAction.VISUAL_PROBE), ("manip_first", GateAction.MANIPULATION_PROBE)):
            env.restore_snapshot(snapshot, branch_id=branch_id)
            restored_memory = json.dumps(env.memory.to_dict(), sort_keys=True)
            records = [env.step_gate(first_action)]
            if env.action_mask()[GateAction.MANIPULATION_PROBE.value] and not env.action_mask()[GateAction.EXECUTE.value] and len(records) < 4:
                records.append(env.step_gate(GateAction.MANIPULATION_PROBE))
            _run_execute_to_completion(env, records, 4)
            branches.append({
                "branch_id": branch_id,
                "first_action": first_action.value,
                "restored_memory_exact": restored_memory == baseline_memory,
                "restored_counters": deepcopy(snapshot.counters),
                "final_counters": deepcopy(env.counters),
                "task_success": any(row["task_success"] for row in records),
                "actions": records,
                "same_total_action_budget": 4,
                "same_sim_time_limit_seconds": env.config.physics.task_time_limit_seconds,
            })
        return {"skipped": False, "split_group": "seed-702-snapshot-0", "branches": branches, "memory_and_counter_isolation": all(row["restored_memory_exact"] for row in branches)}


def environment_audit() -> dict:
    try:
        import imageio
        import PIL
        versions = {"numpy": np.__version__, "pybullet": getattr(pybullet, "__version__", "3.2.7 package"), "imageio": imageio.__version__, "Pillow": PIL.__version__}
    except Exception as exc:
        versions = {"error": str(exc)}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": versions,
        "backend": "PyBullet DIRECT / TinyRenderer / CPU single environment",
        "real_robot_enabled": False,
        "gpu_required": False,
    }


def _write_report(output_dir: str, filename: str, report: dict) -> str:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return str(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)
