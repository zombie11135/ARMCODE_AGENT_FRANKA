"""Collect small counterfactual Gate datasets from recoverable simulation.

This module intentionally has no torch dependency, so it runs inside the
isolated ``.venv-gate-sim`` environment.  Simulator truth is reduced to
training labels only and never serialized into the policy input manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from gate_learning.input_adapter import ACTION_NAMES, from_sim_policy_observation
from gate_sim.environment import GateAction, GateOccludedPickPlaceEnv


SCENE_CYCLE = ("A", "B", "C", "D", "E")


def _fixed_followup(env: GateOccludedPickPlaceEnv, records: List[dict], max_actions: int) -> None:
    """Shared post-branch policy; it never reads scenario or evaluator truth."""

    while len(records) < max_actions and not any(row["task_success"] for row in records):
        mask = env.action_mask()
        if mask[GateAction.EXECUTE.value]:
            action = GateAction.EXECUTE
        elif mask[GateAction.VISUAL_PROBE.value]:
            action = GateAction.VISUAL_PROBE
        elif mask[GateAction.MANIPULATION_PROBE.value]:
            action = GateAction.MANIPULATION_PROBE
        else:
            break
        records.append(env.step_gate(action))
        if records[-1]["primitive_success"] is False:
            break


def _task_text(target_key: str) -> str:
    target = "燕麦盒" if "oat" in target_key else "红色目标盒"
    return f"把{target}抓起并放入紫色目标容器后释放"


def _split_for_index(index: int, total: int) -> str:
    if total <= 1:
        return "train"
    validation_count = max(1, int(round(total * 0.2)))
    return "validation" if index >= total - validation_count else "train"


def collect_counterfactual_initial_states(
    *,
    output_dir: str,
    seeds: Sequence[int],
    max_actions: int = 5,
) -> Dict[str, Any]:
    """Evaluate every available first action from each initial snapshot."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) > 20:
        raise ValueError("development collection is capped at 20 seeds")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "counterfactual_manifest.jsonl"
    rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for index, seed in enumerate(seeds):
        scene = SCENE_CYCLE[index % len(SCENE_CYCLE)]
        with GateOccludedPickPlaceEnv(output_dir=str(destination / "episodes")) as env:
            observation = env.reset(int(seed), scene)
            root_episode_id = env.episode_id
            snapshot = env.save_snapshot()
            task_text = _task_text(env.target_key)
            item = from_sim_policy_observation(observation, task_text=task_text)
            initial_counts = dict(env.counters)
            initial_time = float(env.sim_time_seconds)
            remaining_budget = max(
                1e-6, float(env.config.physics.task_time_limit_seconds) - initial_time
            )
            action_values = [0.0] * len(ACTION_NAMES)
            action_values_valid = [False] * len(ACTION_NAMES)
            probe_gain = [0.0, 0.0]
            probe_gain_valid = [False, False]
            risk = [0.0] * len(ACTION_NAMES)
            risk_valid = [False] * len(ACTION_NAMES)
            branch_summaries: Dict[str, Any] = {}

            for action_index, action_name in enumerate(ACTION_NAMES):
                if not bool(item.action_mask[action_index]):
                    branch_summaries[action_name] = {
                        "available": False,
                        "task_success": None,
                        "primitive_success": None,
                    }
                    continue
                env.restore_snapshot(snapshot, branch_id=f"cf-{action_name.lower()}")
                records = [env.step_gate(GateAction(action_name))]
                _fixed_followup(env, records, max_actions=max_actions)
                final_success = bool(any(record["task_success"] for record in records))
                elapsed = float(env.sim_time_seconds - initial_time)
                # Weight-free target: fraction of the identical remaining time
                # budget left after a successful completion; failures are zero.
                action_values[action_index] = (
                    max(0.0, (remaining_budget - elapsed) / remaining_budget)
                    if final_success else 0.0
                )
                action_values_valid[action_index] = True
                first_effect = records[0].get("probe_effect") or {}
                risk[action_index] = 1.0 if (
                    env.counters["unexpected_disturbances"]
                    > initial_counts["unexpected_disturbances"]
                    or env.counters["drops"] > initial_counts["drops"]
                ) else 0.0
                risk_valid[action_index] = True
                if action_name in {"VISUAL_PROBE", "MANIPULATION_PROBE"}:
                    probe_index = 0 if action_name == "VISUAL_PROBE" else 1
                    gain = first_effect.get("visibility_delta")
                    if gain is not None:
                        probe_gain[probe_index] = float(gain)
                        probe_gain_valid[probe_index] = True
                branch_summaries[action_name] = {
                    "available": True,
                    "task_success": final_success,
                    "primitive_success": records[0]["primitive_success"],
                    "actions": [record["action"] for record in records],
                    "action_count": len(records),
                    "sim_time_seconds": elapsed,
                    "action_value": action_values[action_index],
                    "visibility_delta": first_effect.get("visibility_delta"),
                    "improved_target_confirmation": first_effect.get(
                        "improved_target_confirmation"
                    ),
                    "unexpected_disturbances": int(
                        env.counters["unexpected_disturbances"]
                        - initial_counts["unexpected_disturbances"]
                    ),
                    "drops": int(env.counters["drops"] - initial_counts["drops"]),
                }

            artifacts = item.metadata["artifact_refs"]
            sample_id = f"seed-{int(seed):06d}-initial"
            rows.append({
                "sample_id": sample_id,
                "split_group": sample_id,
                "split": _split_for_index(index, len(seeds)),
                "rgb": str(Path(artifacts["rgb"]).resolve()),
                "depth": str(Path(artifacts["depth"]).resolve()),
                "task_text": task_text,
                "state_vector": item.state_vector.tolist(),
                "action_mask": item.action_mask.astype(bool).tolist(),
                "action_values": action_values,
                "action_values_valid": action_values_valid,
                "feasibility": item.action_mask.astype(float).tolist(),
                "feasibility_valid": [True] * len(ACTION_NAMES),
                "probe_gain": probe_gain,
                "probe_gain_valid": probe_gain_valid,
                "risk": risk,
                "risk_valid": risk_valid,
                "history_rgb": [],
                "label_definition": "successful_remaining_time_fraction_under_fixed_followup_v1",
            })
            report_rows.append({
                "sample_id": sample_id,
                "seed": int(seed),
                "scene_type": scene,
                "source_episode_id": root_episode_id,
                "task_text": task_text,
                "branches": branch_summaries,
            })

    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "multimodal-gate-counterfactual-collection-v1",
        "manifest": str(manifest_path.resolve()),
        "samples": len(rows),
        "max_actions_per_branch": max_actions,
        "same_followup_policy_for_all_branches": True,
        "policy_input_contains_scene_type": False,
        "policy_input_contains_evaluator_state": False,
        "label_definition": (
            "zero on failure; otherwise the fraction of the identical remaining simulation "
            "time budget left after task success under the fixed follow-up policy"
        ),
        "rows": report_rows,
    }
    report_path = destination / "collection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect recoverable-sim counterfactual Gate labels")
    parser.add_argument("--output-dir", default="runs/gate_learning/counterfactual_dev")
    parser.add_argument("--seeds", type=int, default=5, help="number of fixed seeds, capped at 20")
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--max-actions", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1 or args.seeds > 20:
        parser.error("--seeds must be in [1, 20]")
    result = collect_counterfactual_initial_states(
        output_dir=args.output_dir,
        seeds=list(range(args.seed_start, args.seed_start + args.seeds)),
        max_actions=args.max_actions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
