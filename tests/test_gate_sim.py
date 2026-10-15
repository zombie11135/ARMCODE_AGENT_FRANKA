"""Phase-1 simulator acceptance tests (stdlib unittest compatible)."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from gate_sim import GateOccludedPickPlaceEnv
from gate_sim.config import GateSimConfig
from gate_sim.runner import run_method
from nesyro_gate.learnable_gate import GATE_FEATURE_NAMES


class GateSimulationTests(unittest.TestCase):
    def test_textured_daily_clutter_is_physical_visible_and_optional(self) -> None:
        daily = {
            "daily_oat_box", "daily_orange_can",
            "daily_purple_cup", "daily_cyan_bottle",
        }
        with tempfile.TemporaryDirectory() as directory:
            with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                observation = env.reset(31, "D")
                evaluator = env.get_evaluator_state()
                self.assertTrue(daily.issubset(evaluator["object_states"]))
                self.assertTrue(daily.issubset(observation["policy_observation"]["current_estimates"]))
                for key in daily:
                    metadata = evaluator["object_states"][key]["object_metadata"]
                    self.assertEqual(metadata["texture_version"], "procedural-daily-products-v1")
                    self.assertTrue(Path(metadata["texture_path"]).exists())
                rgb = observation["policy_observation"]["rgb"]
                self.assertGreater(len(np.unique(rgb.reshape(-1, 3), axis=0)), 30)
                counters_before = dict(env.counters)
                overview = env.render_overview(640, 360)
                self.assertEqual(overview.shape, (360, 640, 3))
                self.assertEqual(env.counters, counters_before)

            plain_config = replace(GateSimConfig(), textured_daily_clutter_enabled=False)
            with GateOccludedPickPlaceEnv(config=plain_config, output_dir=directory) as env:
                env.reset(31, "D")
                self.assertTrue(daily.isdisjoint(env.get_evaluator_state()["object_states"]))

    def test_reset_render_stability_and_policy_truth_separation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for seed, scene in enumerate("ABCD"):
                with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                    observation = env.reset(seed, scene)
                    policy = observation["policy_observation"]
                    self.assertEqual(policy["rgb"].shape, (144, 192, 3))
                    self.assertEqual(policy["depth_m"].shape, (144, 192))
                    serialized = json.dumps({key: value for key, value in policy.items() if key not in {"rgb", "depth_m"}})
                    self.assertNotIn("body_id", serialized)
                    self.assertNotIn("scene_type", serialized)
                    self.assertNotIn("true_blocker", serialized)
                    stable, details = env._check_scene_stability()
                    self.assertTrue(stable, details)

    def test_daily_target_occlusion_probe_then_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                observation = env.reset(1004, "E")
                self.assertEqual(env.target_key, "target_daily_oat_box")
                self.assertIn(env.target_key, observation["policy_observation"]["current_estimates"])
                self.assertFalse(observation["action_mask"]["EXECUTE"])
                self.assertEqual(env._select_manipulation_candidate(), "distractor_blue")

                probe = env.step_gate("MANIPULATION_PROBE")
                self.assertTrue(probe["primitive_success"], probe)
                self.assertEqual(probe["executor_feedback"]["candidate"], "distractor_blue")
                self.assertTrue(env.action_mask()["EXECUTE"])

                grasp = env.step_gate("EXECUTE")
                place = env.step_gate("EXECUTE")
                self.assertTrue(grasp["primitive_success"], grasp)
                self.assertTrue(place["primitive_success"], place)
                self.assertTrue(place["task_success"], place)
                evaluator = env.get_evaluator_state()
                self.assertTrue(evaluator["target_in_goal"])
                self.assertTrue(evaluator["target_released"])

    def test_existing_23d_schema_is_exact_and_unknown_semantics_are_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                env.reset(0, "A")
                adapter = env.build_gate_input_dict()
                self.assertEqual(adapter["feature_names"], GATE_FEATURE_NAMES)
                self.assertEqual(len(adapter["raw_values"]), 23)
                for name in ("ocr_match_score", "vlm_match_score"):
                    index = GATE_FEATURE_NAMES.index(name)
                    self.assertIsNone(adapter["raw_values"][index])
                    self.assertFalse(adapter["validity"][index])

    def test_unavailable_action_keeps_outcome_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                env.reset(2, "C")
                result = env.step_gate("EXECUTE")
                self.assertFalse(result["available"])
                self.assertIsNone(result["primitive_success"])
                self.assertEqual(result["failure_reason"], "insufficient_target_information")

    def test_candidate_selector_has_no_evaluator_or_body_truth_dependency(self) -> None:
        source = inspect.getsource(GateOccludedPickPlaceEnv._select_manipulation_candidate)
        self.assertNotIn("get_evaluator_state", source)
        self.assertNotIn("object_bodies", source)
        self.assertNotIn("scenario_type", source)
        self.assertNotIn("true_blocker", source)

    def test_three_physical_demo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases = [
                ("direct", 100, "A"),
                ("observe_first", 101, "B"),
                ("observe_then_manipulate", 102, "C"),
            ]
            for method, seed, scene in cases:
                with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                    env.reset(seed, scene)
                    result = run_method(env, method)
                    self.assertTrue(result["task_success"], (method, result))
                    contacts = [row["executor_feedback"] for row in result["actions"] if row["action"] == "EXECUTE" and row["primitive_success"]]
                    self.assertTrue(any(row.get("bilateral_contact") for row in contacts), method)
                    if method == "observe_then_manipulate":
                        self.assertEqual(result["counts"]["manipulation_probes"], 1)

    def test_decision_outcome_log_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with GateOccludedPickPlaceEnv(output_dir=directory) as env:
                env.reset(0, "A")
                result = env.step_gate("EXECUTE")
                path = Path(directory) / env.episode_id / "gate_experience.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["event"] for row in rows], ["gate_decision", "gate_outcome"])
                self.assertEqual(rows[0]["decision_id"], rows[1]["decision_id"])
                self.assertEqual(rows[0]["decision_id"], result["decision_id"])
                self.assertIn("snapshot_reference", rows[0])


if __name__ == "__main__":
    unittest.main()
