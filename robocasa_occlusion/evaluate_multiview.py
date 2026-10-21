"""Evaluate occlusion evidence against independent counterfactual render truth."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from occlusion_reasoning.multiview import MultiViewConfig
from occlusion_reasoning.planar import PlanarConfig
from occlusion_reasoning.scene import infer_scene
from .probe_multiview import CAMERAS, geometry_groups, load_view, save_result


MODES = ("clean", "depth_1mm_dropout10", "depth_3mm_dropout30", "mask_erode1",
         "mask_dilate1", "pose_2mm_0.2deg", "pose_5mm_1deg", "miss_target_both",
         "miss_occluder_left")


def perturb(observations, names, truth, mode, seed):
    observations = copy.deepcopy(observations)
    rng = np.random.default_rng(seed)
    for view_index, (packet, arrays) in enumerate(observations):
        if mode.startswith("depth_"):
            sigma, dropout = (.001, .1) if mode == "depth_1mm_dropout10" else (.003, .3)
            arrays["depth_m"] += rng.normal(0, sigma, arrays["depth_m"].shape).astype(np.float32)
            arrays["depth_m"][rng.random(arrays["depth_m"].shape) < dropout] = 0
        if mode in ("mask_erode1", "mask_dilate1"):
            op = cv2.erode if mode == "mask_erode1" else cv2.dilate
            for node in packet["relation_graph"]["nodes"]:
                key = node["raw_mask_key"]
                arrays[key] = op(arrays[key].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if mode.startswith("pose_") and view_index == 0:
            distance, angle = (.002, .2) if mode == "pose_2mm_0.2deg" else (.005, 1.)
            pose = np.asarray(packet["frame"]["T_reference_from_camera_optical"]).copy()
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
            direction = rng.normal(size=3)
            pose[:3, 3] += direction / np.linalg.norm(direction) * distance
            packet["frame"]["T_reference_from_camera_optical"] = pose.tolist()
        if mode == "miss_target_both":
            packet["relation_graph"]["nodes"] = [
                n for n in packet["relation_graph"]["nodes"] if names[n["node_id"]] != truth["target"]]
        if mode == "miss_occluder_left" and view_index == 0:
            occluders = {e["foreground"] for e in truth["occlusion_relations"]}
            packet["relation_graph"]["nodes"] = [
                n for n in packet["relation_graph"]["nodes"] if names[n["node_id"]] not in occluders]
    return observations


def score(result, names, truth, observations):
    mapping = {e["entity_id"]: {names[n] for n in e["members"]} for e in result["entities"]}
    predicted, direct, planar = set(), set(), set()
    view_predicted, view_direct = set(), set()
    for edge in result["relation_graph"]["edges"]:
        front, back = mapping[edge["source"]], mapping[edge["target"]]
        # A wrong identity merge is not silently removed from the false positives.
        pair = ("|".join(sorted(front)), "|".join(sorted(back)))
        predicted.add(pair)
        view_predicted.add((*pair, edge["view_id"]))
        if edge["relation"] == "occludes_observed_surface":
            direct.add(pair)
            view_direct.add((*pair, edge["view_id"]))
        else:
            planar.add(pair)
    all_truth = {(e["foreground"], e["background"]) for e in truth["occlusion_relations"]}
    observable_truth = {(e["foreground"], e["background"]) for e in truth["occlusion_relations"]
                        if e["background_observed_in_any_view"]}
    view_truth = {(e["foreground"], e["background"], e["view"]) for e in truth["occlusion_relations"]}
    accepted = [a for a in result["associations"] if a["status"] == "accepted"]
    shared_names = set.intersection(*[
        {names[n["node_id"]] for n in packet["relation_graph"]["nodes"]}
        for packet, _ in observations])
    correct_matches = sum(len({names[n] for n in a["nodes"]}) == 1 for a in accepted)
    return {
        "tp": len(predicted & all_truth), "fp": len(predicted - all_truth),
        "fn_observable": len(observable_truth - predicted),
        "fn_all": len(all_truth - predicted),
        "truth_pairs": len(all_truth), "observable_truth_pairs": len(observable_truth),
        "direct_tp": len(direct & all_truth), "direct_fp": len(direct - all_truth),
        "planar_tp": len(planar & all_truth), "planar_fp": len(planar - all_truth),
        "view_tp": len(view_predicted & view_truth), "view_fp": len(view_predicted - view_truth),
        "view_truth_count": len(view_truth),
        "direct_view_tp": len(view_direct & view_truth), "direct_view_fp": len(view_direct - view_truth),
        "accepted_associations": len(accepted), "correct_associations": correct_matches,
        "co_visible_object_pairs": len(shared_names),
        "unresolved_front_surfaces": len(result["unresolved_front_surfaces"]),
        "cross_view_geometry_rejected": int(not result["geometry_health"]["cross_view_inference_enabled"]),
        "predicted_pairs": sorted(predicted), "truth": sorted(all_truth),
        "missed_observable_pairs": sorted(observable_truth - predicted),
        "false_pairs": sorted(predicted - all_truth),
        "both_hidden_target_invented": (
            not any(names[n["node_id"]] == truth["target"] for p, _ in observations
                    for n in p["relation_graph"]["nodes"])
            and any(truth["target"] in pair for pair in predicted)),
    }


def evaluate(scene_dir, output_dir, modes=MODES):
    scene_dir, output_dir = Path(scene_dir), Path(output_dir)
    manifest_path = scene_dir / "cases.json"
    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    mv_config, planar_config = MultiViewConfig(), PlanarConfig()
    report = {
        "input_suite": str(scene_dir.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "multiview_config": asdict(mv_config), "planar_config": asdict(planar_config),
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path("occlusion_reasoning")/name for name in ("multiview.py", "planar.py", "scene.py")]
        },
        "metric_scope": "object-pair union across views; separate view-specific and direct/prior counts",
        "truth_source": "counterfactual isolation render, >=6 pixels and >=3% full projected silhouette",
        "observation_mode": "oracle visible instance masks; optional controlled faults; no recognition algorithm",
        "limitations": f"{len(manifest['cases'])} arrangements in one kitchen/profile, fixed cameras; not hardware or grasp success",
        "cases": {mode: [] for mode in modes},
    }
    # Record the configuration/code before processing this suite.
    (output_dir / "evaluation_protocol.json").write_text(json.dumps({k:v for k,v in report.items() if k != "cases"}, indent=2))
    for index, case in enumerate(manifest["cases"]):
        root = scene_dir / case["case_id"]
        groups = geometry_groups(root)
        loaded = [load_view(root, camera, groups) for camera in CAMERAS]
        original = [x[0] for x in loaded]
        names = {k:v for x in loaded for k,v in x[1].items()}
        for mode in modes:
            observations = perturb(original, names, case, mode, seed=10000+index)
            start = time.perf_counter()
            result = infer_scene(observations, multiview_config=mv_config, planar_config=planar_config)
            duration = time.perf_counter() - start
            row = score(result, names, case, observations)
            row.update(case_id=case["case_id"], scenario=case["scenario"], runtime_s=duration)
            report["cases"][mode].append(row)
            if mode == "clean":
                save_result(root, observations, result, output_dir / "clean" / case["case_id"], save_packets=False)
        print(case["case_id"], {mode: (report["cases"][mode][-1]["tp"],
                                     report["cases"][mode][-1]["fp"],
                                     report["cases"][mode][-1]["fn_observable"])
                               for mode in modes}, flush=True)
    summaries = {}
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    for mode, rows in report["cases"].items():
        totals = {k: sum(r[k] for r in rows) for k in rows[0] if type(rows[0][k]) is int}
        totals.update(
            case_count=len(rows),
            precision=ratio(totals["tp"], totals["tp"]+totals["fp"]),
            observable_recall=ratio(totals["tp"], totals["observable_truth_pairs"]),
            all_truth_recall=ratio(totals["tp"], totals["truth_pairs"]),
            association_precision=ratio(totals["correct_associations"], totals["accepted_associations"]),
            association_recall=ratio(totals["correct_associations"], totals["co_visible_object_pairs"]),
            runtime_median_s=float(np.median([r["runtime_s"] for r in rows])),
            runtime_p95_s=float(np.quantile([r["runtime_s"] for r in rows], .95)),
            clear_scenes_with_false_edges=sum(r["scenario"] == "clear" and r["fp"] > 0 for r in rows),
            invented_hidden_targets=sum(r["both_hidden_target_invented"] for r in rows),
        )
        summaries[mode] = totals
    report["summary"] = summaries
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(summaries, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("runs/occlusion_multiview/geometry_holdout"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/occlusion_multiview/holdout_results"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if args.scene_dir.resolve() == args.output_dir.resolve():
        parser.error("output must differ from input")
    evaluate(args.scene_dir, args.output_dir, args.modes)


if __name__ == "__main__":
    main()
