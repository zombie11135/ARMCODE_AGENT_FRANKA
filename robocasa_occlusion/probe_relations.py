"""Exercise single-frame relations on saved RoboCasa renders with visible-mask oracle.

This explicitly replaces instance segmentation, not occlusion inference. Only
rendered, visible object pixels are passed to the algorithm, with anonymous IDs.
No target identity, hidden silhouette, object pose or true relation is passed.
The separate validation at the end uses identity.json to check the answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np

from occlusion_reasoning.observations import InstanceEvidence, build_observation
from occlusion_reasoning.relations import extract_relations
from occlusion_reasoning.extract_relations import write_result


def visible_instances(segmentation, geom_groups):
    """Return anonymous visible masks and a separate evaluator-only ID mapping."""
    visible = []
    for object_name, geom_ids in geom_groups.items():
        # MuJoCo mjOBJ_GEOM is 5; the second channel is the rendered geom ID.
        mask = (segmentation[..., 0] == 5) & np.isin(segmentation[..., 1], geom_ids)
        y, x = np.nonzero(mask)
        if len(y):
            visible.append((float(x.mean()), float(y.mean()), object_name, mask,
                            [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]))
    instances, evaluator_ids = [], {}
    for index, (_, _, name, mask, box) in enumerate(sorted(visible, key=lambda item: item[:2])):
        source_id = f"visible_{index:03d}"
        instances.append(InstanceEvidence(source_id, mask, box, 1.0))
        evaluator_ids[source_id] = name
    return instances, evaluator_ids


def probe(scene_root, output_root, camera="inspection"):
    import mujoco  # offline model names only; no simulator step or renderer

    scene_root, output_root = Path(scene_root), Path(output_root)
    if scene_root.resolve() == output_root.resolve():
        raise ValueError("probe output must differ from saved scene directory")
    checks, summary = {}, {}
    for index, scenario in enumerate(("clear", "label_occluded", "fully_occluded")):
        root = scene_root / scenario
        model = mujoco.MjModel.from_xml_path(str((root / "evaluation/scene.xml").resolve()))
        groups = {}
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            match = re.match(r"^(carton_\d+|blocker_small|blocker_large)_", name)
            if match:
                groups.setdefault(match[1], []).append(geom_id)
        del model
        segmentation = np.load(root / f"evaluation/{camera}_segmentation.npy", allow_pickle=False)
        instances, evaluator_ids = visible_instances(segmentation, groups)
        depth = np.load(root / f"observation/{camera}_depth_m.npy", allow_pickle=False)
        camera_data = json.loads((root / f"observation/{camera}_camera.json").read_text())
        k = np.asarray(camera_data["K"])
        packet, arrays = build_observation(
            depth_m=depth, intrinsics={"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]},
            instances=instances, frame_id=f"frame_{index:03d}", query="target",
            alignment_status="recorded_aligned",
            reference_frame="simulation_world",
            T_reference_from_camera_optical=camera_data["T_world_from_camera_optical"],
        )
        packet["provenance"] = {
            "backend": "simulation_visible_segmentation_oracle",
            "geometry_source": "rendered_visible_pixels_and_depth",
            "object_poses_used": False, "target_identity_used": False,
        }
        result = extract_relations(packet, arrays)
        result["input_mode"] = "simulation_visible_segmentation_oracle"
        case_output = output_root / scenario
        observation_output = case_output / "observation"
        observation_output.mkdir(parents=True, exist_ok=True)
        packet["artifacts"] = {"arrays": "observation_arrays.npz"}
        np.savez_compressed(observation_output / "observation_arrays.npz", **arrays)
        (observation_output / "scene_observation.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        rgb = cv2.imread(str(root / f"observation/{camera}_rgb.png"))
        if rgb is None:
            raise FileNotFoundError(root / f"observation/{camera}_rgb.png")
        write_result(packet, arrays, result, case_output / "relations", rgb)

        # Evaluation only, after inference. This information never enters the packet.
        identity = json.loads((root / "evaluation/identity.json").read_text())
        names = {n["node_id"]: evaluator_ids[n["source_ids"][0]]
                 for n in packet["relation_graph"]["nodes"]}
        inferred = {(names[e["source"]], names[e["target"]])
                    for e in result["relation_graph"]["edges"] if e["relation"] == "possible_occludes"}
        if scenario == "clear":
            checks["clear_no_occlusion_hypothesis"] = len(inferred) == 0
        elif scenario == "label_occluded":
            checks["partial_correct_occlusion_direction"] = ("blocker_small", identity["target"]) in inferred
        else:
            checks["fully_hidden_target_not_in_observation_nodes"] = identity["target"] not in names.values()
        summary[scenario] = {
            "visible_instances": len(instances),
            "possible_occlusion_edges": len(inferred),
            "pair_statuses": [p["status"] for p in result["pairs"]],
        }
    validation = {
        "passed": all(checks.values()), "checks": checks, "scenarios": summary,
        "scope": "three_saved_states_with_oracle_visible_masks_not_real_perception_accuracy",
    }
    (output_root / "probe_validation.json").write_text(json.dumps(validation, indent=2))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("runs/robocasa_graspable"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/occlusion_relations/robocasa_graspable"))
    parser.add_argument("--camera", choices=["inspection", "overhead"], default="inspection")
    args = parser.parse_args()
    result = probe(args.scene_dir, args.output_dir, args.camera)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
