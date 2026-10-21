"""Anonymous visible-mask adapter and separate evaluator for saved RoboCasa views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np

from occlusion_reasoning.scene import infer_scene
from occlusion_reasoning.observations import build_observation
from .probe_relations import visible_instances

CAMERAS = ("left_oblique", "robot0_eye_in_hand")


def geometry_groups(root):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str((Path(root) / "evaluation/scene.xml").resolve()))
    groups = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        match = re.match(r"^(carton_\d+|blocker_small|blocker_large)_", name)
        if match:
            groups.setdefault(match[1], []).append(geom_id)
    return groups


def load_view(root, camera, groups=None):
    root = Path(root)
    groups = geometry_groups(root) if groups is None else groups
    seg = np.load(root / f"evaluation/{camera}_segmentation.npy", allow_pickle=False)
    instances, evaluator_ids = visible_instances(seg, groups)
    depth = np.load(root / f"observation/{camera}_depth_m.npy", allow_pickle=False)
    meta = json.loads((root / f"observation/{camera}_camera.json").read_text())
    k = np.asarray(meta["K"])
    packet, arrays = build_observation(
        depth_m=depth, intrinsics=dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2]),
        instances=instances, frame_id=camera, query="unspecified_target",
        alignment_status="recorded_aligned", timestamp_s=meta["capture_timestamp_s"],
        camera_id=meta["camera_id"], capture_group_id=meta["capture_group_id"],
        timestamp_domain=meta["timestamp_domain"],
        reference_frame="simulation_world",
        T_reference_from_camera_optical=meta["T_world_from_camera_optical"],
    )
    packet["provenance"] = {
        "backend": "simulation_visible_segmentation_oracle",
        "object_poses_used": False, "hidden_geometry_used": False, "target_identity_used": False,
    }
    names = {node["node_id"]: evaluator_ids[node["source_ids"][0]]
             for node in packet["relation_graph"]["nodes"]}
    return (packet, arrays), names


def annotate(root, observations, result, output):
    images = []
    for packet, arrays in observations:
        camera = packet["frame"]["camera_id"]
        image = cv2.imread(str(Path(root) / f"observation/{camera}_rgb.png"))
        if image is None:
            raise FileNotFoundError(camera)
        for node in packet["relation_graph"]["nodes"]:
            mask = np.asarray(arrays[node["raw_mask_key"]], np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (80, 255, 80), 2)
            y, x = np.nonzero(mask)
            if len(x):
                entity = result["node_to_entity"][node["node_id"]]
                cv2.putText(image, entity, (int(x.mean()), int(y.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for edge in result["relation_graph"]["edges"]:
            if edge["view_id"] == camera:
                color = (255, 0, 255) if edge["relation"] == "occludes_observed_surface" else (0, 165, 255)
                for x, y in edge["support_pixels_sample_xy"]:
                    cv2.circle(image, (x, y), 2, color, -1)
        image = cv2.copyMakeBorder(image, 55, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(image, camera + " | magenta: measured; orange: planar prior",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(image)
    cv2.imwrite(str(output), np.concatenate(images, axis=1))


def save_result(root, observations, result, output, *, save_packets=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "multiview_relations.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if save_packets:
        for packet, arrays in observations:
            directory = output / packet["frame"]["camera_id"]
            directory.mkdir(exist_ok=True)
            packet = {**packet, "artifacts": {"arrays": "observation_arrays.npz"}}
            (directory / "scene_observation.json").write_text(json.dumps(packet, indent=2, allow_nan=False))
            np.savez_compressed(directory / "observation_arrays.npz", **arrays)
    annotate(root, observations, result, output / "evidence.png")


def evaluate_names(result, names):
    # This function is deliberately downstream of inference.
    entity_names = {e["entity_id"]: {names[n] for n in e["members"]} for e in result["entities"]}
    edges = []
    for edge in result["relation_graph"]["edges"]:
        edges.append({
            "foreground": sorted(entity_names[edge["source"]]),
            "background": sorted(entity_names[edge["target"]]),
            "view": edge["view_id"], "pixels": edge["unique_support_pixels"],
            "relation": edge["relation"],
            "evidence": edge["evidence"],
            "support_fraction": edge.get("fraction_of_projected_observed_surface",
                                        edge.get("fraction_of_completed_patch")),
            "median_depth_gap_m": edge["depth_gap_median_m"],
        })
    return {
        "association_count": sum(a["status"] == "accepted" for a in result["associations"]),
        "incorrect_associations": [sorted(v) for v in entity_names.values() if len(v) != 1],
        "edges": edges, "unresolved_count": len(result["unresolved_front_surfaces"]),
    }


def probe(scene_dir, output_dir):
    results = {}
    for scenario in ("clear", "label_occluded", "fully_occluded"):
        root = Path(scene_dir) / scenario
        groups = geometry_groups(root)
        loaded = [load_view(root, camera, groups) for camera in CAMERAS]
        observations = [x[0] for x in loaded]
        names = {k: v for x in loaded for k, v in x[1].items()}
        result = infer_scene(observations)
        save_result(root, observations, result, Path(output_dir) / scenario)
        results[scenario] = evaluate_names(result, names)
    Path(output_dir, "evaluation.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("runs/robocasa_graspable"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/occlusion_multiview/initial"))
    args = parser.parse_args()
    if args.scene_dir.resolve() == args.output_dir.resolve():
        parser.error("output must differ from input")
    print(json.dumps(probe(args.scene_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
