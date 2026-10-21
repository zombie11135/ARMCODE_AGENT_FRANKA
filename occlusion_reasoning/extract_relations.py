"""CLI and review visualization for single-frame relation evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .relations import RelationConfig, extract_relations


def visualize_relations(packet, arrays, result, color_bgr=None):
    height, width = arrays["depth_m"].shape
    if color_bgr is None:
        canvas = np.full((height, width, 3), 235, np.uint8)
    else:
        if color_bgr.shape != (height, width, 3):
            raise ValueError("RGB shape does not match observation")
        canvas = color_bgr.copy()
    centers = {}
    names = {node["node_id"]: f"N{i}" for i, node in enumerate(packet["relation_graph"]["nodes"])}
    for index, node in enumerate(packet["relation_graph"]["nodes"]):
        mask = np.asarray(arrays[node["raw_mask_key"]], bool)
        if not mask.any():
            continue
        color = np.array([(47 + 83 * index) % 220, (157 + 59 * index) % 220,
                          (211 + 97 * index) % 220], float)
        canvas[mask] = .7 * canvas[mask] + .3 * color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(x) for x in color), 2)
        y, x = np.nonzero(mask)
        center = (int(np.median(x)), int(np.median(y)))
        centers[node["node_id"]] = center
        cv2.putText(canvas, names[node["node_id"]], center, cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, names[node["node_id"]], center, cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 1, cv2.LINE_AA)
    active = [p for p in result["pairs"] if p["status"] != "no_near_boundary_evidence"]
    for pair in active:
        for sample in pair["samples"]:
            for key in ("a_xy", "b_xy"):
                cv2.circle(canvas, tuple(sample[key]), 1, (255, 0, 255), -1)
        for edge in pair["edges"]:
            if edge["relation"] == "possible_occludes":
                cv2.arrowedLine(canvas, centers[edge["source"]], centers[edge["target"]],
                                (0, 120, 255), 2, cv2.LINE_AA, tipLength=.15)
    panel_width = 540
    out_height = max(height, 160 + 60 * min(18, len(active)))
    output = np.full((out_height, width + panel_width, 3), 245, np.uint8)
    output[:height, :width] = canvas
    lines = [
        "Single-frame relation evidence",
        "Orange arrow: possible occlusion (not confirmed)",
        "Magenta: sampled neighboring boundaries",
        "Scores: heuristic; not probabilities",
    ]
    for pair in active[:18]:
        delta = pair["depth"]["median_b_minus_a_m"]
        delta_text = "unknown" if delta is None else f"{delta * 1000:+.1f} mm"
        lines.extend([
            f"{names[pair['a']]} / {names[pair['b']]}: {pair['status']}",
            f"  depth B-A: {delta_text}; valid pairs: {pair['depth']['valid_pair_count']}",
        ])
    counts = Counter(p["status"] for p in result["pairs"])
    if not active:
        lines.append("No neighboring-boundary evidence in this frame.")
    if len(active) > 18:
        lines.append("Additional pairs are listed in relations.json.")
    lines.append(f"Pairs: {len(result['pairs'])}; distant: {counts['no_near_boundary_evidence']}")
    for index, line in enumerate(lines):
        cv2.putText(output, line, (width + 14, 28 + index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (35, 35, 35), 1, cv2.LINE_AA)
    return output


def write_result(packet, arrays, result, output_dir, color_bgr=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    cv2.imwrite(str(output_dir / "relations_overlay.png"),
                visualize_relations(packet, arrays, result, color_bgr))


def export_relations(observation_path, output_dir, config=None, rgb_path=None):
    observation_path = Path(observation_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == observation_path.parent:
        raise ValueError("use a separate output directory for relations")
    packet = json.loads(observation_path.read_text())
    array_path = observation_path.parent / packet["artifacts"]["arrays"]
    with np.load(array_path, allow_pickle=False) as stored:
        arrays = dict(stored)
    result = extract_relations(packet, arrays, config)
    result["input"] = {
        "observation": str(observation_path),
        "observation_sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
    }
    color = None
    if rgb_path is None and packet.get("provenance", {}).get("snapshot_dir"):
        candidate = Path(packet["provenance"]["snapshot_dir"]) / "color.png"
        rgb_path = candidate if candidate.is_file() else None
    if rgb_path is not None:
        rgb_path = Path(rgb_path)
        expected = packet.get("provenance", {}).get("rgb_sha256")
        if expected and hashlib.sha256(rgb_path.read_bytes()).hexdigest() != expected:
            raise ValueError("RGB hash differs from observation capture")
        color = cv2.imread(str(rgb_path))
        if color is None:
            raise FileNotFoundError(rgb_path)
    write_result(packet, arrays, result, output_dir, color)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--neighbor-radius-px", type=float, default=6.0)
    parser.add_argument("--depth-threshold-m", type=float, default=.008)
    parser.add_argument("--no-convex-prior", action="store_true")
    args = parser.parse_args()
    result = export_relations(
        args.observation, args.output_dir,
        RelationConfig(neighbor_radius_px=args.neighbor_radius_px,
                       min_depth_delta_m=args.depth_threshold_m,
                       convex_completion_prior=not args.no_convex_prior),
        args.rgb,
    )
    print(json.dumps({"output": str(args.output_dir.resolve()),
                      "pair_statuses": dict(Counter(p["status"] for p in result["pairs"])),
                      "edge_count": len(result["relation_graph"]["edges"])}, indent=2))


if __name__ == "__main__":
    main()
