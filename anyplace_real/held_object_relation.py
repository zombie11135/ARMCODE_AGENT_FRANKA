"""Estimate a per-grasp TCP-to-object relation from synchronized poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import load_json, rotation_checks, write_json


def average_tcp_object_relations(relations):
    relations = [np.asarray(value, dtype=np.float64) for value in relations]
    if len(relations) < 3:
        raise ValueError("at least three synchronized pose samples are required")
    if any(value.shape != (4, 4) for value in relations):
        raise ValueError("every relation must be 4x4")
    translations = np.asarray([value[:3, 3] for value in relations])
    rotations = Rotation.from_matrix(np.asarray([value[:3, :3] for value in relations]))
    mean_rotation = rotations.mean()
    mean_translation = np.mean(translations, axis=0)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = mean_rotation.as_matrix()
    result[:3, 3] = mean_translation
    translation_errors = np.linalg.norm(translations - mean_translation[None, :], axis=1)
    rotation_errors = (mean_rotation.inv() * rotations).magnitude()
    return result, {
        "translation_errors_m": translation_errors,
        "rotation_errors_deg": np.degrees(rotation_errors),
        "translation_rms_m": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rms_deg": float(np.sqrt(np.mean(np.degrees(rotation_errors) ** 2))),
        "rotation_max_deg": float(np.max(np.degrees(rotation_errors))),
    }


def run(manifest_path: Path, output_dir: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("example_only", False):
        raise ValueError("example_only=true: replace poses and set it to false")
    if manifest.get("transform_semantics") != "T_A_B_maps_B_coordinates_into_A":
        raise ValueError("transform_semantics must be explicitly confirmed")
    samples = manifest.get("samples") or []
    relations = []
    rows = []
    for index, sample in enumerate(samples):
        base_tcp = np.asarray(sample["T_base_tcp"], dtype=np.float64)
        base_object = np.asarray(sample["T_base_object"], dtype=np.float64)
        if base_tcp.shape != (4, 4) or base_object.shape != (4, 4):
            raise ValueError("sample transforms must be 4x4")
        relation = np.linalg.inv(base_tcp) @ base_object
        relations.append(relation)
        rows.append(
            {
                "index": index,
                "source": sample.get("source"),
                "T_tcp_object": relation.tolist(),
            }
        )
    mean, scatter = average_tcp_object_relations(relations)
    thresholds = manifest.get("acceptance", {})
    translation_rms_limit = float(thresholds.get("maximum_translation_rms_m", 0.003))
    rotation_rms_limit = float(thresholds.get("maximum_rotation_rms_deg", 3.0))
    passed = bool(
        scatter["translation_rms_m"] <= translation_rms_limit
        and scatter["rotation_rms_deg"] <= rotation_rms_limit
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "T_tcp_object_candidate.npy"
    np.save(str(candidate_path), mean)
    accepted_path = None
    if passed:
        accepted_path = output_dir / "T_tcp_object.npy"
        np.save(str(accepted_path), mean)
    report = {
        "status": "accepted" if passed else "rejected_scatter_too_large",
        "offline_only": True,
        "robot_connection_attempted": False,
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "formula": "T_tcp_object = inverse(T_base_tcp) @ T_base_object",
        "T_tcp_object_candidate": mean.tolist(),
        "rotation_check": rotation_checks(mean),
        "scatter": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in scatter.items()
        },
        "acceptance": {
            "maximum_translation_rms_m": translation_rms_limit,
            "maximum_rotation_rms_deg": rotation_rms_limit,
            "passed": passed,
        },
        "samples": rows,
        "outputs": {
            "candidate": str(candidate_path),
            "accepted": None if accepted_path is None else str(accepted_path),
        },
        "warning": "This relation is valid only for this grasp. Re-estimate after every grasp or detected slip.",
    }
    write_json(output_dir / "held_object_relation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate T_tcp_object from synchronized saved poses")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
