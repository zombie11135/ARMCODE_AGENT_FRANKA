"""Precompute frozen embeddings while preserving raw RGB-D references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from gate_learning.encoders import FrozenImageTextEncoder
from gate_learning.input_adapter import ACTION_NAMES, MULTIMODAL_STATE_FEATURE_NAMES, load_rgb_depth


REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "split_group",
    "rgb",
    "depth",
    "task_text",
    "state_vector",
    "action_mask",
    "action_values",
    "action_values_valid",
}


def read_manifest(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_MANIFEST_FIELDS.difference(row)
            if missing:
                raise ValueError(f"manifest line {line_number} missing {sorted(missing)}")
            if len(row["state_vector"]) != len(MULTIMODAL_STATE_FEATURE_NAMES):
                raise ValueError(f"manifest line {line_number} has wrong state dimension")
            for key in ("action_mask", "action_values", "action_values_valid"):
                if len(row[key]) != len(ACTION_NAMES):
                    raise ValueError(f"manifest line {line_number} has wrong {key} dimension")
            forbidden = {"scene_type", "scenario_type", "true_blocker", "evaluator_state"}.intersection(row)
            if forbidden:
                raise ValueError(
                    f"manifest line {line_number} leaks generation/evaluator fields: {sorted(forbidden)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def cache_manifest(
    manifest_path: str,
    output_dir: str,
    encoder: FrozenImageTextEncoder,
) -> Dict[str, Any]:
    rows = read_manifest(manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "index.jsonl"
    cached_rows = []
    for row in rows:
        rgb, depth = load_rgb_depth(row["rgb"], row["depth"])
        image_features = encoder.encode_images([rgb])[0].numpy().astype(np.float32)
        text_features = encoder.encode_texts([row["task_text"]])[0].numpy().astype(np.float32)
        history_paths = list(row.get("history_rgb") or [])[-4:]
        if history_paths:
            from PIL import Image

            history_arrays = [
                np.asarray(Image.open(Path(path)).convert("RGB"), dtype=np.uint8)
                for path in history_paths
            ]
        else:
            history_arrays = []
        if history_arrays:
            history_features = encoder.encode_images(history_arrays).numpy().astype(np.float32)
        else:
            history_features = np.empty((0, encoder.feature_dim), dtype=np.float32)
        safe_id = str(row["sample_id"]).replace("/", "_").replace("\\", "_")
        cache_path = destination / f"{safe_id}.npz"
        np.savez_compressed(
            cache_path,
            image_features=image_features,
            text_features=text_features,
            history_features=history_features,
            depth_m=depth.astype(np.float32),
            state=np.asarray(row["state_vector"], dtype=np.float32),
            action_mask=np.asarray(row["action_mask"], dtype=np.bool_),
            action_values=np.asarray(row["action_values"], dtype=np.float32),
            action_values_valid=np.asarray(row["action_values_valid"], dtype=np.float32),
            **{
                key: np.asarray(row[key], dtype=np.float32)
                for key in (
                    "feasibility",
                    "feasibility_valid",
                    "probe_gain",
                    "probe_gain_valid",
                    "risk",
                    "risk_valid",
                    "visibility",
                    "visibility_valid",
                    "target_confirmed",
                    "target_confirmed_valid",
                )
                if key in row
            },
        )
        cached = {
            "sample_id": row["sample_id"],
            "split": row.get("split", "train"),
            "split_group": row["split_group"],
            "cache": str(cache_path.resolve()),
            "raw_rgb": str(Path(row["rgb"]).resolve()),
            "raw_depth": str(Path(row["depth"]).resolve()),
            "task_text": row["task_text"],
        }
        cached_rows.append(cached)
    with index_path.open("w", encoding="utf-8") as stream:
        for row in cached_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "schema_version": "multimodal-gate-cache-v1",
        "samples": len(cached_rows),
        "index": str(index_path.resolve()),
        "encoder": encoder.metadata(),
        "raw_inputs_retained": True,
    }
    with (destination / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return metadata
