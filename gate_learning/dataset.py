"""Torch dataset for frozen multimodal Gate feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_cache_index(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("cache index is empty")
    group_splits: Dict[str, str] = {}
    for row in rows:
        group = str(row["split_group"])
        split = str(row.get("split", "train"))
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(f"split_group {group!r} appears in both {previous!r} and {split!r}")
    return rows


class CachedGateDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        with np.load(row["cache"], allow_pickle=False) as values:
            item = {key: values[key].copy() for key in values.files}
        item["sample_id"] = row["sample_id"]
        return item


def collate_cached_gate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_history = max(item["history_features"].shape[0] for item in batch)
    feature_dim = batch[0]["image_features"].shape[-1]
    history = np.zeros((len(batch), max_history, feature_dim), dtype=np.float32)
    history_valid = np.zeros((len(batch), max_history), dtype=np.bool_)
    for row, item in enumerate(batch):
        count = item["history_features"].shape[0]
        if count:
            history[row, :count] = item["history_features"]
            history_valid[row, :count] = True

    depths = []
    for item in batch:
        value = torch.from_numpy(item["depth_m"]).float().unsqueeze(0).unsqueeze(0)
        depths.append(F.interpolate(value, size=(144, 192), mode="nearest")[0])
    collated = {
        "sample_id": [item["sample_id"] for item in batch],
        "image_features": torch.from_numpy(
            np.stack([item["image_features"] for item in batch])
        ).float(),
        "text_features": torch.from_numpy(
            np.stack([item["text_features"] for item in batch])
        ).float(),
        "history_features": torch.from_numpy(history).float() if max_history else None,
        "history_valid": torch.from_numpy(history_valid) if max_history else None,
        "depth_m": torch.stack(depths),
        "state": torch.from_numpy(np.stack([item["state"] for item in batch])).float(),
        "action_mask": torch.from_numpy(
            np.stack([item["action_mask"] for item in batch])
        ).bool(),
        "action_values": torch.from_numpy(
            np.stack([item["action_values"] for item in batch])
        ).float(),
        "action_values_valid": torch.from_numpy(
            np.stack([item["action_values_valid"] for item in batch])
        ).float(),
    }
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
    ):
        if all(key in item for item in batch):
            collated[key] = torch.from_numpy(np.stack([item[key] for item in batch])).float()
    return collated
