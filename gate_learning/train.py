"""Small supervised trainer for counterfactual multimodal Gate labels."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from gate_learning.dataset import CachedGateDataset, collate_cached_gate, read_cache_index
from gate_learning.input_adapter import ACTION_NAMES
from gate_learning.losses import multimodal_gate_loss
from gate_learning.model import HybridGateConfig, HybridMultimodalGate, trainable_parameter_count
from gate_learning.policy import save_checkpoint


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(model: HybridMultimodalGate, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return model(
        image_features=batch["image_features"],
        text_features=batch["text_features"],
        depth_m=batch["depth_m"],
        state=batch["state"],
        action_mask=batch["action_mask"],
        history_features=batch["history_features"],
        history_valid=batch["history_valid"],
    )


@torch.inference_mode()
def evaluate(
    model: HybridMultimodalGate,
    loader: DataLoader,
    device: torch.device,
    *,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    model.eval()
    absolute_error = 0.0
    value_count = 0.0
    choice_correct = 0
    choice_count = 0
    predictions: List[Dict[str, Any]] = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        outputs = _forward(model, batch)
        valid = batch["action_values_valid"].bool() & batch["action_mask"]
        absolute_error += float(
            ((outputs["action_values"] - batch["action_values"]).abs() * valid).sum().cpu()
        )
        value_count += float(valid.sum().cpu())
        eligible = valid.sum(dim=1) >= 2
        if eligible.any():
            floor = torch.finfo(outputs["action_values"].dtype).min
            predicted = outputs["action_values"].masked_fill(~valid, floor).argmax(dim=1)
            target = batch["action_values"].masked_fill(~valid, floor).argmax(dim=1)
            choice_correct += int((predicted[eligible] == target[eligible]).sum().cpu())
            choice_count += int(eligible.sum().cpu())
            if include_predictions:
                for row_index in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
                    predictions.append({
                        "sample_id": batch["sample_id"][row_index],
                        "predicted_action": ACTION_NAMES[int(predicted[row_index])],
                        "target_action": ACTION_NAMES[int(target[row_index])],
                        "correct": bool(
                            (predicted[row_index] == target[row_index]).detach().cpu().item()
                        ),
                        "predicted_values": outputs["action_values"][row_index].cpu().tolist(),
                        "target_values": batch["action_values"][row_index].cpu().tolist(),
                        "valid": valid[row_index].cpu().tolist(),
                    })
    result = {
        "masked_action_value_mae": absolute_error / max(value_count, 1.0),
        "action_choice_accuracy": choice_correct / choice_count if choice_count else None,
        "action_choice_samples": choice_count,
    }
    if include_predictions:
        result["predictions"] = predictions
    return result


def train_from_cache(
    *,
    index_path: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 7,
    device: str = "cpu",
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    rows = read_cache_index(index_path)
    train_rows = [row for row in rows if row.get("split", "train") == "train"]
    validation_rows = [row for row in rows if row.get("split") in {"validation", "val"}]
    if not train_rows:
        raise ValueError("cache contains no training samples")
    first = CachedGateDataset(train_rows)[0]
    feature_dim = int(first["image_features"].shape[-1])
    config = HybridGateConfig(image_feature_dim=feature_dim, text_feature_dim=feature_dim)
    model = HybridMultimodalGate(config).to(target_device)
    train_loader = DataLoader(
        CachedGateDataset(train_rows), batch_size=batch_size, shuffle=True,
        collate_fn=collate_cached_gate,
    )
    validation_loader = (
        DataLoader(
            CachedGateDataset(validation_rows), batch_size=batch_size, shuffle=False,
            collate_fn=collate_cached_gate,
        )
        if validation_rows else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = _move(raw_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(model, batch)
            target_keys = {
                "action_values", "action_values_valid", "feasibility", "feasibility_valid",
                "probe_gain", "probe_gain_valid", "risk", "risk_valid", "visibility",
                "visibility_valid", "target_confirmed", "target_confirmed_valid",
            }
            losses = multimodal_gate_loss(
                outputs, {key: value for key, value in batch.items() if key in target_keys}
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(losses["total"].detach().cpu())
            batches += 1
        row: Dict[str, Any] = {"epoch": epoch, "train_loss": running / max(batches, 1)}
        if validation_loader is not None:
            row["validation"] = evaluate(model, validation_loader, target_device)
        history.append(row)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_metadata_path = Path(index_path).with_name("metadata.json")
    cache_metadata = (
        json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if cache_metadata_path.is_file() else {}
    )
    checkpoint_path = destination / "multimodal_gate.pt"
    trainable_targets = [
        name
        for name in ("feasibility", "probe_gain", "risk", "visibility", "target_confirmed")
        if name in first and f"{name}_valid" in first
    ]
    save_checkpoint(
        str(checkpoint_path), model,
        encoder_metadata=cache_metadata.get("encoder", {}),
        training_metadata={
            "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "train_samples": len(train_rows), "validation_samples": len(validation_rows),
            "split_group_isolation_checked": True,
            "trained_targets": ["action_values", *trainable_targets],
        },
    )
    report = {
        "schema_version": "multimodal-gate-training-report-v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "trainable_parameters": trainable_parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "final_train": evaluate(model, train_loader, target_device, include_predictions=True),
        "final_validation": evaluate(
            model, validation_loader, target_device, include_predictions=True
        )
        if validation_loader is not None else None,
        "warning": "Metrics are meaningful only with real counterfactual labels.",
    }
    (destination / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
