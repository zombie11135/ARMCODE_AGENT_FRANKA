"""Masked objectives for counterfactual Gate supervision."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    valid = valid.to(dtype=values.dtype, device=values.device)
    denominator = valid.sum().clamp_min(1.0)
    return (values * valid).sum() / denominator


def multimodal_gate_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    auxiliary_weight: float = 0.2,
) -> Dict[str, Tensor]:
    """Compute losses while leaving every unexecuted/unknown label unused."""

    q_error = F.smooth_l1_loss(
        outputs["action_values"], targets["action_values"], reduction="none"
    )
    q_loss = _masked_mean(q_error, targets["action_values_valid"])
    components: Dict[str, Tensor] = {"action_value": q_loss}

    optional = (
        ("feasibility", "feasibility_logits", "feasibility", "feasibility_valid", "bce"),
        ("visibility", "visibility_logit", "visibility", "visibility_valid", "bce"),
        (
            "target_confirmed",
            "target_confirmed_logit",
            "target_confirmed",
            "target_confirmed_valid",
            "bce",
        ),
        ("probe_gain", "probe_gain", "probe_gain", "probe_gain_valid", "huber"),
        ("risk", "risk_logits", "risk", "risk_valid", "bce"),
    )
    for name, output_key, target_key, valid_key, kind in optional:
        if target_key not in targets or valid_key not in targets:
            continue
        if kind == "bce":
            error = F.binary_cross_entropy_with_logits(
                outputs[output_key], targets[target_key], reduction="none"
            )
        else:
            error = F.smooth_l1_loss(outputs[output_key], targets[target_key], reduction="none")
        components[name] = _masked_mean(error, targets[valid_key])

    auxiliary = sum(value for key, value in components.items() if key != "action_value")
    total = q_loss + float(auxiliary_weight) * auxiliary
    return {"total": total, **components}
