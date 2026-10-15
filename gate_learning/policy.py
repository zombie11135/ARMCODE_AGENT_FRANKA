"""Inference wrapper connecting raw multimodal input to the Gate network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from gate_learning.encoders import FrozenImageTextEncoder
from gate_learning.input_adapter import ACTION_NAMES, MultimodalGateInput
from gate_learning.model import HybridGateConfig, HybridMultimodalGate


@dataclass(frozen=True)
class MultimodalGatePrediction:
    action: str
    action_values: Dict[str, float]
    action_mask: Dict[str, bool]
    auxiliary: Dict[str, Any]


def _depth_tensor(depth_m: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.asarray(depth_m, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


class MultimodalGatePolicy:
    """Run a trained fusion head with a frozen image/text encoder."""

    def __init__(
        self,
        network: HybridMultimodalGate,
        encoder: FrozenImageTextEncoder,
        *,
        device: str = "cpu",
        trained_targets: Sequence[str] = (),
    ) -> None:
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.encoder = encoder
        self.trained_targets = frozenset(str(name) for name in trained_targets)
        if int(encoder.feature_dim) != network.config.image_feature_dim:
            raise ValueError("encoder image dimension does not match network checkpoint")
        if int(encoder.feature_dim) != network.config.text_feature_dim:
            raise ValueError("encoder text dimension does not match network checkpoint")

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        encoder: FrozenImageTextEncoder,
        *,
        device: str = "cpu",
    ) -> "MultimodalGatePolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != "multimodal-gate-checkpoint-v1":
            raise ValueError("unsupported multimodal Gate checkpoint schema")
        config = HybridGateConfig(**checkpoint["model_config"])
        network = HybridMultimodalGate(config)
        network.load_state_dict(checkpoint["state_dict"], strict=True)
        expected_encoder = dict(checkpoint.get("encoder") or {})
        actual_encoder = encoder.metadata()
        for key in ("backend", "feature_dim", "checkpoint_sha256"):
            if key in expected_encoder and expected_encoder[key] != actual_encoder.get(key):
                raise ValueError(f"encoder provenance mismatch for {key}")
        trained_targets = (checkpoint.get("training") or {}).get("trained_targets", ())
        return cls(network, encoder, device=device, trained_targets=trained_targets)

    def _encode_history(self, frames: Sequence[np.ndarray]) -> Optional[Tensor]:
        if not frames:
            return None
        return self.encoder.encode_images(frames[-self.network.config.max_history :]).unsqueeze(0).to(
            self.device
        )

    @torch.inference_mode()
    def predict(self, item: MultimodalGateInput) -> MultimodalGatePrediction:
        item.validate()
        mask = torch.from_numpy(item.action_mask.astype(np.bool_)).unsqueeze(0).to(self.device)
        if not bool(mask.any()):
            raise ValueError("all Gate actions are unavailable")
        image = self.encoder.encode_images([item.rgb]).to(self.device)
        text = self.encoder.encode_texts([item.task_text]).to(self.device)
        history = self._encode_history(item.history_rgb)
        history_valid = None
        if history is not None:
            history_valid = torch.ones(history.shape[:2], dtype=torch.bool, device=self.device)
        outputs = self.network(
            image_features=image,
            text_features=text,
            depth_m=_depth_tensor(item.depth_m, self.device),
            state=torch.from_numpy(item.state_vector).unsqueeze(0).to(self.device),
            action_mask=mask,
            history_features=history,
            history_valid=history_valid,
        )
        index = int(self.network.select_actions(outputs, mask)[0])
        values = outputs["action_values"][0].cpu().tolist()
        probe_values = outputs["probe_gain"][0].cpu().tolist()
        feasibility_values = outputs["feasibility_logits"][0].sigmoid().cpu().tolist()
        risk_values = outputs["risk_logits"][0].sigmoid().cpu().tolist()
        return MultimodalGatePrediction(
            action=ACTION_NAMES[index],
            action_values={name: float(value) for name, value in zip(ACTION_NAMES, values)},
            action_mask={name: bool(value) for name, value in zip(ACTION_NAMES, item.action_mask)},
            auxiliary={
                "visibility_probability": {
                    "value": float(outputs["visibility_logit"][0].sigmoid().cpu())
                    if "visibility" in self.trained_targets else None,
                    "valid": "visibility" in self.trained_targets,
                },
                "target_confirmed_probability": {
                    "value": float(outputs["target_confirmed_logit"][0].sigmoid().cpu())
                    if "target_confirmed" in self.trained_targets else None,
                    "valid": "target_confirmed" in self.trained_targets,
                },
                "probe_gain": {
                    "values": {"visual": float(probe_values[0]), "manipulation": float(probe_values[1])}
                    if "probe_gain" in self.trained_targets else None,
                    "valid": "probe_gain" in self.trained_targets,
                },
                "feasibility_probability": {
                    "values": {name: float(value) for name, value in zip(ACTION_NAMES, feasibility_values)}
                    if "feasibility" in self.trained_targets else None,
                    "valid": "feasibility" in self.trained_targets,
                },
                "risk_probability": {
                    "values": {name: float(value) for name, value in zip(ACTION_NAMES, risk_values)}
                    if "risk" in self.trained_targets else None,
                    "valid": "risk" in self.trained_targets,
                },
            },
        )


def save_checkpoint(
    path: str,
    network: HybridMultimodalGate,
    *,
    encoder_metadata: Dict[str, Any],
    training_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save weights plus exact encoder/model provenance."""

    torch.save(
        {
            "schema_version": "multimodal-gate-checkpoint-v1",
            "state_dict": network.state_dict(),
            "model_config": network.config.to_dict(),
            "encoder": dict(encoder_metadata),
            "training": dict(training_metadata or {}),
        },
        path,
    )
