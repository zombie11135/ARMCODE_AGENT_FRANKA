"""Lightweight fusion network for multimodal Gate action values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from gate_learning.input_adapter import ACTION_NAMES, MULTIMODAL_STATE_FEATURE_NAMES


@dataclass(frozen=True)
class HybridGateConfig:
    image_feature_dim: int = 512
    text_feature_dim: int = 512
    state_dim: int = len(MULTIMODAL_STATE_FEATURE_NAMES)
    model_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 256
    depth_tokens_per_side: int = 4
    max_history: int = 4
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DepthPatchEncoder(nn.Module):
    """Encode metric depth into spatial tokens instead of a fake RGB channel."""

    def __init__(self, model_dim: int, tokens_per_side: int = 4) -> None:
        super().__init__()
        self.tokens_per_side = int(tokens_per_side)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, model_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.tokens_per_side, self.tokens_per_side)),
        )

    def forward(self, depth_m: Tensor) -> Tensor:
        if depth_m.ndim != 4 or depth_m.shape[1] != 1:
            raise ValueError("depth_m must have shape [B,1,H,W]")
        valid = torch.isfinite(depth_m) & (depth_m > 0)
        cleaned = torch.where(valid, depth_m, torch.zeros_like(depth_m))
        # Compress long tails while keeping metres and a separate validity map.
        cleaned = torch.log1p(cleaned.clamp(min=0.0, max=5.0))
        encoded = self.net(torch.cat((cleaned, valid.to(depth_m.dtype)), dim=1))
        return encoded.flatten(2).transpose(1, 2)


class HybridMultimodalGate(nn.Module):
    """Fuse image/task/depth/state/history and predict masked action values.

    The frozen visual-language encoder is intentionally outside this module.
    ``image_features`` may be one global token ``[B,D]`` (MobileCLIP baseline)
    or spatial patch tokens ``[B,N,D]`` (future SigLIP2 backend).
    """

    def __init__(self, config: HybridGateConfig = HybridGateConfig()) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.image_projection = nn.Linear(config.image_feature_dim, dim)
        self.text_projection = nn.Linear(config.text_feature_dim, dim)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.history_projection = nn.Linear(config.image_feature_dim, dim)
        self.depth_encoder = DepthPatchEncoder(dim, config.depth_tokens_per_side)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # cls, image, text, depth, state, history
        self.type_embedding = nn.Embedding(6, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.final_norm = nn.LayerNorm(dim)
        self.action_value_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        self.feasibility_head = nn.Linear(dim, 4)
        self.visibility_head = nn.Linear(dim, 1)
        self.target_confirmed_head = nn.Linear(dim, 1)
        self.probe_gain_head = nn.Linear(dim, 2)
        self.risk_head = nn.Linear(dim, 4)
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def _ensure_tokens(features: Tensor, name: str) -> Tensor:
        if features.ndim == 2:
            return features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(f"{name} must have shape [B,D] or [B,N,D]")
        return features

    def forward(
        self,
        *,
        image_features: Tensor,
        text_features: Tensor,
        depth_m: Tensor,
        state: Tensor,
        action_mask: Tensor,
        history_features: Optional[Tensor] = None,
        history_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        image_features = self._ensure_tokens(image_features, "image_features")
        text_features = self._ensure_tokens(text_features, "text_features")
        batch = image_features.shape[0]
        if state.shape != (batch, self.config.state_dim):
            raise ValueError(f"state must have shape [B,{self.config.state_dim}]")
        if action_mask.shape != (batch, len(ACTION_NAMES)):
            raise ValueError("action_mask must have shape [B,4]")

        tokens = []
        padding = []

        def append(value: Tensor, type_index: int, valid: Optional[Tensor] = None) -> None:
            value = value + self.type_embedding.weight[type_index].view(1, 1, -1)
            tokens.append(value)
            if valid is None:
                padding.append(torch.zeros(value.shape[:2], dtype=torch.bool, device=value.device))
            else:
                padding.append(~valid.to(dtype=torch.bool, device=value.device))

        append(self.cls_token.expand(batch, -1, -1), 0)
        append(self.image_projection(image_features), 1)
        append(self.text_projection(text_features), 2)
        append(self.depth_encoder(depth_m), 3)
        append(self.state_projection(state).unsqueeze(1), 4)
        if history_features is not None:
            history_features = self._ensure_tokens(history_features, "history_features")
            if history_features.shape[1] > self.config.max_history:
                history_features = history_features[:, -self.config.max_history :]
                if history_valid is not None:
                    history_valid = history_valid[:, -self.config.max_history :]
            if history_valid is None:
                history_valid = torch.ones(
                    history_features.shape[:2], dtype=torch.bool, device=history_features.device
                )
            append(self.history_projection(history_features), 5, history_valid)

        fused_tokens = torch.cat(tokens, dim=1)
        key_padding_mask = torch.cat(padding, dim=1)
        fused = self.fusion(fused_tokens, src_key_padding_mask=key_padding_mask)
        decision_token = self.final_norm(fused[:, 0])
        raw_values = self.action_value_head(decision_token)
        mask = action_mask.to(dtype=torch.bool, device=raw_values.device)
        masked_values = raw_values.masked_fill(~mask, torch.finfo(raw_values.dtype).min)
        return {
            "action_values": raw_values,
            "masked_action_values": masked_values,
            "feasibility_logits": self.feasibility_head(decision_token),
            "visibility_logit": self.visibility_head(decision_token).squeeze(-1),
            "target_confirmed_logit": self.target_confirmed_head(decision_token).squeeze(-1),
            "probe_gain": self.probe_gain_head(decision_token),
            "risk_logits": self.risk_head(decision_token),
            "decision_embedding": decision_token,
        }

    @torch.no_grad()
    def select_actions(self, outputs: Dict[str, Tensor], action_mask: Tensor) -> Tensor:
        if not torch.all(action_mask.to(torch.bool).any(dim=1)):
            raise ValueError("cannot select an action when all actions are masked")
        return outputs["masked_action_values"].argmax(dim=-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
