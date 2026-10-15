"""Frozen image/text encoder backends for the multimodal Gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class FrozenImageTextEncoder(Protocol):
    feature_dim: int

    def encode_images(self, images: Sequence[np.ndarray]) -> Tensor: ...

    def encode_texts(self, texts: Sequence[str]) -> Tensor: ...

    def metadata(self) -> Dict[str, Any]: ...


def _pil_rgb(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class MobileCLIPFrozenEncoder:
    """Use the project's existing MobileCLIP-B checkpoint without fine tuning.

    This is the immediately runnable baseline.  It returns global embeddings;
    its Chinese text discrimination must be benchmarked before deployment.
    """

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        import mobileclip

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = torch.device(device)
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            "mobileclip_b", pretrained=str(self.checkpoint), device=str(self.device)
        )
        self.tokenizer = mobileclip.get_tokenizer("mobileclip_b")
        self.model.eval()
        self.model.requires_grad_(False)
        self.feature_dim = int(self.model.projection_dim)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[np.ndarray]) -> Tensor:
        batch = torch.stack([self.preprocess(_pil_rgb(image)) for image in images]).to(self.device)
        encoded = self.model.encode_image(batch)
        return torch.nn.functional.normalize(encoded.float(), dim=-1).cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        tokens = self.tokenizer([str(text) for text in texts]).to(self.device)
        encoded = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(encoded.float(), dim=-1).cpu()

    def metadata(self) -> Dict[str, Any]:
        import hashlib

        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "backend": "mobileclip_b_global",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": digest.hexdigest(),
            "feature_dim": self.feature_dim,
            "loaded_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "frozen": True,
            "spatial_tokens": False,
        }


class SigLIP2FrozenEncoder:
    """Optional Hugging Face SigLIP2 backend.

    No model is downloaded implicitly when ``local_files_only=True``.  This
    backend is included so a vetted local multilingual checkpoint can replace
    MobileCLIP without changing the fusion model or dataset schema.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        *,
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.feature_dim = int(getattr(self.model.config, "projection_dim"))

    @torch.inference_mode()
    def encode_images(self, images: Sequence[np.ndarray]) -> Tensor:
        inputs = self.processor(images=[_pil_rgb(image) for image in images], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        encoded = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(encoded.float(), dim=-1).cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        inputs = self.processor(
            text=[str(text) for text in texts], padding="max_length", return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        encoded = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(encoded.float(), dim=-1).cpu()

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "siglip2_global",
            "model_name_or_path": self.model_name_or_path,
            "feature_dim": self.feature_dim,
            "loaded_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "frozen": True,
            "spatial_tokens": False,
        }


def create_frozen_encoder(
    backend: str,
    *,
    checkpoint: str,
    device: str = "cpu",
    local_files_only: bool = True,
) -> FrozenImageTextEncoder:
    backend = str(backend).strip().lower()
    if backend == "mobileclip":
        return MobileCLIPFrozenEncoder(checkpoint, device=device)
    if backend == "siglip2":
        return SigLIP2FrozenEncoder(
            checkpoint, device=device, local_files_only=local_files_only
        )
    raise ValueError(f"unknown encoder backend: {backend!r}")


def cosine_similarity_matrix(features: Tensor) -> Tensor:
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features @ features.transpose(0, 1)
