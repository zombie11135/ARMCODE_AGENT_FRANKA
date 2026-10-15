from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def snapshot_identity(snapshot: Path, grasp_input: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if grasp_input is not None:
        value = str(grasp_input.get("query") or "").strip()
        if value:
            return value
    parts = Path(snapshot).name.split("_", 2)
    if len(parts) == 3 and parts[2].strip():
        return parts[2].strip()
    return None


def masked_crop(color_bgr: np.ndarray, mask: np.ndarray, bbox: Sequence[float]) -> Image.Image:
    height, width = color_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid crop bbox")
    crop = color_bgr[y1:y2, x1:x2].copy()
    active = np.asarray(mask, dtype=bool)[y1:y2, x1:x2]
    # Neutral background prevents table colour from becoming an identity shortcut.
    crop[~active] = 127
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class MobileCLIPGallery:
    """Reference-image retrieval used to bind a requested identity to one YOLOE mask."""

    def __init__(self, checkpoint: Path, device: str = "cuda:0") -> None:
        import mobileclip

        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            "mobileclip_b", pretrained=str(self.checkpoint), device=device
        )
        self.model.eval()
        self.labels: List[str] = []
        self.paths: List[str] = []
        self.features: Optional[np.ndarray] = None

    def encode(self, images: Iterable[Image.Image], batch_size: int = 16) -> np.ndarray:
        images = list(images)
        if not images:
            return np.empty((0, 512), np.float32)
        features = []
        started = time.perf_counter()
        for start in range(0, len(images), batch_size):
            tensor = torch.stack([self.preprocess(image) for image in images[start : start + batch_size]])
            tensor = tensor.to(self.device)
            with torch.inference_mode():
                encoded = self.model.encode_image(tensor)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.detach().cpu().numpy().astype(np.float32))
        self.last_encode_seconds = time.perf_counter() - started
        return np.concatenate(features, axis=0)

    def build_from_snapshots(self, root: Path) -> Dict[str, Any]:
        images: List[Image.Image] = []
        labels: List[str] = []
        paths: List[str] = []
        for snapshot in sorted(Path(root).glob("*")):
            input_path = snapshot / "grasp_input.json"
            crop_path = snapshot / "crop.png"
            if not input_path.exists() or not crop_path.exists():
                continue
            with input_path.open("r", encoding="utf-8") as stream:
                grasp_input = json.load(stream)
            label = snapshot_identity(snapshot, grasp_input)
            if not label:
                continue
            images.append(Image.open(crop_path).convert("RGB"))
            labels.append(label)
            paths.append(str(snapshot.resolve()))
        self.features = self.encode(images)
        self.labels = labels
        self.paths = paths
        return {
            "reference_count": len(labels),
            "class_counts": {label: labels.count(label) for label in sorted(set(labels))},
            "encoding_seconds": self.last_encode_seconds,
        }

    def scores(
        self,
        query_features: np.ndarray,
        requested_identity: Optional[str] = None,
        exclude_snapshot: Optional[Path] = None,
        top_k_references: int = 3,
    ) -> List[Dict[str, Any]]:
        if self.features is None:
            raise RuntimeError("gallery has not been built")
        query_features = np.asarray(query_features, dtype=np.float32)
        similarities = query_features @ self.features.T
        excluded = str(exclude_snapshot.resolve()) if exclude_snapshot else None
        classes = sorted(set(self.labels))
        output: List[Dict[str, Any]] = []
        for row in similarities:
            class_scores: Dict[str, float] = {}
            for label in classes:
                values = [
                    float(value)
                    for value, ref_label, ref_path in zip(row, self.labels, self.paths)
                    if ref_label == label and ref_path != excluded
                ]
                if not values:
                    continue
                values.sort(reverse=True)
                class_scores[label] = float(np.mean(values[: max(1, top_k_references)]))
            ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
            output.append(
                {
                    "predicted_identity": ranked[0][0] if ranked else None,
                    "predicted_score": ranked[0][1] if ranked else None,
                    "runner_up_identity": ranked[1][0] if len(ranked) > 1 else None,
                    "score_margin": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None,
                    "requested_identity": requested_identity,
                    "requested_score": class_scores.get(requested_identity) if requested_identity else None,
                    "class_scores": class_scores,
                }
            )
        return output
