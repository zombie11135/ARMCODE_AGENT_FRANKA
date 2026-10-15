from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch


class YOLOEInstanceSegmenter:
    """Thin adapter around the official THU-MIG YOLOE implementation."""

    def __init__(
        self,
        yoloe_root: Path,
        checkpoint: Path,
        prompts: Iterable[str],
        device: str = "cuda:0",
        confidence: float = 0.15,
        image_size: int = 640,
    ) -> None:
        self.yoloe_root = Path(yoloe_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.prompts = [str(value).strip() for value in prompts if str(value).strip()]
        self.device = device
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        if not self.prompts:
            raise ValueError("at least one YOLOE prompt is required")

        from ultralytics import YOLOE

        previous = Path.cwd()
        try:
            # The official text-model adapter resolves mobileclip_blt.pt from cwd.
            os.chdir(self.yoloe_root)
            self.model = YOLOE(str(self.checkpoint)).to(self.device)
            embeddings = self.model.get_text_pe(self.prompts)
            self.model.set_classes(self.prompts, embeddings)
        finally:
            os.chdir(previous)

    def predict(self, color_bgr: np.ndarray) -> Dict[str, Any]:
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        result = self.model.predict(
            rgb,
            verbose=False,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
        )[0]
        elapsed = time.perf_counter() - started
        instances: List[Dict[str, Any]] = []
        if result.boxes is None or result.masks is None:
            return {"instances": instances, "elapsed_seconds": elapsed, "model_speed_ms": result.speed}

        from ultralytics.utils.ops import scale_masks

        masks = scale_masks(result.masks.data[:, None], result.orig_shape, padding=True)[:, 0]
        masks = (masks > 0.5).detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for index, (box, confidence, class_index, mask) in enumerate(
            zip(boxes, confidences, classes, masks)
        ):
            instances.append(
                {
                    "instance_id": "yoloe_{:03d}".format(index),
                    "bbox_xyxy": [float(value) for value in box],
                    "confidence": float(confidence),
                    "class_index": int(class_index),
                    "class_name": str(result.names[int(class_index)]),
                    "mask": mask.astype(bool),
                }
            )
        return {"instances": instances, "elapsed_seconds": elapsed, "model_speed_ms": result.speed}

    def warmup(self, shape: tuple = (720, 1280, 3)) -> float:
        dummy = np.zeros(shape, dtype=np.uint8)
        started = time.perf_counter()
        self.predict(dummy)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - started
