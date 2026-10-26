"""Offline SAM3 visible-mask proposals for camera-only occlusion experiments.

Generic category prompts propose objects, never read medicine text or select a
target. Their coverage is incomplete; masks are predictions, not visible truth.
"""

from contextlib import nullcontext
from pathlib import Path
import time

import cv2
import numpy as np
from runtime.perception_pipeline import serialized_resource


class SAM3Proposals:
    def __init__(self, checkpoint, *, device="cuda", confidence=.35):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.torch = torch
        self.device = device
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        self.confidence = float(confidence)
        started = time.perf_counter()
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint), load_from_HF=False,
            device=device, eval_mode=True, enable_segmentation=True,
            enable_inst_interactivity=False, compile=False)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence)
        self.load_seconds = time.perf_counter()-started

    @serialized_resource('gpu')
    def predict(self, color_bgr, prompts, *, min_mask_pixels=32):
        from PIL import Image
        torch = self.torch
        height, width = color_bgr.shape[:2]
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.device.startswith("cuda") else nullcontext())
        started = time.perf_counter()
        rows, counts = [], {}
        with torch.inference_mode(), autocast:
            state = self.processor.set_image(Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)))
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                prediction = self.processor.set_text_prompt(prompt=str(prompt), state=state)
                scores = prediction["scores"].float().cpu().numpy().reshape(-1)
                boxes = prediction["boxes"].float().cpu().numpy().reshape(-1, 4)
                masks = prediction["masks"].cpu().numpy().reshape(-1, height, width).astype(bool)
                counts[str(prompt)] = len(scores)
                for score, box, mask in zip(scores, boxes, masks):
                    if int(mask.sum()) < min_mask_pixels:
                        continue
                    rows.append({"source_id": f"proposal_{len(rows):03d}", "prompt": str(prompt),
                                 "confidence": float(score), "bbox_xyxy": box.tolist(), "mask": mask})
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return rows, {"backend": "sam3_generic_category_prompts", "prompts": list(prompts),
                      "confidence_threshold": self.confidence,
                      "proposals_per_prompt": counts, "minimum_mask_pixels": min_mask_pixels,
                      "inference_seconds": time.perf_counter()-started,
                      "checkpoint": str(self.checkpoint),
                      "coverage": "predicted_candidates_not_complete_visible_instance_segmentation",
                      "ocr_used": False, "target_identity_used": False}
