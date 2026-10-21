# services/sam3_bbox_segmenter_service.py

from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import cv2
from runtime.perception_pipeline import serialized_resource
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SAM3BBoxSegmenterService:
    """
    使用 GroundingDINO bbox 作为 SAM3 box prompt 进行精细分割。

    输入:
        image_path: 腕部相机 RGB 图像路径
        bbox: GroundingDINO 输出的 [x1, y1, x2, y2] 像素坐标

    输出:
        mask_path
        mask_npy_path
        overlay_path
        masked_image_path
        crop_path
        mask_area
    """

    def __init__(
        self,
        model_dir: str = "/home/wpy/models/sam3",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.0,
        use_autocast: bool = True,
        enable_inst_interactivity: bool = True,
        compile_model: bool = False,
        prompt_mode: str = "geometric",
    ):
        if prompt_mode not in ("geometric", "instance"):
            raise ValueError("prompt_mode must be geometric or instance")
        if prompt_mode == "instance" and not enable_inst_interactivity:
            raise ValueError("instance box prompts require enable_inst_interactivity=True")
        self.prompt_mode = prompt_mode
        self.model_dir = Path(model_dir)
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self._find_checkpoint(self.model_dir)
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.enable_inst_interactivity = enable_inst_interactivity
        self.compile_model = compile_model
        self.use_autocast = self._should_use_autocast(use_autocast)

        print("[SAM3BBox] Loading SAM3 image model...")
        print(f"[SAM3BBox] model_dir: {self.model_dir}")
        print(f"[SAM3BBox] checkpoint_path: {self.checkpoint_path}")
        print(f"[SAM3BBox] device: {self.device}")
        print(f"[SAM3BBox] use_autocast: {self.use_autocast}")

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint_path),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=self.enable_inst_interactivity,
            compile=self.compile_model,
        )

        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

        print("[SAM3BBox] SAM3 model loaded successfully.")

    @serialized_resource('gpu')
    def segment_by_bbox(
        self,
        image_path: str,
        bbox: List[int],
        output_prefix: str = "runs/current_capture/sam3_bbox_segment",
        expand_bbox_px: int = 5,
        select_by: str = "iou",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        用 bbox prompt 分割单个目标。

        Args:
            image_path:
                输入图像路径。

            bbox:
                GroundingDINO 输出的 [x1, y1, x2, y2] 像素坐标。

            output_prefix:
                输出文件前缀。

            expand_bbox_px:
                将 bbox 向外扩展几个像素。通常 5~15 比较合适。

            select_by:
                SAM3 可能返回多个 mask。
                - "iou": 选择预测 box 与输入 bbox IoU 最大的 mask
                - "score": 选择最高分 mask
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image_h, image_w = image_bgr.shape[:2]

        bbox = self._expand_and_clip_bbox(
            bbox=bbox,
            image_width=image_w,
            image_height=image_h,
            expand_px=expand_bbox_px,
        )

        image_pil = Image.open(image_path).convert("RGB")

        # GroundingDINO: [x1, y1, x2, y2] pixel
        # SAM3: [cx, cy, w, h] normalized
        sam3_box = self._xyxy_to_norm_cxcywh(
            bbox=bbox,
            image_width=image_w,
            image_height=image_h,
        )

        with torch.inference_mode():
            with self._autocast_context():
                state = self.processor.set_image(image_pil)

                if self.prompt_mode == "instance":
                    # The detector already chose one object. SAM3's interactive
                    # instance API takes pixel XYXY, not normalized CXCYWH.
                    masks, scores, _ = self.model.predict_inst(
                        state, point_coords=None, point_labels=None,
                        box=np.asarray(bbox, dtype=np.float32)[None, :],
                        multimask_output=False,
                    )
                    boxes = None
                else:
                    # Retain the original visual-exemplar/concept API.
                    if hasattr(self.processor, "reset_all_prompts"):
                        self.processor.reset_all_prompts(state)
                    state = self.processor.add_geometric_prompt(
                        state=state, box=sam3_box, label=True,
                    )
                    masks = state.get("masks", None)
                    boxes = state.get("boxes", None)
                    scores = state.get("scores", None)

        if masks is None or self._num_masks(masks) == 0:
            raise RuntimeError(
                "SAM3 没有返回 mask。可以尝试增大 expand_bbox_px，"
                "或者降低 confidence_threshold。"
            )

        selected_mask, selected_index, selected_score, selected_box = self._select_best_mask(
            masks=masks,
            boxes=boxes,
            scores=scores,
            input_bbox=bbox,
            image_size=(image_h, image_w),
            select_by=select_by,
        )

        outputs = self._save_outputs(
            image_bgr=image_bgr,
            mask=selected_mask,
            bbox=bbox,
            output_prefix=output_prefix,
        )

        outputs.update(
            {
                "image_path": image_path,
                "bbox": bbox,
                "sam3_box_cxcywh_norm": sam3_box,
                "selected_index": selected_index,
                "selected_score": selected_score,
                "selected_box": selected_box,
                "segmenter": "sam3_bbox",
                "prompt_mode": self.prompt_mode,
            }
        )

        return outputs

    @staticmethod
    def _find_checkpoint(model_dir: Path) -> Path:
        if not model_dir.exists():
            raise FileNotFoundError(f"SAM3 model_dir not found: {model_dir}")

        preferred = model_dir / "sam3.pt"
        if preferred.exists():
            return preferred

        pt_files = sorted(model_dir.glob("*.pt"))
        if len(pt_files) == 0:
            pt_files = sorted(model_dir.rglob("*.pt"))

        if len(pt_files) == 0:
            raise FileNotFoundError(f"No .pt checkpoint found under: {model_dir}")

        return pt_files[0]

    def _should_use_autocast(self, use_autocast: bool) -> bool:
        if not use_autocast:
            return False

        if not str(self.device).startswith("cuda"):
            return False

        if not torch.cuda.is_available():
            return False

        # 2080Ti 是 7.5，不建议 bf16 autocast
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8

    def _autocast_context(self):
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _expand_and_clip_bbox(
        bbox: List[int],
        image_width: int,
        image_height: int,
        expand_px: int,
    ) -> List[int]:
        x1, y1, x2, y2 = bbox

        x1 = int(round(x1 - expand_px))
        y1 = int(round(y1 - expand_px))
        x2 = int(round(x2 + expand_px))
        y2 = int(round(y2 + expand_px))

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

        return [x1, y1, x2, y2]

    @staticmethod
    def _xyxy_to_norm_cxcywh(
        bbox: List[int],
        image_width: int,
        image_height: int,
    ) -> List[float]:
        x1, y1, x2, y2 = bbox

        cx = ((x1 + x2) / 2.0) / image_width
        cy = ((y1 + y2) / 2.0) / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height

        return [float(cx), float(cy), float(w), float(h)]

    @staticmethod
    def _num_masks(masks) -> int:
        if isinstance(masks, torch.Tensor):
            if masks.ndim == 2:
                return 1
            return int(masks.shape[0])
        if isinstance(masks, list):
            return len(masks)
        arr = np.asarray(masks)
        if arr.ndim == 2:
            return 1
        return int(arr.shape[0])

    @staticmethod
    def _tensor_or_array_to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x)

    def _select_best_mask(
        self,
        masks,
        boxes,
        scores,
        input_bbox: List[int],
        image_size: Tuple[int, int],
        select_by: str,
    ):
        image_h, image_w = image_size

        masks_np = self._tensor_or_array_to_numpy(masks)

        # 可能是 [N, 1, H, W] / [N, H, W] / [H, W]
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0, :, :]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]

        n = masks_np.shape[0]

        scores_np = self._tensor_or_array_to_numpy(scores)
        if scores_np is None:
            scores_np = np.ones((n,), dtype=np.float32)
        scores_np = scores_np.reshape(-1)
        if len(scores_np) < n:
            scores_np = np.pad(scores_np, (0, n - len(scores_np)), constant_values=1.0)

        boxes_np = self._tensor_or_array_to_numpy(boxes)
        if boxes_np is not None:
            boxes_np = boxes_np.reshape(-1, 4)

        if select_by == "iou" and boxes_np is not None and len(boxes_np) >= n:
            ious = [
                self._bbox_iou(input_bbox, boxes_np[i].tolist())
                for i in range(n)
            ]
            selected_index = int(np.argmax(ious))
        else:
            selected_index = int(np.argmax(scores_np[:n]))

        mask = masks_np[selected_index]

        # 如果 mask 尺寸不是原图尺寸，resize 回原图
        if mask.shape[0] != image_h or mask.shape[1] != image_w:
            mask_t = torch.from_numpy(mask).float()[None, None, :, :]
            mask_t = F.interpolate(
                mask_t,
                size=(image_h, image_w),
                mode="bilinear",
                align_corners=False,
            )
            mask = mask_t[0, 0].numpy()

        mask_u8 = (mask > 0).astype(np.uint8) * 255

        selected_score = float(scores_np[selected_index])
        selected_box = None
        if boxes_np is not None and len(boxes_np) > selected_index:
            selected_box = [float(v) for v in boxes_np[selected_index].tolist()]

        return mask_u8, selected_index, selected_score, selected_box

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return float(inter_area / union)

    @staticmethod
    def _save_outputs(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
        output_prefix: str,
    ) -> Dict[str, Any]:
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if mask.max() <= 1:
            mask = mask * 255

        x1, y1, x2, y2 = bbox

        mask_path = f"{output_prefix}_mask.png"
        mask_npy_path = f"{output_prefix}_mask.npy"
        overlay_path = f"{output_prefix}_overlay.png"
        masked_image_path = f"{output_prefix}_masked.png"
        crop_path = f"{output_prefix}_crop.png"

        cv2.imwrite(mask_path, mask)
        np.save(mask_npy_path, mask)

        overlay = image_bgr.copy()
        color_layer = image_bgr.copy()
        color_layer[mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.65, color_layer, 0.35, 0)

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.imwrite(overlay_path, overlay)

        masked = image_bgr.copy()
        masked[mask == 0] = 0
        cv2.imwrite(masked_image_path, masked)

        crop = image_bgr[y1:y2, x1:x2]
        cv2.imwrite(crop_path, crop)

        mask_area = int((mask > 0).sum())

        return {
            "mask_path": mask_path,
            "mask_npy_path": mask_npy_path,
            "overlay_path": overlay_path,
            "masked_image_path": masked_image_path,
            "crop_path": crop_path,
            "mask_area": mask_area,
        }
