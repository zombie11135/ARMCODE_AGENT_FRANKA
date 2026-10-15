# services/segmenter_service.py

from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np


class BaseSegmenterService:
    """
    分割服务基类。
    所有 segmenter 都建议统一提供 segment_by_bbox() 接口。
    """

    def segment_by_bbox(
        self,
        image_path: str,
        bbox: List[int],
        output_prefix: str = "runs/current_capture/segment",
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _clip_bbox(bbox, width: int, height: int):
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width - 1))
        y2 = max(0, min(int(y2), height - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

        return [x1, y1, x2, y2]

    @staticmethod
    def _save_outputs(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
        output_prefix: str,
    ) -> Dict[str, Any]:
        """
        保存：
        - mask
        - overlay 可视化图
        - masked image
        - crop image
        """
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if mask.max() <= 1:
            mask = mask * 255

        h, w = mask.shape[:2]
        img_h, img_w = image_bgr.shape[:2]

        if h != img_h or w != img_w:
            raise ValueError(
                f"Mask size {mask.shape} does not match image size {image_bgr.shape}"
            )

        x1, y1, x2, y2 = bbox

        mask_path = f"{output_prefix}_mask.png"
        overlay_path = f"{output_prefix}_overlay.png"
        masked_image_path = f"{output_prefix}_masked.png"
        crop_path = f"{output_prefix}_crop.png"

        # 1. 保存 mask
        cv2.imwrite(mask_path, mask)

        # 2. 保存 overlay
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

        # 3. 保存 masked image
        masked = image_bgr.copy()
        masked[mask == 0] = 0
        cv2.imwrite(masked_image_path, masked)

        # 4. 保存 crop
        crop = image_bgr[y1:y2, x1:x2]
        cv2.imwrite(crop_path, crop)

        mask_area = int((mask > 0).sum())

        return {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "masked_image_path": masked_image_path,
            "crop_path": crop_path,
            "mask_area": mask_area,
        }


class BBoxSegmenterService(BaseSegmenterService):
    """
    临时可运行版本：
    直接用 bbox 生成矩形 mask。

    用途：
    - 先打通 segment_object skill
    - 后面替换成 SAM3SegmenterService
    """

    def __init__(self, padding: int = 0):
        self.padding = padding

    def segment_by_bbox(
        self,
        image_path: str,
        bbox: List[int],
        output_prefix: str = "runs/current_capture/segment",
        **kwargs,
    ) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        height, width = image_bgr.shape[:2]

        x1, y1, x2, y2 = bbox

        if self.padding > 0:
            x1 -= self.padding
            y1 -= self.padding
            x2 += self.padding
            y2 += self.padding

        bbox = self._clip_bbox([x1, y1, x2, y2], width, height)

        x1, y1, x2, y2 = bbox

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        outputs = self._save_outputs(
            image_bgr=image_bgr,
            mask=mask,
            bbox=bbox,
            output_prefix=output_prefix,
        )

        outputs.update(
            {
                "image_path": image_path,
                "bbox": bbox,
                "segmenter": "bbox_segmenter",
            }
        )

        return outputs


class SAM3SegmenterServiceTemplate(BaseSegmenterService):
    """
    SAM3 接入模板。

    你后面只需要把 _predict_mask_with_sam3() 里的内容替换成你现有的 SAM3 推理代码。
    """

    def __init__(self, sam3_model=None, device: str = "cuda"):
        self.sam3_model = sam3_model
        self.device = device

    def segment_by_bbox(
        self,
        image_path: str,
        bbox: List[int],
        output_prefix: str = "runs/current_capture/segment",
        **kwargs,
    ) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        height, width = image_bgr.shape[:2]
        bbox = self._clip_bbox(bbox, width, height)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = self._predict_mask_with_sam3(
            image_rgb=image_rgb,
            bbox=bbox,
            **kwargs,
        )

        if mask is None:
            raise RuntimeError("SAM3 failed to return a valid mask.")

        if mask.shape[:2] != image_bgr.shape[:2]:
            raise RuntimeError(
                f"SAM3 mask shape {mask.shape} does not match image shape {image_bgr.shape}"
            )

        mask = (mask > 0).astype(np.uint8) * 255

        outputs = self._save_outputs(
            image_bgr=image_bgr,
            mask=mask,
            bbox=bbox,
            output_prefix=output_prefix,
        )

        outputs.update(
            {
                "image_path": image_path,
                "bbox": bbox,
                "segmenter": "sam3",
            }
        )

        return outputs

    def _predict_mask_with_sam3(
        self,
        image_rgb: np.ndarray,
        bbox: List[int],
        **kwargs,
    ) -> np.ndarray:
        """
        在这里接入你的 SAM3 代码。

        输入：
            image_rgb: RGB 图像，numpy array, H x W x 3
            bbox: [x1, y1, x2, y2]

        输出：
            mask: H x W 的 bool 或 0/1 mask

        你需要把下面的 NotImplementedError 换成自己的 SAM3 推理逻辑。
        """
        raise NotImplementedError(
            "请在 SAM3SegmenterServiceTemplate._predict_mask_with_sam3() 中接入你的 SAM3 推理代码。"
        )