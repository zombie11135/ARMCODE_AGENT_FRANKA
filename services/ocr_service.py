# services/ocr_service.py

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class PaddleOCRService:
    """
    PaddleOCR 服务。

    兼容 PaddleOCR 3.x：
    - 初始化用 device="cpu" / "gpu"
    - 推荐推理接口 ocr.predict(image_path)

    兼容 PaddleOCR 2.x：
    - 如果当前版本存在 ocr.ocr()，也会尝试使用。
    """

    def __init__(
        self,
        lang: str = "ch",
        device: str = "cpu",
        use_textline_orientation: bool = True,
        text_detection_model_name: Optional[str] = None,
        text_recognition_model_name: Optional[str] = None,
        show_log: bool = False,
        **kwargs,
    ):
        self.lang = lang
        self.device = device
        self.use_textline_orientation = use_textline_orientation
        self.show_log = show_log

        print("[OCR] Loading PaddleOCR...")
        print(
            f"[OCR] lang={lang}, device={device}, "
            f"use_textline_orientation={use_textline_orientation}"
        )

        from paddleocr import PaddleOCR

        # PaddleOCR 3.x 参数
        params = {
            "lang": lang,
            "device": device,
            "use_textline_orientation": use_textline_orientation,
        }

        # 这几个可选参数只有你需要指定模型时再传
        if text_detection_model_name is not None:
            params["text_detection_model_name"] = text_detection_model_name

        if text_recognition_model_name is not None:
            params["text_recognition_model_name"] = text_recognition_model_name

        # 允许外部补充 PaddleOCR 3.x 支持的参数
        params.update(kwargs)

        try:
            self.ocr = PaddleOCR(**params)

        except Exception as e:
            print(f"[OCR] PaddleOCR 3.x init failed: {e}")
            print("[OCR] Trying PaddleOCR 2.x compatible init...")

            # PaddleOCR 2.x 兼容兜底
            # 注意：只有在你安装的是 2.x 时才会走通
            legacy_params = {
                "lang": lang,
                "use_angle_cls": use_textline_orientation,
                "show_log": show_log,
            }

            # 2.x 用 use_gpu；3.x 不支持，所以只在 fallback 里用
            legacy_params["use_gpu"] = device.startswith("gpu") or device == "cuda"

            self.ocr = PaddleOCR(**legacy_params)

        print("[OCR] PaddleOCR loaded.")

    def read_image(
        self,
        image_path: str,
        output_prefix: str = "runs/current_capture/ocr",
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        raw_result = self._run_ocr(image_path)
        items = self._parse_ocr_result(raw_result)

        items = [
            item for item in items
            if item.get("confidence", 0.0) >= min_confidence
        ]

        vis_path = self._draw_ocr_result(
            image_path=image_path,
            items=items,
            output_path=f"{output_prefix}_vis.png",
        )

        texts = [item["text"] for item in items]
        confidences = [item["confidence"] for item in items]

        return {
            "image_path": image_path,
            "texts": texts,
            "text": " ".join(texts),
            "confidences": confidences,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "items": items,
            "raw_result": raw_result,
            "vis_path": vis_path,
        }

    def read_region(
        self,
        image_path: str,
        bbox: List[int],
        output_prefix: str = "runs/current_capture/ocr_region",
        padding: int = 5,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image_h, image_w = image_bgr.shape[:2]

        x1, y1, x2, y2 = self._expand_and_clip_bbox(
            bbox=bbox,
            image_width=image_w,
            image_height=image_h,
            padding=padding,
        )

        crop = image_bgr[y1:y2, x1:x2]

        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        crop_path = f"{output_prefix}_crop.png"
        cv2.imwrite(crop_path, crop)

        result = self.read_image(
            image_path=crop_path,
            output_prefix=output_prefix,
            min_confidence=min_confidence,
        )

        for item in result["items"]:
            if item.get("box") is not None:
                item["box_in_crop"] = item["box"]
                item["box"] = [
                    [float(px + x1), float(py + y1)]
                    for px, py in item["box_in_crop"]
                ]

        result.update(
            {
                "source_image_path": image_path,
                "bbox": [x1, y1, x2, y2],
                "crop_path": crop_path,
            }
        )

        return result

    def _run_ocr(self, image_path: str):
        """
        PaddleOCR 3.x 优先使用 predict。
        PaddleOCR 2.x 使用 ocr。
        """
        if hasattr(self.ocr, "predict"):
            return self.ocr.predict(image_path)

        if hasattr(self.ocr, "ocr"):
            try:
                return self.ocr.ocr(
                    image_path,
                    cls=self.use_textline_orientation,
                )
            except TypeError:
                return self.ocr.ocr(image_path)

        raise RuntimeError("Unknown PaddleOCR API: neither predict() nor ocr() exists.")

    def _parse_ocr_result(self, raw_result) -> List[Dict[str, Any]]:
        """
        兼容 PaddleOCR 2.x / 3.x 返回格式。

        2.x 常见格式：
        [
            [
                [box, (text, score)],
                ...
            ]
        ]

        3.x 常见结果对象可能支持：
        - dict
        - json / res 字段
        - rec_texts / rec_scores / rec_polys
        """
        items = []

        def to_plain(obj):
            """
            PaddleOCR 3.x 的返回对象有时不是普通 dict，
            这里尽量转成 dict。
            """
            if isinstance(obj, dict):
                return obj

            # 有些结果对象可能有 json 属性
            if hasattr(obj, "json"):
                try:
                    value = obj.json
                    if isinstance(value, dict):
                        return value
                    if callable(value):
                        value = value()
                        if isinstance(value, dict):
                            return value
                except Exception:
                    pass

            # 有些结果对象可能有 res 属性
            if hasattr(obj, "res"):
                try:
                    value = obj.res
                    if isinstance(value, dict):
                        return value
                except Exception:
                    pass

            return obj

        def parse_node(node):
            node = to_plain(node)

            if isinstance(node, dict):
                rec_texts = (
                    node.get("rec_texts")
                    or node.get("texts")
                    or node.get("text")
                )

                rec_scores = (
                    node.get("rec_scores")
                    or node.get("scores")
                    or node.get("confidences")
                )

                rec_boxes = (
                    node.get("rec_polys")
                    or node.get("rec_boxes")
                    or node.get("dt_polys")
                    or node.get("boxes")
                )

                if isinstance(rec_texts, str):
                    rec_texts = [rec_texts]

                if rec_texts is not None:
                    for i, text in enumerate(rec_texts):
                        score = 0.0
                        if rec_scores is not None and i < len(rec_scores):
                            score = float(rec_scores[i])

                        box = None
                        if rec_boxes is not None and i < len(rec_boxes):
                            box = self._normalize_box(rec_boxes[i])

                        items.append(
                            {
                                "text": str(text),
                                "confidence": score,
                                "box": box,
                            }
                        )
                    return

                for v in node.values():
                    parse_node(v)
                return

            if isinstance(node, (list, tuple)):
                # PaddleOCR 2.x 单条记录：[box, (text, score)]
                if len(node) == 2:
                    maybe_box, maybe_text_score = node

                    if self._looks_like_box(maybe_box) and isinstance(
                        maybe_text_score,
                        (list, tuple),
                    ) and len(maybe_text_score) >= 2:
                        text = str(maybe_text_score[0])
                        score = float(maybe_text_score[1])
                        box = self._normalize_box(maybe_box)

                        items.append(
                            {
                                "text": text,
                                "confidence": score,
                                "box": box,
                            }
                        )
                        return

                for x in node:
                    parse_node(x)

        parse_node(raw_result)

        items = [
            item for item in items
            if item.get("text", "").strip()
        ]

        return items

    @staticmethod
    def _looks_like_box(x) -> bool:
        try:
            arr = np.asarray(x)
            return arr.ndim == 2 and arr.shape[1] == 2
        except Exception:
            return False

    @staticmethod
    def _normalize_box(box) -> Optional[List[List[float]]]:
        if box is None:
            return None

        arr = np.asarray(box, dtype=np.float32)

        if arr.ndim == 1 and len(arr) == 4:
            x1, y1, x2, y2 = arr.tolist()
            return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.tolist()

        return None

    @staticmethod
    def _expand_and_clip_bbox(
        bbox: List[int],
        image_width: int,
        image_height: int,
        padding: int,
    ) -> List[int]:
        x1, y1, x2, y2 = bbox

        x1 = int(round(x1 - padding))
        y1 = int(round(y1 - padding))
        x2 = int(round(x2 + padding))
        y2 = int(round(y2 + padding))

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")

        return [x1, y1, x2, y2]

    @staticmethod
    def _draw_ocr_result(
        image_path: str,
        items: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        for idx, item in enumerate(items):
            box = item.get("box")
            conf = item.get("confidence", 0.0)

            if box is None:
                continue

            pts = np.asarray(box, dtype=np.int32)

            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )

            x, y = pts[0]
            label = f"text_{idx + 1} {conf:.2f}"

            cv2.putText(
                image,
                label,
                (int(x), max(20, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

        return output_path