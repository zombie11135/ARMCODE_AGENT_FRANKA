# services/grounding_dino_detector_service.py

from typing import Any, Dict, List, Optional

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


TARGET_PRESETS = {
    # 药盒：不要写 pill box / box / package 这种太泛的词
    "medicine_box": [
        "rectangular medicine carton",
        "pharmaceutical paper box",
        "medicine package with printed text",
        "drug package box",
        "medicine bag",
        "medicine package"
    ],

    # 水果：水果通常比较好检测
    "fruit": [
        "apple",
        "orange",
        "banana",
        "pear",
        "fruit",
    ],

    # 零食：先用包装类描述，不要写 food 这种太泛的词
    "snack": [
        "snack package",
        "snack bag",
        "biscuit package",
        "cookie package",
        "candy package",
        "food package with printed text",
    ],

    # 托盘建议后面单独做，先保留
    "tray": [
        "round pink plastic tray",
        "shallow pink tray",
    ],
}


class GroundingDINODetectorService:
    """
    GroundingDINO 检测服务。

    用法：
        detector.detect(image_path, target_name="medicine_box")
        detector.detect(image_path, target_name="fruit")
        detector.detect(image_path, target_name="snack")

    返回：
        {
            "image_path": ...,
            "detections": [
                {
                    "object_id": "obj_1",
                    "name": "medicine_box",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.83,
                    "phrase": "rectangular medicine carton"
                }
            ]
        }
    """

    def __init__(
        self,
        model_name_or_path: str = "/home/wpy/models/grounding-dino-tiny",
        device: Optional[str] = None,
        threshold: float = 0.30,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.50,
        local_files_only: bool = True,
        use_fp16: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.local_files_only = local_files_only
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")

        print(f"[GroundingDINO] Loading model from: {model_name_or_path}")
        print(f"[GroundingDINO] device={self.device}, fp16={self.use_fp16}")

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model.half()

        print("[GroundingDINO] Model loaded successfully.")

    def detect(
        self,
        image_path: str,
        target_name: str,
        threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        max_objects: int = 20,
        enable_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        检测指定类别。

        target_name 推荐使用：
            medicine_box
            fruit
            snack
            tray

        也可以传自然语言，例如：
            "apple"
            "orange"
            "medicine box"
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image_height, image_width = image_bgr.shape[:2]

        image_pil = Image.open(image_path).convert("RGB")

        target_key, phrases = self._resolve_target(target_name)
        text_prompt = self._build_text_prompt(phrases)

        threshold = self.threshold if threshold is None else threshold
        text_threshold = self.text_threshold if text_threshold is None else text_threshold

        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        if self.use_fp16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self._post_process(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        raw_detections = []

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            score = float(score.detach().cpu().item())
            phrase = str(label)

            bbox = self._clip_bbox(
                [x1, y1, x2, y2],
                image_width=image_width,
                image_height=image_height,
            )

            if not self._valid_bbox(bbox):
                continue

            det = {
                "object_id": f"obj_{len(raw_detections) + 1}",
                "name": target_key,
                "bbox": [int(v) for v in bbox],
                "confidence": score,
                "phrase": phrase,
                "image_path": image_path,
            }

            raw_detections.append(det)

        # NMS 去重
        detections = self._nms(raw_detections, self.nms_iou_threshold)

        # 类别专属后处理过滤
        if enable_filter:
            detections = self._filter_by_target_rules(
                detections=detections,
                target_key=target_key,
                image_width=image_width,
                image_height=image_height,
            )

        detections = detections[:max_objects]

        for i, det in enumerate(detections):
            det["object_id"] = f"obj_{i + 1}"

        return {
            "image_path": image_path,
            "image_width": image_width,
            "image_height": image_height,
            "target_name": target_name,
            "target_key": target_key,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "detections": detections,
        }

    def detect_all_task_objects(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """
        针对你的桌面任务，依次检测药盒、水果、零食。
        注意：这里是分开检测，不是把所有 prompt 混在一起。
        """
        all_detections = []

        for target_name, threshold, text_threshold in [
            ("medicine_box", 0.32, 0.28),
            ("fruit", 0.30, 0.25),
            ("snack", 0.32, 0.28),
        ]:
            result = self.detect(
                image_path=image_path,
                target_name=target_name,
                threshold=threshold,
                text_threshold=text_threshold,
                enable_filter=True,
            )

            all_detections.extend(result["detections"])

        all_detections = self._nms(all_detections, self.nms_iou_threshold)

        for i, det in enumerate(all_detections):
            det["object_id"] = f"obj_{i + 1}"

        return {
            "image_path": image_path,
            "detections": all_detections,
        }

    def _post_process(
        self,
        outputs,
        input_ids,
        threshold: float,
        text_threshold: float,
        target_sizes,
    ):
        """
        兼容不同 transformers 版本：
        有些版本用 threshold；
        有些版本旧代码里叫 box_threshold。
        """
        try:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )

    @staticmethod
    def _resolve_target(target_name: str):
        name = target_name.lower().strip()

        medicine_alias = [
            "medicine",
            "medicine_box",
            "medicine box",
            "drug",
            "drug box",
            "药盒",
        ]

        fruit_alias = [
            "fruit",
            "水果",
            "apple",
            "苹果",
            "orange",
            "橙子",
            "banana",
            "香蕉",
            "pear",
            "梨",
        ]

        snack_alias = [
            "snack",
            "零食",
            "biscuit",
            "cookie",
            "candy",
            "chips",
            "snack package",
        ]

        tray_alias = [
            "tray",
            "pink tray",
            "托盘",
            "粉色托盘",
            "plate",
        ]

        if name in medicine_alias:
            return "medicine_box", TARGET_PRESETS["medicine_box"]

        if name in fruit_alias:
            # 如果是具体水果，就优先只检测具体水果
            if name in ["apple", "苹果"]:
                return "fruit", ["apple"]
            if name in ["orange", "橙子"]:
                return "fruit", ["orange"]
            if name in ["banana", "香蕉"]:
                return "fruit", ["banana"]
            if name in ["pear", "梨"]:
                return "fruit", ["pear"]
            return "fruit", TARGET_PRESETS["fruit"]

        if name in snack_alias:
            return "snack", TARGET_PRESETS["snack"]

        if name in tray_alias:
            return "tray", TARGET_PRESETS["tray"]

        # 自定义目标名
        return name.replace(" ", "_"), [target_name]

    @staticmethod
    def _build_text_prompt(phrases: List[str]) -> str:
        """
        GroundingDINO 推荐用英文句号分隔类别短语。
        """
        clean = []
        for p in phrases:
            p = p.strip().lower()
            if p and p not in clean:
                clean.append(p)

        return ". ".join(clean) + "."

    @staticmethod
    def _clip_bbox(
        bbox,
        image_width: int,
        image_height: int,
    ):
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(float(x1), image_width - 1))
        y1 = max(0, min(float(y1), image_height - 1))
        x2 = max(0, min(float(x2), image_width - 1))
        y2 = max(0, min(float(y2), image_height - 1))

        return [x1, y1, x2, y2]

    @staticmethod
    def _valid_bbox(bbox) -> bool:
        x1, y1, x2, y2 = bbox
        return x2 > x1 and y2 > y1

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

        return inter_area / union

    def _nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        detections = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True,
        )

        kept = []

        for det in detections:
            keep = True

            for old in kept:
                iou = self._bbox_iou(det["bbox"], old["bbox"])

                # 同一类才强 NMS
                if det["name"] == old["name"] and iou > iou_threshold:
                    keep = False
                    break

            if keep:
                kept.append(det)

        return kept

    def _filter_by_target_rules(
        self,
        detections: List[Dict[str, Any]],
        target_key: str,
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        filtered = []

        for det in detections:
            bbox = det["bbox"]

            if target_key == "medicine_box":
                if not self._is_reasonable_medicine_box(
                    bbox,
                    image_width,
                    image_height,
                ):
                    continue

            elif target_key == "snack":
                if not self._is_reasonable_package(
                    bbox,
                    image_width,
                    image_height,
                ):
                    continue

            elif target_key == "fruit":
                if not self._is_reasonable_fruit(
                    bbox,
                    image_width,
                    image_height,
                ):
                    continue

            filtered.append(det)

        return filtered

    @staticmethod
    def _is_reasonable_medicine_box(
        bbox,
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        药盒过滤规则：
        - 面积不能太小，过滤绿色小方块这类误检
        - 药盒通常是矩形，不是很小的正方形
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return False

        area = w * h
        image_area = image_width * image_height
        area_ratio = area / image_area

        aspect = max(w / h, h / w)

        if area_ratio < 0.015:
            return False

        if area_ratio > 0.55:
            return False

        if aspect < 1.15:
            return False

        if aspect > 6.5:
            return False

        return True

    @staticmethod
    def _is_reasonable_package(
        bbox,
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        零食包装通常也是中等大小矩形或袋状区域。
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return False

        area_ratio = (w * h) / (image_width * image_height)

        if area_ratio < 0.01:
            return False

        if area_ratio > 0.65:
            return False

        return True

    @staticmethod
    def _is_reasonable_fruit(
        bbox,
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        水果可能接近圆形，所以不能用药盒那种长宽比过滤。
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return False

        area_ratio = (w * h) / (image_width * image_height)
        aspect = max(w / h, h / w)

        if area_ratio < 0.005:
            return False

        if area_ratio > 0.50:
            return False

        if aspect > 3.0:
            return False

        return True