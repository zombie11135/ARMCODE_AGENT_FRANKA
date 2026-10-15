from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from .geometry import build_instance_geometry, load_depth_m, mask_iou
from .retrieval import MobileCLIPGallery, masked_crop, snapshot_identity
from .run_snapshot import DEFAULT_PROMPTS, PROJECT_ROOT, save_overlay
from .yoloe_backend import YOLOEInstanceSegmenter


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshots = [
        path
        for path in sorted(args.snapshots_root.resolve().glob("*"))
        if (path / "grasp_input.json").exists()
    ]
    if args.max_cases is not None:
        snapshots = snapshots[: args.max_cases]

    segmenter = YOLOEInstanceSegmenter(
        args.yoloe_root,
        args.checkpoint,
        args.prompts,
        device=args.device,
        confidence=args.confidence,
        image_size=args.image_size,
    )
    warmup_seconds = segmenter.warmup()
    gallery = MobileCLIPGallery(args.mobileclip_checkpoint, device=args.device)
    gallery_report = gallery.build_from_snapshots(args.snapshots_root)

    cases: List[Dict[str, Any]] = []
    for case_index, snapshot in enumerate(snapshots):
        with (snapshot / "grasp_input.json").open("r", encoding="utf-8") as stream:
            grasp_input = json.load(stream)
        color = cv2.imread(str(snapshot / "color.png"), cv2.IMREAD_COLOR)
        if color is None:
            continue
        depth_m = load_depth_m(snapshot / "depth.png", grasp_input.get("depth_scale", 1000.0))
        old_mask = np.load(str(snapshot / "mask.npy")) > 0
        target_bbox = grasp_input["bbox"]
        intrinsics = grasp_input["depth_intrinsics"]
        expected = snapshot_identity(snapshot, grasp_input)
        if expected is None:
            continue

        prediction = segmenter.predict(color)
        instances = prediction["instances"]
        for instance in instances:
            instance["evaluator_bbox_iou"] = bbox_iou(instance["bbox_xyxy"], target_bbox)
        best_proposal = max(instances, key=lambda item: item["evaluator_bbox_iou"], default=None)
        confidence_selected = max(instances, key=lambda item: item["confidence"], default=None)

        retrieval_started = time.perf_counter()
        if instances:
            crops = [masked_crop(color, item["mask"], item["bbox_xyxy"]) for item in instances]
            features = gallery.encode(crops)
            retrieval = gallery.scores(features, expected, exclude_snapshot=snapshot)
            for instance, score in zip(instances, retrieval):
                instance["retrieval"] = score
            retrieval_selected = max(
                instances,
                key=lambda item: (
                    item["retrieval"]["requested_score"]
                    if item["retrieval"]["requested_score"] is not None
                    else -1.0
                ),
            )
        else:
            retrieval_selected = None
        retrieval_seconds = time.perf_counter() - retrieval_started

        old_geometry = build_instance_geometry(color, depth_m, old_mask, intrinsics, erosion_px=2)
        new_geometry = (
            build_instance_geometry(color, depth_m, retrieval_selected["mask"], intrinsics, erosion_px=2)
            if retrieval_selected is not None
            else None
        )
        case = {
            "snapshot": str(snapshot),
            "expected_identity": expected,
            "candidate_count": len(instances),
            "yoloe_seconds": prediction["elapsed_seconds"],
            "retrieval_seconds": retrieval_seconds,
            "best_proposal_bbox_iou": best_proposal["evaluator_bbox_iou"] if best_proposal else 0.0,
            "confidence_selected_bbox_iou": (
                confidence_selected["evaluator_bbox_iou"] if confidence_selected else 0.0
            ),
            "retrieval_selected_bbox_iou": (
                retrieval_selected["evaluator_bbox_iou"] if retrieval_selected else 0.0
            ),
            "retrieval_predicted_identity": (
                retrieval_selected["retrieval"]["predicted_identity"] if retrieval_selected else None
            ),
            "retrieval_requested_score": (
                retrieval_selected["retrieval"]["requested_score"] if retrieval_selected else None
            ),
            "historical_mask_iou": (
                mask_iou(retrieval_selected["mask"], old_mask) if retrieval_selected else 0.0
            ),
            "old_mask_quality": old_geometry["quality"],
            "new_mask_quality": new_geometry["quality"] if new_geometry else None,
        }
        cases.append(case)

        if case_index < args.save_visualizations or case["retrieval_selected_bbox_iou"] < 0.5:
            if retrieval_selected is not None:
                save_overlay(
                    output / (snapshot.name + "_retrieval_selected.png"),
                    color,
                    retrieval_selected["mask"],
                    retrieval_selected["bbox_xyxy"],
                    "{} {:.3f}".format(
                        expected, retrieval_selected["retrieval"]["requested_score"] or 0.0
                    ),
                )

    yolo_times = [case["yoloe_seconds"] for case in cases]
    retrieval_times = [case["retrieval_seconds"] for case in cases]
    summary = {
        "case_count": len(cases),
        "proposal_recall_bbox_iou_0_5": float(
            np.mean([case["best_proposal_bbox_iou"] >= 0.5 for case in cases])
        ),
        "proposal_recall_bbox_iou_0_75": float(
            np.mean([case["best_proposal_bbox_iou"] >= 0.75 for case in cases])
        ),
        "confidence_selection_success_iou_0_5": float(
            np.mean([case["confidence_selected_bbox_iou"] >= 0.5 for case in cases])
        ),
        "retrieval_selection_success_iou_0_5": float(
            np.mean([case["retrieval_selected_bbox_iou"] >= 0.5 for case in cases])
        ),
        "retrieval_selected_identity_top1_accuracy": float(
            np.mean(
                [case["retrieval_predicted_identity"] == case["expected_identity"] for case in cases]
            )
        ),
        "median_best_proposal_bbox_iou": median(
            [case["best_proposal_bbox_iou"] for case in cases]
        ),
        "median_retrieval_selected_bbox_iou": median(
            [case["retrieval_selected_bbox_iou"] for case in cases]
        ),
        "median_historical_mask_iou": median([case["historical_mask_iou"] for case in cases]),
        "median_yoloe_seconds": median(yolo_times),
        "p95_yoloe_seconds": percentile(yolo_times, 95),
        "median_retrieval_seconds": median(retrieval_times),
        "median_old_valid_depth_ratio": median(
            [case["old_mask_quality"]["valid_depth_ratio"] for case in cases]
        ),
        "median_new_valid_depth_ratio": median(
            [
                case["new_mask_quality"]["valid_depth_ratio"]
                for case in cases
                if case["new_mask_quality"] is not None
            ]
        ),
        "median_old_largest_component_ratio": median(
            [case["old_mask_quality"]["largest_component_ratio"] for case in cases]
        ),
        "median_new_largest_component_ratio": median(
            [
                case["new_mask_quality"]["largest_component_ratio"]
                for case in cases
                if case["new_mask_quality"] is not None
            ]
        ),
    }
    report = {
        "status": "development_benchmark",
        "offline_only": True,
        "reference_warning": (
            "Stored bbox/mask are historical pipeline outputs, not human ground truth. "
            "BBox IoU measures recovery of the previously selected target; mask IoU is agreement only."
        ),
        "selection_isolation": (
            "Historical bbox/mask are evaluator-only. Runtime target selection uses the requested identity "
            "and a MobileCLIP gallery built from other saved captures."
        ),
        "warmup_seconds": warmup_seconds,
        "gallery": gallery_report,
        "summary": summary,
        "cases": cases,
    }
    with (output / "benchmark.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots-root", type=Path, default=PROJECT_ROOT / "runs/grasp_snapshots")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/perception_v2/benchmark")
    parser.add_argument("--yoloe-root", type=Path, default=PROJECT_ROOT / "dependencies/yoloe")
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "dependencies/yoloe/pretrain/yoloe-v8s-seg.pt"
    )
    parser.add_argument(
        "--mobileclip-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "dependencies/yoloe/mobileclip_blt.pt",
    )
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--save-visualizations", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
