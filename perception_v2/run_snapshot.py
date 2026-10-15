from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .geometry import (
    build_instance_geometry,
    load_camera_to_ee,
    load_depth_m,
    mask_iou,
    pose_mrad_to_transform,
    write_ply,
)
from .retrieval import MobileCLIPGallery, masked_crop, snapshot_identity
from .yoloe_backend import YOLOEInstanceSegmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ["medicine carton", "pharmaceutical package", "printed rectangular box"]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)
    np.save(str(path.with_suffix(".npy")), np.asarray(mask, dtype=np.uint8))


def save_overlay(path: Path, color: np.ndarray, mask: np.ndarray, bbox, label: str) -> None:
    overlay = color.copy().astype(np.float32)
    active = np.asarray(mask, dtype=bool)
    overlay[active] = 0.55 * overlay[active] + 0.45 * np.array([180, 60, 220], np.float32)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 220, 40), 2)
    cv2.putText(overlay, label, (max(0, x1), max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3)
    cv2.putText(overlay, label, (max(0, x1), max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.clip(overlay, 0, 255).astype(np.uint8))


def _comparison_panel(
    color: np.ndarray,
    mask: Optional[np.ndarray],
    title: str,
    tint_bgr: tuple[int, int, int],
) -> np.ndarray:
    panel = color.copy().astype(np.float32)
    if mask is not None:
        active = np.asarray(mask, dtype=bool)
        panel[active] = 0.55 * panel[active] + 0.45 * np.asarray(tint_bgr, np.float32)
    panel = np.clip(panel, 0, 255).astype(np.uint8)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        panel,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def save_comparison(
    path: Path,
    color: np.ndarray,
    historical_mask: Optional[np.ndarray],
    semantic_mask: np.ndarray,
    visible_mask: np.ndarray,
    grasp_mask: np.ndarray,
) -> None:
    """Save a compact diagnostic, keeping evaluator and runtime masks visibly separate."""
    panels = [
        _comparison_panel(
            color,
            historical_mask,
            "Historical mask (evaluator only)" if historical_mask is not None else "RGB input",
            (30, 30, 230),
        ),
        _comparison_panel(color, semantic_mask, "YOLOE semantic mask", (180, 60, 220)),
        _comparison_panel(color, visible_mask, "Depth-clean visible surface", (230, 180, 40)),
        _comparison_panel(color, grasp_mask, "Eroded grasp mask", (40, 220, 40)),
    ]
    height, width = color.shape[:2]
    target_width = min(width, 800)
    target_height = max(1, int(round(height * target_width / width)))
    panels = [cv2.resize(panel, (target_width, target_height)) for panel in panels]
    canvas = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def camera_to_base_from_snapshot(
    grasp_input: Dict[str, Any], handeye_path: Optional[Path]
) -> Optional[np.ndarray]:
    pose = grasp_input.get("capture_ee_pose_mrad") or grasp_input.get("current_ee_pose_mrad")
    if pose is None or handeye_path is None:
        return None
    return pose_mrad_to_transform(pose) @ load_camera_to_ee(handeye_path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    snapshot = args.snapshot_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (snapshot / "grasp_input.json").open("r", encoding="utf-8") as stream:
        grasp_input = json.load(stream)
    color_path = snapshot / "color.png"
    depth_path = snapshot / "depth.png"
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(color_path)
    depth_m = load_depth_m(depth_path, float(grasp_input.get("depth_scale", 1000.0)))
    intrinsics = grasp_input.get("depth_intrinsics") or grasp_input.get("color_intrinsics")
    if intrinsics is None:
        raise ValueError("snapshot has no camera intrinsics")

    backend = YOLOEInstanceSegmenter(
        yoloe_root=args.yoloe_root,
        checkpoint=args.checkpoint,
        prompts=args.prompts,
        device=args.device,
        confidence=args.confidence,
        image_size=args.image_size,
    )
    warmup_seconds = backend.warmup(color.shape) if args.warmup else None
    prediction = backend.predict(color)
    instances = prediction["instances"]

    reference_mask = None
    reference_metrics = None
    reference_path = snapshot / "mask.npy"
    if args.evaluate_reference_mask and reference_path.exists():
        reference_mask = np.load(str(reference_path)) > 0
        for instance in instances:
            instance["evaluator_reference_iou"] = mask_iou(instance["mask"], reference_mask)
        reference_metrics = {
            "reference_kind": "historical_pipeline_mask_not_human_ground_truth",
            "best_candidate_iou": max(
                (item["evaluator_reference_iou"] for item in instances), default=0.0
            ),
        }

    for instance in instances:
        instance_dir = output / instance["instance_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        save_mask(instance_dir / "semantic_mask_raw.png", instance["mask"])
        save_overlay(
            instance_dir / "semantic_overlay.png",
            color,
            instance["mask"],
            instance["bbox_xyxy"],
            "{} {:.3f}".format(instance["class_name"], instance["confidence"]),
        )

    retrieval_report = None
    requested_identity = args.query or snapshot_identity(snapshot, grasp_input)
    if args.query:
        identity_source = "cli_query"
    elif str(grasp_input.get("query") or "").strip():
        identity_source = "snapshot_grasp_input_query"
    elif requested_identity:
        identity_source = "legacy_snapshot_directory_name"
    else:
        identity_source = None
    retrieval_used = bool(args.gallery_root is not None and instances and requested_identity)
    if retrieval_used:
        gallery = MobileCLIPGallery(args.mobileclip_checkpoint, device=args.device)
        gallery_report = gallery.build_from_snapshots(args.gallery_root)
        crops = [masked_crop(color, item["mask"], item["bbox_xyxy"]) for item in instances]
        features = gallery.encode(crops)
        scores = gallery.scores(features, requested_identity, exclude_snapshot=snapshot)
        for instance, score in zip(instances, scores):
            instance["retrieval"] = score
        selected = max(
            instances,
            key=lambda item: (
                item["retrieval"]["requested_score"]
                if item["retrieval"]["requested_score"] is not None
                else -1.0
            ),
        )
        retrieval_report = {
            "gallery": gallery_report,
            "requested_identity": requested_identity,
            "requested_identity_source": identity_source,
            "selection_policy": "requested_identity_mobileclip_gallery",
        }
    else:
        # Runtime selection uses model evidence only. Historical mask is evaluator-only.
        selected = max(instances, key=lambda item: item["confidence"], default=None)
        if args.gallery_root is not None and instances and not requested_identity:
            retrieval_report = {
                "status": "unavailable",
                "reason": "missing_requested_identity",
                "selection_policy": "highest_yoloe_confidence_fallback",
            }
    selected_report = None
    if selected is not None:
        camera_to_base = camera_to_base_from_snapshot(
            grasp_input, args.handeye_config.resolve() if args.handeye_config else None
        )
        geometry = build_instance_geometry(
            color,
            depth_m,
            selected["mask"],
            intrinsics,
            camera_to_base=camera_to_base,
            erosion_px=args.erosion_px,
        )
        selected_dir = output / selected["instance_id"]
        save_mask(selected_dir / "visible_surface_mask.png", geometry["visible_surface_mask"])
        save_mask(selected_dir / "grasp_mask.png", geometry["grasp_mask"])
        save_comparison(
            output / "comparison.png",
            color,
            reference_mask,
            selected["mask"],
            geometry["visible_surface_mask"],
            geometry["grasp_mask"],
        )
        write_ply(selected_dir / "pointcloud_camera.ply", geometry["points_camera"], geometry["colors"])
        if geometry["points_base"] is not None:
            write_ply(selected_dir / "pointcloud_xarm_base.ply", geometry["points_base"], geometry["colors"])
        selected_report = {
            "selection_policy": (
                "requested_identity_mobileclip_gallery"
                if retrieval_used
                else "highest_yoloe_confidence_no_evaluator_input"
            ),
            "instance_id": selected["instance_id"],
            "class_name": selected["class_name"],
            "confidence": selected["confidence"],
            "bbox_xyxy": selected["bbox_xyxy"],
            "identity_status": (
                "retrieval_candidate_unverified"
                if retrieval_used
                else "unavailable_in_mask_backend"
            ),
            "retrieval": selected.get("retrieval"),
            "quality": geometry["quality"],
            "camera_cloud_stats": geometry["camera_cloud_stats"],
            "base_cloud_stats": geometry["base_cloud_stats"],
            "artifacts": {
                "semantic_mask": str(selected_dir / "semantic_mask_raw.npy"),
                "visible_surface_mask": str(selected_dir / "visible_surface_mask.npy"),
                "grasp_mask": str(selected_dir / "grasp_mask.npy"),
                "comparison": str(output / "comparison.png"),
                "pointcloud_camera": str(selected_dir / "pointcloud_camera.ply"),
                "pointcloud_xarm_base": (
                    str(selected_dir / "pointcloud_xarm_base.ply")
                    if geometry["points_base"] is not None
                    else None
                ),
            },
        }

    serializable_instances = [
        {key: value for key, value in item.items() if key != "mask"} for item in instances
    ]
    report = {
        "status": "ok",
        "offline_only": True,
        "snapshot": str(snapshot),
        "requested_identity": requested_identity,
        "requested_identity_source": identity_source,
        "model": {
            "backend": "official_thu_mig_yoloe",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint.resolve()),
            "prompts": args.prompts,
            "confidence": args.confidence,
            "image_size": args.image_size,
            "device": args.device,
        },
        "timing": {
            "warmup_seconds": warmup_seconds,
            "prediction_wall_seconds": prediction["elapsed_seconds"],
            "model_speed_ms": prediction["model_speed_ms"],
        },
        "instances": serializable_instances,
        "selected": selected_report,
        "retrieval": retrieval_report,
        "evaluator_reference": reference_metrics,
    }
    with (output / "perception_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline YOLOE RGB-D perception on a saved snapshot.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yoloe-root", type=Path, default=PROJECT_ROOT / "dependencies/yoloe")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "dependencies/yoloe/pretrain/yoloe-v8s-seg.pt",
    )
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--erosion-px", type=int, default=2)
    parser.add_argument(
        "--handeye-config",
        type=Path,
        default=PROJECT_ROOT / "config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--evaluate-reference-mask", action="store_true")
    parser.add_argument("--gallery-root", type=Path)
    parser.add_argument("--query")
    parser.add_argument(
        "--mobileclip-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "dependencies/yoloe/mobileclip_blt.pt",
    )
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
