"""Offline adapter for saved perception_v2 reports; no model/hardware imports.

Usage:
    python -m occlusion_reasoning.from_perception --report REPORT.json
        --snapshot-dir SNAPSHOT --query TARGET --frame-id FRAME --output-dir OUT

Legacy snapshots without alignment metadata require --assume-aligned. This is
recorded as an assumption, not a calibration check. Optional --camera-pose reads
JSON containing reference_frame and T_reference_from_camera_optical (metres).
It must describe this capture; current robot poses are deliberately not reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from perception_v2.geometry import load_depth_m
from .observations import InstanceEvidence, build_observation


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_report(
    report_path: Path,
    snapshot_dir: Path,
    output_dir: Path,
    *,
    query: str,
    frame_id: str,
    assume_aligned: bool = False,
    camera_pose: dict | None = None,
    duplicate_iou: float = 0.90,
) -> dict:
    report_path, snapshot_dir = report_path.resolve(), snapshot_dir.resolve()
    output_dir = output_dir.resolve()
    # Never overwrite the input snapshots or model outputs.
    if output_dir == snapshot_dir or output_dir == report_path.parent:
        raise ValueError("output_dir must differ from the input directories")
    report = json.loads(report_path.read_text())
    metadata_path = snapshot_dir / "grasp_input.json"
    metadata = json.loads(metadata_path.read_text())
    recorded_alignment = metadata.get("depth_aligned_to_color")
    if recorded_alignment is False:
        raise ValueError("depth is explicitly marked unaligned; register it before export")
    if recorded_alignment is not True and not assume_aligned:
        raise ValueError("alignment metadata missing; use --assume-aligned only for known aligned captures")
    if metadata.get("depth_scale") is None:
        raise ValueError("snapshot must specify depth_scale; units cannot be guessed")
    depth_scale = float(metadata["depth_scale"])
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    color_path, depth_path = snapshot_dir / "color.png", snapshot_dir / "depth.png"
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(color_path)
    depth = load_depth_m(depth_path, depth_scale)
    if depth.shape != color.shape[:2]:
        raise ValueError("RGB and aligned depth shapes differ")
    intrinsic = metadata.get("depth_intrinsics") or metadata.get("color_intrinsics")
    if not intrinsic:
        raise ValueError("aligned camera intrinsics missing")
    instances = []
    for row in report["instances"]:
        source_id = str(row["instance_id"])
        if source_id in {".", ".."} or Path(source_id).name != source_id:
            raise ValueError("invalid source instance directory")
        mask_path = report_path.parent / source_id / "semantic_mask_raw.npy"
        mask = np.load(mask_path, allow_pickle=False)
        # Explicit field allowlist: selected/evaluator/legacy directory identities
        # must not become task identities or geometric evidence.
        retrieval = row.get("retrieval") or {}
        scores = retrieval.get("class_scores") or {}
        instances.append(InstanceEvidence(
            source_id=source_id, mask=mask, bbox_xyxy=row["bbox_xyxy"],
            detection_score=row["confidence"], category=row.get("class_name"),
            identity_scores=scores,
        ))
    pose = camera_pose or {}
    packet, arrays = build_observation(
        depth_m=depth, intrinsics=intrinsic, instances=instances,
        frame_id=frame_id, query=query,
        alignment_status="recorded_aligned" if recorded_alignment is True else "assumed_aligned",
        timestamp_s=metadata.get("capture_timestamp_s"),
        camera_id=metadata.get("camera_id"),
        capture_group_id=metadata.get("capture_group_id"),
        timestamp_domain=metadata.get("timestamp_domain"),
        reference_frame=pose.get("reference_frame"),
        T_reference_from_camera_optical=pose.get("T_reference_from_camera_optical"),
        duplicate_iou=duplicate_iou,
    )
    packet["provenance"] = {
        "backend": "perception_v2_saved_report", "report_path": str(report_path),
        "snapshot_dir": str(snapshot_dir), "report_sha256": _digest(report_path),
        "rgb_sha256": _digest(color_path), "depth_sha256": _digest(depth_path),
        "pairing_status": "user_supplied_snapshot_pairing_not_capture_hash_verified",
        "detector_prompts": (report.get("model") or {}).get("prompts"),
        "scope": "only_detected_candidates_no_complete_scene_coverage_claim",
    }
    if metadata.get("capture_ee_pose_mrad") is not None:
        packet["provenance"]["capture_ee_pose_available"] = True
    if metadata.get("current_ee_pose_mrad") is not None and not pose:
        packet["provenance"]["legacy_current_ee_pose_ignored"] = True
    if "predicted_identity" in (report.get("selected") or {}):
        packet["provenance"]["selected_identity_not_used"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "observation_arrays.npz", **arrays)
    canvas = color.copy()
    for index, node in enumerate(packet["relation_graph"]["nodes"]):
        mask = arrays[node["raw_mask_key"]]
        tint = np.array([(59 + 83 * index) % 255, (177 + 47 * index) % 255,
                         (211 + 101 * index) % 255], dtype=float)
        canvas[mask] = np.clip(0.65 * canvas[mask] + 0.35 * tint, 0, 255)
        x1, y1, x2, y2 = map(lambda x: int(round(x)), node["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(int(x) for x in tint), 2)
        caption = f"instance_{index:03d} / {len(node['source_ids'])} proposal(s)"
        cv2.putText(canvas, caption, (max(0, x1), max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
        cv2.putText(canvas, caption, (max(0, x1), max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imwrite(str(output_dir / "instances_overlay.png"), canvas)
    packet["artifacts"] = {"arrays": "observation_arrays.npz", "overlay": "instances_overlay.png"}
    (output_dir / "scene_observation.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--assume-aligned", action="store_true")
    parser.add_argument("--camera-pose", type=Path)
    parser.add_argument("--duplicate-iou", type=float, default=.90)
    args = parser.parse_args()
    packet = export_report(
        args.report, args.snapshot_dir, args.output_dir, query=args.query,
        frame_id=args.frame_id, assume_aligned=args.assume_aligned,
        camera_pose=json.loads(args.camera_pose.read_text()) if args.camera_pose else None,
        duplicate_iou=args.duplicate_iou,
    )
    print(json.dumps({
        "output": str(args.output_dir.resolve()),
        "proposals": len(packet["proposals"]),
        "instance_hypotheses": len(packet["relation_graph"]["nodes"]),
        "alignment": packet["frame"]["rgb_depth_alignment"],
        "graph_status": packet["relation_graph"]["status"],
        "belief_status": packet["target_candidates"]["status"],
        "missing": packet["missing_for_full_reasoning"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
