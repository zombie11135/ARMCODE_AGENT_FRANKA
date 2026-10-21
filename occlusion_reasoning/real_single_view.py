"""Analyze saved camera-only RGB-D snapshots; no camera/robot connection.

Each view has its OWN optical reference frame, with an identity self-transform.
This enables local plane fitting without pretending to know an external camera
pose. The different reference frames deliberately fail the dual-view contract.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .observations import InstanceEvidence, build_observation
from .relations import extract_relations
from .planar import extract_planar_hypotheses
from .extract_relations import write_result


def prepare_observation(snapshot_dir, proposals):
    root = Path(snapshot_dir)
    metadata = json.loads((root / "grasp_input.json").read_text())
    if metadata.get("depth_aligned_to_color") is not True:
        raise ValueError("recorded RGB/depth alignment required")
    depth = np.load(root / "depth_m.npy", allow_pickle=False)
    color = cv2.imread(str(root / "color.png"))
    if color is None or color.shape[:2] != depth.shape:
        raise ValueError("aligned color and depth dimensions differ")
    intrinsic = metadata["depth_intrinsics"]
    if np.any(np.asarray(intrinsic.get("coeffs", []), float) != 0):
        raise ValueError("rectify nonzero lens distortion before pinhole occlusion reasoning")
    camera = metadata["camera_id"]
    instances = [InstanceEvidence(p["source_id"], p["mask"], p["bbox_xyxy"], p["confidence"])
                 for p in proposals]
    packet, arrays = build_observation(
        depth_m=depth,
        intrinsics=dict(fx=intrinsic["fx"], fy=intrinsic["fy"], cx=intrinsic["ppx"], cy=intrinsic["ppy"]),
        instances=instances, frame_id=f"{metadata['capture_group_id']}/{camera}", query="unspecified_target",
        alignment_status="recorded_aligned", timestamp_s=metadata["capture_timestamp_s"],
        camera_id=camera, capture_group_id=metadata["capture_group_id"],
        timestamp_domain=metadata["timestamp_domain"], reference_frame=f"{camera}_optical_local",
        T_reference_from_camera_optical=np.eye(4))
    packet["provenance"] = {
        "backend": "real_saved_rgbd_with_predicted_visible_masks",
        "snapshot_dir": str(root.resolve()),
        "rgb_sha256": hashlib.sha256((root / "color.png").read_bytes()).hexdigest(),
        "depth_sha256": hashlib.sha256((root / "depth_m.npy").read_bytes()).hexdigest(),
        "reference_frame_scope": "camera_local_self_transform_not_external_calibration",
        "external_camera_pose_available": False, "robot_control_initialized": False,
        "robot_state_queried": False, "target_identity_used": False,
    }
    return packet, arrays, color


def save_view(snapshot_dir, proposals, backend_report, output_dir, *, raw_proposals=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    packet, arrays, color = prepare_observation(snapshot_dir, proposals)
    packet["provenance"]["segmentation"] = backend_report
    observation = output_dir / "observation"
    observation.mkdir()
    packet["artifacts"] = {"arrays": "observation_arrays.npz"}
    np.savez_compressed(observation / "observation_arrays.npz", **arrays)
    (observation / "scene_observation.json").write_text(json.dumps(packet, indent=2, allow_nan=False))
    started = time.perf_counter()
    boundary = extract_relations(packet, arrays)
    planar = extract_planar_hypotheses(packet, arrays)
    reasoning_seconds = time.perf_counter()-started
    write_result(packet, arrays, boundary, output_dir / "boundary", color)
    (output_dir / "planar_relations.json").write_text(json.dumps(planar, indent=2, allow_nan=False))
    (output_dir / "segmentation_report.json").write_text(json.dumps({
        **backend_report, "proposals": [{k:v for k,v in p.items() if k != "mask"} for p in proposals]
    }, indent=2, allow_nan=False))
    if raw_proposals is not None:
        np.savez_compressed(output_dir / "raw_proposal_masks.npz",
                            **{p["source_id"]: p["mask"] for p in raw_proposals})
    canvas = color.copy()
    centers, labels, qualities = {}, {}, []
    for index, node in enumerate(packet["relation_graph"]["nodes"]):
        mask = arrays[node["raw_mask_key"]]
        y, x = np.nonzero(mask)
        if not len(x):
            continue
        tint = np.array([(59+index*83)%220, (177+index*47)%220, (211+index*101)%220])
        canvas[mask] = np.clip(.7*canvas[mask]+.3*tint, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(map(int, tint)), 2)
        center = (int(np.median(x)), int(np.median(y)))
        centers[node["node_id"]], labels[node["node_id"]] = center, f"N{index}"
        cv2.putText(canvas, f"N{index}", center, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        cv2.putText(canvas, f"N{index}", center, cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        touches_border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        qualities.append({"node_id": node["node_id"], "display_id": f"N{index}",
                          "touches_image_border": touches_border, **node["quality"]})
    candidates = [e for e in boundary["relation_graph"]["edges"] if e["relation"] == "possible_occludes"]
    candidates.extend(planar["edges"])
    pairs = set()
    named = []
    for edge in candidates:
        pair = (edge["source"], edge["target"])
        named.append({**edge, "foreground_display_id": labels.get(pair[0]),
                      "background_display_id": labels.get(pair[1])})
        if pair not in pairs and all(n in centers for n in pair):
            cv2.arrowedLine(canvas, centers[pair[0]], centers[pair[1]], (0, 165, 255),
                            2, cv2.LINE_AA, tipLength=.15)
        pairs.add(pair)
    canvas = cv2.copyMakeBorder(canvas, 65, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(canvas, f"{packet['frame']['camera_id']}: {len(qualities)} instance hypotheses, {len(pairs)} possible pairs",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Orange: hypotheses only | no cross-camera pose | no robot connection",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_dir / "evidence.png"), canvas):
        raise IOError("failed to save evidence overlay")
    summary = {"camera_id": packet["frame"]["camera_id"], "instance_hypotheses": len(qualities),
               "raw_proposals": len(proposals), "possible_occlusion_pairs": len(pairs),
               "candidates": named, "node_quality": qualities,
               "reasoning_seconds": reasoning_seconds, "accuracy_evaluated": False,
               "evaluation_status": "pipeline_smoke_test_no_human_occlusion_ground_truth",
               "cross_view_inference_run": False, "external_camera_pose_used": False,
               "robot_control_initialized": False, "robot_state_queried": False}
    (output_dir / "single_view_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/seur/data/wpy/models/sam3/sam3.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=.35)
    from perception_v2.proposal_refinement import DEFAULT_OCCLUSION_PROMPTS
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_OCCLUSION_PROMPTS))
    parser.add_argument("--proposal-mode", choices=["multiscale", "full"], default="multiscale")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from perception_v2.sam3_proposals import SAM3Proposals
    backend = SAM3Proposals(args.checkpoint, device=args.device, confidence=args.confidence)
    report = {"mode": "saved_camera_only_single_view", "cross_view_inference_run": False,
              "robot_control_initialized": False, "robot_state_queried": False,
              "accuracy_evaluated": False, "human_ground_truth_available": False,
              "proposal_mode": args.proposal_mode, "prompts": args.prompts,
              "confidence_threshold": args.confidence,
              "model_load_seconds": backend.load_seconds, "views": {}}
    capture_report = json.loads((args.capture_dir / 'capture_report.json').read_text())
    cameras = tuple(capture_report['cameras'])
    if not cameras or not set(cameras) <= {'global', 'wrist'}:
        raise ValueError('capture report must list wrist/global camera snapshots')
    for camera in cameras:
        snapshot = args.capture_dir / camera
        color = cv2.imread(str(snapshot / "color.png"))
        if color is None:
            raise FileNotFoundError(snapshot / "color.png")
        raw = None
        if args.proposal_mode == "multiscale":
            from perception_v2.proposal_refinement import refine_proposals
            proposals, backend_report, raw = refine_proposals(color, args.prompts, backend)
        else:
            proposals, backend_report = backend.predict(color, args.prompts)
        report["views"][camera] = save_view(snapshot, proposals, backend_report,
                                           args.output_dir / camera, raw_proposals=raw)
        print(json.dumps({"camera": camera, "instances": report["views"][camera]["instance_hypotheses"],
                          "possible_pairs": report["views"][camera]["possible_occlusion_pairs"]}), flush=True)
    (args.output_dir / "test_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()
