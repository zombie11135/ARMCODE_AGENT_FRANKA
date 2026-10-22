"""Offline geometric fusion for a stationary camera pair with measured extrinsics.

The reference is the global optical frame, not a robot base. This command never
connects to cameras, robot state, motion or gripper services. The relative pose
is valid only while both cameras (including the wrist camera) stay fixed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .extract_multiview import load_observation
from .observations import _validate_transform
from .scene import infer_scene
from .relations import extract_relations
from .multiview import MultiViewConfig, _prepare, _project
from .planar import PlanarConfig


def relative_pose_from_board(T_global_from_board, T_wrist_from_board):
    """Both board poses must describe the SAME stationary board at one instant.

    Numeric composition only: this neither detects a board nor validates a
    calibration. Repeated independent board poses/reprojection checks are needed.
    """
    global_pose = _validate_transform(T_global_from_board)
    wrist_pose = _validate_transform(T_wrist_from_board)
    return global_pose @ np.linalg.inv(wrist_pose)


def bind_static_pair(observations, calibration, serials):
    if calibration.get("schema") != "static_camera_pair_v1":
        raise ValueError("expected static_camera_pair_v1 calibration")
    if calibration.get("status") not in {"measured", "validated"}:
        raise ValueError("relative camera pose has not been measured")
    if (calibration.get("translation_unit") != "m"
            or calibration.get("matrix_convention") != "p_global = T_global_from_wrist @ p_wrist"):
        raise ValueError("calibration requires metric optical coordinates and explicit transform direction")
    if calibration.get("validity") != "both_camera_poses_unchanged":
        raise ValueError("calibration must declare its stationary camera-pair scope")
    if calibration.get("camera_serials") != serials or set(serials) != {"global", "wrist"}:
        raise ValueError("calibration camera serials do not match the captured devices")
    if len(set(serials.values())) != 2:
        raise ValueError("two different camera devices are required")
    if len(observations) != 2 or {p["frame"]["camera_id"] for p, _ in observations} != {"global", "wrist"}:
        raise ValueError("one global and one wrist observation are required")
    pose = _validate_transform(calibration["T_global_optical_from_wrist_optical"])
    bound = []
    for packet, original_arrays in observations:
        packet, arrays = copy.deepcopy(packet), dict(original_arrays)
        frame = packet["frame"]
        camera = frame["camera_id"]
        transform = np.eye(4) if camera == "global" else pose
        frame["reference_frame"] = "measured_static_global_optical"
        frame["T_reference_from_camera_optical"] = transform.tolist()
        # Keep exported reference clouds coherent too. Core inference still
        # recomputes geometry from raw visible masks and depth, never this cache.
        k = frame["intrinsics"]
        depth = np.asarray(arrays["depth_m"])
        for node in packet["relation_graph"]["nodes"]:
            raw = np.asarray(arrays[node["raw_mask_key"]], bool)
            y, x = np.nonzero(raw & np.isfinite(depth) & (depth > 0))
            z = depth[y, x].astype(float)
            points = np.column_stack(((x-k["cx"])*z/k["fx"], (y-k["cy"])*z/k["fy"], z))
            key = node.get("observed_points_reference_key") or node["node_id"]+"_reference_points"
            node["observed_points_reference_key"] = key
            arrays[key] = points @ transform[:3, :3].T + transform[:3, 3]
        packet.setdefault("provenance", {}).update({
            "reference_frame_scope": "measured_stationary_camera_pair_global_optical_reference",
            "external_camera_pose_available": True,
            "external_pose_type": "relative_camera_pair_not_robot_base_pose",
            "camera_pair_stationarity_assumed": True,
            "robot_control_initialized": False, "robot_state_queried": False,
        })
        bound.append((packet, arrays))
    return bound


def render_fusion(observations, result, output):
    images = []
    for packet, arrays in observations:
        camera = packet["frame"]["camera_id"]
        rgb = Path(packet["provenance"]["snapshot_dir"])/"color.png"
        canvas = cv2.imread(str(rgb))
        if canvas is None:
            raise FileNotFoundError(rgb)
        for node in packet["relation_graph"]["nodes"]:
            mask = np.asarray(arrays[node["raw_mask_key"]], np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (80, 255, 80), 1)
            y, x = np.nonzero(mask)
            if len(x):
                label = result["node_to_entity"][node["node_id"]]
                cv2.putText(canvas, label, (int(np.median(x)), int(np.median(y))),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        for edge in result["relation_graph"]["edges"]:
            if edge["view_id"] != camera:
                continue
            color = (255, 0, 255) if edge["relation"] == "occludes_observed_surface" else (0, 165, 255)
            for x, y in edge.get("support_pixels_sample_xy", []):
                cv2.circle(canvas, (x, y), 1, color, -1)
        canvas = cv2.copyMakeBorder(canvas, 60, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(canvas, f"{camera} | geometry: {result['geometry_health']['status']}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Entity IDs | magenta: cross-view evidence; orange: local prior",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .39, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(canvas)
    height = max(im.shape[0] for im in images)
    images = [cv2.copyMakeBorder(im, 0, height-im.shape[0], 0, 0, cv2.BORDER_CONSTANT) for im in images]
    if not cv2.imwrite(str(output), np.hstack(images)):
        raise IOError(output)


def render_alignment(observations, result, output_dir):
    """Show actual projection depth discrepancies, without calling them calibration accuracy."""
    config = MultiViewConfig(**result['config'])
    prepared = [_prepare(packet, arrays, config) for packet, arrays in observations]
    canvases = [cv2.imread(str(Path(packet['provenance']['snapshot_dir'])/'color.png'))
                for packet, _ in observations]
    rows = []
    for s, source in enumerate(prepared):
        destination = prepared[1-s]
        for node, points in zip(source['nodes'], source['points']):
            projection = _project(source, destination, points, config)
            x, y = projection['pixels'].T
            measured = destination['depth'][y, x]
            valid = (projection['owner'] >= 0) & np.isfinite(measured) & (measured > 0)
            delta = projection['delta'][valid]
            if not len(delta):
                continue
            rows.append({'source_node': node['node_id'], 'destination_camera': destination['frame']['camera_id'],
                         'tested_interior_pixels': len(delta),
                         'depth_delta_q10_q50_q90_mm': (np.quantile(delta, [.1, .5, .9])*1000).tolist(),
                         'in_front_inconsistent_pixels': int((delta < -config.occlusion_depth_margin_m).sum()),
                         'meaning': 'projected_depth_minus_measured_depth_not_calibration_error_bound'})
            for selection, color in [
                    (projection['same'], (0, 220, 0)),
                    (projection['behind'], (0, 165, 255)),
                    (projection['delta'] < -config.occlusion_depth_margin_m, (0, 0, 255))]:
                pixels = projection['pixels'][selection & valid]
                stride = max(1, int(np.ceil(len(pixels)/2000)))
                for u, v in pixels[::stride]:
                    cv2.circle(canvases[1-s], (int(u), int(v)), 1, color, -1)
    output_dir = Path(output_dir)
    (output_dir/'alignment_report.json').write_text(json.dumps({
        'geometry_health': result['geometry_health'], 'projections': rows,
        'accuracy_evaluated': False,
        'legend': {'green': 'within_same_surface_depth_tolerance',
                   'red': 'projected_surface_in_measured_free_space',
                   'orange': 'projected_surface_behind_measured_depth_not_necessarily_distinct_object'}
    }, indent=2, allow_nan=False))
    panels = []
    for view, canvas in zip(prepared, canvases):
        canvas = cv2.copyMakeBorder(canvas, 45, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(canvas, view['frame']['camera_id']+' | green: agree; red: inconsistent; orange: behind',
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1)
        panels.append(canvas)
    height = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, height-p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
    if not cv2.imwrite(str(output_dir/'alignment.png'), np.hstack(panels)):
        raise IOError('Could not save alignment diagnostic image')


def fuse(analysis_dir, calibration_path, output_dir, *, config=None):
    analysis_dir, calibration_path, output_dir = map(Path, (analysis_dir, calibration_path, output_dir))
    calibration = json.loads(calibration_path.read_text())
    loaded = [load_observation(analysis_dir/c/"observation/scene_observation.json")
              for c in ("global", "wrist")]
    observations = [item[0] for item in loaded]
    serials = {}
    for packet, arrays in observations:
        source = Path(packet["provenance"]["snapshot_dir"])
        metadata = json.loads((source/"grasp_input.json").read_text())
        if hashlib.sha256((source/"color.png").read_bytes()).hexdigest() != packet["provenance"]["rgb_sha256"]:
            raise ValueError("saved image no longer matches its observation packet")
        if hashlib.sha256((source/"depth_m.npy").read_bytes()).hexdigest() != packet["provenance"]["depth_sha256"]:
            raise ValueError("saved depth no longer matches its observation packet")
        if not np.array_equal(arrays['depth_m'], np.load(source/'depth_m.npy', allow_pickle=False), equal_nan=True):
            raise ValueError('Observation depth arrays differ from the original saved RGB-D')
        serials[packet["frame"]["camera_id"]] = metadata["device_serial"]
    base_bound = calibration.get('schema') == 'robot_base_capture_geometry_v1'
    if base_bound:
        from .robot_pair import bind_base_pair, verify_snapshot
        geometry_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        for packet, _ in observations:
            camera = packet['frame']['camera_id']
            source = Path(packet['provenance']['snapshot_dir'])
            metadata = json.loads((source/'grasp_input.json').read_text())
            if (packet['provenance'].get('geometry_sha256') != geometry_hash
                    or metadata.get('geometry_sha256') != geometry_hash):
                raise ValueError('Observation and capture geometry hashes differ; reanalyze the matching capture')
            verify_snapshot(calibration, camera, metadata, source)
        bound = bind_base_pair(observations, calibration, serials)
    else:
        bound = bind_static_pair(observations, calibration, serials)
    config = config or {}
    multiview_options = ({'max_timestamp_delta_s': calibration['max_frame_receive_skew_s']}
                         if base_bound else {})
    multiview_options.update(config.get('multiview', {}))
    result = infer_scene(bound,
        multiview_config=MultiViewConfig(**multiview_options),
        planar_config=PlanarConfig(**config.get('planar', {})))
    result["single_view_boundary_evidence"] = []
    for packet, arrays in bound:
        boundary = extract_relations(packet, arrays)
        result["single_view_boundary_evidence"].append(boundary)
        for edge in boundary["relation_graph"]["edges"]:
            result["relation_graph"]["edges"].append({
                **edge,
                "source": result["node_to_entity"][edge["source"]],
                "target": result["node_to_entity"][edge["target"]],
                "observation_source_node": edge["source"],
                "observation_target_node": edge["target"],
                "view_id": packet["frame"]["camera_id"],
                "evidence": ("single_view_boundary_convex_completion_prior"
                             if edge["relation"] == "possible_occludes" else "single_view_boundary_depth_order"),
            })
    result["real_camera_fusion"] = {
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "input_provenance": [item[1] for item in loaded],
        "coordinate_reference": "fr3_link0" if base_bound else "global_camera_optical_not_robot_base",
        "camera_pair_and_scene_stationarity_assumed": True,
        "exposure_synchronization_verified": False,
        "robot_control_initialized": False, "robot_state_queried": False,
        "accuracy_evaluated": False,
    }
    if base_bound:
        result['real_camera_fusion'].update(
            capture_robot_state_queried=True,
            calibration_status={k:v['status'] for k,v in calibration['calibrations'].items()},
            operator_verification_note=calibration.get('operator_verification_note', ''),
            validity='this_stationary_capture_group_only',
            robot_motion_commanded=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    for packet, arrays in bound:
        directory = output_dir/packet["frame"]["camera_id"]
        directory.mkdir()
        packet["artifacts"] = {"arrays": "observation_arrays.npz"}
        (directory/"scene_observation.json").write_text(json.dumps(packet, indent=2, allow_nan=False))
        np.savez_compressed(directory/"observation_arrays.npz", **arrays)
    (output_dir/"multiview_relations.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    render_fusion(bound, result, output_dir/"evidence.png")
    render_alignment(bound, result, output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument('--config', type=Path, help='JSON with multiview/planar parameter dictionaries')
    args = parser.parse_args()
    result = fuse(args.analysis_dir, args.calibration, args.output_dir,
                  config=json.loads(args.config.read_text()) if args.config else None)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "geometry_health": result["geometry_health"]["status"],
                      "all_relation_edges": len(result["relation_graph"]["edges"]),
                      "accepted_cross_view_associations": sum(a['status'] == 'accepted' for a in result['associations']),
                      "measured_cross_view_occlusion_edges": sum(e['relation'] == 'occludes_observed_surface'
                                                                for e in result['relation_graph']['edges']),
                      "single_view_possible_occlusion_edges": sum(e['relation'] == 'possible_occludes'
                                                                 for e in result['relation_graph']['edges']),
                      "occlusion_evidence_edges": sum(e["relation"] in {"occludes_observed_surface", "possible_occludes"}
                                                      for e in result["relation_graph"]["edges"])}))


if __name__ == "__main__":
    main()
