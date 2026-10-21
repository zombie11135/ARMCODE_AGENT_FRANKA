"""Meaningful contracts at the perception/reasoning boundary; no trained models."""

import json

import cv2
import numpy as np
import pytest

from occlusion_reasoning.from_perception import export_report
from occlusion_reasoning.observations import InstanceEvidence, build_observation


def build(depth, instances, **kwargs):
    return build_observation(
        depth_m=depth, intrinsics={"fx": 100, "fy": 100, "cx": 2, "cy": 2},
        instances=instances, frame_id=kwargs.pop("frame_id", "f0"), query="target",
        alignment_status="recorded_aligned", **kwargs,
    )


def instance(mask, source="det0", score=.8, **kwargs):
    return InstanceEvidence(source, mask, [0, 0, 5, 5], score, **kwargs)


def test_occluded_thin_fragments_survive_and_invalid_depth_stays_unknown():
    depth = np.full((5, 5), .4, dtype=np.float32)
    mask = np.zeros((5, 5), bool)
    mask[1, :2], mask[4, 4], mask[0, 0], mask[0, 1] = True, True, True, True
    depth[4, 4] = .7  # legitimate different visible face; do not median-band away
    depth[0, 0], depth[0, 1] = 0, np.nan
    packet, arrays = build(depth, [instance(mask, identity_scores={"target": .92})])
    node = packet["relation_graph"]["nodes"][0]
    assert arrays[node["raw_mask_key"]].sum() == 5
    assert arrays[node["depth_supported_mask_key"]].sum() == 3
    assert node["quality"]["depth_missing_pixels"] == 2
    points = arrays[node["observed_points_camera_key"]]
    np.testing.assert_allclose(points[-1], [.014, .014, .7], atol=1e-7)
    assert packet["target_candidates"]["hypotheses"][0]["probability"] is None
    assert packet["target_candidates"]["hypotheses"][0]["identity_confirmed"] is False
    assert node["track_id"] is None


def test_duplicate_prompts_are_grouped_but_different_visible_masks_are_not():
    first = np.zeros((5, 5), bool)
    second = first.copy()
    first[:2, :] = True
    second[3:, :] = True
    rows = [
        instance(first, "box", .7), instance(first, "carton", .9),
        instance(second, "other", .8),
    ]
    packet, arrays = build(np.ones((5, 5)), rows)
    nodes = packet["relation_graph"]["nodes"]
    assert len(nodes) == 2  # identical bounding boxes are not enough to merge
    assert nodes[0]["source_ids"] == ["carton", "box"]
    assert nodes[0]["grouping"] == "high_mask_overlap_hypothesis"
    assert len(packet["proposals"]) == 3  # keep original alternatives
    assert arrays["unassigned_valid_depth"].sum() == 5  # includes background
    assert arrays["overlapping_node_masks"].sum() == 0
    assert packet["relation_graph"]["edges"] == []
    assert packet["relation_graph"]["status"] == "edges_not_inferred"


def test_camera_pose_transforms_observed_points_without_claiming_object_pose():
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    packet, arrays = build(np.ones((5, 5)), [instance(np.ones((5, 5)))],
                           reference_frame="base", T_reference_from_camera_optical=transform)
    node = packet["relation_graph"]["nodes"][0]
    camera = arrays[node["observed_points_camera_key"]]
    reference = arrays[node["observed_points_reference_key"]]
    np.testing.assert_allclose(reference - camera, np.tile([1, 2, 3], (25, 1)))
    assert "not_full_object_pose" in node["geometry_scope"]
    second, _ = build(np.ones((5, 5)), [instance(np.ones((5, 5)))], frame_id="f1")
    assert node["node_id"] != second["relation_graph"]["nodes"][0]["node_id"]


def test_no_detections_does_not_mean_empty_scene_or_absent_target():
    packet, arrays = build(np.ones((5, 5)), [])
    assert not packet["relation_graph"]["nodes"]
    assert arrays["unassigned_valid_depth"].all()
    hypotheses = packet["target_candidates"]["hypotheses"]
    assert {h["kind"] for h in hypotheses} == {"unobserved_or_missed", "target_absent"}
    assert all(h["probability"] is None for h in hypotheses)


def test_reject_mismatched_masks_and_invalid_camera_pose():
    with pytest.raises(ValueError, match="mask shape"):
        build(np.ones((5, 5)), [instance(np.ones((4, 5)))])
    with pytest.raises(ValueError, match="rigid transform"):
        build(np.ones((5, 5)), [], reference_frame="base",
              T_reference_from_camera_optical=np.zeros((4, 4)))


def save_fixture(tmp_path, alignment=None):
    snapshot, report_dir = tmp_path / "misleading_target_name", tmp_path / "perception"
    snapshot.mkdir()
    report_dir.mkdir()
    cv2.imwrite(str(snapshot / "color.png"), np.full((5, 5, 3), 100, np.uint8))
    cv2.imwrite(str(snapshot / "depth.png"), np.full((5, 5), 400, np.uint16))
    metadata = {"depth_scale": 1000, "depth_intrinsics": {"fx": 100, "fy": 100, "cx": 2, "cy": 2},
                "current_ee_pose_mrad": [1, 2, 3, 0, 0, 0], "query": "wrong_query"}
    if alignment is not None:
        metadata["depth_aligned_to_color"] = alignment
    (snapshot / "grasp_input.json").write_text(json.dumps(metadata))
    report = {
        "selected": {"predicted_identity": "evaluator_answer", "confidence": 1},
        "evaluator_reference": {"target_id": "evaluator_answer"},
        "instances": [
            {"instance_id": "candidate0", "bbox_xyxy": [0, 0, 5, 5], "confidence": .8,
             "class_name": "box", "evaluator_reference_iou": 1,
             "retrieval": {"predicted_identity": "evaluator_answer",
                           "class_scores": {"target": .92, "distractor": .91}}},
            {"instance_id": "candidate1", "bbox_xyxy": [0, 0, 5, 5], "confidence": .7},
        ],
    }
    for index, row in enumerate(report["instances"]):
        folder = report_dir / row["instance_id"]
        folder.mkdir()
        mask = np.zeros((5, 5), np.uint8)
        mask[index * 3:index * 3 + 2, :] = 1
        np.save(folder / "semantic_mask_raw.npy", mask)
    report_path = report_dir / "perception_report.json"
    report_path.write_text(json.dumps(report))
    return snapshot, report_path


def test_saved_adapter_retains_all_candidates_and_excludes_evaluator_fields(tmp_path):
    snapshot, report = save_fixture(tmp_path)
    packet = export_report(report, snapshot, tmp_path / "out",
                           query="target", frame_id="test", assume_aligned=True)
    assert packet["task"]["query"] == "target"
    assert len(packet["relation_graph"]["nodes"]) == 2
    assert packet["frame"]["T_reference_from_camera_optical"] is None
    assert packet["frame"]["rgb_depth_alignment"] == "assumed_aligned"
    assert "evaluator_answer" not in json.dumps(packet)
    assert "evaluator_reference_iou" not in json.dumps(packet)
    assert packet["provenance"]["legacy_current_ee_pose_ignored"]
    assert (tmp_path / "out" / "instances_overlay.png").is_file()
    with np.load(tmp_path / "out" / "observation_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["depth_m"], .4)


@pytest.mark.parametrize("alignment,assume", [(None, False), (False, True)])
def test_unknown_or_explicitly_bad_alignment_is_not_silently_accepted(tmp_path, alignment, assume):
    snapshot, report = save_fixture(tmp_path, alignment)
    with pytest.raises(ValueError, match="align"):
        export_report(report, snapshot, tmp_path / "out",
                      query="target", frame_id="test", assume_aligned=assume)
