"""Contract and counterexample tests for single-frame relation hypotheses."""

import copy
import json

import numpy as np
import pytest

from occlusion_reasoning.observations import InstanceEvidence, build_observation
from occlusion_reasoning.relations import RelationConfig, extract_relations
from occlusion_reasoning.extract_relations import export_relations, visualize_relations


def packet_for(masks, depths=None):
    shape = masks[0].shape
    depth = np.full(shape, .9, np.float32)
    instances = []
    for i, mask in enumerate(masks):
        if depths is not None:
            depth[mask] = depths[i]
        instances.append(InstanceEvidence(f"d{i}", mask, [0, 0, shape[1], shape[0]], .9))
    return build_observation(
        depth_m=depth, intrinsics={"fx": 100, "fy": 100, "cx": 40, "cy": 40},
        instances=instances, frame_id="frame", query="target", alignment_status="recorded_aligned")


def covered_box():
    front = np.zeros((80, 80), bool)
    front[15:45, 30:50] = True
    back = np.zeros_like(front)
    back[25:65, 15:65] = True
    back &= ~front
    return packet_for([front, back], [.4, .45])


def occlusion_edges(result):
    return [e for e in result["relation_graph"]["edges"] if e["relation"] == "possible_occludes"]


def test_partial_cover_has_local_depth_evidence_and_directed_hypothesis():
    packet, arrays = covered_box()
    result = extract_relations(packet, arrays)
    pair = result["pairs"][0]
    assert pair["status"] == "possible_occlusion"
    assert pair["depth"]["order"] == "a_closer"
    assert pair["depth"]["median_b_minus_a_m"] == pytest.approx(.05)
    assert pair["completion"]["front_overlap_pixels"] > 100
    edge = occlusion_edges(result)[0]
    assert edge["source"] == packet["relation_graph"]["nodes"][0]["node_id"]
    assert edge["target"] == packet["relation_graph"]["nodes"][1]["node_id"]
    assert edge["confirmed"] is False and edge["calibrated_probability"] is None
    assert packet["relation_graph"]["edges"] == []  # input is not mutated
    json.dumps(result, allow_nan=False)


def test_identity_scores_and_node_order_do_not_change_physical_direction():
    packet, arrays = covered_box()
    original = occlusion_edges(extract_relations(packet, arrays))[0]
    packet["task"]["query"] = "completely different task"
    packet["target_candidates"] = {"probability": 1, "fake_ground_truth": "hidden"}
    packet["relation_graph"]["nodes"].reverse()
    second = occlusion_edges(extract_relations(packet, arrays))[0]
    assert (original["source"], original["target"]) == (second["source"], second["target"])


def test_neighboring_convex_boxes_at_different_depth_are_not_proven_occluders():
    a = np.zeros((80, 80), bool)
    a[20:60, 10:30] = True
    b = np.zeros_like(a)
    b[20:60, 30:50] = True
    packet, arrays = packet_for([a, b], [.4, .45])
    result = extract_relations(packet, arrays)
    assert result["pairs"][0]["status"] == "local_depth_order_only"
    assert not occlusion_edges(result)
    assert result["relation_graph"]["edges"][0]["scope"] == "sampled_boundary_only"


def test_thin_visible_rim_does_not_dilute_shared_boundary_evidence():
    front = np.zeros((80, 80), bool)
    front[20:60, 20:60] = True
    back = np.zeros_like(front)
    back[17:63, 17:63] = True
    back &= ~front
    packet, arrays = packet_for([front, back], [.4, .45])
    pair = extract_relations(packet, arrays)["pairs"][0]
    assert pair["status"] == "possible_occlusion"
    assert pair["boundary"]["contact_fraction"] == pytest.approx(1)


def test_nearby_enclosing_contour_with_visible_background_gap_is_not_contact():
    front = np.zeros((80, 80), bool)
    front[25:55, 25:55] = True
    back = np.zeros_like(front)
    back[15:65, 15:65] = True
    back[22:58, 22:58] = False  # three pixels of observed background around the foreground
    packet, arrays = packet_for([front, back], [.4, .45])
    result = extract_relations(packet, arrays)
    assert result["pairs"][0]["status"] == "local_depth_order_only"
    assert not occlusion_edges(result)


def test_large_bbox_overlap_does_not_replace_boundary_evidence():
    a = np.zeros((80, 80), bool)
    a[10:20, 10:20] = True
    b = np.zeros_like(a)
    b[60:70, 60:70] = True
    packet, arrays = packet_for([a, b], [.4, .5])  # both boxes cover entire image
    result = extract_relations(packet, arrays)
    assert result["pairs"][0]["status"] == "no_near_boundary_evidence"
    assert not result["relation_graph"]["edges"]


@pytest.mark.parametrize("missing", [0, np.nan])
def test_depth_holes_remain_unknown_without_interior_or_neighbor_fabrication(missing):
    packet, arrays = covered_box()
    mask = arrays[packet["relation_graph"]["nodes"][1]["raw_mask_key"]]
    arrays["depth_m"][mask] = missing
    result = extract_relations(packet, arrays)
    pair = result["pairs"][0]
    assert pair["status"] == "insufficient_local_depth"
    assert pair["depth"]["valid_pair_count"] == 0
    assert not result["relation_graph"]["edges"]
    json.dumps(result, allow_nan=False)


def test_sub_threshold_depth_and_opposing_orders_are_not_forced_into_edges():
    packet, arrays = covered_box()
    mask = arrays[packet["relation_graph"]["nodes"][1]["raw_mask_key"]]
    arrays["depth_m"][mask] = .402
    assert extract_relations(packet, arrays)["pairs"][0]["status"] == "no_resolved_depth_step"
    a = np.zeros((80, 80), bool)
    a[10:70, 10:30] = True
    b = np.zeros_like(a)
    b[10:70, 30:60] = True
    packet, arrays = packet_for([a, b], [.4, .45])
    arrays["depth_m"][40:70, 30:60] = .35
    result = extract_relations(packet, arrays)
    assert result["pairs"][0]["status"] == "mixed_local_depth_order"
    assert not result["relation_graph"]["edges"]


def test_overlapping_masks_and_third_instance_are_reported_as_conflicts():
    a = np.zeros((80, 80), bool)
    a[10:70, 10:35] = True
    b = np.zeros_like(a)
    b[10:70, 25:55] = True
    packet, arrays = packet_for([a, b], [.4, .5])
    assert extract_relations(packet, arrays)["pairs"][0]["status"] == "mask_assignment_conflict"
    a[:] = False
    a[:, 10:30] = True
    b[:] = False
    b[:, 32:55] = True
    c = np.zeros_like(a)
    c[:, 30:32] = True
    packet, arrays = packet_for([a, b, c], [.4, .5, .3])
    pair = extract_relations(packet, arrays)["pairs"][0]
    assert pair["status"] == "boundary_blocked_by_third_instance"
    assert not pair["edges"]


def test_convex_completion_is_optional_and_does_not_create_hidden_nodes():
    packet, arrays = covered_box()
    result = extract_relations(packet, arrays, RelationConfig(convex_completion_prior=False))
    assert result["pairs"][0]["status"] == "local_depth_order_only"
    assert not occlusion_edges(result)
    # Completely invisible targets have no mask/node; do not manufacture a relation.
    packet["relation_graph"]["nodes"] = packet["relation_graph"]["nodes"][:1]
    assert not extract_relations(packet, arrays)["pairs"]
    packet["relation_graph"]["nodes"] = []
    assert not extract_relations(packet, arrays)["relation_graph"]["node_ids"]


def test_sparse_depth_is_not_promoted_by_a_few_good_samples():
    packet, arrays = covered_box()
    mask = arrays[packet["relation_graph"]["nodes"][1]["raw_mask_key"]]
    arrays["depth_m"][mask] = np.nan
    arrays["depth_m"][44:47, 50:53] = .45
    result = extract_relations(packet, arrays)
    assert result["pairs"][0]["status"] == "insufficient_local_depth"


def test_local_boundary_order_is_used_even_when_whole_object_medians_disagree():
    packet, arrays = covered_box()
    back = arrays[packet["relation_graph"]["nodes"][1]["raw_mask_key"]]
    arrays["depth_m"][back] = .2  # most of this surface is nearer than object A
    local = np.zeros_like(back)
    local[23:48, 27:53] = True
    arrays["depth_m"][back & local] = .45  # the contacting boundary is farther
    assert np.median(arrays["depth_m"][back]) < .4
    pair = extract_relations(packet, arrays)["pairs"][0]
    assert pair["depth"]["order"] == "a_closer"
    assert pair["depth"]["median_b_minus_a_m"] == pytest.approx(.05)


def test_depth_hole_at_interface_does_not_fall_back_to_valid_object_interior():
    packet, arrays = covered_box()
    back = arrays[packet["relation_graph"]["nodes"][1]["raw_mask_key"]]
    local = np.zeros_like(back)
    local[20:53, 20:60] = True
    arrays["depth_m"][back & local] = np.nan
    assert np.isfinite(arrays["depth_m"][back]).any()
    result = extract_relations(packet, arrays)
    assert result["pairs"][0]["status"] == "insufficient_local_depth"
    assert not result["relation_graph"]["edges"]


def test_saved_packet_export_and_overlay(tmp_path):
    packet, arrays = covered_box()
    source = tmp_path / "input"
    source.mkdir()
    packet["artifacts"] = {"arrays": "observation_arrays.npz"}
    np.savez_compressed(source / "observation_arrays.npz", **arrays)
    path = source / "scene_observation.json"
    path.write_text(json.dumps(packet))
    result = export_relations(path, tmp_path / "out")
    assert (tmp_path / "out" / "relations_overlay.png").is_file()
    assert json.loads((tmp_path / "out" / "relations.json").read_text())["pairs"][0]["status"] == "possible_occlusion"
    assert visualize_relations(packet, arrays, result).shape[1] > arrays["depth_m"].shape[1]
    with pytest.raises(ValueError, match="separate output"):
        export_relations(path, source)


def test_reject_bad_units_shapes_and_configuration():
    packet, arrays = covered_box()
    bad = copy.deepcopy(packet)
    bad["frame"]["depth_units"] = "millimetres"
    with pytest.raises(ValueError, match="metres"):
        extract_relations(bad, arrays)
    with pytest.raises(ValueError, match="mask dimensions"):
        broken = dict(arrays)
        broken[packet["relation_graph"]["nodes"][0]["raw_mask_key"]] = np.zeros((2, 2))
        extract_relations(packet, broken)
    with pytest.raises(ValueError, match="positive"):
        extract_relations(packet, arrays, RelationConfig(min_depth_delta_m=-1))
    with pytest.raises(ValueError, match="integer"):
        extract_relations(packet, arrays, RelationConfig(depth_patch_radius_px=2.0))


def test_simulation_visible_oracle_never_creates_fully_hidden_nodes():
    from robocasa_occlusion.probe_relations import visible_instances

    segmentation = np.zeros((4, 4, 2), int)
    segmentation[:, :, 0] = 5
    segmentation[:, :, 1] = 10
    instances, evaluator_ids = visible_instances(
        segmentation, {"blocker": [10], "invisible_target": [20]})
    assert len(instances) == 1
    assert instances[0].source_id == "visible_000"
    assert instances[0].category is None
    assert instances[0].identity_scores is None
    assert "invisible_target" not in evaluator_ids.values()
