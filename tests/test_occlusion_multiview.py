"""Geometry/contract tests independent of RoboCasa labels and hidden state."""

import copy
import json

import numpy as np
import pytest

from occlusion_reasoning.multiview import MultiViewConfig, extract_multiview
from occlusion_reasoning.observations import InstanceEvidence, build_observation
from occlusion_reasoning.planar import PlanarConfig, extract_planar_hypotheses
from occlusion_reasoning.scene import infer_scene


def make_packet(depth, masks, camera, x=0):
    transform = np.eye(4)
    transform[0, 3] = x
    instances = [InstanceEvidence(f"anonymous_{i}", mask, [0, 0, depth.shape[1], depth.shape[0]], .9)
                 for i, mask in enumerate(masks)]
    packet, arrays = build_observation(
        depth_m=depth, intrinsics=dict(fx=140, fy=140, cx=80, cy=80),
        instances=instances, frame_id=camera, query="not_used",
        alignment_status="recorded_aligned", timestamp_s=1.,
        reference_frame="world", T_reference_from_camera_optical=transform)
    packet["frame"].update(camera_id=camera, capture_group_id="pair_1", timestamp_domain="test")
    return packet, arrays


def ray_box_views(*, merge=False, baseline=-.5, second_x=0, boxes=None, object_names=None):
    # Independent analytic ray/AABB renderer, not the algorithm's reprojection.
    boxes = boxes or [
        (np.array([-.14, -.16, .95]), np.array([.14, .16, 1.05])),
        (np.array([-.1, -.1, 1.2]), np.array([.1, .1, 1.3])),
    ]
    result, names = [], {}
    object_names = object_names or ("front", "back")
    for camera, x in [("left", baseline), ("right", second_x)]:
        yy, xx = np.indices((160, 160))
        rays = np.stack(((xx-80)/140, (yy-80)/140, np.ones_like(xx)), axis=-1)
        origin = np.array([x, 0, 0])
        depths = []
        for low, high in boxes:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (low-origin)/rays, (high-origin)/rays
            entry, leave = np.minimum(t1, t2).max(axis=-1), np.maximum(t1, t2).min(axis=-1)
            depths.append(np.where((leave >= entry) & (entry > 0), entry, np.inf))
        depths = np.asarray(depths)
        owner = depths.argmin(axis=0)
        depth = depths.min(axis=0)
        valid = np.isfinite(depth)
        depth[~valid] = 3.
        mask_names = [(owner == i) & valid for i in range(len(boxes))]
        present = [i for i, mask in enumerate(mask_names) if mask.any()]
        masks = [valid] if merge else [mask_names[i] for i in present]
        packet, arrays = make_packet(depth, masks, camera, x)
        for node, index in zip(packet["relation_graph"]["nodes"], [0] if merge else present):
            names[node["node_id"]] = object_names[index]
        result.append((packet, arrays))
    return result, names


def named_edges(result, names):
    mapping = {e["entity_id"]: {names[node] for node in e["members"]} for e in result["entities"]}
    return {(tuple(sorted(mapping[e["source"]])), tuple(sorted(mapping[e["target"]])), e["view_id"])
            for e in result["relation_graph"]["edges"]}


def test_one_view_fully_hidden_object_is_recovered_without_identity_input():
    observations, names = ray_box_views()
    result = extract_multiview(observations)
    assert len(observations[0][0]["relation_graph"]["nodes"]) == 2
    assert len(observations[1][0]["relation_graph"]["nodes"]) == 1
    assert (("front",), ("back",), "right") in named_edges(result, names)
    assert len(result["entities"]) == 2
    assert all(e["calibrated_probability"] is None for e in result["relation_graph"]["edges"])
    json.dumps(result, allow_nan=False)


def test_one_pixel_visible_rim_survives_without_inventing_interior_depth():
    observations, names = ray_box_views()
    packet, arrays = observations[0]
    node = next(n for n in packet["relation_graph"]["nodes"] if names[n["node_id"]] == "back")
    mask = arrays[node["raw_mask_key"]]
    column = int(mask.sum(axis=0).argmax())
    thin = np.zeros_like(mask)
    thin[:, column] = mask[:, column]
    arrays[node["raw_mask_key"]] = thin
    result = extract_multiview(observations)
    projection = next(p for p in result["projections"] if p["source_node"] == node["node_id"])
    assert projection["association_source_interior_pixels"] == 0
    assert projection["source_valid_surface_pixels"] >= 8
    assert (("front",), ("back",), "right") in named_edges(result, names)


def test_isolated_unreliable_depth_pixels_do_not_make_an_edge():
    observations, names = ray_box_views()
    packet, arrays = observations[0]
    node = next(n for n in packet["relation_graph"]["nodes"] if names[n["node_id"]] == "back")
    mask = arrays[node["raw_mask_key"]]
    yy, xx = np.indices(mask.shape)
    arrays["depth_m"][mask & ((yy % 3 != 0) | (xx % 3 != 0))] = 0
    result = extract_multiview(observations)
    assert not result["relation_graph"]["edges"]


def test_both_hidden_does_not_create_an_object_node():
    observations, _ = ray_box_views(baseline=0)
    result = extract_multiview(observations)
    assert len(result["entities"]) == 1
    assert not result["relation_graph"]["edges"]
    assert result["unobserved_hypothesis"]["probability"] is None


def test_two_distinct_occluders_can_be_reported_for_the_same_object():
    boxes = [
        (np.array([-.14, -.16, .95]), np.array([0, .16, 1.05])),
        (np.array([0, -.16, .95]), np.array([.14, .16, 1.05])),
        (np.array([-.1, -.1, 1.2]), np.array([.1, .1, 1.3])),
    ]
    observations, names = ray_box_views(
        baseline=-.5, second_x=.5, boxes=boxes, object_names=("front_left", "front_right", "back"))
    result = extract_multiview(observations)
    pairs = {(front, back) for front, back, _ in named_edges(result, names)}
    assert (("front_left",), ("back",)) in pairs
    assert (("front_right",), ("back",)) in pairs


def test_same_object_self_occlusion_is_not_an_inter_object_edge():
    observations, _ = ray_box_views(merge=True)
    separate, _ = ray_box_views()
    packet, arrays = separate[0]
    front_mask = arrays[packet["relation_graph"]["nodes"][0]["raw_mask_key"]]
    yy, xx = np.indices(front_mask.shape)
    # Missing front-surface samples allow some back-surface samples to survive
    # the source object's destination Z buffer, while shared surface remains.
    observations[0][1]["depth_m"][front_mask & ((xx + yy) % 3 != 0)] = 0
    result = extract_multiview(observations)
    assert len(result["entities"]) == 1
    assert any(p["behind_measured_surface_pixels"] > 0 for p in result["projections"])
    assert not result["relation_graph"]["edges"]


def test_ambiguous_one_to_two_match_does_not_force_an_identity_merge():
    depth = np.ones((160, 160), np.float32)
    whole = np.zeros_like(depth, bool)
    whole[30:130, 30:130] = True
    first = whole.copy()
    first[:, 80:] = False
    second = whole & ~first
    observations = [make_packet(depth, [whole], "left"),
                    make_packet(depth, [first, second], "right")]
    result = extract_multiview(observations)
    assert len(result["associations"]) == 2
    assert all(a["status"] == "ambiguous" for a in result["associations"])
    assert len(result["entities"]) == 3
    assert not result["relation_graph"]["edges"]


def test_conflicting_masks_are_excluded_from_projection_votes():
    observations, names = ray_box_views()
    packet, arrays = observations[1]
    original = packet["relation_graph"]["nodes"][0]
    duplicate = copy.deepcopy(original)
    duplicate["node_id"] += "_conflict"
    duplicate["raw_mask_key"] = "conflicting_mask"
    arrays["conflicting_mask"] = arrays[original["raw_mask_key"]].copy()
    packet["relation_graph"]["nodes"].append(duplicate)
    result = extract_multiview(observations)
    assert not result["relation_graph"]["edges"]
    assert any(p["mask_conflict_or_boundary_pixels"] > 0 for p in result["projections"])


def test_observation_contract_retains_synchronization_metadata():
    packet, _ = build_observation(
        depth_m=np.ones((2, 2)), intrinsics=dict(fx=10, fy=10, cx=1, cy=1),
        instances=[], frame_id="frame", query="unused", alignment_status="recorded_aligned",
        camera_id="left", capture_group_id="capture_001", timestamp_domain="host_monotonic",
        timestamp_s=1, reference_frame="robot_base", T_reference_from_camera_optical=np.eye(4))
    assert packet["frame"]["camera_id"] == "left"
    assert packet["frame"]["capture_group_id"] == "capture_001"
    assert packet["frame"]["timestamp_domain"] == "host_monotonic"


def test_unknown_identity_without_distinctness_witness_stays_unresolved():
    observations, names = ray_box_views()
    packet, arrays = observations[0]
    packet["relation_graph"]["nodes"] = [
        n for n in packet["relation_graph"]["nodes"] if names[n["node_id"]] == "back"]
    result = extract_multiview(observations)
    assert not result["relation_graph"]["edges"]
    assert result["unresolved_front_surfaces"]


def test_missed_foreground_segmentation_does_not_become_named_occlusion():
    observations, _ = ray_box_views()
    observations[1][0]["relation_graph"]["nodes"] = []
    result = extract_multiview(observations)
    assert not result["relation_graph"]["edges"]
    assert any(p["unassigned_depth_pixels"] > 0 for p in result["projections"])


@pytest.mark.parametrize("depth", [0, np.nan, np.inf])
def test_missing_destination_depth_is_unknown(depth):
    observations, _ = ray_box_views()
    observations[1][1]["depth_m"][:] = depth
    result = extract_multiview(observations)
    assert not result["relation_graph"]["edges"]
    assert any(p["missing_destination_depth_pixels"] > 0 for p in result["projections"])


def test_reprojection_in_front_is_inconsistency_not_reverse_occlusion():
    observations, _ = ray_box_views()
    observations[1][1]["depth_m"][:] = 10
    result = extract_multiview(observations)
    assert any(p["in_front_inconsistent_pixels"] > 0 for p in result["projections"])
    assert not result["relation_graph"]["edges"]


def test_out_of_frame_is_unknown():
    observations, _ = ray_box_views()
    observations[1][0]["frame"]["T_reference_from_camera_optical"][0][3] = 20
    result = extract_multiview(observations)
    assert all(p["unique_projected_pixels"] == 0 for p in result["projections"])
    assert not result["relation_graph"]["edges"]


def test_zbuffer_does_not_count_multiple_source_pixels_as_independent_votes():
    observations, _ = ray_box_views()
    result = extract_multiview(observations)
    assert any(p["duplicate_destination_pixels_removed"] > 0 for p in result["projections"])
    for p in result["projections"]:
        assert p["unique_projected_pixels"] <= p["source_valid_surface_pixels"]
    for edge in result["relation_graph"]["edges"]:
        projection = next(p for p in result["projections"] if p["source_node"] == edge["reprojected_node"])
        assert edge["unique_support_pixels"] <= projection["unique_projected_pixels"]


def test_semantics_cached_clouds_view_and_node_order_do_not_drive_inference():
    observations, names = ray_box_views()
    original = named_edges(extract_multiview(observations), names)
    changed = copy.deepcopy(observations)
    for packet, arrays in changed:
        packet["task"] = {"query": "fake ground truth"}
        packet["target_candidates"] = {"secret": "front"}
        packet["provenance"] = {"truth": "back"}
        packet["relation_graph"]["nodes"].reverse()
        for node in packet["relation_graph"]["nodes"]:
            node["category"] = "wrong_name"
            node["observed_surface_camera"] = {"centroid_m": [99, 99, 99]}
            arrays[node["observed_points_camera_key"]][:] = 999
    assert named_edges(extract_multiview(changed[::-1]), names) == original
    assert observations[0][0]["relation_graph"]["edges"] == []


@pytest.mark.parametrize("field,value", [
    ("capture_group_id", "another_state"), ("timestamp_domain", "another_clock"),
    ("reference_frame", "another_world"), ("timestamp_s", 10),
    ("T_reference_from_camera_optical", None), ("rgb_depth_alignment", "assumed_aligned"),
    ("camera_id", "left"), ("depth_definition", "range"),
])
def test_invalid_acquisition_contract_is_rejected(field, value):
    observations, _ = ray_box_views()
    observations[1][0]["frame"][field] = value
    with pytest.raises(ValueError):
        extract_multiview(observations)


def planar_scene(adjacent=False):
    front, back = np.zeros((160, 160), bool), np.zeros((160, 160), bool)
    back[30:130, 30:130] = True
    if adjacent:
        front[30:130, 130:150] = True
    else:
        front[50:110, 50:110] = True
        back &= ~front
    depth = np.full((160, 160), 3., np.float32)
    depth[back], depth[front] = 1., .8
    return make_packet(depth, [front, back], "single")


def test_planar_completion_handles_contact_without_a_dominant_boundary_step():
    packet, arrays = planar_scene()
    result = extract_planar_hypotheses(packet, arrays)
    assert len(result["edges"]) == 1
    edge = result["edges"][0]
    assert edge["source"] == packet["relation_graph"]["nodes"][0]["node_id"]
    assert edge["target"] == packet["relation_graph"]["nodes"][1]["node_id"]
    assert edge["depth_gap_median_m"] == pytest.approx(.2)
    assert edge["confirmed"] is False
    assert edge["relation"] == "possible_occludes"


def test_side_by_side_planar_surfaces_do_not_imply_occlusion():
    assert not extract_planar_hypotheses(*planar_scene(adjacent=True))["edges"]


def test_completion_cannot_distinguish_a_real_hole_and_must_remain_a_prior():
    # Same RGB-D can describe an actual ring-shaped object, with an independent
    # object inside its hole. The proposed hidden plane is NOT an observation.
    edge = extract_planar_hypotheses(*planar_scene())["edges"][0]
    assert "prior" in edge["evidence"] and not edge["confirmed"]


def test_planar_prior_can_be_disabled_for_nonplanar_objects():
    packet, arrays = planar_scene()
    second = copy.deepcopy(packet)
    second["frame"].update(camera_id="second", frame_id="second")
    for node in second["relation_graph"]["nodes"]:
        node["node_id"] = node["node_id"].replace("single", "second")
    observations = [(packet, arrays), (second, copy.deepcopy(arrays))]
    assert infer_scene(observations)["relation_graph"]["edges"]
    assert not infer_scene(observations, planar_prior=False)["relation_graph"]["edges"]


def test_inconsistent_geometry_disables_fusion_but_retains_local_hypotheses():
    packet, arrays = planar_scene()
    second = copy.deepcopy(packet)
    second["frame"].update(camera_id="second", frame_id="second")
    second["frame"]["T_reference_from_camera_optical"][2][3] += .03
    for node in second["relation_graph"]["nodes"]:
        node["node_id"] = node["node_id"].replace("single", "second")
    observations = [(packet, arrays), (second, copy.deepcopy(arrays))]
    direct = extract_multiview(observations)
    assert direct["geometry_health"]["status"] == "rejected_inconsistent_depth"
    assert not direct["relation_graph"]["edges"]
    assert not any(a["status"] == "accepted" for a in direct["associations"])
    combined = infer_scene(observations)
    assert combined["relation_graph"]["edges"]
    assert all(e["relation"] == "possible_occludes" for e in combined["relation_graph"]["edges"])


@pytest.mark.parametrize("kwargs", [
    {"occlusion_depth_margin_m": .001}, {"association_best_ratio": 1},
    {"min_occlusion_pixels": 1.2}, {"boundary_margin_px": -1},
    {"min_association_fraction": 2}, {"association_depth_tolerance_m": np.nan},
])
def test_invalid_multiview_configuration(kwargs):
    with pytest.raises(ValueError):
        MultiViewConfig(**kwargs)


def test_invalid_planar_configuration():
    with pytest.raises(ValueError):
        PlanarConfig(plane_tolerance_m=.01)
    with pytest.raises(ValueError):
        PlanarConfig(ransac_trials=1.5)
