"""Counterexamples for learned relation supplementation and official pair ordering."""
from copy import deepcopy
import numpy as np
import pytest
from occlusion_reasoning.pretrained_order import (
    InstaOrderPredictor, prepare_pair, supplement_relations,
)


def case():
    pair = dict(pair_id='p', a='a', b='b', status='no_resolved_depth_step',
                mask_overlap={'fraction_of_smaller_mask': 0}, boundary={'near_pair_count': 20, 'contact_fraction': .2},
                depth={'order': 'unknown'}, edges=[])
    geometry = dict(config={'max_mask_overlap_fraction': .05, 'contact_distance_px': 2.}, pairs=[pair],
                    relation_graph={'nodes': [], 'edges': []})
    score = dict(pair_id='p', a='a', b='b', a_over_b=.8, b_over_a=.1)
    return geometry, score


def test_missing_depth_does_not_discard_visual_occlusion_and_inputs_unchanged():
    geometry, score = case()
    before = deepcopy(geometry)
    result = supplement_relations(geometry, [], [score])
    edge, = result['relation_graph']['edges']
    assert (edge['source'], edge['target']) == ('a', 'b')
    assert edge['confirmed'] is False and edge['calibrated_probability'] is None
    assert geometry == before


@pytest.mark.parametrize('failure', ['distant', 'third_object', 'overlap', 'opposing_depth', 'bidirectional'])
def test_confident_model_is_not_enough_when_evidence_conflicts(failure):
    geometry, score = case()
    pair = geometry['pairs'][0]
    if failure in ('distant', 'third_object'):
        pair['boundary']['near_pair_count'] = 0
        pair['status'] = 'no_near_boundary_evidence' if failure=='distant' else 'boundary_blocked_by_third_instance'
    elif failure == 'overlap':
        pair['mask_overlap']['fraction_of_smaller_mask'] = .2
    elif failure == 'opposing_depth':
        pair['depth']['order'] = 'b_closer'
    else:
        score['b_over_a'] = .9
    result = supplement_relations(geometry, [], [score])
    assert result['relation_graph']['edges'] == []
    assert all(not d['accepted'] for d in result['model_decisions'])


def test_preserves_baseline_without_duplicates_or_silent_direction_reversal():
    geometry, score = case()
    edge = dict(source='a', target='b', relation='possible_occludes', evidence_source='geometry')
    assert supplement_relations(geometry, [edge], [score])['relation_graph']['edges'] == [edge]
    reverse = dict(score, a_over_b=.1, b_over_a=.9)
    result = supplement_relations(geometry, [edge], [reverse])
    assert result['relation_graph']['edges'] == [edge]
    assert result['model_decisions'][0]['rejection_reason'] == 'opposing_baseline_edge'


def test_swapping_pair_swaps_only_mask_channels_without_mutating_masks():
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    a = np.zeros((90, 120), bool); a[:30, :20] = True
    b = np.zeros_like(a); b[30:70, 20:40] = True
    original = a.copy()
    first = prepare_pair(rgb, a, b, [0, 0, 20, 30], [20, 30, 40, 70])
    second = prepare_pair(rgb, b, a, [20, 30, 40, 70], [0, 0, 20, 30])
    np.testing.assert_array_equal(first, second[[1, 0, 2, 3, 4]])
    assert first.shape == (5, 256, 256) and first.dtype == np.float32
    assert set(np.unique(first[:2])) <= {0., 1.}
    np.testing.assert_array_equal(a, original)


def test_official_output_index_and_swap_averaging():
    torch = pytest.importorskip('torch')
    predictor = InstaOrderPredictor.__new__(InstaOrderPredictor)
    predictor.torch, predictor.device, predictor.batch_size = torch, 'cpu', 8
    class Model:
        calls = 0
        def __call__(self, batch):
            self.calls += 1
            probability = [.1, .8] if self.calls % 2 else [.6, .2]
            return torch.logit(torch.tensor([probability]*len(batch)))
    predictor.model = Model()
    nodes = [dict(node_id=k, bbox_xyxy=[1, 1, 9, 9]) for k in ('a', 'b')]
    masks = {k: np.ones((10, 10), bool) for k in ('a', 'b')}
    result, = predictor.predict(np.zeros((10, 10, 3), np.uint8), nodes, masks,
                                [dict(pair_id='p', a='a', b='b')])
    assert result['a_over_b'] == pytest.approx(.7)
    assert result['b_over_a'] == pytest.approx(.15)
    assert predictor.predict(np.zeros((10, 10, 3), np.uint8), nodes, masks, []) == []


@pytest.mark.parametrize('gap, expected_edges', [(0, 1), (3, 0)])
def test_nearby_masks_with_background_gap_defer_even_a_confident_model(gap, expected_edges):
    from occlusion_reasoning.observations import InstanceEvidence, build_observation
    from occlusion_reasoning.relations import extract_relations
    left = np.zeros((60, 60), bool); left[10:40, 10:25] = True
    right = np.zeros_like(left); right[10:40, 25+gap:40+gap] = True
    depth = np.full(left.shape, .6, np.float32)
    packet, arrays = build_observation(
        depth_m=depth, intrinsics={'fx': 100, 'fy': 100, 'cx': 30, 'cy': 30},
        instances=[InstanceEvidence('left', left, [10, 10, 25, 40], .9),
                   InstanceEvidence('right', right, [25+gap, 10, 40+gap, 40], .9)],
        frame_id='test', query='irrelevant', alignment_status='recorded_aligned')
    boundary = extract_relations(packet, arrays)
    pair, = boundary['pairs']
    assert pair['boundary']['near_pair_count'] > 0
    score = dict(pair_id=pair['pair_id'], a=pair['a'], b=pair['b'], a_over_b=.999, b_over_a=.01)
    result = supplement_relations(boundary, [], [score])
    assert len(result['relation_graph']['edges']) == expected_edges
    if gap:
        deferred, = result['deferred_candidates']
        assert deferred['model_score'] == .999
        assert deferred['status'] == 'needs_boundary_verification'
        assert deferred['interpretation'] == 'insufficient_boundary_support_not_proven_absence'
    else:
        assert result['deferred_candidates'] == []


def test_contact_gate_does_not_erase_independent_geometry_edges():
    geometry, score = case()
    geometry['pairs'][0]['boundary']['contact_fraction'] = 0
    edge = dict(source='a', target='b', relation='possible_occludes', evidence_source='planar')
    result = supplement_relations(geometry, [edge], [score])
    assert result['relation_graph']['edges'] == [edge]
    assert result['deferred_candidates'][0]['rejection_reason'] == 'no_visible_boundary_contact'


def test_small_contact_arc_is_not_required_to_cover_whole_boundary():
    geometry, score = case()
    geometry['pairs'][0]['boundary']['contact_fraction'] = .16
    assert len(supplement_relations(geometry, [], [score])['relation_graph']['edges']) == 1
