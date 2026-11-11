import numpy as np
import pytest

from occlusion_reasoning.observations import InstanceEvidence, build_observation
from occlusion_reasoning.relations import extract_relations
from occlusion_reasoning.structure import summarize_structure, root_priority, vertical_relations


def frame(masks, depths):
    depth = np.full(masks[0].shape, .9, np.float32)
    for mask, value in zip(masks, depths):
        depth[mask] = value
    return build_observation(depth_m=depth, intrinsics=dict(fx=100, fy=100, cx=40, cy=40),
        instances=[InstanceEvidence(str(i), m, [0, 0, 80, 80], .9) for i, m in enumerate(masks)],
        frame_id='complex', query='target', alignment_status='recorded_aligned',
        reference_frame='test_camera', T_reference_from_camera_optical=np.eye(4))


def test_small_occluders_collectively_cover_one_missing_silhouette():
    back = np.zeros((80, 80), bool);back[5:75, 5:75] = True
    back[20:60, 20:60] = False
    fronts = []
    for start in (21, 31, 41):
        mask = np.zeros_like(back);mask[20:27, start:start+8] = True;fronts.append(mask)
    packet, arrays = frame(fronts+[back], [.4]*3+[.5])
    result = extract_relations(packet, arrays)
    groups = result['collective_occlusions']
    assert len(groups) == 1 and len(groups[0]['sources']) == 3
    edges = [e for e in result['relation_graph']['edges'] if e['relation'] == 'possible_occludes']
    assert len(edges) == 3
    assert all(e['evidence'] == 'collective_convex_completion' for e in edges)
    for pair in result['pairs']:
        if pair.get('collective_group_id'):
            assert pair['completion']['front_overlap_fraction'] < .1


def test_collective_evidence_does_not_fill_missing_depth():
    back = np.zeros((80, 80), bool);back[5:75, 5:75] = True;back[20:60, 20:60] = False
    a = np.zeros_like(back);a[20:27, 21:29] = True
    b = np.zeros_like(back);b[20:27, 41:49] = True
    packet, arrays = frame([a, b, back], [0, .4, .5])
    assert not extract_relations(packet, arrays)['collective_occlusions']


def test_mixed_depth_keeps_two_local_orders_without_inventing_global_direction():
    a = np.zeros((80, 80), bool);a[10:70, 10:30] = True
    b = np.zeros_like(a);b[10:70, 30:60] = True
    packet, arrays = frame([a, b], [.4, .45])
    arrays['depth_m'][40:70, 30:60] = .35
    pair = extract_relations(packet, arrays)['pairs'][0]
    assert pair['status'] == 'mixed_local_depth_order'
    assert {r['order'] for r in pair['local_regions']} == {'a_closer', 'b_closer'}
    assert pair['edges'] == []
    assert all(r['scope'] == 'local_interface_only' for r in pair['local_regions'])


def test_diamond_dependencies_have_layers_but_no_fabricated_direct_edge():
    pairs = [dict(source=a, target=b) for a,b in [('a','b'),('a','c'),('b','t'),('c','t'),('d','t')]]
    structure = summarize_structure(['a','b','c','d','t'], pairs)['targets']['t']
    assert structure['clearing_layers'] == [['a','d'], ['b','c']]
    assert structure['direct_blockers'] == ['b','c','d']
    assert structure['multi_layer'] and structure['multiple_direct_blockers']
    assert root_priority('a', 't', pairs)[0] == 2
    assert root_priority('d', 't', pairs)[0] == 1
    assert len(pairs) == 5


def test_cycle_conflict_does_not_hide_behind_an_exposed_root():
    pairs = [dict(source=a, target=b) for a,b in [('r','a'),('a','b'),('b','a'),('b','t')]]
    result = summarize_structure(['r','a','b','t'], pairs)['targets']['t']
    assert result['has_dependency_conflict']
    assert result['unresolved_dependency_ids'] == ['a','b']


def test_stack_uses_base_gravity_instead_of_camera_depth():
    x, y = np.meshgrid(np.linspace(-.04,.04,12), np.linspace(-.04,.04,12))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size,.4)])
    arrays = {'top':points.copy(), 'bottom':points.copy()}
    arrays['bottom'][:,2] = .45
    nodes = [dict(node_id=n, observed_points_camera_key=n) for n in arrays]
    transform = np.diag([1.,-1.,-1.,1.]);transform[2,3] = .8
    edges = vertical_relations(nodes, arrays, transform)
    assert len(edges) == 1 and edges[0]['source'] == 'top'
    assert edges[0]['target'] == 'bottom'
    assert edges[0]['observed_surface_gap_m'] == pytest.approx(.05)
    assert edges[0]['confirmed'] is False
    arrays['bottom'][:,0] += .2
    assert vertical_relations(nodes, arrays, transform) == []


def test_planar_completion_can_accumulate_several_small_foregrounds():
    from occlusion_reasoning.planar import extract_planar_hypotheses
    back=np.zeros((80,80),bool);back[5:75,5:75]=True
    fronts=[]
    for y,x in [(20,20),(20,45),(45,20),(45,45)]:
        mask=np.zeros_like(back);mask[y:y+10,x:x+10]=True;fronts.append(mask);back &= ~mask
    packet,arrays=frame(fronts+[back],[.4]*4+[.5])
    result=extract_planar_hypotheses(packet,arrays)
    assert len(result['collective_occlusions'])==1
    assert len(result['edges'])==4
    assert all(e['fraction_of_completed_patch'] < .05 and e['collectively_supported'] for e in result['edges'])


def graph_fixture():
    nodes=[dict(node_id=n,detection_score=.5,quality=dict(raw_mask_pixels=100,depth_supported_pixels=100),
                observed_surface_camera=dict(point_count=100)) for n in ['a','b','c','d','t']]
    pairs=[dict(source=a,target=b) for a,b in [('a','b'),('a','c'),('b','t'),('c','t'),('d','t')]]
    return dict(primary=dict(relation_graph=dict(nodes=nodes),possible_pairs=pairs,frame=dict(width=100,height=100)),
                target_association=dict(status='associated',node_id='t'))


def test_blocker_selection_prefers_shared_prerequisite_over_detection_score():
    from services.wrist_occlusion_service import choose_blocker
    graph=graph_fixture()
    graph['primary']['relation_graph']['nodes'][3]['detection_score']=.99
    result=choose_blocker(graph)
    assert result['blocker']['node_id']=='a'
    assert result['dependency']['clearing_layers']==[['a','d'],['b','c']]


def test_local_crossing_conflict_requests_another_view_instead_of_clearing():
    from services.wrist_occlusion_service import choose_blocker
    graph=graph_fixture()
    graph['primary']['local_depth_regions']=[dict(source='a',target='t',possible_local_occlusion=True)]
    assert choose_blocker(graph)['status']=='local_depth_order_conflict'
    assert choose_blocker(graph)['blocker'] is None
