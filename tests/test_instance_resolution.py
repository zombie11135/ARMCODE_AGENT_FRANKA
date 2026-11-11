import numpy as np
import pytest

from perception_v2.proposal_refinement import select_visible_hypotheses


def scene():
    shape=(100,140)
    def mask(x0,y0,x1,y1):
        m=np.zeros(shape,bool);m[y0:y1,x0:x1]=True;return m
    left=mask(20,20,55,80);right=mask(85,20,120,80)
    cover=mask(55,10,85,90);small=mask(60,30,80,50)
    depth=np.full(shape,.8);depth[left|right]=.6;depth[cover]=.5;depth[small]=.4
    def row(key,m,score):
        y,x=np.nonzero(m)
        return dict(source_id=key,mask=m,confidence=score,prompt='box',
                    bbox_xyxy=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])
    rows=[row('left',left,.9),row('right',right,.88),row('cover',cover,.95),
          row('small',small,.7),row('whole',left|right,.65)]
    k=dict(fx=200.,fy=200.,cx=70.,cy=50.)
    return rows,depth,k


def test_small_foreground_survives_large_mask_and_fragments_use_joint_model_mask():
    rows,depth,k=scene();original={p['source_id']:p['mask'].copy() for p in rows}
    selected,audit=select_visible_hypotheses(rows,depth_m=depth,intrinsics=k,enable_joint_reconciliation=True)
    byid={p['source_id']:p for p in selected}
    assert set(byid)=={'cover','small','whole'}
    assert not np.any(byid['cover']['mask'] & byid['small']['mask'])
    np.testing.assert_array_equal(byid['whole']['mask'],original['whole'])
    assert byid['whole']['confidence']==.65
    assert audit['fragment_reconciliations'][0]['fragment_source_ids']==['left','right']
    assert not audit['hidden_pixels_filled']
    for p in rows:np.testing.assert_array_equal(p['mask'],original[p['source_id']])


def test_same_depth_nested_hypothesis_is_not_invented_as_a_new_block():
    rows,depth,k=scene();depth[rows[3]['mask']]=.5
    selected,audit=select_visible_hypotheses(rows,depth_m=depth,intrinsics=k,enable_joint_reconciliation=True)
    assert 'small' not in {p['source_id'] for p in selected}
    assert audit['nested_foreground_recoveries']==[]


@pytest.mark.parametrize('missing', ['joint_prediction','closer_gap','depth','same_surface'])
def test_disjoint_regions_are_not_merged_without_joint_geometry_evidence(missing):
    rows,depth,k=scene()
    if missing=='joint_prediction':rows=[p for p in rows if p['source_id']!='whole']
    elif missing=='closer_gap':depth[rows[2]['mask']]=.7
    elif missing=='depth':depth[:]=0
    else:depth[rows[1]['mask']]=.66
    _,audit=select_visible_hypotheses(rows,depth_m=depth,intrinsics=k,enable_joint_reconciliation=True)
    assert audit['fragment_reconciliations']==[]


def test_reconciled_object_keeps_multilayer_occlusion_dependencies():
    from occlusion_reasoning.observations import InstanceEvidence,build_observation
    from occlusion_reasoning.planar import extract_planar_hypotheses
    rows,depth,k=scene();selected,_=select_visible_hypotheses(rows,depth_m=depth,intrinsics=k,enable_joint_reconciliation=True)
    instances=[InstanceEvidence(p['source_id'],p['mask'],p['bbox_xyxy'],p['confidence']) for p in selected]
    packet,arrays=build_observation(depth_m=depth,intrinsics=k,instances=instances,
        frame_id='nested',query='unknown',alignment_status='recorded_aligned',
        reference_frame='camera_local',T_reference_from_camera_optical=np.eye(4))
    nodes=packet['relation_graph']['nodes'];mapping={s:n['node_id'] for n in nodes for s in n['source_ids']}
    edges=extract_planar_hypotheses(packet,arrays)['edges'];pairs={(e['source'],e['target']) for e in edges}
    assert (mapping['small'],mapping['cover']) in pairs
    assert (mapping['cover'],mapping['whole']) in pairs


def test_default_keeps_fragments_separate_and_recovers_small_foreground():
    rows,depth,k=scene()
    selected,audit=select_visible_hypotheses(rows,depth_m=depth,intrinsics=k)
    byid={p['source_id']:p for p in selected}
    assert set(byid)=={'left','right','cover','small'}
    assert not audit['joint_reconciliation_enabled']
    assert not audit['fragment_reconciliations'] and 'fragment_associations' not in audit
    assert audit['nested_foreground_recoveries']
    assert not np.any(byid['cover']['mask'] & byid['small']['mask'])
    from occlusion_reasoning.observations import InstanceEvidence,build_observation
    from occlusion_reasoning.planar import extract_planar_hypotheses
    instances=[InstanceEvidence(p['source_id'],p['mask'],p['bbox_xyxy'],p['confidence']) for p in selected]
    packet,arrays=build_observation(depth_m=depth,intrinsics=k,instances=instances,
        frame_id='separate',query='unknown',alignment_status='recorded_aligned',
        reference_frame='camera_local',T_reference_from_camera_optical=np.eye(4))
    mapping={s:n['node_id'] for n in packet['relation_graph']['nodes'] for s in n['source_ids']}
    edges=extract_planar_hypotheses(packet,arrays)['edges']
    pairs={(e['source'],e['target']) for e in edges}
    assert (mapping['small'],mapping['cover']) in pairs
    # Rectangular side fragments alone do not establish hidden continuation.
    assert (mapping['cover'],mapping['left']) not in pairs
    assert (mapping['cover'],mapping['right']) not in pairs


def test_agent_defaults_skip_fragment_models_and_keep_geometry_enabled():
    from runtime.config import SimpleAgentSystemConfig
    from runtime.services import build_occlusion_service
    from services.wrist_occlusion_service import WristOcclusionService
    from services.runtime_service_cache import clear_service_cache
    cfg=SimpleAgentSystemConfig(goal='test',enable_geometry_occlusion=True,
        geometry_occlusion_checkpoint='not-loaded.pt',planner_llm_base_url='http://127.0.0.1:8010/v1',
        planner_llm_model='test')
    assert not cfg.enable_fragment_association and not cfg.enable_vlm_fragment_assistance
    assert not WristOcclusionService('not-loaded.pt').enable_fragment_association
    clear_service_cache()
    try:
        service=build_occlusion_service(cfg)
        assert not service.enable_fragment_association and service.fragment_advisor is None
    finally:
        clear_service_cache()


def test_separate_regions_with_visible_recesses_retain_multilayer_chain():
    from occlusion_reasoning.observations import InstanceEvidence,build_observation
    from occlusion_reasoning.planar import extract_planar_hypotheses
    rows,depth,k=scene()
    # Each lower region has its own visible recess around the foreground, so
    # occlusion is supported without assuming the two regions are one object.
    rows[0]['mask'][80:,20:55]=True
    rows[0]['mask'][90:,20:70]=True
    rows[1]['mask'][80:,85:120]=True
    rows[1]['mask'][90:,75:120]=True
    for row in rows[:2]:
        depth[row['mask']]=.6
        y,x=np.nonzero(row['mask'])
        row['bbox_xyxy']=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    rows=rows[:4]
    selected,audit=select_visible_hypotheses(rows,depth_m=depth,intrinsics=k)
    instances=[InstanceEvidence(p['source_id'],p['mask'],p['bbox_xyxy'],p['confidence']) for p in selected]
    packet,arrays=build_observation(depth_m=depth,intrinsics=k,instances=instances,
        frame_id='recesses',query='unknown',alignment_status='recorded_aligned',
        reference_frame='camera_local',T_reference_from_camera_optical=np.eye(4))
    mapping={s:n['node_id'] for n in packet['relation_graph']['nodes'] for s in n['source_ids']}
    pairs={(e['source'],e['target']) for e in extract_planar_hypotheses(packet,arrays)['edges']}
    assert (mapping['small'],mapping['cover']) in pairs
    assert (mapping['cover'],mapping['left']) in pairs
    assert (mapping['cover'],mapping['right']) in pairs
    assert not audit['fragment_reconciliations']
