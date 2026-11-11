from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import json

import numpy as np
import pytest

from perception_v2.fragment_association import (
    FragmentAssociationMemory, associate_fragments, validate_directions)
from perception_v2.sam3_fragment_validator import SAM3FragmentValidator


def fixture_scene():
    shape=(100,140)
    def mask(x0,y0,x1,y1):
        m=np.zeros(shape,bool);m[y0:y1,x0:x1]=True;return m
    left=mask(20,20,55,80);right=mask(85,20,120,80);cover=mask(55,10,85,90)
    depth=np.full(shape,.8);depth[left|right]=.6;depth[cover]=.5
    image=np.full((*shape,3),120,np.uint8)
    rows=[]
    for name,m in [('left',left),('right',right),('cover',cover)]:
        y,x=np.nonzero(m)
        rows.append(dict(source_id=name,mask=m,confidence=.9,prompt='box',
                         bbox_xyxy=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]))
    return rows,depth,image,dict(fx=200.,fy=200.,cx=70.,cy=50.)


def context(t=100.,epoch=1,scope='camera-A',pose=None):
    return dict(timestamp=t,scene_epoch=epoch,scope=scope,pose=pose or [0.]*6)


def agreeing_validator(image,masks,negatives):
    return dict(masks=[masks[0]|masks[1],masks[0]|masks[1]],
                scores=[.9,.9],image_encodings=1,decoder_calls=2)


def run(rows,depth,image,k,validator,**kwargs):
    return associate_fragments(rows,depth,k,image,validator,**kwargs)


def test_confirmed_group_preserves_both_fragments_and_never_fills_occluder():
    rows,d,img,k=fixture_scene();original=[r['mask'].copy() for r in rows]
    selected,audit=run(rows,d,img,k,agreeing_validator)
    group=next(r for r in selected if r.get('object_id'))
    np.testing.assert_array_equal(group['mask'],original[0]|original[1])
    assert not np.any(group['mask']&original[2])
    assert len(group['visible_fragments'])==2
    assert {f['object_id'] for f in group['visible_fragments']}=={group['object_id']}
    assert audit['model_pair_calls']==1 and audit['groups']==1
    for r,m in zip(rows,original):np.testing.assert_array_equal(r['mask'],m)


@pytest.mark.parametrize('case',['one_direction','blocker','extra_pixels','different_masks'])
def test_association_requires_two_consistent_clean_predictions(case):
    rows,d,img,k=fixture_scene();a,b,cover=[r['mask'] for r in rows];whole=a|b
    preds=[whole.copy(),whole.copy()]
    if case=='one_direction':preds[1]=b
    elif case=='blocker':preds=[whole|cover,whole|cover]
    elif case=='extra_pixels':preds=[np.ones(a.shape,bool)]*2
    else:preds=[a,b]
    assert not validate_directions([a,b],cover,dict(masks=preds))['accepted']


def test_failed_validation_does_not_poison_memory():
    rows,d,img,k=fixture_scene();memory=FragmentAssociationMemory()
    v=lambda image,masks,neg:dict(masks=masks)
    selected,audit=run(rows,d,img,k,v,memory=memory,context=context())
    assert audit['groups']==0 and not memory.entries
    assert len(selected)==3


def test_memory_reuses_id_with_new_source_ids_and_without_new_model_call():
    rows,d,img,k=fixture_scene();memory=FragmentAssociationMemory();v=Mock(side_effect=agreeing_validator)
    first,_=run(rows,d,img,k,v,memory=memory,context=context())
    for i,r in enumerate(rows):r['source_id']='new_'+str(i)
    second,audit=run(rows,d,img,k,v,memory=memory,context=context(101.))
    assert v.call_count==1 and audit['model_pair_calls']==0
    assert next(r['object_id'] for r in first if r.get('object_id'))==next(r['object_id'] for r in second if r.get('object_id'))
    assert audit['checks'][0]['source']=='short_term_memory'


@pytest.mark.parametrize('changed',['scene','camera','translation','rotation','expiry','backwards_time','no_epoch','no_pose','depth','appearance'])
def test_memory_invalidates_on_changed_evidence(changed):
    rows,d,img,k=fixture_scene();memory=FragmentAssociationMemory();v=Mock(side_effect=agreeing_validator)
    run(rows,d,img,k,v,memory=memory,context=context())
    ctx=context(101.)
    if changed=='scene':ctx['scene_epoch']=2
    elif changed=='camera':ctx['scope']='camera-B'
    elif changed=='translation':ctx['pose'][0]=.02
    elif changed=='rotation':ctx['pose'][5]=.1
    elif changed=='expiry':ctx['timestamp']=131.
    elif changed=='backwards_time':ctx['timestamp']=99.
    elif changed=='no_epoch':ctx['scene_epoch']=None
    elif changed=='no_pose':ctx['pose']=None
    elif changed=='depth':d[rows[0]['mask']|rows[1]['mask']]=.62
    elif changed=='appearance':img[rows[0]['mask']|rows[1]['mask']]=220
    _,audit=run(rows,d,img,k,v,memory=memory,context=ctx)
    assert audit['model_pair_calls']==1 and v.call_count==2


def test_memory_reuse_does_not_extend_confirmation_expiry():
    rows,d,img,k=fixture_scene();memory=FragmentAssociationMemory();v=Mock(side_effect=agreeing_validator)
    for t in (100.,120.,131.):run(rows,d,img,k,v,memory=memory,context=context(t))
    assert v.call_count==2


def test_depth_contradiction_drops_old_geometry_association():
    rows,d,img,k=fixture_scene();memory=FragmentAssociationMemory()
    run(rows,d,img,k,agreeing_validator,memory=memory,context=context())
    d[rows[0]['mask']]=.68
    v=Mock(side_effect=AssertionError('Incompatible surfaces should not be queried'))
    _,audit=run(rows,d,img,k,v,memory=memory,context=context(101.))
    assert not memory.entries and audit['groups']==0


def test_model_budget_does_not_forge_success():
    rows,d,img,k=fixture_scene();v=Mock(side_effect=AssertionError('Budget exhausted'))
    selected,audit=run(rows,d,img,k,v,max_pairs=0)
    assert len(selected)==3 and audit['model_pair_calls']==0 and audit['groups']==0


def test_one_encoding_two_independent_positive_prompts():
    rows,d,img,k=fixture_scene();masks=[r['mask'] for r in rows[:2]]
    calls=[];states=[]
    def encode(image):
        states.append(object());return states[-1]
    def predict(state,**kwargs):
        assert state is states[0]
        calls.append(kwargs)
        return np.zeros((3,100,140),bool),np.array([.3,.8,.5]),None
    backend=SimpleNamespace(device='cpu',torch=SimpleNamespace(inference_mode=nullcontext),
        processor=SimpleNamespace(set_image=encode),model=SimpleNamespace(predict_inst=predict))
    prediction=SAM3FragmentValidator(backend)(img,masks,rows[2]['mask'])
    assert len(states)==1 and len(calls)==2
    for i,call in enumerate(calls):
        coords,labels=call['point_coords'],call['point_labels']
        assert list(labels).count(1)==1
        px,py=coords[0].astype(int)
        assert masks[i][py,px] and not masks[1-i][py,px]
    assert prediction['image_encodings']==1 and prediction['decoder_calls']==2


def test_group_evidence_survives_saved_observation(tmp_path):
    from occlusion_reasoning.real_single_view import save_view
    import cv2
    rows,d,img,k=fixture_scene();selected,audit=run(rows,d,img,k,agreeing_validator)
    snap=tmp_path/'capture';snap.mkdir()
    cv2.imwrite(str(snap/'color.png'),img);np.save(snap/'depth_m.npy',d)
    (snap/'grasp_input.json').write_text(json.dumps(dict(depth_aligned_to_color=True,
        camera_id='wrist',capture_group_id='test',capture_timestamp_s=100.,timestamp_domain='host',
        depth_intrinsics=dict(fx=k['fx'],fy=k['fy'],ppx=k['cx'],ppy=k['cy']))))
    out=tmp_path/'analysis';save_view(snap,selected,dict(visible_hypothesis_selection={}),out)
    packet=json.loads((out/'observation/scene_observation.json').read_text())
    node=next(n for n in packet['relation_graph']['nodes'] if n.get('object_id'))
    assert node['track_id']==node['object_id']
    with np.load(out/'observation/observation_arrays.npz') as arrays:
        fragments=[arrays[key] for key in node['visible_fragment_mask_keys']]
        np.testing.assert_array_equal(fragments[0]|fragments[1],arrays[node['raw_mask_key']])
    metadata=json.loads((out/'segmentation_report.json').read_text())
    assert len(next(p for p in metadata['proposals'] if p.get('object_id'))['visible_fragments'])==2
