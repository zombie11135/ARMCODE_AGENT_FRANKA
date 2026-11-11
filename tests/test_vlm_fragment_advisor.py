import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from services.vlm_fragment_advisor import VLMFragmentAdvisor
from perception_v2.fragment_association import associate_fragments
from test_fragment_association import fixture_scene, context


def test_vlm_same_object_is_advice_and_never_overrides_failed_geometry_validation(tmp_path):
    rows, depth, image, k = fixture_scene()
    advisor=Mock(return_value=dict(relation='same_object',advisory_only=True))
    validator=lambda image,masks,neg:dict(masks=masks)
    selected,audit=associate_fragments(rows,depth,k,image,validator,advisor=advisor)
    assert advisor.call_count==1 and audit['groups']==0
    assert len(selected)==len(rows)
    assert audit['checks'][0]['vlm_advice']['relation']=='same_object'
    for a,b in zip(sorted(rows,key=lambda x:x['source_id']), sorted(selected,key=lambda x:x['source_id'])):
        np.testing.assert_array_equal(a['mask'],b['mask'])


def test_geometric_depth_conflict_is_not_sent_to_vlm():
    rows,depth,image,k=fixture_scene()
    depth[rows[1]['mask']]=.9
    advisor=Mock()
    selected,audit=associate_fragments(rows,depth,k,image,Mock(),advisor=advisor)
    assert audit['candidate_pairs']==0
    advisor.assert_not_called()


def test_exact_frame_cache_and_scene_change(tmp_path):
    rows,depth,image,k=fixture_scene()
    path=tmp_path/'rgb.png';cv2.imwrite(str(path),image)
    client=Mock(model_name='test')
    client.describe_images.return_value=json.dumps(dict(relation='uncertain',reason='遮挡无法确认'))
    advisor=VLMFragmentAdvisor(client)
    masks=[r['mask'] for r in rows[:2]]
    first=advisor.assess(path,masks,context(),tmp_path/'advice')
    second=advisor.assess(path,masks,context(101.),tmp_path/'advice')
    assert not first['cache_hit'] and second['cache_hit'] and client.describe_images.call_count==1
    advisor.assess(path,masks,context(102.,epoch=2),tmp_path/'advice')
    assert client.describe_images.call_count==2
    image[0,0]=0;cv2.imwrite(str(path),image)
    advisor.assess(path,masks,context(103.,epoch=2),tmp_path/'advice')
    assert client.describe_images.call_count==3
    inputs=client.describe_images.call_args.args[0]
    assert inputs[0]==str(path) and len(inputs)==3
    x0,y0,x1,y1=rows[0]['bbox_xyxy']
    np.testing.assert_array_equal(cv2.imread(inputs[1]),image[y0:y1,x0:x1])
    assert 'interior_pixel_xy' in client.describe_images.call_args.args[1]


@pytest.mark.parametrize('reply',[None,'not JSON','{"relation":"same_object","reason":"ok","move":true}',
                                  '{"relation":"yes","reason":"ok"}'])
def test_unavailable_or_invalid_advice_preserves_geometry_and_is_not_cached(tmp_path,reply):
    rows,depth,image,k=fixture_scene();path=tmp_path/'rgb.png';cv2.imwrite(str(path),image)
    client=Mock(model_name='test')
    if reply is None:client.describe_images.side_effect=TimeoutError('endpoint')
    else:client.describe_images.return_value=reply
    advisor=VLMFragmentAdvisor(client)
    result=advisor.assess(path,[r['mask'] for r in rows[:2]],context(),tmp_path/'advice')
    assert result['relation']=='uncertain' and result['status']=='unavailable'
    assert not advisor.cache
    assert not result['mask_merge_authorized']


def test_config_can_disable_only_vlm_and_preserve_geometry_service():
    from runtime.config import SimpleAgentSystemConfig
    from runtime.services import build_occlusion_service
    from services.runtime_service_cache import clear_service_cache
    cfg=SimpleAgentSystemConfig(goal='test',enable_geometry_occlusion=True,
        enable_fragment_association=True,enable_vlm_fragment_assistance=True,
        geometry_occlusion_checkpoint='unused-in-construction.pt',
        planner_llm_base_url='http://127.0.0.1:8010/v1',planner_llm_model='test')
    clear_service_cache()
    try:
        enabled=build_occlusion_service(cfg)
        assert enabled.fragment_advisor is not None and enabled.backend is None
        cfg.enable_vlm_fragment_assistance=False
        disabled=build_occlusion_service(cfg)
        assert disabled is not enabled and disabled.fragment_advisor is None
        assert disabled.enable_fragment_association
    finally:
        clear_service_cache()
