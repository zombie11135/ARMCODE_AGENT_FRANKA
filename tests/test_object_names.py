"""Names/brands never become guessed robot targets; all tests use saved/mock inputs."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

from agents.plan_graph import PlanGraph, PlanNode
from agents.planning_agent import PlanningAgent
from agents.simple_code_generator import SimpleCodeGenerator
from agents.simple_execution_runtime import SimpleExecutionRuntime
from core.context import RuntimeContext
from core.skill_base import SkillResult
from services.object_names import ObjectNameResolver, NameResolutionRequired, bind_plan_names, handle_name_command
from services.object_reference_service import ObjectReferenceService, ReferenceSettings, ReferenceGallery
from skill_lib.simple_actions import grasp, observe, place
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill


class ColorEncoder:
    def __init__(self): self.calls = 0
    def encode(self, images, batch_size=16):
        self.calls += 1
        return np.array([np.asarray(im).mean(axis=(0, 1))+.01 for im in images])


@pytest.fixture
def scene(tmp_path):
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'test')
    encoder = ColorEncoder()
    service = ObjectReferenceService(ReferenceSettings(gallery_dir=str(tmp_path/'gallery'),
        checkpoint=str(checkpoint), device='cpu', log_dir=str(tmp_path/'logs')), encoder=encoder)
    for name, color in [('red product', 'red'), ('blue product', 'blue')]:
        path = tmp_path/f'{color}.png'; Image.new('RGB', (20, 20), color).save(path)
        service.gallery.register(name=name, image_path=path, category='cube', detector_prompt='cube',
            aliases=[f'{color} cube'], attributes={'brand':'999'}, descriptors=['demo brand'])
    image = tmp_path/'scene.png'
    im=Image.new('RGB', (40,20), 'red'); im.paste('blue', (20,0,40,20)); im.save(image)
    detector = Mock()
    detector.detect.return_value={'detections':[
        {'object_id':'a','bbox':[0,0,20,20],'confidence':.99},
        {'object_id':'b','bbox':[20,0,40,20],'confidence':.99}]}
    context = RuntimeContext(object_references=service, detector=detector, camera=Mock())
    context.camera.capture_wrist.return_value={'color_path':str(image)}
    return service, context, image, encoder


def test_schema1_read_is_nonmutating_and_ids_survive_rename(tmp_path):
    gallery=ReferenceGallery(tmp_path)
    raw=json.dumps({'schema_version':1,'objects':[{'name':'old name','aliases':[],
        'category':'cube','detector_prompt':'cube','references':[]}]})
    gallery.manifest.write_text(raw)
    original=gallery.resolve('old name')['object_id']
    assert gallery.manifest.read_text()==raw
    gallery.update('old name', name='new full name', aliases=['short'])
    assert gallery.resolve('short')['object_id']==original
    assert gallery.resolve('old name')['name']=='new full name'
    assert gallery.read()['schema_version']==2


def test_exact_alias_id_and_unicode_normalization(scene):
    service, _, _, encoder=scene
    resolver=ObjectNameResolver(service)
    red=service.resolve('red product')
    for query in ['red product','red cube',' ＲＥＤ　ＣＵＢＥ ',red['object_id']]:
        assert resolver.lookup(query)['object_id']==red['object_id']
    assert encoder.calls==0
    assert resolver.lookup('999')['status']=='needs_scene'
    assert len(resolver.lookup('999')['candidates'])==2


def test_unknown_or_similar_names_are_not_automatically_aliased(scene):
    service, _, _, _=scene
    resolver=ObjectNameResolver(service)
    assert resolver.lookup('red product 10mg')['status']=='unknown'
    assert resolver.lookup('red products')['status']=='unknown'
    assert resolver.mentions('抓取19999药盒')==[]


def test_longest_registered_name_does_not_also_match_brand(scene):
    service, _, _, _=scene
    service.gallery.update('red product', aliases=['999 red product'])
    mentions=ObjectNameResolver(service).mentions('抓取999 red product')
    assert len(mentions)==1 and mentions[0]['status']=='resolved'


def test_temporary_binding_is_scoped_and_cannot_override_another_name(scene):
    service, _, _, _=scene
    target=service.resolve('red product')['object_id']
    bound=ObjectNameResolver(service, {'999':target})
    assert bound.lookup('999')['object_id']==target
    assert ObjectNameResolver(service).lookup('999')['status']=='needs_scene'
    with pytest.raises(ValueError, match='覆盖'):
        ObjectNameResolver(service, {'blue product':target})


def test_shared_brand_with_two_visible_products_requests_clarification(scene):
    service, context, image, _=scene
    result=ObjectNameResolver(service).resolve_in_image('999',context,str(image))
    assert result['status']=='clarification_required'
    assert result['reason']=='multiple_visible_identities'
    assert context.detector.detect.call_count==1  # both names share candidate extraction
    context.camera.capture_wrist.assert_not_called()  # supplied image only


def test_shared_brand_selects_only_confirmed_visible_identity(scene):
    service, context, image, _=scene
    context.detector.detect.return_value['detections']=context.detector.detect.return_value['detections'][:1]
    result=ObjectNameResolver(service).resolve_in_image('999',context,str(image))
    assert result['status']=='resolved' and result['source']=='scene_reference'
    assert result['canonical_name']=='red product'


def test_no_confirmed_match_never_selects_highest_failed_score(scene):
    service, context, image, _=scene
    Image.new('RGB',(40,20),'green').save(image)
    result=ObjectNameResolver(service).resolve_in_image('999',context,str(image))
    assert result['status']=='clarification_required'
    assert result['reason']=='no_confirmed_visible_identity'


def test_visual_error_for_another_candidate_does_not_prove_uniqueness(scene):
    service,context,image,_=scene
    with patch('skill_lib.perception.reference_recognition.RecognizeObjectSkill.run',
               side_effect=[SkillResult(True,{}),SkillResult(False,{'stage':'reference_error'},error='model failed')]):
        result=ObjectNameResolver(service).resolve_in_image('999',context,str(image))
    assert result['status']=='clarification_required' and result['reason']=='incomplete_visual_evidence'


def make_graph(target):
    return PlanGraph(goal='test',nodes=[
        PlanNode('a','pick','pick','pick',skill='grasp',args={'target':target}),
        PlanNode('b','place','place','place',skill='place',args={'target':'pink tray'})])


def test_planner_cannot_silently_turn_brand_into_one_product(scene):
    service, _, _, _=scene
    graph=bind_plan_names(make_graph('red product'),'把999放到盘子',service)
    assert graph.nodes[0].args['target']=='999'
    assert len(graph.nodes[0].args['name_candidate_ids'])==2
    assert graph.nodes[1].args['target']=='pink tray'


def test_explicit_name_wins_and_destination_is_not_repurposed_as_pick_target(scene):
    service, _, _, _=scene
    graph=bind_plan_names(make_graph('red cube'),'把red cube放到盘子',service)
    actions=SimpleCodeGenerator().build_actions(graph)
    assert actions[0].target==service.resolve('red product')['object_id']
    graph=bind_plan_names(make_graph('unregistered medicine'),'把药盒放到red cube旁边',service)
    assert graph.nodes[0].args['target']=='unregistered medicine'


def test_existing_llm_call_gets_catalog_and_returns_stable_id(scene):
    service, context, _, _=scene
    context.vlm=Mock()
    context.vlm.chat.return_value=json.dumps(make_graph('red cube').to_dict())
    graph=PlanningAgent(context,use_llm=True).plan('抓取red cube放到粉色盘子',scene_text='boxes')
    assert graph.nodes[0].args['target']==service.resolve('red product')['object_id']
    assert context.vlm.chat.call_count==1


def test_name_edits_do_not_reencode_gallery_or_change_identity(scene):
    service, _, image, encoder=scene
    candidate=[{'bbox':[0,0,20,20]}]
    first=service.match(image_path=image,candidates=candidate,query='red product')
    before=encoder.calls
    service.gallery.update('red product',name='full red product',aliases=['short red'],attributes={'specification':'10ml'})
    second=service.match(image_path=image,candidates=candidate,query='short red')
    assert second['accepted'] and encoder.calls==before
    assert first['gallery_revision']==second['gallery_revision']
    assert first['identity_id']==second['identity_id']
    assert second['identity']=='full red product'


def test_ocr_fallback_receives_readable_name_and_aliases_not_id(scene):
    service, context, image, _=scene
    target=service.resolve('red product')['object_id']
    skill=FindAndGraspMedicineSkill(context)
    with patch.object(skill,'_recognize_legacy',return_value=SkillResult(True,{})) as legacy:
        result=skill._recognize_image(image_path=str(image),query=target,ocr_verify_strict_mode=True)
    assert legacy.call_args.kwargs['query']=='red product'
    assert 'red cube' in legacy.call_args.kwargs['aliases']
    assert result.data['target_identity']['object_id']==target


def test_execution_preflight_unifies_aliases_and_preserves_place(scene,tmp_path):
    service,context,_,_=scene
    runtime=SimpleExecutionRuntime(SimpleNamespace(plan_json=str(tmp_path/'plan.json'),target_name_bindings={}))
    actions=runtime._resolve_action_targets(context,[observe('red product'),grasp('red cube'),place('pink tray')])
    assert actions[0].target==actions[1].target==service.resolve('red product')['object_id']
    assert actions[2].target=='pink tray'
    context.camera.capture_wrist.assert_not_called()


def test_ambiguous_execution_stops_before_hardware_session(scene,tmp_path):
    service,context,_,_=scene
    config=SimpleNamespace(camera_warmup_s=0,hardware_config_path='profile',execute_robot=True,
                           plan_json=str(tmp_path/'plan.json'),target_name_bindings={})
    runtime=SimpleExecutionRuntime(config)
    camera_service=Mock()
    with patch.object(runtime,'_build_execution_context',return_value=(context,camera_service)), \
         patch('hardware.runtime.require_task_geometry'),patch('hardware.runtime.build_hardware') as hardware, \
         patch.object(runtime,'_build_robot') as robot:
        with pytest.raises(NameResolutionRequired):
            runtime.run([grasp('999'),place('pink tray')])
    hardware.assert_not_called(); robot.assert_not_called()
    assert context.camera.capture_wrist.call_count==1


def test_interactive_bind_uses_ids_and_never_edits_gallery(scene):
    service,_,_,_=scene
    before=service.gallery.manifest.read_bytes()
    args=SimpleNamespace(hardware_config='test')
    with patch('services.object_reference_service.build_reference_service',return_value=service):
        assert handle_name_command('/bind 999 = red product',args)
        assert args.next_task_name_bindings['999']==service.resolve('red product')['object_id']
        handle_name_command('/bind-clear',args)
        assert args.next_task_name_bindings=={}
    assert service.gallery.manifest.read_bytes()==before


def test_ambiguity_returns_report_without_automatic_replan(scene,tmp_path):
    from agents.simple_agent_system import SimpleArmCodeAgentSystem, SimpleAgentSystemConfig
    service,_,_,_=scene
    system=SimpleArmCodeAgentSystem(SimpleAgentSystemConfig(goal='999',
        project_root=str(tmp_path), failure_report_path=str(tmp_path/'failure.json')))
    result=ObjectNameResolver(service).lookup('999')
    with patch.object(system,'_run_task',side_effect=NameResolutionRequired(result)):
        outcome=system.run()
    assert not outcome.success and outcome.replan_attempts==0
    assert json.loads((tmp_path/'failure.json').read_text())['automatic_replan'] is False


def test_specification_change_requires_explicit_metadata_edit(scene):
    service,_,image,_=scene
    service.gallery.update('red product',attributes={'specification':'10ml'})
    before=service.gallery.manifest.read_bytes()
    with pytest.raises(ValueError,match='specification differs'):
        service.gallery.register(name='red product',image_path=image,category='cube',detector_prompt='cube',
                                 attributes={'specification':'20ml'})
    assert service.gallery.manifest.read_bytes()==before


def test_bad_alias_can_be_removed_without_losing_images(scene):
    service,_,_,_=scene
    before=service.resolve('red product')
    service.gallery.update('red product',remove_aliases=['red cube'])
    assert service.resolve('red cube') is None
    assert service.resolve('red product')['references']==before['references']


def test_reusing_released_old_name_allocates_a_distinct_id(scene):
    service,_,image,_=scene
    original=service.resolve('red product')['object_id']
    service.gallery.update('red product',name='renamed product',remove_aliases=['red product'])
    service.gallery.register(name='red product',image_path=image,category='cube',detector_prompt='cube')
    assert service.resolve('red product')['object_id'] != original
    assert service.resolve('renamed product')['object_id'] == original


def test_registration_cli_roundtrip_without_models_or_hardware(tmp_path):
    import subprocess, sys
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    photo=tmp_path/'crop.png'; Image.new('RGB',(20,20),'red').save(photo)
    settings=tmp_path/'settings.yaml'
    settings.write_text(yaml.safe_dump(dict(enabled=True,gallery_dir=str(tmp_path/'gallery'),
        checkpoint=str(tmp_path/'not-loaded.pt'),log_dir=str(tmp_path/'logs'),device='cpu')))
    def cli(*args):
        run=subprocess.run([sys.executable,'-m','hardware.reference_cli','--config',str(settings),*args],
                           cwd=root,text=True,capture_output=True,check=True)
        return json.loads(run.stdout)
    created=cli('register','demo cube','--category','cube','--prompt','cube',
        '--alias','full cube name','--brand','demo','--images',str(photo))
    identity=created['references'][0]['object_id']
    assert cli('resolve','full cube name')['object_id']==identity
    assert cli('resolve','demo')['status']=='needs_scene'
    cli('update',identity,'--name','renamed cube','--alias','short cube')
    assert cli('resolve','short cube')['object_id']==identity
    assert cli('resolve','demo','--bind',f'demo={identity}')['source']=='task_binding'


def test_cli_consumes_binding_even_when_task_fails(scene):
    from skill_lab import run_agent_system as cli
    service,_,_,_=scene
    identity=service.resolve('red product')['object_id']
    args=SimpleNamespace(next_task_name_bindings={'999':identity},hardware_config=None,
                         simple_actions=True,debug_output=False,no_run_executor=False)
    captured=[]
    def config_builder(args,**kwargs):
        cfg=SimpleNamespace(target_name_bindings=dict(args.next_task_name_bindings))
        captured.append(cfg)
        return cfg
    result=SimpleNamespace(success=False,replan_attempts=0,failure_report_path='test',replan_md=None,error='test ambiguity')
    with patch.object(cli,'build_config',side_effect=config_builder),patch.object(cli,'SimpleArmCodeAgentSystem') as system:
        system.return_value.run.return_value=result
        assert not cli.run_goal(args,'抓取999',[])
        assert args.next_task_name_bindings=={}
        assert captured[0].target_name_bindings=={'999':identity}
        cli.run_goal(args,'抓取999',[])
        assert captured[1].target_name_bindings=={}
