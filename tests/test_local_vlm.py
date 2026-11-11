"""Local routing, GPU request coordination and replay scoring without model imports."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from runtime.local_vlm import PROFILES, apply_local_endpoint, server_command
from runtime.vlm_benchmark import expected_recovery_actions, score_case, summarize
from agents.planning_agent import PlanningAgent
from agents.simple_code_generator import SimpleCodeGenerator
from agents.occlusion_recovery_planner import proposal_from_choice, RecoveryDecisionError
from core.context import RuntimeContext
from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient


def test_local_selection_overrides_all_role_specific_remote_defaults():
    args = SimpleNamespace(local_vlm='qwen3vl-8b-awq', local_vlm_port=8123,
        planner_llm_base_url='https://old.example/v1', fallback_vlm_model='old-model')
    apply_local_endpoint(args)
    for prefix in ('api', 'vlm', 'planner_llm', 'fallback_vlm', 'grasp_vdm', 'vlm_occlusion'):
        assert getattr(args, prefix+'_base_url') == 'http://127.0.0.1:8123/v1'
        assert getattr(args, prefix+'_model') == 'qwen3vl-8b-awq'
        assert getattr(args, 'api_key' if prefix == 'api' else prefix+'_api_key') == 'local'


def test_no_local_selection_preserves_provider_settings():
    args = SimpleNamespace(api_base_url='https://configured.example/v1')
    apply_local_endpoint(args)
    assert vars(args) == {'api_base_url': 'https://configured.example/v1'}


def test_local_profile_uses_isolated_backend_and_retains_multimodal_capacity(tmp_path, monkeypatch):
    python = tmp_path/'python'
    python.touch()
    monkeypatch.setenv('ARM_LOCAL_VLM_PYTHON', str(python))
    monkeypatch.setenv('ARM_LOCAL_VLM_ROOT', str(tmp_path))
    root = tmp_path/'qwen35-9b-awq'
    root.mkdir()
    (root/'config.json').write_text('{}')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'part.safetensors'}}))
    with pytest.raises(FileNotFoundError, match='incomplete'):
        server_command('qwen35-9b-awq')
    (root/'part.safetensors').touch()
    command = server_command('qwen35-9b-awq')
    assert command[0] == str(python)
    assert command[command.index('--host')+1] == '127.0.0.1'
    assert json.loads(command[command.index('--limit-mm-per-prompt')+1]) == {'image': 4, 'video': 0}
    assert command[command.index('--gpu-memory-utilization')+1] == '0.45'
    assert all(len(p['revision']) == 40 for p in PROFILES.values())


@pytest.mark.parametrize('mode', ['chat', 'vision'])
def test_local_calls_use_gpu_coordinator_and_disable_thinking(mode, tmp_path):
    client = OpenAIChatGPTVLMClient('http://127.0.0.1:8010/v1', 'local', 'qwen35-9b-awq')
    response = Mock(status_code=200)
    response.json.return_value = {'choices': [{'message': {'content': '{"ok":true}'}}]}
    image = tmp_path/'color.png'
    image.write_bytes(b'image fixture')
    with patch('services.openai_chatgpt_vlm_client._post_local', return_value=response) as local, \
         patch('services.openai_chatgpt_vlm_client.requests.post') as remote:
        if mode == 'chat':
            assert client.chat('test') == '{"ok":true}'
        else:
            assert client.describe_images([str(image)], 'test') == '{"ok":true}'
        assert local.call_args.kwargs['json']['chat_template_kwargs'] == {'enable_thinking': False}
        remote.assert_not_called()


def test_remote_calls_keep_provider_payload_unchanged():
    client = OpenAIChatGPTVLMClient('https://provider.example/v1', 'test', 'existing-model')
    response = Mock(status_code=200)
    response.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
    with patch('services.openai_chatgpt_vlm_client.requests.post', return_value=response) as post:
        assert client.chat('hello') == 'ok'
        assert 'chat_template_kwargs' not in post.call_args.kwargs['json']


def test_replay_does_not_treat_historical_action_or_parseable_json_as_correctness():
    snapshot = dict(snapshot_id='id', allowed_actions=[{'action_id': 'pause', 'action': 'pause'},
        {'action_id': 'clear:n1', 'action': 'clear'}])
    case = dict(kind='recovery', snapshot=snapshot,
        expected_action_ids=expected_recovery_actions(snapshot), historical_action_id='pause')
    wrong = score_case(case, json.dumps(dict(snapshot_id='id', action_id='pause', reason='停下')))
    assert wrong['legal_action'] and wrong['historical_agreement'] and not wrong['pass_check']
    right = score_case(case, json.dumps(dict(snapshot_id='id', action_id='clear:n1', reason='推进清障')))
    assert right['pass_check'] and not right['historical_agreement']


def test_visual_boolean_labels_are_not_satisfied_by_strings():
    case = dict(kind='vision', expected={'target_occluded': False})
    assert score_case(case, '{"target_occluded":false}')['pass_check']
    assert not score_case(case, '{"target_occluded":"false"}')['pass_check']


def test_failed_requests_remain_in_latency_and_quality_counts():
    summary = summarize([dict(kind='plan', pass_check=True, strict_json=True, call_wall_s=1.),
                         dict(kind='plan', pass_check=False, call_wall_s=9.)])
    assert summary['kinds']['plan'] == dict(count=2, passed=1, strict_json_count=1,
        median_s=5., p95_s=9., total_tokens=0)


def test_compact_plan_builds_existing_action_graph_without_changing_targets():
    data = {'actions': [{'op': 'grasp', 'target': '西瓜霜'}, {'op': 'place', 'target': 'blue tray'},
                        {'op': 'grasp', 'target': '抗病毒口服液'}, {'op': 'place', 'target': 'pink tray'}]}
    graph = PlanningAgent._compact_actions_to_graph(data, '原始任务', ['保留执行状态'])
    assert graph.goal == '原始任务' and graph.constraints == ['保留执行状态']
    assert [(a.op, a.target) for a in SimpleCodeGenerator().build_actions(graph)] == [
        ('grasp', '西瓜霜'), ('place', 'blue tray'), ('grasp', '抗病毒口服液'), ('place', 'pink tray')]
    assert [(e.source, e.target) for e in graph.edges] == [('n0','n1'), ('n1','n2'), ('n2','n3')]


@pytest.mark.parametrize('data', [{'actions': []}, {'actions': [{'op': 'move', 'target': 'x'}]},
                                 {'actions': [{'op': 'grasp', 'target': ''}]},
                                 {'actions': [], 'goal': '替换任务'}])
def test_invalid_compact_plans_are_not_executable(data):
    with pytest.raises(ValueError):
        PlanningAgent._compact_actions_to_graph(data, '原始任务', [])


def test_compact_model_is_used_through_the_real_planner_pipeline():
    client = SimpleNamespace(compact_planning=True, chat=Mock(return_value=json.dumps({'actions': [
        {'op': 'grasp', 'target': '西瓜霜'}, {'op': 'place', 'target': 'blue tray'}]})))
    graph = PlanningAgent(context=RuntimeContext(vlm=client), use_llm=True).plan('将西瓜霜放到蓝色盘子中')
    assert graph.metadata['planner'] == 'llm'
    assert graph.metadata['response_format'] == 'compact_actions_v1'
    assert [(a.op,a.target) for a in SimpleCodeGenerator().build_actions(graph)] == [
        ('grasp','西瓜霜'), ('place','blue tray')]


def test_compact_recovery_binds_full_ids_from_the_same_request_menu():
    snapshot = dict(snapshot_id='current-snapshot', allowed_actions=[{'action_id': 'pause'},
        {'action_id': 'clear:long-observation-id/wrist/instance_002'}])
    assert proposal_from_choice({'choice': 1, 'reason': '推进清障'}, snapshot) == dict(
        snapshot_id='current-snapshot', action_id='clear:long-observation-id/wrist/instance_002', reason='推进清障')
    for choice in (-1, 2, True, '1'):
        with pytest.raises(RecoveryDecisionError):
            proposal_from_choice({'choice': choice, 'reason': '测试'}, snapshot)


def test_sam_weights_are_shared_only_for_compatible_model_settings(tmp_path):
    import sys
    from services.runtime_service_cache import clear_service_cache
    from services.sam3_model_pool import image_model
    builder = Mock(side_effect=lambda **kwargs: SimpleNamespace(to=lambda **kwargs: object()))
    clear_service_cache()
    with patch.dict(sys.modules, {'sam3.model_builder': SimpleNamespace(build_sam3_image_model=builder)}):
        a = image_model(tmp_path/'sam3.pt', device='cpu', interactive=True)
        assert image_model(tmp_path/'sam3.pt', device='cpu', interactive=True) is a
        assert image_model(tmp_path/'sam3.pt', device='cpu', interactive=False) is not a
        assert image_model(tmp_path/'other.pt', device='cpu', interactive=True) is not a
        assert builder.call_count == 3
    clear_service_cache()


def test_local_server_gpu_override_does_not_change_perception_environment(tmp_path, monkeypatch):
    import os
    from runtime.local_vlm import service_environment
    monkeypatch.setenv('ARM_LOCAL_VLM_CACHE', str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.delenv('ARM_LOCAL_VLM_CUDA_VISIBLE_DEVICES', raising=False)
    assert service_environment()['CUDA_VISIBLE_DEVICES'] == '0'
    monkeypatch.setenv('ARM_LOCAL_VLM_CUDA_VISIBLE_DEVICES', '1')
    assert service_environment()['CUDA_VISIBLE_DEVICES'] == '1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'cuda:0', 'cuda:1'])
def test_sam_weights_match_the_requested_inference_device(tmp_path, device):
    import sys
    from services.runtime_service_cache import clear_service_cache
    from services.sam3_model_pool import image_model

    class Model:
        device = 'cpu'
        def to(self, *, device):
            self.device = device
            return self
        def infer(self, input_device):
            if self.device != input_device:
                raise RuntimeError('Input and weights are on different devices')
            return 'inference completed'

    # Emulate the affected SDK: an indexed CUDA device leaves weights on CPU.
    def build(**kwargs):
        model = Model()
        if kwargs['device'] == 'cuda':
            model.device = 'cuda'
        return model

    clear_service_cache()
    try:
        with patch.dict(sys.modules, {'sam3.model_builder': SimpleNamespace(build_sam3_image_model=build)}):
            model = image_model(tmp_path/'sam3.pt', device=device)
            assert model.infer(device) == 'inference completed'
    finally:
        clear_service_cache()
