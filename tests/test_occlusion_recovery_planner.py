"""Recovery menus/LLM validation without endpoints, cameras or robot devices."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agents.occlusion_recovery_planner import (
    OcclusionRecoveryPlanner, RecoveryDecisionError, build_recovery_snapshot, validate_proposal)
from services.wrist_occlusion_service import choose_blocker


@pytest.fixture
def graph():
    nodes = []
    for i, xy in enumerate(((.4, .1), (.5, .1))):
        nodes.append(dict(node_id=f'node_{i}', display_id=f'N{i}', bbox_xyxy=[10+i*20, 10, 40+i*20, 40],
            quality=dict(depth_coverage_ratio=.95, raw_mask_pixels=200, depth_supported_pixels=190), quality_issues=[],
            detection_score=.9-i*.1, observed_surface_camera=dict(centroid_m=[*xy, .5])))
    return dict(report_path='not-a-live-observation.json', report_sha256='example-sha',
        source_binding={}, observation_provenance=dict(epoch=7),
        capture_geometry=dict(before=[0.]*6, calibration=dict(cameras=dict(wrist=dict(
            T_flange_camera=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])))),
        primary=dict(frame=dict(width=80, height=60), relation_graph=dict(nodes=nodes),
            possible_pairs=[dict(source='node_0', target='node_1')]),
        target_association=dict(status='target_not_identified', node_id=None, candidates=[]))


def snapshot(graph, **kwargs):
    return build_recovery_snapshot(graph=graph, decision=choose_blocker(graph, allow_search=True),
        target='西瓜霜', goal='将西瓜霜放到蓝色盘子中', **kwargs)


def test_partial_depth_is_diagnostic_and_legacy_quality_flag_does_not_block_clear(graph):
    graph['primary']['relation_graph']['nodes'][0]['quality']['depth_coverage_ratio'] = .708
    graph['primary']['relation_graph']['nodes'][0]['quality_issues'] = ['low_depth_coverage']
    s = snapshot(graph, clear_limit=3, buffers=[[450,-400,300,3.14,0,-1.57]])
    assert s['geometry_status'] == 'exploratory_clear_candidate'
    assert {a['action_id'] for a in s['allowed_actions']} == {'pause', 'observe:current', 'clear:node_0'}
    assert s['nodes'][0]['depth_coverage'] == .708
    assert s['clear_unavailable_reasons'] == []
    assert s['geometry_requirements']['min_depth_coverage'] is None
    assert s['geometry_requirements']['depth_coverage_filter_enabled'] is False
    assert s['nodes'][0]['quality_issues'] == []
    assert s['clear_remaining'] == 3 and not s['buffer_slots'][0]['occupied']


def test_same_view_retry_does_not_hide_a_legal_clear_or_another_view(graph):
    history = [dict(event='recovery_observation', target='西瓜霜', view_id='current',
                    observation_success=True, target_identified=False)]
    s = snapshot(graph, buffers=[[450,-400,300,3.14,0,-1.57]], history=history,
                 observation_count=1, views={'view_1': [400,100,600,3.14,0,0]})
    choices = {a['action_id'] for a in s['allowed_actions']}
    assert s['current_view_retry_used'] and s['observation_remaining'] == 1
    assert 'observe:current' not in choices
    assert 'clear:node_0' in choices and 'observe:view_1' in choices
    assert s['clear_unavailable_reasons'] == []


def test_observation_budget_exhaustion_does_not_disable_legal_clear(graph):
    s = snapshot(graph, buffers=[[450,-400,300,3.14,0,-1.57]], observation_count=2)
    assert {a['action_id'] for a in s['allowed_actions']} == {'pause', 'clear:node_0'}
    assert s['observation_remaining'] == 0 and s['clear_remaining'] == 1


@pytest.mark.parametrize('change', ['clear', 'different_view'])
def test_scene_or_view_change_allows_one_new_same_view_retry(graph, change):
    history = [dict(event='recovery_observation', target='西瓜霜', view_id='current')]
    if change == 'clear':
        history.append(dict(event='clear', target='西瓜霜', stage='finished', movement_completed=True))
    else:
        history.append(dict(event='recovery_observation', target='西瓜霜', view_id='view_1'))
    s = snapshot(graph, history=history, observation_count=1, observation_limit=3)
    assert not s['current_view_retry_used']
    assert 'observe:current' in {a['action_id'] for a in s['allowed_actions']}


def test_empty_depth_runtime_still_reports_reason_without_recapture_or_clear(graph, tmp_path, capsys):
    from core.context import RuntimeContext
    from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig
    from skill_lib.actions.llm_occlusion_recovery import recover_once
    graph['primary']['relation_graph']['nodes'][0]['quality'].update(
        depth_coverage_ratio=0., depth_supported_pixels=0)
    graph['primary']['relation_graph']['nodes'][0]['quality_issues'] = ['no_depth_support']
    planner = Mock()
    planner.decide.side_effect = lambda s: dict(snapshot_id=s['snapshot_id'], action_id='pause',
                                               reason='候选没有有效点云，需新视角')
    robot = SimpleRobotActions(RuntimeContext(occlusion_recovery_planner=planner), SimpleRobotConfig(
        enable_occlusion_recovery=True, occlusion_recovery_max_attempts=1,
        geometry_occlusion_buffer_poses_mmrad=[[450,-400,300,3.14,0,-1.57]],
        nesyro_gate_trace_path=str(tmp_path/'gate.jsonl')))
    robot.geometry_recovery_state['observation_counts']['西瓜霜'] = 1
    robot.geometry_recovery_state['history'].append(dict(event='recovery_observation',
        target='西瓜霜', view_id='current', observation_success=True, target_identified=False))
    robot._observe_medicine_once = Mock(side_effect=AssertionError('No redundant recapture'))
    robot._invoke_clear_occlusion_skill = Mock(side_effect=AssertionError('No unsupported clear'))
    kind, result = recover_once(robot, '西瓜霜', graph, choose_blocker(graph, allow_search=True))
    assert kind == 'stop' and not result.success and '清障尚未执行' in result.error
    assert result.data['recovery_snapshot']['observation_remaining'] == 1
    output = capsys.readouterr().out
    assert 'depth=0.0%' in output and 'issues=no_depth_support' in output
    assert 'depth coverage filter: disabled' in output and 'required=80.0%' not in output
    assert 'clear_remaining=1' in output and 'free_buffers=1' in output
    robot._observe_medicine_once.assert_not_called()
    robot._invoke_clear_occlusion_skill.assert_not_called()


def test_retired_depth_threshold_argument_cannot_reenable_the_gate(graph):
    node = graph['primary']['relation_graph']['nodes'][0]
    node['quality']['depth_coverage_ratio'] = .714
    node['quality_issues'] = ['low_depth_coverage']
    assert choose_blocker(graph, allow_search=True, min_depth_fraction=.99)['blocker']['node_id'] == 'node_0'
    graph['target_association'] = dict(status='associated', node_id='node_0')
    assert choose_blocker(graph)['status'] == 'no_occlusion_evidence'


@pytest.mark.parametrize('issue', ['truncated_at_image_border', 'overlapping_visible_mask_assignments'])
def test_image_quality_warnings_do_not_veto_target_or_blocker_with_measured_depth(graph, issue):
    from copy import deepcopy
    original = deepcopy(graph)
    graph['primary']['relation_graph']['nodes'][0]['quality_issues'] = ['low_depth_coverage', issue]
    annotated = deepcopy(graph)
    s = snapshot(graph, buffers=[[450,-400,300,3.14,0,-1.57]])
    assert s['geometry_status'] == 'exploratory_clear_candidate'
    assert 'clear:node_0' in {a['action_id'] for a in s['allowed_actions']}
    assert s['nodes'][0]['quality_issues'] == []
    assert s['nodes'][0]['quality_diagnostics'] == ['low_depth_coverage', issue]
    assert graph == annotated  # Saved evidence and its hashes remain usable.
    graph['target_association'] = dict(status='associated', node_id='node_0')
    assert choose_blocker(graph)['status'] == 'no_occlusion_evidence'
    # A real relation still selects its upstream blocker, even with a border warning.
    graph['target_association'] = dict(status='associated', node_id='node_1')
    assert choose_blocker(graph)['blocker']['node_id'] == 'node_0'
    assert graph['primary']['possible_pairs'] == original['primary']['possible_pairs']


@pytest.mark.parametrize('missing', ['mask', 'depth'])
def test_missing_grasp_inputs_still_block_even_when_image_quality_warnings_are_diagnostic(graph, missing):
    node = graph['primary']['relation_graph']['nodes'][0]
    node['quality_issues'] = ['truncated_at_image_border', 'overlapping_visible_mask_assignments']
    if missing == 'mask':
        node['quality']['raw_mask_pixels'] = 0
    else:
        node['quality']['depth_supported_pixels'] = 0
    assert choose_blocker(graph, allow_search=True)['status'] == 'blocker_quality_insufficient'
    graph['target_association'] = dict(status='associated', node_id='node_0')
    assert choose_blocker(graph)['status'] == 'target_quality_insufficient'


def test_occupied_slots_and_spent_clear_budget_are_not_reset(graph):
    s = snapshot(graph, buffers=[[450,-400,300,3.14,0,-1.57]],
        slots_used=1, clear_count=1, clear_limit=1, observation_count=2)
    assert s['allowed_actions'] == [dict(action_id='pause', action='pause')]
    assert s['buffer_slots'][0]['occupied'] and s['clear_remaining'] == 0


def test_next_clear_uses_only_free_slot_and_excludes_occupied_buffer_region(graph):
    buffers = [[700,-400,300,3.14,0,-1.57], [500,-400,300,3.14,0,-1.57]]
    s = snapshot(graph, buffers=buffers, slots_used=1, clear_count=1, clear_limit=2)
    clear = next(a for a in s['allowed_actions'] if a['action'] == 'clear')
    assert clear['buffer_slot'] == 1 and clear['node_id'] == 'node_0'
    buffers[0][:2] = [400, 100]  # The candidate surface is now inside the occupied region.
    blocked = snapshot(graph, buffers=buffers, slots_used=1, clear_count=1, clear_limit=2)
    assert not any(a['action'] == 'clear' for a in blocked['allowed_actions'])
    assert 'inside_occupied_buffer_xy_region' in blocked['nodes'][0]['exclusions']


def test_occupied_buffer_requires_capture_geometry_before_another_clear(graph):
    graph['capture_geometry'] = None
    s = snapshot(graph, buffers=[[700,-400,300,3.14,0,-1.57], [500,-400,300,3.14,0,-1.57]],
        slots_used=1, clear_count=1, clear_limit=2)
    assert not any(a['action'] == 'clear' for a in s['allowed_actions'])


@pytest.mark.parametrize('damage', ['old_frame', 'unknown_node', 'arbitrary_pose', 'threshold', 'empty_reason'])
def test_unavailable_or_stale_model_choices_are_rejected(graph, damage):
    s = snapshot(graph)
    proposal = dict(snapshot_id=s['snapshot_id'], action_id='observe:current', reason='获取新深度')
    if damage == 'old_frame': proposal['snapshot_id'] = 'old-epoch'
    elif damage == 'unknown_node': proposal['action_id'] = 'clear:other_frame/node_0'
    elif damage == 'arbitrary_pose': proposal['xyz'] = [450,-400,300]
    elif damage == 'threshold': proposal['min_depth'] = .1
    else: proposal['reason'] = ''
    with pytest.raises(RecoveryDecisionError): validate_proposal(proposal, s)


def test_model_response_is_logged_and_validated_without_tool_execution(graph, tmp_path):
    s = snapshot(graph)
    client = Mock(model_name='offline-test')
    client.chat.return_value = json.dumps(dict(snapshot_id=s['snapshot_id'],
        action_id='observe:current', reason='深度不足，重新采集'))
    result = OcclusionRecoveryPlanner(client, tmp_path).decide(s)
    assert result['action'] == 'observe'
    log = json.loads(open(result['trace_path']).read())
    assert log['accepted'] and log['snapshot'] == s
    client.chat.assert_called_once()


def test_invalid_model_response_is_audited_and_cannot_run_an_action(graph, tmp_path):
    client = Mock(model_name='offline-test')
    client.chat.return_value = '{"action_id":"grasp-anyway"}'
    with pytest.raises(RecoveryDecisionError):
        OcclusionRecoveryPlanner(client, tmp_path).decide(snapshot(graph))
    logs = list(tmp_path.glob('*.json'))
    assert len(logs) == 1 and not json.loads(logs[0].read_text())['accepted']


def test_preferred_blocker_cannot_bypass_geometry(graph):
    decision = choose_blocker(graph, allow_search=True, preferred_node_id='node_0')
    assert decision['blocker']['node_id'] == 'node_0'
    with pytest.raises(ValueError):
        choose_blocker(graph, allow_search=True, preferred_node_id='node_1')


def test_recovery_client_uses_configured_planner_without_starting_services(tmp_path):
    from agents.occlusion_recovery_planner import build_recovery_planner
    config = SimpleNamespace(planner_llm_base_url='http://offline.invalid/v1',
        planner_llm_api_key='test-only', planner_llm_model='offline-test',
        vlm_timeout=120, capture_dir=str(tmp_path))
    with patch('services.openai_chatgpt_vlm_client.OpenAIChatGPTVLMClient') as client:
        result = build_recovery_planner(config)
        client.assert_called_once_with(base_url=config.planner_llm_base_url,
            api_key='test-only', model_name='offline-test', timeout=30)
        result.client.chat.assert_not_called()


def test_cli_rejects_recovery_without_geometry_before_profile_or_device_setup(monkeypatch):
    import sys
    from app import cli
    monkeypatch.setattr(sys, 'argv', ['app', '--enable_llm_occlusion_recovery'])
    with patch('hardware.runtime.apply_cli_profile') as apply:
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 2
        apply.assert_not_called()
