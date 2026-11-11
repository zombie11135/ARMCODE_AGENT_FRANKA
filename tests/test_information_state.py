from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from runtime.information_state import (assess_information, information_change,
    record_information, invalidate_information)
from agents.occlusion_recovery_planner import build_recovery_snapshot, OcclusionRecoveryPlanner


def graph(localized=False, blocked=True, coverage=.8):
    nodes = [dict(node_id=f'n{i}', bbox_xyxy=[i*20, 0, i*20+10, 20], detection_score=.9,
                  quality=dict(depth_coverage_ratio=coverage, raw_mask_pixels=200,
                               depth_supported_pixels=int(200*coverage)), quality_issues=[],
                  observed_surface_camera=dict(point_count=int(200*coverage))) for i in range(2)]
    return dict(report_path='frame-a.json', report_sha256='sha-a', timestamp=1.,
                primary=dict(frame=dict(width=100,height=100), relation_graph=dict(nodes=nodes),
                             possible_pairs=[dict(source='n0',target='n1')] if blocked else []),
                target_association=dict(status='associated' if localized else 'target_not_identified',
                                        node_id='n1' if localized else None))


def test_unseen_target_is_unknown_even_when_scene_has_no_edges():
    state = assess_information(graph(blocked=False), '药盒')
    assert state['task_evidence_score'] == 0
    assert state['components']['target_measured_depth'] is None
    assert state['next_stage'] == 'information_acquisition'
    assert not state['task_completion_ready'] and not state['score_authorizes_motion']


def test_same_observation_and_genuine_regression_do_not_receive_action_bonus():
    before = assess_information(graph(True, False), '药盒')
    assert information_change(before, deepcopy(before))['task_evidence_delta'] == 0
    after = assess_information(graph(False, True), '药盒')
    after.update(source_report='frame-b.json', source_sha256='sha-b')
    change = information_change(before, after)
    assert change['task_evidence_delta'] < 0
    assert not change['causal_action_effect_proven']


def test_new_target_is_not_a_comparable_progress_measure():
    assert not information_change(assess_information(graph(), 'A'),
                                  assess_information(graph(), 'B'))['comparable']


def test_partial_depth_is_score_diagnostic_not_a_new_grasp_gate():
    state = assess_information(graph(True, False, .2), '药盒')
    assert state['next_stage'] == 'grasp_planning'
    assert state['task_evidence_score'] == 76
    assert 'motion_feasibility' in state['unchecked']
    no_depth = assess_information(graph(True, False, 0), '药盒')
    assert no_depth['next_stage'] == 'information_acquisition'


def test_clearance_can_improve_evidence_without_claiming_success():
    before = assess_information(graph(True, True), '药盒')
    after = assess_information(graph(True, False), '药盒')
    after['source_report']='new-frame.json'
    assert information_change(before, after)['task_evidence_delta'] == 30
    assert not after['task_completion_ready']


def test_invalidation_retains_last_measurement_and_waits_for_observation(tmp_path):
    robot = SimpleNamespace(information_states={}, geometry_recovery_state={'history':[]}, config=SimpleNamespace(
        nesyro_gate_trace_path=str(tmp_path/'trace.jsonl')))
    record_information(robot, '药盒', graph(True, True))
    score=robot.information_states['药盒']['task_evidence_score']
    invalidate_information(robot, 'grasp_started')
    assert not robot.information_states['药盒']['valid']
    assert robot.information_states['药盒']['task_evidence_score']==score
    record_information(robot, '药盒', graph(True, True))
    lines=[json.loads(s) for s in (tmp_path/'information_trace.jsonl').read_text().splitlines()]
    assert lines[1]['current_score'] is None
    assert lines[2]['change']['task_evidence_delta']==0
    record_information(robot, '药盒', graph(True, True))
    assert len((tmp_path/'information_trace.jsonl').read_text().splitlines())==3
    invalidate_information(robot,'new_task')
    assert not robot.information_states


def test_recovery_receives_hypotheses_and_keeps_expected_gain_unknown():
    from services.wrist_occlusion_service import choose_blocker
    g=graph()
    g['primary']['segmentation_selection']=dict(fragment_associations=dict(checks=[
        dict(accepted=False,fragment_source_ids=['a','b'],vlm_advice=dict(relation='same_object'))]))
    snapshot=build_recovery_snapshot(graph=g, decision=choose_blocker(g,allow_search=True),
                                     target='药盒',planning_only=True)
    assert snapshot['fragment_hypotheses'][0]['vlm_advice']['relation']=='same_object'
    assert snapshot['information_state']['scene_diagnostics']['unresolved_fragment_pairs']==1
    clear=next(a for a in snapshot['action_information_opportunities'] if a['action_id']=='plan_clear:n0')
    assert clear['expected_score_gain'] is None
    assert clear['requires_post_action_observation'] and clear['directly_reveals']==['n1']
    assert 'fragment_hypotheses' in OcclusionRecoveryPlanner.build_prompt(snapshot,compact=True)


def test_resolved_joint_prediction_no_longer_counts_as_unresolved():
    g=graph()
    g['primary']['segmentation_selection']=dict(fragment_associations=dict(checks=[
        dict(accepted=False,resolved_by_joint_prediction=True,vlm_advice=dict(relation='same_object'))]))
    assert assess_information(g,'药盒')['scene_diagnostics']['unresolved_fragment_pairs']==0


def test_real_observation_publication_and_scene_invalidation_are_wired(tmp_path):
    from core.context import RuntimeContext
    from skill_lib.simple_actions import SimpleRobotActions, SimpleRobotConfig
    from skill_lib.actions.observation_cache import _publish, _observation
    robot=SimpleRobotActions(RuntimeContext(), SimpleRobotConfig(
        nesyro_gate_trace_path=str(tmp_path/'gate.jsonl')))
    data={'geometry_occlusion':graph(True,False)}
    _publish(robot,'药盒',_observation('medicine','药盒',data),data)
    assert robot.information_states['药盒']['next_stage']=='grasp_planning'
    assert robot.geometry_recovery_state['history'][-1]['event']=='information_update'
    robot._discard_scene_evidence('transport_started')
    assert not robot.information_states['药盒']['valid']
