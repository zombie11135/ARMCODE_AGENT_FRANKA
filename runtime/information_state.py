"""Task-conditioned evidence, not a calibrated visibility/success probability.

Only fresh observations update scores. Actions invalidate current evidence;
their expected benefit and measured subsequent change remain separate.
"""
import json
from pathlib import Path
import time


WEIGHTS = dict(target_localization=40., target_measured_depth=30., target_relation_clarity=30.)


def assess_information(graph, target):
    from services.wrist_occlusion_service import choose_blocker
    view = graph['primary']
    association = graph['target_association']
    nodes = view['relation_graph']['nodes']
    node = next((n for n in nodes if n['node_id'] == association.get('node_id')), None)
    localized = node is not None and association['status'] == 'associated'
    decision = choose_blocker(graph, allow_search=True)
    checks = view.get('segmentation_selection', {}).get('fragment_associations', {}).get('checks', [])
    unresolved = [c for c in checks if not c['accepted'] and not c.get('resolved_by_joint_prediction')]
    depth = float(node['quality']['depth_coverage_ratio']) if localized else None
    clear = decision['status'] == 'no_occlusion_evidence'
    # Clarity means absence of detected blocking evidence, not hidden-area coverage.
    components = dict(target_localization=float(localized), target_measured_depth=depth,
                      target_relation_clarity=float(clear) if localized else None)
    score = round(sum(WEIGHTS[k]*(v if v is not None else 0.) for k, v in components.items()), 2)
    missing = []
    if not localized:
        missing.append('target_localization')
    if depth is None or node['quality'].get('depth_supported_pixels', 0) == 0:
        missing.append('target_measured_depth')
    if not clear:
        missing.append('target_occlusion_or_relation_resolution')
    return dict(schema='task_information_v1', target=target, valid=True,
        source_report=graph.get('report_path'), source_sha256=graph.get('report_sha256'),
        timestamp=graph.get('timestamp'), provenance=graph.get('observation_provenance'),
        task_evidence_score=score, components=components, weights=WEIGHTS.copy(),
        unknown_components=[k for k, v in components.items() if v is None], missing_evidence=missing,
        next_stage='grasp_planning' if clear else 'information_acquisition',
        geometry_status=decision['status'],
        scene_diagnostics=dict(visible_instances=len(nodes), possible_occlusion_edges=len(view['possible_pairs']),
            unresolved_fragment_pairs=len(unresolved),
            vlm_same_object_hypotheses=sum(c.get('vlm_advice', {}).get('relation') == 'same_object' for c in unresolved)),
        task_completion_ready=False, unchecked=['collision_free_grasp', 'motion_feasibility',
            'holding_verification', 'destination_surface', 'placement_verification'],
        calibrated_probability=None, score_authorizes_motion=False)


def information_change(before, after):
    if not before or before['target'] != after['target']:
        return dict(comparable=False, task_evidence_delta=None, reason='no_previous_same_target_observation')
    # Identical source observations, including cache hits, are not new evidence.
    same = (before.get('source_report'), before.get('source_sha256')) == (
        after.get('source_report'), after.get('source_sha256'))
    return dict(comparable=True, repeated_observation=same,
        task_evidence_delta=round(after['task_evidence_score']-before['task_evidence_score'], 2),
        scene_diagnostic_delta={k: after['scene_diagnostics'][k]-before['scene_diagnostics'][k]
                                for k in after['scene_diagnostics']},
        causal_action_effect_proven=False)


def action_information_opportunities(snapshot):
    candidates = {c['node_id']: c for c in snapshot['information_candidates']}
    rows = []
    for action in snapshot['allowed_actions']:
        row = dict(action_id=action['action_id'], expected_score_gain=None,
                   expected_gain_calibrated=False, requires_post_action_observation=False)
        if action['action'] in ('clear', 'plan_clear'):
            c = candidates[action['node_id']]
            row.update(opportunity='reveal_occluded_regions', reachable_branches=c['reveal_branch_count'],
                       directly_reveals=c['directly_reveals'], requires_post_action_observation=True,
                       physical_cost='grasp_transport_release', target_location_proven=False)
        elif action['action'] in ('observe', 'plan_observe'):
            row.update(opportunity='remeasure_current_view' if action['view_id'] == 'current' else 'change_view',
                       physical_cost='capture_only' if action['view_id'] == 'current' else 'view_motion_and_capture',
                       requires_post_action_observation=True)
        else:
            row.update(opportunity='none', physical_cost='none')
        rows.append(row)
    return rows


def _append(robot, record):
    path = Path(robot.config.nesyro_gate_trace_path).with_name('information_trace.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')


def record_information(robot, target, graph):
    state = assess_information(graph, target)
    previous = robot.information_states.get(target)
    delta = information_change(previous, state)
    # A cache publication can restore current validity without adding progress.
    robot.information_states[target] = state
    if delta.get('repeated_observation') and previous.get('valid'):
        return state
    robot.geometry_recovery_state['history'].append(dict(event='information_update', target=target,
        task_evidence_score=state['task_evidence_score'], change=delta,
        missing_evidence=state['missing_evidence'], source_report=state['source_report']))
    _append(robot, dict(event='observation', at=time.time(), information=state, change=delta,
                        preceding_invalidation=(previous or {}).get('invalidated_by')))
    print(f"[information] {target}: {state['task_evidence_score']:.1f}/100; "
          f"next={state['next_stage']}; delta={delta['task_evidence_delta']}")
    return state


def invalidate_information(robot, reason):
    if reason == 'new_task':
        robot.information_states.clear()
        return
    for target, state in robot.information_states.items():
        if state['valid']:
            state['valid'] = False
            state['invalidated_by'] = reason
            _append(robot, dict(event='invalidated', at=time.time(), target=target, reason=reason,
                                current_score=None, last_observed_score=state['task_evidence_score']))
