"""Execute one validated recovery choice inside the existing robot task session."""
from copy import deepcopy
import math
from pathlib import Path

from agents.occlusion_recovery_planner import build_recovery_snapshot, validate_proposal
from core.skill_base import SkillResult
from runtime.observation_memory import fresh_camera
from services.object_reference_service import atomic_json
from services.wrist_occlusion_service import choose_blocker, verify_binding, sha


def save_recovery_state(robot):
    path = Path(robot.config.nesyro_gate_trace_path).with_name('occlusion_recovery_state.json')
    atomic_json(path, dict(schema=1, scope='audit_of_current_task_not_automatic_restart_restore',
        clear_counts=robot.occlusion_recovery_counts, buffer_slots_used=robot.geometry_buffer_slots_used,
        configured_buffer_poses_mmrad=robot.config.geometry_occlusion_buffer_poses_mmrad,
        state=robot.geometry_recovery_state))


def record_clear(robot, target, *, graph, blocker, slot, pose, stage, success=None):
    robot.geometry_recovery_state['history'].append(dict(event='clear', stage=stage,
        target=target, source_report=graph['report_path'], node_id=blocker['node_id'],
        bbox=blocker['bbox_xyxy'], buffer_slot=slot, buffer_pose_mmrad=pose,
        movement_completed=success, visual_effect_verified=False))
    save_recovery_state(robot)


def recover_once(robot, target, graph, decision):
    """Return ('refreshed', None), ('clear', decision), or ('stop', SkillResult)."""
    from .geometry_occlusion import _failure, observation_current
    cfg = robot.config
    state = robot.geometry_recovery_state
    budget = cfg.geometry_occlusion_observation_budget
    if type(budget) is not int or not 0 <= budget <= 10:
        raise ValueError('Recovery observation budget must be 0..10')
    poses = cfg.geometry_occlusion_recovery_view_poses_mmrad
    if any(len(p) != 6 or not all(math.isfinite(float(v)) for v in p) for p in poses):
        raise ValueError('Recovery views require finite flange XYZ/RPY in mm/rad')
    views = {f'view_{i+1}': list(p) for i, p in enumerate(poses)} if cfg.execute else {}
    snapshot = build_recovery_snapshot(graph=graph, decision=decision, target=target,
        goal=getattr(robot.context, 'task_goal', target),
        clear_count=robot.occlusion_recovery_counts.get(target, 0),
        clear_limit=cfg.occlusion_recovery_max_attempts,
        buffers=cfg.geometry_occlusion_buffer_poses_mmrad, slots_used=robot.geometry_buffer_slots_used,
        observation_count=state['observation_counts'].get(target, 0), observation_limit=budget,
        history=state['history'], views=views, allow_clear=cfg.enable_occlusion_recovery,
        max_image_fraction=cfg.occlusion_blocker_max_image_area_ratio)
    actions = snapshot['allowed_actions']
    print('[occlusion recovery] geometry='+decision['status'],
          'clear_candidates='+str(sum(a['action'] == 'clear' for a in actions)),
          'clear_remaining='+str(snapshot['clear_remaining']),
          'free_buffers='+str(sum(not b['occupied'] for b in snapshot['buffer_slots'])))
    print('[occlusion recovery] depth coverage filter: disabled (coverage is diagnostic only)')
    if snapshot['clear_unavailable_reasons']:
        print('[occlusion recovery] clear unavailable:', ', '.join(snapshot['clear_unavailable_reasons']))
        sources = {p['source'] for p in snapshot['possible_pairs']}
        for node in snapshot['nodes']:
            is_target = node['node_id'] == snapshot['target_association']['node_id']
            if ((is_target and decision['status'] == 'target_quality_insufficient')
                    or (node['node_id'] in sources and not node['geometry_eligible'])):
                print('[occlusion recovery] rejected '+('target' if is_target else 'blocker'), node['display_id'],
                      'depth='+format(node['depth_coverage'], '.1%'),
                      'issues='+','.join(node['quality_issues']))
    if snapshot['current_view_retry_used']:
        print('[occlusion recovery] same-view retry already used; choose a legal clear, configured view or pause')
    print('[occlusion recovery] available:', ', '.join(a['action_id'] for a in actions))
    planner = robot.context.require('occlusion_recovery_planner')
    try:
        proposal = planner.decide(deepcopy(snapshot))
        # Validate again at the execution boundary, even for alternate/test clients.
        selected = validate_proposal({k: proposal.get(k) for k in ('snapshot_id', 'action_id', 'reason')}, snapshot)
    except Exception as error:
        return 'stop', _failure('geometry_occlusion_recovery_planner_failed',
            '恢复规划未生成可执行决策，未执行额外动作。', recovery_pending=True,
            error_type=type(error).__name__, recovery_snapshot=snapshot)
    state['history'].append(dict(event='llm_choice', target=target, **selected))
    save_recovery_state(robot)
    print('[occlusion recovery]', selected['action_id'], selected['reason'])
    if selected['action'] == 'pause':
        reason = selected['reason']
        if 'geometry:blocker_quality_insufficient' in snapshot['clear_unavailable_reasons']:
            reason = ('遮挡候选仍有掩码或几何质量问题；清障尚未执行，详见候选诊断。模型说明：'+reason)
        return 'stop', _failure('geometry_occlusion_recovery_paused', reason,
            recovery_pending=True, recovery_snapshot=snapshot, recovery_state=deepcopy(state))
    # A model response never authorizes using evidence changed during the request.
    if sha(graph['report_path']) != graph['report_sha256']:
        return 'stop', _failure('geometry_occlusion_report_changed', 'LLM 推理期间关系图文件改变。')
    verify_binding(graph['source_binding'])
    if selected['action'] == 'clear':
        if not observation_current(robot, robot.observations.get(target)):
            return 'stop', _failure('geometry_occlusion_recovery_stale', 'LLM 返回时观测已过期，需要新观测。',
                                   recovery_pending=True)
        return 'clear', choose_blocker(graph, allow_search=cfg.occlusion_allow_unlocalized_target,
            max_image_fraction=cfg.occlusion_blocker_max_image_area_ratio,
            preferred_node_id=selected['node_id'])
    # Reserve before capture/motion; a failed attempt also consumes this budget.
    state['observation_counts'][target] = state['observation_counts'].get(target, 0)+1
    save_recovery_state(robot)
    view = selected['view_id']
    if view != 'current' and not robot._move_xarm(views[view]):
        return 'stop', _failure('geometry_occlusion_recovery_view_failed', '移动到恢复观测位失败。')
    with fresh_camera(robot.context.camera, 'llm_occlusion_recovery'):
        result = robot._observe_medicine_once(target)
    state['history'].append(dict(event='recovery_observation', target=target,
        view_id=view, observation_success=result.success,
        target_identified=result.data.get('target_identified', False),
        report=result.data.get('geometry_occlusion', {}).get('report_path')))
    save_recovery_state(robot)
    if not result.success:
        return 'stop', _failure('geometry_occlusion_recovery_observation_failed',
                               '恢复观测未获取有效场景。', observation_error=result.error)
    return 'refreshed', None
