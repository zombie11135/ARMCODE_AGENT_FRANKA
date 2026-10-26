"""Bind target identity to wrist evidence, clear one exposed ancestor, observe anew."""
import json
from pathlib import Path
import time

from core.skill_base import SkillResult
from runtime.observation_memory import memory_for, fresh_camera, pose_distance, invalidate_camera
from services.wrist_occlusion_service import choose_blocker, verify_binding, sha, associate_target


def _failure(kind, message, **evidence):
    return SkillResult(False, dict(failure_type=kind, automatic_replan=False, **evidence), error=message)


def freeze_geometry_views(context, captured=None, auxiliary_capture=None):
    """Acquire/freeze on the device owner; background analysis needs no camera."""
    service, camera = context.require('occlusion'), context.require('camera')
    capture, root = service.freeze(captured if captured is not None else camera.capture_wrist())
    auxiliary, auxiliary_error = None, None
    if getattr(camera, 'active_camera_names', ()) == ('global', 'wrist'):
        try:
            aux = auxiliary_capture if auxiliary_capture is not None else camera.capture_global()
            if aux.get('camera_name') == 'wrist' or aux.get('view_fallback'):
                raise ValueError('Wrist fallback cannot be labeled as fixed auxiliary')
            auxiliary = service.freeze(aux, 'global')
        except Exception as error:
            auxiliary_error = str(error)
    return capture, root, auxiliary, auxiliary_error


def analyze_geometry_views(context, cfg, target, views):
    """Pure saved-frame inference; never acquire images or query/move the arm."""
    from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
    try:
        capture, root, auxiliary, auxiliary_error = views
        service = context.require('occlusion')
        finder = FindAndGraspMedicineSkill(context)
        identity = finder.run(query=target, prepared_capture=capture, camera_name='wrist',
            plan_grasp=False, execute=False, min_score=cfg.grasp_min_score,
            detection_threshold=cfg.grasp_box_threshold, text_threshold=cfg.grasp_text_threshold,
            verify_ocr=cfg.verify_ocr, ocr_verify_strict_mode=cfg.ocr_verify_strict_mode,
            ocr_verify_fuzzy_threshold=cfg.ocr_verify_fuzzy_threshold)
        selected = identity.data.get('selected_instance') if identity.success else None
        graph = service.analyze(capture, root, selected)
        aux_report = None
        if auxiliary is not None:
            try:
                aux_report = service.analyze(*auxiliary)
            except Exception as error:
                auxiliary_error = str(error)
        data = dict(identity.data if identity.success else {}, target_identified=bool(identity.success),
                    target_query=target, geometry_capture=capture, geometry_occlusion=graph,
                    geometry_auxiliary=aux_report, geometry_auxiliary_error=auxiliary_error,
                    recognition_error=None if identity.success else identity.error)
        if not identity.success:
            data.update(selected_instance=None, image_path=capture['color_path'],
                        grasp_input=dict(capture, bbox=None, query=target,
                                         observation_provenance=capture.get('observation_cache')))
        # A usable scene observation can succeed even when target identity is
        # unavailable. This is not a successful target detection/grasp permission.
        return SkillResult(True, data, message='腕部关系证据已更新；目标身份独立记录。')
    except Exception as error:
        return _failure('geometry_occlusion_observation_failed', str(error))


def observe_geometry(robot, target):
    try:
        views = freeze_geometry_views(robot.context)
        return analyze_geometry_views(robot.context, robot.config, target, views)
    except Exception as error:
        return _failure('geometry_occlusion_observation_failed', str(error))


def observation_current(robot, observation):
    if not isinstance(observation, dict) or not observation.get('result', {}).get('geometry_occlusion'):
        return False
    capture = observation['result']['geometry_capture']
    memory = memory_for(robot.context.camera)
    if memory is None:
        return False  # Uncalibrated callers must acquire again; no indefinite cache.
    provenance = capture.get('observation_cache') or {}
    if not memory.commit(provenance.get('epoch'), lambda: None):
        return False
    if provenance.get('scope') != robot.context.camera.observation_scope:
        return False
    age = time.time()-capture['timestamp']
    graph = observation['result']['geometry_occlusion']
    task_snapshot = (graph['report_path'], graph['report_sha256']) in robot._pipeline_frame_bindings
    # Startup work can outlast the image-cache TTL without changing the table.
    # This owner-registered frame lasts within this task/epoch only; its actual
    # timestamp, calibrated source pose and all file bindings remain unchanged.
    if not 0 <= age or (not task_snapshot and age > memory.policy.max_age_s):
        return False
    source_pose = capture.get('capture_geometry', {}).get('before')
    if source_pose is None:
        return False
    distance, angle = pose_distance(source_pose, robot.context.camera.observation_pose())
    return distance <= memory.policy.position_radius_m and angle <= memory.policy.orientation_radius_rad


def validate_prepared(context, prepared, query):
    """Recheck identity, source files and epoch at the grasp-planning boundary."""
    if prepared.get('target_query') != query or not prepared.get('target_identified'):
        raise ValueError('Prepared grasp must identify this requested target')
    graph = prepared['geometry_occlusion']
    if sha(graph['report_path']) != graph['report_sha256']:
        raise ValueError('Occlusion report changed after analysis')
    stored = json.loads(Path(graph['report_path']).read_text())
    if any(stored[key] != graph[key] for key in ('primary', 'target_association', 'source_binding',
                                                'observation_provenance', 'capture_geometry')):
        raise ValueError('Prepared graph differs from the saved occlusion evidence')
    if choose_blocker(graph)['status'] != 'no_occlusion_evidence':
        raise ValueError('Prepared target is blocked, ambiguous, or has insufficient geometry')
    capture = prepared['geometry_capture']
    verify_binding(graph['source_binding'])
    for key, binding in graph['source_binding'].items():
        if str(Path(capture[key]).resolve()) != binding['path']:
            raise ValueError('Prepared capture differs from occlusion analysis')
    if capture.get('observation_cache') != graph.get('observation_provenance'):
        raise ValueError('Prepared capture epoch differs from occlusion analysis')
    if capture.get('capture_geometry') != graph.get('capture_geometry'):
        raise ValueError('Prepared capture-time flange geometry changed')
    memory = memory_for(context.camera)
    if memory is not None and not memory.commit((capture.get('observation_cache') or {}).get('epoch'), lambda: None):
        raise ValueError('Scene changed after occlusion analysis')
    node_id = graph['target_association']['node_id']
    association = associate_target(graph['primary']['relation_graph']['nodes'], prepared.get('selected_instance'))
    if association['node_id'] != node_id:
        raise ValueError('Target selection changed after graph association')
    node = next(n for n in graph['primary']['relation_graph']['nodes'] if n['node_id'] == node_id)
    if sha(node['visible_mask_path']) != node['visible_mask_sha256']:
        raise ValueError('Prepared target visible mask changed')
    return node


def _trace(robot, event, **data):
    record = dict(event=event, timestamp=time.time(), **data)
    robot.geometry_occlusion_events.append(record)
    path = Path(robot.config.nesyro_gate_trace_path).with_name('occlusion_graph_trace.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))+'\n')


def prepare_grasp(robot, target):
    try:
        return _prepare_grasp(robot, target)
    except Exception as error:
        return _failure('geometry_occlusion_error', str(error), error_type=type(error).__name__)


def _prepare_grasp(robot, target):
    cfg = robot.config
    camera = robot.context.camera
    # Clear attempts and LLM observation attempts have independent finite budgets.
    # Keep one robot session so neither counters nor occupied slots reset on recovery.
    while True:
        observation = robot.observations.get(target)
        if not observation_current(robot, observation):
            result = robot._observe_medicine_once(target)
            if not result.success:
                return result
            observation = robot.observations.get(target)
        if not observation or not observation.get('result', {}).get('geometry_occlusion'):
            return _failure('geometry_occlusion_missing', '当前目标缺少腕部关系图。')
        data = observation['result']
        graph = data['geometry_occlusion']
        if sha(graph['report_path']) != graph['report_sha256']:
            return _failure('geometry_occlusion_report_changed', '关系图文件已改变，需要重新观测。')
        verify_binding(graph['source_binding'])
        decision = choose_blocker(graph, allow_search=cfg.occlusion_allow_unlocalized_target,
            max_image_fraction=cfg.occlusion_blocker_max_image_area_ratio)
        _trace(robot, 'decision', target=target, status=decision['status'],
               report=graph['report_path'], provenance=graph['observation_provenance'],
               blocker_id=(decision.get('blocker') or {}).get('node_id'))
        if decision['status'] == 'no_occlusion_evidence':
            validate_prepared(robot.context, data, target)
            return SkillResult(True, dict(geometry_occlusion=decision, graph_report=graph['report_path']))
        if cfg.enable_llm_occlusion_recovery:
            from .llm_occlusion_recovery import recover_once
            kind, recovered = recover_once(robot, target, graph, decision)
            if kind == 'refreshed':
                continue
            if kind == 'stop':
                return recovered
            decision = recovered
        blocker = decision.get('blocker')
        if blocker is None:
            return _failure('geometry_occlusion_unresolved', '目标身份、几何或遮挡关系尚不能支持抓取。',
                            geometry_decision=decision, graph_report=graph['report_path'])
        if not cfg.enable_occlusion_recovery:
            return _failure('geometry_occlusion_clear_required', '目标存在潜在遮挡，尚未启用清障恢复。',
                            geometry_decision=decision)
        count = robot.occlusion_recovery_counts.get(target, 0)
        if count >= max(0, int(cfg.occlusion_recovery_max_attempts)):
            return _failure('geometry_occlusion_budget_exhausted', '已达到本任务目标的清障次数上限。')
        slots = cfg.geometry_occlusion_buffer_poses_mmrad
        import math
        if (any(len(slot) != 6 or not all(math.isfinite(float(x)) for x in slot) for slot in slots)
                or len({tuple(slot) for slot in slots}) != len(slots)):
            return _failure('geometry_occlusion_buffer_invalid', '缓冲槽位必须为不同的六维有限位姿。')
        index = robot.geometry_buffer_slots_used
        if index >= len(slots):
            return _failure('geometry_occlusion_buffer_missing', '请配置足够的独立空闲缓冲位姿；不会复用已占用槽位。')
        evidence = dict(node_id=blocker['node_id'], target_node_id=decision.get('target_node_id'),
            exploratory=decision['exploratory'], report_path=graph['report_path'], report_sha256=sha(graph['report_path']),
            buffer_pose_mmrad=slots[index], max_image_fraction=cfg.occlusion_blocker_max_image_area_ratio)
        original_pose = data['geometry_capture'].get('capture_geometry', {}).get('before')
        if cfg.execute and original_pose is None:
            return _failure('geometry_occlusion_capture_pose_missing', '清障后返回观测位置需要拍照时的法兰位姿。')
        observation['geometry_blocker'] = evidence
        # Reserve before execution. A failed/partial move does not free a slot.
        if cfg.execute:
            robot.geometry_buffer_slots_used += 1
            robot.occlusion_recovery_counts[target] = count+1
            if cfg.enable_llm_occlusion_recovery:
                from .llm_occlusion_recovery import record_clear
                record_clear(robot, target, graph=graph, blocker=blocker, slot=index,
                             pose=slots[index], stage='reserved')
        result = robot._invoke_clear_occlusion_skill(target)
        if cfg.execute and cfg.enable_llm_occlusion_recovery:
            record_clear(robot, target, graph=graph, blocker=blocker, slot=index,
                         pose=slots[index], stage='finished', success=result.success)
        _trace(robot, 'clear_result', target=target, success=result.success, error=result.error,
               buffer_slot=index, exploratory=decision['exploratory'], result=result.data)
        if not result.success:
            return _failure('geometry_occlusion_clear_failed', '清障失败，停止目标抓取并保留现场状态。',
                            clear_result=result.data, clear_error=result.error,
                            holding_state='unknown_if_execution_started')
        if not cfg.execute:
            return _failure('geometry_occlusion_dry_run_clear_plan',
                            '已规划遮挡搬运；dry-run 未改变场景，不继续假设目标已暴露。', clear_plan=result.data)
        # Mutation hooks have already invalidated graph and target evidence.
        # Also invalidate on non-calibrated adapters, then return to the actual
        # acquisition pose rather than a guessed home pose.
        invalidate_camera(camera, 'geometry_clear_reobserve', discard_scene=True)
        robot._discard_scene_evidence('geometry_clear_reobserve')
        pose = [v*1000 for v in original_pose[:3]]+list(original_pose[3:])
        if not robot._move_xarm(pose):
            return _failure('geometry_occlusion_return_view_failed', '清障后返回原观测位姿失败。')
        if cfg.observe_wait_s > 0:
            time.sleep(cfg.observe_wait_s)
        with fresh_camera(camera, 'geometry_clear_reobserve'):
            refreshed = robot._observe_medicine_once(target)
        _trace(robot, 'reobserve', target=target, success=refreshed.success,
               graph_report=refreshed.data.get('geometry_occlusion', {}).get('report_path'))
        if not refreshed.success:
            return _failure('geometry_occlusion_refresh_failed', '清障后观测失败，禁止使用旧目标坐标。',
                            refresh_error=refreshed.error)
