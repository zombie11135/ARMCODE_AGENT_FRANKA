"""Saved first-frame analysis overlapping foreground container observation.

Only the owner acquires images, moves devices, and publishes robot state. The
single background job can warm model candidates, but cannot select an action,
reserve a buffer, or execute a grasp.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import time

from core.context import RuntimeContext
from services.object_reference_service import atomic_json


class TaskPerceptionPipeline:
    def __init__(self, robot, target, views, trace_path):
        self.robot, self.target, self.views = robot, target, views
        self.trace_path = Path(trace_path)
        self.events = []
        live = robot.context
        # Do not share any camera, arm, gripper or mutable task state with work.
        self.context = RuntimeContext(**{name: getattr(live, name, None) for name in (
            'vlm', 'detector', 'segmenter', 'ocr', 'anygrasp', 'transforms',
            'object_references', 'perception_workers', 'occlusion')}, event_callback=self.events.append)
        self.config = deepcopy(robot.config)
        self.started = time.perf_counter()
        self.record = dict(target=target, status='running',
            source_capture=views[0]['color_path'], acquired_at=views[0]['timestamp'],
            provenance=deepcopy(views[0].get('observation_cache')), devices_used_by_worker=False)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='task-perception')
        self.future = self.pool.submit(self._work)
        self.consumed = False

    def _work(self):
        from skill_lib.actions.geometry_occlusion import analyze_geometry_views
        started = time.perf_counter()
        result = analyze_geometry_views(self.context, self.config, self.target, self.views)
        self.record['analysis_s'] = time.perf_counter()-started
        if result.success and self.context.transforms is not None:
            started = time.perf_counter()
            try:
                self._prefetch_candidates(result.data)
            except Exception as error:
                # Speculation failing does not fail the physical task; the
                # normal planner will report any actual planning failure.
                self.record['candidate_prefetch_error'] = type(error).__name__
            self.record['candidate_prefetch_s'] = time.perf_counter()-started
        self.record['background_total_s'] = time.perf_counter()-self.started
        return result

    def _prefetch_candidates(self, data):
        from services.wrist_occlusion_service import choose_blocker
        from skill_lib.manipulation.clear_occlusion_to_buffer import _shrink_bbox
        from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
        cfg, graph = self.config, data['geometry_occlusion']
        decision = choose_blocker(graph, allow_search=cfg.occlusion_allow_unlocalized_target,
                                  max_image_fraction=cfg.occlusion_blocker_max_image_area_ratio)
        node = decision.get('blocker')
        if decision['status'] == 'no_occlusion_evidence':
            node = next(n for n in graph['primary']['relation_graph']['nodes']
                        if n['node_id'] == graph['target_association']['node_id'])
            bbox = node['bbox_xyxy']
        elif node is not None and cfg.enable_occlusion_recovery and cfg.occlusion_recovery_max_attempts > 0 and cfg.geometry_occlusion_buffer_poses_mmrad:
            bbox = _shrink_bbox(node['bbox_xyxy'], cfg.occlusion_blocker_bbox_shrink_ratio)
        else:
            self.record['candidate_prefetch'] = 'no_candidate_for_current_action'
            return
        capture = data['geometry_capture']
        provenance = capture.get('observation_cache') or {}
        result = self.context.require('anygrasp').plan_grasp(
            color_path=capture['color_path'], points_path=capture['points_path'],
            capture_geometry=capture['capture_geometry'], bbox=bbox,
            mask_npy_path=node['visible_mask_path'], mask_path=None,
            intrinsics=PlanGraspWithAnyGraspSkill._normalize_intrinsics(
                capture.get('depth_intrinsics') or capture.get('color_intrinsics')),
            bbox_filter_margin_px=cfg.bbox_filter_margin_px,
            bbox_filter_inner_margin_ratio=cfg.bbox_filter_inner_margin_ratio,
            retain_all_candidates=True, apply_object_mask=True,
            collision_detection=cfg.collision_detection,
            scene_token=[provenance.get('scope'), provenance.get('scene_epoch')],
            output_json_path=str(Path(graph['report_path']).with_name('prefetched_candidates.json')))
        self.record.update(candidate_prefetch='ready', candidate_node_id=node['node_id'],
                           candidate_run_dir=result.get('run_dir'))

    def publish(self):
        """Join after returning home; atomically publish only current evidence."""
        from skill_lib.actions.geometry_occlusion import observation_current
        from skill_lib.actions.observation_cache import _key, _observation, _publish
        from runtime.observation_memory import memory_for
        self.record['elapsed_before_join_s'] = time.perf_counter()-self.started
        waiting = time.perf_counter()
        try:
            result = self.future.result()
            self.record['join_wait_s'] = time.perf_counter()-waiting
            self.consumed = True
            for event in self.events:
                self.robot.context.emit(event)
            if not result.success:
                self.record.update(status='analysis_failed', error=result.error)
                print('[perception_pipeline] analysis failed; use normal foreground observation')
                return False
            observation = _observation('medicine', self.target, result.data)
            graph = result.data['geometry_occlusion']
            binding = (graph['report_path'], graph['report_sha256'])
            self.robot._pipeline_frame_bindings.add(binding)
            self.record.update(validity='current_task_and_observation_epoch',
                               capture_age_s=time.time()-self.views[0]['timestamp'])
            if not observation_current(self.robot, observation):
                self.robot._pipeline_frame_bindings.discard(binding)
                self.record['status'] = 'discarded_observation_not_current'
                print('[perception_pipeline] frame invalidated or away from source pose; use new observation')
                return False
            capture = result.data['geometry_capture']
            provenance = capture['observation_cache']
            memory = memory_for(self.robot.context.camera)
            def publish():
                memory.put(_key(self.robot, 'medicine', self.target),
                    dict(data=result.data, observation=observation, epoch=provenance['epoch']),
                    capture['capture_geometry']['before'], capture['timestamp'],
                    provenance['scope'], provenance['epoch'])
                _publish(self.robot, self.target, observation, result.data)
            accepted = memory.commit(provenance['epoch'], publish)
            if not accepted:
                self.robot._pipeline_frame_bindings.discard(binding)
            self.record['status'] = 'published' if accepted else 'discarded_scene_changed'
            print('[perception_pipeline]', self.record['status'], self.target,
                  'join_wait_s='+format(self.record['join_wait_s'], '.3f'))
            return accepted
        finally:
            atomic_json(self.trace_path, self.record)

    def close(self):
        self.future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if not self.consumed:
            self.record['status'] = 'discarded_task_ended'
            atomic_json(self.trace_path, self.record)


def start_task_perception_pipeline(robot, actions, captures, config):
    from runtime.observation_memory import memory_for
    from skill_lib.actions.geometry_occlusion import freeze_geometry_views
    target = next((a.target for a in actions if a.op == 'grasp'), None)
    if (not getattr(config, 'enable_perception_pipeline', True)
            or not robot.config.enable_geometry_occlusion or target is None
            or memory_for(robot.context.camera) is None or captures is None):
        return None
    try:
        views = freeze_geometry_views(robot.context, captures['wrist'], captures.get('global'))
        trace = Path(config.nesyro_gate_trace_path).with_name('perception_pipeline.json')
        pipeline = TaskPerceptionPipeline(robot, target, views, trace)
        print('[perception_pipeline] first-frame analysis started for', target)
        return pipeline
    except Exception as error:
        print('[perception_pipeline] startup unavailable:', type(error).__name__)
        return None
