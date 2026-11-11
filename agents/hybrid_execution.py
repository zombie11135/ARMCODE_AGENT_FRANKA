"""Instruction-graph subtasks with bounded code repair and one-step prefetch.

Only the foreground invokes skills or updates task state. Background work is
code generation from task names/constraints, never perception acquisition or
robot trajectories. All attempts share the original hardware/robot session.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import json
import time

from agents.cap_code_generator import CapCodeGenerator, task_pairs
from agents.plan_graph import PlanGraph
from agents.replanning_agent import ReplanningAgent
from core.actions import SimpleAction
from runtime.cap_policy import PolicyError, PolicyInterpreter, PolicyStopped
from runtime.observation_memory import invalidate_camera, memory_for
from runtime.timing import timing_stage
from services.object_reference_service import atomic_json


def node_actions(pair):
    return [SimpleAction('grasp', pair[0]), SimpleAction('place', pair[1])]


def subtask_goal(pair):
    return f'当前子任务：将 {pair[0]} 放到 {pair[1]}。只完成这一对；其他任务由外层调度器执行。'


def policy_client(context):
    return (getattr(context, 'planner_llm', None) or getattr(context, 'fallback_vlm', None)
            or context.require('vlm'))


class HybridTaskRunner:
    def __init__(self, runtime, robot, context, camera_service, source_actions, resolved_actions, client):
        self.runtime, self.robot, self.context, self.camera = runtime, robot, context, camera_service
        self.config, self.state = runtime.config, runtime.task_state
        self.pairs, self.resolved = task_pairs(source_actions), task_pairs(resolved_actions)
        self.graph = PlanGraph.load_json(self.config.plan_json)
        self.generator = CapCodeGenerator(client)
        self.directory = Path(self.config.generated_code).parent / (Path(self.config.generated_code).stem + '_nodes')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace_path = Path(self.config.plan_json).with_name('hybrid_trace.json')
        self.events, self.completed = [], []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-prefetch')
        self.prefetch = None
        self.prefetch_jobs = []
        self.started = time.perf_counter()
        # Stable graph binding by ordered manipulation nodes, including repeated names.
        from agents.simple_code_generator import SimpleCodeGenerator
        manipulation = [n for n in SimpleCodeGenerator._topological_nodes(self.graph) if n.skill in ('grasp', 'place')]
        self.graph_ids = [[n.node_id for n in manipulation[2*i:2*i+2]] for i in range(len(self.pairs))]

    def event(self, event, **data):
        self.state.skill_budgets = dict(
            clear_counts=dict(getattr(self.robot, 'occlusion_recovery_counts', {})),
            buffer_slots_used=getattr(self.robot, 'geometry_buffer_slots_used', 0),
            geometry_observations=dict(getattr(self.robot, 'geometry_recovery_state', {}).get('observation_counts', {})))
        memory = memory_for(self.context.camera)
        if memory is not None:
            self.state.scene = dict(epoch=memory.epoch, scene_epoch=memory.scene_epoch)
        self.state.scene['task_information'] = dict(getattr(self.robot, 'information_states', {}))
        self.state.save()
        self.events.append(dict(event=event, elapsed_s=time.perf_counter()-self.started, **data))
        atomic_json(self.trace_path, dict(schema='hybrid_trace_v1', events=self.events,
                                         completed_actions=self.completed, revision=self.state.revision))

    def generate(self, index, revision, feedback=None):
        constraints = list(self.graph.constraints)
        for node_id in self.graph_ids[index]:
            constraints.extend(self.graph.get_node(node_id).constraints)
            constraints.extend(self.graph.get_node(node_id).preconditions)
        path = self.directory / f'node_{index:03d}_r{revision}.py'
        return self._generate(index, revision, path, constraints, feedback=feedback)

    def _generate(self, index, revision, path, constraints, *, feedback=None, job=None):
        started = time.perf_counter()
        if job is not None:
            job.update(start_s=started-self.started,
                       queue_wait_s=started-self.started-job['submitted_s'])
        try:
            with timing_stage('子任务代码生成', detail=f'node_{index}:r{revision}') as timing:
                timing.update(node=index, revision=revision, speculative=job is not None)
                return self.generator.generate(subtask_goal(self.pairs[index]),
                    node_actions(self.pairs[index]), path, feedback=feedback, constraints=constraints)
        except Exception as error:
            if job is not None:
                job['error_type'] = type(error).__name__
            raise
        finally:
            if job is not None:
                job.update(duration_s=time.perf_counter()-started,
                           finished_s=time.perf_counter()-self.started)

    def start_prefetch(self, index):
        if index >= len(self.pairs) or not self.config.enable_policy_pipeline:
            return
        revision = self.state.revision
        # Freeze the graph inputs before submitting, rather than reading mutable state in the worker.
        constraints = list(self.graph.constraints)
        for node_id in self.graph_ids[index]:
            constraints.extend(self.graph.get_node(node_id).constraints)
            constraints.extend(self.graph.get_node(node_id).preconditions)
        path = self.directory / f'node_{index:03d}_r{revision}.py'
        job = dict(node=index, revision=revision, submitted_s=time.perf_counter()-self.started,
                   outcome='pending')
        job['future'] = self.pool.submit(copy_context().run, self._generate,
            index, revision, path, constraints, job=job)
        self.prefetch_jobs.append(job)
        self.prefetch = job
        self.event('prefetch_started', node=index, revision=revision)

    def source(self, index):
        if index == 0 and self.config.cap_policy_source:
            return self.config.cap_policy_source
        if self.prefetch is not None and (self.prefetch['node'], self.prefetch['revision']) == (index, self.state.revision):
            job, self.prefetch = self.prefetch, None
            future = job['future']
            ready = future.done()
            waiting = time.perf_counter()
            with timing_stage('等待后台子任务代码') as timing:
                timing.update(node=index, revision=job['revision'], ready_before_wait=ready)
                try:
                    result = future.result()
                except PolicyError as error:
                    job.update(outcome='failed', join_wait_s=time.perf_counter()-waiting)
                    timing['status'] = 'failed'
                    self.event('prefetch_failed', node=index, revision=job['revision'],
                               join_wait_s=job['join_wait_s'], error=str(error))
                else:
                    job.update(outcome='used', join_wait_s=time.perf_counter()-waiting)
                    self.event('prefetch_used', node=index, revision=job['revision'],
                               ready_before_wait=ready, join_wait_s=job['join_wait_s'])
                    return result
        return self.generate(index, self.state.revision)

    def discard_prefetch(self, reason):
        if self.prefetch is None:
            return
        job, self.prefetch = self.prefetch, None
        job.update(outcome='invalidated' if reason == 'graph_revision' else 'unused_task_ended',
                   invalidated_s=time.perf_counter()-self.started, cancelled=job['future'].cancel())
        self.event('prefetch_invalidated', node=job['node'], revision=job['revision'],
                   reason=reason, cancelled=job['cancelled'])

    def execute_node(self, index, source):
        target, container = self.pairs[index]
        resolved_target, resolved_container = self.resolved[index]
        executed = False
        completed_before = len(self.completed)
        self.state.nodes[str(index)]['status'] = 'running'
        self.state.save()
        self.runtime.last_failure_report = None
        self.context.manipulation_progress = {}

        def observe(name):
            if name != target:
                raise PolicyError('Observation target is outside the active graph node')
            if not self.state.consume(f'node_{index}:observations', 2 + self.config.max_replan_attempts):
                raise PolicyStopped('Node observation budget exhausted')
            action = SimpleAction('observe', resolved_target)
            self.runtime._execution_checkpoint = dict(action=action, index=len(self.completed),
                                                     completed=list(self.completed), holding=None)
            result = self.robot.observe(resolved_target)
            self.event('observation', node=index, success=result.success, error=result.error)
            return bool(result.success)

        def pick_place(obj, destination):
            nonlocal executed
            if executed or [obj, destination] != [target, container]:
                raise PolicyError('Only the active graph pair may be executed once')
            executed = True
            actions = [SimpleAction(**row) for row in self.completed] + node_actions([resolved_target, resolved_container])
            if not self.runtime._run_actions(self.robot, self.context, self.camera, actions,
                    completed_actions=self.completed, start_index=len(self.completed)):
                raise PolicyStopped('Active manipulation did not complete')
            self.completed = [a.__dict__ for a in actions]
            return True

        try:
            PolicyInterpreter(source, dict(observe=observe, pick_place=pick_place)).run()
            if not executed:
                self.failure(index, 'observation_failed', 'Policy ended without completing its subtask')
                return False
            return True
        except PolicyError as error:
            if self.runtime.last_failure_report is None:
                kind = ('policy_failed_after_completion' if len(self.completed) > completed_before else
                        'observation_budget_exhausted' if isinstance(error, PolicyStopped) else 'policy_failed')
                self.failure(index, kind, str(error))
            return False

    def failure(self, index, kind, error):
        return self.runtime._write_failure_report(SimpleAction('policy', f'node_{index}'), self.resolved[index][0],
            kind, error, completed_actions=self.completed, evidence=dict(automatic_replan=False))

    def repair(self, index):
        report = self.runtime.last_failure_report
        route = report['recovery']['route']
        if route == 'stop' or not self.config.enable_replan:
            return None
        if not self.state.consume(f'node_{index}:repairs', self.config.max_replan_attempts):
            self.event('repair_budget_exhausted', node=index)
            return None
        self.discard_prefetch('graph_revision')
        report = dict(report, failed_node_id=self.graph_ids[index][0])
        with timing_stage('Hybrid节点修复', detail=str(index)):
            self.graph = ReplanningAgent().replan_from_failure_report(self.graph, report)
        self.state.revision += 1
        self.runtime.replan_attempts += 1
        self.graph.metadata['hybrid_revision'] = self.state.revision
        self.graph.save_json(self.config.replan_json)
        self.runtime.replan_paths = dict(replan_json=self.config.replan_json)
        if route in ('refresh', 'graph_repair'):
            invalidate_camera(self.context.camera, 'hybrid_node_repair', discard_scene=False)
            self.robot.observations.pop(self.resolved[index][0], None)
        self.state.save()
        self.event('node_repair', node=index, route=route, failure_type=report['failure_type'])
        return self.generate(index, self.state.revision, feedback=report)

    def run(self):
        try:
            for index in range(len(self.pairs)):
                self.state.nodes[str(index)] = dict(status='running', graph_node_ids=self.graph_ids[index],
                                                     pair=self.resolved[index])
                self.state.save()
                source = self.source(index)
                while True:
                    # A repaired graph invalidates the previous speculation.
                    # Overlap the next revision with EACH current-node attempt.
                    self.start_prefetch(index + 1)
                    if self.execute_node(index, source):
                        break
                    pending = self.runtime.last_failure_report['evidence'].get('recovery_pending', False)
                    self.state.nodes[str(index)]['status'] = 'awaiting_information' if pending else 'failed'
                    if pending:
                        self.event('information_pending', node=index)
                    self.state.save()
                    source = self.repair(index)
                    if source is None:
                        return False
                self.state.nodes[str(index)]['status'] = 'completed'
                for node_id in self.graph_ids[index]:
                    self.graph.get_node(node_id).status = 'completed'
                self.graph.save_json(Path(self.config.plan_json).with_name('hybrid_graph.json'))
                self.state.save()
                self.event('node_completed', node=index)
            return True
        except PolicyError as error:
            self.failure(index, 'policy_failed', str(error))
            return False
        finally:
            self.discard_prefetch('task_ended')
            # A running LLM call has its configured timeout; it never touches devices.
            with timing_stage('等待后台代码线程退出'):
                self.pool.shutdown(wait=True, cancel_futures=True)
            jobs = [{k: v for k, v in job.items() if k != 'future'} for job in self.prefetch_jobs]
            for job in jobs:
                job['wasted_generation_s'] = job.get('duration_s', 0.) if job['outcome'] != 'used' else 0.
                job['work_after_invalidation_s'] = (max(0., job['finished_s'] - max(
                    job['start_s'], job['invalidated_s'])) if 'invalidated_s' in job and 'finished_s' in job else 0.)
            self.event('session_finished', prefetch_jobs=jobs)
