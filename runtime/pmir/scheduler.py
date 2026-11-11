"""Dependency/resource scheduler; serial and visibility-blind controls share safety checks."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
import json
import threading
import time

from runtime.timing import timing_stage


class ForegroundCosts:
    """Exclusive owner-thread wall intervals, including nested adapter calls.

    Worker time is a separate overlay, never added to these totals. Scopes must
    finish before a cooperative generator yields. This is instrumentation only.
    """
    def __init__(self, origin):
        self.origin = self.previous = origin
        self.category = 'runtime_bookkeeping'
        self.intervals = []
        self.closed = False

    def switch(self, category):
        now = time.perf_counter()
        self.intervals.append((self.previous-self.origin, now-self.origin, self.category))
        self.previous, self.category = now, category

    @contextmanager
    def scope(self, category):
        if self.closed:
            yield
            return
        previous = self.category
        self.switch(category)
        try:
            yield
        finally:
            self.switch(previous)

    def finish(self):
        self.switch('finished')
        self.closed = True
        totals = defaultdict(float)
        for start, end, category in self.intervals:
            totals[category] += end-start
        return dict(totals), self.previous-self.origin


def run_compute(node, job, origin, attempt=1):
    # No reference to the scheduler, simulator, or live adapter crosses threads.
    started, cpu = time.perf_counter(), time.thread_time()
    with timing_stage('PMIR计算', detail=f'{node.id}:{node.kind}:attempt{attempt}'):
        result = job()
    components = result.pop('_timing', {})
    for key in ('injected_wait',):
        if key in components:
            components[key] = [v-origin for v in components[key]]
    return result, dict(node=node.id, kind=node.kind, phase=node.phase, target=node.target,
                       physical=False, start_s=started-origin, attempt=attempt,
                       duration_s=time.perf_counter()-started, cpu_s=time.thread_time()-cpu,
                       components=components, thread=threading.get_ident())


class Scheduler:
    def __init__(self, graph, adapter, directory, *, mode='dynamic'):
        if mode not in ('serial', 'naive', 'dynamic'):
            raise ValueError('Unknown scheduler mode')
        graph.validate()
        self.graph, self.adapter, self.directory, self.mode = graph, adapter, directory, mode
        self.nodes = {n.id: n for n in graph.nodes}
        self.state = {n.id: 'pending' for n in graph.nodes}
        self.values, self.resources, self.futures, self.active = {}, {}, {}, {}
        self.events, self.spans, self.wait_windows = [], [], []
        self.attempts = defaultdict(int)
        self.reacquire, self.releases = set(), {}
        self.started = time.perf_counter()
        self.costs = ForegroundCosts(self.started)
        adapter.cost_scope = self.costs.scope
        self.arm_idle_compute_s = 0.
        self.log = (directory/'events.jsonl').open('x', encoding='utf-8')
        adapter.emit = self.emit

    def emit(self, event, **details):
        with self.costs.scope('logging'):
            row = dict(sequence=len(self.events), event=event, wall_s=time.perf_counter()-self.started,
                       sim_time_s=self.adapter.backend.sim_time_s, step=self.adapter.backend.step_count, **details)
            self.events.append(row)
            self.log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            self.log.flush()
        if event == 'event_released':
            self.releases[details['gate']] = details['producer']
            if details['producer'] != 'initial':
                with self.costs.scope('scheduler_decision'):
                    for node in self.graph.nodes:
                        if node.gate == details['gate'] and self.state[node.id] == 'pending':
                            self.graph.link(details['producer'], node.id, 'event',
                                'RGB-D visibility and simulator stability confirmed at phase boundary')

    def available(self, node):
        if any(self.state[p] not in ('done', 'skipped') for p in self.graph.parents(node.id)):
            return False
        if any(resource in self.resources for resource in node.resources):
            return False
        if self.mode == 'naive' and node.kind == 'CAPTURE' and node.id not in self.reacquire:
            return True  # One early computation attempt; never speculative physical action.
        return not node.gate or self.adapter.visible.get(node.target, False)

    def condition(self, node):
        return all(bool(self.values.get(parent, {}).get('success', False)) == expected
                   for parent, expected in node.conditions)

    def start(self, node):
        self.attempts[node.id] += 1
        self.state[node.id] = 'running'
        self.resources.update({resource: node.id for resource in node.resources})
        self.emit('start', node=node.id, kind=node.kind, phase=node.phase, target=node.target,
                  resources=node.resources, attempt=self.attempts[node.id])

    def commit(self, node, result):
        with self.costs.scope('state_validation'):
            valid = self.adapter.valid(node, result)
        if not valid:
            self.emit('invalidated', node=node.id, reason='physical_state_version_changed')
            raise RuntimeError(f'Stale result cannot commit: {node.id}')
        self.values[node.id] = result
        self.state[node.id] = 'done'
        for resource in node.resources:
            self.resources.pop(resource, None)
        self.emit('commit', node=node.id, kind=node.kind, phase=node.phase, target=node.target,
                  result_success=result.get('success', True), attempt=self.attempts[node.id])
        with self.costs.scope('state_validation'):
            self.adapter.after_node(node)

    def retry_hidden_preparation(self, node, result):
        if self.mode != 'naive' or node.kind != 'PERCEIVE' or result.get('success', True):
            return False
        capture = self.nodes[node.inputs[0]] if node.inputs else None
        if capture is None or capture.kind != 'CAPTURE' or not capture.gate:
            return False  # Explicit observe() must preserve false-branch semantics.
        if capture.id in self.reacquire:
            raise RuntimeError(f'Visibility-confirmed reacquisition failed: {node.target}')
        self.reacquire.add(capture.id)
        for key in (node.id, capture.id):
            self.state[key] = 'pending'
            self.values.pop(key, None)
        for resource in node.resources:
            self.resources.pop(resource, None)
        producer = self.releases.get(capture.gate)
        if producer and producer != 'initial':
            self.graph.link(producer, capture.id, 'event', 'Bounded reacquisition after confirmed visibility')
        self.emit('invalidated', node=node.id, target=node.target, attempt=self.attempts[node.id],
                  reason='empty_early_observation', retry='once_after_confirmed_visibility')
        return True

    def blocking_jobs(self):
        """Observed waiting chain to the next physical operation, not a counterfactual speedup."""
        next_node = next((n for n in self.graph.nodes if n.physical and self.state[n.id] == 'pending'), None)
        if next_node is None:
            return []
        ancestors, todo = set(), list(self.graph.parents(next_node.id))
        while todo:
            parent = todo.pop()
            if parent not in ancestors and self.state[parent] not in ('done', 'skipped'):
                ancestors.add(parent)
                todo.extend(self.graph.parents(parent))
        jobs = []
        for node, _ in self.futures.values():
            reason = 'data_dependency' if node.id in ancestors else None
            if reason is None:
                for parent in ancestors:
                    pending = self.nodes[parent]
                    if (self.state[parent] == 'pending' and self.condition(pending)
                            and all(self.state[p] in ('done', 'skipped') for p in self.graph.parents(parent))
                            and set(node.resources).intersection(pending.resources)):
                        reason = 'resource_conflict'
                        break
            if reason:
                jobs.append(dict(node=node.id, attempt=self.attempts[node.id], reason=reason,
                                 blocked_physical=next_node.id))
        return jobs

    def run(self):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pmir-compute')
        start_sim, start_step = self.adapter.backend.sim_time_s, self.adapter.backend.step_count
        error = None
        blocker_key, blockers = None, []
        try:
            with self.costs.scope('state_validation'):
                self.adapter.probe()
            while any(value not in ('done', 'skipped') for value in self.state.values()):
                progressed = False
                for key, (node, future) in list(self.futures.items()):
                    if future.done():
                        result, span = future.result()
                        del self.futures[key]
                        with self.costs.scope('scheduler_decision'):
                            retry = self.retry_hidden_preparation(node, result)
                        self.spans.append(dict(span, consumed=not retry, invalidated=retry))
                        if not retry:
                            self.commit(node, result)
                        progressed = True
                with self.costs.scope('scheduler_decision'):
                    for node in self.graph.nodes:
                        if self.state[node.id] != 'pending':
                            continue
                        if any(self.state[p] not in ('done', 'skipped') for p in self.graph.parents(node.id)):
                            if self.mode == 'serial':
                                break
                            continue
                        if not self.condition(node):
                            self.state[node.id] = 'skipped'
                            self.emit('skipped', node=node.id, reason='untaken_control_branch')
                            progressed = True
                            continue
                        if not self.available(node):
                            if self.mode == 'serial':
                                break
                            continue
                        if self.mode == 'serial' and (self.active or self.futures):
                            break
                        self.start(node)
                        progressed = True
                        if node.physical:
                            self.active[node.id] = (node, self.adapter.physical(node, self.values), time.perf_counter())
                        elif node.owner:
                            started = time.perf_counter()
                            with self.costs.scope('foreground_plan'), timing_stage('PMIR前台', detail=f'{node.id}:{node.kind}'):
                                result = self.adapter.immediate(node, self.values)
                            self.spans.append(dict(node=node.id, kind=node.kind, phase=node.phase, target=node.target,
                                physical=False, start_s=started-self.started, duration_s=time.perf_counter()-started,
                                thread=threading.get_ident(), consumed=True, attempt=self.attempts[node.id]))
                            with self.costs.scope('runtime_bookkeeping'):
                                self.commit(node, result)
                        else:
                            job = self.adapter.worker_job(node, self.values)
                            self.futures[node.id] = (node, pool.submit(copy_context().run, run_compute,
                                node, job, self.started, self.attempts[node.id]))
                        if self.mode == 'serial':
                            break
                for key, (node, iterator, started) in list(self.active.items()):
                    try:
                        with self.costs.scope('robot_control'):
                            next(iterator)
                    except StopIteration as stopped:
                        self.spans.append(dict(node=node.id, kind=node.kind, phase=node.phase,
                            target=node.target, physical=True, start_s=started-self.started,
                            duration_s=time.perf_counter()-started, thread=threading.get_ident(), consumed=True,
                            attempt=self.attempts[node.id]))
                        del self.active[key]
                        self.commit(node, stopped.value)
                    progressed = True
                if not self.active and self.futures:
                    with self.costs.scope('profiling_analysis'):
                        key = (tuple(self.state.values()), tuple(self.futures))
                        if key != blocker_key:
                            blockers = self.blocking_jobs()
                            blocker_key = key
                    waiting = time.perf_counter()
                    with self.costs.scope('compute_wait_poll'):
                        self.adapter.idle_tick()
                    end = time.perf_counter()
                    self.arm_idle_compute_s += end-waiting
                    self.wait_windows.append(dict(start_s=waiting-self.started, end_s=end-self.started, jobs=blockers))
                elif not progressed:
                    pending = [n.id for n in self.graph.nodes if self.state[n.id] == 'pending']
                    raise RuntimeError(f'No runnable operation; unmet gates/dependencies: {pending}')
        except Exception as caught:
            error = f'{type(caught).__name__}: {caught}'
            self.adapter.backend.halted = True
            self.emit('stopped', error=error, completed=self.adapter.completed)
        finally:
            for node, iterator, started in self.active.values():
                self.spans.append(dict(node=node.id, kind=node.kind, phase=node.phase, target=node.target,
                    physical=True, start_s=started-self.started, duration_s=time.perf_counter()-started,
                    thread=threading.get_ident(), consumed=False, attempt=self.attempts[node.id]))
                iterator.close()
            for key in self.state:
                if self.state[key] == 'running':
                    self.state[key] = 'aborted'
            for _, future in self.futures.values():
                future.cancel()
            with self.costs.scope('shutdown_wait'), timing_stage('PMIR后台收尾'):
                pool.shutdown(wait=True, cancel_futures=True)
            for node, future in self.futures.values():
                if not future.cancelled():
                    try:
                        _, span = future.result()
                        self.spans.append(dict(span, consumed=False))
                    except Exception as caught:
                        self.emit('unused_compute_failed', node=node.id, error=type(caught).__name__)
            self.emit('finished', success=error is None, error=error, completed=self.adapter.completed)
            self.log.close()
        totals, wall = self.costs.finish()
        return dict(success=error is None, error=error, node_states=self.state, spans=self.spans,
                    wall_duration_s=wall, foreground_cost_s=totals,
                    sim_duration_s=self.adapter.backend.sim_time_s-start_sim,
                    control_steps=self.adapter.backend.step_count-start_step,
                    idle_control_steps=self.adapter.idle_steps,
                    arm_idle_compute_wall_s=self.arm_idle_compute_s,
                    completed=self.adapter.completed)
