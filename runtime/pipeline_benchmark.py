"""Controlled scheduler replay; no cameras, robot clients, calibration or buffers.

Uses the production perception publisher, Hybrid runner, CaP validator and
timing recorder. Perception and execution are explicitly synthetic. ``local``
uses an already-running LLM for code only, never starts/stops a server.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from agents.hybrid_execution import HybridTaskRunner
from agents.plan_graph import PlanGraph, PlanNode
from agents.simple_execution_runtime import SimpleExecutionRuntime
from core.actions import grasp, place
from core.context import RuntimeContext
from core.skill_base import SkillResult
from runtime.config import SimpleAgentSystemConfig
from runtime.observation_memory import ObservationMemory
from runtime.task_perception_pipeline import TaskPerceptionPipeline
from runtime.timing import task_timing, timing_stage
from services.object_reference_service import atomic_json


VARIANTS = dict(serial=(False, False), perception=(True, False),
                policy=(False, True), both=(True, True))
PAIRS = [['西瓜霜', 'blue tray'], ['抗病毒口服液', 'pink tray']]
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Workload:
    analysis_s: float = .12
    container_s: float = .16
    code_s: float = .10
    execute_s: float = .15


def intervals(spans):
    """Union inclusive spans so nested work is never counted twice."""
    merged = []
    for start, end in sorted((s['start_s'], s['start_s'] + s['duration_s']) for s in spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_seconds(spans):
    return sum(b-a for a, b in intervals(spans))


def overlap_seconds(left, right):
    return sum(max(0., min(b, d)-max(a, c))
               for a, b in intervals(left) for c, d in intervals(right))


def summarize_run(directory):
    timing = json.loads((directory/'task_timing.json').read_text())
    trace = json.loads((directory/'hybrid_trace.json').read_text())
    spans = timing['spans']
    def stage(name, background=None):
        return [s for s in spans if s['name'] == name
                and (background is None or s['background'] == background)]
    jobs = trace['events'][-1].get('prefetch_jobs', [])
    used = {(j['node'], j['revision']) for j in jobs if j['outcome'] == 'used'}
    used_spans = [s for s in stage('子任务代码生成', True)
                  if (s['attributes']['node'], s['attributes']['revision']) in used]
    return dict(duration_s=timing['duration_s'], timing_status=timing['status'],
        perception_overlap_s=overlap_seconds(stage('首帧后台感知总计'), stage('模拟容器预观察')),
        policy_overlap_s=overlap_seconds(stage('子任务代码生成', True), stage('模拟子任务执行')),
        useful_policy_overlap_s=overlap_seconds(used_spans, stage('模拟子任务执行')),
        perception_join_s=interval_seconds(stage('等待后台感知结果')),
        policy_join_s=interval_seconds(stage('等待后台子任务代码')),
        policy_foreground_s=interval_seconds(stage('子任务代码生成', False)),
        policy_shutdown_s=interval_seconds(stage('等待后台代码线程退出')),
        resource_wait_s=interval_seconds(stage('资源锁等待')),
        prefetch_outcomes=dict(Counter(j['outcome'] for j in jobs)),
        prefetch_wasted_s=sum(j['wasted_generation_s'] for j in jobs),
        prefetch_queue_s=sum(j.get('queue_wait_s', 0.) for j in jobs),
        retries=sum(e['event'] == 'node_repair' for e in trace['events']),
        completed_actions=trace['completed_actions'])


class StubClient:
    def __init__(self, delay):
        self.delay = delay

    def chat(self, text, **kwargs):
        pairs = json.loads(text.split('\nRequired pairs: ', 1)[1].split('\n', 1)[0])
        time.sleep(self.delay)
        obj, tray = pairs[0]
        return f'def policy():\n    if observe({obj!r}):\n        pick_place({obj!r}, {tray!r})\n'


class SyntheticEvidence:
    """Only a memory/pose fixture. Has no capture, motion or device methods."""
    observation_scope = 'synthetic_scheduler_fixture_not_robot_calibration'

    def __init__(self):
        self.observation_memory = ObservationMemory()

    def observation_pose(self):
        # Synthetic equality sentinel required by the real publication checks.
        # Never loaded from historical records or passed to an actuator.
        return [0.] * 6


def run_trial(directory, variant, scenario, workload, client, mode):
    directory.mkdir(parents=True, exist_ok=False)
    perception_enabled, policy_enabled = VARIANTS[variant]
    actions = [action for obj, tray in PAIRS for action in (grasp(obj), place(tray))]
    config = SimpleAgentSystemConfig(goal='；'.join(f'将 {a} 放入 {b}' for a, b in PAIRS),
        policy_backend='hybrid', execute_robot=False, enable_policy_pipeline=policy_enabled,
        enable_perception_pipeline=perception_enabled, plan_json=str(directory/'plan.json'),
        generated_code=str(directory/'policy.py'), replan_json=str(directory/'replan.json'),
        failure_report_path=str(directory/'failure.json'), simple_memory_path=str(directory/'memory.json'))
    graph = PlanGraph(goal=config.goal)
    for i, action in enumerate(actions):
        graph.add_node(PlanNode(str(i), action.target, action.op, action.target,
                               skill=action.op, args={'target': action.target}))
        if i:
            graph.add_edge(str(i-1), str(i))
    graph.save_json(config.plan_json)
    evidence = SyntheticEvidence()
    context = RuntimeContext(camera=evidence)
    runtime = SimpleExecutionRuntime(config)
    runtime._active_context = context
    owner = threading.get_ident()
    robot = SimpleNamespace(context=context, observations={}, _pipeline_frame_bindings=set(),
        information_states={}, geometry_recovery_state={'history': []},
        config=SimpleNamespace(nesyro_gate_trace_path=str(directory/'gate.jsonl')))
    attempts = []
    def observe(target):
        assert threading.get_ident() == owner
        return SkillResult(True, dict(simulated=True, target=target))
    robot.observe = observe

    def execute(robot, context, camera, actions, *, completed_actions, start_index):
        assert threading.get_ident() == owner
        target = actions[start_index].target
        attempts.append(target)
        with timing_stage('模拟子任务执行', detail=target):
            time.sleep(workload.execute_s)
        if len(attempts) == 1 and scenario != 'success':
            kind = 'grasp_pose_not_found' if scenario == 'repair_first' else 'geometry_occlusion_budget_exhausted'
            runtime._write_failure_report(grasp(target), target, kind, 'injected synthetic feedback',
                                          completed_actions=completed_actions)
            return False
        done = list(completed_actions)
        for action in actions[start_index:]:
            done.append(vars(action))
            runtime.task_state.complete(action, done)
        evidence.observation_memory.invalidate('synthetic_manipulation', discard_scene=True)
        robot.observations.clear()
        robot._pipeline_frame_bindings.clear()
        return True
    runtime._run_actions = execute

    capture = dict(color_path='synthetic://fixed-frame', timestamp=time.time(),
        capture_geometry=dict(before=evidence.observation_pose(), synthetic=True),
        observation_cache=dict(scope=evidence.observation_scope, epoch=0, scene_epoch=0))
    scene = dict(primary=dict(relation_graph={'nodes': []}, possible_pairs=[]),
        target_association=dict(status='target_not_found', node_id=None), synthetic=True)
    scene_path = directory/'synthetic_scene.json'
    atomic_json(scene_path, scene)
    scene.update(report_path=str(scene_path), report_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest())

    def analyze(worker_context, cfg, target, views):
        assert all(getattr(worker_context, key) is None for key in ('camera', 'arm', 'gripper', 'robot', 'xarm'))
        with timing_stage('模拟首帧分析'):
            time.sleep(workload.analysis_s)
        return SkillResult(True, dict(geometry_capture=deepcopy(views[0]),
                                     geometry_occlusion=deepcopy(scene), target_identified=False))

    pipeline = runner = None
    metadata = dict(evidence_level='offline_scheduler' if mode == 'stub' else 'real_llm_simulated_execution',
        execute_robot=False, perception_model='synthetic', code_model=mode, variant=variant,
        scenario=scenario, workload=asdict(workload), perception_pipeline=perception_enabled,
        policy_pipeline=policy_enabled, candidate_prefetch='skipped_missing_transforms',
        scope='fixed-frame scheduler replay; excludes capture, planning, model loading and physical motion')
    with patch('skill_lib.actions.geometry_occlusion.analyze_geometry_views', analyze):
        with task_timing(config.goal, directory/'task_timing.json', metadata) as recorder:
            try:
                runner = HybridTaskRunner(runtime, robot, context, None, actions, actions, client)
                # Same first-node cost in all variants, as in the agent coding stage.
                config.cap_policy_source = runner.generate(0, 0)
                if perception_enabled:
                    pipeline = TaskPerceptionPipeline(robot, PAIRS[0][0], (capture, None, None, None),
                                                      directory/'perception_pipeline.json')
                with timing_stage('模拟容器预观察'):
                    time.sleep(workload.container_s)
                if pipeline is None:
                    # Matched work: serialize the SAME frozen-frame analysis after
                    # containers. This is not an end-to-end CLI/camera ablation.
                    pipeline = TaskPerceptionPipeline(robot, PAIRS[0][0], (capture, None, None, None),
                                                      directory/'perception_pipeline.json')
                assert pipeline.publish(), 'Synthetic first frame failed production publication checks'
                success = runner.run()
                recorder.status = 'ok' if success else 'failed'
            finally:
                if pipeline is not None:
                    pipeline.close()
                if runner is not None:
                    runner.pool.shutdown(wait=True, cancel_futures=True)
    row = dict(variant=variant, scenario=scenario, success=success, attempts=attempts,
               directory=str(directory), **summarize_run(directory))
    expected_attempts = ([PAIRS[0][0]] if scenario == 'stop_first' else
                         [PAIRS[0][0], PAIRS[0][0], PAIRS[1][0]] if scenario == 'repair_first' else
                         [pair[0] for pair in PAIRS])
    row['correct'] = (success == (scenario != 'stop_first') and attempts == expected_attempts
                      and row['completed_actions'] == ([] if scenario == 'stop_first' else [vars(a) for a in actions]))
    atomic_json(directory/'result.json', row)
    return row


def aggregate(rows):
    groups = []
    for scenario in dict.fromkeys(r['scenario'] for r in rows):
        for variant in VARIANTS:
            selected = [r for r in rows if r['scenario'] == scenario and r['variant'] == variant]
            if not selected:
                continue
            durations = [r['duration_s'] for r in selected]
            matched = []
            for row in selected:
                baseline = next(r for r in rows if r['scenario'] == scenario
                                and r['variant'] == 'serial' and r['repeat'] == row['repeat'])
                if (row['correct'] and baseline['correct'] and row['success'] and baseline['success']
                        and row['attempts'] == baseline['attempts'] and row['retries'] == baseline['retries']):
                    matched.append(baseline['duration_s']/row['duration_s'])
            groups.append(dict(scenario=scenario, variant=variant, runs=len(selected),
                correct=sum(r['correct'] for r in selected), successful=sum(r['success'] for r in selected),
                median_s=statistics.median(durations), min_s=min(durations), max_s=max(durations),
                paired_speedup_median=statistics.median(matched) if matched else None,
                **{key: statistics.median(r[key] for r in selected) for key in
                   ('perception_overlap_s', 'policy_overlap_s', 'useful_policy_overlap_s', 'perception_join_s', 'policy_join_s',
                    'policy_foreground_s', 'policy_shutdown_s', 'resource_wait_s', 'prefetch_wasted_s')}))
    return groups


def source_fingerprint():
    paths = ['agents/hybrid_execution.py', 'agents/cap_code_generator.py', 'runtime/timing.py',
             'runtime/task_perception_pipeline.py', 'runtime/perception_pipeline.py', 'runtime/pipeline_benchmark.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--scenarios', nargs='+', choices=['success', 'repair_first', 'stop_first'],
                        default=['success', 'repair_first'])
    parser.add_argument('--model-mode', choices=['stub', 'local'], default='stub')
    parser.add_argument('--base-url', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--model', default='qwen3vl-8b-awq')
    for name, value in asdict(Workload()).items():
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=value)
    args = parser.parse_args(argv)
    workload = Workload(**{name: getattr(args, name) for name in asdict(Workload())})
    if args.repeats < 1 or args.warmups < 0 or any(not math.isfinite(v) or v < 0 for v in asdict(workload).values()):
        parser.error('repeats >= 1, warmups >= 0 and finite nonnegative delays required')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    client = StubClient(workload.code_s)
    if args.model_mode == 'local':
        from urllib.parse import urlparse
        from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient
        if urlparse(args.base_url).hostname not in ('localhost', '127.0.0.1', '::1'):
            parser.error('local mode requires a loopback service; no remote calls')
        client = OpenAIChatGPTVLMClient(args.base_url, 'local', args.model, timeout=60)
    manifest = dict(schema='pipeline_scheduler_benchmark_v1', arguments={
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        workload=asdict(workload), pairs=PAIRS, source_sha256=source_fingerprint(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        python=sys.version, interpreter=sys.executable, temperature=0., max_tokens=2048,
        startup='process imports excluded; warmup rounds separate; server cold/warm state unknown',
        hardware_executed=False, perception='synthetic fixed frame; no identity or grasp verification',
        code_delay_applies=args.model_mode == 'stub',
        serial_scope='same production first-frame job joined after containers; not full CLI ablation')
    atomic_json(args.output_dir/'manifest.json', manifest)
    rows, order = [], []
    rng = random.Random(args.seed)
    for repeat in range(-args.warmups, args.repeats):
        cases = [(scenario, variant) for scenario in args.scenarios for variant in VARIANTS]
        rng.shuffle(cases)
        for scenario, variant in cases:
            phase = f'warmup_{-repeat:02d}' if repeat < 0 else f'measured_{repeat:02d}'
            directory = args.output_dir/phase/scenario/variant
            order.append(str(directory.relative_to(args.output_dir)))
            atomic_json(args.output_dir/'order.json', order)
            row = run_trial(directory, variant, scenario, workload, client, args.model_mode)
            row['repeat'] = repeat
            if repeat >= 0:
                rows.append(row)
            atomic_json(args.output_dir/'results.json', rows)
            if not row['correct']:
                raise RuntimeError(f'Scheduler outcome mismatch: {directory}')
    report = dict(schema='pipeline_scheduler_comparison_v1', evidence_level=(
        'offline_scheduler' if args.model_mode == 'stub' else 'real_llm_simulated_execution'),
        groups=aggregate(rows), runs=rows,
        note='Ratios describe this injected workload only, not robot/model perception speedup. '
             'Span unions/intersections are wall time; wasted/queue sums are job work.')
    atomic_json(args.output_dir/'comparison.json', report)
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
