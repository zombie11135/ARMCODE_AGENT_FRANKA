"""Paired serial/dynamic PMIR trials with actual robosuite contact physics.

No hardware/LLM/model-service imports. The optional delay is a named latency
injection, not measured vision-model inference. Outputs never overwrite runs.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
from itertools import permutations
import statistics
import sys
import time

import numpy as np
from PIL import Image

from runtime.pmir import compile_program
from runtime.pmir.scheduler import Scheduler
from runtime.timing import task_timing
from .adapter import Adapter
from .perception import mask_for


PROGRAM = '''def policy():
    pick_place("red_cube", "red_goal")
    pick_place("blue_cube", "blue_goal")
'''
ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def union(items):
    out = []
    for start, end in sorted(items):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def overlap(left, right):
    return sum(max(0., min(b, d)-max(a, c)) for a, b in union(left) for c, d in union(right))


def run_trial(directory, *, source=PROGRAM, scenario='independent', mode='dynamic', seed=7,
              compute_delay_s=0., clock='fast'):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'policy.py').write_text(source)
    started = time.perf_counter()
    graph = compile_program(source)
    compile_s = time.perf_counter()-started
    if any(n.kind == 'BLACKBOX' for n in graph.nodes):
        raise ValueError('A blackbox contract needs an explicit simulation handler before execution')
    write(directory/'graph_initial.json', graph.to_dict())
    (directory/'graph_initial.mmd').write_text(graph.mermaid())
    started = time.perf_counter()
    adapter = Adapter(directory, seed=seed, scenario=scenario, compute_delay_s=compute_delay_s, clock=clock)
    init_s = time.perf_counter()-started
    try:
        before = adapter.snapshot()
        initial_qpos = adapter.backend.env.sim.data.qpos.copy()
        initial_hash = hashlib.sha256(initial_qpos.tobytes()).hexdigest()
        Image.fromarray(before['camera']['rgb']).save(directory/'before.png')
        np.savez_compressed(directory/'before_rgbd.npz', **{
            key: before['camera'][key] for key in ('rgb', 'depth_m', 'K', 'T_world_from_camera_optical')})
        pixels = {t: int(mask_for(before['camera'], t)[0].sum()) for t in adapter.versions}
        if scenario == 'occluded' and (pixels['blue_cube'] >= 25 or pixels['red_cube'] < 25):
            raise RuntimeError(f'Scene failed its initial visibility condition: {pixels}')
        metadata = dict(backend='robosuite_pmir', simulated=True, execute_robot=False, mode=mode,
            scenario=scenario, seed=seed, clock=clock, perception='rgbd_color_known_box_geometry',
            injected_delay_per_perceive_s=compute_delay_s, branch_prediction=False,
            state_monitor='simulator_ground_truth', resource_capacity={'cpu': 1, 'robot': 1, 'gripper': 1, 'camera': 1})
        with task_timing('Two boxes via PMIR', directory/'task_timing.json', metadata) as timing:
            scheduler = Scheduler(graph, adapter, directory, mode=mode)
            result = scheduler.run()
            result['goal_reached'] = adapter.backend.task_status()['complete']
            result['success'] &= result['goal_reached']
            timing.status = 'ok' if result['success'] else 'failed'
        result.update(mode=mode, scenario=scenario, seed=seed, clock=clock,
            injected_delay_per_perceive_s=compute_delay_s, compile_wall_s=compile_s, init_wall_s=init_s,
            initial_qpos_sha256=initial_hash,
            initial_rgb_sha256=hashlib.sha256(before['camera']['rgb'].tobytes()).hexdigest(), initial_pixels=pixels, lift_height_m=adapter.lift_evidence,
            final_task=adapter.backend.task_status(), deadline_misses=adapter.deadline_misses,
            max_lateness_s=adapter.max_lateness_s,
            note='Host wall time and simulated control time are separate. Fast mode freezes physics '
                 'while only computation is pending; real_time mode holds control during waits. '
                 'Neither is physical FR3 throughput. No neural perception or PEIP.')
        compute = [(s['start_s'], s['start_s']+s['duration_s']) for s in result['spans']
                   if s['kind'] in ('PERCEIVE', 'GRASP_PLAN')]
        physical = [(s['start_s'], s['start_s']+s['duration_s']) for s in result['spans'] if s['physical']]
        result['compute_physical_residency_overlap_wall_s'] = overlap(compute, physical)
        result['compute_physics_step_overlap_wall_s'] = overlap(compute, [
            (a-scheduler.started, b-scheduler.started) for a, b in adapter.step_intervals])
        result['task_timing_wall_s'] = json.loads((directory/'task_timing.json').read_text())['duration_s']
        profile = dict(schema='pmir_exclusive_costs_v1', origin='scheduler_start',
            intervals=scheduler.costs.intervals, wait_windows=scheduler.wait_windows,
            note='Owner intervals are exclusive. Worker spans are an overlapping second lane. '
                 'Image acquisition includes backend format conversion/copies; snapshot_copy is the additional frozen copy. '
                 'physics_step includes MuJoCo integration and controller computation, not real arm time.')
        write(directory/'cost_profile.json', profile)
        result.update(cost_metrics(result, profile))
        result['physical_sequence'] = [(e['target'], e['phase']) for e in scheduler.events
                                      if e['event'] == 'start' and e['kind'] in ('MOVE', 'GRIPPER', 'VERIFY')]
        result['event_releases'] = [e for e in scheduler.events if e['event'] == 'event_released']
        result['directory'] = str(directory)
        after = adapter.snapshot()
        Image.fromarray(after['camera']['rgb']).save(directory/'after.png')
        for producer, snapshot in adapter.probe_frames:
            Image.fromarray(snapshot['camera']['rgb']).save(directory/f'visibility_{producer}.png')
            np.savez_compressed(directory/f'visibility_{producer}.npz', **{
                key: snapshot['camera'][key] for key in ('rgb', 'depth_m', 'K', 'T_world_from_camera_optical')})
        graph.validate()
        write(directory/'graph_final.json', graph.to_dict())
        (directory/'graph_final.mmd').write_text(graph.mermaid())
        write(directory/'result.json', result)
        return result
    finally:
        adapter.close()


def cost_metrics(result, profile):
    workers = [s for s in result['spans'] if s['kind'] in ('PERCEIVE', 'GRASP_PLAN')]
    critical, wait_compute = [], []
    critical_by_reason = {'data_dependency': [], 'resource_conflict': []}
    for window in profile['wait_windows']:
        for span in workers:
            start = max(window['start_s'], span['start_s'])
            end = min(window['end_s'], span['start_s']+span['duration_s'])
            if end <= start:
                continue
            wait_compute.append((start, end))
            for job in window['jobs']:
                if (job['node'], job['attempt']) == (span['node'], span['attempt']):
                    critical.append((start, end))
                    critical_by_reason[job['reason']].append((start, end))
    duration = lambda intervals: sum(b-a for a, b in union(intervals))
    injected = sum(s.get('components', {}).get('injected_wait', [0, 0])[1]
                   - s.get('components', {}).get('injected_wait', [0, 0])[0] for s in workers)
    perception = sum(s['duration_s'] for s in workers if s['kind'] == 'PERCEIVE')
    physical_steps = [(a, b) for a, b, category in profile['intervals'] if category == 'physics_step']
    compute = [(s['start_s'], s['start_s']+s['duration_s']) for s in workers]
    return dict(perception_wall_s=perception, injected_wait_wall_s=injected,
        perception_without_injected_wait_s=perception-injected,
        worker_thread_cpu_s=sum(s['cpu_s'] for s in workers),
        grasp_plan_wall_s=sum(s['duration_s'] for s in workers if s['kind'] == 'GRASP_PLAN'),
        perception_attempts=sum(s['kind'] == 'PERCEIVE' for s in workers),
        invalidated_attempts=sum(bool(s.get('invalidated')) for s in workers),
        wasted_compute_wall_s=sum(s['duration_s'] for s in workers if not s.get('consumed')),
        compute_during_wait_s=duration(wait_compute),
        exposed_critical_compute_s=duration(critical),
        critical_by_reason_s={key: duration(value) for key, value in critical_by_reason.items()},
        compute_actual_physics_overlap_s=overlap(compute, physical_steps),
        critical_compute_definition='Measured running compute intersecting no-motion wait windows and '
            'unfinished data/resource chain to next physical operation. Conservative exposed-blocking '
            'time; excludes polling after completion, scheduler delays and hidden computation. '
            'Not a full counterfactual DAG critical-path length and not net speedup.')


def summarize(rows):
    groups, pairs, effects = [], [], []
    keys = sorted({(r['scenario'], r['injected_delay_per_perceive_s'], r['mode']) for r in rows})
    for scenario, delay, mode in keys:
        subset = [r for r in rows if (r['scenario'], r['injected_delay_per_perceive_s'], r['mode']) == (scenario, delay, mode)]
        wall = [r['wall_duration_s'] for r in subset]
        costs = sorted({k for r in subset for k in r['foreground_cost_s']})
        groups.append(dict(scenario=scenario, delay_ms=delay*1000, mode=mode, n=len(subset),
            successes=sum(r['success'] for r in subset), median_wall_s=statistics.median(wall),
            p90_wall_s=float(np.quantile(wall, .9)),
            mean_foreground_cost_s={k: statistics.mean(r['foreground_cost_s'].get(k, 0) for r in subset) for k in costs},
            mean_wall_s=statistics.mean(wall),
            **{'median_'+key: statistics.median(r[key] for r in subset) for key in (
                'sim_duration_s', 'compute_physical_residency_overlap_wall_s', 'compute_actual_physics_overlap_s',
                'arm_idle_compute_wall_s', 'exposed_critical_compute_s', 'perception_wall_s',
                'injected_wait_wall_s', 'perception_without_injected_wait_s', 'worker_thread_cpu_s',
                'invalidated_attempts', 'wasted_compute_wall_s', 'perception_attempts')}))
    for baseline in (r for r in rows if r['mode'] == 'serial'):
        for candidate in (r for r in rows if r['mode'] != 'serial' and all(
                r[k] == baseline[k] for k in ('scenario', 'seed', 'repeat', 'injected_delay_per_perceive_s'))):
            matched = all(baseline[k] == candidate[k] for k in ('initial_qpos_sha256', 'initial_rgb_sha256'))
            same_actions = baseline['physical_sequence'] == candidate['physical_sequence']
            same_steps = baseline['control_steps'] == candidate['control_steps']
            comparable = matched and same_actions and (same_steps or baseline['clock'] != 'fast') and baseline['success'] and candidate['success'] and baseline['completed'] == candidate['completed']
            costs = set(baseline['foreground_cost_s']) | set(candidate['foreground_cost_s'])
            pairs.append(dict(scenario=baseline['scenario'], seed=baseline['seed'], repeat=baseline['repeat'],
                delay_ms=baseline['injected_delay_per_perceive_s']*1000, mode=candidate['mode'],
                identical_initial_state=matched, identical_actions=same_actions, identical_steps=same_steps,
                comparable=bool(comparable),
                saved_wall_s=baseline['wall_duration_s']-candidate['wall_duration_s'] if comparable else None,
                saved_task_timing_s=baseline['task_timing_wall_s']-candidate['task_timing_wall_s'] if comparable else None,
                wall_speedup=baseline['wall_duration_s']/candidate['wall_duration_s'] if comparable else None,
                foreground_saved_s={k: baseline['foreground_cost_s'].get(k, 0)-candidate['foreground_cost_s'].get(k, 0) for k in costs},
                serial_steps=baseline['control_steps'], candidate_steps=candidate['control_steps']))
    rng = np.random.default_rng(20261110)
    for scenario, delay, mode in keys:
        subset = [p for p in pairs if (p['scenario'], p['delay_ms'], p['mode']) == (scenario, delay*1000, mode) and p['comparable']]
        if not subset:
            continue
        savings = np.array([p['saved_wall_s'] for p in subset])
        boots = np.median(rng.choice(savings, (10000, len(savings)), replace=True), axis=1)
        effects.append(dict(scenario=scenario, delay_ms=delay*1000, mode=mode, n=len(subset),
            median_saved_s=float(np.median(savings)), mean_saved_s=float(np.mean(savings)),
            median_speedup=statistics.median(p['wall_speedup'] for p in subset),
            median_saved_bootstrap_95ci_s=np.quantile(boots, [.025, .975]).tolist(),
            wins=int((savings > 0).sum()),
            mean_foreground_saved_s={k: statistics.mean(p['foreground_saved_s'].get(k, 0) for p in subset)
                                     for k in set().union(*(p['foreground_saved_s'] for p in subset))}))
    return dict(schema='pmir_robosuite_comparison_v2', groups=groups, pairs=pairs, effects=effects,
        evidence='Actual MuJoCo grasp/place + known-color RGB-D; declared artificial per-perception latency',
        note='Bootstrap intervals are exploratory with few paired repeats; no real-model/robot claims. '
             'Saved wall time is paired Serial minus candidate, never overlap duration. '
             'Mean exclusive foreground cost differences sum to mean net savings.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--program-file', type=Path)
    parser.add_argument('--scenarios', nargs='+', choices=['independent', 'occluded'], default=['independent', 'occluded'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 8, 9])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--compute-delay-s', type=float, default=0.)
    parser.add_argument('--modes', nargs='+', choices=['serial', 'naive', 'dynamic'], default=['serial', 'dynamic'])
    parser.add_argument('--delays-ms', nargs='+', type=float)
    parser.add_argument('--clock', choices=['fast', 'real_time'], default='fast')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or not math.isfinite(args.compute_delay_s) or args.compute_delay_s < 0:
        parser.error('Invalid repetitions, warmups or compute delay')
    delays = [v/1000 for v in args.delays_ms] if args.delays_ms is not None else [args.compute_delay_s]
    if any(not math.isfinite(v) or v < 0 for v in delays):
        parser.error('Delays must be finite and nonnegative')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.program_file.read_text() if args.program_file else PROGRAM
    files = [p for folder in ('runtime/pmir', 'simulation/pmir', 'simulation/robosuite_backend')
             for p in (ROOT/folder).glob('*.py')]
    write(args.output_dir/'manifest.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        python=sys.version, executable=sys.executable, packages={k: version(k) for k in ('robosuite', 'mujoco', 'numpy', 'scipy')},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        policy=source, physical_robot=False, optional_latency_injection_s=args.compute_delay_s,
        thread_limits={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        resources='One owner thread for MuJoCo/EGL, one CPU worker; no GPU inference backend'))
    for file in files:
        destination = args.output_dir/'source_snapshot'/file.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
    rows, order = [], []
    rng = random.Random(20261110)
    mode_orders = list(permutations(args.modes))
    for repeat in range(-args.warmups, args.repeats):
        blocks = [(seed, scenario, delay) for seed in (args.seeds[:1] if repeat < 0 else args.seeds)
                  for scenario in args.scenarios for delay in (delays[-1:] if repeat < 0 else delays)]
        rng.shuffle(blocks)
        for seed, scenario, delay in blocks:
            # Over six repeats, each method occupies each position twice. Fixed
            # blocks match scene/delay; their order is independently randomized.
            modes = mode_orders[repeat % len(mode_orders)]
            for mode in modes:
                phase = f'warmup_{-repeat:02d}' if repeat < 0 else f'measured_{repeat:02d}'
                directory = args.output_dir/phase/f'seed_{seed}'/scenario
                if args.delays_ms is not None:
                    directory = directory/f'delay_{delay*1000:g}ms'
                directory = directory/mode
                order.append(str(directory.relative_to(args.output_dir)))
                write(args.output_dir/'order.json', order)
                result = run_trial(directory, source=source, scenario=scenario, mode=mode, seed=seed,
                                   compute_delay_s=delay, clock=args.clock)
                result['repeat'] = repeat
                if repeat >= 0:
                    rows.append(result)
                write(args.output_dir/'results.json', rows)
                print(json.dumps(dict(repeat=repeat, delay_ms=delay*1000, **{k: result[k] for k in (
                    'scenario', 'mode', 'success', 'wall_duration_s', 'sim_duration_s', 'invalidated_attempts', 'error')})), flush=True)
    comparison = summarize(rows)
    write(args.output_dir/'comparison.json', comparison)
    print(json.dumps(comparison['groups'], indent=2))
    return 0 if all(row['success'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
