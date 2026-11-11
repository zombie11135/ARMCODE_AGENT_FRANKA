"""Export PMIR graph, paired timeline and measured visibility evidence.

Uses matplotlib from the existing perception environment; the simulation
environment does not need extra plotting dependencies.
"""
import argparse
from collections import defaultdict
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


def draw_graph(data, out):
    groups = defaultdict(list)
    for node in data['nodes']:
        groups[node['target']].append(node)
    positions = {n['id']: (column*4, -row*1.1) for column, nodes in enumerate(groups.values())
                 for row, n in enumerate(nodes)}
    adjacent = defaultdict(set)
    for edge in data['edges']:
        adjacent[edge['source']].add(edge['target'])
    def reaches(source, target):
        todo, seen = list(adjacent[source]), set()
        while todo:
            value = todo.pop()
            if value == target:
                return True
            if value not in seen:
                seen.add(value)
                todo.extend(adjacent[value])
        return False
    fig, ax = plt.subplots(figsize=(10, 14))
    drawn = set()
    for edge in sorted(data['edges'], key=lambda e: e['kind'] != 'event'):
        a, b = edge['source'], edge['target']
        if (a, b) in drawn:
            continue
        if edge['kind'] != 'event' and any(reaches(child, b) for child in adjacent[a] if child != b):
            continue  # Transitive reduction is for display only; JSON retains every reason.
        drawn.add((a, b))
        color = '#c44e52' if edge['kind'] == 'event' else '#4c72b0'
        start, end = positions[a], positions[b]
        cross = start[0] != end[0]
        if cross:
            start, end = (start[0]+1.45, start[1]), (end[0]-1.45, end[1])
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
            shrinkA=0 if cross else 22, shrinkB=0 if cross else 22, color=color, linewidth=1.3,
            linestyle='--' if edge['kind'] == 'event' else '-'))
    for nodes in groups.values():
        for node in nodes:
            x, y = positions[node['id']]
            ax.add_patch(FancyBboxPatch((x-1.4, y-.32), 2.8, .64, boxstyle='round,pad=0.05',
                facecolor='#e4ebf4' if not node['physical'] else '#eee2f4', edgecolor='#536078'))
            ax.text(x, y, f'{node["id"]} {node["kind"]}\n{node["phase"]}', ha='center', va='center', fontsize=8)
    for target, nodes in groups.items():
        ax.text(positions[nodes[0]['id']][0], 1.0, target, ha='center', weight='bold')
    ax.set_xlim(-1.8, max(x for x, y in positions.values())+1.8)
    ax.set_ylim(min(y for x, y in positions.values())-1, 1.6)
    ax.axis('off')
    ax.set_title('PMIR: causal dependencies + measured event release\n'
                 'Red dashed edge: confirmed visibility/stability; resource constraints remain separate', fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'dependency_graph.svg')
    fig.savefig(out/'dependency_graph.png', dpi=130)
    plt.close(fig)


def draw_timeline(directories, out):
    colors = {'physical': '#8c6bb1', 'compute': '#3690c0', 'capture': '#31a354', 'plan': '#fd8d3c'}
    fig, axes = plt.subplots(len(directories), 1, figsize=(12, 3*len(directories)), sharex=True)
    for ax, directory in zip(axes, directories):
        result = json.loads((directory/'result.json').read_text())
        for span in result['spans']:
            kind = 'physical' if span['physical'] else 'compute' if span['kind'] in ('PERCEIVE', 'GRASP_PLAN') else 'capture' if span['kind'] == 'CAPTURE' else 'plan'
            lane = 0 if kind == 'physical' else 2 if kind == 'compute' else 1
            ax.broken_barh([(span['start_s'], span['duration_s'])], (lane-.3, .6),
                           facecolors=colors[kind], edgecolors='white', linewidth=.2)
        for event in result['event_releases']:
            if event.get('target') == 'blue_cube' and event.get('producer') != 'initial':
                ax.axvline(event['wall_s'], color='#c44e52', linestyle='--')
                ax.text(event['wall_s'], 2.4, 'B visible + stable', fontsize=8, color='#c44e52')
                break  # Show the release enabling B's first capture, not its later placement.
        ax.set_yticks([0, 1, 2], ['physical residency', 'owner capture / plan', 'CPU worker'])
        ax.set_ylim(-.5, 2.7)
        ax.set_title(f'{result["mode"]}: wall={result["wall_duration_s"]:.3f}s, simulated={result["sim_duration_s"]:.2f}s', loc='left')
        ax.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Host wall-clock seconds (inclusive operation residency; not FR3 time)')
    fig.legend([Patch(color=v) for v in colors.values()], list(colors), loc='lower center', ncol=4)
    fig.suptitle('Actual MuJoCo contact execution / PMIR scheduler timeline\n'
                 f'{result["scenario"]}; clock={result["clock"]}; '
                 f'injected latency={result["injected_delay_per_perceive_s"]:.3f}s per perception')
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out/'timeline.svg')
    fig.savefig(out/'timeline.png', dpi=130)
    plt.close(fig)



def draw_cost_timeline(directories, out):
    colors = {'physics_step': '#8073ac', 'robot_control': '#b2abd2',
              'image_acquire_format': '#1b9e77', 'image_snapshot_copy': '#66c2a5',
              'state_validation': '#e6ab02', 'scheduler_decision': '#e7298a',
              'compute_wait_poll': '#bababa', 'logging': '#a6761d',
              'runtime_bookkeeping': '#80b1d3', 'foreground_plan': '#fd8d3c',
              'profiling_analysis': '#444444', 'shutdown_wait': '#888888'}
    fig, axes = plt.subplots(len(directories), 1, figsize=(14, 2.8*len(directories)), sharex=True)
    for ax, directory in zip(axes, directories):
        result = json.loads((directory/'result.json').read_text())
        profile = json.loads((directory/'cost_profile.json').read_text())
        for category, color in colors.items():
            intervals = [(a, b-a) for a, b, key in profile['intervals'] if key == category]
            ax.broken_barh(intervals, (-.25, .5), facecolors=color, rasterized=True)
        workers = [s for s in result['spans'] if s['kind'] in ('PERCEIVE', 'GRASP_PLAN')]
        for span in workers:
            ax.broken_barh([(span['start_s'], span['duration_s'])], (.75, .5),
                          facecolors='#d73027' if span.get('invalidated') else '#2166ac')
            interval = span.get('components', {}).get('injected_wait')
            if interval:
                ax.broken_barh([(interval[0], interval[1]-interval[0])], (.92, .16), facecolors='#92c5de')
        for window in profile['wait_windows']:
            for job in window['jobs']:
                for span in workers:
                    if (span['node'], span['attempt']) != (job['node'], job['attempt']):
                        continue
                    start, end = max(window['start_s'], span['start_s']), min(window['end_s'], span['start_s']+span['duration_s'])
                    if end > start:
                        ax.broken_barh([(start, end-start)], (1.5, .2), facecolors='#222222')
        ax.set_yticks([0, 1, 1.6], ['exclusive owner', 'worker / injected wait', 'exposed compute wait'])
        ax.set_ylim(-.5, 1.95)
        ax.set_title(f'{result["mode"]}: wall {result["wall_duration_s"]:.3f}s; '
                     f'exposed compute {1000*result["exposed_critical_compute_s"]:.1f}ms; '
                     f'invalid attempts {result["invalidated_attempts"]}', loc='left', fontsize=10)
        ax.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Host wall seconds; exclusive foreground accounting, overlapping worker lane')
    fig.legend([Patch(color=color) for color in colors.values()], list(colors), loc='lower center', ncol=4, fontsize=8)
    fig.suptitle(f'{result["scenario"]}: {1000*result["injected_delay_per_perceive_s"]:g}ms artificial delay per PERCEIVE\n'
                 'Red worker = discarded hidden-target perception; pale blue = injected wait; no real model/robot')
    fig.tight_layout(rect=(0, .12, 1, .93))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'cost_timeline.{extension}', dpi=150)
    plt.close(fig)


def draw_sweep(run_dir, out):
    comparison = json.loads((run_dir/'comparison.json').read_text())
    groups, effects = comparison['groups'], comparison['effects']
    scenarios = sorted({g['scenario'] for g in groups})
    styles = {'serial': '#666666', 'naive': '#d95f02', 'dynamic': '#1b9e77'}
    fig, axes = plt.subplots(2, len(scenarios), figsize=(12, 8))
    for column, scenario in enumerate(scenarios):
        for mode, color in styles.items():
            subset = [g for g in groups if g['scenario'] == scenario and g['mode'] == mode]
            axes[0, column].plot([g['delay_ms'] for g in subset], [g['median_wall_s'] for g in subset],
                                  'o-', color=color, label=mode)
            subset = [g for g in effects if g['scenario'] == scenario and g['mode'] == mode]
            if subset:
                means = np.array([g['median_saved_s'] for g in subset])*1000
                limits = np.array([g['median_saved_bootstrap_95ci_s'] for g in subset])*1000
                axes[1, column].errorbar([g['delay_ms'] for g in subset], means,
                    yerr=np.maximum(0, np.stack((means-limits[:, 0], limits[:, 1]-means))),
                    marker='o', color=color, label=mode, capsize=4)
        axes[0, column].set_title(scenario)
        axes[0, column].set_ylabel('Median task wall time (s)')
        axes[1, column].set_ylabel('Paired Serial minus candidate (ms)')
        axes[1, column].axhline(0, color='#666666', linewidth=.7)
        for row in range(2):
            axes[row, column].set_xlabel('Injected latency per perception (ms)')
            axes[row, column].set_xticks([0, 50, 100, 200, 500])
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend()
    fig.suptitle('Fixed-seed latency sweep: actual simulation + artificial computation wait\n'
                 'Six paired repeats; 95% paired-median bootstrap intervals are exploratory')
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'latency_sweep.{extension}', dpi=150)
    plt.close(fig)

    categories = {'Physics + controller': ['physics_step', 'robot_control'],
                  'Capture + format + copy': ['image_acquire_format', 'image_snapshot_copy'],
                  'State validation': ['state_validation'], 'Scheduling': ['scheduler_decision'],
                  'Compute wait / poll': ['compute_wait_poll'], 'Logging': ['logging'],
                  'Other + instrumentation': ['runtime_bookkeeping', 'foreground_plan', 'profiling_analysis', 'shutdown_wait']}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        subset = [g for delay in (0, 200, 500) for mode in styles for g in groups
                  if g['scenario'] == scenario and g['delay_ms'] == delay and g['mode'] == mode]
        bottom = np.zeros(len(subset))
        for category, names in categories.items():
            values = np.array([sum(g['mean_foreground_cost_s'].get(k, 0) for k in names) for g in subset])
            ax.bar(np.arange(len(subset)), values, bottom=bottom, label=category)
            bottom += values
        ax.set_xticks(np.arange(len(subset)), [f'{g["mode"][:3]}\n{g["delay_ms"]:g}ms' for g in subset], fontsize=8)
        ax.set_title(scenario)
        ax.set_ylabel('Mean exclusive foreground seconds (sum = mean wall)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Measured time decomposition; background computation overlaps these bars')
    fig.tight_layout(rect=(0, .16, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'time_decomposition.{extension}', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 4))
    for ax, scenario in zip(axes, scenarios):
        for mode, color in list(styles.items())[1:]:
            subset = [g for g in groups if g['scenario'] == scenario and g['mode'] == mode]
            ax.plot([g['delay_ms'] for g in subset], [1000*g['median_wasted_compute_wall_s'] for g in subset],
                    'o-', color=color, label=mode)
        ax.set_title(scenario)
        ax.set_xlabel('Injected latency per perception (ms)')
        ax.set_ylabel('Discarded computation wall time (ms)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Avoidable hidden-target computation; discarded time can itself overlap motion')
    fig.tight_layout(rect=(0, 0, 1, .92))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'invalid_computation.{extension}', dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenario', default='occluded')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--delay-ms', type=float)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.run_dir/f'measured_{args.repeat:02d}'/f'seed_{args.seed}'/args.scenario
    if args.delay_ms is not None:
        root = root/f'delay_{args.delay_ms:g}ms'
    directories = [root/mode for mode in ('serial', 'naive', 'dynamic') if (root/mode).exists()]
    draw_graph(json.loads((directories[-1]/'graph_final.json').read_text()), args.output_dir)
    draw_timeline(directories, args.output_dir)
    if (directories[-1]/'cost_profile.json').exists():
        draw_cost_timeline(directories, args.output_dir)
    if args.sweep:
        draw_sweep(args.run_dir, args.output_dir)
    result = json.loads((directories[-1]/'result.json').read_text())
    reveal = next((e for e in result['event_releases'] if e.get('target') == 'blue_cube' and e['producer'] != 'initial'), None)
    if reveal:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        files = ['before.png', reveal['frame_file'], 'after.png']
        labels = ['Initial: B fully occluded', 'After lifting A: B visible', 'Both boxes released in goal regions']
        for ax, file, title in zip(axes, files, labels):
            ax.imshow(plt.imread(directories[-1]/file))
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(args.output_dir/'visibility_evidence.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
