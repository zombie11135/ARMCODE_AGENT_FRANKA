"""Export a fixed-frame scheduler timeline from saved benchmark reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


STAGES = {
    '子任务代码生成': ('code generation', '#377eb8'),
    '模拟容器预观察': ('synthetic containers', '#ff7f00'),
    '首帧后台感知总计': ('synthetic perception', '#4daf4a'),
    '模拟子任务执行': ('synthetic execution', '#984ea3'),
    '等待后台子任务代码': ('join code', '#a65628'),
    '等待后台感知结果': ('join perception', '#e41a1c'),
    '等待后台代码线程退出': ('shutdown wait', '#666666'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--scenario', choices=['success', 'repair_first', 'stop_first'], default='repair_first')
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, variant in zip(axes, ('serial', 'perception', 'policy', 'both')):
        root = args.run_dir/f'measured_{args.repeat:02d}'/args.scenario/variant
        timing = json.loads((root/'task_timing.json').read_text())
        trace = json.loads((root/'hybrid_trace.json').read_text())
        jobs = trace['events'][-1]['prefetch_jobs']
        wasted = {(j['node'], j['revision']) for j in jobs if j['outcome'] != 'used'}
        for span in timing['spans']:
            if span['name'] not in STAGES:
                continue
            _, color = STAGES[span['name']]
            attrs = span['attributes']
            is_wasted = span['name'] == '子任务代码生成' and span['background'] and (
                attrs['node'], attrs['revision']) in wasted
            lane = 1 if span['name'] == '首帧后台感知总计' else 2 if span['background'] else 0
            ax.broken_barh([(span['start_s'], span['duration_s'])], (lane-.3, .6),
                          facecolors=color, edgecolors='black', linewidth=.3, hatch='////' if is_wasted else None)
        ax.axvline(timing['duration_s'], color='#444444', linestyle=':', linewidth=1)
        ax.set_yticks([0, 1, 2], ['owner', 'perception', 'policy'])
        ax.set_ylim(2.5, -.6)
        ax.set_title(f'{variant} | wall {timing["duration_s"]:.3f} s', loc='left', fontsize=10)
        ax.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Seconds since timing entry (inclusive spans; do not add bars)')
    handles = [Patch(facecolor=color, label=label) for label, color in STAGES.values()]
    handles.append(Patch(facecolor='white', edgecolor='black', hatch='////', label='unused code work'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle(f'{args.run_dir.name} / {args.scenario} / repeat {args.repeat}\n'
                 'Injected workload and simulated feedback; no physical robot or perception model', fontsize=12)
    fig.tight_layout(rect=(0, .08, 1, .93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
