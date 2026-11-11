"""Real-task timing reduction/export. No device or model imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

PHASES = ('CAPTURE', 'PERCEIVE', 'GRASP_PLAN', 'MOTION_PLAN', 'MOVE', 'GRIPPER', 'VERIFY', 'HANDOFF', 'TOTAL')
NAMES = {
    'RGB-D新帧采集': 'CAPTURE',
    '目标身份识别': 'PERCEIVE', 'SAM3分割与遮挡关系构建': 'PERCEIVE',
    '容器检测与坐标计算': 'PERCEIVE',
    '抓取候选服务': 'GRASP_PLAN',
    '夹爪张开': 'GRIPPER', '夹爪闭合': 'GRIPPER',
    '抓取结果实际验证': 'VERIFY', '放置结果实际验证': 'VERIFY',
}
RECOVERY = {'抓取规划重试', '遮挡后二次观察重试', 'Hybrid节点修复', '任务重规划'}

def interval_union(intervals):
    end, total = None, 0.
    for a, b in sorted(intervals):
        total += max(0., b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total

def phases(report):
    spans = report['spans']
    result = {}
    for phase in PHASES[:-2]:
        rows = [s for s in spans if s.get('attributes', {}).get('phase', NAMES.get(s['name'])) == phase]
        aligned = all(s.get('start_s') is not None for s in rows)
        result[phase] = dict(calls=len(rows), failed_calls=sum(s['status'] != 'ok' for s in rows),
            coverage='measured' if rows else 'not_observed',
            work_s=sum(s['duration_s'] for s in rows) if rows else None,
            wall_s=interval_union((s['start_s'], s['start_s']+s['duration_s']) for s in rows) if rows and aligned else None)
    # Interrupted/aborted execution has no confirmed stationary boundary. Do not bridge it.
    moves = [s for s in spans if s.get('attributes', {}).get('phase') == 'MOVE']
    moves.sort(key=lambda s: s['attributes']['worker_start_monotonic_s'])
    requests = sorted([s for s in spans if s['name'] == '机械臂调用' and s['detail'] == 'move'],
                      key=lambda s:s['start_s'])
    request_order = {s['id']:i for i,s in enumerate(requests)}
    gaps = []
    for previous, current in zip(moves, moves[1:]):
        if requests and request_order.get(current['parent_id'], -2) != request_order.get(previous['parent_id'], -2)+1:
            continue
        if (previous['status'] == 'ok' and previous.get('start_s') is not None
                and current.get('start_s') is not None):
            start = previous['start_s']+previous['duration_s']
            stop = current['start_s']
            if stop >= start:
                gaps.append(dict(start_s=start, duration_s=stop-start,
                    previous_request=previous['attributes']['request_id'],
                    next_request=current['attributes']['request_id']))
    result['HANDOFF'] = dict(calls=len(gaps), failed_calls=0,
        coverage='measured' if gaps else 'not_observed',
        work_s=sum(g['duration_s'] for g in gaps) if gaps else None,
        wall_s=sum(g['duration_s'] for g in gaps) if gaps else None, intervals=gaps)
    result['TOTAL'] = dict(calls=1, failed_calls=int(report['status'] != 'ok'),
        coverage='measured', wall_s=report['duration_s'], work_s=report['duration_s'])
    return result

def recovery_events(report):
    return [dict(span_id=s['id'], kind=s['name'], status=s['status'], detail=s['detail'])
            for s in report['spans'] if s['name'] in RECOVERY]

def write_trial_csv(report, path):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out, fieldnames=['run_id','status','phase','coverage','calls','failed_calls','wall_s','work_s'])
        writer.writeheader()
        for phase, row in report['phases'].items():
            writer.writerow(dict(run_id=report['run_id'], status=report['status'], phase=phase,
                **{k:v for k,v in row.items() if k != 'intervals'}))

def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered)-1)*q
    low = int(index)
    return ordered[low] + (ordered[min(low+1,len(ordered)-1)]-ordered[low])*(index-low)

def export(paths, output, experiment, include_dry_run=False):
    reports, seen = [], set()
    for path in paths:
        report=json.loads(Path(path).read_text())
        metadata=report.get('metadata', {})
        if metadata.get('experiment') != experiment or report.get('run_id') in seen:
            continue
        if not include_dry_run and (metadata.get('execute_robot') is not True or
                metadata.get('hardware_profile', {}).get('arm', {}).get('stack', metadata.get('configured_stack')) != 'real'):
            continue
        seen.add(report['run_id'])
        report['source_path']=str(path)
        reports.append(report)
    if not reports:
        raise ValueError('No matching trials; use the exact ARMCODE_TIMING_EXPERIMENT label. No synthetic data are inserted.')
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, groups = [], {}
    for report in reports:
        meta=report['metadata']
        signature=dict(goal=report['goal'], execute_robot=meta.get('execute_robot'),
            conditions=meta.get('conditions', {}), layout=meta.get('layout'),
            hardware_sha256=meta.get('hardware_sha256'), implementation_sha256=meta.get('implementation_sha256'))
        group=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
        events=report.get('recovery_events', [])
        outcome=report['status'] + ('_with_recovery' if events else '_without_recovery')
        groups[group]=signature
        for phase, data in report.get('phases', {}).items():
            rows.append(dict(run_id=report['run_id'],trial=meta.get('trial',''),group=group,
                outcome=outcome,recovery_events=len(events),phase=phase,coverage=data['coverage'],
                wall_s=data['wall_s'],work_s=data['work_s'],calls=data['calls'],failed_calls=data['failed_calls'],
                source_path=report['source_path']))
    summary=[]
    for group in groups:
        for outcome in sorted({r['outcome'] for r in rows if r['group']==group}):
            for phase in PHASES:
                selected=[r for r in rows if (r['group'],r['outcome'],r['phase'])==(group,outcome,phase)]
                values=[r['wall_s'] for r in selected if r['wall_s'] is not None]
                summary.append(dict(group=group,outcome=outcome,phase=phase,trials=len(selected),measured=len(values),
                    mean_s=mean(values) if values else None,p50_s=quantile(values,.5) if values else None,
                    p90_s=quantile(values,.9) if values else None))
    for filename, data in [('trials.csv',rows),('summary.csv',summary)]:
        if data:
            with (output/filename).open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    payload=dict(experiment=experiment,trial_count=len(reports),groups=groups,summary=summary,
        note='Per-trial wall-time union; phases and HANDOFF overlap, never sum to TOTAL. '
             'Missing measurements are null, not zero. Status is software outcome, not independent physical ground truth.',
        reports=reports)
    (output/'baseline.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    return payload

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Directory containing per-task timing/*.json histories')
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-dry-run',action='store_true')
    args=parser.parse_args()
    paths = sorted(args.directory.glob('*.json')) if args.directory.name == 'timing' else sorted(args.directory.glob('**/timing/*.json'))
    result=export(paths,args.output,args.experiment,args.include_dry_run)
    print(f"Exported {result['trial_count']} trials to {args.output}; {len(result['groups'])} condition group(s)")

if __name__ == '__main__':
    main()
