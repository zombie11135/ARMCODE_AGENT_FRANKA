"""Replay planning, recovery and labelled images through VLMs; never run devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from types import SimpleNamespace

from agents.occlusion_recovery_planner import OcclusionRecoveryPlanner, proposal_from_choice, validate_proposal
from agents.plan_graph import PlanGraph
from agents.planning_agent import PlanningAgent
from agents.simple_code_generator import SimpleCodeGenerator
from runtime.local_vlm import ROOT, PROFILES, running_models, start_server, stop_process
from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient


def gpu_used_mib():
    try:
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True, timeout=3)
        return float(value.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class MemorySamples:
    def __enter__(self):
        self.values = []
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.finished.is_set():
            value = gpu_used_mib()
            if value is not None:
                self.values.append(value)
            self.finished.wait(.25)

    def __exit__(self, *args):
        self.finished.set()
        self.thread.join(timeout=4)


class MeasuredClient(OpenAIChatGPTVLMClient):
    def _post(self, payload):
        self.last_metrics = {}
        with MemorySamples() as samples:
            started = time.perf_counter()
            try:
                response = super()._post(payload)
            finally:
                self.last_metrics = dict(http_wall_s=time.perf_counter()-started,
                    device_peak_used_mib=max(samples.values) if samples.values else None)
        if response.status_code == 200:
            doc = response.json()
            self.last_metrics.update(usage=doc.get('usage'),
                finish_reason=doc.get('choices', [{}])[0].get('finish_reason'))
        return response


def parse_json(text):
    return PlanningAgent._extract_json(text)


def expected_recovery_actions(snapshot):
    menu = [a['action_id'] for a in snapshot['allowed_actions']]
    clear = [a for a in menu if a.startswith('clear:')]
    observe = [a for a in menu if a.startswith('observe:')]
    return clear or observe or ['pause']


def build_cases(*, recovery_dir=None, visual_manifest=None, recovery_limit=24):
    """Keep historical action agreement separate from policy compliance/ground truth."""
    cases = []
    plan_examples = [
        ('将西瓜霜放到蓝色盘子中', [('grasp', '西瓜霜'), ('place', 'blue tray')]),
        ('将抗病毒口服液放到粉色盘子中', [('grasp', '抗病毒口服液'), ('place', 'pink tray')]),
        ('将西瓜霜放到蓝色盘子中，将抗病毒口服液放到粉色盘子中',
         [('grasp', '西瓜霜'), ('place', 'blue tray'), ('grasp', '抗病毒口服液'), ('place', 'pink tray')]),
        ('把西瓜霜和抗病毒口服液都放进蓝色盘子',
         [('grasp', '西瓜霜'), ('place', 'blue tray'), ('grasp', '抗病毒口服液'), ('place', 'blue tray')]),
        ('先把抗病毒口服液放进蓝色盘子，再把西瓜霜放进粉色盘子',
         [('grasp', '抗病毒口服液'), ('place', 'blue tray'), ('grasp', '西瓜霜'), ('place', 'pink tray')]),
    ]
    planner = PlanningAgent(context=SimpleNamespace(object_references=None, target_name_bindings={}))
    for i, (goal, actions) in enumerate(plan_examples):
        cases.append(dict(case_id=f'plan_{i}', kind='plan', goal=goal,
            prompt=planner._build_compact_llm_prompt(goal, '', []), max_tokens=2048,
            plan_format='compact_actions_v1',
            expected_actions=[list(a) for a in actions], label_source='explicit_user_goal'))
    seen = set()
    paths = sorted(Path(recovery_dir or ROOT/'runs/current_capture/occlusion_recovery_llm').glob('*.json'))
    # Round-robin menu types so a short run includes clear, observe and pause.
    groups = {}
    for path in paths:
        doc = json.loads(path.read_text())
        snapshot = doc.get('snapshot')
        if not snapshot or snapshot.get('snapshot_id') in seen:
            continue
        seen.add(snapshot['snapshot_id'])
        expected = expected_recovery_actions(snapshot)
        row = dict(case_id='recovery_'+path.stem, kind='recovery', snapshot=snapshot,
            prompt=OcclusionRecoveryPlanner.build_prompt(snapshot, compact=True), max_tokens=256,
            recovery_format='menu_index_v1',
            expected_action_ids=expected, source=str(path),
            historical_action_id=doc.get('decision', {}).get('action_id'),
            label_source='current_prompt_action_preference_not_physical_ground_truth')
        groups.setdefault(expected[0].split(':')[0], []).append(row)
    recovery = []
    while groups and len(recovery) < recovery_limit:
        for key in list(groups):
            if len(recovery) >= recovery_limit:
                break
            recovery.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    cases.extend(recovery)
    if visual_manifest:
        manifest = json.loads(Path(visual_manifest).read_text())
        for row in manifest:
            path = Path(row['image_path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError(f'Visual case image changed: {row["case_id"]}')
            cases.append(dict(row, kind='vision', max_tokens=500,
                label_source='visually_reviewed_saved_image_not_robot_outcome'))
    return cases


def score_case(case, raw):
    parsed = parse_json(raw)
    if case['kind'] == 'plan':
        graph = (PlanningAgent._compact_actions_to_graph(parsed, case['goal'], [])
                 if case.get('plan_format') == 'compact_actions_v1' else PlanGraph.from_dict(parsed))
        actions = [[a.op, a.target] for a in SimpleCodeGenerator().build_actions(graph)]
        manipulation = [a for a in actions if a[0] in ('grasp', 'place')]
        return dict(pass_check=manipulation == case['expected_actions'], actions=actions,
                    minimal_action_match=actions == case['expected_actions'],
                    extra_observation_count=sum(a[0] == 'observe' for a in actions),
                    metric='raw_plan_action_match_without_heuristic_fallback')
    if case['kind'] == 'recovery':
        if case.get('recovery_format') == 'menu_index_v1':
            parsed = proposal_from_choice(parsed, case['snapshot'])
        decision = validate_proposal(parsed, case['snapshot'])
        return dict(pass_check=decision['action_id'] in case['expected_action_ids'],
            legal_action=True, action_id=decision['action_id'],
            historical_agreement=decision['action_id'] == case.get('historical_action_id'),
            metric='menu_legality_and_prompt_preference_not_task_success')
    expected = case.get('expected', {})
    return dict(pass_check=all(type(parsed.get(k)) is type(v) and parsed[k] == v for k, v in expected.items()),
                prediction=parsed, metric='reviewed_image_field_match')


def run_cases(client, cases, output, *, repeats=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclude the first model request from reported warm-request latency.
    warm_started = time.perf_counter()
    client.chat('只回复 READY。', max_tokens=8, temperature=0.)
    vision = next((case for case in cases if case['kind'] == 'vision'), None)
    if vision:
        client.describe_images([vision['image_path']], '只回复 READY。', max_tokens=8, temperature=0.)
    warmup_s = time.perf_counter()-warm_started
    records = []
    for repeat in range(repeats):
        for case in cases:
            record = dict(case_id=case['case_id'], kind=case['kind'], repeat=repeat,
                          hardware_accessed=False, label_source=case['label_source'])
            started = time.perf_counter()
            try:
                if case['kind'] == 'vision':
                    raw = client.describe_images([case['image_path']], case['prompt'],
                        max_tokens=case['max_tokens'], temperature=0., response_format=case.get('response_format'))
                else:
                    raw = client.chat(case['prompt'], max_tokens=case['max_tokens'], temperature=0.)
                record['raw_response'] = raw
                try:
                    json.loads(raw)
                    record['strict_json'] = True
                except json.JSONDecodeError:
                    record['strict_json'] = False
                record.update(score_case(case, raw))
            except Exception as error:
                record.update(pass_check=False, error_type=type(error).__name__)
            record['call_wall_s'] = time.perf_counter()-started
            record.update(getattr(client, 'last_metrics', {}))
            records.append(record)
            with (output/'records.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            print(client.model_name, case['case_id'], record['pass_check'],
                  round(record['call_wall_s'], 3), flush=True)
    summary = summarize(records)
    summary.update(model=client.model_name, warmup_s=warmup_s,
        hardware_accessed=False, cold_start_excluded=True,
        memory_measurement='whole-device nvidia-smi samples every 250ms; brief peaks can be missed')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize(records):
    result = dict(case_count=len(records), kinds={})
    for kind in sorted({r['kind'] for r in records}):
        rows = [r for r in records if r['kind'] == kind]
        latency = sorted(r['call_wall_s'] for r in rows)
        result['kinds'][kind] = dict(count=len(rows), passed=sum(r['pass_check'] for r in rows),
            strict_json_count=sum(r.get('strict_json', False) for r in rows),
            median_s=statistics.median(latency), p95_s=latency[max(0, int(.95*len(latency)+.999)-1)],
            total_tokens=sum((r.get('usage') or {}).get('completion_tokens', 0) for r in rows))
    peaks = [r['device_peak_used_mib'] for r in records if r.get('device_peak_used_mib') is not None]
    result['device_peak_used_mib'] = max(peaks) if peaks else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--gpu-fraction', type=float, default=.45)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visual-manifest', type=Path)
    parser.add_argument('--recovery-dir', type=Path)
    parser.add_argument('--recovery-limit', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--include-remote', action='store_true')
    args = parser.parse_args()
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        value = os.environ.get(key, '')
        if value.lower().startswith('socks://'):
            os.environ[key] = 'socks5h://'+value[len('socks://'):]
    if args.output.exists():
        parser.error('Choose a new output directory; previous measurements are retained')
    if running_models(args.port):
        parser.error('Benchmark needs an unused port; do not stop an unrelated server')
    cases = build_cases(recovery_dir=args.recovery_dir, visual_manifest=args.visual_manifest,
                       recovery_limit=args.recovery_limit)
    args.output.mkdir(parents=True)
    (args.output/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    summaries = []
    if args.include_remote:
        base = os.environ['ARM_AGENT_API_BASE_URL']
        model = os.environ['ARM_AGENT_API_MODEL']
        client = MeasuredClient(base, os.getenv('ARM_AGENT_API_KEY', ''), model, timeout=120)
        summaries.append(run_cases(client, cases, args.output/'remote', repeats=args.repeats))
    for profile in args.models:
        started = time.perf_counter()
        process = start_server(profile, port=args.port, gpu_fraction=args.gpu_fraction)
        try:
            startup_s = time.perf_counter()-started
            client = MeasuredClient(f'http://127.0.0.1:{args.port}/v1', 'local', profile, timeout=120)
            summary = run_cases(client, cases, args.output/profile, repeats=args.repeats)
            summary.update(startup_s=startup_s, profile=PROFILES[profile], gpu_fraction=args.gpu_fraction)
            summaries.append(summary)
            (args.output/'comparison.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
        finally:
            stop_process(process)
            time.sleep(2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
