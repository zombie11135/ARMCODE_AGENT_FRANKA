"""Call the configured LLM on a saved failure; never open cameras or robot devices."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.occlusion_recovery_planner import OcclusionRecoveryPlanner, build_recovery_snapshot
from services.object_reference_service import atomic_json
from services.wrist_occlusion_service import choose_blocker, sha, verify_binding


def replay_inputs(failure_path, trace_path):
    failure = json.loads(Path(failure_path).read_text())
    graph_path = failure['evidence']['graph_report']
    graph = json.loads(Path(graph_path).read_text())
    graph['report_sha256'] = sha(graph_path)
    verify_binding(graph['source_binding'])
    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
    end = next(i for i in range(len(rows)-1, -1, -1)
               if rows[i].get('event') == 'reobserve' and rows[i].get('graph_report') == graph_path)
    clear = next(row for row in reversed(rows[:end]) if row.get('event') == 'clear_result')
    if clear.get('success') is not True:
        raise ValueError('This replay requires a completed clear before the failed re-observation')
    slot = clear['buffer_slot']
    if slot != 0:
        raise ValueError('Replay requires complete slot history when more than one slot was used')
    result = clear['result']
    history = [dict(event='clear', stage='finished', target=clear['target'],
        source_report=None, node_id=result['blocker']['node_id'],
        bbox=result['blocker']['bbox'], buffer_slot=slot,
        buffer_pose_mmrad=result['buffer_pose_mmrad'], movement_completed=True,
        visual_effect_verified=False)]
    return failure, graph, [result['buffer_pose_mmrad']], history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure-report', type=Path, default=ROOT/'runs/current_plan/failure_report.json')
    parser.add_argument('--trace', type=Path, default=ROOT/'runs/current_plan/occlusion_graph_trace.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clear-limit', type=int, default=1, help='Replay budget assumption; original CLI is not recorded in graph')
    parser.add_argument('--include-hypothetical-view', action='store_true',
                        help='Also test a hypothetical named viewpoint; no real pose is configured or executed')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; historical replay results are not overwritten')
    args.output.mkdir(parents=True)
    for name in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
        value = os.environ.get(name, '')
        if value.lower().startswith('socks://'):
            os.environ[name] = 'socks5h://'+value[len('socks://'):]
    from services.openai_chatgpt_vlm_client import OpenAIChatGPTVLMClient
    base = (os.getenv('PLANNER_LLM_FALLBACK_BASE_URL') or os.getenv('PLANNER_API_BASE_URL')
            or os.getenv('ARM_AGENT_API_BASE_URL'))
    key = (os.getenv('PLANNER_LLM_FALLBACK_API_KEY') or os.getenv('PLANNER_API_KEY')
           or os.getenv('ARM_AGENT_API_KEY'))
    model = (os.getenv('PLANNER_LLM_FALLBACK_MODEL') or os.getenv('PLANNER_API_MODEL')
             or os.getenv('ARM_AGENT_API_MODEL'))
    if not base or not model:
        raise ValueError('Source config/local.env or provide the configured planner environment first')
    planner = OcclusionRecoveryPlanner(OpenAIChatGPTVLMClient(base, key or '', model, timeout=30),
                                       args.output/'model_calls')
    failure, graph, buffers, history = replay_inputs(args.failure_report, args.trace)
    atomic_json(args.output/'failure_report.json', failure)
    (args.output/'source_graph.original.json').write_bytes(Path(graph['report_path']).read_bytes())
    decision = choose_blocker(graph, allow_search=True)
    runs = []
    cases = [('actual_failure', 0, {}), ('observation_budget_exhausted', 2, {})]
    if args.include_hypothetical_view:
        cases.insert(1, ('hypothetical_registered_other_view', 0, {'right_oblique': None}))
    for name, used, views in cases:
        snapshot = build_recovery_snapshot(graph=graph, decision=decision,
            target=failure['medicine_query'], goal=failure['goal'], clear_count=1,
            clear_limit=args.clear_limit, buffers=buffers, slots_used=1,
            observation_count=used, observation_limit=2, history=history, views=views)
        choice = planner.decide(snapshot)
        if choice['action'] == 'clear':
            raise AssertionError('The recorded clear used the only buffer slot; no further clear is allowed')
        if used == 2 and choice['action'] != 'pause':
            raise AssertionError('No observation budget or clear action remains')
        record = dict(case=name, snapshot=snapshot, choice=choice,
            input_kind=('actual_saved_failure_with_hypothetical_tool_availability' if views else
                        'same_actual_saved_failure_with_explicit_replay_budgets'),
            hardware_accessed=False, action_executed=False)
        atomic_json(args.output/(name+'.json'), record)
        runs.append(record)
        print(name, choice['action_id'], choice['reason'], flush=True)
    summary = dict(model=model, case_count=len(runs), clear_limit_assumed=args.clear_limit,
        hardware_accessed=False, cameras_opened=False, robot_initialized=False,
        model_choices=[dict(case=r['case'], action_id=r['choice']['action_id'], reason=r['choice']['reason']) for r in runs],
        note='Real LLM calls on historical evidence. Budgets are replay parameters, not a reconstructed original CLI. No motion or new observations executed.')
    atomic_json(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Replay failed:', type(error).__name__, '(inspect model_calls for validation evidence)', file=sys.stderr)
        raise SystemExit(1)
