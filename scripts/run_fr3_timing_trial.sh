#!/usr/bin/env bash
# One operator-started trial; never loops or resets objects automatically.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash scripts/run_fr3_timing_trial.sh TRIAL [existing agent flags...]\n' >&2
  printf 'Only --execute_robot enables real motion, as in the existing agent.\n' >&2
  exit 2
fi
trial="$1"
shift
if [[ ! "$trial" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'TRIAL must contain only letters, digits, underscore or hyphen.\n' >&2
  exit 2
fi
export ARMCODE_TIMING_EXPERIMENT="${ARMCODE_TIMING_EXPERIMENT:-fr3_two_object_baseline_v1}"
export ARMCODE_TIMING_TRIAL="$trial"
export ARMCODE_TIMING_LAYOUT="${ARMCODE_TIMING_LAYOUT:-table_v1}"
# Trial label + random suffix avoids carrying a previous task's completion/failure state.
mkdir -p runs
trial_dir="$(mktemp -d "$PWD/runs/fr3-timing-${trial}-XXXXXXXX")"
printf '[baseline] trial=%s; artifacts=%s\n' "$trial" "$trial_dir"
exec bash run_fr3_agent.sh \
  --goal '将西瓜霜放到蓝色盘子中，然后将抗病毒口服液放到粉色盘子中' \
  --local_vlm qwen3vl-8b-awq --use_llm_planner --use-sam-for-grasp \
  --plan_json "$trial_dir/plan_graph.json" --plan_md "$trial_dir/plan_graph.md" \
  --replan_json "$trial_dir/replan_graph.json" --replan_md "$trial_dir/replan_graph.md" \
  --execution_state_path "$trial_dir/execution_state.json" \
  --failure_report_path "$trial_dir/failure_report.json" \
  --simple_memory_path "$trial_dir/simple_execution_memory.json" \
  --generated_code "$trial_dir/actions.py" "$@"
