#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/load_local_env.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "${FR3_AGENT_PYTHON:-python3}" -u scripts/check_offline_stack.py \
  --image runs/live_scene_decision/20261022_194928/capture/wrist/color.png "$@"
