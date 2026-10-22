#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/load_local_env.sh
FR3_AGENT_PYTHON="${FR3_AGENT_PYTHON:-python3}"
if [[ $# -eq 0 ]]; then
  set -- collect
fi
exec "$FR3_AGENT_PYTHON" -u -m hardware.eye_to_hand_capture "$@"
