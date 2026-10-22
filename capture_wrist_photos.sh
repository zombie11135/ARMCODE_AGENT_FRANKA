#!/usr/bin/env bash
set -euo pipefail
CAPTURE_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$CAPTURE_PROJECT_DIR"
source scripts/load_local_env.sh
CAPTURE_PYTHON="${ARM_AGENT_PYTHON:-${FR3_AGENT_PYTHON:-python3}}"
exec "$CAPTURE_PYTHON" -u -m hardware.wrist_raw_capture "$@"
