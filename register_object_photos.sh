#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
source scripts/load_local_env.sh
ARM_AGENT_PYTHON="${ARM_AGENT_PYTHON:-python3}"
exec "$ARM_AGENT_PYTHON" -u -m hardware.reference_capture "$@"
