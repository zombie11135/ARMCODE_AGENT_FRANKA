#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/scripts/load_local_env.sh"
ANYGRASP_PYTHON="${ANYGRASP_PYTHON:-python3}"
if [[ -n "${ANYGRASP_COMPAT_LIBS:-}" ]]; then
  export LD_LIBRARY_PATH="$ANYGRASP_COMPAT_LIBS:${LD_LIBRARY_PATH:-}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd -- "$PROJECT_DIR"
exec "$ANYGRASP_PYTHON" -u -m hardware.anygrasp_offline "$@"
