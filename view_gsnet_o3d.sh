#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/load_local_env.sh
GSNET_PICK_PYTHON="${GSNET_PICK_PYTHON:-python3}"
exec "$GSNET_PICK_PYTHON" -u -m hardware.gsnet_o3d "$@"
