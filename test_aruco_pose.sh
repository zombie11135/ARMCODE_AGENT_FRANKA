#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/load_local_env.sh
ARUCO_POSE_PYTHON="${FR3_AGENT_PYTHON:-python3}"
exec "$ARUCO_POSE_PYTHON" -u -m hardware.aruco_pose "$@"
