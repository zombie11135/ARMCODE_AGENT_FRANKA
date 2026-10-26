#!/usr/bin/env bash
set -euo pipefail
GRIPPER_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$GRIPPER_PROJECT_DIR"
GRIPPER_EXPERIMENT_PYTHON="${ARM_AGENT_PYTHON:-/home/seur/data/wpy/conda_envs/env_armcode_agent/bin/python}"
exec "$GRIPPER_EXPERIMENT_PYTHON" -u -m hardware.gripper_contact_cli "$@"
