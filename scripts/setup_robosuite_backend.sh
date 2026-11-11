#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
sim_python="${SIM_PYTHON:-python3.11}"
sim_env="$project_dir/.venv-robosuite"
"$sim_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
if [[ ! -x "$sim_env/bin/python" ]]; then
    "$sim_python" -m venv --copies "$sim_env"
fi
"$sim_env/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Existing simulation venv must use Python 3.11"'
if [[ -n "${SIM_OFFLINE_SITE:-}" ]]; then
    "$sim_env/bin/python" scripts/robosuite_offline_install.py \
        "$SIM_OFFLINE_SITE" "${SIM_ROBOSUITE_SOURCE:?Set the pinned robosuite source directory}"
else
    "$sim_env/bin/python" -m pip install -r requirements-robosuite-backend.txt
fi
"$sim_env/bin/python" -m pip check
echo 'Ready: .venv-robosuite/bin/python -m simulation.robosuite_backend.demo'
