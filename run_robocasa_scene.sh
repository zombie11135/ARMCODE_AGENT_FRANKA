#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ ! -x .venv-robocasa/bin/python ]]; then
    echo "Run: bash scripts/setup_robocasa_scene.sh" >&2
    exit 1
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec .venv-robocasa/bin/python -m robocasa_occlusion.cli "$@"
