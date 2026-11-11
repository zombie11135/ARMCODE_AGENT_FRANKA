#!/usr/bin/env bash
set -euo pipefail
VLM_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$VLM_PROJECT_DIR"
source scripts/load_local_env.sh
export ARM_LOCAL_VLM_ROOT="${ARM_LOCAL_VLM_ROOT:-$VLM_PROJECT_DIR/models/local-vlm}"
export ARM_LOCAL_VLM_CACHE="${ARM_LOCAL_VLM_CACHE:-$VLM_PROJECT_DIR/runs/local_vlm/cache}"
export UV_CACHE_DIR="$ARM_LOCAL_VLM_CACHE/uv"
export HF_HOME="$ARM_LOCAL_VLM_CACHE/huggingface"
export TMPDIR="$ARM_LOCAL_VLM_CACHE/tmp"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p -- "$TMPDIR" "$ARM_LOCAL_VLM_ROOT"
VLM_UV="${ARM_LOCAL_VLM_UV:-uv}"
if [[ ! -x .venv-vlm-local/bin/python ]]; then
  "$VLM_UV" venv --python "${ARM_LOCAL_VLM_BASE_PYTHON:-python3.11}" .venv-vlm-local
fi
"$VLM_UV" pip install --python .venv-vlm-local/bin/python -r requirements-vlm-local.txt
for VLM_PROFILE in qwen35-9b-awq qwen3vl-8b-awq; do
  .venv-vlm-local/bin/python -m runtime.local_vlm download --model "$VLM_PROFILE"
done
