#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
command -v uv >/dev/null || { echo "Install uv, then run this script again." >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_dir/dependencies/.uv-cache}"
mkdir -p dependencies

clone_pinned() {
    local url="$1" dir="$2" commit="$3"
    if [[ ! -d "$dir" ]]; then
        git init "$dir"
        git -C "$dir" remote add origin "$url"
        git -C "$dir" fetch --depth 1 origin "$commit"
        git -C "$dir" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || {
        echo "Unexpected source revision in $dir; preserving your checkout." >&2
        exit 1
    }
}

# Latest upstream revisions verified when this scene was built (2026-10-17).
clone_pinned https://github.com/robocasa/robocasa.git dependencies/robocasa365 a9e152e2eea16d342dac40e4577fb4c68b932854
clone_pinned https://github.com/ARISE-Initiative/robosuite.git dependencies/robosuite 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
if [[ ! -x .venv-robocasa/bin/python ]]; then
    uv venv --python 3.11 .venv-robocasa
fi
uv pip install --python .venv-robocasa/bin/python -r requirements-robocasa-scene.txt
uv pip install --python .venv-robocasa/bin/python --no-deps -e dependencies/robosuite -e dependencies/robocasa365
.venv-robocasa/bin/python -m robocasa_occlusion.download_assets
echo "Ready: bash run_robocasa_scene.sh render --all"
