#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_ROOT"
TEST_PYTHON="${OCCLUSION_CAMERA_PYTHON:-/home/seur/data/wpy/conda_envs/env_armcode_agent/bin/python}"
if [[ ! -x "$TEST_PYTHON" ]]; then
  printf 'Set OCCLUSION_CAMERA_PYTHON to a Python with RealSense/OpenCV/SAM3 installed.\n' >&2
  exit 1
fi
TEST_COMMAND="${1:-capture}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TEST_COMMAND" in
  capture) exec "$TEST_PYTHON" -u -m occlusion_reasoning.camera_only "$@" ;;
  analyze) exec "$TEST_PYTHON" -u -m occlusion_reasoning.real_single_view "$@" ;;
  fuse) exec "$TEST_PYTHON" -u -m occlusion_reasoning.real_fusion "$@" ;;
  *) printf 'Usage: bash run_occlusion_camera_test.sh {capture|analyze|fuse} [options]\n' >&2; exit 2 ;;
esac
