#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_ROOT"
source scripts/load_local_env.sh
TEST_PYTHON="${OCCLUSION_CAMERA_PYTHON:-${FR3_AGENT_PYTHON:-python3}}"
if ! command -v "$TEST_PYTHON" >/dev/null 2>&1; then
  printf 'Set OCCLUSION_CAMERA_PYTHON to a Python with RealSense/OpenCV/SAM3 installed.\n' >&2
  exit 1
fi
TEST_COMMAND="${1:-capture}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TEST_COMMAND" in
  capture) exec "$TEST_PYTHON" -u -m occlusion_reasoning.camera_only "$@" ;;
  capture-base) exec "$TEST_PYTHON" -u -m occlusion_reasoning.robot_pair "$@" ;;
  dual-view) exec "$TEST_PYTHON" -u -m occlusion_reasoning.robot_pair --analyze "$@" ;;
  analyze) exec "$TEST_PYTHON" -u -m occlusion_reasoning.real_single_view "$@" ;;
  fuse) exec "$TEST_PYTHON" -u -m occlusion_reasoning.real_fusion "$@" ;;
  *) printf 'Usage: bash run_occlusion_camera_test.sh {capture|capture-base|dual-view|analyze|fuse} [options]\n' >&2; exit 2 ;;
esac
