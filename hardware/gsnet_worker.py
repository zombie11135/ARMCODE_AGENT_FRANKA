"""Isolated GSNet worker; shares GraspNet-format filtering with the baseline."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from hardware.graspnet_worker_common import roi_mask, select_top_one, run as run_filtered


def run(request):
    root = Path(request['gsnet_root']).resolve()
    sys.path[:0] = [str(root), str(root / 'utils')]
    import demo_local as gsnet
    from realsense_preview import render
    return run_filtered(request, gsnet, render=render)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()))
