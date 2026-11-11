"""Run the two-object task in two fresh processes and compare physics/RGB-D."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from .demo import ROOT, write_json


def smoke(output, seed=7):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('run_a', 'run_b'):
        subprocess.run([sys.executable, '-m', 'simulation.robosuite_backend.demo',
                        '--seed', str(seed), '--output-dir', str(output / name)], check=True, cwd=ROOT)
    first, second = output / 'run_a', output / 'run_b'
    reports = [json.loads((p / 'summary.json').read_text()) for p in (first, second)]
    assert all(r['success'] and len(r['results']) == 2 for r in reports)
    for report in reports:
        assert all(r['success'] and r['data']['lifted'] for r in report['results'])
        assert report['task']['complete']
    for phase in ('before', 'after'):
        with np.load(first / f'{phase}_rgbd.npz') as a, np.load(second / f'{phase}_rgbd.npz') as b:
            assert a['rgb'].shape == (240, 320, 3) and a['rgb'].dtype == np.uint8
            assert a['depth_m'].shape == (240, 320) and a['depth_m'].dtype == np.float32
            assert np.isfinite(a['depth_m']).all() and (a['depth_m'] > 0).all()
            for key in a.files:
                np.testing.assert_allclose(a[key], b[key], atol=1e-7, rtol=0)
    # Wall-clock timing is intentionally excluded from deterministic comparison.
    def without_wall(value):
        if isinstance(value, dict):
            return {k: without_wall(v) for k, v in value.items() if not k.startswith('wall_')}
        if isinstance(value, list):
            return [without_wall(v) for v in value]
        return value
    events = [[without_wall(json.loads(line)) for line in (p / 'events.jsonl').read_text().splitlines()]
              for p in (first, second)]
    assert events[0] == events[1], 'Seeded physics/events differ between processes'
    assert without_wall(reports[0]) == without_wall(reports[1])
    report = dict(success=True, seed=seed, fresh_processes=2, objects_per_run=2,
                  rgbd_atol=1e-7, physics_and_events_identical=True,
                  wall_time_compared=False, output_dir=str(output))
    write_json(output / 'smoke_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / 'runs/robosuite' / ('smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print(json.dumps(smoke(output.resolve(), args.seed), ensure_ascii=False))


if __name__ == '__main__':
    main()
