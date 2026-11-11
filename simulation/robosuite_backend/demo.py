"""Two-object demonstration; no LLM, camera service, or hardware imports."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform

import numpy as np
from PIL import Image

from .backend import RobosuiteBackend
from .config import SimConfig
from .policy import TWO_OBJECT_POLICY, run_policy

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_observation(directory, name, observation):
    data = observation.data
    camera = data['camera']
    Image.fromarray(camera['rgb']).save(directory / f'{name}_rgb.png')
    np.savez_compressed(directory / f'{name}_rgbd.npz', rgb=camera['rgb'], depth_m=camera['depth_m'],
                        K=camera['K'], T_world_from_camera_optical=camera['T_world_from_camera_optical'])
    metadata = {k: v for k, v in data.items() if k != 'camera'}
    metadata['camera'] = {k: v for k, v in camera.items() if k not in ('rgb', 'depth_m', 'K', 'T_world_from_camera_optical')}
    write_json(directory / f'{name}_state.json', metadata)


def run_demo(output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'config.json', config.to_dict())
    (output_dir / 'policy.py').write_text(TWO_OBJECT_POLICY)
    with RobosuiteBackend(config, log_path=output_dir / 'events.jsonl') as backend:
        save_observation(output_dir, 'before', backend.observe())
        try:
            results = run_policy(backend)
        except Exception as error:
            write_json(output_dir / 'summary.json', dict(success=False, error=str(error), task=backend.task_status()))
            save_observation(output_dir, 'after', backend.observe())
            raise
        save_observation(output_dir, 'after', backend.observe())
        # GL context is current after observe(). Record the actual renderer.
        from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR, GL_VERSION
        renderer = {name: glGetString(key).decode() for name, key in (
            ('renderer', GL_RENDERER), ('vendor', GL_VENDOR), ('version', GL_VERSION))}
        write_json(output_dir / 'manifest.json', dict(
            backend='robosuite', simulated=True, seed=config.seed, config=config.to_dict(),
            python=platform.python_version(), platform=platform.platform(), renderer=renderer,
            mujoco_gl=os.environ.get('MUJOCO_GL'), egl_device_id=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
            robosuite_revision='5ce6643f3092639d08f7b0f90ed1c6a84f50552c',
            packages={d.metadata['Name']: d.version for d in distributions()},
            time_note='sim_duration_s is simulated physics time; wall_duration_s is host runtime, neither is robot execution time'))
        summary = dict(success=backend.task_status()['complete'], simulated=True, seed=config.seed,
                       results=[asdict(result) for result in results], task=backend.task_status())
        write_json(output_dir / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/simulation/robosuite_panda.json')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = SimConfig.load(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    output = args.output_dir or ROOT / 'runs/robosuite' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    summary = run_demo(output, config)
    print(json.dumps(dict(success=summary['success'], seed=config.seed, output_dir=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
