"""Render a wide Panda view and a GIF of the actual two-object simulation."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .backend import RobosuiteBackend
from .config import SimConfig
from .demo import ROOT, write_json, save_observation
from .policy import run_policy


def render_demo(output, seed=7):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = SimConfig(seed=seed, camera='frontview', image_width=960, image_height=720)
    with RobosuiteBackend(config, log_path=output / 'events.jsonl') as backend:
        sim = backend.env.sim
        camera_id = sim.model.camera_name2id(config.camera)
        eye = np.array([1.3, 1.4, 1.8])
        lookat = np.array([-0.2, 0, 1.0])
        backward = eye - lookat
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        sim.model.cam_pos[camera_id] = eye
        sim.model.cam_quat[camera_id] = Rotation.from_matrix(
            np.column_stack([right, up, backward])).as_quat()[[3, 0, 1, 2]]
        sim.forward()
        save_observation(output, 'before', backend.observe())

        def frame():
            rgb = sim.render(width=640, height=480, camera_name=config.camera)
            return Image.fromarray(np.flipud(rgb).copy())

        frames = [frame()]
        step_records = [dict(step=backend.step_count, sim_time_s=backend.sim_time_s, phase=backend.phase)]
        original_step = backend._step

        def recorded_step(position, gripper):
            original_step(position, gripper)
            if backend.step_count % 6 == 0:
                frames.append(frame())
                step_records.append(dict(step=backend.step_count, sim_time_s=backend.sim_time_s,
                                         phase=backend.phase))

        # Local rendering wrapper only; the existing controller and skill code
        # still performs every physics step, with no changes to its actions.
        backend._step = recorded_step
        results = run_policy(backend)
        save_observation(output, 'after', backend.observe())
        frames.append(frame())
        step_records.append(dict(step=backend.step_count, sim_time_s=backend.sim_time_s, phase=backend.phase))
        frames[0].save(output / 'pick_place.gif', save_all=True, append_images=frames[1:],
                       duration=[800] + [100] * (len(frames) - 2) + [1500], loop=0)
        write_json(output / 'visualization.json', dict(
            success=backend.task_status()['complete'], seed=seed, config=config.to_dict(),
            camera_position_m=eye.tolist(), camera_lookat_m=lookat.tolist(),
            camera_frame='sim_world', frames=step_records, gif='pick_place.gif',
            playback_note='Approximately 3x simulated time; first and last frames held. Not real robot time.',
            lift_heights_m=[result.data['lift_height_m'] for result in results]))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / 'runs/robosuite' / ('visual_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    render_demo(output, args.seed)


if __name__ == '__main__':
    main()
