"""Synchronous, simulation-only implementation of the restricted skill API."""
from dataclasses import replace
from importlib.metadata import version
import platform
import time

import numpy as np
from scipy.spatial.transform import Rotation

from core.skill_base import SkillResult
from .config import SimConfig
from .disturbances import DisturbanceHook
from .events import EventLog
from .rendering import configure_renderer


class SkillFailure(RuntimeError):
    pass


class RobosuiteBackend:
    """Owns one Panda simulation and an exclusive JSONL log.

    Call from one thread. Object poses are simulator ground truth, not estimated
    perception. Only reset writes object qpos; skills use controller actions.
    """
    def __init__(self, config: SimConfig | None = None, *, log_path,
                 disturbance_hook: DisturbanceHook | None = None):
        self.config = config or SimConfig()
        self.log = EventLog(log_path)
        self.disturbance_hook = disturbance_hook
        self.env = None
        self.episode = -1
        self.step_count = 0
        self.phase = 'initializing'
        try:
            configure_renderer()
            self.reset()
        except Exception:
            self.close()
            raise

    @property
    def sim_time_s(self):
        return float(self.env.sim.data.time) if self.env is not None else 0.0

    def _event(self, event, **payload):
        self.log.write(event, episode=self.episode, step=self.step_count,
                       sim_time_s=self.sim_time_s, **payload)

    def reset(self, seed: int | None = None):
        from .environment import PandaTabletop

        if seed is not None:
            self.config = replace(self.config, seed=seed)
        if self.env is not None:
            self.env.close()
        # Rebuild all controller and RNG state, including gripper command history.
        self.env = PandaTabletop(self.config)
        self.env.reset()
        self.episode += 1
        self.step_count = 0
        self.halted = False
        self.phase = 'reset'
        self.rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 1]))
        self._stable = {name: 0 for name in self.env.objects}
        site = self.env.robots[0].eef_site_id['right']
        self._orientation = Rotation.from_matrix(self.env.sim.data.site_xmat[site].reshape(3, 3)).as_rotvec()
        home = self._eef_position()
        for _ in range(20):
            self._step(home, -1)
        self.phase = 'idle'
        self._event('reset', seed=self.config.seed, config=self.config.to_dict(),
                    versions={name: version(name) for name in ('robosuite', 'mujoco', 'numpy', 'scipy')},
                    python=platform.python_version(), objects=self.object_states(),
                    disturbance_hook=None if self.disturbance_hook is None else type(self.disturbance_hook).__name__)
        return self.robot_state()

    def _eef_position(self):
        robot = self.env.robots[0]
        return self.env.sim.data.site_xpos[robot.eef_site_id['right']].copy()

    def robot_state(self):
        robot, data = self.env.robots[0], self.env.sim.data
        site = robot.eef_site_id['right']
        return dict(
            joint_names=list(robot.robot_model.joints),
            joint_positions_rad=data.qpos[robot._ref_joint_pos_indexes].tolist(),
            joint_velocities_rad_s=data.qvel[robot._ref_joint_vel_indexes].tolist(),
            gripper_positions_m=data.qpos[robot._ref_gripper_joint_pos_indexes['right']].tolist(),
            eef_position_m=data.site_xpos[site].tolist(),
            eef_quaternion_xyzw=Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_quat().tolist(),
            frame='sim_world', phase=self.phase, halted=self.halted,
        )

    def object_states(self):
        data = self.env.sim.data
        states = {}
        for name, obj in self.env.objects.items():
            body = self.env.body_ids[name]
            velocity = np.array(data.get_joint_qvel(obj.joints[0]))
            states[name] = dict(position_m=data.body_xpos[body].tolist(),
                quaternion_xyzw=data.body_xquat[body][[1, 2, 3, 0]].tolist(),
                linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
                angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
                frame='sim_world', source='simulator_ground_truth')
        return states

    def observe(self, target: str | None = None) -> SkillResult:
        from .environment import GOALS
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map)

        if target is not None and target not in self.env.objects and target not in GOALS:
            return self._rejected('observe', 'unknown_target', target=target)
        started = time.perf_counter()
        cfg, sim = self.config, self.env.sim
        rgb, depth_buffer = sim.render(width=cfg.image_width, height=cfg.image_height,
                                       camera_name=cfg.camera, depth=True)
        rgb = np.flipud(rgb).copy()
        depth_m = np.flipud(get_real_depth_map(sim, depth_buffer)).astype(np.float32)
        data = dict(backend='robosuite', simulated=True, target=target,
            episode=self.episode, step=self.step_count, sim_time_s=self.sim_time_s,
            robot=self.robot_state(), objects=self.object_states(),
            goals={name: dict(position_m=[*xy, cfg.table_height], frame='sim_world') for name, xy in GOALS.items()},
            camera=dict(name=cfg.camera, rgb=rgb, depth_m=depth_m, color_order='RGB',
                image_origin='top_left', depth_convention='optical_z_m',
                optical_axes='x_right_y_down_z_forward',
                K=get_camera_intrinsic_matrix(sim, cfg.camera, cfg.image_height, cfg.image_width),
                T_world_from_camera_optical=get_camera_extrinsic_matrix(sim, cfg.camera)),
            task=self.task_status(), wall_duration_s=time.perf_counter() - started)
        self._event('observe', target=target, success=True, image_shape=list(rgb.shape),
                    objects=data['objects'], wall_duration_s=data['wall_duration_s'])
        return SkillResult(True, data, message='Simulation RGB-D and ground-truth state')

    def _placement_metrics(self, name, destination):
        from .environment import GOALS

        obj, cfg = self.env.objects[name], self.config
        state = self.object_states()[name]
        pos = np.array(state['position_m'])
        xy_error = float(np.linalg.norm(pos[:2] - GOALS[destination]))
        z_error = abs(float(pos[2]) - cfg.table_height - cfg.cube_half_size)
        gripper = self.env.robots[0].gripper['right']
        released = not self.env.check_contact(gripper.contact_geoms, obj.contact_geoms)
        on_table = self.env.check_contact(obj.contact_geoms, 'table_collision')
        settled = state['linear_speed_m_s'] < cfg.settle_speed and state['angular_speed_rad_s'] < cfg.settle_angular_speed
        return dict(inside_goal=xy_error < cfg.goal_xy_tolerance,
                    correct_height=z_error < cfg.goal_z_tolerance, released=bool(released),
                    on_table=bool(on_table), settled=bool(settled), xy_error_m=xy_error, z_error_m=z_error)

    @staticmethod
    def _placed(metrics):
        return all(metrics[k] for k in ('inside_goal', 'correct_height', 'released', 'on_table', 'settled'))

    def task_status(self):
        from .environment import TASK

        objects = {}
        for name, goal in TASK:
            metrics = self._placement_metrics(name, goal)
            metrics.update(destination=goal, stable_steps=self._stable[name],
                           complete=self._placed(metrics) and self._stable[name] >= self.config.stable_steps)
            objects[name] = metrics
        return dict(complete=all(row['complete'] for row in objects.values()), objects=objects)

    def _step(self, position, gripper):
        from .environment import TASK

        if self.disturbance_hook is not None:
            details = self.disturbance_hook(self.env, step=self.step_count,
                                            sim_time_s=self.sim_time_s, rng=self.rng)
            if details is not None:
                self._event('disturbance', details=details)
        action = np.concatenate([position, self._orientation, [gripper]])
        self.env.step(action)
        self.step_count += 1
        for name, goal in TASK:
            self._stable[name] = self._stable[name] + 1 if self._placed(self._placement_metrics(name, goal)) else 0

    def _move(self, phase, target, gripper, *, hold_steps=8):
        self.phase = phase
        self._event('phase_start', phase=phase, target_position_m=np.asarray(target).tolist())
        target = np.asarray(target, dtype=float)
        command = self._eef_position()
        stable = 0
        for _ in range(self.config.phase_timeout_steps):
            delta = target - command
            distance = np.linalg.norm(delta)
            if distance > self.config.waypoint_step:
                command += delta * (self.config.waypoint_step / distance)
            else:
                command = target.copy()
            self._step(command, gripper)
            error = float(np.linalg.norm(self._eef_position() - target))
            stable = stable + 1 if error < self.config.position_tolerance else 0
            if stable >= hold_steps:
                self._event('phase_end', phase=phase, position_error_m=error)
                return
        raise SkillFailure(f'{phase}: end effector did not reach waypoint')

    def _rejected(self, operation, error, **data):
        self._event('skill_rejected', operation=operation, success=False, error=error, **data)
        return SkillResult(False, dict(simulated=True, backend='robosuite', **data), error=error)

    def pick_place(self, target: str, container: str) -> SkillResult:
        from .environment import GOALS

        if target not in self.env.objects or container not in GOALS:
            return self._rejected('pick_place', 'unknown_target_or_destination', target=target, container=container)
        if self.halted:
            return self._rejected('pick_place', 'reset_required_after_failure', target=target, container=container)
        start_wall, start_sim, start_step = time.perf_counter(), self.sim_time_s, self.step_count
        self._event('skill_start', operation='pick_place', target=target, container=container)
        error = None
        lifted = False
        lift_height = 0.0
        try:
            cfg = self.config
            source = np.array(self.object_states()[target]['position_m'])
            above = np.array([*source[:2], cfg.transport_height])
            self._move('raise', [*self._eef_position()[:2], cfg.transport_height], -1)
            self._move('approach', above, -1)
            self._move('descend', source, -1)
            self._move('close_gripper', source, 1, hold_steps=25)
            if not self.env._check_grasp(self.env.robots[0].gripper['right'], self.env.objects[target]):
                raise SkillFailure('grasp_failed: both finger pads must contact the object')
            self._move('lift', above, 1)
            lift_height = self.object_states()[target]['position_m'][2] - source[2]
            lifted = bool(lift_height > 0.06)
            if not lifted:
                raise SkillFailure('lift_failed: object did not rise by 0.06 m')
            dest_xy = GOALS[container]
            self._move('transport', [*dest_xy, cfg.transport_height], 1)
            # Preserve measured object-to-grip-site offset, then lower to table.
            obj_pos = np.array(self.object_states()[target]['position_m'])
            offset = obj_pos - self._eef_position()
            place = np.array([*dest_xy, cfg.table_height + cfg.cube_half_size + 0.002]) - offset
            self._move('lower', place, 1)
            self._move('release', place, -1, hold_steps=25)
            self._move('retreat', [*dest_xy, cfg.transport_height], -1)
            self.phase = 'settle'
            stable = 0
            for _ in range(cfg.phase_timeout_steps):
                self._step(np.array([*dest_xy, cfg.transport_height]), -1)
                stable = stable + 1 if self._placed(self._placement_metrics(target, container)) else 0
                if stable >= cfg.stable_steps:
                    break
            else:
                raise SkillFailure('placement_failed: object did not settle in destination')
        except SkillFailure as caught:
            error = str(caught)
            self.halted = True
        except Exception as caught:
            self.halted = True
            self._event('skill_error', operation='pick_place', phase=self.phase, error=str(caught))
            raise
        success = error is None
        data = dict(backend='robosuite', simulated=True, target=target, container=container,
                    phase=self.phase, lifted=lifted, lift_height_m=float(lift_height),
                    control_steps=self.step_count-start_step, sim_duration_s=self.sim_time_s-start_sim,
                    wall_duration_s=time.perf_counter()-start_wall,
                    placement=self._placement_metrics(target, container), task=self.task_status(),
                    objects=self.object_states(), robot=self.robot_state())
        self._event('skill_result', operation='pick_place', success=success, error=error, result=data)
        self.phase = 'idle' if success else self.phase
        return SkillResult(success, data, message='Simulation pick/place completed' if success else '', error=error)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
