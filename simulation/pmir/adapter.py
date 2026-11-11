"""Cooperative micro-skills over existing Panda physics and completion checks.

One owner calls every MuJoCo/EGL API. Workers receive only immutable RGB-D and
version tokens. Poses from RGB-D drive approach; simulator truth is explicitly
used for state-version monitoring, grasp contact, placement offset and success.
"""
from contextlib import nullcontext
from functools import partial, wraps
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.robosuite_backend import RobosuiteBackend, SimConfig
from simulation.robosuite_backend.backend import SkillFailure
from .perception import compute, mask_for


def measured(category):
    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            with self.cost_scope(category):
                return method(self, *args, **kwargs)
        return call
    return decorate


class StaleEvidence(RuntimeError):
    pass


class Adapter:
    def __init__(self, output, *, seed=7, scenario='independent', compute_delay_s=0., clock='fast'):
        self.owner = threading.get_ident()
        self.cost_scope = lambda category: nullcontext()
        self.backend = RobosuiteBackend(SimConfig(seed=seed, spawn_jitter=0 if scenario == 'occluded' else .015),
                                       log_path=output/'physics_events.jsonl')
        # Instance-local instrumentation; existing backend implementation is unchanged.
        def wrap_call(owner, name, category):
            original = getattr(owner, name)
            @wraps(original)
            def call(*args, **kwargs):
                with self.cost_scope(category):
                    return original(*args, **kwargs)
            setattr(owner, name, call)
        for owner, name, category in (
            (self.backend.env, 'step', 'physics_step'),
            (self.backend, '_placement_metrics', 'state_validation'),
            (self.backend, 'object_states', 'state_validation'),
            (self.backend.env, '_check_grasp', 'state_validation'),
            (self.backend.log, 'write', 'logging'),
        ):
            wrap_call(owner, name, category)
        self.output, self.scenario = output, scenario
        self.settings = dict(compute_delay_s=compute_delay_s)
        self.clock = clock
        self.grip = -1
        self.last_tick = None
        self.idle_steps = self.deadline_misses = 0
        self.max_lateness_s = 0.
        self.snapshot_id = 0
        self.step_intervals = []
        self.versions = {name: 0 for name in self.backend.env.objects}
        self.anchors = {name: np.array(s['position_m']) for name, s in self.backend.object_states().items()}
        self.orientations = {name: np.array(s['quaternion_xyzw']) for name, s in self.backend.object_states().items()}
        self.visible = {}
        self.visibility_events = []
        self.probe_frames = []
        self.emit = lambda *args, **kwargs: None
        self.completed = []
        self.lift_evidence = {}
        if scenario == 'occluded':
            # Episode setup changes only a synthetic camera. No object teleport
            # after initialization. Both boxes share x at this scene's seed.
            cam = self.backend.env.sim.model.camera_name2id(self.backend.config.camera)
            eye, center = np.array([-.07, -1., .84]), np.array([-.07, .12, .82])
            back = (eye-center)/np.linalg.norm(eye-center)
            right = np.cross([0, 0, 1], back)
            right /= np.linalg.norm(right)
            up = np.cross(back, right)
            model = self.backend.env.sim.model
            model.cam_pos[cam] = eye
            model.cam_quat[cam] = Rotation.from_matrix(np.column_stack([right, up, back])).as_quat()[[3, 0, 1, 2]]
            self.backend.env.sim.forward()

    def assert_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('MuJoCo and EGL belong to the scheduler owner')

    def snapshot(self):
        self.assert_owner()
        with self.cost_scope('image_acquire_format'):
            data = self.backend.observe().data
        self.snapshot_id += 1
        with self.cost_scope('image_snapshot_copy'):
            camera = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data['camera'].items()}
            for value in camera.values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
        return dict(camera=camera, read_versions=dict(self.versions), snapshot_id=self.snapshot_id,
                    episode=self.backend.episode, step=self.backend.step_count)

    @measured('state_validation')
    def probe(self, producer='initial'):
        snap = self.snapshot()
        self.probe_frames.append((producer, snap))
        states = self.backend.object_states()
        for target in self.versions:
            pixels = int(mask_for(snap['camera'], target)[0].sum())
            stable = (states[target]['linear_speed_m_s'] < self.backend.config.settle_speed
                      and states[target]['angular_speed_rad_s'] < self.backend.config.settle_angular_speed)
            was = self.visible.get(target, False)
            self.visible[target] = pixels >= 25 and stable
            row = dict(target=target, producer=producer, visible_pixels=pixels, stable=stable,
                       visible=self.visible[target], version=self.versions[target],
                       snapshot_id=snap['snapshot_id'], monitor='rgbd_mask_plus_simulator_velocity')
            row['frame_file'] = f'visibility_{producer}.png'
            self.emit('visibility_checked', **row)
            if not was and self.visible[target]:
                self.visibility_events.append(row)
                self.emit('event_released', gate=f'visible:{target}', **row)
        return snap

    @measured('state_validation')
    def valid(self, node, value):
        versions = value.get('read_versions', {})
        return node.target not in versions or versions[node.target] == self.versions[node.target]

    def worker_job(self, node, values):
        inputs = tuple(values[parent] for parent in node.inputs)
        return partial(compute, node.kind, node.target, inputs, dict(self.settings))

    def immediate(self, node, values):
        self.assert_owner()
        if node.kind == 'CAPTURE':
            return self.snapshot()
        if node.kind == 'MOTION_PLAN':
            evidence = values[node.inputs[0]]
            if not self.valid(node, evidence):
                raise StaleEvidence(f'Changed pose before planning: {node.target}')
            from simulation.robosuite_backend.environment import GOALS
            b, cfg = self.backend, self.backend.config
            source = np.asarray(evidence['position_m'])
            return dict(source=source, raise_to=np.array([*b._eef_position()[:2], cfg.transport_height]),
                        above=np.array([*source[:2], cfg.transport_height]),
                        destination_xy=np.array(GOALS[node.destination]),
                        read_versions=dict(evidence['read_versions']), success=True,
                        planner='nominal_cartesian_waypoints_not_collision_search')
        if node.kind == 'JOIN':
            return dict(success=True)
        raise ValueError(f'No simulation handler for {node.kind}')

    def _step(self, position, grip, *, idle=False):
        self.assert_owner()
        if self.clock == 'real_time' and self.last_tick is not None:
            deadline = self.last_tick + 1/self.backend.config.control_freq
            remaining = deadline-time.perf_counter()
            if remaining > 0:
                with self.cost_scope('control_pacing_wait'):
                    time.sleep(remaining)
            else:
                self.max_lateness_s = max(self.max_lateness_s, -remaining)
                self.deadline_misses += int(-remaining > .005)
        self.last_tick = time.perf_counter()
        self.backend._step(position, grip)
        self.step_intervals.append((self.last_tick, time.perf_counter()))
        self.grip = grip
        self.idle_steps += int(idle)
        self._monitor_versions()

    @measured('state_validation')
    def _monitor_versions(self):
        for name, state in self.backend.object_states().items():
            position = np.asarray(state['position_m'])
            quat = np.asarray(state['quaternion_xyzw'])
            # Independent-object effects are checked against observed simulator
            # state on every control step, including accidental contacts.
            changed = np.linalg.norm(position-self.anchors[name]) > .0015
            changed |= 2*np.arccos(np.clip(abs(np.dot(quat, self.orientations[name])), 0., 1.)) > .04
            if changed:
                self.versions[name] += 1
                self.anchors[name], self.orientations[name] = position.copy(), quat.copy()
                self.emit('state_changed', target=name, version=self.versions[name], source='simulator_ground_truth')

    def idle_tick(self):
        if self.clock == 'real_time':
            self._step(self.backend._eef_position(), self.grip, idle=True)
        else:
            time.sleep(.001)

    def physical(self, node, values):
        b, cfg = self.backend, self.backend.config
        plan = values[node.inputs[0]]
        if node.phase == 'raise' and not self.valid(node, plan):
            raise StaleEvidence(f'Changed pose before physical commit: {node.target}')
        dest = plan['destination_xy']
        grip, hold = (-1 if node.phase in ('raise', 'approach', 'descend', 'release', 'retreat', 'settle') else 1), 8
        if node.phase == 'raise':
            target = plan['raise_to']
        elif node.phase in ('approach', 'lift'):
            target = plan['above']
        elif node.phase in ('descend', 'close'):
            target = plan['source']
        elif node.phase in ('transport', 'retreat', 'settle'):
            target = np.array([*dest, cfg.transport_height])
        elif node.phase == 'lower':
            offset = np.asarray(b.object_states()[node.target]['position_m'])-b._eef_position()
            target = np.array([*dest, cfg.table_height+cfg.cube_half_size+.002])-offset
            plan['place'] = target
        else:
            target = plan['place']
        if node.phase in ('close', 'release'):
            hold = 25
        b.phase = node.phase
        b._event('phase_start', phase=node.phase, target=node.target, target_position_m=target.tolist())
        command, stable = b._eef_position(), 0
        for _ in range(cfg.phase_timeout_steps):
            delta = target-command
            distance = np.linalg.norm(delta)
            command = command+delta*(cfg.waypoint_step/distance) if distance > cfg.waypoint_step else target.copy()
            self._step(command, grip)
            if node.phase == 'settle':
                good = b._placed(b._placement_metrics(node.target, node.destination))
                required = cfg.stable_steps
            else:
                good = np.linalg.norm(b._eef_position()-target) < cfg.position_tolerance
                required = hold
            stable = stable+1 if good else 0
            yield None  # Scheduler may capture snapshots and consume CPU jobs here.
            if stable >= required:
                break
        else:
            b.halted = True
            raise SkillFailure(f'{node.phase}: waypoint or placement timeout')
        if node.phase == 'close' and not b.env._check_grasp(b.env.robots[0].gripper['right'], b.env.objects[node.target]):
            raise SkillFailure('Both finger pads must contact the object')
        if node.phase == 'lift':
            height = b.object_states()[node.target]['position_m'][2]-plan['source'][2]
            if height <= .06:
                raise SkillFailure('Object failed to rise by 0.06 m')
            self.lift_evidence[node.target] = float(height)
        if node.phase == 'settle':
            self.completed.append((node.target, node.destination))
        b._event('phase_end', phase=node.phase, target=node.target)
        return dict(success=True, phase=node.phase)

    def after_node(self, node):
        if node.phase in ('lift', 'release', 'settle'):
            self.probe(node.id)

    def close(self):
        self.assert_owner()
        self.backend.close()
