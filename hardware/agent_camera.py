"""Stationary, immutable RGB-D observations bound to the calibrated FR3 flange."""
from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import time
import uuid
import threading

import numpy as np

from .fr3 import FrankaMoveItArm
from .gsnet_single import pose_difference
from .transforms import load_transforms
from runtime.observation_memory import ObservationMemory, stationary_camera, StaleObservationError


class CalibratedCamera:
    def __init__(self, camera, config, arm=None):
        self.camera = camera
        self.config = config
        self.arm = arm or FrankaMoveItArm(config.arm, execution_enabled=False)
        self.transforms = load_transforms(config)
        self.observation_memory = ObservationMemory(config.observation_cache, clock=lambda: time.time())
        self._capture_local = threading.local()
        self.last_wrist_capture = None

    @property
    def last_wrist_capture(self):
        return getattr(self._capture_local, 'last', None)

    @last_wrist_capture.setter
    def last_wrist_capture(self, value):
        self._capture_local.last = value

    @property
    def observation_scope(self):
        payload = json.dumps({'camera': asdict(self.config.cameras),
                              'calibration': self.transforms.summary(),
                              'perception': asdict(self.config.perception)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def active_camera_names(self):
        return ('wrist',) if self.config.cameras.mode == 'wrist' else ('global', 'wrist')

    @property
    def perception_workers(self):
        return self.config.perception.parallel_candidate_workers

    @stationary_camera
    def observation_pose(self):
        return self.arm.get_pose().values

    def invalidate_observations(self, reason='external_scene_change'):
        self.observation_memory.invalidate(reason, discard_scene=True)
        self.last_wrist_capture = None

    def __getattr__(self, name):
        return getattr(self.camera, name)

    @stationary_camera
    def capture_global(self):
        if self.config.cameras.mode == 'wrist':
            capture = self.capture_wrist()
            capture['requested_view'] = 'global'
            capture['view_fallback'] = 'wrist_only_mode'
            return capture
        return self._freeze(self.camera.capture_global())

    @stationary_camera
    def capture_wrist(self):
        if self.config.arm.stack != 'real':
            raise RuntimeError('Live images require real capture-time FK, not fake joint states')
        before = self.arm.get_pose()
        memory = self.observation_memory
        scope = self.observation_scope
        cached = memory.get(('capture', 'wrist'), before.values, scope)
        if cached is not None:
            if not memory.commit(cached['observation_cache']['epoch'], lambda: None):
                raise StaleObservationError('Scene changed during cached capture lookup')
            cached['observation_cache']['hit'] = True
            self.last_wrist_capture = cached
            print('[observation_cache] wrist image hit; original acquisition geometry retained')
            return cached
        epoch = memory.epoch
        before_time = time.time()
        first = self.camera.capture_wrist()
        frame_number = first['color_frame_number']
        deadline = time.monotonic() + 5
        while True:
            time.sleep(.03)
            capture = self.camera.capture_wrist()
            if (capture['color_frame_number'] >= frame_number + 2
                    and capture['timestamp'] >= before_time):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('No fresh stationary wrist RGB-D frame')
        after = self.arm.get_pose()
        difference = pose_difference(before, after)
        if difference['translation_m'] > .001 or difference['rotation_rad'] > .01:
            raise RuntimeError('Robot moved while capturing: take a fresh stationary image')
        serial = capture['device_serial']
        optical = self.transforms.cameras['wrist']['optical_frame']
        self.transforms._check_capture('wrist', serial, optical)
        if not capture.get('depth_aligned_to_color') or capture.get('points_map') is None:
            raise RuntimeError('GSNet requires aligned RGB-D and SDK point cloud from the same frame')
        if np.any(np.asarray(capture['color_intrinsics'].get('coeffs', [0.] * 5)) != 0):
            raise ValueError('GSNet target pixel filtering requires zero-distortion color intrinsics')
        capture['capture_geometry'] = {
            'before': before.values, 'after': after.values,
            'before_unix_s': before_time, 'after_unix_s': time.time(),
            'stationarity': difference, 'serial': serial, 'optical_frame': optical,
            'calibration': self.transforms.summary(),
        }
        capture['camera_name'] = 'wrist'
        capture['observation_cache'] = {'hit': False, 'epoch': epoch,
                                      'scene_epoch': memory.scene_epoch,
                                      'acquired_at': capture['timestamp'], 'scope': scope}
        frozen = self._freeze(capture)
        if not memory.commit(epoch, lambda: memory.put(
                ('capture', 'wrist'), frozen, before.values, capture['timestamp'], scope, epoch)):
            raise StaleObservationError('Scene changed while acquiring RGB-D; capture again')
        self.last_wrist_capture = frozen
        return frozen

    def capture(self, camera_name):
        if camera_name == 'wrist':
            return self.capture_wrist()
        if camera_name == 'global':
            return self.capture_global()
        raise ValueError(camera_name)

    def _freeze(self, capture):
        import cv2
        result = dict(capture)
        folder = Path(self.camera.save_dir).resolve() / 'observations' / uuid.uuid4().hex
        folder.mkdir(parents=True)
        for key, filename in [('color', 'color.png'), ('depth', 'depth.png'), ('depth_vis', 'depth_vis.png')]:
            if result.get(key) is not None:
                target = folder / filename
                if not cv2.imwrite(str(target), result[key]):
                    raise IOError(f'Cannot save {target}')
                result[key + '_path'] = str(target)
        points = result.pop('points_map', None)
        if points is not None:
            np.save(folder / 'points.npy', points)
            result['points_path'] = str(folder / 'points.npy')
        serializable = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        (folder / 'capture.json').write_text(json.dumps(serializable, indent=2))
        return result
