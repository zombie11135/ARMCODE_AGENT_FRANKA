"""Bounded, process-local observation memory. Poses are flange metres/radians.

No robot or camera dependencies: a caller supplies measured poses and the
original acquisition time. A hit never updates the acquisition pose or age.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import time
from functools import wraps
import threading
import weakref


class StaleObservationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObservationCachePolicy:
    enabled: bool = True
    position_radius_m: float = .02
    orientation_radius_rad: float = math.radians(5)
    max_age_s: float = 30.
    max_entries: int = 16

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('observation_cache.enabled must be boolean')
        for name in ('position_radius_m', 'orientation_radius_rad', 'max_age_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'observation_cache.{name} must be finite and positive')
        if self.orientation_radius_rad > math.pi:
            raise ValueError('orientation_radius_rad must be <= pi')
        if type(self.max_entries) is not int or self.max_entries <= 0:
            raise ValueError('observation_cache.max_entries must be a positive integer')


def _rotation(pose):
    roll, pitch, yaw = pose[3:]
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return (cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
            sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, -sp, cp*sr, cp*cr)


def pose_distance(a, b):
    a, b = tuple(a), tuple(b)
    if len(a) != 6 or len(b) != 6 or not all(math.isfinite(v) for v in a+b):
        raise ValueError('Expected two finite 6D poses in metres/radians')
    translation = math.sqrt(sum((x-y)**2 for x, y in zip(a[:3], b[:3])))
    # trace(Ra.T Rb); handles Euler wraparound and equivalent Euler poses.
    cosine = (sum(x*y for x, y in zip(_rotation(a), _rotation(b))) - 1) / 2
    return translation, math.acos(max(-1., min(1., cosine)))


class ObservationMemory:
    def __init__(self, policy=None, clock=time.time):
        self.policy = policy or ObservationCachePolicy()
        self.clock = clock
        self.epoch = 0
        self.scene_epoch = 0
        self.entries = OrderedDict()
        self._sequence = 0
        self.stats = {'hits': 0, 'misses': 0, 'invalidations': 0}
        self.last_invalidation = None
        self._lock = threading.RLock()
        # Capture and physical execution share this lock; model inference does
        # not hold it. Always acquire device_lock before _lock if both are used.
        self.device_lock = threading.RLock()
        self._changing = 0
        self._listeners = []
        self._local = threading.local()

    def on_invalidate(self, callback):
        reference = weakref.WeakMethod(callback) if getattr(callback, '__self__', None) is not None else weakref.ref(callback)
        with self._lock:
            self._listeners.append(reference)

    def commit(self, epoch, publish, *, scene_only=False):
        """Validate and publish to Agent state atomically with invalidation."""
        with self._lock:
            current = self.scene_epoch if scene_only else self.epoch
            if self._changing or epoch != current:
                self.stats.setdefault('stale_rejections', 0)
                self.stats['stale_rejections'] += 1
                return False
            publish()
            return True

    @contextmanager
    def mutation(self, reason):
        """One physical action, with start/end invalidation even on exceptions."""
        with self.device_lock:
            with self._lock:
                self._changing += 1
            try:
                self.invalidate(reason + '_started', discard_scene=True)
                yield
            finally:
                with self._lock:
                    try:
                        self.invalidate(reason + '_finished', discard_scene=True)
                    finally:
                        self._changing -= 1

    def require_stable(self):
        with self._lock:
            if self._changing:
                raise StaleObservationError('Scene is changing; observation cannot be published')

    def invalidate(self, reason='explicit_refresh', *, discard_scene=False):
        with self._lock:
            self.entries.clear()
            self.epoch += 1
            if discard_scene:
                self.scene_epoch += 1
            self.stats['invalidations'] += 1
            self.last_invalidation = reason
            live = []
            for reference in self._listeners:
                callback = reference()
                if callback is not None:
                    if discard_scene:
                        callback(reason)
                    live.append(reference)
            self._listeners = live

    @contextmanager
    def fresh(self, reason='force_refresh'):
        """Invalidate prior evidence and bypass both image and skill-result hits."""
        self.invalidate(reason)
        self._local.bypass = getattr(self._local, 'bypass', 0) + 1
        try:
            yield
        finally:
            self._local.bypass -= 1

    def put(self, key, value, pose, acquired_at, scope, epoch=None):
        with self._lock:
            return self._put(key, value, pose, acquired_at, scope, epoch)

    def _put(self, key, value, pose, acquired_at, scope, epoch):
        if self._changing:
            return
        if not self.policy.enabled or (epoch is not None and epoch != self.epoch):
            return
        pose = tuple(pose)
        pose_distance(pose, pose)
        age = self.clock() - acquired_at
        if not math.isfinite(age) or not 0 <= age <= self.policy.max_age_s:
            return
        self._sequence += 1
        self.entries[self._sequence] = (key, deepcopy(value), pose, acquired_at, scope, self.epoch)
        while len(self.entries) > self.policy.max_entries:
            self.entries.popitem(last=False)

    def get(self, key, pose, scope):
        with self._lock:
            return self._get(key, pose, scope)

    def _get(self, key, pose, scope):
        now = self.clock()
        candidates = []
        for token, (entry_key, value, source_pose, acquired_at, source_scope, epoch) in list(self.entries.items()):
            age = now - acquired_at
            if not 0 <= age <= self.policy.max_age_s or epoch != self.epoch:
                del self.entries[token]
                continue
            if (self._changing or getattr(self._local, 'bypass', 0) or not self.policy.enabled
                    or entry_key != key or source_scope != scope):
                continue
            distance, angle = pose_distance(source_pose, pose)
            if distance <= self.policy.position_radius_m and angle <= self.policy.orientation_radius_rad:
                candidates.append((distance/self.policy.position_radius_m + angle/self.policy.orientation_radius_rad,
                                   -acquired_at, token, value))
        if not candidates:
            self.stats['misses'] += 1
            return None
        _, _, token, value = min(candidates, key=lambda item: item[:3])
        self.entries.move_to_end(token)
        self.stats['hits'] += 1
        return deepcopy(value)


def memory_for(camera):
    """Ignore arbitrary/mock attributes and cameras without calibrated memory."""
    memory = getattr(camera, 'observation_memory', None)
    return memory if isinstance(memory, ObservationMemory) else None


def invalidate_camera(camera, reason, *, discard_scene=True):
    memory = memory_for(camera)
    if memory is not None:
        memory.invalidate(reason, discard_scene=discard_scene)


@contextmanager
def fresh_camera(camera, reason='force_refresh'):
    memory = memory_for(camera)
    if memory is None:
        yield
    else:
        with memory.fresh(reason):
            yield


def fresh_skill_camera(reason):
    def decorate(function):
        @wraps(function)
        def invoke(skill, *args, **kwargs):
            with fresh_camera(skill.context.camera, reason):
                return function(skill, *args, **kwargs)
        return invoke
    return decorate


def stationary_camera(function):
    """Camera methods only; serialize frame/FK binding against physical actions."""
    @wraps(function)
    def invoke(camera, *args, **kwargs):
        with camera.observation_memory.device_lock:
            camera.observation_memory.require_stable()
            return function(camera, *args, **kwargs)
    return invoke


def scene_mutation(reason):
    """Decorate low-level execution: first argument is a context or its owner."""
    def decorate(function):
        @wraps(function)
        def invoke(owner, *args, **kwargs):
            context = getattr(owner, 'context', owner)
            memory = memory_for(getattr(context, 'camera', None))
            if memory is None:
                return function(owner, *args, **kwargs)
            with memory.device_lock:
                plan = kwargs.get('plan', args[0] if args and isinstance(args[0], dict) else {})
                provenance = plan.get('grasp_input', {}).get('observation_provenance')
                if provenance is not None and not memory.commit(provenance['epoch'], lambda: None):
                    raise StaleObservationError('Scene changed after grasp planning; replan before execution')
                place_provenance = plan.get('observation_provenance')
                if place_provenance is not None and not memory.commit(
                        place_provenance['scene_epoch'], lambda: None, scene_only=True):
                    raise StaleObservationError('Scene changed after place planning; observe container again')
                with memory.mutation(reason):
                    return function(owner, *args, **kwargs)
        return invoke
    return decorate
