"""Reuse successful observation skills using their actual RGB-D source pose."""
from functools import wraps
from contextlib import contextmanager
from inspect import signature

from core.skill_base import SkillResult
from runtime.observation_memory import memory_for, fresh_camera, invalidate_camera


def _key(robot, kind, target):
    context = robot.context
    # Include selection/OCR settings and live backend identities. This memory is
    # deliberately not restored from task JSON written by an earlier process.
    return ('skill', kind, str(target), repr(vars(robot.config)),
            tuple(id(getattr(context, name, None)) for name in
                  ('detector', 'segmenter', 'ocr', 'object_references', 'occlusion')))


def _observation(kind, target, data):
    if kind == 'container':
        from skill_lib.actions.targets import container_key, parse_container_spec
        return dict(kind=kind, target=target, container_key=container_key(parse_container_spec(target)), place_plan=data)
    return dict(kind=kind, target=target, selected_instance=data.get('selected_instance'),
                grasp_input=data.get('grasp_input'), result=data)


def _publish(robot, target, observation, data):
    robot.observations[target] = observation
    if observation['kind'] == 'container':
        key = observation['container_key']
        robot.place_plans[key] = data
        robot.place_counts.setdefault(key, 0)


def _stale_result():
    return SkillResult(False, data={'failure_type': 'stale_observation', 'retryable': True},
                       error='Scene changed during perception; acquire a new observation')


def recall_observation(robot, kind, target):
    camera = robot.context.camera
    memory = memory_for(camera)
    if memory is None:
        return None
    cached = memory.get(_key(robot, kind, target), camera.observation_pose(), camera.observation_scope)
    if cached is None:
        return None
    if not memory.commit(cached['epoch'], lambda: _publish(robot, target, cached['observation'], cached['data'])):
        return None
    print(f'[observation_cache] {kind}({target}) skill hit; perception skipped')
    return SkillResult(success=True, data=cached['data'], message='pose-matched observation reused')


def reuse_observation(kind):
    def decorate(function):
        target_parameter = tuple(signature(function).parameters)[1]
        @wraps(function)
        def invoke(robot, *args, **kwargs):
            target = args[0] if args else kwargs[target_parameter]
            hit = recall_observation(robot, kind, target)
            if hit is not None:
                return hit
            camera = robot.context.camera
            memory = memory_for(camera)
            previous = getattr(camera, 'last_wrist_capture', None) if memory is not None else None
            start_epoch = memory.epoch if memory is not None else None
            key = _key(robot, kind, target)
            try:
                result = function(robot, *args, **kwargs)
            except BaseException:
                if memory is not None:
                    memory.commit(start_epoch, lambda: memory.invalidate('observation_exception'))
                raise
            if result.success and memory is None:
                _publish(robot, target, _observation(kind, target, result.data), result.data)
            if memory is not None:
                if not result.success:
                    # A failed detector must not repeatedly see the same cached
                    # frame on retries or active-view probes.
                    source = getattr(camera, 'last_wrist_capture', None)
                    epoch = (source.get('observation_cache', {}).get('epoch', start_epoch)
                             if isinstance(source, dict) and source is not previous else start_epoch)
                    if not memory.commit(epoch, lambda: memory.invalidate('observation_failed')):
                        return _stale_result()
                else:
                    source = getattr(camera, 'last_wrist_capture', None)
                    if isinstance(source, dict) and source is not previous:
                        geometry = source.get('capture_geometry', {})
                        provenance = source.get('observation_cache', {})
                        if geometry.get('before') is not None and 'scope' in provenance and 'epoch' in provenance:
                            if kind == 'container':
                                result.data['observation_provenance'] = dict(provenance)
                            observation = _observation(kind, target, result.data)
                            def publish():
                                memory.put(key, {'data': result.data, 'observation': observation,
                                                 'epoch': provenance['epoch']},
                                           geometry['before'], source['timestamp'],
                                           provenance['scope'], provenance['epoch'])
                                _publish(robot, target, observation, result.data)
                            if (provenance['scope'] == camera.observation_scope
                                    and memory.commit(provenance['epoch'], publish)):
                                return result
                    return _stale_result()
            return result
        return invoke
    return decorate


@contextmanager
def manipulation_observations(robot, label):
    """No pre-action bypass: planning can reuse images. Always invalidate after.

    FR3 execution also invalidates immediately before its first gripper/motion
    command so internal recovery cannot see evidence from before a failed move.
    """
    camera = getattr(robot.context, 'camera', None)
    memory = memory_for(camera)
    try:
        yield
    finally:
        if robot.config.execute and memory is not None:
            memory.invalidate(label + '_attempt_completed', discard_scene=True)


def invalidate_after_manipulation(label):
    def decorate(function):
        @wraps(function)
        def invoke(robot, *args, **kwargs):
            with manipulation_observations(robot, label):
                try:
                    result = function(robot, *args, **kwargs)
                except BaseException:
                    invalidate_camera(getattr(robot.context, 'camera', None), label + '_exception')
                    raise
                if not result.success:
                    invalidate_camera(getattr(robot.context, 'camera', None), label + '_failed')
                return result
        return invoke
    return decorate


def require_fresh_observation(label):
    def decorate(function):
        @wraps(function)
        def invoke(robot, *args, **kwargs):
            with fresh_camera(getattr(robot.context, 'camera', None), label):
                return function(robot, *args, **kwargs)
        return invoke
    return decorate
