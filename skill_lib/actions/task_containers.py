"""Container locations are fixed for one task; RGB-D evidence is still transient.

Only successful, complete base-frame placement plans enter this registry. It is
owned by one robot facade and is never restored from the execution audit JSON.
"""
from copy import deepcopy
import math
from uuid import uuid4

from core.skill_base import SkillResult
from .targets import container_key, parse_container_spec


POSE_FIELDS = ('approach_pose_mmrad', 'place_pose_mmrad', 'lift_pose_mmrad')


def complete_place_plan(plan):
    try:
        return all(len(plan[field]) == 6 and all(
            not isinstance(value, bool) and math.isfinite(float(value))
            for value in plan[field]) for field in POSE_FIELDS)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def remember_container(robot, target, plan):
    if not complete_place_plan(plan):
        return
    key = container_key(parse_container_spec(target))
    plan['task_container'] = dict(task_id=robot._container_task_id, key=key,
                                  fixed_for_task=True, coordinate_frame='base')
    robot._task_container_plans[key] = dict(target=target, plan=deepcopy(plan))


def _publish(robot, target, key, plan):
    robot.place_plans[key] = plan
    robot.place_counts.setdefault(key, 0)
    robot.observations[target] = dict(kind='container', target=target,
                                      container_key=key, place_plan=plan)


def recall_container(robot, target):
    key = container_key(parse_container_spec(target))
    record = robot._task_container_plans.get(key)
    if record is None or not is_task_container_plan(robot, record['plan']):
        return None
    plan = deepcopy(record['plan'])
    _publish(robot, target, key, plan)
    print(f'[container_cache] {target}: reuse fixed base-frame plan for this task')
    return SkillResult(True, data=plan, message='fixed container plan reused within task')


def is_task_container_plan(robot, plan):
    """Trust only this task's registered poses (including configured slot offsets)."""
    if not isinstance(plan, dict):
        return False
    binding = plan.get('task_container')
    if not isinstance(binding, dict) or binding.get('task_id') != robot._container_task_id:
        return False
    record = robot._task_container_plans.get(binding.get('key'))
    if (record is None or binding != record['plan'].get('task_container')
            or not complete_place_plan(plan)):
        return False
    index = plan.get('place_slot_index', 0)
    if type(index) is not int or index < 0:
        return False
    expected = robot._offset_place_plan(record['plan'], index)
    return all(list(plan[field]) == list(expected[field]) for field in POSE_FIELDS)


def discard_scene_evidence(robot, reason):
    robot.observations.clear()
    robot.place_plans.clear()
    if reason == 'new_task':
        robot._task_container_plans.clear()
        robot._container_task_id = uuid4().hex
        robot.place_counts.clear()
        return
    # Source images/bboxes remain historical metadata, never current RGB-D cache
    # entries. Only these registered base-frame placement facts are republished.
    for key, record in robot._task_container_plans.items():
        _publish(robot, record['target'], key, deepcopy(record['plan']))


def forget_container(robot, target):
    key = container_key(parse_container_spec(target))
    robot._task_container_plans.pop(key, None)
    robot.place_plans.pop(key, None)
    for alias, observation in list(robot.observations.items()):
        if observation.get('kind') == 'container' and observation.get('container_key') == key:
            robot.observations.pop(alias, None)
    from runtime.observation_memory import memory_for
    memory = memory_for(getattr(robot.context, 'camera', None))
    if memory is not None:
        memory.invalidate('explicit_container_refresh')
