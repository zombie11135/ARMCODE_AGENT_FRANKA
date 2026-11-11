"""Standalone binding to the existing bounded interpreter, not a scheduler."""
from runtime.cap_policy import PolicyError, PolicyInterpreter, PolicyStopped


TWO_OBJECT_POLICY = '''\
def policy():
    if not observe("red_cube"):
        return False
    if not pick_place("red_cube", "red_goal"):
        return False
    if not observe("blue_cube"):
        return False
    return pick_place("blue_cube", "blue_goal")
'''


def run_policy(backend, source=TWO_OBJECT_POLICY):
    # Importing environment here is safe: backend has already configured EGL.
    from .environment import TASK

    completed = 0
    observations = {target: 0 for target, _ in TASK}
    results = []

    def observe(target):
        if target not in observations:
            raise PolicyError(f'Unbound observation target: {target}')
        observations[target] += 1
        if observations[target] > 2:
            raise PolicyError(f'Observation budget exhausted: {target}')
        return bool(backend.observe(target).success)

    def pick_place(target, destination):
        nonlocal completed
        if completed >= len(TASK) or (target, destination) != TASK[completed]:
            raise PolicyError('Unplanned, reordered or repeated manipulation')
        result = backend.pick_place(target, destination)
        results.append(result)
        if not result.success:
            raise PolicyStopped(result.error)
        completed += 1
        return True

    PolicyInterpreter(source, dict(observe=observe, pick_place=pick_place)).run()
    if completed != len(TASK) or not backend.task_status()['complete']:
        raise PolicyStopped('Policy ended before both objects were placed and stable')
    return results
