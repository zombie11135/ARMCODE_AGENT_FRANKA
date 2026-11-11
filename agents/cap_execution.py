"""Execute a generated task policy inside the existing camera/hardware session."""
import hashlib
from pathlib import Path
import time

from agents.cap_code_generator import task_pairs
from core.actions import SimpleAction
from runtime.cap_policy import PolicyError, PolicyStopped, PolicyInterpreter
from services.object_reference_service import atomic_json


def run_policy(runtime, robot, context, camera_service, source_actions, resolved_actions):
    source = runtime.config.cap_policy_source
    pairs = task_pairs(source_actions)
    resolved_pairs = task_pairs(resolved_actions)
    aliases = {a.target: b.target for a, b in zip(source_actions, resolved_actions) if a.op in ('grasp', 'observe')}
    completed, trace = [], []
    observation_counts = {}
    pair_index = 0
    started = time.monotonic()

    def observe(target):
        if target not in aliases:
            raise PolicyError(f'Unbound observation target: {target}')
        observation_counts[target] = observation_counts.get(target, 0) + 1
        if observation_counts[target] > 2:
            raise PolicyError('Policy observation budget exhausted for target: ' + target)
        action = SimpleAction('observe', aliases[target])
        runtime._execution_checkpoint = dict(action=action, index=len(completed), completed=list(completed), holding=None)
        context.manipulation_progress = {}
        result = robot.observe(action.target)
        trace.append(dict(op='observe', target=target, success=result.success, error=result.error,
                          elapsed_s=time.monotonic()-started))
        return bool(result.success)

    def pick_place(target, container):
        nonlocal pair_index, completed
        if pair_index >= len(pairs) or [target, container] != pairs[pair_index]:
            raise PolicyError('Policy attempted an unplanned, reordered or repeated manipulation')
        obj, destination = resolved_pairs[pair_index]
        actions = [SimpleAction(**row) for row in completed] + [SimpleAction('grasp', obj), SimpleAction('place', destination)]
        success = runtime._run_actions(robot, context, camera_service, actions,
                                       completed_actions=completed, start_index=len(completed))
        trace.append(dict(op='pick_place', target=target, container=container, success=success,
                          elapsed_s=time.monotonic()-started))
        if not success:
            raise PolicyStopped('Guarded manipulation failed; policy cannot retry physical actions')
        completed = [a.__dict__ for a in actions]
        pair_index += 1
        return True

    success, error = False, None
    try:
        PolicyInterpreter(source, dict(observe=observe, pick_place=pick_place)).run()
        if pair_index != len(pairs):
            raise PolicyError('Policy stopped before completing all requested pairs (observation may have failed)')
        success = True
        return True
    except PolicyStopped as caught:
        error = str(caught)
        # _run_actions already wrote the original evidence; preserve it.
        return False
    except PolicyError as caught:
        error = str(caught)
        runtime._write_failure_report(SimpleAction('policy', 'cap'), None, 'policy_failed', error,
            evidence=dict(automatic_replan=False, trace=trace), completed_actions=completed)
        return False
    except Exception as caught:
        error = str(caught)
        raise
    finally:
        atomic_json(Path(runtime.config.plan_json).with_name('cap_trace.json'), dict(
            backend='cap_skill_policy_v1', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            success=success, error=error, completed_pairs=pair_index,
            events=trace, elapsed_s=time.monotonic()-started))
