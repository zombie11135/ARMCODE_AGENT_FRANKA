"""Generate hierarchical task-level language model programs over skill APIs."""
import hashlib
import json
from pathlib import Path

from runtime.cap_policy import PolicyError, PolicyInterpreter, validate_policy


CONTRACT = '''Write Python policy code only (no explanation, imports or markdown).
Define policy() with no arguments. Public APIs:
- observe(object_name) -> bool: acquire/identify the named object, real feedback.
- pick_place(object_name, container_name) -> bool: guarded grasp, place, verify/home.
  A manipulation failure stops execution; do NOT retry this call in policy code.
Only use these exact task names and perform the requested pick/place pairs in order,
once each. Every successful run must finish every pair. You may define helper
functions; undefined helpers will be synthesized in a bounded subsequent pass.
Allowed Python: local assignment, strings/bools/small positive integers, list,
for over lists or range(1..16), if/else, and/or/not, comparisons, return, functions.
No attributes, indexing, imports, while, exceptions, external libraries, robot
coordinates or device access. Observation retries must be bounded (at most 2).
Example for a hypothetical single pair ["box", "tray"]:
def policy():
    if ensure_visible("box"):
        pick_place("box", "tray")
def ensure_visible(target):
    for attempt in range(2):
        if observe(target):
            return True
    return False
'''


def task_pairs(actions):
    pairs, holding = [], None
    for action in actions:
        if action.op == 'grasp':
            if holding is not None:
                raise PolicyError('Nested grasp in task graph')
            holding = action.target
        elif action.op == 'place':
            if holding is None:
                raise PolicyError('Placement without grasp in task graph')
            pairs.append([holding, action.target]);holding = None
    if holding is not None or not pairs:
        raise PolicyError('CaP tasks require complete grasp/place pairs')
    return pairs


def code_only(text):
    value = str(text).strip()
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return value + '\n'


class CapCodeGenerator:
    def __init__(self, client):
        self.client = client

    def generate(self, goal, actions, output_path, *, feedback=None, constraints=()):
        pairs = task_pairs(actions)
        prompt = CONTRACT + '\nUser goal: ' + goal + '\nRequired pairs: ' + json.dumps(pairs, ensure_ascii=False)
        prompt += '\nActive graph constraints: ' + json.dumps(list(constraints), ensure_ascii=False)
        if feedback is not None:
            prompt += '\nPrevious attempt feedback (repair this subtask only): ' + json.dumps({
                key: feedback.get(key) for key in ('failure_type', 'error', 'recovery', 'failed_action', 'completed_actions')}, ensure_ascii=False)
        records = []
        source = ''
        validated = False
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self.client.chat(text=prompt, max_tokens=2048, temperature=0.0)
            records.append(dict(prompt=prompt, response=response))
            source = code_only(response)
            for _ in range(3):
                _, missing = validate_policy(source, allow_undefined=True)
                if not missing:
                    break
                prompt = (CONTRACT + '\nExisting policy:\n' + source + '\nDefine ONLY these missing functions: '
                          + json.dumps(missing) + '\nDo not redefine policy or existing functions.')
                response = self.client.chat(text=prompt, max_tokens=2048, temperature=0.0)
                records.append(dict(prompt=prompt, response=response))
                source += '\n' + code_only(response)
            validate_policy(source)
            # Non-actuating optimistic branch rehearsal detects missing/duplicate task work.
            completed = []
            def observe(name):
                if name not in {pair[0] for pair in pairs}:
                    raise PolicyError('Unknown observation target')
                return True
            def pick(obj, container):
                if len(completed) >= len(pairs) or [obj, container] != pairs[len(completed)]:
                    raise PolicyError('Policy changes task order, target binding or repeats completed work')
                completed.append([obj, container]);return True
            PolicyInterpreter(source, dict(observe=observe, pick_place=pick)).run()
            if completed != pairs:
                raise PolicyError('Policy does not complete the task when observations succeed')
            validated = True
            path.write_text(source, encoding='utf-8')
            return source
        except PolicyError:
            raise
        except Exception as error:
            raise PolicyError(f'CaP generation failed: {error}') from error
        finally:
            path.with_suffix('.cap.json').write_text(json.dumps(dict(
                backend='cap_skill_policy_v1', contract_version=1, goal=goal, pairs=pairs,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(), source=source,
                constraints=list(constraints), feedback=feedback,
                generations=records, validated=validated, validation='bounded_ast_and_optimistic_dry_run'),
                ensure_ascii=False, indent=2), encoding='utf-8')
