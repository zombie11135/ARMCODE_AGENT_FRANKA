"""Name resolution over the local object registry; no model or hardware at import.

Strong names resolve deterministically. Shared descriptions require current image
evidence or an explicit task-scoped binding. Model guesses never edit the registry.
"""
from __future__ import annotations

import copy
import re
import time
import unicodedata
import uuid


def normalize_name(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def legacy_object_id(name):
    # Deterministic IDs let schema-1 galleries be read without a write operation.
    return "ref_" + uuid.uuid5(uuid.NAMESPACE_URL, "armcode-object:" + normalize_name(name)).hex[:20]


class NameResolutionRequired(RuntimeError):
    def __init__(self, result):
        self.result = result
        candidates = result.get("candidates", [])
        choices = "；".join(f"{c['name']} [{c['object_id']}]" for c in candidates)
        message = f"目标名称“{result.get('query', '')}”尚未唯一确定（{result.get('reason', result.get('status'))}）。"
        if choices:
            message += f"候选：{choices}。"
        message += "请使用完整名称重新输入任务，或用 /bind 简称 = 物品编号 指定下一次任务的含义。"
        super().__init__(message)


class ObjectNameResolver:
    def __init__(self, service, bindings=None):
        self.service = service
        self.bindings = {}
        for alias, target in (bindings or {}).items():
            obj = service.resolve(target)
            if obj is None:
                raise ValueError(f"临时绑定目标必须是已登记的明确名称或编号: {target}")
            key = normalize_name(alias)
            if not key:
                raise ValueError("临时名称不能为空")
            strong = service.resolve(alias)
            if strong is not None and strong['object_id'] != obj['object_id']:
                raise ValueError("临时绑定不能覆盖另一个物品的正式名称/编号/别名")
            self.bindings[key] = obj['object_id']

    @staticmethod
    def _candidate(obj):
        return {key: copy.deepcopy(obj.get(key)) for key in
                ('object_id', 'name', 'aliases', 'attributes', 'descriptors', 'category', 'detector_prompt')}

    def lookup(self, query):
        started = time.perf_counter()
        key = normalize_name(query)
        objects = self.service.gallery.read()['objects'] if self.service.settings.enabled else []
        source, candidates = 'unknown', []
        if key in self.bindings:
            source = 'task_binding'
            candidates = [o for o in objects if o['object_id'] == self.bindings[key]]
        else:
            by_id = [o for o in objects if normalize_name(o['object_id']) == key]
            strong = [o for o in objects if key in
                      [normalize_name(s) for s in [o['name'], *o.get('aliases', [])]]]
            weak = [o for o in objects if key and key in
                    [normalize_name(s) for s in [o.get('attributes', {}).get('brand', ''),
                                                *o.get('descriptors', [])]]]
            if by_id:
                source, candidates = 'object_id', by_id
            elif strong:
                candidates = list({o['object_id']: o for o in [*strong, *weak]}.values())
                source = 'exact_name_or_alias' if len(candidates) == 1 else 'shared_description'
            elif weak:
                source, candidates = 'shared_description', weak
        status = ('needs_scene' if source == 'shared_description' else
                  'resolved' if len(candidates) == 1 else 'unknown')
        return dict(query=str(query), normalized_query=key, status=status, source=source,
                    object_id=candidates[0]['object_id'] if status == 'resolved' else None,
                    canonical_name=candidates[0]['name'] if status == 'resolved' else None,
                    candidates=[self._candidate(o) for o in candidates],
                    elapsed_s=time.perf_counter()-started)

    def mentions(self, text):
        """Longest non-overlapping registry phrases; not a general language parser."""
        text = normalize_name(text)
        tokens = set(self.bindings)
        for obj in self.service.catalog():
            tokens.update(normalize_name(s) for s in [obj['name'], obj['object_id'],
                *obj.get('aliases', []), obj.get('attributes', {}).get('brand', ''),
                *obj.get('descriptors', [])])
        spans = []
        for token in sorted(tokens - {''}, key=lambda s: (-len(s), s)):
            for match in re.finditer(re.escape(token), text):
                start, end = match.span()
                if (token[0].isascii() and token[0].isalnum() and start and
                        text[start-1].isascii() and text[start-1].isalnum()):
                    continue
                if (token[-1].isascii() and token[-1].isalnum() and end < len(text) and
                        text[end].isascii() and text[end].isalnum()):
                    continue
                if not any(start < b and end > a for a, b, _ in spans):
                    spans.append((start, end, self.lookup(token)))
        return [r for _, _, r in sorted(spans, key=lambda item: item[0])]

    def resolve_in_image(self, query, context, image_path):
        started = time.perf_counter()
        resolution = self.lookup(query)
        if resolution['status'] != 'needs_scene':
            return resolution
        from skill_lib.perception.reference_recognition import RecognizeObjectSkill
        resolution['name_lookup_s'] = resolution['elapsed_s']
        matches, attempts = [], []
        incomplete = False
        for obj in resolution['candidates']:
            result = RecognizeObjectSkill(context).run(image_path=image_path, query=obj['object_id'], segment=False)
            attempts.append(dict(object_id=obj['object_id'], success=result.success, data=result.data, error=result.error))
            if result.success:
                matches.append(obj)
            elif result.data.get('stage') in ('reference_error', 'reference_disabled', 'unregistered_target'):
                incomplete = True
        resolution.update(scene_attempts=attempts, image_path=str(image_path))
        if len(matches) == 1 and not incomplete:
            resolution.update(status='resolved', source='scene_reference', object_id=matches[0]['object_id'],
                              canonical_name=matches[0]['name'])
        else:
            resolution.update(status='clarification_required',
                              reason='incomplete_visual_evidence' if incomplete else
                                     'multiple_visible_identities' if matches else 'no_confirmed_visible_identity')
        resolution['elapsed_s'] = time.perf_counter() - started
        return resolution


def display_name(context, query):
    service = getattr(context, 'object_references', None)
    obj = service.resolve(query) if service is not None else None
    return obj['name'] if obj else query


def handle_name_command(command, args):
    """Interactive metadata commands. No camera/arm/gripper access."""
    operation = command.split(' ', 1)[0]
    if operation not in ('/names', '/bind', '/bind-clear'):
        return False
    if operation == '/bind-clear':
        args.next_task_name_bindings = {}
        print('[names] 已清空下一次任务的临时名称绑定。')
        return True
    from types import SimpleNamespace
    from services.object_reference_service import build_reference_service
    service = build_reference_service(SimpleNamespace(hardware_config_path=getattr(args, 'hardware_config', None)))
    if service is None:
        print('[names] 当前配置没有启用参考图库。')
        return True
    if operation == '/names':
        import json
        print(json.dumps(service.catalog(), ensure_ascii=False, indent=2))
        print('[names] 下一次任务绑定:', getattr(args, 'next_task_name_bindings', {}))
        return True
    body = command[len('/bind'):].strip()
    if '=' not in body:
        print('[names] 用法：/bind 简称 = 已登记的完整名称或 ref_编号；仅对下一次任务生效。')
        return True
    alias, target = (s.strip() for s in body.split('=', 1))
    try:
        new = dict(getattr(args, 'next_task_name_bindings', {}))
        new[alias] = target
        resolver = ObjectNameResolver(service, new)
        args.next_task_name_bindings = dict(resolver.bindings)
        print(f'[names] 下一次任务：{alias} → {service.resolve(target)["name"]}；任务开始后自动消费此绑定。')
    except ValueError as error:
        print(f'[names] {error}')
    return True


def bind_plan_names(graph, task_goal, service, bindings=None):
    """Deterministic binding after the LLM, including preservation of ambiguous brands."""
    if service is None:
        return graph
    resolver = ObjectNameResolver(service, bindings)
    mentions = resolver.mentions(task_goal)
    explicit = {r['object_id'] for r in mentions if r['status'] == 'resolved'}
    records = []
    for node in graph.nodes:
        if node.skill not in ('grasp', 'observe') or not isinstance(node.args, dict):
            continue
        original = str(node.args.get('target') or '')
        result = resolver.lookup(original)
        # Do not let the planner guess one product when only a shared name was supplied.
        weak = [r for r in mentions if r['status'] == 'needs_scene' and
                not explicit.intersection(c['object_id'] for c in r['candidates']) and
                (result['object_id'] in {c['object_id'] for c in r['candidates']} or
                 normalize_name(original) == r['normalized_query'])]
        if len(weak) == 1:
            result = weak[0]
        if result['status'] == 'resolved':
            node.args.update(target=result['object_id'], target_name=result['canonical_name'],
                             requested_name=result['query'])
        elif result['status'] == 'needs_scene':
            node.args.update(target=result['query'], requested_name=result['query'],
                             name_candidate_ids=[c['object_id'] for c in result['candidates']])
        records.append(dict(node_id=node.node_id, original_target=original, **result))
    graph.metadata['name_resolution'] = records
    return graph
