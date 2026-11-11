"""Restricted AST -> skill contracts -> PMIR, without eval/exec.

Supports literal/local names, bounded literal loops, observe/pick_place, and
if [not] observe(name). No arbitrary Python, recursion, while or device access.
Explicit observe preserves its source position. Implicit skill preparation can
run early; physical commits retain program order. Unsupported syntax is rejected.
"""
import ast

from .graph import Graph, Node


class SkillRegistry:
    def __init__(self):
        self.skills = {'observe': (1, 'observe'), 'pick_place': (2, 'pick_place')}

    def register_blackbox(self, name, arity):
        if name in self.skills or arity < 0:
            raise ValueError('Duplicate skill or invalid arity')
        self.skills[name] = (arity, 'blackbox')


def conflicts(left, right):
    return bool(left and right and ('*' in left or '*' in right or set(left) & set(right)))


class Compiler:
    def __init__(self, registry):
        self.registry, self.graph = registry, Graph()
        self.last_commit = None
        self.barrier = None
        self.observations = {}
        self.conditions = ()
        self.line = 0

    def add(self, kind, phase, target, destination=None, *, inputs=(), resources=(), reads=(),
            writes=(), owner=False, physical=False, gate=None):
        if len(self.graph.nodes) >= 1024:
            raise ValueError('PMIR node budget exceeded')
        node = Node(f'u{len(self.graph.nodes):03d}', kind, phase, target, destination,
                    tuple(inputs), tuple(resources), tuple(reads), tuple(writes), owner, physical,
                    gate, self.conditions, self.line)
        for parent in inputs:
            self.graph.link(parent, node.id, 'data', 'Consumes immutable result')
        for parent, expected in self.conditions:
            self.graph.link(parent, node.id, 'control', f'observe result must be {expected}')
        if self.barrier:
            self.graph.link(self.barrier, node.id, 'control', 'Opaque skill may affect all state')
        # Effects are provided once per skill, then applied to arbitrary programs.
        for earlier in self.graph.nodes:
            if conflicts(earlier.writes, node.reads):
                self.graph.link(earlier.id, node.id, 'physical', 'Read after possible state write')
            if conflicts(earlier.writes, node.writes):
                self.graph.link(earlier.id, node.id, 'physical', 'Ordered writes to shared state')
            if conflicts(earlier.reads, node.writes):
                self.graph.link(earlier.id, node.id, 'physical', 'Preserve preceding state read')
        self.graph.nodes.append(node)
        return node.id

    def perceive(self, target, *, explicit=False):
        capture = self.add('CAPTURE', 'snapshot', target, resources=('camera',), owner=True,
            reads=(f'pose:{target}', 'fixed_camera'), gate=None if explicit else f'visible:{target}')
        if explicit and self.last_commit:
            self.graph.link(self.last_commit, capture, 'control', 'Explicit observe occurs after preceding skill')
        result = self.add('PERCEIVE', 'rgbd_color_box', target, inputs=(capture,),
                          resources=('cpu',), reads=(f'pose:{target}',))
        self.observations[target] = result
        return result

    def call(self, call, env):
        if not isinstance(call.func, ast.Name) or call.func.id not in self.registry.skills or call.keywords:
            raise ValueError('Only registered positional skill calls are supported')
        arity, expansion = self.registry.skills[call.func.id]
        args = [self.literal(n, env) for n in call.args]
        if len(args) != arity or any(type(a) is not str for a in args):
            raise ValueError('Skill arguments must be bound string names with registered arity')
        self.line = call.lineno
        target = args[0] if args else call.func.id
        if expansion == 'observe':
            return self.perceive(target, explicit=True)
        if expansion == 'blackbox':
            previous = list(self.graph.nodes)
            terminal = self.add('BLACKBOX', call.func.id, target,
                args[1] if len(args) > 1 else None, owner=True, physical=True,
                resources=('robot', 'gripper', 'camera', 'cpu'), reads=('*',), writes=('*',))
            self.graph.nodes[-1].arguments = tuple(args)
            for node in previous:
                self.graph.link(node.id, terminal, 'control', 'Opaque skill serialization barrier')
            self.barrier = self.last_commit = terminal
            self.observations.clear()
            return terminal
        destination = args[1]
        observation = self.observations.pop(target, None) or self.perceive(target)
        self.observations.pop(target, None)
        grasp = self.add('GRASP_PLAN', 'top_down_box', target, inputs=(observation,),
                         resources=('cpu',), reads=(f'pose:{target}',))
        plan = self.add('MOTION_PLAN', 'cartesian_waypoints', target, destination, inputs=(grasp,),
                        owner=True, resources=('cpu',), reads=(f'pose:{target}', 'robot', f'goal:{destination}'))
        if self.last_commit:
            self.graph.link(self.last_commit, plan, 'control', 'Current robot state and ordered physical commits')
        parent = plan
        for phase in ('raise', 'approach', 'descend', 'close', 'lift', 'transport', 'lower', 'release', 'retreat', 'settle'):
            kind = 'GRIPPER' if phase in ('close', 'release') else 'VERIFY' if phase == 'settle' else 'MOVE'
            writes = ('robot',) if phase in ('raise', 'approach', 'descend') else ('robot', f'pose:{target}')
            current = self.add(kind, phase, target, destination, inputs=(plan,), owner=True,
                               physical=True, resources=('robot', 'gripper'), writes=writes)
            self.graph.link(parent, current, 'control', 'Skill phase order and physical verification')
            parent = current
        self.last_commit = parent
        return parent

    def literal(self, node, env):
        if isinstance(node, ast.Constant) and type(node.value) in (str, bool, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 16:
            return tuple(self.literal(n, env) for n in node.elts)
        raise ValueError('Only literals and locally bound values are supported')

    def block(self, statements, env, *, terminal_returns=False):
        for index, stmt in enumerate(statements):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                env[stmt.targets[0].id] = self.literal(stmt.value, env)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self.call(stmt.value, env)
            elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and not stmt.orelse:
                values = self.literal(stmt.iter, env)
                if not isinstance(values, tuple) or len(values) > 16:
                    raise ValueError('Loops require at most 16 literal elements')
                for value in values:
                    env[stmt.target.id] = value
                    self.block(stmt.body, env)
            elif isinstance(stmt, ast.If):
                test, expected = stmt.test, True
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    test, expected = test.operand, False
                if not isinstance(test, ast.Call) or not isinstance(test.func, ast.Name) or test.func.id != 'observe':
                    raise ValueError('PMIR v1 condition must be observe(name) or not observe(name)')
                predicate = self.call(test, env)
                old, previous_commit = self.conditions, self.last_commit
                self.observations.clear()
                # No branch prediction: guarded computation waits for the real predicate.
                self.conditions = old + ((predicate, expected),)
                first_begin = len(self.graph.nodes)
                self.block(stmt.body, dict(env))
                first_nodes = self.graph.nodes[first_begin:]
                self.last_commit = previous_commit
                self.observations.clear()
                self.conditions = old + ((predicate, not expected),)
                second_begin = len(self.graph.nodes)
                self.block(stmt.orelse, dict(env))
                second_nodes = self.graph.nodes[second_begin:]
                self.conditions = old
                # Join every branch leaf, including observation-only branches.
                # Merely joining the last physical commit can release too early.
                leaves = []
                for branch in (first_nodes, second_nodes):
                    ids = {n.id for n in branch}
                    leaves.extend(n.id for n in branch if not any(
                        e.source == n.id and e.target in ids for e in self.graph.edges))
                merge = self.add('JOIN', 'branch', '', owner=True, inputs=tuple(dict.fromkeys([predicate, *leaves])))
                self.last_commit = self.barrier = merge
                self.observations.clear()
            elif isinstance(stmt, ast.Return) and terminal_returns and index == len(statements)-1:
                if isinstance(stmt.value, ast.Call):
                    self.call(stmt.value, env)
                elif stmt.value is not None and self.literal(stmt.value, env) is not True:
                    raise ValueError('Early/failure returns need a future control-flow extension')
            elif isinstance(stmt, ast.Pass):
                continue
            else:
                raise ValueError(f'Unsupported PMIR statement: {type(stmt).__name__} at line {stmt.lineno}')


def compile_program(source, registry=None):
    if len(source) > 32768:
        raise ValueError('Program exceeds 32 KiB')
    tree = ast.parse(source)
    body = tree.body
    if len(body) == 1 and isinstance(body[0], ast.FunctionDef):
        fn = body[0]
        if (fn.name != 'policy' or fn.decorator_list or fn.args.args or fn.args.posonlyargs
                or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.returns):
            raise ValueError('Expected undecorated policy() with no arguments')
        body = fn.body
    compiler = Compiler(registry or SkillRegistry())
    compiler.block(body, {}, terminal_returns=True)
    if not compiler.graph.nodes:
        raise ValueError('Empty program')
    compiler.graph.validate()
    return compiler.graph
