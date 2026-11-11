"""Small bounded Python policy interpreter. Does NOT use eval/exec.

The public contract deliberately contains perception and compound manipulation
skills only. This is a task-level LMP baseline, not arbitrary Python or control
of joint positions. Validation alone is not a physical safety guarantee.
"""
import ast
import math


class PolicyError(ValueError):
    pass


class PolicyStopped(PolicyError):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


APIS = {'observe', 'pick_place', 'range'}
NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
         ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
         ast.If, ast.For, ast.Call, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
         ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Pass)


def validate_policy(source, *, allow_undefined=False):
    if len(source) > 32768:
        raise PolicyError('Policy source exceeds 32 KiB')
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PolicyError(f'Invalid policy syntax: {error}') from error
    nodes = list(ast.walk(tree))
    if len(nodes) > 2000:
        raise PolicyError('Policy AST too large')
    functions = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            raise PolicyError('Only function definitions are allowed at module scope')
        if node.name in functions or node.name in APIS:
            raise PolicyError('Duplicate or reserved function name')
        functions[node.name] = node
    if 'policy' not in functions or functions['policy'].args.args:
        raise PolicyError('Define policy() with no arguments')
    calls = set()
    for node in nodes:
        if not isinstance(node, NODES):
            raise PolicyError(f'Unsupported policy syntax: {type(node).__name__}')
        if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef)):
            name = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else node.name
            if name.startswith('_') or len(name) > 80:
                raise PolicyError('Private/oversized identifiers are forbidden')
        if isinstance(node, ast.FunctionDef):
            if node not in tree.body or node.decorator_list or node.returns:
                raise PolicyError('Nested functions, annotations and decorators are forbidden')
        if isinstance(node, ast.arguments):
            if node.posonlyargs or node.kwonlyargs or node.vararg or node.kwarg or node.defaults or node.kw_defaults:
                raise PolicyError('Use plain positional function arguments')
            if len({arg.arg for arg in node.args}) != len(node.args):
                raise PolicyError('Duplicate function arguments')
        if isinstance(node, ast.arg) and node.annotation:
            raise PolicyError('Annotations are forbidden')
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise PolicyError('Only assignment to a local name is allowed')
        if isinstance(node, ast.For) and (not isinstance(node.target, ast.Name) or node.orelse):
            raise PolicyError('Use simple bounded for loops without else')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise PolicyError('Only positional calls to named policy functions or skill APIs are allowed')
            calls.add(node.func.id)
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) not in (str, int, float, bool, type(None)):
                raise PolicyError('Unsupported constant')
            if isinstance(value, str) and len(value) > 512:
                raise PolicyError('String too large')
            if type(value) in (int, float) and (abs(value) > 10000 or not math.isfinite(value)):
                raise PolicyError('Numeric constant outside policy bounds')
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > 32:
            raise PolicyError('Policy list too long')
    missing = calls - functions.keys() - APIS
    if missing and not allow_undefined:
        raise PolicyError(f'Undefined helpers: {sorted(missing)}')
    return tree, sorted(missing)


class PolicyInterpreter:
    def __init__(self, source, api, *, max_steps=2000, max_api_calls=64):
        tree, _ = validate_policy(source)
        self.functions = {node.name: node for node in tree.body}
        self.api = api
        self.steps, self.calls, self.depth = max_steps, max_api_calls, 0

    def tick(self):
        self.steps -= 1
        if self.steps < 0:
            raise PolicyError('Policy instruction budget exhausted')

    def run(self):
        return self.call('policy', [])

    def call(self, name, args):
        self.tick()
        if name == 'range':
            if not 1 <= len(args) <= 3 or any(type(v) is not int for v in args):
                raise PolicyError('range requires integer arguments')
            try:
                result = range(*args)
                if len(result) > 16:
                    raise PolicyError('range has more than 16 iterations')
            except (ValueError, OverflowError) as error:
                raise PolicyError(str(error)) from error
            return result
        if name in self.api:
            expected = 1 if name == 'observe' else 2
            if len(args) != expected or any(type(value) is not str for value in args):
                raise PolicyError('Skill APIs require target name strings')
            self.calls -= 1
            if self.calls < 0:
                raise PolicyError('Policy skill-call budget exhausted')
            return self.api[name](*args)
        fn = self.functions[name]
        if len(args) != len(fn.args.args):
            raise PolicyError(f'Argument count mismatch: {name}')
        if self.depth >= 8:
            raise PolicyError('Policy call depth exhausted')
        self.depth += 1
        try:
            self.block(fn.body, dict(zip([arg.arg for arg in fn.args.args], args)))
        except _Return as result:
            return result.value
        finally:
            self.depth -= 1

    def block(self, statements, env):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Return):
                raise _Return(self.expr(node.value, env) if node.value else None)
            if isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif isinstance(node, ast.Assign):
                env[node.targets[0].id] = self.expr(node.value, env)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.For):
                values = self.expr(node.iter, env)
                if not isinstance(values, (list, tuple, range)) or len(values) > 32:
                    raise PolicyError('Only bounded lists/ranges are iterable')
                for value in values:
                    env[node.target.id] = value
                    self.block(node.body, env)
            elif isinstance(node, ast.Pass):
                pass

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise PolicyError(f'Unknown local variable: {node.id}')
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expr(value, env) for value in node.elts]
        if isinstance(node, ast.Call):
            return self.call(node.func.id, [self.expr(arg, env) for arg in node.args])
        if isinstance(node, ast.UnaryOp):
            return not self.expr(node.operand, env)
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                result = self.expr(value, env)
                if isinstance(node.op, ast.And) and not result:
                    return result
                if isinstance(node.op, ast.Or) and result:
                    return result
            return result
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, right in zip(node.ops, node.comparators):
                value = self.expr(right, env)
                operations = {ast.Eq: lambda: left == value, ast.NotEq: lambda: left != value,
                              ast.Lt: lambda: left < value, ast.LtE: lambda: left <= value,
                              ast.Gt: lambda: left > value, ast.GtE: lambda: left >= value}
                try:
                    matches = operations[type(op)]()
                except TypeError as error:
                    raise PolicyError('Incompatible comparison operands') from error
                if not matches:
                    return False
                left = value
            return True
        raise PolicyError(f'Unsupported expression: {type(node).__name__}')
