# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check module call delegation without importing optional model dependencies."""

import ast
from pathlib import Path

_BASE = "megatron.core.transformer.module.MegatronModule"


def _module_overrides(sources):
    """Find overrides through transitive inheritance, including imported aliases."""
    aliases, bases, classes = {}, {}, []
    for source_module, source in sources.items():
        is_package = source_module.endswith('.__init__')
        module = source_module.removesuffix('.__init__')
        tree = ast.parse(source, filename=module)

        def resolve(expr):
            name = ast.unparse(expr)
            return f"{module}.{name}"

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                parts = module.split('.')
                package = parts[: len(parts) - node.level + int(is_package)] if node.level else []
                origin = '.'.join([*package, node.module or '']).rstrip('.')
                for alias in node.names:
                    aliases[f"{module}.{alias.asname or alias.name}"] = f"{origin}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    aliases[f"{module}.{name}"] = alias.name if alias.asname else name
            elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[f"{module}.{target.id}"] = resolve(node.value)
            elif isinstance(node, ast.ClassDef):
                name = f"{module}.{node.name}"
                bases.setdefault(name, set()).update(resolve(base) for base in node.bases)
                classes.append((name, node))

    def canonical(name):
        seen = set()
        while name not in seen:
            seen.add(name)
            parts = name.split('.')
            for end in range(len(parts), 0, -1):
                prefix = '.'.join(parts[:end])
                if prefix in aliases:
                    name = '.'.join([aliases[prefix], *parts[end:]])
                    break
            else:
                return name
        return name

    descendants = {_BASE}
    while True:
        found = {
            name
            for name, parents in bases.items()
            if any(canonical(p) in descendants for p in parents)
        }
        if found <= descendants:
            break
        descendants.update(found)
    # A mixin earlier in the MRO can also replace a module's __call__.
    checked = descendants.copy()
    while True:
        parents = {canonical(parent) for name in checked for parent in bases.get(name, ())}
        if parents <= checked:
            break
        checked.update(parents)
    for name, node in classes:
        if name not in checked or name == _BASE:
            continue
        for member in node.body:
            if (
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == '__call__'
            ):
                yield name, member
            elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                targets = member.targets if isinstance(member, ast.Assign) else [member.target]
                if any(isinstance(t, ast.Name) and t.id == '__call__' for t in targets):
                    raise AssertionError(
                        f"{name}: define __call__ as a checked wrapper, not an alias"
                    )


def _delegates(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == '__call__'
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == 'super'
        and not node.func.value.args
        and not node.func.value.keywords
    )


def _check_wrapper(method):
    """Allow direct wrappers; put execution branches in _call_with_bound_args."""
    assert isinstance(method, ast.FunctionDef) and not method.decorator_list
    assert isinstance(method.body[-1], ast.Return), "__call__ must return its delegated result"
    returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
    delegates = [node for node in ast.walk(method) if _delegates(node)]
    assert len(delegates) == len(returns), "Delegate exactly once per return"
    for result in returns:
        assert (
            sum(_delegates(node) for node in ast.walk(result)) == 1
        ), "Every return must wrap super().__call__()"
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in (
                'forward',
                '_call_with_bound_args',
                '_call_impl',
                '_wrapped_call_impl',
            ), "Do not bypass MegatronModule.__call__"
            if node.func.attr == '__call__':
                assert _delegates(node), "Delegate through super().__call__()"
        if (
            isinstance(node, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.FunctionDef))
            and node is not method
        ):
            assert not any(
                _delegates(child) for child in ast.walk(node)
            ), "Do not hide delegation in conditional expressions or nested functions"


def test_repository_overrides_only_wrap_base_call():
    root = Path(__file__).resolve().parents[3]
    sources = {
        '.'.join(path.relative_to(root).with_suffix('').parts): path.read_text()
        for path in (root / 'megatron').rglob('*.py')
    }
    overrides = list(_module_overrides(sources))
    assert overrides, "No subclasses found; check source discovery"
    for name, method in overrides:
        try:
            _check_wrapper(method)
        except AssertionError as error:
            raise AssertionError(f"{name}:{method.lineno}: {error}") from error
