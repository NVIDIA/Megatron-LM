# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check stacklevel usage for warnings.warn() calls in megatron/core.

warnings.warn() inside a function should pass stacklevel so the warning points
at the caller. module-level calls keep the default, and so does __post_init__
(it runs under the dataclass-generated __init__ whose frame reports as
'<string>', so no constant stacklevel is correct there).
"""

import ast
from pathlib import Path

MEGATRON_CORE = Path(__file__).resolve().parents[2] / "megatron" / "core"


class _WarnCollector(ast.NodeVisitor):
    """Collect function-level warnings.warn() calls missing stacklevel."""

    def __init__(self):
        self.function_stack = []
        self.violations = []

    def _visit_function(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node):
        func = node.func
        is_warnings_warn = (
            isinstance(func, ast.Attribute)
            and func.attr == "warn"
            and isinstance(func.value, ast.Name)
            and func.value.id == "warnings"
        )
        if (
            is_warnings_warn
            and self.function_stack
            and "__post_init__" not in self.function_stack
            and not any(keyword.arg == "stacklevel" for keyword in node.keywords)
        ):
            self.violations.append(node.lineno)
        self.generic_visit(node)


def test_function_level_warn_calls_pass_stacklevel():
    """Every function-level warnings.warn() call must set stacklevel."""
    repo_root = MEGATRON_CORE.parent.parent
    violations = []
    for path in sorted(MEGATRON_CORE.rglob("*.py")):
        collector = _WarnCollector()
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        violations.extend(
            f"{path.relative_to(repo_root).as_posix()}:{lineno}" for lineno in collector.violations
        )
    assert not violations, (
        "warnings.warn() calls inside functions must pass an explicit stacklevel "
        "(stacklevel=2 for direct calls, deeper for nested helpers): " + ", ".join(violations)
    )
