# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static import guards for the runtime, model, and primitive layers."""

from __future__ import annotations

import ast
from pathlib import Path

AGENT_COMPOSE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = AGENT_COMPOSE_ROOT / "megatron" / "lite"
LAYER_ROOTS = {
    "primitive": PACKAGE_ROOT / "primitive",
    "model": PACKAGE_ROOT / "model",
    "runtime": PACKAGE_ROOT / "runtime",
}
DENIED_IMPORTS = {
    "primitive": ("megatron.lite.model", "megatron.lite.runtime"),
    "model": ("megatron.lite.runtime",),
    "runtime": (),
}
SHARED_CONTRACTS = ("megatron.lite.runtime.contracts",)


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def test_layer_import_boundaries() -> None:
    violations: list[str] = []
    for layer, root in LAYER_ROOTS.items():
        for path in sorted(root.rglob("*.py")):
            for lineno, module in _imports(path):
                if any(_matches(module, allowed) for allowed in SHARED_CONTRACTS):
                    continue
                for denied in DENIED_IMPORTS[layer]:
                    if _matches(module, denied):
                        rel = path.relative_to(AGENT_COMPOSE_ROOT)
                        violations.append(f"{rel}:{lineno}: {module} imports {denied}")
    assert violations == []
