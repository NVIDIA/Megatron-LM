# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static contracts of the indexer top-k package.

* Only ``plugins/env.py`` names the plugins' environment keys or touches ``os.environ``.
* Only ``plugins/`` imports external modules by path or manipulates ``sys.modules``.
* Nothing in the package uses ``torch.distributed``: selection issues no collectives.
* The package imports without Triton, DeepGEMM or cuDNN, and without Megatron Core.

The per-module bindings (``modules/attention/indexer_topk.py``) follow the same source rules.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = LITE_ROOT / "megatron" / "lite" / "primitive" / "kernels" / "indexer_topk"
PLUGINS_ROOT = PACKAGE_ROOT / "plugins"
ENV_MODULE = PLUGINS_ROOT / "env.py"
PACKAGE = "megatron.lite.primitive.kernels.indexer_topk"
BINDINGS_MODULE = (
    LITE_ROOT / "megatron" / "lite" / "primitive" / "modules" / "attention" / "indexer_topk.py"
)

_ENVIRONMENT_ATTRIBUTES = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}
_EXTERNAL_LOADERS = {
    "__import__",
    "exec_module",
    "import_module",
    "module_from_spec",
    "spec_from_file_location",
    "spec_from_loader",
    "ModuleSpec",
}
_ALLOWED_TOP_LEVEL = ("torch", "megatron.lite.primitive")
_GUARDED_TOP_LEVEL = ("triton",)


def _package_files() -> list[Path]:
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert ENV_MODULE in files and PACKAGE_ROOT / "__init__.py" in files
    return files


def _source_files() -> list[Path]:
    """The package and the bindings module: every file the source rules apply to."""
    assert BINDINGS_MODULE.is_file()
    return [*_package_files(), BINDINGS_MODULE]


def _relative(path: Path) -> str:
    return str(path.relative_to(LITE_ROOT))


def _names_in(node: ast.AST) -> set[str]:
    """Every attribute and name identifier used under ``node``."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.alias):
            names.add(child.name.rsplit(".", 1)[-1])
    return names


def test_only_env_module_mentions_sglang_env() -> None:
    violations = []
    for path in _source_files():
        if path == ENV_MODULE:
            continue
        text = path.read_text(encoding="utf-8")
        if "SGLANG_LITETOPK" in text or "LITETOPK_GRAFT" in text:
            violations.append(f"{_relative(path)}: names a plugin environment key")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _ENVIRONMENT_ATTRIBUTES
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                violations.append(f"{_relative(path)}:{node.lineno}: os.{node.attr}")
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                touched = {alias.name for alias in node.names} & _ENVIRONMENT_ATTRIBUTES
                if touched:
                    violations.append(f"{_relative(path)}:{node.lineno}: from os import {touched}")
    assert violations == []


def test_only_plugins_load_external_modules() -> None:
    violations = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        in_plugins = PLUGINS_ROOT in path.parents
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            else:
                continue
            location = f"{_relative(path)}:{node.lineno}"
            if name in _EXTERNAL_LOADERS and not in_plugins:
                violations.append(f"{location}: {name}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in {"modules", "path", "meta_path", "path_hooks"}
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and (node.attr != "modules" or not in_plugins)
            ):
                violations.append(f"{location}: sys.{node.attr}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and not in_plugins:
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(module.split(".")[0] == "importlib" for module in modules):
                    violations.append(f"{_relative(path)}:{node.lineno}: imports importlib")
    assert violations == []


def test_no_torch_distributed_in_package() -> None:
    violations = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            location = f"{_relative(path)}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("torch.distributed") for alias in node.names):
                    violations.append(f"{location}: import torch.distributed")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("torch.distributed") or (
                    module == "torch" and "distributed" in _names_in(node)
                ):
                    violations.append(f"{location}: from {module} import ...")
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "distributed"
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            ):
                violations.append(f"{location}: torch.distributed")
    assert violations == []


def _top_level_imports(tree: ast.Module) -> list[tuple[int, str, bool]]:
    """Module-level imports as (line, module, guarded by try) triples."""
    imports = []

    def visit(statements: list[ast.stmt], guarded: bool) -> None:
        for statement in statements:
            if isinstance(statement, ast.Import):
                imports.extend((statement.lineno, a.name, guarded) for a in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                if statement.level == 0 and statement.module:
                    imports.append((statement.lineno, statement.module, guarded))
            elif isinstance(statement, ast.Try):
                visit(statement.body, True)
                for handler in statement.handlers:
                    visit(handler.body, guarded)
                visit(statement.orelse, guarded)
                visit(statement.finalbody, guarded)
            elif isinstance(statement, ast.If):
                visit(statement.body, guarded)
                visit(statement.orelse, guarded)

    visit(tree.body, False)
    return imports


def _matches(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def test_top_level_imports_are_torch_stdlib_or_lite_primitive() -> None:
    violations = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, module, guarded in _top_level_imports(tree):
            root = module.split(".")[0]
            if (
                root == "__future__"
                or root in sys.stdlib_module_names
                or _matches(module, _ALLOWED_TOP_LEVEL)
                or (guarded and _matches(module, _GUARDED_TOP_LEVEL))
            ):
                continue
            violations.append(f"{_relative(path)}:{line}: {module}")
    assert violations == []


def test_package_import_is_light() -> None:
    modules = sorted(
        ".".join((PACKAGE, *path.relative_to(PACKAGE_ROOT).with_suffix("").parts)).removesuffix(
            ".__init__"
        )
        for path in _package_files()
    )
    script = f"""
import importlib, json, sys
blocked = ("triton", "triton.language", "deep_gemm", "cudnn")
for name in blocked:
    sys.modules[name] = None  # any import of these raises ImportError
loaded = [importlib.import_module(name).__name__ for name in {modules!r}]
print(json.dumps({{
    "loaded": loaded,
    "still_blocked": [name for name in blocked if sys.modules.get(name, 0) is None],
    "megatron_core": sorted(n for n in sys.modules if n.split(".")[:2] == ["megatron", "core"]),
    "model_packages": sorted(n for n in sys.modules if n.startswith("megatron.lite.model")),
}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(LITE_ROOT), environment.get("PYTHONPATH")))
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded"] == modules
    assert report["still_blocked"] == ["triton", "triton.language", "deep_gemm", "cudnn"]
    assert report["megatron_core"] == []
    assert report["model_packages"] == []
