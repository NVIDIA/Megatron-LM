# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static import boundary and fresh-process full-package import guard."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "megatron/lite"


def test_vllm_imports_are_allowlisted_and_delayed():
    violations = []
    paths = [*ROOT.rglob("*.py"), *(ROOT.parents[1] / "examples/verl").rglob("*.py")]
    for path in paths:
        relative = (
            path.relative_to(ROOT).as_posix()
            if ROOT in path.parents
            else path.relative_to(ROOT.parents[1]).as_posix()
        )
        allowed = relative.startswith(
            ("model/deepseek_v41/vllm/", "examples/verl/verl_mlite/rollout/")
        )

        def visit(node, in_function=False):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    names = [node.args[0].value]
            if any(isinstance(n, str) and (n == "vllm" or n.startswith("vllm.")) for n in names):
                if not allowed or not in_function:
                    violations.append((relative, node.lineno))
            for field, value in ast.iter_fields(node):
                deferred = in_function or (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                    and field == "body"
                )
                for child in value if isinstance(value, list) else [value]:
                    if isinstance(child, ast.AST):
                        visit(child, deferred)

        visit(ast.parse(path.read_text()))
    assert not violations, violations


def test_full_package_walk_never_requests_vllm(tmp_path):
    script = r'''
import importlib, importlib.abc, json, pathlib, sys
requests = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            requests.append(fullname)
            raise AssertionError("FORBIDDEN_VLLM_IMPORT:" + fullname)
sys.meta_path.insert(0, Guard())
import megatron.lite
root = pathlib.Path(megatron.lite.__file__).parent
errors, imported = {}, []
for path in sorted(root.rglob("*.py")):
    pieces = path.relative_to(root).with_suffix("").parts
    if pieces[-1] == "__init__": pieces = pieces[:-1]
    name = ".".join(("megatron", "lite", *pieces))
    try:
        importlib.import_module(name)
        imported.append(name)
    except Exception as exc:
        errors[name] = type(exc).__name__ + ":" + str(exc)
loaded = [n for n in sys.modules if n == "vllm" or n.startswith("vllm.")]
report = {"requests": requests, "loaded": loaded, "imported": imported, "other_errors": errors}
pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
assert not requests and not loaded, report
for required in ("megatron.lite.model.deepseek_v41.lite.protocol", "megatron.lite.model.deepseek_v41.vllm.linear", "megatron.lite.model.deepseek_v41.vllm.mhc"):
    assert required in imported, errors.get(required)
try:
    importlib.import_module("vllm")
except AssertionError:
    pass
else:
    raise AssertionError("import guard positive control did not reject vllm")
'''
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "walk.json")], check=True, timeout=240
    )
