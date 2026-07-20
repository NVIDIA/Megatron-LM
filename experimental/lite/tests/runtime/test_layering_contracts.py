# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static layering guards for MLite public data boundaries and imports.

Bench, runtime, model, and primitive layers have directional import boundaries.
The bench layer must hand the runtime only a model-agnostic batch:
``packed_seq_params`` / ``position_ids`` are transient THD metadata that may be
materialized only at the immediate forward boundary, inside the explicitly marked
allow range in the bridge runtime.

The guard enforces the contract; it does not police prose. Substring scans ignore
string/comment content (so a docstring may still document a dependency), while the
import denylist enforces the core layer boundaries.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterable
from pathlib import Path

LITE_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = LITE_ROOT / "examples" / "bench"
RUNTIME_ROOT = LITE_ROOT / "megatron" / "lite" / "runtime"
MODEL_ROOT = LITE_ROOT / "megatron" / "lite" / "model"
PRIMITIVE_ROOT = LITE_ROOT / "megatron" / "lite" / "primitive"
BRIDGE_RUNTIME = RUNTIME_ROOT / "backends" / "bridge" / "runtime.py"

ALLOW_BEGIN = "MLITE_LAYERING_ALLOW_BRIDGE_FORWARD_METADATA_BEGIN"
ALLOW_END = "MLITE_LAYERING_ALLOW_BRIDGE_FORWARD_METADATA_END"
LAYER_ROOTS = {
    "bench": BENCH_ROOT,
    "runtime": RUNTIME_ROOT,
    "model": MODEL_ROOT,
    "primitive": PRIMITIVE_ROOT,
}
MODEL_PACKAGE_PREFIXES = (
    "megatron.lite.model.deepseek_v4",
    "megatron.lite.model.glm5",
    "megatron.lite.model.kimi_k2",
    "megatron.lite.model.qwen3_5",
    "megatron.lite.model.qwen3_moe",
)
MODEL_NAME_TERMS = {"deepseek_v4", "glm5", "kimi_k2", "qwen3", "qwen3_5", "qwen3_moe"}
DENIED_IMPORT_PREFIXES = {
    "bench": ("examples.verl", "megatron.lite.model", "verl", "verl_mlite"),
    "runtime": ("examples", *MODEL_PACKAGE_PREFIXES, "verl", "verl_mlite"),
    "model": ("examples", "megatron.lite.runtime.backends", "verl", "verl_mlite"),
    "primitive": (
        "examples",
        "megatron.lite.model",
        "megatron.lite.runtime.backends",
        "megatron.lite.runtime.megatron_utils",
        "verl",
        "verl_mlite",
    ),
}
IMPORT_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "megatron/lite/primitive/ops/linear_cross_entropy.py": (
        "verl.utils.kernel.linear_cross_entropy",
    ),
}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _allowlisted_import(relative_path: str, module: str) -> bool:
    return any(
        _matches_prefix(module, allowed)
        for allowed in IMPORT_ALLOWLIST.get(relative_path, ())
    )


def _code_lines(path: Path) -> list[str]:
    """Return source lines with string/comment spans blanked out.

    The contract guard targets executable references, not documentation: a
    docstring or comment may legitimately mention ``packed_seq_params`` or a model
    name to explain provenance. Blanking string/comment tokens keeps those out of
    the substring scan while preserving exact line numbers. Falls back to raw lines
    if the file does not tokenize (fail toward flagging, never toward hiding).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blanked = [list(line) for line in lines]
    masked_types = {tokenize.STRING, tokenize.COMMENT}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        masked_types.add(fstring_middle)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return lines
    for tok in tokens:
        if tok.type not in masked_types:
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            chars = blanked[row - 1]
            start = scol if row == srow else 0
            end = ecol if row == erow else len(chars)
            for col in range(start, min(end, len(chars))):
                chars[col] = " "
    return ["".join(chars) for chars in blanked]


def _allow_ranges(path: Path) -> list[range]:
    if path != BRIDGE_RUNTIME:
        return []

    ranges: list[range] = []
    start: int | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if ALLOW_BEGIN in line:
            if start is not None:
                raise AssertionError(f"nested allow range in {path}")
            start = lineno
        elif ALLOW_END in line:
            if start is None:
                raise AssertionError(f"unmatched allow range end in {path}:{lineno}")
            ranges.append(range(start, lineno + 1))
            start = None
    if start is not None:
        raise AssertionError(f"unclosed allow range in {path}")
    return ranges


def _violations(paths: Iterable[Path], denied_terms: set[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        ranges = _allow_ranges(path)
        for lineno, line in enumerate(_code_lines(path), start=1):
            if any(lineno in allowed for allowed in ranges):
                continue
            for term in sorted(denied_terms):
                if term in line:
                    rel = path.relative_to(LITE_ROOT)
                    found.append(f"{rel}:{lineno}: {term}")
    return found


def _import_violations(layer: str) -> list[str]:
    denied = DENIED_IMPORT_PREFIXES[layer]
    found: list[str] = []
    for path in _python_files(LAYER_ROOTS[layer]):
        rel = path.relative_to(LITE_ROOT).as_posix()
        for lineno, module in _imported_modules(path):
            if _allowlisted_import(rel, module):
                continue
            for prefix in denied:
                if _matches_prefix(module, prefix):
                    found.append(f"{rel}:{lineno}: {module} matches denied {prefix}")
    return found


def test_layer_import_boundaries() -> None:
    violations = []
    for layer in LAYER_ROOTS:
        violations.extend(_import_violations(layer))
    assert violations == []


def test_import_allowlist_entries_are_live() -> None:
    stale: list[str] = []
    for relative_path, allowed_modules in IMPORT_ALLOWLIST.items():
        path = LITE_ROOT / relative_path
        if not path.is_file():
            stale.append(f"{relative_path}: file missing")
            continue
        modules = {module for _, module in _imported_modules(path)}
        for allowed_module in allowed_modules:
            if not any(_matches_prefix(module, allowed_module) for module in modules):
                stale.append(f"{relative_path}: no import matches {allowed_module}")
    assert stale == []


def test_bench_layer_does_not_see_model_internal_batch_fields() -> None:
    violations = _violations(
        _python_files(BENCH_ROOT),
        {"packed_seq_params", "position_ids", "to_bridge_dict"},
    )
    assert violations == []


def test_runtime_packed_seq_params_is_bridge_forward_transient_only() -> None:
    violations = _violations(_python_files(RUNTIME_ROOT), {"packed_seq_params"})
    assert violations == []


def test_primitive_layer_is_model_name_agnostic() -> None:
    violations = _violations(_python_files(PRIMITIVE_ROOT), MODEL_NAME_TERMS)
    assert violations == []
