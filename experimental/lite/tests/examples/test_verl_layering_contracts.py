# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static boundaries for the optional VERL example connector."""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterable
from pathlib import Path

LITE_ROOT = Path(__file__).resolve().parents[2]
VERL_MLITE_ROOT = LITE_ROOT / "examples" / "verl" / "verl_mlite"

MODEL_PACKAGE_PREFIXES = (
    "megatron.lite.model.deepseek_v4",
    "megatron.lite.model.glm5",
    "megatron.lite.model.kimi_k2",
    "megatron.lite.model.qwen3_5",
    "megatron.lite.model.qwen3_moe",
)
DENIED_IMPORT_PREFIXES = ("examples.bench", *MODEL_PACKAGE_PREFIXES)


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


def _import_violations() -> list[str]:
    found: list[str] = []
    for path in _python_files(VERL_MLITE_ROOT):
        relative_path = path.relative_to(LITE_ROOT).as_posix()
        for lineno, module in _imported_modules(path):
            for prefix in DENIED_IMPORT_PREFIXES:
                if _matches_prefix(module, prefix):
                    found.append(f"{relative_path}:{lineno}: {module} matches denied {prefix}")
    return found


def _code_lines(path: Path) -> list[str]:
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
    for token in tokens:
        if token.type not in masked_types:
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            chars = blanked[row - 1]
            start = start_col if row == start_row else 0
            end = end_col if row == end_row else len(chars)
            for column in range(start, min(end, len(chars))):
                chars[column] = " "
    return ["".join(chars) for chars in blanked]


def _violations(paths: Iterable[Path], denied_terms: set[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        for lineno, line in enumerate(_code_lines(path), start=1):
            for term in sorted(denied_terms):
                if term in line:
                    relative_path = path.relative_to(LITE_ROOT)
                    found.append(f"{relative_path}:{lineno}: {term}")
    return found


def test_verl_connector_import_boundaries() -> None:
    assert _import_violations() == []


def test_verl_mlite_layer_does_not_see_model_internal_batch_fields() -> None:
    violations = _violations(
        _python_files(VERL_MLITE_ROOT), {"packed_seq_params", "position_ids", "to_bridge_dict"}
    )
    assert violations == []
