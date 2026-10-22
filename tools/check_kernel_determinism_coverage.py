# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Require determinism tests for kernel-related changes.

Run by the ``linting`` job in ``.github/workflows/cicd-main.yml`` on every PR push. It reads
the kernel registry in ``tests/unit_tests/determinism/kernels/manifest.py`` and inspects the
files the PR changes relative to its base:

1. A changed file that bears a kernel -- it lives in one of ``KERNEL_DIRECTORIES``, matches
   one of ``KERNEL_CONTENT_PATTERNS``, or is already registered -- must be registered in
   the manifest with a bit-exact test (or an explicit ``exempt_reason``). Not overridable:
   this is the "every kernel has a determinism test" invariant.
2. When a registered kernel source changes, at least one of its determinism tests must
   change in the same PR, so the test is revisited together with the kernel. Override by
   adding the ``EXEMPT_LABEL`` label to the PR (refactors, comment-only edits) or, locally,
   with ``--no-require-test-update``.

Standard library only -- the linting job has no torch. The manifest is loaded by file path
so importing it does not pull in ``tests.unit_tests``.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tests/unit_tests/determinism/kernels/manifest.py"
SOURCE_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h")


def load_manifest(path: Path = MANIFEST_PATH) -> ModuleType:
    """Import the manifest module from ``path`` without importing its parent packages."""
    spec = importlib.util.spec_from_file_location("kernel_determinism_manifest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load manifest from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules[module.__name__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def changed_files(base_ref: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return repo-relative paths added/copied/modified/renamed relative to ``base_ref``."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "--merge-base", base_ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_source_under_megatron(path: str) -> bool:
    """Return True for source files under ``megatron/`` (Python, CUDA, C++)."""
    return path.startswith("megatron/") and path.endswith(SOURCE_SUFFIXES)


def in_kernel_directory(path: str, kernel_directories: tuple[str, ...]) -> bool:
    """Return True if ``path`` lies under one of the manifest's kernel directories."""
    return any(path.startswith(directory.rstrip("/") + "/") for directory in kernel_directories)


def strip_comments_and_strings(text: str) -> str:
    """Blank out comments and string literals of Python source, preserving layout.

    Kernel patterns must match code, not prose: a docstring that mentions
    ``torch.compile`` is not a kernel. Tokens are replaced with spaces so line and
    column positions (and the ``^`` anchors in the patterns) are unchanged. Falls
    back to the raw text when the source does not tokenize.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return text
    lines = text.splitlines(keepends=True)
    skip = {tokenize.COMMENT, tokenize.STRING}
    skip |= {
        getattr(tokenize, name)
        for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
        if hasattr(tokenize, name)
    }
    for tok in tokens:
        if tok.type not in skip:
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        for row in range(srow, erow + 1):
            line = lines[row - 1]
            start = scol if row == srow else 0
            end = ecol if row == erow else len(line.rstrip("\r\n"))
            lines[row - 1] = line[:start] + " " * (end - start) + line[end:]
    return "".join(lines)


def matches_kernel_pattern(text: str, patterns) -> list[str]:
    """Return the patterns in ``patterns`` that match the code in ``text``."""
    code = strip_comments_and_strings(text)
    return [pattern for pattern in patterns if re.search(pattern, code, re.MULTILINE)]


def is_kernel_bearing(path: str, manifest: ModuleType, repo_root: Path = REPO_ROOT) -> str:
    """Return why ``path`` counts as kernel-bearing, or ``""`` if it does not."""
    if manifest.entries_for(path):
        return "registered in the manifest"
    if not is_source_under_megatron(path):
        return ""
    if path.endswith("__init__.py"):
        return ""
    if in_kernel_directory(path, manifest.KERNEL_DIRECTORIES):
        return "lives in a kernel directory"
    file = repo_root / path
    if not file.is_file():
        return ""
    try:
        text = file.read_text(errors="replace")
    except OSError:
        return ""
    if path.endswith((".cu", ".cuh")):
        return "is a CUDA source"
    hits = matches_kernel_pattern(text, manifest.KERNEL_CONTENT_PATTERNS)
    if hits:
        return f"matches kernel pattern(s) {', '.join(repr(h) for h in hits)}"
    return ""


def check(
    files: list[str],
    manifest: ModuleType,
    labels: set[str],
    require_test_update: bool = True,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Return human-readable violations for ``files`` (empty list means the PR passes)."""
    violations: list[str] = []
    changed = set(files)
    exempt = manifest.EXEMPT_LABEL in labels
    manifest_display = MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()

    for path in sorted(changed):
        reason = is_kernel_bearing(path, manifest, repo_root)
        if not reason:
            continue
        entries = manifest.entries_for(path)
        if not entries:
            violations.append(
                f"{path}: {reason} but is not registered in {manifest_display}. "
                "Add a KernelEntry naming its bit-exact determinism test (see "
                "docs/developer/determinism/testing.md)."
            )
            continue
        for entry in entries:
            if not entry.tests and not entry.exempt_reason:
                violations.append(
                    f"{path}: manifest entry {entry.name!r} has neither tests nor an exempt_reason."
                )
                continue
            if not require_test_update or not entry.tests:
                continue
            if any(test in changed for test in entry.tests):
                continue
            message = (
                f"{path}: kernel {entry.name!r} changed but none of its determinism tests did "
                f"({', '.join(entry.tests)}). Update the test to cover the change"
            )
            if exempt:
                logger.warning("%s -- allowed by label %r.", message, manifest.EXEMPT_LABEL)
            else:
                violations.append(
                    f"{message}, or add the {manifest.EXEMPT_LABEL!r} label if the change "
                    "cannot affect numerics."
                )
    return violations


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base-ref", default="origin/main", help="Git ref the PR is compared against."
    )
    parser.add_argument(
        "--files", nargs="*", help="Changed files to check instead of computing them with git."
    )
    parser.add_argument(
        "--labels", default="", help="Comma-separated PR labels (for the exemption label)."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--no-require-test-update",
        action="store_true",
        help="Only enforce registration; do not require the determinism test to change.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the check on the PR's changed files and return the process exit code."""
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)
    files = args.files if args.files is not None else changed_files(args.base_ref)
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    violations = check(files, manifest, labels, require_test_update=not args.no_require_test_update)
    if violations:
        logger.error(
            "Kernel determinism coverage check failed (%d issue(s)):\n%s",
            len(violations),
            "\n".join(f"  - {v}" for v in violations),
        )
        return 1
    logger.info("Kernel determinism coverage check passed for %d changed file(s).", len(files))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
