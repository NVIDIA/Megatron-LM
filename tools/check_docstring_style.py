# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""
Check that docstrings use the Google style expected by the documentation builder.

The Sphinx configuration renders every megatron/core docstring through
`sphinx.ext.napoleon.docstring.GoogleDocstring` (see `docs/conf.py` and
`docs/autodoc2_docstrings_parser.py`). Docstrings written in other styles are passed through
unconverted and render as garbled text in the generated API docs. Ruff's pydocstyle rules
(`convention=google`, see `pyproject.toml`) catch most of this, but they cannot detect:

- reST/Sphinx-style field lists (`:param x:`, `:returns:`, ...),
- NumPy-style sections (e.g., `Parameters` followed by a dashed underline), including header names
  Ruff's pydocstyle rules do not know (e.g., `Params` / `Inputs`),
- RST roles (``:class:`X```, ``:meth:`X```, ...) and RST directives (`.. note::`, ...). Docstrings
  are rendered as MyST markdown (`autodoc2_render_plugin = "myst"` in `docs/conf.py`), so references
  must use MyST syntax (``{class}`X```) and notes/admonitions must use MyST directive fences, e.g.:

      ```{note}
      Something noteworthy.
      ```

This script flags both. Run it as part of `tools/autoformat.sh` or standalone:

    python3 tools/check_docstring_style.py megatron/core
"""

import argparse
import ast
import re
import sys
from pathlib import Path

# NumPy-style section headers, each followed by a line of dashes. Includes informal drift (e.g.,
# "Params", "Inputs") that Ruff's pydocstyle rules cannot see because they only know Google section
# names.
NUMPY_SECTION_NAMES = {
    "Shortcuts",
    "Parameters",
    "Params",
    "Inputs",
    "Outputs",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "See Also",
    "Notes",
    "Warnings",
    "Attributes",
    "Methods",
    "Examples",
    "Extended Summary",
    "References",
    "Versionadded",
    "Versionchanged",
    "Deprecated",
}
# Also catch NumPy-style headers with names unknown to Google style (Ruff's pydocstyle rules only
# recognize Google section names and miss these).
NUMPY_SECTION_RE = re.compile(r"^\s{0,8}(\w[\w ]*?)\s*$")
SECTION_UNDERLINE_RE = re.compile(r"^\s*[-=~=+]{3,}\s*$")
# reST/Sphinx-style field lists.
REST_FIELD_RE = re.compile(
    r"^\s*:(param|parameter|arg|argument|key|keyword|type|returns?|rtype|"
    r"raises?|except|exception|yield[s]?|ytype|var[vtype]?|meta)\b",
    re.IGNORECASE,
)
# RST roles, e.g., ":class:`X`" – must be MyST roles ("{class}`X`") instead. Role names must start
# with a letter so that unrelated colon patterns (e.g., times like "12:00:`x`") do not trigger false
# positives.
RST_ROLE_RE = re.compile(r":([a-zA-Z][\w:-]*):`([^`]*)`")
# RST directives, e.g., ".. note::" or ".. code-block:: python" – must be MyST fenced directives
# instead. Optional trailing arguments are part of the directive, so the line is not required to end
# at "::".
RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+[\w-]+\s*::")
# Opening fence of a MyST directive (e.g., "```{note}"); its content is regular docstring markup and
# must still be checked.
MYST_DIRECTIVE_FENCE_RE = re.compile(r"^(```|~~~)\s*\{")


def check_docstring(filename: Path, name: str, lineno: int, docstring: str) -> list[str]:
    """Return a list of style violations for one docstring."""
    issues = []
    lines = docstring.splitlines()
    in_code_fence = False
    pending_directive_close = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            if pending_directive_close:
                # Closing fence of a MyST directive fence; its content was checked like normal
                # docstring text.
                pending_directive_close = False
            elif MYST_DIRECTIVE_FENCE_RE.match(stripped):
                pending_directive_close = True
            else:
                # Plain or language-tagged code fence; skip example code.
                in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        if RST_DIRECTIVE_RE.match(line):
            issues.append(
                f"{filename}:{lineno + i}: RST directive (docstring '{name}'); "
                "use a MyST fenced directive (e.g., '```{note}')"
            )
        for match in RST_ROLE_RE.finditer(line):
            issues.append(
                f"{filename}:{lineno + i}: RST role ':{match.group(1)}:' "
                f"(docstring '{name}'); use the MyST role "
                f"'{{{match.group(1)}}}'"
            )
        match = NUMPY_SECTION_RE.match(line)
        if (
            match
            and match.group(1) in NUMPY_SECTION_NAMES
            and i + 1 < len(lines)
            # The NumPy spec requires the underline to be at least as long
            # as the section title; shorter dashes are plain prose/dividers.
            and len(lines[i + 1].strip()) >= len(match.group(1))
            and SECTION_UNDERLINE_RE.match(lines[i + 1])
        ):
            issues.append(
                f"{filename}:{lineno + i}: NumPy-style section "
                f"'{match.group(1)}' (docstring '{name}'); use Google style "
                f"(e.g., '{match.group(1)}:')"
            )
        if REST_FIELD_RE.match(line):
            issues.append(
                f"{filename}:{lineno + i}: reST-style field list "
                f"(docstring '{name}'); use Google style (e.g., 'Args:')"
            )
    return issues


def iter_docstrings(tree: ast.AST):
    """Yield (node name, line number, docstring) for every documented node."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc_node = node.body[0] if node.body else None
        if isinstance(doc_node, ast.Expr) and isinstance(doc_node.value, ast.Constant):
            value = doc_node.value.value
            if isinstance(value, str) and value.strip():
                name = getattr(node, "name", "<module>")
                yield name, doc_node.lineno, value


def main() -> int:
    """Run the checker over the given files or directories."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="Python files or directories to check")
    args = parser.parse_args()

    issues = []
    for raw_path in args.paths:
        path = Path(raw_path)
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for py_file in files:
            # Docstring rules are not enforced in tests, consistent with Ruff's per-file-ignores for
            # "tests/**" and "*_test.py".
            parts = py_file.parts
            if "tests" in parts or py_file.name.endswith("_test.py"):
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError as e:
                issues.append(f"{py_file}:{e.lineno}: SyntaxError: {e.msg}")
                continue
            for name, lineno, docstring in iter_docstrings(tree):
                issues.extend(check_docstring(py_file, name, lineno, docstring))

    for issue in issues:
        print(issue)
    print(f"Checked docstring styles, found {len(issues)} issue(s).")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
