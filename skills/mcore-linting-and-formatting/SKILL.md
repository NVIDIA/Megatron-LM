---
name: mcore-linting-and-formatting
description: Linting and formatting for Megatron-LM. Covers running autoformat.sh, tools (ruff, black, isort, pylint, mypy), and code style rules.
license: Apache-2.0
when_to_use: Running linting or autoformat; fixing style violations before a PR; 'pre-commit fails', 'ruff error', 'isort', 'mypy', 'style violation', 'how do I format', 'autoformat.sh'.
metadata:
  author: Oliver Koenig <okoenig@nvidia.com>
---

# Linting and Formatting

---

## Running the Formatter

Run before opening a PR:

```bash
# Check mode (no changes applied)
BASE_REF=main CHECK_ONLY=true SKIP_DOCS=false bash tools/autoformat.sh

# Fix mode
BASE_REF=main CHECK_ONLY=false bash tools/autoformat.sh
```

Tools invoked: `black`, `isort`, `pylint`, `ruff`, `mypy`.

---

## Import Ordering

After editing imports in any Python files, always run `uv run isort` on those
files before committing:

```bash
uv run isort <file1>.py <file2>.py
```

---

## Setting Up the Linting Group

Inside the development container, from the repository root:

```bash
uv sync --locked --only-group linting
export PATH="$PWD/.venv/bin:$PATH"
```

This installs `ruff`, `black`, `isort`, `pylint` — the same tools used by
`tools/autoformat.sh` and CI's `linting` job.

---

## Code Style Rules

- **Type hints**: required on all public API functions. Prefer `X | None` for new interfaces; preserve nearby annotation conventions when extending existing code.
- **Docstrings**: Google-style on all public classes and functions.
- **Naming**: follow Python conventions — `snake_case` for functions and variables, `PascalCase` for classes.
- **Line length**: 100 characters (configured in `pyproject.toml` and `.pylintrc`).
- **No bare `except`**: always catch specific exception types.

See the [repository style guide](../../style-guide.md) for the full conventions
and the distinction between recommendations and enforced checks.
