# Repository Guidelines

## Skills

The `skills/` directory contains structured guides for common tasks (running
tests, building containers, managing dependencies, submitting SLURM jobs, etc.).
**Always read the relevant `SKILL.md` before starting any task it covers —
skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Contributing

### Pull Requests

- All PRs must be created as **drafts**. Use `gh pr create --draft` or the GitHub UI draft option.
- Never push branches directly to `https://github.com/NVIDIA/Megatron-LM`. You must push your branch to a personal fork (e.g. `https://github.com/<your-username>/Megatron-LM`), then open a PR from the fork's branch against `NVIDIA/Megatron-LM`.
- Read @docs/developer/contribute.md for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Quality

- Megatron-LM linting is driven by `tools/autoformat.sh`, not the
  Megatron-Bridge pre-commit workflow. Before pushing Python changes under
  `megatron/core` or `tests`, run the CI-equivalent check:
  `BASE_REF=main CHECK_ONLY=true SKIP_DOCS=false bash tools/autoformat.sh`.
  This script requires Git 2.31.0 or newer.
- To apply the same formatting locally, run
  `BASE_REF=main CHECK_ONLY=false bash tools/autoformat.sh`.
- For a narrow changed-file pass, use the same tool settings as CI:
  `uv run black --skip-magic-trailing-comma --skip-string-normalization <files>`,
  `uv run isort <files>`, `uv run pylint <files>`, and
  `uv run ruff check <files>`.
- Black uses `line_length = 100` from `pyproject.toml` and
  `--skip-magic-trailing-comma`; do not rely on trailing commas to preserve
  multiline formatting.
- After editing imports in any Python files, always run `uv run isort` on
  those files to fix import order before committing.
