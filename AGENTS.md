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

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.
