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
- Commit PR changes with both `-s` and `-S`: `-s` adds the required `Signed-off-by` trailer, and `-S` signs the commit so copy-pr-bot and `/ok to test` can verify the pushed commit without manually specifying the SHA.
- Read @docs/developer/contribute.md for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.

### Megatron Core Process Groups

- In `megatron/core` production code, avoid adding new direct reads of global
  process groups from `parallel_state` (for example,
  `parallel_state.get_tensor_model_parallel_group()` or directly imported
  `get_*_group()` helpers). Prefer accepting a `ProcessGroupCollection` or an
  explicit `torch.distributed.ProcessGroup` from the caller and passing that
  through.
- Allowed compatibility points include `megatron/core/parallel_state.py`,
  `megatron/core/process_groups_config.py`, initialization/bootstrap code that
  materializes a `ProcessGroupCollection` from MPU globals, tests, docs, and
  migration fallbacks with an explicit comment.
- This guidance targets Megatron Core library code. Do not apply it to
  `megatron/training` or other training-loop code unless the PR explicitly
  opts into that migration.
- In reviews, flag new direct `parallel_state.get_*_group()` usage in
  `megatron/core` unless it is one of the compatibility points above. This is
  advisory guidance, not a CI gate.
