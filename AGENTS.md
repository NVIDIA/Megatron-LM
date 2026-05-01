# Repository Guidelines

## Contributing

### Pull Requests

- All PRs must be created as **drafts**. Use `gh pr create --draft` or the GitHub UI draft option.
- Never push branches directly to `https://github.com/NVIDIA/Megatron-LM`. You must push your branch to a personal fork (e.g. `https://github.com/<your-username>/Megatron-LM`), then open a PR from the fork's branch against `NVIDIA/Megatron-LM`.
- Read @docs/developer/contribute.md for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.
