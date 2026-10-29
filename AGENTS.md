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
- When rebasing, cherry-picking or committing, sign the commits with both `-s` and `-S`: `-s` adds the required
  `Signed-off-by` trailer, and `-S` signs the commit so copy-pr-bot and
  `/ok to test` can verify the pushed commit without manually specifying the SHA.
  `/ok to test` is not required for PR authors in `NVIDIA/mcore-engineers` as long as
  they sign their commits with `-S`.
- Authors in `NVIDIA/mcore-engineers` should sign using their NVIDIA emails so they
  are automatically added to the right user groups on the internal Slack
  workspace.
- Before merging any PR, check for open PRs whose GitHub base/diffbase is
  `pull-request/<this PR number>`. Retarget every such dependent PR to `main`
  before merging; otherwise GitHub may close it and discard review discussion
  or approvals.
- Read [docs/developer/contribute.md](./docs/developer/contribute.md) for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Provenance

- Best effort to avoid copying code from other repositories. Implement changes
  independently.
- If copying code is necessary, add the source URL, revision, and license to
 all headers of files that contain such code. In addition, ensure the
 "Code provenance" section is filled in within the PR description.
- Preserve all required license and copyright notices and attribution for any
  external code in the proposed changes.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.

### Megatron Core Process Groups

The `parallel_state` functions that read the global parallel grid are
deprecated in 0.21 and removed in 0.23.
[docs/developer/parallel-state-deprecation.md](./docs/developer/parallel-state-deprecation.md)
lists their replacements and describes the warnings and checks summarized here.

- Code takes its process groups from the `ProcessGroupCollection` of the model
  that owns the operation, or an explicit `torch.distributed.ProcessGroup`,
  passed by its caller. New code requires its groups: do not add a `None`
  default that falls back to `parallel_state`. An entry point outside
  `megatron/core` that has no caller uses the collection that
  `initialize_model_parallel()` returned or
  `parallel_state.get_registered_process_groups()`.
- Deprecated: the `parallel_state` group, rank and size accessors, stage
  predicates, virtual-pipeline getters and setters, rank and size overrides,
  memory-buffer aliases, and `ProcessGroupCollection.use_mpu_process_groups()`.
  Called from outside `megatron/core`, they emit
  `ParallelStateDeprecationWarning`. Bootstrap (`initialize_model_parallel()`,
  `build_process_group_collection()`), teardown (`destroy_model_parallel()`)
  and the registry (`register_process_groups()`,
  `get_registered_process_groups()`) are not deprecated.
- A compatibility fallback for callers that do not pass their groups yet reads
  the grid only through the warned helpers of
  `megatron/core/process_groups_config.py`: `resolve_process_groups()` or
  another guarded helper, a `with global_process_group_fallback(...)` block
  around the reads, or, in `megatron/core` only,
  `warn_global_process_group_fallback()` before them. When it falls back, each
  emits a `ProcessGroupFallbackWarning` dated by `deprecated_in` and
  `removed_in`: 0.21 and 0.23, the defaults, for a fallback that first warns
  in 0.21. `resolve_gtp_remat_group(owner=None)` does not warn; it is only for
  a caller that has already warned. A comment does not make a read a fallback.
- In `megatron/core`, `tools/check_process_group_usage.py` counts the calls of
  the deprecated functions, of `use_mpu_process_groups()` and
  `get_registered_process_groups()`, and of the guarded helpers
  (`resolve_process_groups()`, `resolve_gtp_remat_group()`, and
  `ProcessGroupCollection.setup_process_groups_for_ddp()` and
  `setup_process_groups_for_optimizer()`). Only `parallel_state.py` and
  `process_groups_config.py` are exempt. A call that
  `tools/process_group_usage_allowlist.json` does not hold fails the check,
  even a warned or guarded one, and so does an allowlisted read that loses its
  warning. An allowlisted read may become warned or guarded in the same
  function or class; `--update` records that and drops the entries of removed
  reads. A read moved to another function or file needs a hand-edited entry
  that a reviewer approves.
- Outside `megatron/core`, `tests/unit_tests/test_parallel_state_accessor_uses.py`
  covers the code under `megatron/` (including `megatron/training`),
  `examples/`, `tools/`, `tasks/`, `scripts/` and `docs/`, and the top-level
  scripts; it skips `tasks/eval_utils.py`, which cannot be imported. It allows
  a call of a deprecated function or of `use_mpu_process_groups()` only in a
  `global_process_group_fallback()` block, including one that a function of
  the same module returns, or in bootstrap code that ignores
  `ParallelStateDeprecationWarning` in a `warnings.catch_warnings()` block.
  Under pytest, `pyproject.toml` ignores the warning when a test calls a
  deprecated function, since tests may use the global grid, and turns it into
  an error when the caller is a module of the `megatron`, `examples`, `tools`,
  `tasks` or `scripts` packages or one of the top-level scripts that it names,
  such as `megatron.training` or `pretrain_gpt`.
- To find fallbacks, train with `--strict-process-group-fallbacks` or run
  pytest with
  `-W error::megatron.core.process_groups_config.ProcessGroupFallbackWarning`.
- In reviews, flag a new read of the global grid or a new fallback in
  `megatron/core`, and a deprecated call outside the allowed blocks in the
  code that the guard covers. The checker's CI step only reports (see
  [#6307](https://github.com/NVIDIA/Megatron-LM/issues/6307)), so reviewers
  enforce it.
