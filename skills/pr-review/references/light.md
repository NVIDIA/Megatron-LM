# Light review

A quick pass for things that are cheap to spot and expensive to miss. Keep it
concise and actionable — the value of this depth is that an author gets it back
in minutes, so a long list of maybes defeats the point.

Prerequisite: the mandatory workflow in `../SKILL.md` (diff → domain skills →
this file → applicable style guidance → review).

## Focus ONLY on

- Critical bugs or logic errors
- Typos in code, comments, or strings
- Missing or insufficient test coverage for changed code
  - If the PR adds a new feature or significant functionality without
    corresponding tests, suggest adding tests
  - If the PR fixes a bug that was not caught by an existing unit test, suggest
    adding a regression test to prevent recurrence
- Outdated or inaccurate documentation affected by the changes
- Clear violations of [style-guide.md](../../../style-guide.md) in changed
  Python code that configured linters/formatters do not enforce
- New reads of the global parallel grid (see
  [parallel-state-deprecation.md](../../../docs/developer/parallel-state-deprecation.md))
  - In `megatron/core`, outside `parallel_state.py` and
    `process_groups_config.py`, flag added calls of the deprecated
    `parallel_state` functions, `use_mpu_process_groups()`,
    `get_registered_process_groups()` or the guarded helpers of
    `process_groups_config.py`, such as `resolve_process_groups()`, unless the
    call replaces an existing read in the same function
  - Outside `megatron/core` and the tests, including `megatron/training`, flag
    a deprecated call or `use_mpu_process_groups()` that is not in a
    `global_process_group_fallback()` block or in bootstrap code that ignores
    `ParallelStateDeprecationWarning`
  - Prefer passing a `ProcessGroupCollection` or explicit
    `torch.distributed.ProcessGroup` from the caller
- GPU kernel changes without a determinism test
  - If the PR adds or changes a GPU kernel (Triton kernel,
    `jit_fuser`/`torch.compile` function, CUDA extension, Transformer Engine
    or external-library dispatch, or a scatter/index accumulation), check that
    a bit-exact determinism test under
    `tests/unit_tests/determinism/kernels/` is added or updated and that the
    kernel is registered in `tests/unit_tests/determinism/kernels/manifest.py`
    (see `docs/developer/determinism/testing.md`)

## Do NOT comment on

- Subjective style preferences or issues already covered by configured linters/formatters
- Minor naming suggestions
- Architectural opinions or refactoring ideas unrelated to style-guide.md
- Performance unless there is a clear, measurable issue

These are the strict-review categories. Raising them here turns a fast sanity
check into a design debate the author did not ask for; if they want that, they
will run `/review mode=strict`.

## Completion

It is perfectly acceptable to have nothing to comment on. If so, approve:

```bash
gh pr review $PR_NUMBER --repo $REPO --approve --body "LGTM"
```
