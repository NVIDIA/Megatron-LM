# Light review

A quick pass for things that are cheap to spot and expensive to miss. Keep it
concise and actionable — the value of this depth is that an author gets it back
in minutes, so a long list of maybes defeats the point.

Prerequisite: the mandatory workflow in `../SKILL.md` (diff → domain skills →
this file → review).

## Focus ONLY on

- Critical bugs or logic errors
- Typos in code, comments, or strings
- Missing or insufficient test coverage for changed code
  - If the PR adds a new feature or significant functionality without
    corresponding tests, suggest adding tests
  - If the PR fixes a bug that was not caught by an existing unit test, suggest
    adding a regression test to prevent recurrence
- Outdated or inaccurate documentation affected by the changes
- New direct global process group access in `megatron/core` production code
  - Flag added calls to `parallel_state.get_*_group()` or directly imported
    `get_*_group()` helpers unless they are in `parallel_state.py`,
    `process_groups_config.py`, initialization/bootstrap code that materializes
    a `ProcessGroupCollection`, tests, docs, or an explicitly documented
    migration fallback
  - Prefer passing a `ProcessGroupCollection` or explicit
    `torch.distributed.ProcessGroup` from the caller

## Do NOT comment on

- Style preferences or formatting
- Minor naming suggestions
- Architectural opinions or refactoring ideas
- Performance unless there is a clear, measurable issue

These are the strict-review categories. Raising them here turns a fast sanity
check into a design debate the author did not ask for; if they want that, they
will run `/claude strict-review`.

## Completion

It is perfectly acceptable to have nothing to comment on. If so, approve:

```bash
gh pr review $PR_NUMBER --repo $REPO --approve --body "LGTM"
```
