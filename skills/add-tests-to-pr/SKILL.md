---
name: add-tests-to-pr
description: Orchestrator that adds inference tests to an existing pull request. Given a PR, it checks out the branch, asks which test types to add (unit / functional / performance) as checkboxes, authors them via the add-inference-* skills, verifies them on the cluster via the run-inference-* skills, then pushes a <branch>-tests branch to the PR's remote. - 'add tests to this PR', 'add inference tests to PR #1234', 'write and verify tests for this pull request', 'cover this PR with unit/functional/perf tests'.
when_to_use: A PR needs inference test coverage and you want one command to check it out, pick test types interactively, author + run them, and push a tests branch. Wraps add-inference-unit-tests, add-inference-functional-tests, add-inference-performance-test and their run-* counterparts.
---

# Add inference tests to a pull request

This is an **orchestrator skill**. It does not itself know how to write or run
any individual test — it sequences the six paired inference-test skills around
a single PR and handles the git plumbing (checkout, branch, push). The actual
authoring and execution is delegated, by name, to:

- Unit: [[add-inference-unit-tests]] → [[run-inference-unit-tests]]
- Functional: [[add-inference-functional-tests]] → [[run-inference-functional-tests]]
- Performance: [[add-inference-performance-test]] → [[run-inference-performance-tests]]

Cluster execution for every `run-*` step (and golden/baseline capture in the
functional and perf `add-*` steps) goes through cog — see [[cog-setup-and-help]].
The `run-*` skills auto-bootstrap cog if it is missing; don't reinvent that here.

## Inputs

The caller provides a **PR reference**: a number (`1234`), a URL, or `owner:branch`.
If none is given, ask for it before doing anything else.

---

## Step 1 — Check out the PR branch

```bash
# Confirm the PR exists and capture its head branch + head repo (the remote we push back to).
gh pr view <PR> --json number,headRefName,headRepositoryOwner,headRepository,isCrossRepository,url
gh pr checkout <PR>          # creates/updates the local branch tracking the PR head
ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
```

Record three things and keep them for later:
- `ORIG_BRANCH` — the PR's head branch name. The tests branch will be `${ORIG_BRANCH}-tests`.
- The **head remote** — where the PR branch lives. `gh pr checkout` adds it as a
  remote automatically. This is the push target (per the user's choice: push to
  the PR's own remote, not a separate fork). Resolve it with
  `git config branch.${ORIG_BRANCH}.remote`.
- Whether it is cross-repository (a contributor fork) or a branch on the base repo.

> **Guard (CLAUDE.md):** the repo policy is *never push branches directly to
> `NVIDIA/Megatron-LM`*. If the resolved head remote points at `NVIDIA/Megatron-LM`,
> stop and confirm with the user before pushing `${ORIG_BRANCH}-tests` there.

## Step 2 — Read the diff to scope the tests

```bash
gh pr diff <PR> --name-only            # which files changed
git diff main...HEAD -- megatron/core/inference   # the inference-relevant changes
```

Summarize, in one or two lines, **what changed under `megatron/core/inference`**
(new function, changed sampling path, new model wiring, perf-sensitive kernel,
etc.). This summary is the context you pass into each `add-*` skill so it tests
the right thing — do not skip it. If the PR touches nothing inference-related,
say so and ask the user whether to proceed anyway.

## Step 3 — Ask which test types to add (checkboxes)

Use `AskUserQuestion` with **multiSelect: true** — the user may pick any
combination:

- **Unit** — fast, local authoring; cluster only at the verify step.
- **Functional** — needs a checkpoint on the cluster FS + golden-value capture via cog.
- **Performance** — needs a model `.args` file + baseline capture via cog.

Show, next to each option, the rough cost (functional and perf both require cog +
a cluster run, so they are slower and need a registered cluster). Carry the
selected set into Step 4.

## Step 4 — For each selected type: author, then verify

Process the selected types **in this order** (cheapest first): unit → functional → perf.
For each one:

1. **Author.** Invoke the matching `add-*` skill, handing it the diff summary from
   Step 2. Let that skill drive its own questions (functional/perf will ask for
   model, parallelism, checkpoint path, test-case name — that is expected; do not
   try to answer for the user). It produces the test files:
   - unit → `tests/unit_tests/inference/<category>/test_*.py`
   - functional → `tests/functional_tests/test_cases/<family>/<test_case>/` (+ golden values)
   - perf → `tests/performance_tests/test_cases/<family>/<test_case>/` (+ baseline values)
2. **Lint imports (CLAUDE.md).** If you added or changed imports in any `.py` file,
   run `uv run isort` on those files before moving on.
3. **Verify on the cluster.** Invoke the matching `run-*` skill on the just-added
   test (a single test case where the run skill supports it, e.g.
   `family/test_case`). This is the "make sure it works" step the user asked for.
   - If it **passes**, record the result and continue.
   - If it **fails**, fix the test (or the captured golden/baseline) and re-run.
     Do not push failing tests. If a failure looks like a real product bug in the
     PR rather than a bad test, surface it to the user instead of masking it.

Keep a running tally of what was added and each verify result for the final report.

## Step 5 — Create the tests branch, commit, push

Only after every selected type passes its verify step:

```bash
git checkout -b "${ORIG_BRANCH}-tests"
git add tests/                      # stage only the new/updated test artifacts
git status                          # confirm scope — do NOT sweep in unrelated changes
git commit -m "test: add inference tests for ${ORIG_BRANCH}"   # imperative mood (contribute.md)
git push -u <head-remote> "${ORIG_BRANCH}-tests"
```

- Commit message subject in the **imperative mood**, proper English (contribute.md).
- Stage **only** the test files you created — never touch code outside the test
  scope (contribute.md: "Touch anything outside the stated scope").
- Push to the **head remote** resolved in Step 1 (subject to the NVIDIA guard).

## Step 6 — Report

Tell the user, concisely:
- Which test types were added and the file paths.
- The verify result for each (pass + key numbers, or what was fixed).
- The branch pushed and its remote.
- A ready-to-run draft-PR command (all PRs must be **drafts** per CLAUDE.md):
  ```bash
  gh pr create --draft --base <PR-base-branch> --head <head-owner>:${ORIG_BRANCH}-tests \
    --title "test: add inference tests for ${ORIG_BRANCH}"
  ```
  Offer to run it; do not open the PR unless the user asks.

---

## Notes & failure modes

- **Nothing selected** in Step 3 → stop, nothing to do.
- **cog not set up** → the `run-*` skills bootstrap it via [[cog-setup-and-help]];
  let them. If bootstrap itself fails, report the cog error code and stop before
  the git steps.
- **Functional/perf need cluster artifacts** (checkpoints, `.args` files) that may
  not exist. The `add-*` skills own that; if they block on a missing artifact,
  relay the blocker rather than fabricating one.
- **Verify is mandatory before push.** The whole point of this orchestrator is
  that the pushed branch contains *working* tests. Never skip Step 4.3 to save time.
- **Idempotency:** if `${ORIG_BRANCH}-tests` already exists, ask whether to reuse,
  amend, or pick a new suffix rather than force-pushing blindly.
