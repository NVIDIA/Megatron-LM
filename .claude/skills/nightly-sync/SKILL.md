---
name: nightly-sync
description: Domain knowledge for the nightly main-to-dev sync workflow. Covers merge strategy, CI architecture, failure investigation, and known issues.
---

# Nightly Sync: Main to Dev

This skill is read by the automated sync bot during the nightly-sync-main-to-dev
workflow. It contains all domain knowledge for merging main into dev, resolving
conflicts, iterating on CI, and shipping the PR.

---

## Phase 1: Create the Sync Branch and Merge

### Branch Setup

1. Create branch `$BRANCH` from `origin/dev`
2. Merge: `git merge origin/main -X theirs --no-edit`
3. If conflicts remain (e.g. add/add), resolve by favoring main

### Preserving Dev-Only Additions

Do NOT blanket-override all shared files with main's version. Dev has features
not yet in main (new classes, new modules, new tests). The merge preserves both
sides' non-conflicting additions — only intervene where there is an actual
conflict.

### Squash-Merge Chain Detection

Dev often develops features as a chain of PRs (PR1 → PR2 → PR3) where each
builds on the last. When PR1 is squash-merged to main, git sees main's squashed
version and dev's original commits as unrelated changes. `-X theirs` will pick
main's PR1 code and silently discard PR2/PR3's improvements on dev.

After the merge, check for this pattern:

1. For each file where `-X theirs` resolved a conflict, run
   `git log --oneline origin/dev -- <file>` to see if dev has commits that
   came AFTER the code main is bringing in.
2. If dev has follow-up commits (bug fixes, refactors, extensions), **favor
   dev's version** for those sections.
3. If the conflict is just main bringing in a clean copy of what dev already
   has (no follow-ups), main's version is fine.

Practical check: run `git diff origin/dev -- <file>` on conflicted files. If
dev's code was removed or reverted, investigate whether dev's version is the
more evolved one.

### Files to Override from Main

These files have known semantic conflicts where dev's versions reference args
or APIs that main removed or renamed. Take main's version with
`git checkout origin/main -- <file>`:

- `pyproject.toml` and `uv.lock` — lock-file consistency for container build
- `docker/Dockerfile.ci.dev` — build compatibility
- `megatron/training/training.py` — references dev-only args
- `megatron/training/initialize.py` — references dev-only args
- `megatron/training/utils.py` — references dev-only args
- `megatron/training/datasets/data_samplers.py` — references dev-only args
- `megatron/core/optimizer/layer_wise_optimizer.py` — constructor signature

**Important:** Taking main's `pyproject.toml`/`uv.lock` changes the container's
dependency versions. This can break dev-only code that depends on newer library
versions (e.g. TransformerEngine features). If CI failures trace back to version
mismatches, you may need to keep dev's dependency versions for specific packages.

### Special Handling: data_schedule.py

Main and dev have completely different classes in this file:
- Main: `HybridCPDataLoaderWrapper` (imported by main's `training.py`)
- Dev: `BasePackingScheduler`, `DpBalancedScheduler`,
  `DefaultDynamicCPScheduler`, `wrap_data_iterator`,
  `get_batch_on_this_rank_for_sequence_packing` (imported by `pretrain_gpt.py`
  and tests)

**Do NOT take either version wholesale.** Keep dev's file and append main's
`HybridCPDataLoaderWrapper` class (plus any missing imports like
`BalancedCPScheduler`, `Any`, `List`) at the end.

### Restore Deleted Files

Compare `git ls-tree` between `origin/main` and HEAD to find files in main
that are missing from the merged tree. For each:
- **Restore** if main's code imports/references it and would break without it
  (e.g. `hybrid_cp_schedule.py` if `data_schedule.py` imports from it)
- **Do NOT restore** if dev intentionally deleted it — check
  `git log origin/dev -- <file>` for the deletion commit to understand intent
- When in doubt, check whether any file in the merged tree imports from the
  missing file. If nothing imports it, skip it.

### Formatting

Run on ALL changed Python files (relative to `origin/dev`), in this order:

1. `black` (version 24, `--config pyproject.toml`)
2. `isort`
3. Order matters: black first, then isort — reverse order can undo isort's work
4. `pylint` on changed `megatron/core/` files — fix missing-docstring and
   line-too-long violations before pushing

### Commit and Push

Commit everything and push the branch.

---

## Phase 2: Create the Draft PR

- Title: `chore: nightly sync main into dev ($DATE)`
- Create as **draft**: `gh pr create --draft`
- Body should include:
  1. Summary of what was synced (number of commits from main)
  2. List of files where main's version was taken over the merge
  3. List of files that were deleted in dev but restored (and why)
  4. The remerge-diff output (`git show --remerge-diff HEAD` on the merge
     commit) so reviewers can inspect ONLY the conflict resolutions. If the
     output is very long, summarize conflicts by file and put the full diff
     in a collapsed `<details>` block. If git is too old for `--remerge-diff`,
     note the git version and describe the merge strategy used instead.
- Save the PR number for later phases

---

## Phase 3: CI Iteration

### CI Architecture

- **`Nemo_CICD_Test`** is a downstream gate job aggregating unit test,
  integration test, and other results. If it fails, investigate the upstream
  jobs it depends on — do NOT debug the gate itself.
- **Integration tests** (H100, GB200) may be skipped for non-maintainer PRs.
  This is expected; the `Nemo_CICD_Test` gate will fail as a result.
- **`tests/unit_tests/conftest.py`** imports from `megatron.training.training`,
  so a broken import in `training.py` (or anything it transitively imports)
  cascades to fail ALL test suites. If every test job fails with ImportError,
  check the training.py import chain first.

### Trigger and Poll

1. Comment on PR: `/ok to test <SHA>`
2. Find CI run: `gh run list --json` filtered by
   `headBranch == "pull-request/<PR_NUMBER>"` (no `--branch` flag available)
3. Poll with a foreground Bash while-loop (`sleep 120`) using:
   `gh api repos/$REPO/actions/runs/<RUN_ID>/jobs?per_page=100`
4. As soon as ANY job fails, investigate immediately — don't wait for the
   full run. Other jobs continue in parallel while you diagnose and fix.

### Failure Investigation

1. Fetch logs: `gh api repos/$REPO/actions/jobs/<JOB_ID>/logs`
2. Grep for: `ImportError`, `ModuleNotFoundError`, `FAILED`,
   `would reformat`, `line-too-long`, `Traceback`
3. Read the error, understand root cause, fix the code

### Common Issues

- **ImportError for a class/module:** Dev test imports a class from a file
  where we took main's version. Restore only the missing class/function —
  not the entire file. If a file's classes are completely different between
  main and dev, keep both sets of code.
- **Formatting failures (black/pylint):** Run `black --config pyproject.toml`
  on offending files. For pylint long-line or missing-docstring, edit directly.
- **Circular imports:** `isort` can reorder imports in a way that introduces
  circular dependencies (e.g. `megatron/legacy/model/__init__.py`). Check
  `git diff` on `__init__.py` files to see if import order changed.
- **Dependency version mismatches:** Taking main's `pyproject.toml`/`uv.lock`
  can change library versions in the CI container. Dev-only code may depend on
  newer versions (e.g. TransformerEngine's `single_grouped_weight`). If failures
  trace to missing kwargs or changed APIs in third-party libs, this is the cause.

### Pre-Existing Failure Verification

**You MUST empirically verify before classifying any failure as pre-existing.**

1. `gh pr list --repo $REPO --base dev --state merged --limit 3`
2. `gh pr checks <PR_NUMBER> --repo $REPO` on a recently merged dev PR
3. If the same test bucket **passes on recent dev CI** → the failure is
   sync-caused. You must fix it.
4. Only if the test **also fails on recent dev CI** can you classify it as
   pre-existing. Document with the dev PR number and CI run as evidence.

### Fix and Re-trigger

- Fix sync-caused failures, commit, push
- Re-trigger CI immediately (`/ok to test <new SHA>`)
- Don't wait for the rest of the current run to finish
- Repeat until all fixable issues are resolved

---

## Phase 4: Mark PR Ready

Only after the FINAL CI run has **completed** and previously-failing jobs
now **pass** (or are empirically verified as pre-existing):

```
gh pr ready <PR_NUMBER> --repo $REPO
```

"Fixed" means the fix was pushed AND the CI run finished AND the job passes.
Pushing a fix is not enough — wait for confirmation.

Comment on PR confirming which checks passed, listing any pre-existing
failures with evidence (dev PR number + CI result), and stating it is ready
for human review.

---

## Rules

- Prioritize main over dev on genuine conflicts. Preserve dev-only additions
  that do not conflict.
- CI triggers via comment: `/ok to test <sha>`
- CI runs appear on branch `pull-request/<PR_NUMBER>`
- Git committer identity: `svcnvidia-nemo-ci`
- After editing imports, run `isort` on those files
- **Push directly to NVIDIA/Megatron-LM** (not a fork). The bot uses a PAT
  with write access. CLAUDE.md says "never push directly" but that rule is
  for human contributors — the sync bot is an exception.
