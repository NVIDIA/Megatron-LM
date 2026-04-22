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

- `megatron/training/training.py` — references dev-only args
- `megatron/training/initialize.py` — references dev-only args
- `megatron/training/utils.py` — references dev-only args
- `megatron/training/datasets/data_samplers.py` — references dev-only args
- `megatron/core/optimizer/layer_wise_optimizer.py` — constructor signature

**IMPORTANT: Do NOT take main's `pyproject.toml`, `uv.lock`, or
`docker/Dockerfile.ci.dev`.** These three files are a tightly coupled
triple — the Dockerfile's `uv sync` command must match the dependency
groups in `pyproject.toml`, and `uv.lock` must be consistent with both.
Main's versions are missing dev-only dependencies (e.g.
`fast-hadamard-transform`, correct TransformerEngine revision) and the
`--group no_pypi_wheels` flag needed to install them. Keep dev's versions
of all three files.

**NEVER manually edit `uv.lock`.** It is a machine-generated lockfile. If
it needs to change, it must be regenerated with `uv lock` inside a CUDA
container (see `.claude/skills/build-and-test/SKILL.md`).

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

### Execution model: one step, no background

You run inside ONE GitHub Actions step. The moment you stop emitting
tool calls, the step ends and the runner container is destroyed. Any
background process you started dies with it. There is NO persistent
session and NO future wakeup. See the workflow prompt's "NO background
tasks" block for the full ban list.

Practical rule: every wait for CI to resolve is a SINGLE foreground Bash
tool call that blocks inline until the wait is resolved.

### The Fix-Then-Retrigger Loop

Two nested loops. Do NOT conflate them:

- The **outer loop** is YOUR sequence of tool calls (each iteration: one
  `/ok to test`, one blocking poll, maybe one fix-and-push). It is NOT a
  Bash loop. It advances because you make new tool calls.
- The **inner loop** is a single blocking Bash tool call using
  `while true; do ... sleep 120; done`. It runs during one iteration of
  the outer loop and ends when CI reaches a terminal state for that
  iteration.

The outer loop terminates ONLY when Phase 4's gate is satisfied.

**Source of truth:** `gh pr view <PR_NUMBER> --repo $REPO --json statusCheckRollup`.
This lists every required check, including external status contexts
(GitLab CI, `copy-pr-bot`, etc.) that `gh api .../actions/runs/.../jobs`
does NOT show.

**Outer-loop iteration (each iteration is a few tool calls):**

1. `latest_sha=$(git rev-parse HEAD)` (one Bash call).
2. Post `/ok to test $latest_sha` on the PR:
   `gh pr comment <PR_NUMBER> --repo $REPO --body "/ok to test $latest_sha"`
3. ONE blocking Bash tool call. This is the inner loop. Copy this
   template verbatim, only changing `REPO` and `PR`:

   ```bash
   REPO='NVIDIA/Megatron-LM'
   PR='<PR_NUMBER>'
   # Names matched case-insensitively, anchored to the START of the name.
   EXEMPT='copy-pr-bot|is-not-external-contributor|greptile|coderabbit|codeowners|.*review|.*approval|codecov|coverage|build-docs|doc-build|readthedocs|sphinx'
   # Sentinel check that tells us CI has fully run. Update this if the
   # aggregate gate job is renamed.
   SENTINEL='Nemo_CICD_Test'

   while true; do
     # Normalize both CheckRun (.status / .conclusion) and StatusContext
     # (.state) entries into the same {name, status, conclusion} shape.
     rollup=$(gh pr view "$PR" --repo "$REPO" --json statusCheckRollup --jq '
       .statusCheckRollup[] | [
         (.name // .context // "?"),
         (if .__typename == "StatusContext" then
            (if (.state == "PENDING" or .state == "EXPECTED") then "IN_PROGRESS"
             else "COMPLETED" end)
          else (.status // "UNKNOWN") end),
         (if .__typename == "StatusContext" then
            (if .state == "SUCCESS" then "SUCCESS"
             elif (.state == "FAILURE" or .state == "ERROR") then "FAILURE"
             else "NEUTRAL" end)
          else (.conclusion // "UNKNOWN") end)
       ] | @tsv')

     # Sentinel: do NOT declare green until the CI aggregate gate has
     # reached a terminal state. Before /ok to test triggers the run,
     # the sentinel is absent; while CI is running, it's IN_PROGRESS.
     sentinel_line=$(printf '%s\n' "$rollup" | awk -F'\t' -v s="$SENTINEL" '$1 == s')
     sentinel_status=$(printf '%s\n' "$sentinel_line" | awk -F'\t' 'NR==1 {print $2}')
     if [ "$sentinel_status" != "COMPLETED" ]; then
       echo "=== $(date -u) waiting for $SENTINEL (status: ${sentinel_status:-absent}) ==="
       sleep 120
       continue
     fi

     # Classify non-exempt checks (exempt list applied to the NAME only).
     non_exempt=$(printf '%s\n' "$rollup" | awk -F'\t' -v p="^($EXEMPT)" 'tolower($1) !~ tolower(p)')
     failed=$(printf '%s\n' "$non_exempt" | awk -F'\t' '$2 == "COMPLETED" && $3 !~ /^(SUCCESS|SKIPPED|NEUTRAL)$/')
     pending=$(printf '%s\n' "$non_exempt" | awk -F'\t' '$2 != "COMPLETED"')

     if [ -n "$failed" ]; then
       echo "=== NON-EXEMPT FAILURES ==="
       printf '%s\n' "$failed"
       echo "RESULT=FAILURE"
       exit 0
     fi
     if [ -n "$pending" ]; then
       # Sentinel is COMPLETED but a non-exempt check is still pending —
       # rare but possible. Keep waiting; do NOT ship.
       echo "=== $(date -u) sentinel done but non-exempt checks still pending ==="
       printf '%s\n' "$pending"
       sleep 120
       continue
     fi

     echo "=== ALL NON-EXEMPT CHECKS COMPLETED GREEN ==="
     printf '%s\n' "$non_exempt"
     echo "RESULT=GREEN"
     exit 0
   done
   ```

   This Bash call blocks for as long as CI takes (minutes to hours). Do
   NOT split it into many short polls interleaved with other tool calls
   — that wastes `--max-turns` and creates windows where you could lose
   track of the loop state.

4. Read the tool output:
   - If `RESULT=FAILURE`: use `gh api repos/$REPO/actions/jobs/<JOB_ID>/logs`
     (or the equivalent for external contexts) to diagnose, fix the code,
     commit, push. Then start a NEW outer-loop iteration at step 1 with
     the new HEAD SHA.
   - If `RESULT=GREEN`: outer loop is done. Proceed to Phase 4.

**Why not wait-for-run-to-register first?** `gh pr comment` with
`/ok to test <sha>` is handled by `copy-pr-bot`, which takes a few
seconds to trigger the CI run. The `statusCheckRollup` poll in step 3
will initially show checks in `PENDING` / `QUEUED`; that's fine — the
inner loop treats those as "keep waiting" and will see them advance as
CI progresses. No separate registration poll needed.

### Anti-Patterns (what went wrong on run 24800621116)

- **Do NOT classify a queued/in-progress job as "infrastructure-
  blocked" and ship.** A stuck queue drains eventually — wait. If the
  job eventually passes, great; if it fails, go fix it.
- **Do NOT mark ready while any required check is `PENDING` /
  `QUEUED` / `IN_PROGRESS` on the HEAD SHA.** A push is not a pass;
  only a `COMPLETED` + green status is.
- **Do NOT declare an untested job "pre-existing."** Pre-existing
  means the test ran to completion and failed the same way on recent
  dev CI. A job that never ran on your PR cannot be pre-existing.
- **Do NOT use `gh api .../actions/runs/.../jobs` alone** as the gate
  signal. External status contexts (GitLab CI pipelines, copy-pr-bot
  status, etc.) do NOT appear there. Use `statusCheckRollup`.
- **Do NOT start any background process.** No `&`, no `nohup`, no
  `run_in_background: true`, no `ScheduleWakeup`. The GitHub Actions
  step owns your shell; when the step ends, every background process
  is killed and cannot resume.

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

---

## Phase 4: Mark PR Ready — Strict Gate

Run `gh pr ready` ONLY when every non-exempt required check on the latest
CI run (against the current HEAD SHA) satisfies BOTH:

1. `status == "completed"` — NOT `queued`, `in_progress`, `pending`,
   `waiting`, or `requested`.
2. `conclusion ∈ {"success", "skipped", "neutral"}`.

If a non-exempt check is pending/queued/in-progress: keep polling; do not
run `gh pr ready`. If it fails: go back to Phase 3's loop.

The exempt list (approval/coverage/docs) is defined in Phase 3; only those
checks may be ignored.

A pre-existing failure (same test failing identically on recent dev CI)
may be accepted, but ONLY after it has fully run, been empirically
verified against dev, and documented in the PR body with evidence (dev PR
number + CI run URL).

```
gh pr ready <PR_NUMBER> --repo $REPO
```

Then comment on the PR confirming which checks passed, listing any
documented pre-existing failures with evidence, and stating it is ready
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
