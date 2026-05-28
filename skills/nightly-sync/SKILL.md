---
name: nightly-sync
description: Domain knowledge for the nightly main-to-dev sync workflow. Covers merge strategy, CI architecture, failure investigation, and known issues.
license: Apache-2.0
when_to_use: Working on the nightly sync PR; investigating a nightly sync failure; resolving merge conflicts between main and dev; 'nightly sync failed', 'main-to-dev merge', 'sync bot'.
---

# Nightly Sync: Main to Dev

This skill is read by the automated sync bot during the nightly-sync-main-to-dev
workflow. It contains all domain knowledge for merging main into dev, resolving
conflicts, iterating on CI, and shipping the PR.

Detailed shell templates and historical regression examples live in
[detailed procedures](references/detailed-procedures.md). Read that reference
when running pre-push invariant checks or the Phase 3 CI polling loop.

---

## Phase 1: Create the Sync Branch and Merge

### Branch Setup

1. Create branch `$BRANCH` from `origin/dev`
2. Merge: `git merge origin/main --no-edit`
3. Resolve conflicts surgically. Do NOT use global `-X theirs`, and do NOT
   blanket-checkout main's version of a shared file. Main's version may be
   taken wholesale only for files in "Files to Override from Main" below, or
   when you have identified a specific main commit that intentionally removes
   the dev-only code. For all other conflicts, combine both sides so recent
   dev-only additions remain present.

### Preserving Dev-Only Additions

Do NOT blanket-override all shared files with main's version. Dev has features
not yet in main (new classes, new modules, new tests). The merge preserves both
sides' non-conflicting additions — only intervene where there is an actual
conflict.

### Squash-Merge Chain Detection

Dev often develops features as a chain of PRs (PR1 → PR2 → PR3) where each
builds on the last. When PR1 is squash-merged to main, git sees main's squashed
version and dev's original commits as unrelated changes. A conflict resolution
that blindly picks main can silently discard PR2/PR3's improvements on dev.

After the merge, check for this pattern:

1. For each conflicted file, run `git log --oneline origin/dev -- <file>` to
   see if dev has commits that came AFTER the code main is bringing in.
2. If dev has follow-up commits (bug fixes, refactors, extensions), **favor
   dev's version** for those sections.
3. If the conflict is just main bringing in a clean copy of what dev already
   has (no follow-ups), main's version is fine.

Practical check: run `git diff origin/dev -- <file>` on conflicted files. If
dev's code was removed or reverted, investigate whether dev's version is the
more evolved one.

Real examples from PR #4291:
- `emerging_optimizers.py`: Main's version was MORE complete — it squash-merged
  dev's PRs plus added more. Taking main for that section was correct.
- `distrib_optimizer.py`: Main overwrote dev's `GroupedQuantizedTensor` support.
  Had to restore `_is_distopt_quantized_param` and the expanded
  `_expand_quantized_param_shard_for_cast` loop while keeping main's NVFP4
  additions. This required a surgical merge combining sections from both.

Key insight: squash-merge chains can go in EITHER direction. Sometimes main
is ahead (it squash-merged dev's work + more), sometimes dev is ahead (it has
follow-up PRs). Always diff both ways before deciding which version to favor.

Real example from PR #4882 / PR #4318:
- `transformer_engine.py`: main had unrelated `TEFusedMLP` refactors, while dev
  had the new `TEFusedDenseMLP` class. The sync kept the config flag and test
  but dropped the class and `gpt_layer_specs.py` selection. That is a merge
  accident: restore the dev class and selection while preserving main's
  `TEFusedMLP.as_mlp_submodule` refactor.

### Files to Override from Main

These files have known semantic conflicts where dev's versions reference args
or APIs that main removed or renamed. Take main's version with
`git checkout origin/main -- <file>`:

- `megatron/training/training.py` — references dev-only args
- `megatron/training/initialize.py` — references dev-only args
- `megatron/training/utils.py` — references dev-only args
- `megatron/training/datasets/data_samplers.py` — references dev-only args
- `megatron/core/optimizer/layer_wise_optimizer.py` — constructor signature

**Caveat for ALL overrides:** After taking main's version of any file, you
MUST run the API Mismatch Detection procedure (see below) on that file.
Taking main's caller code while keeping dev's callee implementations is the
#1 source of sync bugs.

**IMPORTANT: Do NOT take main's `pyproject.toml`, `uv.lock`, or
`docker/Dockerfile.ci.dev`.** These three files are a tightly coupled
triple — the Dockerfile's `uv sync` command must match the dependency
groups in `pyproject.toml`, and `uv.lock` must be consistent with both.
Main's versions are missing dev-only dependencies (e.g.
`fast-hadamard-transform`, correct TransformerEngine revision) and the
`--group no_pypi_wheels` flag needed to install them. Keep dev's versions
of all three files.

**IMPORTANT: `.github/CODEOWNERS` must NEVER be modified by the sync
bot under any circumstances.** Dev's CODEOWNERS is intentionally
different from main's — do not take main's version, do not merge them,
do not touch the file. If the merge produces a conflict or a non-zero
diff against `origin/dev` on this path, restore dev's version verbatim:

```
git checkout origin/dev -- .github/CODEOWNERS
```

Then verify with `git diff origin/dev -- .github/CODEOWNERS` — output
must be empty. Modifying CODEOWNERS triggers spurious reviewer
requests and conflicts with the dev team's governance; rolling back a
CODEOWNERS change after the PR lands is painful.

**NEVER manually edit `uv.lock`.** It is a machine-generated lockfile. If
it needs to change, it must be regenerated with `uv lock` inside a CUDA
container (see `.claude/skills/build-and-test/SKILL.md`).

### Git Source Reconciliation (pyproject.toml)

After keeping dev's `pyproject.toml`, check whether main has added NEW git
sources to `[tool.uv.sources]` that don't exist in dev's version. Main's
merged code may import from packages only available at specific git revisions.

1. Diff the `[tool.uv.sources]` sections:
   `git show origin/main:pyproject.toml` vs `git show origin/dev:pyproject.toml`
2. For each git source in main but not dev, add it to dev's `pyproject.toml`
3. For sources in both but at different revisions, check whether dev's revision
   works. If dev's revision is broken (TOML parse errors, missing classes main's
   code imports), take main's revision instead.

Real examples from PR #4291:
- `nvidia-resiliency-ext`: Main's `torch.py` imports `get_write_results_queue`
  which only existed in main's pinned git revision, not on PyPI. Had to add
  main's git source to dev's pyproject.toml.
- `nemo-run`: Dev's pinned revision had a TOML parse error with uv 0.7.2.
  Had to swap to main's revision.

After any changes to `pyproject.toml`, regenerate `uv.lock` inside a CUDA
container:
```bash
docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c "pip install uv==0.7.2 && cd /workspace && \
  uv venv .venv --system-site-packages && uv sync --only-group build && uv lock"
# Clean up root-owned .venv:
docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c "rm -rf /workspace/.venv"
```

### API Mismatch Detection (Post-Merge Audit)

The merge can create "Frankenstein" code where main's callers use dev's
implementations (or vice versa) with different method signatures. This
compiles fine but fails at runtime.

After the merge, audit cross-boundary call sites:

1. Identify files where main's version was taken (`-X theirs` or explicit
   `git checkout origin/main`)
2. For each, find all external call sites: classes it instantiates, methods
   it calls on imported objects, functions from other modules it invokes
3. Verify method names, parameter counts, and signatures match between the
   caller and the implementation in the merged tree
4. Pay special attention to "interface" modules (files defining base classes)
   — if main and dev evolved the interface differently, every caller and
   implementer must agree

Real examples from PR #4291:
- `multi_latent_attention.py` (main) called `off_interface.group_commit()`
  but dev's interface only had `group_offload()` — method renamed
- `mamba_model.py` (main) called `init_chunk_handler(3 params)` but dev's
  interface required 6 params — signature expanded on dev
- `mamba_model.py` called `mark_not_offloadable()` but dev had
  `mark_not_offload()` — method renamed
- `bulk_offload()` did `.remove()` after `bulk_offload_group()` already
  `.pop()`d the same item — double-removal from a list

Practical detection:
```bash
# For each file taken from main, find what it imports and calls
grep -rn "from <module> import\|<module>\." megatron/
# Cross-reference with the actual implementations in the merged tree
```

### File-Specific Merge Lessons

These lessons were learned from PR #4291. They may recur if the same files
continue to diverge:

- `gated_delta_net.py`: If the merge creates code calling non-existent helper
  methods (e.g. `_resolve_cu_seqlens`), take dev's version wholesale.
- `model_chunk_schedule_plan.py`: Watch for missing imports (e.g.
  `CudaGraphScope`) silently dropped during conflict resolution.
- `fine_grained_activation_offload.py`: Critical interface file used by many
  callers. If main and dev have divergent method names/signatures, prefer
  dev's implementation and patch main-originated callers to match.
- `distrib_optimizer.py`: Dev may have broader type abstractions (e.g.
  `_is_distopt_quantized_param` covering both FP8 and GroupedQuantizedTensor).
  Main may simplify to explicit type checks. Restore dev's abstractions.

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

### Pre-push invariant checks

Before every `git push` in this workflow, run the invariant script in
[detailed procedures](references/detailed-procedures.md#pre-push-invariant-checks).
The CODEOWNERS check and dev-feature preservation audit are hard aborts; never
push if either fails. The dependency-triple check is a warning because git-source
reconciliation can produce legitimate diffs.

### Commit and Push

Phase 1 produces a single commit on the sync branch. The merge itself
creates the merge commit; fold any post-merge work (formatting,
conflict surgery, restored files, regenerated `uv.lock`) into it
rather than stacking a second commit:

```bash
git add -A
git commit --amend --no-edit  # rewrites the merge commit's tree;
                              # parents are preserved.
git push -u origin "$BRANCH"  # only non-force push of the run.
```

Once pushed, this commit is immutable for the rest of the run.
Phase 3 fixes go into a separate rolling fix commit on top (see
Phase 3 step 4 and the two-commit policy in Rules).

---

## Phase 2: Create the Draft PR

- Title: `chore: nightly sync main into dev ($DATE)`
- Create as **draft**: `gh pr create --draft`
- Body should include:
  1. Summary of what was synced (number of commits from main)
  2. **Python-only line-change stats**, so reviewers can gauge the real
     code surface (excluding golden-value JSON, uv.lock, etc.). Compute
     with:

     ```bash
     git diff --numstat origin/dev...HEAD -- '*.py' \
       | awk 'BEGIN{a=0;d=0} {a+=$1; d+=$2} END{
           printf "Python lines: +%d / -%d across %d files\n", a, d, NR
         }'
     ```

     Include the exact line (e.g. `Python lines: +1234 / -567 across 42 files`)
     in the PR body so reviewers see it at a glance.
  3. List of files where main's version was taken over the merge
  4. List of files that were deleted in dev but restored (and why)
  5. The remerge-diff output (`git show --remerge-diff HEAD` on the merge
     commit) so reviewers can inspect ONLY the conflict resolutions. If the
     output is very long, summarize conflicts by file and put the full diff
     in a collapsed `<details>` block. If git is too old for `--remerge-diff`,
     note the git version and describe the merge strategy used instead.
- Save the PR number for later phases
- **Add the `Run functional tests` and `Run MBridge tests` labels** to the
  PR immediately after creation. The `Run functional tests` label ensures
  `/ok to test` triggers the full CI suite (unit tests + functional/
  integration tests with 100-step training and golden value comparison).
  The `Run MBridge tests` label triggers the MBridge test suite. Without
  these labels, only a lightweight subset runs.
  ```bash
  gh pr edit <PR_NUMBER> --repo $REPO \
    --add-label "Run functional tests" \
    --add-label "Run MBridge tests"
  ```

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

Use two nested loops: an outer sequence of tool calls and one inner blocking
Bash poll per CI iteration. The source of truth is
`gh pr view <PR_NUMBER> --repo $REPO --json statusCheckRollup`, because it
includes external contexts such as GitLab CI and `copy-pr-bot`.

For each iteration, get the current SHA, post `/ok to test <sha>`, then run the
blocking polling template in
[detailed procedures](references/detailed-procedures.md#ci-polling-template).
If it reports `RESULT=FAILURE`, fix the code and update the single rolling fix
commit. If it reports `RESULT=GREEN`, proceed to Phase 4.

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
- **Do NOT push directly to `pull-request/<PR_NUMBER>` branches.**
  The community bot manages those branches when it processes
  `/ok to test`. Pushing to them directly breaks the CI trigger
  mechanism. Always push to your own sync branch (e.g.
  `main2dev/<DATE>`) instead.
- **Do NOT forget the `Run functional tests` and `Run MBridge tests`
  labels.** Without `Run functional tests`, the internal GitLab
  functional tests do not run; without `Run MBridge tests`, the
  MBridge test suite does not run.

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
- **API mismatch (AttributeError / TypeError at runtime):** Main's callers
  reference methods that don't exist (or have different signatures) in dev's
  implementations. See "API Mismatch Detection" in Phase 1. Fix by adding
  shims, renaming methods, or adjusting call signatures.
- **Infrastructure / network failures (apt-get, pip download):** Errors like
  `archive.ubuntu.com unreachable` or `Connection timed out` during package
  installation are transient CI infrastructure issues, not code problems.
  Retry CI with the same SHA. Do not investigate as code failures.

### Pre-Existing Failure Verification

**You MUST empirically verify before classifying any failure as pre-existing.**

1. `gh pr list --repo $REPO --base dev --state merged --limit 3`
2. `gh pr checks <PR_NUMBER> --repo $REPO` on a recently merged dev PR
3. If the same test bucket **passes on recent dev CI** → the failure is
   sync-caused. You must fix it.
4. Only if the test **also fails on recent dev CI** can you classify it as
   pre-existing. Document with the dev PR number and CI run as evidence.

### Internal GitLab Functional Tests

GitHub CI covers unit tests and some integration tests. Internal GitLab
(`gitlab-master.nvidia.com`) runs additional functional tests on
H100/GB200 hardware that may reveal issues GitHub CI does not catch.
These surface in `statusCheckRollup` as external status contexts (the
bash template already handles them via the `__typename == "StatusContext"`
branch).

- Fine-grained activation offloading failures, for example, only showed
  up in GitLab functional tests during PR #4291
- If GitHub CI passes but a reviewer reports GitLab failures,
  investigate with the same rigor as GitHub CI failures
- The sync PR should ideally pass both GitHub and GitLab CI before
  merge, but GitHub CI passing (i.e. the Phase 4 gate above) is the
  minimum before `gh pr ready`

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

Then comment on the PR confirming it is ready for human review. The
comment should include:
- Which non-exempt checks passed (summary from the bash template's
  final `ALL NON-EXEMPT CHECKS COMPLETED GREEN` output)
- Any documented pre-existing failures with evidence (dev PR number +
  CI run URL showing the same failure on recent dev CI)
- Which files were taken from main vs. merged manually
- Any API mismatches detected and fixed
- Any `pyproject.toml` git source reconciliation performed
- Links to the CI runs that validated the fixes

---

## Rules

- Do not globally prioritize main over dev. Preserve recent dev-only additions
  unless a specific main commit intentionally removed them. Main may win
  wholesale only for the explicit override list above; otherwise resolve
  conflicts by combining main's incoming changes with dev's still-unmerged
  features.
- **Two-commit policy:** the PR contains at most two bot-authored
  commits — the Phase 1 merge commit (immutable once pushed) and a
  single rolling fix commit on top. The fix commit is created on
  the first Phase 3 failure (normal push) and amended on every
  subsequent failure (`git commit --amend --no-edit` +
  `git push --force-with-lease`). Never modify the Phase 1 commit
  after pushing it; never let the fix-commit count exceed one.
- CI triggers via comment: `/ok to test <sha>`
- CI runs appear on branch `pull-request/<PR_NUMBER>`
- Git committer identity: `svcnvidia-nemo-ci`
- After editing imports, run `isort` on those files
- **Push directly to NVIDIA/Megatron-LM** (not a fork). The bot uses a PAT
  with write access. CLAUDE.md says "never push directly" but that rule is
  for human contributors — the sync bot is an exception.
