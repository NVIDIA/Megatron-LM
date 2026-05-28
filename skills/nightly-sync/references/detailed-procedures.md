# Nightly Sync Detailed Procedures

Load this reference only when executing the nightly sync workflow. Keep the
main `SKILL.md` focused on workflow decisions; this file holds copyable shell
templates and historical regression examples.

## Pre-push Invariant Checks

Run before every `git push` in the nightly sync workflow, including the initial
Phase 1 push and every Phase 3 fix push. If CODEOWNERS or the dev-feature audit
fails, stop and fix before pushing. The dependency-management triple check is a
warning because git-source reconciliation can legitimately change
`pyproject.toml`.

```bash
MERGE_COMMIT=$(git rev-list --min-parents=2 --max-count=1 HEAD || true)
if [ -n "$MERGE_COMMIT" ]; then
  DEV_REF="${MERGE_COMMIT}^1"
  MAIN_REF="${MERGE_COMMIT}^2"
else
  DEV_REF="origin/dev"
  MAIN_REF="origin/main"
fi

# 1. CODEOWNERS must be identical to dev's.
if ! git diff --quiet "$DEV_REF" HEAD -- .github/CODEOWNERS; then
  echo "ABORT: .github/CODEOWNERS differs from dev. Restore with:"
  echo "  git checkout $DEV_REF -- .github/CODEOWNERS"
  exit 1
fi

# 2. Dependency-management triple must be identical to dev's.
for f in pyproject.toml uv.lock docker/Dockerfile.ci.dev; do
  if ! git diff --quiet "$DEV_REF" HEAD -- "$f"; then
    echo "WARNING: $f differs from dev"
  fi
done

# 3. Dev-feature preservation audit.
INTENTIONAL_OVERRIDE_REGEX='^(megatron/training/training\.py|megatron/training/initialize\.py|megatron/training/utils\.py|megatron/training/datasets/data_samplers\.py|megatron/core/optimizer/layer_wise_optimizer\.py)$'
SKIP_REGEX='^(pyproject\.toml|uv\.lock|docker/Dockerfile\.ci\.dev|\.github/CODEOWNERS)$'

VIOLATIONS=0
for f in $(git diff --name-only "$DEV_REF"..HEAD \
            -- '*.py' '*.md' '*.yaml' '*.yml' '*.toml' \
               '*.sh' '*.cpp' '*.cu' '*.h' \
            | sort -u); do
  [[ "$f" =~ $SKIP_REGEX ]] && continue
  [[ "$f" =~ $INTENTIONAL_OVERRIDE_REGEX ]] && continue
  git cat-file -e "HEAD:$f" 2>/dev/null || continue

  missing=$(comm -23 \
              <(git show "$DEV_REF:$f"  2>/dev/null | sort -u) \
              <(git show "$MAIN_REF:$f" 2>/dev/null | sort -u) \
            | comm -23 - <(git show "HEAD:$f" 2>/dev/null | sort -u) \
            | grep -E '[[:alnum:]_]' \
            || true)

  if [ -n "$missing" ]; then
    echo "=== $f ==="
    printf '%s\n' "$missing"
    VIOLATIONS=$((VIOLATIONS + $(printf '%s\n' "$missing" | grep -c .)))
  fi
done

if [ "$VIOLATIONS" -gt 0 ]; then
  echo "ABORT: $VIOLATIONS dev-only line(s) dropped by the merge. For each:"
  echo "  (a) MAIN INTENTIONALLY REMOVED -- find the specific commit in"
  echo "      'git log origin/main -- <file>' that removed it; document the"
  echo "      SHA in the PR body, then the drop is acceptable."
  echo "  (b) MERGE ACCIDENT -- main never explicitly touched that line."
  echo "      RESTORE the dev line (Edit/Write to put it back)."
  echo "Default to (b); only declare (a) with a specific main commit as evidence."
  exit 1
fi
```

Regressions this audit would have flagged:

- `transformer_layer.py` lost `_forward_mlp_router(input_ids=None)`.
- `token_dispatcher.py` lost the `num_sms_preprocessing_api=...` kwarg on the `_HybridEPManager` call.
- `moe_layer.py` lost `self._maybe_record_overload_factor(...)`.
- `gpt_dynamic_inference_with_coordinator.py` lost `from megatron.training.arguments import parse_and_validate_args`.
- `datasets/readme.md` lost the dev-only "Packing Scheduler" section.
- PR #4882 / PR #4318 dropped `TEFusedDenseMLP` and `gpt_layer_specs.py` selection while leaving the config flag and unit test.
- `data_samplers.py`, `utils.py`, and `training.py` kept `args.hybrid_context_parallel` instead of `args.dynamic_context_parallel` after taking main's version.

## CI Polling Template

Use this as the single blocking Bash call inside each Phase 3 outer-loop
iteration. It waits for the aggregate gate, classifies non-exempt checks, and
prints `RESULT=FAILURE` or `RESULT=GREEN`.

```bash
REPO='NVIDIA/Megatron-LM'
PR='<PR_NUMBER>'
# Names matched case-insensitively, anchored to the START of the name.
EXEMPT='copy-pr-bot|is-not-external-contributor|greptile|coderabbit|codeowners|.*review|.*approval|codecov|coverage|build-docs|doc-build|readthedocs|sphinx'
SENTINEL='Nemo_CICD_Test'

while true; do
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

  sentinel_line=$(printf '%s\n' "$rollup" | awk -F'\t' -v s="$SENTINEL" '$1 == s')
  sentinel_status=$(printf '%s\n' "$sentinel_line" | awk -F'\t' 'NR==1 {print $2}')
  if [ "$sentinel_status" != "COMPLETED" ]; then
    echo "=== $(date -u) waiting for $SENTINEL (status: ${sentinel_status:-absent}) ==="
    sleep 120
    continue
  fi

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

Do not split this into many short polls. `/ok to test <sha>` may take a few
seconds to register through `copy-pr-bot`; the polling template handles absent,
pending, queued, and in-progress checks.

## Fix Commit Update

When the polling template reports `RESULT=FAILURE`, diagnose logs and update the
single rolling fix commit on top of the immutable Phase 1 merge commit:

```bash
git add -A
if git rev-parse --verify HEAD^2 >/dev/null 2>&1; then
  git commit -m "fix: post-CI corrections"
  git push origin "$BRANCH"
else
  git commit --amend --no-edit
  git push --force-with-lease origin "$BRANCH"
fi
```

Use `--force-with-lease`, not `--force`; if a human pushed onto the branch,
fetch and decide how to proceed instead of clobbering their work.
