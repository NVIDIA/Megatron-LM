---
name: bump-base-image
description: Bump the NVIDIA PyTorch base image (`nvcr.io/nvidia/pytorch:<YY.MM>-py3`) used by Megatron-LM CI. Covers the two pin sites (GitHub CI in `docker/.ngc_version.dev` and GitLab CI in `.gitlab/stages/01.build.yml`), the post-bump CI loop (re-run functional tests, refresh golden values, mark broken tests), and the gotchas that bit PRs #4611 and #4688.
when_to_use: User wants to upgrade the PyTorch container (e.g. "bump base image to 26.04"); CI is failing after a previous bump because the GitLab pin was missed; functional tests are failing with `lm loss` / `num-zeros` / `iteration-time` drift right after a container bump; a functional test hangs, times out, or OOMs after a bump; the user mentions `.ngc_version.dev`, `nvcr.io/nvidia/pytorch`, "container base image", or "Update Docker image version".
---

# Bump the PyTorch base image

End-to-end workflow for moving Megatron-LM's CI to a newer `nvcr.io/nvidia/pytorch:<YY.MM>-py3` container. The most common failure mode is forgetting that **GitHub CI and GitLab CI have separate pins** — a bump that only touches the former lands green, then breaks GitLab CI on `main` and forces an immediate follow-up PR. Always update both in the same PR.

## Inputs to gather from the user

1. **Target tag**, e.g. `26.04-py3`. NVIDIA NGC PyTorch containers are released as `nvcr.io/nvidia/pytorch:YY.MM-py3`.
2. **Scope** — usually `dev` only. The `lts` pin (`docker/.ngc_version.lts`, plus the `IMAGE_TYPE: lts` rows in GitLab) is bumped on a different cadence; only touch it if the user explicitly asks.
3. **Workflow run ID** (optional but typical) — after the first CI run, the user will provide a GitHub Actions run ID for golden-value refresh.

## Workflow

```
- [ ] Step 1: Update the GitHub CI pin (docker/.ngc_version.dev)
- [ ] Step 2: Update the GitLab CI pin (.gitlab/stages/01.build.yml)
- [ ] Step 3: Open the PR with the `Run functional tests` label
- [ ] Step 4: Re-run failing tests via `/ok to test <commit-sha>`
- [ ] Step 5: For golden-value drift → refresh with the `update-golden-values` skill
- [ ] Step 6: For hangs / real regressions → mark tests `mr-broken` and file tracking issues
- [ ] Step 7: Verify both pins are in sync before merging
```

### Step 1 — GitHub CI pin

`docker/.ngc_version.dev` is a single-line file consumed by `docker/Dockerfile.ci.dev` (via `FROM_IMAGE_NAME=$(cat docker/.ngc_version.dev)`). Overwrite it:

```bash
echo 'nvcr.io/nvidia/pytorch:<YY.MM>-py3' > docker/.ngc_version.dev
```

The file has no trailing newline historically; preserving or adding one is fine — the build args treat the value as `$(cat ...)`. Do **not** touch `docker/.ngc_version.lts` unless bumping LTS too.

### Step 2 — GitLab CI pin

GitLab CI does **not** read `docker/.ngc_version.dev`. It hardcodes `BASE_IMAGE` in a `parallel: matrix:` block. Update the two `IMAGE_TYPE: dev` rows (one per platform):

```yaml
# .gitlab/stages/01.build.yml — under test:pre_build_image -> parallel.matrix
- IMAGE: CI_MCORE_DEV_IMAGE
  FILE: Dockerfile.ci.dev
  IMAGE_TYPE: dev
  BASE_IMAGE: nvcr.io/nvidia/pytorch:<YY.MM>-py3   # amd64 row
  PLATFORM: amd64
- IMAGE: CI_MCORE_DEV_IMAGE
  FILE: Dockerfile.ci.dev
  IMAGE_TYPE: dev
  BASE_IMAGE: nvcr.io/nvidia/pytorch:<YY.MM>-py3   # arm64 row
  PLATFORM: arm64
```

Leave the `IMAGE_TYPE: lts` rows alone. Quick sanity check before commit:

```bash
rg -n '^\s*BASE_IMAGE: nvcr\.io/nvidia/pytorch:' .gitlab/stages/01.build.yml
# expect:  lts pin × 2 unchanged, dev pin × 2 == new tag
```

### Step 3 — Open the PR

- Title convention: `chore: Update Docker image version to <YY.MM>-py3` (see #4611).
- **Apply the `Run functional tests` label** before the first push. This unlocks the full functional matrix on the PR; without it the bump only runs the standard GH PR checks and you'll miss the drift.
- Push as draft first if you're still iterating; the bot will auto-draft otherwise.

### Step 4 — Re-running CI on a new commit

For PRs from forks (the typical contributor case), each new commit needs an explicit `/ok to test <commit-sha>` PR comment to authorize NVIDIA runners (see the `copy-pr-bot` flow in #4611). One comment per commit. If `copy-pr-bot` reports "had a problem deploying to test", just push another commit (or re-issue the comment after the next push); the deploy is per-commit, not per-comment.

### Step 5 — Golden-value drift

Container bumps shift CUDA / cuBLAS / cuDNN / kernel autotuning, which moves `lm loss`, `num-zeros`, `iteration-time`, and `mem-*` metrics on a large fraction of functional tests. This is **expected** and is not a correctness regression — refresh the golden values rather than chasing each test.

Hand off to the `update-golden-values` skill with:

- `--source github`
- `--pipeline-id <WORKFLOW_RUN_ID>` from the failing CI run
- `--only-failing` (refresh just the trajectories that drifted)

PR #4611 refreshed **78 golden-value files** across `dev_dgx_h100` and `dev_dgx_gb200` for GPT / MoE / MIMO / hybrid suites in a single pass via this exact flow. The per-metric relative-difference summary the skill produces is the recommended PR description blurb — reviewers expect to see it.

### Step 6 — Real regressions: mark broken, don't block the bump

A small number of tests will genuinely break (hangs, OOM, real numerical regressions). Don't gate the base-image bump on fixing them — that conflates two changes. Instead:

1. **File a GitHub issue** describing the failure mode and linking the failing CI run.
2. **Flip the test's scope to the `-broken` variant** in the recipe YAML under `tests/test_utils/recipes/<arch>/`, with an inline comment that references the issue. Pattern:

   ```yaml
   - test_case: [hybrid_dynamic_inference_tp1_ep8_nanov3_chunked_prefill]
     products:
       - environment: [dev]
         # Broken: hangs on repeat iter 3, exceeds 1h job limit — see issue #<N>.
         scope: [mr-broken, mr-github-broken]      # was: [mr, mr-github]
         platforms: [dgx_h100]
   ```

   Scope mapping (replace, don't append):

   | Before        | After                |
   | ------------- | -------------------- |
   | `mr`          | `mr-broken`          |
   | `mr-github`   | `mr-github-broken`   |
   | `nightly`     | `nightly-broken`     |

   The recipe still runs in the `-broken` scope, but failures stop blocking PR merges.

### Step 7 — Sync check before merging

The single biggest failure mode of this workflow is shipping #4611 without #4688. Before you ask for the merge, confirm both pins resolve to the same tag:

```bash
echo -n "ngc_version.dev: " && cat docker/.ngc_version.dev
echo
echo "gitlab dev rows:"
rg -n '^\s*BASE_IMAGE: nvcr\.io/nvidia/pytorch:' .gitlab/stages/01.build.yml \
  | rg -B1 'IMAGE_TYPE: dev' \
  | rg 'BASE_IMAGE'
```

All three lines should show `nvcr.io/nvidia/pytorch:<YY.MM>-py3`. If they don't, fix it before merge — otherwise GitLab CI keeps building on the old container and the next person hits the same trap.

## File-touch cheat sheet

| Path                                                                                          | Edit                                                                                   |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `docker/.ngc_version.dev`                                                                     | Overwrite with new `nvcr.io/nvidia/pytorch:<YY.MM>-py3`                                |
| `.gitlab/stages/01.build.yml`                                                                 | Update both `IMAGE_TYPE: dev` `BASE_IMAGE:` rows (amd64 + arm64)                       |
| `tests/functional_tests/test_cases/**/golden_values_dev_dgx_{h100,gb200}.json`                | Refresh via the `update-golden-values` skill                                           |
| `tests/test_utils/recipes/<arch>/<suite>.yaml`                                                | Flip drifting / hanging cases to `mr-broken` / `mr-github-broken` with an issue link    |
| `docker/.ngc_version.lts`, `.gitlab/stages/01.build.yml` `IMAGE_TYPE: lts` rows               | **Skip unless explicitly bumping LTS.** LTS has its own release cadence.                |

## Gotchas

- **GitHub vs GitLab pins are independent.** `docker/.ngc_version.dev` only drives GitHub CI's local container build via `Dockerfile.ci.dev`. GitLab CI has its own hardcoded `BASE_IMAGE:` matrix in `.gitlab/stages/01.build.yml`. PR #4688 existed solely because #4611 forgot the second one — don't repeat this.
- **Don't bump LTS along with dev.** The `IMAGE_TYPE: lts` rows and `docker/.ngc_version.lts` are stability-pinned for the `container::lts` label path. Bump them in a dedicated PR with its own LTS validation.
- **Don't fix golden-value drift by hand.** Use `tests/test_utils/python_scripts/download_golden_values.py` via the `update-golden-values` skill. Hand-editing the JSONs invites diff noise and relative-difference regressions on subsequent bumps.
- **`mr-broken` is a real scope, not a comment marker.** It keeps the recipe wired into the matrix (so it stays discoverable and runnable on demand) without gating merges. Don't delete the test case from the recipe.
- **`/ok to test` is per-commit.** A new force-push or fixup commit needs a fresh `/ok to test <sha>` comment to re-trigger NVIDIA-runner CI on a fork PR.
- **Don't merge until the GitLab pin matches.** Use the Step 7 grep before requesting review.

## Related skills

- [update-golden-values](../update-golden-values/SKILL.md) — call this as soon as the first post-bump CI run finishes and you have a workflow run ID with failing golden checks. Produces the per-metric relative-difference summary you paste into the PR description.
- [build-and-dependency](../build-and-dependency/SKILL.md) — for verifying the new image builds locally before opening the PR (`docker build --target main --build-arg FROM_IMAGE_NAME=$(cat docker/.ngc_version.dev) ...`).
- [cicd](../cicd/SKILL.md) — for the PR scope-label semantics (`Run functional tests`, `complexity::*`) and the `copy-pr-bot` flow.
