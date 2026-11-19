---
name: mcore-cicd
description: CI/CD reference for Megatron-LM. Covers CI pipeline structure, PR scope labels, triggering internal GitLab CI (which force-pushes the current branch to a pull-request/BRANCH ref — always dry-run and verify the destination first; never run against shared or protected branches), and CI failure investigation. Use when investigating a CI failure; understanding the pipeline structure; which CI label to attach; triggering internal GitLab CI; 'CI is red', 'how do I trigger CI', 'PR labels', 'where are the logs', 'pull-request branch'.
license: Apache-2.0
metadata:
  author: Oliver Koenig <okoenig@nvidia.com>
---

# CI/CD Guide

---

## Answer-First CI Facts

For PR-label or trigger questions, lead with the exact values:

- `Run Merge-Queue jobs`: selects the actual merge-group configuration:
  `scope=L1`, `n_repeat=1`, `lightweight=false`, `cadence=mergegroup`, no cadence
  bypass, full unit buckets, dev images, AWS H100 routing, MBridge L1, no NeMo RL.
  Normal training steps and golden-value comparisons remain enabled. PR approval
  controls and platform/downstream eligibility still apply.
- `Run functional tests`: `scope=L1`, `n_repeat=5`, `lightweight=false`.
- `Run tests`: `scope=L1`, `n_repeat=1`, `lightweight=true` (four steps, no golden comparison).
- No scope label: `scope=L0`, `n_repeat=2`, `lightweight=false`.
- `container::lts` switches the container image path to LTS outside merge-queue
  mode. **Opt-in only — attach it solely when the user
  explicitly asks for LTS validation; never add it on your own initiative, even
  for a container or dependency change.**
- `Run MBridge tests` additionally triggers the MBridge L1 suite.
- `Run NeMoRL tests` additionally triggers NeMo RL's Megatron functional test suite
  outside merge-queue mode.
- ⚠️ **WARNING — destructive remote write.** `tools/trigger_internal_ci.py`
  **force-pushes the current branch** to the internal GitLab remote as
  `pull-request/<branch>`. Always run with `--dry-run` first and confirm the
  destination ref before invoking it without the flag. Never run against a
  shared or protected branch — only target your own pull-request branch.
  Safe preflight: `python tools/trigger_internal_ci.py --gitlab-origin gitlab --dry-run`.
  Add the optional `--functional-test-*` flags only after the dry-run output
  matches the intended destination.

---

## CI Pipeline Structure

The main workflow is `.github/workflows/cicd-main.yml`. It triggers on pushes
to branches matching `pull-request/[0-9]+` and `deploy-release/*`, on merge
groups, and on manual dispatch.

```text
is-not-external-contributor
  └─ pre-flight
       └─ configure          # determines scope, container tag, n_repeat
            ├─ linting
            ├─ cicd-container-build
            │    ├─ cicd-parse-unit-tests → cicd-unit-tests-latest
            │    ├─ cicd-parse-integration-tests-h100 → cicd-integration-tests-latest-h100
            │    └─ cicd-parse-integration-tests-gb200 → cicd-integration-tests-latest-gb200 (maintainers only)
            └─ Nemo_CICD_Test  # final pass/fail gate
```

Images are pushed to:

- AWS ECR: `766267172432.dkr.ecr.us-east-1.amazonaws.com/…`
- GCP Artifact Registry: `us-east4-docker.pkg.dev/nv-projdgxchipp-20260113193621/megatron-lm/…`

---

## CI Test Scope Labels

The CI pipeline reads PR labels to decide test scope, n_repeat, and container image.

**Decision tree (first match wins):**

| Condition | `scope` | `n_repeat` | `lightweight` | Notes |
|-----------|---------|-----------|---------------|-------|
| Merge group or `Run Merge-Queue jobs` | `L1` | 1 | false | Shared configuration; wins over other scope/image/downstream labels |
| `Run tests` | `L1` | 1 | true | Lightweight run without golden comparisons |
| `Run functional tests` | `L1` | 5 | false | Five full repetitions |
| Schedule / manual dispatch | `L1` | 5 | false | CI workload default |
| _(no scope label)_ | `L0` | 2 | false | PR default |

`Run Merge-Queue jobs` and actual merge groups share `MERGE_QUEUE_JOBS` in the
configure step. They use `cadence=mergegroup`, retain cadence filtering, disable
selective unit tests, select dev images and AWS H100 routing, request MBridge L1,
and skip NeMo RL. Non-maintainer or non-member PRs retain ephemeral runner routing.
Ordinary `Run tests` and `Run functional tests` labels bypass cadence filtering.
The merge-queue and functional-test labels share the approval queue's global
functional-test slot and are held when `ALLOW_PR_RUN_ALL_TESTS=false`.

**Orthogonal image label:**

| Label | Effect |
|-------|--------|
| **`container::lts`** | Build on the older long-term-support NGC PyTorch base instead of `dev`'s latest; ignored in merge-queue mode |
| **`Run MBridge tests`** | Also triggers the MBridge L1 test suite |
| **`Run NeMoRL tests`** | Also triggers NeMo RL's Megatron functional test suite outside merge-queue mode |

Adding a label does not itself start `cicd-main.yml`; apply it before the next
synthetic PR push or rerun the workflow after applying it.

For combining queued changes into one validation PR, use
[batch-and-test-mergequeue-prs](../batch-and-test-mergequeue-prs/SKILL.md).

### Which label to attach when opening a PR

| Changed paths / nature of change | Label to attach |
|----------------------------------|-----------------|
| Docs only (`docs/`, `*.md`, docstrings) | _(none)_ |
| CI/tooling only (`.github/`, `tools/`, `Makefile`) | _(none)_ |
| Touches MBridge integration | add `Run MBridge tests` |
| Could affect NeMo RL's Megatron integration | add `Run NeMoRL tests` |

---

## Triggering Internal CI

Use `tools/trigger_internal_ci.py` after the internal GitLab remote and
`GITLAB_TOKEN` are configured; see @tools/trigger_internal_ci.md for setup
details. First run a dry run and verify the destination ref:

```bash
python tools/trigger_internal_ci.py --gitlab-origin gitlab --dry-run
```

The script force-pushes the current branch to `pull-request/<branch>` before
triggering the pipeline. Only target your own pull-request branch, never a shared
or protected branch. Add optional `--functional-test-*` flags only after the
dry-run output matches the intended destination.

---

## CI Failure Investigation

CI branches always follow the pattern `pull-request/<number>`.

### Locating the PR from a CI Branch

```bash
# Extract PR number from the current branch
PR_NUMBER=$(git rev-parse --abbrev-ref HEAD | grep -oP '(?<=pull-request/)\d+')

# Fetch the PR metadata (title, labels, author, base branch)
gh pr view "$PR_NUMBER" --repo NVIDIA/Megatron-LM

# Show the changeset for that PR
gh pr diff "$PR_NUMBER" --repo NVIDIA/Megatron-LM
```

### Reading CI Job Logs

```bash
# List recent workflow runs for the PR
gh run list --repo NVIDIA/Megatron-LM --branch "pull-request/$PR_NUMBER"

# Stream failing job output
gh run view <run-id> --repo NVIDIA/Megatron-LM --log-failed
```

Full per-rank logs are **not** in the runner stdout. They are uploaded as
GitHub artifacts named `logs-<test_case>-<run_id>-<uuid>`.

```bash
# 1. Find artifact name
gh run view <run-id> --repo NVIDIA/Megatron-LM --json artifacts \
  --jq '.artifacts[].name'

# 2. Download the artifact zip
gh run download <run-id> --repo NVIDIA/Megatron-LM \
  --name "logs-<artifact-name>" -D ./ci-logs

# 3. Locate which rank logs contain errors
grep -r -l "ERROR\|Traceback\|FAILED\|fatal" ./ci-logs/

# 4. Log files can exceed 10 000 lines — never read a full log at once.
wc -l ./ci-logs/<test>/<attempt>/attempt_0/<rank>/stderr.log
sed -n '1,200p' ./ci-logs/.../stderr.log   # read in chunks
```

### Identifying Failure Root Cause

1. **Linting failure** — re-run `tools/autoformat.sh` locally; the diff shows exactly what needs to change.
2. **Container build failure** — inspect the `cicd-container-build` job log.
3. **Unit test failure** — the failing bucket is in the `cicd-unit-tests-latest` job matrix.
4. **Functional test failure** — look at the `cicd-integration-tests-*` job. Start with `stdout.log` for rank 0.
5. **Flaky test** — the runner retries automatically up to 3 times. If all retries exhausted and the pattern matches a known transient (NCCL, ECC, segfault), it is infrastructure noise.

### Correlating a Failure with the PR Changeset

```bash
# Find unit tests that cover a changed source file
grep -r "from megatron.core.transformer.attention" tests/unit_tests/ -l

# Check CODEOWNERS for reviewer assignment
cat .github/CODEOWNERS | grep "<changed-path>"
```
