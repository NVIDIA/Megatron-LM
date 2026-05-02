---
name: cicd
description: CI/CD reference for Megatron-LM. Covers CI pipeline structure, PR scope labels, triggering internal GitLab CI, and CI failure investigation.
when_to_use: Investigating a CI failure; understanding the pipeline structure; which CI label to attach; triggering internal GitLab CI; 'CI is red', 'how do I trigger CI', 'PR labels', 'where are the logs', 'pull-request branch'.
---

# CI/CD Guide

---

## CI Pipeline Structure

The main workflow is `.github/workflows/cicd-main.yml`. It triggers on pushes
to branches matching `pull-request/[0-9]+` and `deploy-release/*`, on merge
groups, on a daily schedule, and on manual dispatch.

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
| Merge group | `mr-github` | 1 | false | Automatic, no label needed |
| Label: **`Run tests`** | `mr-github` | 1 | **true** | Trains 4 steps, no golden-value compare |
| Label: **`Run functional tests`** | `mr-github` | 5 | **false** | Trains 100 steps, golden-value compare |
| _(no label)_ | `mr-github-slim` | 5 | false | Slim subset only |

**Orthogonal image label:**

| Label | Effect |
|-------|--------|
| **`container::lts`** | Use the LTS base image instead of `dev` (combinable with any scope label) |
| **`Run MBridge tests`** | Also triggers the MBridge L1 test suite |

### Which label to attach when opening a PR

| Changed paths / nature of change | Label to attach |
|----------------------------------|-----------------|
| Docs only (`docs/`, `*.md`, docstrings) | _(none)_ |
| CI/tooling only (`.github/`, `tools/`, `Makefile`) | _(none)_ |
| Test files only (`tests/`) — existing tests, no new golden values | `Run tests` |
| **New test cases added** (no golden values exist yet) | `Run functional tests` |
| **Re-enabling a disabled test** (scope `-broken` → active) | `Run functional tests` |
| Non-numerical library code (logging, error handling, CLI flags, refactors) | `Run tests` |
| Could affect training numerics (model arch, attention, optimizer, distributed, MoE routing) | `Run functional tests` |
| Container or dependency changes (`docker/`, `pyproject.toml`, `uv.lock`) | `Run tests` + `container::lts` |
| Touches MBridge integration | add `Run MBridge tests` |

**Rule of thumb:** default to `Run tests`. Always use `Run functional tests` when the PR adds new test cases (golden values must be generated) or when the change could plausibly shift loss curves.

---

## Triggering Internal CI

Use `tools/trigger_internal_ci.py` to push the current branch to the internal
GitLab remote and trigger a pipeline — without touching the GitLab UI.
Full setup and usage details: @tools/trigger_internal_ci.md.

**Prerequisites** (one-time):

```bash
# 1. Add the internal GitLab remote
git remote add gitlab git@<gitlab-hostname>:ADLR/Megatron-LM.git

# 2. Create a personal access token with 'api' scope on your GitLab profile,
#    then store it:
export GITLAB_TOKEN=glpat-<your-token>
```

**Usage:**

```bash
python tools/trigger_internal_ci.py \
  --gitlab-origin gitlab \
  [--functional-test-scope mr] \
  [--functional-test-repeat 5] \
  [--functional-test-cases all] \
  [--dry-run]
```

The script force-pushes the current branch as `pull-request/<branch>` and
prints the resulting pipeline URL.

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
