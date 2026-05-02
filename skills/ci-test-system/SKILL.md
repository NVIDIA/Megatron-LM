---
name: ci-test-system
description: Test system, CI pipeline, and CI failure investigation for Megatron-LM. Covers test layout, recipe YAML structure, adding unit and functional tests, CI scope labels, triggering internal GitLab CI, pipeline structure, and debugging CI failures.
TRIGGER when: user asks to run tests, add a test, investigate a CI failure, understand the CI pipeline, or work with test recipes; user opens or pushes to a PR and needs to know which CI label to attach; user wants to trigger the internal GitLab CI pipeline; user asks to download golden values or references a pipeline/run ID in the context of golden values.
DO NOT TRIGGER when: user is only setting up the dev environment or managing dependencies (use build-and-dependency instead).
---

# Test System & CI Guide

---

## Test Layout

```text
tests/
├── unit_tests/          # pytest, 1 node × 8 GPUs, torch.distributed runner
├── functional_tests/    # end-to-end shell + training scripts
│   └── test_cases/
│       └── {model}/{test_case}/
│           ├── model_config.yaml          # training args
│           └── golden_values_{env}_{platform}.json
└── test_utils/
    ├── recipes/
    │   ├── h100/        # YAML recipes for H100 jobs
    │   └── gb200/       # YAML recipes for GB200 jobs
    └── python_scripts/  # helpers (recipe_parser, golden-value download, …)
```

---

## How Tests Execute

The GitHub Actions runner invokes `launch_nemo_run_workload.py`, which uses
**nemo-run** to launch a `DockerExecutor` container. The repo is bind-mounted
at `/opt/megatron-lm`; training data is mounted at `/mnt/artifacts`.

**Unit tests** are dispatched through `torch.distributed.run`:

- Ranks 0 and 3 are tee-d to stdout; all other ranks write only to log files.
- Per-rank log files land at `{assets_dir}/logs/1/` and are uploaded as a
  GitHub artifact after the run.

**Functional tests** are driven by
`tests/functional_tests/shell_test_utils/run_ci_test.sh`. Only rank 0 runs the
pytest validation step; training output from all ranks is uploaded as an artifact.

**Flaky-failure auto-retry**: `launch_nemo_run_workload.py` retries up to
**3 times** for known transient patterns (NCCL timeout, ECC error, segfault,
HuggingFace connectivity, …) before declaring a genuine failure.

---

## Recipe YAML Structure

Recipes live in `tests/test_utils/recipes/` and are parsed by
`tests/test_utils/python_scripts/recipe_parser.py`. Each file expands a
cartesian `products` block into individual workload specs:

```yaml
type: basic
format_version: 1
maintainers: [mcore]
loggers: [stdout]
spec:
  name: "{test_case}_{environment}_{platforms}"
  model: gpt              # maps to tests/functional_tests/test_cases/{model}/
  build: mcore-pyt-{environment}
  nodes: 1
  gpus: 8
  n_repeat: 5
  platforms: dgx_h100
  time_limit: 1800
  script_setup: |
    ...
  script: |-
    bash tests/functional_tests/shell_test_utils/run_ci_test.sh ...
products:
  - test_case: [my_test]
    products:
      - environment: [dev, lts]
        scope: [mr-github]
        platforms: [dgx_h100]
```

Key runtime placeholders: `{assets_dir}`, `{artifacts_dir}`, `{test_case}`,
`{environment}`, `{platforms}`, `{n_repeat}`.

### CI Test Scope Labels

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

### Disabling a Test Without Deleting It

To temporarily disable a test case in a recipe YAML, suffix its `scope` value
with `-broken` — **do not delete the entry**:

```yaml
# before (test runs in CI)
scope: [mr-github]

# after (test is skipped; entry preserved for easy re-enable)
scope: [mr-github-broken]
```

This applies to any scope token (`mr-github`, `mr-github-slim`, `mr-gitlab`,
etc.). Deleting the entry entirely would require recreating the test case
definition when the fix lands.

### Which label to attach when opening a PR

Apply this logic based on what the PR changes:

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

## Adding a Unit Test

1. Create `tests/unit_tests/<category>/test_<name>.py`.
2. Use fixtures from `tests/unit_tests/conftest.py`.
3. Apply markers as needed:
   - `@pytest.mark.internal` — skipped on `legacy` tag
   - `@pytest.mark.flaky` — skipped in `lts` environment
   - `@pytest.mark.experimental` — `latest` tag only
4. Verify locally inside the container:

   ```bash
   pytest -xvs tests/unit_tests/<category>/test_<name>.py
   ```

5. If the test needs a dedicated CI bucket, add an entry to
   `tests/test_utils/recipes/h100/unit-tests.yaml`.

---

## Adding a Functional / Integration Test

1. Create `tests/functional_tests/test_cases/<model>/<test_name>/`.
2. Write `model_config.yaml` with `MODEL_ARGS`, `ENV_VARS`, and `TEST_TYPE`.
3. Add a YAML recipe under `tests/test_utils/recipes/h100/` (and `gb200/` if
   needed). Required fields: `scope`, `environment`, `platform`, `n_repeat`,
   `time_limit`.
4. Push the PR, add the label **"Run functional tests"** to trigger a full run.
5. After a successful run, download golden values:

   ```bash
   python tests/test_utils/python_scripts/download_golden_values.py \
     --source github --pipeline-id <run-id>
   ```

6. Commit the downloaded golden values.

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

## CI Pipeline

The main workflow is `.github/workflows/cicd-main.yml`. It triggers on pushes
to branches matching `pull-request/[0-9]+` and `deploy-release/*`, on merge
groups, on a daily schedule, and on manual dispatch.

### Pipeline Structure

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

# List the files changed in the PR
gh pr view "$PR_NUMBER" --repo NVIDIA/Megatron-LM --json files --jq '.files[].path'
```

If the branch name contains a non-numeric suffix (e.g. `pull-request/my-branch`),
search by branch name instead:

```bash
gh pr list --repo NVIDIA/Megatron-LM --head "pull-request/my-branch"
```

### Reading CI Job Logs

```bash
# List recent workflow runs for the PR
gh run list --repo NVIDIA/Megatron-LM --branch "pull-request/$PR_NUMBER"

# Show summary of a specific run
gh run view <run-id> --repo NVIDIA/Megatron-LM

# Stream the GitHub Actions runner output (stdout of ranks 0 and 3 only)
gh run view <run-id> --repo NVIDIA/Megatron-LM --log-failed
```

Full per-rank logs are **not** in the runner stdout. They are uploaded as
GitHub artifacts named `logs-<test_case>-<run_id>-<uuid>`.

If the runner output does not show a clear error, download the full artifact
and crawl all rank logs:

```bash
# 1. Find the artifact name for the failing run
gh run view <run-id> --repo NVIDIA/Megatron-LM --json artifacts \
  --jq '.artifacts[].name'

# 2. Download the artifact zip
gh run download <run-id> --repo NVIDIA/Megatron-LM \
  --name "logs-<artifact-name>" -D ./ci-logs

# 3. Locate which rank logs contain errors (file list only, no content yet)
grep -r -l "ERROR\|Traceback\|FAILED\|fatal" ./ci-logs/

# 4. Log files can exceed 10 000 lines — never read a full log at once.
#    Check size first, then read in chunks of ~200 lines:
wc -l ./ci-logs/<test>/<attempt>/attempt_0/<rank>/stderr.log
sed -n '1,200p'   ./ci-logs/.../stderr.log   # chunk 1
sed -n '201,400p' ./ci-logs/.../stderr.log   # chunk 2
# … continue until the traceback / error is found, then stop.
```

Inside the artifact the log tree mirrors the container's `assets_dir`:

```text
ci-logs/
└── <test-name>/
    └── <attempt>/
        └── attempt_0/
            └── <rank>/
                ├── stdout.log
                └── stderr.log
```

### Identifying Failure Root Cause

1. **Linting failure** — re-run `tools/autoformat.sh` locally; the diff shows
   exactly what needs to change.
2. **Container build failure** — inspect the `cicd-container-build` job log.
   Common causes: new dependency with conflicting pins in `uv.lock`, or a broken
   git-sourced package revision.
3. **Unit test failure** — the failing bucket is identified in the
   `cicd-unit-tests-latest` job matrix. Ranks 0 and 3 appear in runner stdout;
   for failures on other ranks download the artifact and check per-rank logs.
   Re-run the specific bucket locally inside the container:

   ```bash
   bash tests/unit_tests/run_ci_test.sh \
     --tag latest --environment dev \
     --bucket "<glob from recipe>" \
     --log-dir ./logs
   ```

4. **Functional test failure** — look at the `cicd-integration-tests-*` job.
   Failures in lightweight mode indicate a crash; failures with golden-value
   mismatch indicate a numerical regression. Only rank 0 runs pytest validation,
   so start with `stdout.log` for rank 0 in the artifact.
5. **Flaky test** — the runner retries automatically up to 3 times for known
   transient patterns. If the job exhausted all retries and the failure matches
   one of those patterns it is infrastructure noise, not a code regression.
   Mark genuinely non-deterministic tests with `@pytest.mark.flaky` and open a
   follow-up issue.

### Correlating a Failure with the PR Changeset

```bash
# Find unit tests that cover a changed source file
grep -r "from megatron.core.transformer.attention" tests/unit_tests/ -l

# Check CODEOWNERS for reviewer assignment
cat .github/CODEOWNERS | grep "<changed-path>"
```

Use this mapping to determine whether the failure is directly caused by the
PR's changes or is a pre-existing issue on `main`.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Port collision on multi-GPU runs | torchrun binding conflicts | Use `torch.distributed.run` via the container entry point |
| Test passes locally but fails in CI | Different environment or data path | Check `DATA_PATH`, `DATA_CACHE_PATH`, and the `environment` tag (`dev` vs `lts`) |
| Golden value mismatch after a code change | Numerical regression | Download new golden values via `download_golden_values.py` after a clean run |
| `cicd-integration-tests-gb200` not triggered | GB200 jobs require maintainer status | Ask a maintainer to trigger, or add the `Run functional tests` label |
