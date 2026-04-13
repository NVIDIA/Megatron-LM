---
name: build-and-test
description: Developer environment setup, CI/CD workflows, and CI failure debugging for Megatron-LM. Covers container-based development, uv package management, linting, running tests, CI failure investigation, and common pitfalls.
TRIGGER when: user asks to add, remove, or update a dependency; user edits or asks about pyproject.toml or uv.lock; user asks to set up a dev environment or run tests; user asks about a CI failure or build error.
DO NOT TRIGGER when: user is only reading or discussing code unrelated to dependencies, build, or CI.
---

# Developer Guide

This guide covers the recommended development workflow for Megatron-LM.
The core principle: **build and develop inside containers** — the CI container
ships the correct CUDA toolkit, PyTorch build, and pre-compiled native extensions
(TransformerEngine, DeepEP, …) that cannot be reproduced on a bare host.

---

## Why Containers

Megatron-LM depends on CUDA, NCCL, PyTorch with GPU support, TransformerEngine,
and optional components like ModelOpt and DeepEP. Installing these on a bare host
is fragile and hard to reproduce. The project ships Dockerfiles that pin every
dependency.

**Use the container as your development environment.** This guarantees:

- Identical CUDA / NCCL / cuDNN versions across all developers and CI.
- `uv.lock` resolves the same way locally and in CI.
- GPU-dependent operations (training, testing) work out of the box.

### Step 1 — Acquire an Image

**Option A — NVIDIA-internal: pull a CI-built image**

> ⚠️ Requires access to the internal GitLab instance.
> See `tools/trigger_internal_ci.md` for setup (adding the git remote, obtaining a token).

The internal GitLab CI publishes images to its container registry.
Derive the registry host from your configured `gitlab` remote — the same
host you use for `trigger_internal_ci.py`:

```bash
# Derive host from your 'gitlab' remote:
GITLAB_HOST=$(git remote get-url gitlab | sed 's/.*@\(.*\):.*/\1/')

docker pull ${GITLAB_HOST}/adlr/megatron-lm/mcore_ci_dev:main
```

**Option B — Build from scratch (works for everyone)**

```bash
# dev image (default)
docker build \
  --build-arg FROM_IMAGE_NAME=$(cat docker/.ngc_version.dev) \
  --build-arg IMAGE_TYPE=dev \
  -f docker/Dockerfile.ci.dev \
  -t megatron-lm:local .

# lts image
docker build \
  --build-arg FROM_IMAGE_NAME=$(cat docker/.ngc_version.lts) \
  --build-arg IMAGE_TYPE=lts \
  -f docker/Dockerfile.ci.dev \
  -t megatron-lm:local-lts .
```

Which image variant is used is controlled by the PR label `container::lts`;
absent that label, `dev` is used.

### Step 2 — Launch the Container

**Option A — Local Docker runtime**

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  megatron-lm:local \
  bash -c "<your command>"
```

**Option B — Slurm cluster (for those without a local Docker runtime)**

NVIDIA clusters typically use [Pyxis](https://github.com/NVIDIA/pyxis) +
[enroot](https://github.com/NVIDIA/enroot). Request an interactive session:

```bash
srun \
  --nodes=1 --gpus-per-node=8 \
  --container-image megatron-lm:local \
  --container-mounts $(pwd):/workspace \
  --container-workdir /workspace \
  --pty bash
```

For clusters that require a `.sqsh` archive first:

```bash
enroot import -o megatron-lm.sqsh dockerd://megatron-lm:local
srun \
  --nodes=1 --gpus-per-node=8 \
  --container-image $(pwd)/megatron-lm.sqsh \
  --container-mounts $(pwd):/workspace \
  --container-workdir /workspace \
  --pty bash
```

---

## Dependency Management

Dependencies are declared in `pyproject.toml`. The venv lives at `/opt/venv`
inside the container (already on `PATH`).

> **All `uv` operations must be run inside the container.**
> Never run `uv sync` / `uv pip install` on the host.

### uv Dependency Groups

| Group | Purpose |
|-------|---------|
| `training` | Runtime training extras |
| `dev` | Full dev environment (TransformerEngine, ModelOpt, …) |
| `lts` | LTS-safe subset (no ModelOpt) |
| `test` | pytest, coverage, nemo-run |
| `linting` | ruff, black, isort, pylint |
| `build` | Cython, pybind11, nvidia-mathdx |

Install commands (inside the container):

```bash
# Full dev + test environment
uv sync --locked --group dev --group test

# Linting only
uv sync --locked --only-group linting

# LTS environment
uv sync --locked --group lts --group test
```

Several dependencies are sourced directly from git (TransformerEngine, nemo-run,
FlashMLA, Emerging-Optimizers, nvidia-resiliency-ext). The locked `uv.lock` file
pins exact revisions; update it with `uv lock` when changing `pyproject.toml`.

### Adding a New Dependency

Follow this three-step workflow:

1. **Acquire a container image** — see [Step 1](#step-1--acquire-an-image) above.
2. **Launch the container interactively** — see [Step 2](#step-2--launch-the-container) above.
3. **Update the lock file inside the container**, then commit it:

   ```bash
   # Inside the container:
   uv add <package>          # adds to pyproject.toml and resolves
   uv lock                   # regenerates uv.lock
   # Exit the container, then on the host:
   git add pyproject.toml uv.lock
   git commit -S -s -m "build: add <package> dependency"
   ```

---

## Linting

Run before opening a PR:

```bash
# Check mode (no changes applied)
BASE_REF=main CHECK_ONLY=true SKIP_DOCS=false bash tools/autoformat.sh
```

Tools invoked: `black`, `isort`, `pylint`, `ruff`, `mypy`.

---

## Running Tests

### Test Layout

```text
tests/
├── unit_tests/          # pytest, 1 node × 8 GPUs, torch.distributed runner
├── functional_tests/    # end-to-end shell + training scripts
└── test_utils/
    ├── recipes/
    │   ├── h100/        # YAML recipes for H100 jobs
    │   └── gb200/       # YAML recipes for GB200 jobs
    └── python_scripts/  # helpers (recipe_parser, golden-value download, …)
```

### How Tests Execute

All tests run on a **single DGX H100 node (8 GPUs)**. The GitHub Actions runner
invokes `launch_nemo_run_workload.py`, which uses **nemo-run** to launch a
`DockerExecutor` container. The repo is bind-mounted at `/opt/megatron-lm`;
training data is mounted at `/mnt/artifacts`.

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

### Recipe YAML Structure

Recipes live in `tests/test_utils/recipes/` and are parsed by
`tests/test_utils/python_scripts/recipe_parser.py`. Each file expands a
cartesian `products` block into individual workload specs:

```yaml
type: basic
format_version: 1
spec:
  name: "{test_case}_{environment}_{platforms}_{tag}"
  model: gpt
  nodes: 1
  gpus: 8
  platforms: dgx_h100
  time_limit: 1800
  script_setup: |
    ...
  script: |-
    bash tests/unit_tests/run_ci_test.sh \
      --tag {tag} \
      --environment {environment} \
      --bucket "tests/unit_tests/models/**/*.py" \
      --log-dir {assets_dir}/logs/1/
products:
  - test_case: [my_test]
    environment: [dev, lts]
    tag: [latest, legacy]
    scope: [mr-github]
    n_repeat: [1]
    time_limit: [1800]
```

Key runtime placeholders: `{assets_dir}`, `{artifacts_dir}`, `{test_case}`,
`{environment}`, `{platforms}`, `{tag}`, `{n_repeat}`.

### Adding a Unit Test

1. Create `tests/unit_tests/<category>/test_<name>.py`.
2. Use fixtures from `tests/unit_tests/conftest.py`.
3. Apply markers as needed:
   - `@pytest.mark.internal` — skipped on `legacy` tag
   - `@pytest.mark.flaky` — skipped in `lts` environment
   - `@pytest.mark.experimental` — `latest` tag only
4. Verify the test runs locally inside the container:

   ```bash
   pytest -xvs tests/unit_tests/<category>/test_<name>.py
   ```

5. If the test needs a dedicated CI bucket, add an entry to
   `tests/test_utils/recipes/h100/unit-tests.yaml`.

### Adding a Functional / Integration Test

1. Create `tests/functional_tests/test_cases/<model>/<test_name>/`.
2. Write the shell test script; use `{assets_dir}` for all output paths.
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

### CI Test Scope Labels

| PR label | Scope | Behaviour |
|----------|-------|-----------|
| _(none)_ | `mr-github-slim` | Lightweight subset, fast feedback |
| `Run tests` | `mr-github` | Full suite, lightweight mode (4 steps, no golden compare) |
| `Run functional tests` | `mr-github` | Full suite, 100-step training + golden compare, n_repeat=5 |
| `container::lts` | _(any)_ | Use the LTS base image instead of dev |

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
| `uv sync --locked` fails | Dependency conflict or stale `uv.lock` | Re-run `uv lock` inside the container and commit updated lock |
| `ModuleNotFoundError` after pip install | pip installed outside the uv-managed venv | Use `uv add` and `uv sync`, never bare `pip install` |
| `uv: command not found` inside container | Wrong container image | Use the `megatron-lm` image built from `Dockerfile.ci.dev` |
| `No space left on device` during uv ops | Cache fills container's `/root/.cache/` | Mount a host cache dir via `-v $HOME/.cache/uv:/root/.cache/uv` |
| Pre-commit fails with linting errors | Code style violations | Run `BASE_REF=main CHECK_ONLY=false bash tools/autoformat.sh` |
| Port collision on multi-GPU runs | torchrun binding conflicts | Use `torch.distributed.run` via the container entry point |
