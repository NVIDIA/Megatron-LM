# AGENT.md

Guidance for AI agents operating in this repository.

---

## 1. CI/CD Overview

### Pipeline entry point

The main workflow is `.github/workflows/cicd-main.yml`. It triggers on pushes
to branches matching `pull-request/[0-9]+` and `deploy-release/*`, on merge
groups, on a daily schedule, and on manual dispatch.

High-level stage order:

```
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

### Container build

Dockerfiles live in `docker/`. The default CI image is built from
`docker/Dockerfile.ci.dev`; the LTS variant uses `docker/Dockerfile.ci.lts`.

Which image is used is controlled by the PR label `container::lts`; absent that
label, `dev` is used. Images are pushed to:

- AWS ECR: `766267172432.dkr.ecr.us-east-1.amazonaws.com/…`
- GCP Artifact Registry: `us-east4-docker.pkg.dev/nv-projdgxchipp-20260113193621/megatron-lm/…`

The container is tagged by PR number and commit SHA for full traceability.
Build caching is enabled via the registry.

### Dependency management

> **All `uv` operations must be run inside the `megatron-core` CI container.**
> Never run `uv sync` / `uv pip install` on the host — the container provides
> the correct CUDA toolkit, PyTorch build, and pre-compiled native extensions
> (TransformerEngine, DeepEP, …) that cannot be reproduced outside of it.

#### Getting the container

**Option A — use a pre-built image** (fastest):

```bash
# Images are tagged by PR number and commit SHA.
# Pull the latest image built for the current PR branch:
PR_NUMBER=$(git rev-parse --abbrev-ref HEAD | grep -oP '(?<=pull-request/)\d+')
docker pull 766267172432.dkr.ecr.us-east-1.amazonaws.com/megatron-lm:${PR_NUMBER}

# Or pull the image built from main:
docker pull 766267172432.dkr.ecr.us-east-1.amazonaws.com/megatron-lm:main
```

**Option B — build from scratch**:

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

#### Running work inside the container

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  megatron-lm:local \
  bash -c "<your command>"
```

#### uv dependency groups

Dependencies are declared in `pyproject.toml`. The venv lives at `/opt/venv`
inside the container (already on `PATH`).

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

### Linting

Run before opening a PR:

```bash
# Check mode (no changes applied)
BASE_REF=main CHECK_ONLY=true SKIP_DOCS=false bash tools/autoformat.sh
```

Tools invoked: `black`, `isort`, `pylint`, `ruff`, `mypy`.

---

## 2. Test onboarding

### Test layout

```
tests/
├── unit_tests/          # pytest, 1 node × 8 GPUs, torch.distributed runner
├── functional_tests/    # end-to-end shell + training scripts
└── test_utils/
    ├── recipes/
    │   ├── h100/        # YAML recipes for H100 jobs
    │   └── gb200/       # YAML recipes for GB200 jobs
    └── python_scripts/  # helpers (recipe_parser, golden-value download, …)
```

### How tests execute

All tests run on a **single DGX H100 node (8 GPUs)**. The GitHub Actions runner
invokes `launch_nemo_run_workload.py`, which uses **nemo-run** to launch a
`DockerExecutor` container. The repo is bind-mounted into the container at
`/opt/megatron-lm`; training data is mounted at `/mnt/artifacts`.

**Unit tests** are dispatched through `torch.distributed.run`:

```
--tee 3 --redirects 3 --log-dir {assets_dir}/logs/1/
```

- Ranks 0 and 3 (first and last) are tee-d to stdout; all other ranks write
  only to log files.
- Per-rank log files land at `{assets_dir}/logs/1/` and are uploaded as a
  GitHub artifact after the run.

**Functional tests** are driven by
`tests/functional_tests/shell_test_utils/run_ci_test.sh`. Only the node with
`SLURM_NODEID=0` runs the pytest validation step; training output from all ranks
is written to `{assets_dir}/` and uploaded as an artifact.

**Flaky-failure auto-retry**: `launch_nemo_run_workload.py` reads
`assets_dir/logs/*/*/attempt_0/*/std*.log` across all ranks after a failure.
If it matches a known transient pattern (NCCL timeout, ECC error, segfault,
HuggingFace connectivity, …) the job is retried up to **3 times** before being
declared a genuine failure.

### Recipe YAML structure

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
  time_limit: 1800          # seconds
  script_setup: |           # runs before the test (git ops, env setup)
    ...
  script: |-                # the actual test command; use {assets_dir} for output paths
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

### Adding a unit test

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

### Adding a functional / integration test

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

### CI test scope labels

| PR label | Scope | Behaviour |
|----------|-------|-----------|
| _(none)_ | `mr-github-slim` | Lightweight subset, fast feedback |
| `Run tests` | `mr-github` | Full suite, lightweight mode (4 steps, no golden compare) |
| `Run functional tests` | `mr-github` | Full suite, 100-step training + golden compare, n_repeat=5 |
| `container::lts` | _(any)_ | Use the LTS base image instead of dev |

---

## 3. CI-failure assistance

### Locating the PR from a CI branch

CI branches always follow the pattern `pull-request/<number>`. To find the
associated PR:

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

### Reading CI job logs

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
#    Chunk each file and read one chunk at a time to stay within context limits.
#    First, check how large a file is:
wc -l ./ci-logs/<test>/<attempt>/attempt_0/<rank>/stderr.log

#    Then read in chunks of ~200 lines, advancing the offset until the error is found:
#    (using the Read tool with offset/limit, or via sed for shell use)
sed -n '1,200p'   ./ci-logs/.../stderr.log   # chunk 1
sed -n '201,400p' ./ci-logs/.../stderr.log   # chunk 2
# … continue until the traceback / error is found, then stop.
```

Inside the artifact the log tree mirrors the container's `assets_dir`:

```
ci-logs/
└── <test-name>/
    └── <attempt>/
        └── attempt_0/
            └── <rank>/
                ├── stdout.log
                └── stderr.log
```

The glob `assets_dir/logs/*/*/attempt_0/*/std*.log` captures all ranks across
all attempts — this is also what the flaky-failure detector reads before
deciding whether to retry.

### Identifying failure root cause

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
   transient patterns (NCCL timeout, ECC error, segfault, HuggingFace
   connectivity, …). If the job exhausted all retries and the failure matches
   one of those patterns it is infrastructure noise, not a code regression.
   For genuinely non-deterministic test logic, mark with `@pytest.mark.flaky`
   (or `@pytest.mark.flaky_in_dev`) and open a follow-up issue.

### Correlating a failure with the PR changeset

After extracting the PR number and fetching the diff (see above), map changed
files to test ownership:

```bash
# Find unit tests that cover a changed source file
# e.g. megatron/core/transformer/attention.py → tests/unit_tests/models/
grep -r "from megatron.core.transformer.attention" tests/unit_tests/ -l

# Check CODEOWNERS for reviewer assignment
cat .github/CODEOWNERS | grep "<changed-path>"
```

Use this mapping to determine whether the failure is directly caused by the
PR's changes or is a pre-existing issue on `main`.
