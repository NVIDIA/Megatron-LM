---
name: mcore-testing
description: Test system for Megatron-LM. Covers test layout, recipe YAML structure, adding and running unit and functional tests, golden values, marker filters, and CI parity. Use when adding or running a unit or functional test; understanding the test layout; writing a recipe YAML; downloading or updating golden values; reproducing a test failure locally; 'how do I add a test', 'run unit tests', 'pytest fails', 'test layout', 'golden values', 'recipe YAML', 'marker filter'.
license: Apache-2.0
metadata:
  author: Oliver Koenig <okoenig@nvidia.com>
---

# Testing Guide

---

## Answer-First Testing Facts

For questions about disabling tests without deleting them:

- Functional recipe entries stay in YAML; disable by suffixing scope with
  `-broken`, for example `scope: [mr-github]` -> `scope: [mr-github-broken]`.
- Unit-test skips use pytest markers instead: `@pytest.mark.flaky_in_dev` skips
  in the default dev environment, and `@pytest.mark.flaky` skips in LTS.
- Do not delete the test case or recipe entry when the goal is discoverability
  and easy re-enable.

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

### Disabling a Test Without Deleting It

To temporarily disable a test case in a recipe YAML, suffix its `scope` value
with `-broken` — **do not delete the entry**:

```yaml
# before (test runs in CI)
scope: [mr-github]

# after (test is skipped; entry preserved for easy re-enable)
scope: [mr-github-broken]
```

---

## Running Unit Tests Locally

All unit tests initialize a `torch.distributed` group, so every invocation
requires GPU access and must go through `torch.distributed.run`:

```bash
# Full suite
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests

# Single file
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/models/test_gpt_model.py

# Single test
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor

# Filter by name substring
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests -k optimizer
```

### Marker filters

```bash
# Exclude flaky tests during development
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests -m "not flaky and not flaky_in_dev"

# Include experimental tests
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests --experimental
```

### CI parity

Use `tests/unit_tests/run_ci_test.sh` to reproduce a CI bucket failure exactly.
For ad-hoc runs, prefer the direct `torch.distributed.run` invocations above.

### Gotchas

- `pyproject.toml` sets `addopts = --durations=15 -s -rA` — stdout is not
  captured (`-s`), so ranks interleave during multi-rank runs. Override with
  `--capture=fd` when debugging a specific rank.
- `tests/unit_tests/conftest.py` looks for test data under `/opt/data` and
  attempts a download if missing. Supply it manually or skip data-dependent
  tests when running outside the canonical container.

---

## Adding a Unit Test

### Choose focused behavioral coverage

- **Add the minimum tests needed to protect changed behavior.** Bug fixes
  should include a focused regression test where practical.
- **Inspect nearby tests before adding coverage.** Extend or parameterize
  existing tests when appropriate; do not duplicate behavior checks.
  Identical line coverage alone does not establish redundancy.
- **Do not add tests for trivial changes.** Documentation, formatting,
  simple re-exports, and straightforward getters/setters generally do not
  need new unit tests unless they protect a meaningful contract or known
  regression.
- **Assert meaningful results against independent expectations.** Do not
  reproduce production logic, assert only mocked return values, or check
  private call sequences without a behavioral reason.
- **Use representative cases and relevant boundaries.** Do not generate
  exhaustive matrices of shapes, dtypes, seeds, and TP/PP/DP configurations
  by default. Each additional dimension or case should address a distinct
  failure mode.
- **Reuse fixtures and helpers and keep compute costs proportional.** Avoid
  unnecessary GPU allocation, distributed launches, large tensors, model
  downloads, or new dependencies when a smaller test adequately validates
  the behavior. Follow the target suite's runner and fixture requirements;
  retain GPU or distributed execution when it is needed to exercise the
  changed behavior. These scope rules do not waive the kernel determinism
  coverage required in the implementation steps below.

### Test behavior, not configuration snapshots

- **Do not test recipe or test-script settings.** Do not add a unit test
  that only asserts that a key in an existing config, or an argument in an
  existing functional or nightly test script, has a given value. Do not
  unit-test functional test scripts at all; running them is their test.
  Recipe and script changes have a reason; snapshot assertions only fail CI
  on the next intentional change. A check across all configs that prevents
  a known issue is appropriate, such as rejecting a parallelism combination
  with a documented accuracy bug.
- **Do not add a unit test that only checks that an individual shipped
  example config loads or validates.** Put shared loading, inheritance, and
  schema-validation checks in a parametrized config-validation test. Extend
  that coverage when a new kind of config needs a check instead of adding
  a separate test for each config. Check for existing coverage first; do not
  assume every shipped config is already covered.
- **Do not add a unit test that only asserts a config field's default
  value.** A default change affects every run that omits the key; its reason
  belongs in the PR, not in a test that fails on the change. Test what the
  code does with the value instead, such as rejecting an invalid combination
  or computing a derived property.
- **Do not add a unit test that only checks what Pydantic already does.**
  This includes omitted fields falling back to defaults, passed values
  reading back unchanged, and rejection of missing required fields or
  violations of a `Literal`, type, or `Field` constraint (`ge`, `gt`,
  `allow_inf_nan`, `PositiveInt`). Test these cases only when our own code
  sits between the input and the model, such as a custom merge or a
  `mode="before"` validator. For a custom validator, testing that an edge
  value is accepted is appropriate: constructing the model without an error
  is the check, so do not read the value back.

### Implementation steps

1. Create `tests/unit_tests/<category>/test_<name>.py`.
2. Use fixtures from `tests/unit_tests/conftest.py`.
3. Apply markers as needed:
   - `@pytest.mark.internal` — skipped on `legacy` tag
   - `@pytest.mark.flaky_in_dev` — skipped in `dev` environment (CI default; use this to disable a flaky test without blocking the standard pipeline)
   - `@pytest.mark.flaky` — skipped in `lts` environment
   - `@pytest.mark.experimental` — `latest` tag only
4. Verify locally (see Running Unit Tests Locally above).
5. If the test needs a dedicated CI bucket, add an entry to
   `tests/test_utils/recipes/h100/unit-tests.yaml`.
6. If the change adds or modifies a GPU kernel (Triton, `jit_fuser` /
   `torch.compile`, CUDA extension, TE or external-library dispatch, or a
   scatter/index accumulation), add or update its bit-exact replay test under
   `tests/unit_tests/determinism/kernels/` and register it in
   `tests/unit_tests/determinism/kernels/manifest.py`. The `linting` CI job
   (`tools/check_kernel_determinism_coverage.py`) fails kernel PRs without
   this; the `determinism-exempt` label overrides it for non-numeric edits.
   See `docs/developer/determinism/testing.md`.

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

Golden values keep the full `float32` precision of the TensorBoard scalars,
record it as `"value_precision": "full"`, and deterministic test cases are
compared bit-exactly against them. Never round or hand-edit them:
`tools/check_golden_values.py` (run by the `linting` CI job on changed golden
files) rejects golden files of deterministically compared cases whose metrics
lack the `full` marker. Legacy files (no marker) are still compared at five
decimals until they are regenerated from a CI run.

---

Golden comparisons report absolute and relative errors; see [diagnostics](../../docs/developer/determinism/testing.md).

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Test passes locally but fails in CI | Different environment or data path | Check `DATA_PATH`, `DATA_CACHE_PATH`, and the `environment` tag (`dev` vs `lts`) |
| Golden value mismatch after a code change | Numerical regression | Download new golden values via `download_golden_values.py` after a clean run |
| `cicd-integration-tests-gb200` not triggered | GB200 jobs require maintainer status | Ask a maintainer to trigger, or add the `Run functional tests` label |
