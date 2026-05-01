---
name: testing
description: Test system for Megatron-LM. Covers test layout, recipe YAML structure, adding unit and functional tests, golden values, and running tests locally.
when_to_use: Adding a unit or functional test; understanding the test layout; writing a recipe YAML; downloading or updating golden values; reproducing a test failure locally; 'how do I add a test', 'test layout', 'golden values', 'recipe YAML'.
---

# Testing Guide

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

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Test passes locally but fails in CI | Different environment or data path | Check `DATA_PATH`, `DATA_CACHE_PATH`, and the `environment` tag (`dev` vs `lts`) |
| Golden value mismatch after a code change | Numerical regression | Download new golden values via `download_golden_values.py` after a clean run |
| `cicd-integration-tests-gb200` not triggered | GB200 jobs require maintainer status | Ask a maintainer to trigger, or add the `Run functional tests` label |
