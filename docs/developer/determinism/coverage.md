---
orphan: true
---

# Measured determinism coverage

Kernel registration says where tests belong. Measured coverage says which
declared cases actually completed a replay protocol on a particular revision,
software stack, GPU, and rank count.

The initial producer annotates local fused-activation tests. The distributed
cross-entropy test is not annotated: its process-group contract needs a separate
adapter before its evidence can be reused by a recipe. Other kernel families
remain visible in `inventory_without_declared_cases`; they are not included in
the case percentage. This is incremental onboarding, not whole-model coverage.

## Adding a case

Mark a positive replay test with an operation ID from `kernels/manifest.py` and
an implementation identifier that distinguishes backend and numerical variants:

```python
@pytest.mark.determinism_case(
    op_id="fused_bias_swiglu", implementation="torch.compile:bias_swiglu"
)
def test_swiglu_replay():
    # Construct representative inputs, then call the existing replay helper.
    assert_replays_bit_exact(fn, inputs, replays=3, backward=True)
```

Parametrized tests declare separate cases. The harness records tensor shapes,
strides, dtypes, gradient requirements, deterministic-algorithm mode, and the
actual comparison protocol. Autocast, TF32, and cuDNN dispatch settings are
recorded at each call, and the run context includes GPU driver versions.
A test-file mapping or a marker alone cannot pass.
Do not annotate artificial negative controls as production operations.

The current protocol compares outputs and gradients within one process. It does
not certify fresh-process dispatch, checkpoint restart, arbitrary shapes,
unobserved internal kernels, or complete mutable training state. Correctness
against an independent reference is a separate check.

## Running and reporting

The dedicated H100 kernel recipe enables collection and writes reports beside
the uploaded logs, including on test failure. For a local GPU run, use a fresh
output directory for each launch:

```bash
export DETERMINISM_EVIDENCE_RUN_ID=my-run
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export MAMBA_DETERMINISTIC=1 CAUSAL_CONV1D_DETERMINISTIC=1
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest \
  -p tools.determinism.pytest_plugin \
  --determinism-evidence-dir /tmp/my-run/shards \
  tests/unit_tests/determinism/kernels/test_fused_activations.py
python -m tools.determinism.coverage /tmp/my-run/shards \
  --revision "$(git rev-parse HEAD)" --output /tmp/my-run/coverage.json
```

The report also writes `coverage.md`. Collection persists each rank's declared
cases before execution and updates observations as comparisons finish. Reusing
a directory with duplicate rank shards is rejected rather than silently picking
the best retry.

| Status | Meaning |
| --- | --- |
| `verified_deterministic` | Nonempty replay evidence passed, all required ranks completed, and source provenance is current and clean |
| `verified_nondeterministic` | An explicit identical-input replay mismatch was observed |
| `not_verified` | Missing replay/rank, skipped or failed setup/teardown, interrupted session, or stale/dirty source |

Expected failures are classified using the typed replay observation. An xfail
caused by a missing dependency is unverified. A numerical mismatch remains
nondeterministic even if pytest expected it. An XPASS is eligible only when the
underlying replay actually completed.

For the declared selected cases, deterministic coverage is `D / (D + N + U)`;
verification coverage is `(D + N) / (D + N + U)`. An empty denominator produces
null percentages. Always publish the selected case list and the unannotated
inventory alongside these metrics; narrowing test selection changes the scope.

The JSON schema is versioned. Consumers must match source and environment
context before reusing observations. The `implementation` identifier is an
author-maintained dispatch contract, not automatic introspection into TE or
compiled kernels.

CPU contract tests can run without importing the GPU test conftest:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/unit_tests/determinism_reporting \
  tests/unit_tests/determinism_reporting/test_coverage_evidence.py
```
