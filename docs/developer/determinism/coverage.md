---
orphan: true
---

# Measured determinism coverage

Kernel registration says where tests belong. Measured coverage says which
declared cases actually completed a replay protocol on a particular revision,
software stack, GPU, and rank count.

The producer annotates local fused activations, TE normalization and attention,
single-rank embedding accumulation, the MoE router GEMM, and SSM decode. These
selected cases cover eleven manifest families, not every variant within them.
The distributed cross-entropy test is not annotated: its process-group contract
needs a separate adapter before its evidence can be reused by a recipe. Other kernel families
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
Triton cache policy, cache directory, and all `TRITON_AUTOTUNE_BLOCK_*` overrides
are recorded both at collection and at the replay call. A different policy or
block override cannot reuse passing evidence. A cache path alone does not prove
identical cache contents or selected configurations; SSM onboarding must also
validate those before making a cross-process claim.
A test-file mapping or a marker alone cannot pass.
Do not annotate artificial negative controls as production operations.

Module/closure tests pass explicit `configuration` to the harness for options
outside the tensor arguments: normalization, attention backend, router dtype,
embedding branch/group size, or the SSM policy override. These `test:`
implementation IDs and configuration fields deliberately require an explicit
recipe adapter; the current function-only inventory cannot reuse them from
matching input shapes alone. Decode evidence is forward-only and includes
the mutated state tensor; it makes no claim about an SSM training backward.

The current protocol compares outputs and gradients within one process. It does
not certify fresh-process dispatch, checkpoint restart, arbitrary shapes,
unobserved internal kernels, or complete mutable training state. Correctness
against an independent reference is a separate check.

## Running and reporting

The dedicated H100 and GB200 kernel recipes enable collection and write reports beside
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

CI passes `--require-verified`: at least one selected case must have passing,
nonempty comparisons on every required rank. Empty selections, all-skipped runs,
and incomplete rank sets fail this gate while retaining available diagnostics.
`--require-case '*pattern*'` additionally requires a nonempty matching selection
and verified-deterministic status for every match. These are execution gates;
they do not require all inventory families to be verified.

## GB200 model replay

The GB200 recipe has separate four-GPU kernel and model buckets, excluded from
the general bucket. `launch_on_gb200` selects only compatible GPT/hybrid
TP/PP/VPP/EP/FSDP cells. Eight-GPU cells remain in the H100 matrix; FSDP cells
require the exact world size so an `fsdp4` label cannot silently exercise eight
shards. The model bucket also selects tensorwise/delayed FP8, MXFP8, and NVFP4.
MXFP8 and NVFP4 each have a required-case gate: skipping either cannot pass.

`determinism_model(model_id=...)` plus `--determinism-evidence-scope model`
uses the same per-rank protocol with a distinct `model_determinism_replay`
report kind. The recipe consumer rejects this kind as operator evidence.
Models compare output and parameter-gradient bytes, including signed zeros
and NaN payloads. Pipeline cells compare the final loss scalar broadcast to
all ranks and each rank's parameter gradients, not every activation tensor.
The report records actual comparison counts, skips, and Torch's warn-only
setting. It does not certify full training state, fresh-process replay, or
checkpoint restart.

GB200 sets `CUDA_DEVICE_MAX_CONNECTIONS=32` before CUDA initialization to allow
scheduling contention. The kernel bucket also runs the harness's injected
signed-zero/NaN mismatch controls, which are excluded from production coverage.
Collection and CPU contract checks do not establish GPU pass rates; the first
protected-runner execution is required to validate this selection and runtime.

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
