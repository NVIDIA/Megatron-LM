---
orphan: true
---

# Kernel Determinism Testing

> For the determinism contract and the operation catalog, refer to
> [`status.md`](./status.md) and [`op-catalog.md`](./op-catalog.md).

Every GPU kernel Megatron dispatches must have a bit-exact determinism test,
and every pull request that adds or changes a kernel must add or update that
test. This page describes what counts as a kernel, where the tests live, how
the requirement is enforced, and how to add a test.

## What counts as a kernel

Any code path where Megatron itself launches or selects a GPU kernel whose
numerical result could depend on scheduling:

- Triton `@triton.jit` kernels (and TileLang / cuTile kernels)
- `@jit_fuser` / `torch.compile` fused functions
- C++/CUDA extensions (`load_inline`, `CUDAExtension`, `.cu` sources)
- Transformer Engine and external-library dispatch with algorithm choices
  (grouped GEMM, attention backends, causal_conv1d, mamba_ssm, FLA, DeepEP,
  cuTile, FlashInfer, apex multi-tensor kernels)
- torch ops with a non-deterministic accumulation: `scatter_add_`,
  `index_add_`, `index_put_(accumulate=True)`, `bincount`, embedding backward

`tests/unit_tests/determinism/kernels/manifest.py` lists the patterns
(`KERNEL_CONTENT_PATTERNS`) and the directories (`KERNEL_DIRECTORIES`) that
make a file *kernel-bearing*. External-library patterns match *call sites*
(`causal_conv1d_fn(`, `tex.rmsnorm_fwd(`, `buffer.dispatch(`, ...), never
bare imports, so a module that only imports a class or checks `isinstance`
is not kernel-bearing. Modules that select among or call external kernels
without defining one (the Mamba mixer, gated delta product, RoPE dispatch,
FP8 master-weight casts, ...) are registered with `kind="dispatch"`; they are
covered by the replay tests of the kernels they call, named in the entry's
`notes`, and by module-level tests where those exist.

## Where the tests live

| Layer | Location | What it asserts |
| --- | --- | --- |
| Kernel | `tests/unit_tests/determinism/kernels/test_*.py` | One kernel family per file. Each kernel is run several times on identical inputs and every output tensor and gradient must be byte-identical (`harness.assert_replays_bit_exact`, `assert_module_replays_bit_exact`; `harness.bytes_equal` compares bit patterns, so signed zeros and NaN payloads must match too). |
| Module / model | `tests/unit_tests/determinism/correctness/` | GPT, TransformerBlock, HybridModel and FP8/FP4 recipes across parallelism cells (`BitExactRunner`). |
| End to end | functional tests with `--deterministic-mode` | Loss and `num-zeros` compared against golden values at their recorded precision; legacy goldens use five decimals. |

The manifest (`tests/unit_tests/determinism/kernels/manifest.py`) is the
registry that ties kernel source files to the tests that cover them. Each
`KernelEntry` names the source files, the tests, the kernel kind, and either a
non-empty `tests` tuple or an explicit `exempt_reason` (for example a
multi-rank NVLS collective that needs an NVLink peer group, or an allocator
with no compute kernel). Exemptions are visible coverage debt, not silence.

## Golden-value error diagnostics

The functional comparison logs differences even when they pass tolerance. For
each metric/check, it reports the check result, comparison precision, `atol`,
`rtol`, and the number of samples outside tolerance. It shows the samples with
the largest absolute and relative errors, plus the first sample outside
tolerance if that sample is not already shown. Each sample includes its step,
golden value, actual value, and allowed absolute error:

- Absolute error: `abs(actual - golden)`.
- Relative error: `abs(actual - golden) / abs(golden)`, also shown as a percentage.
- Allowed absolute error: `atol + rtol * abs(golden)`.

Relative error is undefined for a zero golden value and nonzero actual value;
the absolute tolerance still applies. Non-finite or missing samples have
undefined errors and are excluded from the finite-sample extrema.

The diagnostics use the same precision and aggregation as the check. Exact
checks honor the golden's precision marker, approximate checks retain their
five-decimal comparison, and iteration-time errors compare unrounded medians
over the selected steady-state samples. Existing tolerance, failure-budget,
and legacy-placeholder policies are unchanged. In particular, an approximate
check can pass with some samples outside tolerance; the report shows those
outliers even when the check passes. Exact checks retain zero tolerance.

A large error is a signal to investigate numerical correctness. A small
historical-golden difference can also result from a valid change in reduction
order. Use independent-reference correctness checks and same-implementation
replay to assess the change before reviewing a golden refresh. Reporting an
error never updates a golden value.

## How the requirement is enforced

1. **Repository invariant** (`tests/unit_tests/determinism/kernels/test_manifest.py`,
   CPU only, runs in the unit-test bucket): every kernel-bearing file under
   `megatron/` is registered, every registered path exists, and every entry
   has tests or an exemption. Adding a Triton kernel to a new file without
   registering it fails the unit tests.
2. **Pull-request gate** (`tools/check_kernel_determinism_coverage.py`, run by
   the `linting` job in `.github/workflows/cicd-main.yml` on PR pushes):
   - a changed kernel-bearing file must be registered (not overridable);
   - a changed registered kernel source must come with a change to at least
     one of its determinism tests. Override with the `determinism-exempt`
     PR label when the change cannot affect numerics (comment-only edits,
     refactors); the check then logs the exemption instead of failing.
3. **Review**: the PR template checkbox and the `/claude review` prompt ask
   for the test explicitly.

The full kernel bucket runs in the H100 unit-test recipe. Dedicated GB200
kernel and model buckets select a four-GPU subset with `launch_on_gb200`;
larger configurations stay in the H100 matrix. Replay reports require real
comparisons on every rank, and the GB200 model gate requires MXFP8 and NVFP4
evidence explicitly. See [measured coverage](./coverage.md) for the selection,
scope, and artifacts. Scheduling a test does not establish a hardware pass.

Run the gate locally against `main`:

```bash
python3 tools/check_kernel_determinism_coverage.py --base-ref origin/main
```

## Adding a kernel test

1. Put the test in the `tests/unit_tests/determinism/kernels/test_*.py`
   module that matches the kernel family (or add one; the package `__init__`
   pins the determinism environment at import).
2. Use the harness:

   ```python
   from tests.unit_tests.determinism.kernels.harness import (
       assert_replays_bit_exact,
       assert_module_replays_bit_exact,
       deterministic_algorithms,
       seeded,
   )

   seeded()
   x = torch.randn(16384, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
   assert_replays_bit_exact(my_kernel, (x,), replays=3, contention=True, what="my_kernel")
   ```

   `contention=True` runs the replays under side-stream GEMM pressure so
   ordering-dependent reductions surface. Size inputs so many CTAs contend on
   the same outputs; a two-block reduction can be deterministic by accident.
   Kernels that select a branch on `torch.are_deterministic_algorithms_enabled()`
   should be exercised on both branches with `deterministic_algorithms(...)`.
   Kernels that consume the RNG (dropout) use `restore_rng=True`.
3. When the default path is known to race (atomics), add a negative control
   with `count_differing_replays(...) > 0`, generously sized, so the strict
   assertion is known to be sensitive (see `test_moe_kernels.py`).
4. Register the kernel in `manifest.py`: source files, the test file, kind,
   and a note on the non-determinism mechanism and how the deterministic
   branch is selected.
5. Run the file on a GPU node:

   ```bash
   uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
     tests/unit_tests/determinism/kernels/test_my_family.py \
     tests/unit_tests/determinism/kernels/test_manifest.py
   ```

## Author accuracy and sensitivity checks

Repeatability alone cannot detect a consistently wrong kernel. For new or
updated supported cases, also compare outputs and every input gradient against
an independent implementation, with explicit dtype-specific `rtol` and `atol`.
Use `tools.determinism.reference.assert_reference_close` with the replay's
actual output/gradient dictionaries, an independently computed reference pair,
`harness.replay_signature(inputs, backward=True)`, and a versioned reference ID.
The signature must include the same explicit `configuration` as the replay.
The helper records per-tensor error magnitudes, violations, and a sample;
it rejects missing keys, incompatible shapes/dtypes, and nonfinite values.

For reductions, supply explicit `ReductionReference` entries computed from
independent terms, keyed by `gradient:<tensor name>`. Document any rounding before
the sum and retain ideal values using `mathematical_reference`. Reduction checks
apply both a component budget based on accumulation precision/length/conditioning
and an L2 guard using the original tolerances; see the
[accuracy policy](./coverage.md#independent-accuracy-and-sensitivity). These
worst-case budgets are conservative and do not prove kernel correctness.
Set `numerical_controls=True` to require out-of-budget perturbations to fail for
every tensor. This exercises the numerical gate separately from byte sensitivity.

Use `assert_replay_sensitivity` with the same actual pair and the real byte
comparator. It first checks the unchanged baseline, then flips a bit separately
in every output and gradient. All perturbations must be detected. This verifies
comparator wiring; it does not replace scheduling contention or an observed
atomic-race negative control. Synthetic perturbations never count as production
nondeterminism evidence.

Register each required parametrized pytest node ID in `KernelEntry.author_tests`.
The H100 and GB200 kernel report gates require both checks for every listed
case on every rank, matching the replay signature and tensor counts. Deleted,
renamed, skipped, incomplete, or stale cases fail the gate. Adoption starts with
biased SwiGLU, weighted SwiGLU, and weighted squared ReLU in FP32 and BF16;
other entries retain their existing replay contract without an accuracy claim.
See `test_mlp_activation_author_evidence` for the pilot implementation and
[measured coverage](./coverage.md) for report semantics and validation limits.

PR evidence should link the reference/replay artifacts and separate, uninstrumented
forward/backward timings, with the source revision, hardware, inputs, software
versions, and deterministic/default settings. The performance driver can join
these artifacts using the shared local-activation contract; missing/mismatched
phases remain unverified and unbudgeted timing rows cannot give a performance
pass. Calibrated hardware budgets and broad CI enforcement remain follow-up work;
these accuracy checks impose no performance threshold. Investigate a reference or historical-golden mismatch
before accepting a changed baseline, even when same-implementation replay passes.
