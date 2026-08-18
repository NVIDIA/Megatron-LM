# SM100 fused GDR backward

This package implements the training backward path for the Megatron Core
internal Gated Delta Rule (GDR) backend on Blackwell SM100 GPUs. It fuses the
chunk-local and recurrent-state gradient work into one CuTe DSL launch.

## Recommended usage

Model code should select the internal backend through `TransformerConfig`:

```python
config = TransformerConfig(
    ...,
    gdn_gdr_backend="internal",
    gdn_gdr_recompute_h=False,
)
```

Runtime dispatch is controlled by `MCORE_GDN_INTERNAL_BACKEND`:

- `auto` (default): use CuTe DSL for supported inputs and otherwise use FLA.
- `cute`: require the CuTe DSL path and raise an error for unsupported inputs.
- `fla`: bypass CuTe DSL.

The production backward call path is:

```text
implementation.InternalChunkGatedDeltaRuleFunction.backward
  -> implementation._cutedsl_backward
  -> implementation._call_fused_gdr_bwd_cute
  -> fused_bwd.fused_gdr_bwd
  -> launcher.launch_fused_gdr_bwd
  -> kernel.FusedGdrBwdKernel
```

The low-level wrapper is an internal interface. Callers outside this package
should use the configured GDN layer so that packing, dtype conversion,
metadata creation, fallback, and autograd wiring remain consistent.
For supported fused-backward shapes, `gdn_gdr_recompute_h` controls the
activation-memory/compute tradeoff for both forward implementations:

- `False` (default): the FLA forward in `auto` mode or the fused forward in
  explicit `cute` mode saves each chunk's input recurrent state `h`. Autograd
  passes it directly to the fused backward.
- `True`: forward drops `h`; backward reconstructs it from saved `A`, K, V,
  gates, and beta before launching the fused backward.

The standalone `gdn_gdr_backend="fla"` backend bypasses this internal autograd
path, so the flag has no effect there. The GB200 E2E test asserts that both
fused kernels run and compares the full forward-plus-backward path with FLA.

## Package structure

- `fused_bwd.py` validates the public kernel contract, packs dense batches,
  creates variable-length metadata, allocates outputs, and restores output
  shapes and dtypes.
- `launcher.py` owns DLPack descriptors, dynamic token-mode annotations,
  specialization keys, thread-safe compilation caching, and stream-bound
  launch preparation.
- `layouts.py` owns the ten oriented MMA variants, nineteen logical
  operation bindings, canonical and packed SMEM layouts, accumulator-layout
  checks, and the trace-local TMA descriptor bundle.
- `storage.py` owns the decorated shared-memory struct, TMEM live-range
  contract, resource budget, and named SMEM/TMEM view builders.
- `kernel.py` contains the role schedule, pipeline ordering, tensor-bound TMA
  descriptor construction, allocation/free ordering, and the
  `FusedGdrBwdKernel` entry point.
- `tcgen05_ws.py` contains the small low-level helpers needed for SM100 tensor
  memory and `tcgen05` operations.

## Input contract

`fused_bwd.fused_gdr_bwd` accepts one or more packed sequences:

- CUDA 13 or newer on an SM100 GPU; all tensors are contiguous and on the
  same device.
- `q`, `k`, `v`, and `do`: BF16 `[1, N, 64, 128]`.
- `a`: BF16 `[1, N, 64, 64]`.
- `g` and `beta`: FP32 `[1, N, 64]`.
- `h`: BF16 `[1, N / 64, 64, 128, 128]`.
- `dht`: FP32 `[B, 64, 128, 128]`.
- `cu_seqlens`: contiguous CUDA int32 `[B + 1]`, starts at zero, ends at `N`,
  and describes `B` logical sequences. Each sequence length is a positive
  multiple of 64.
- `chunk_size=64`, `state_v_first=False`, no grouped-query head mapping, and a
  finite positive `scale`.

The physical leading dimension of token tensors is always one because the
wrapper uses packed THD storage. `B` is the logical sequence count encoded by
`cu_seqlens`; it is not restricted to two. Dense BTHD inputs with any positive
batch size are flattened and receive generated offsets before this interface.

The result is `(dq, dk, dv, dg, dbeta, dh0)`. Token gradients match their
corresponding input shapes; `dh0` is FP32 `[B, 64, 128, 128]`.

## Algorithm and schedule

The kernel launches one CTA for each logical `(sequence, head)` pair and walks
that sequence's 64-token chunks in reverse. It keeps recurrent state gradients
and MMA accumulators in TMEM while using TMA and mbarrier pipelines to stage
Q/K/V, gates, beta, `A`, output gradients, and saved forward states.

The pinned MMA schedule has 16 dependency phases. In broad terms it:

1. Reconstructs the chunk-local score path and corrected value gradients.
2. Propagates `dH` through the recurrent state and produces `dV`.
3. Forms the `A`, gate, and beta gradient terms.
4. Combines state and chunk-local contributions for `dQ` and `dK`.
5. Stores `dQ`, `dK`, `dV`, `dg`, and `dbeta`, then emits the initial-state
   gradient after the first chunk.

The 384-thread CTA has these specialized roles:

- Warps 0-3: state/K/V consumers, reverse-state propagation, and `dK`/`dV`
  epilogues.
- Warps 4-7: A/Q consumers, `dQ`, `dA`, gate, and beta work.
- Warp 8: all `tcgen05` MMA issue.
- Warp 9: TMA input loads.
- Warp 10: reserved producer slot; intentionally idle in the current schedule.
- Warp 11: drains staged `dQ` tiles to global memory.

The implementation allocates 480 TMEM columns in four legal power-of-two
blocks and phase-aliases accumulator ranges only when their live intervals do
not overlap. Shared storage is checked at import time against the SM100 opt-in
limit. `get_layout_budget()` exposes the actual shared-memory, TMEM-column, and
thread budgets for tests and diagnostics.

Layout metadata is resolved by logical operation name at trace time and
flattened into explicit arguments before the `@cute.kernel` boundary. The
device schedule therefore does not depend on correlated tuple positions such
as a variant or packed-layout index. Packed 64x128 operands are validated
against their canonical physical coordinate-to-address mapping after explicit
linear regrouping.

## Compilation and caching

The token mode is marked dynamic, so one specialization can launch different
packed token counts when the remaining static contract matches. Dtypes, head
counts, logical sequence count, uniform sequence length, device capability,
and IKET instrumentation are compilation-key fields. Consequently:

- The first call for a new key includes CuTe DSL JIT compilation.
- Steady-state benchmarks must warm the key before timing.
- Each distinct logical batch size or uniform sequence length can add a cache
  entry for the process lifetime.
- Prepared launches bind tensor descriptors and the caller's current CUDA
  stream; tensors are retained until replay completes.

## CI coverage

CPU unit tests in `tests/unit_tests/ssm/test_internal_gdn_backend.py` keep a
small representative matrix for BF16 support, dtype rejection, dense packing,
the arbitrary packed-batch contract, and the named layout/storage structure.
They do not duplicate a large shape sweep.

`tests/unit_tests/ssm/test_internal_gdn_backend_e2e.py` is marked
`launch_on_gb200` and runs one explicit-`cute` full-fused E2E case (`B=2`, `T=8192`,
`H=64`, `D=128`, BF16). Deterministic Q/K/V inputs use standard deviation 0.1
to keep the long recurrence finite, and backward uses a fixed BF16 random
`grad_output` without normalization by tensor size. It compares output and all
five input gradients with FLA: output uses `atol=rtol=1e-2`, Q/K/V gradients
use `5e-2`, and gate/beta gradients use `1e-1`, matching the repository's
packed parameter-gradient tolerance. The test then records the median
CUDA-event time after two warmups and five samples. The performance guard
allows 10% noise relative to FLA. JIT time is excluded.

## Notes

- The adapter promotes gate and beta data to FP32 at the low-level boundary and
  restores their gradient dtypes afterward.
- When no final-state gradient is supplied, the adapter reuses a cached zero
  `dht` tensor of the required logical batch shape.
- Metadata is cached by `cu_seqlens` identity and tensor version; in-place
  mutation invalidates the cached entry.
- Unsupported shapes and dtypes fall back only in `auto` mode. `cute` mode is
  useful in CI and debugging because it turns accidental fallback into an
  explicit failure.
- Performance results are not hard-coded here. Reports should identify the
  commit, GPU, CUDA/CuTe DSL versions, shape, warmup count, sample count, and
whether JIT time is included.

For layout-only development, run the CPU contract test first, then compile the
SM100 probe and both uniform and packed-variable specializations on GB200.
Correctness must compare the full fused forward/backward result against FLA.
Performance comparisons for a structural refactor use at least 20 interleaved
baseline/candidate samples after warmup; cold JIT time is reported separately.
