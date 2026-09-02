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
path, so the flag has no effect there.

### Context parallelism

`GatedDeltaNet.forward` preserves MCore's established chunkwise-CP contract:
dense SBHD input with `B > 1` is rejected. Use packed THD input, or use dense
input with `B = 1`. Packed input carries validated global CPU sequence offsets
through the CP context; the internal adapter converts those offsets to the
rank-local metadata consumed by the fused backward.

For context parallelism, the internal backend keeps the established FLA local
forward path; CP boundary preprocessing may use FLA or the in-tree CuTe DSL
kernels. During backward it first runs the CP preprocessing sequence:
recompute `w`/`u`, reconstruct the local recurrent state when needed, compute
the local `dv`, and run the CP AllGather/merge preprocessing to produce the
rank-local boundary gradient `dht`. It then passes that `dht` and the saved or
recomputed local state to `fused_gdr_bwd`. The collective communication remains
outside the CuTe DSL kernel.

This is a correctness-first integration and intentionally duplicates local
`dv` and recurrent-state work that the fused kernel performs again. CP inputs
must satisfy the fused backward contract (BF16, 64 heads, head dimension 128);
`auto` falls back to the CP-aware FLA backward when they do not, while `cute`
reports the unsupported contract explicitly. CP4 E2E validation must verify
that the selected CP preprocessing kernels and fused backward are invoked. The
GB200 E2E coverage exercises packed THD CP4 with two logical sequences and
compares output plus all five GDR input gradients against FLA. The non-CP case
asserts that both CuTe DSL forward and backward kernels run.

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
- `h`: BF16 `[1, C, 64, 128, 128]`, where
  `C = sum(ceil(sequence_length / 64))`.
- `dht`: FP32 `[B, 64, 128, 128]`.
- `cu_seqlens`: contiguous CUDA int32 `[B + 1]`, starts at zero, ends at `N`,
  and describes `B` positive-length logical sequences. Sequence lengths do
  not need to be multiples of 64.
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

For a final partial chunk, the kernel computes `valid_tokens` from the logical
sequence boundary. Invalid rows are neutralized in shared memory before use,
the cumulative gate is extended from the last valid row, and every token
gradient store is predicated by `valid_tokens`. No padded token tensor is
allocated and no input is copied. The aligned specialization retains the
existing unmasked TMA load/store path.

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
`launch_on_gb200`. It calls the internal GDR backend directly for non-CP and CP4
with `B=2`, `T=8192`, `H=64`, `D=128`, and BF16. The CP4 parameter manually
builds a single-sequence CP context; it validates FLA CP forward plus fused CuTe
backward, but it does not exercise dense `GatedDeltaNet.forward`. The test
compares output and all five input gradients with FLA and records CUDA-event
timing after two warmups and five samples, with a 10% regression allowance.
JIT time is excluded.

The production packed-THD CP4 aligned/tail validation is recorded in
`fused_gdr_cp_cute/kernel.md`; it is separate from the checked-in CI test.

## Notes

- The adapter promotes gate and beta data to FP32 at the low-level boundary and
  restores their gradient dtypes afterward.
- When no final-state gradient is supplied, the adapter reuses a stream-scoped
  zero-`dht` tensor. This LRU cache is bounded to 256 MiB.
- Uniform dense-batch offsets are cached by device, batch size, and local
  sequence length. Packed variable-length metadata retains the identity/version
  cache described below.
- Metadata is cached by `cu_seqlens` identity and tensor version; in-place
  mutation invalidates the cached entry. Standard packed-batch construction
  supplies a trusted CPU mirror once, which GatedDeltaNet validates and reuses
  across layers. Q/KV validation and equality checks therefore do not observe
  CUDA results in Python. Chunk offsets are generated lazily only by the
  internal path and reuse the identity/version cache. Device-only offsets are
  not treated as validated metadata; direct calls without a CPU mirror fall
  back in `auto` mode and fail in `cute` mode.
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
