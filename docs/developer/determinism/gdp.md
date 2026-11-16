# Batch-invariant gated delta product

GDP normally uses different chunked training and recurrent decoding arithmetic.
Floating-point rounding then depends on where prefill stops and decode begins.
The opt-in backend below gives training, prefill, and decode a shared BF16
arithmetic contract, including GDN with one update per token.

## Configuration

Enable `batch_invariant_mode` and initialize the selected batch-invariant runtime
before constructing the model, as for the existing attention and projection
backends. Set `gdp_batch_invariant_block_size` to 16 (default) or 32. Blocks count
**internal updates**, not tokens. The block size is an immutable numerical
setting: training and serving must agree. C32 gives better long-sequence
training throughput in the measured GB200 configuration; C16 has lower decode
latency at the measured small batch sizes.

The GDP path supports BF16, convolution width four, CP=1, and no sequence
parallelism. CuTeDSL GDP, FP8/FP4, selective GDP recompute/offload, prefix caching,
and speculative buffers are rejected. Whole-layer checkpointing is supported.
Packed training currently pads sequences to a common length; padding overhead
has not been optimized.

## Arithmetic and cache

With state orientation `[K,V]`, an update is
`Sbar = exp(g) * S; u = beta * (v - k @ Sbar); S = Sbar + outer(k, u)`.
For GDP, apply the token's decay on its first update and read its query after
its last update.

Within a fixed block, construct the strictly lower triangular matrix
`A[i,j] = -beta[i] * exp(p[i] - p[j]) * dot(k[i], k[j])`, where `p` is a sequential
FP32 gate prefix. Build rounded inverse-factor rows in chronological order with
an explicit adjacent-pair addition tree. Project from BF16 boundary state, pin
the FP32 RHS multiply/subtract/multiply instructions, and explicitly round the
factors, RHS, residuals, attention weights, and output to BF16. Only completed
blocks update the FP32 boundary state. A generic reduction or disabling compiler
FMA fusion alone is insufficient to preserve the required rounding.

The inference allocator owns six cache fields: boundary state, keys, gate
prefix inputs, inverse factors, RHS, and cursor. Save/restore and slot movement
must preserve all six. Partial blocks remain in the cache across calls. The
training API's returned boundary state alone cannot resume a partial block.
Slot IDs must be unique and in range; `-1` denotes inactive incremental lanes.

The backward differentiates the saved rounded computation with straight-through
cast derivatives. It uses a reverse state scan and blockwise triangular adjoint
solve. For incoming factor gradient `dT`, compute
`Z = inverse(I - transpose(A)) @ dT`, then
`dA = Z @ transpose(T_saved)` on the strict lower triangle. Merging small
triangular blocks avoids the cancellation of full-matrix power doubling on
correlated keys. Compensated BF16 products preserve FP32 operand residuals.
Only first-order gradients are supported.

Output and gradient kernels skip queries that internal GDP updates do not emit.
Small-batch forward partitions disjoint value columns, with one writer for shared
cache fields. These layout choices preserve the same forward arithmetic.

This contract is not byte-compatible with native FLA/CuTeDSL arithmetic. Both
training and serving must use it. It establishes GDP parity, not an automatic
guarantee for every surrounding model operation or downstream logprob normalizer.

## Validation

From the Megatron-LM root, with its GPU dependencies installed:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest -q \
  tests/unit_tests/ssm/test_gdp_batch_invariant_recurrence.py \
  tests/unit_tests/ssm/test_gdp_batch_invariant.py \
  tests/unit_tests/determinism/kernels/test_gdp_batch_invariant.py
```

The tests cover independent gradient oracles, correlated keys, multiple value
tiles, exact uneven chunking through 8193 tokens, batch isolation, packed and
static inference, allocator cache replay, and small hybrid-model logits/raw
logprobs.

Performance measurements should use prepared recurrence inputs, report forward,
forward-plus-backward and decode separately, and distinguish CUDA-graph replay
from eager launch overhead. Record the kernel source hash and dependency versions.
Native FLA output is not an accuracy oracle: compare approximation accuracy with
an independent FP64 recurrence and check native repeat differences separately.

## Reproducing performance measurements

The benchmark uses prepared BF16 recurrence inputs and CUDA graph replay, with
native FLA, C16, and C32 on the same GPU. It records the recurrence source hash,
software versions, training forward/backward, parallel prefill, incremental prompt
append, and single-token decode latency. It does not measure projection, attention,
or full-model throughput.

```bash
uv run python tools/benchmark_gdp_batch_invariant.py --updates 1 3 --output gdp-benchmark.json
```

Fresh static prefill writes parallel-prepared results directly into allocator-owned
cache storage. Continuation and dynamic packed prefill use the incremental path,
which retains partial blocks. Prefill does not allocate or write the RHS tape that
is needed only by backward.

## Measured performance

GB200, PyTorch 2.11.0+cu130, Triton 3.6.0; CUDA graph replay. Prepared recurrence
inputs have 4096 tokens, 24 heads, K=128, V=64 and three internal updates per token.
All entries below are forward plus backward in milliseconds.

| Batch | Native FLA | Batch-invariant C16 | Batch-invariant C32 |
| --- | ---: | ---: | ---: |
| 1 | 2.18 | 4.59 | 3.19 |
| 8 | 11.24 | 17.17 | 12.72 |

The shared convolution backward uses deterministic tiled partial sums instead of
materializing full token-by-channel gathers and products for each tap. A paired
complete-mixer benchmark (hidden size 1536, the same 24-head geometry, C32) measures
the effect of that change, including projections, normalization and recurrence:

| Batch | Forward + backward before (ms) | After (ms) | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 8.35 | 5.89 | 29% |
| 8 | 60.74 | 38.52 | 37% |

These are component timings, not whole-model or distributed-training throughput.
The optimized arithmetic still has a cost relative to native FLA. Inference
convolution skips the FP32 preactivation buffer used by backward; convolution
parameter-gradient scratch scales with the number of 256-row tiles.
