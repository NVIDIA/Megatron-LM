# SM100 fused GDR forward

This package implements the chunked Gated Delta Rule forward path for
Blackwell SM100 with CuTe DSL. It computes the sequence output while carrying
the recurrent state across 64-token chunks and can materialize the auxiliary
tensors needed by training.

## Recommended usage

Select the internal backend through `TransformerConfig`:

```python
config = TransformerConfig(
    ...,
    gdn_gdr_backend="internal",
    gdn_gdr_recompute_h=False,
)
```

`gdn_gdr_recompute_h=False` (default) saves each chunk's input recurrent state
for the fused backward. Set it to `True` to omit that activation and recompute
it during backward. The flag applies to both the FLA forward used by `auto`
mode and the fused forward used by explicit `cute` mode; it does not affect the
standalone `gdn_gdr_backend="fla"` backend.

`MCORE_GDN_INTERNAL_BACKEND` controls dispatch inside the internal backend:

- `auto` (default) uses CuTe DSL for supported inputs and otherwise falls back
  to FLA.
- `cute` requires the CuTe DSL path and reports unsupported inputs as errors.
- `fla` bypasses the CuTe DSL path.

The production call path is:

```text
implementation._cutedsl_forward
  -> chunk_gated_delta_rule_prefill_cute
  -> cutedsl_fused_chunk_gdn_fwd_sm100
  -> GatedDeltaNetChunkedKernel
```

Model code should use the configured GDN layer rather than invoking the
launcher directly.

## Supported wrapper contract

`chunk_gated_delta_rule_prefill_cute` accepts packed THD tensors:

- CUDA 13 or newer on an SM100 GPU.
- Contiguous `q`, `k`, and `v` with shape `[total_tokens, heads, 128]`.
- `q`, `k`, and `v` must share the same FP16 or BF16 dtype.
- `g` and `beta` have shape `[total_tokens, output_heads]` and are converted
  to contiguous FP32 tensors before launch.
- `cu_seqlens` is required; every sequence length must be a multiple of the
  64-token chunk size.
- In-kernel QK L2 normalization and non-empty initial state are not supported
  by the public Megatron wrapper.

The wrapper returns the output tensor, or `(output, final_state)` when
`output_final_state=True`. It also accepts preallocated output, `A`, final
state, per-chunk input state `h`, and checkpoint buffers for the training
path.
When `output_h` is provided, it stores every 64-token chunk's input state and
therefore requires `checkpoint_every_n_tokens=64`. The BF16 TMA kernel path
initializes the first saved state of each packed sequence in-kernel; non-BF16
fallback paths keep the wrapper-side initialization.


## Algorithm

For each 64-token chunk, the kernel keeps the recurrent state `S` in TMEM and
performs seven tensor-core GEMM stages:

1. `K @ K^T` builds lower-triangular intra-chunk scores.
2. `Q @ K^T` builds output scores.
3. `K @ S` applies the previous state to the keys.
4. `Q @ S` computes the inter-chunk output contribution.
5. `A_inverse @ V` produces corrected value vectors.
6. The scaled QK scores multiply the corrected values for the intra-chunk
   output.
7. `K^T @ delta` produces the recurrent-state update.

The epilogue combines the intra- and inter-chunk output terms and advances the
state using the final cumulative gate value for the chunk. When requested, the
kernel also stores `A`, corrected values, `W`, chunk states, final state, or
periodic state checkpoints.

## SM100 execution structure

The persistent kernel launches 12 warps with specialized roles:

- Warps 0-3 compute gate transfer factors, score epilogues, and the
  hierarchical triangular inverse.
- Warps 4-7 compute state/value corrections, state updates, and output
  epilogues.
- Warp 8 issues the seven `tcgen05` MMA stages.
- Warps 9-10 perform TMA loads for Q/K/V and gate/beta tensors.
- Warp 11 stores the output epilogue.

The implementation uses approximately 225.5 KB of shared memory for staged
Q/K/V, intermediate matrices, output staging, and gate data. TMEM holds the
FP32 recurrent state and MMA accumulators. K is double-buffered so the next
chunk can be prefetched while the current chunk is processed.

## Notes

- The token dimension is dynamic, while head counts, dtypes, and optional
  output features are part of the CuTe DSL compilation-cache key.
- Dense BTHD inputs are flattened to packed THD layout by the Megatron adapter.
- The adapter supplies chunk-local cumulative log2 gates to the kernel. With
  `gdn_gdr_recompute_h=False`, it requests both `A` and per-chunk input state
  `h` and passes them directly to the fused backward kernel. With the flag set,
  it requests only `A` and drops `h` after producing the forward output.
- Native partial chunk tails are not enabled by the public wrapper; supported
  sequence lengths are 64-token aligned.
- This document does not record performance numbers. Benchmark results should
  identify the exact commit, GPU, CUDA/CuTe DSL version, shape, warmup, and
  iteration count.
