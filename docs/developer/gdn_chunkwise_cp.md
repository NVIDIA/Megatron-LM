<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Gated DeltaNet chunkwise context parallelism

Gated DeltaNet (GDN) supports two context-parallel layouts. The default
`linear_cp_mode="headwise"` exchanges sequence shards for head shards and runs
FLA over the full sequence. `linear_cp_mode="chunkwise"` retains all TP-local
heads and computes on contiguous sequence shards.

## Configuration

```python
config = TransformerConfig(
    # Supply the remaining model dimensions and parallelism settings as usual.
    linear_cp_mode="chunkwise",
    linear_cp_layout="contiguous",
    gdn_chunkwise_cp_state_mode="recurrent",  # or "parallel" (default)
)
```

These are `TransformerConfig` fields; this change does not add standalone
training CLI flags. GDN parameter shapes and checkpoint names are unchanged.
Chunkwise head divisibility depends on TP rather than TP times CP.

GPT residual streams remain in the attention layout. The GDN module converts
zigzag input to contiguous before its projection, and restores the output layout.
HybridStack continues to own its stack-level layout conversion and passes cached
packed-sequence CP metadata through TransformerLayer.

## State propagation modes

### Parallel summaries

`gdn_chunkwise_cp_state_mode="parallel"` reuses the existing GDP CP backend with
one Householder update. Local kernels build affine state summaries; the shared
CP implementation composes preceding summaries and the backward suffix summaries.
This path supports ordinary batches and packed THD sequences with global sequence
IDs. It reuses the packed-sequence metadata and differentiable convolution halo
exchange rather than exchanging all projected features across the sequence.

### Recurrent boundary states

`gdn_chunkwise_cp_state_mode="recurrent"` uses native FLA GDN kernels and passes
actual FP32 boundary states between adjacent CP ranks. Backward passes boundary
adjoints in reverse rank order. WY preparation, output formation, and the remaining
local gradient algebra are distributed; boundary recurrence has a sequential
dependency across ranks.

This mode addresses two sources of BF16 differences from headwise execution:

1. GDP single-update kernels and native GDN kernels use different intermediate
   rounding paths.
2. Composing affine summaries does not reproduce every rounding step of a
   recurrence that casts FP32 state to BF16 for matrix multiplication.

The native algorithm uses 64-token blocks. When a CP boundary cuts a block,
inputs are redistributed with a differentiable all-to-all to preserve the global
block boundaries. Neutral tokens (`q=k=v=0`, `g=beta=0`) are appended only after
the real sequence. Output is redistributed back to the original shard. Already
aligned shards avoid this redistribution. This also handles sequences shorter
than 64 times the CP size, where some temporary shards contain only padding.

## Supported configurations

- Training paths only, including full-layer activation recomputation and gated
  output-norm recomputation.
- Equal key and value head dimensions are currently required.
- Packed THD input is supported by the parallel-summary mode; recurrent mode
  rejects it explicitly.
- Chunkwise mode requires `linear_cp_layout="contiguous"` and rejects GDN2,
  pre-GDR fusion, and deterministic mode.
- The existing local-shard convolution halo requirement still applies.
- Dynamic inference and CUDA-graph support are not added by this change.

The implementation was validated with FLA 0.5.2, PyTorch 2.9.1+cu129,
Transformer Engine 2.19.0, and causal-conv1d 1.6.0 on SM90 GPUs. Numerical equality
on this matrix is not a guarantee for every FLA version, device, or input.

## Tests

Run the self-contained real-NCCL suite in an environment containing the required
CUDA dependencies:

```bash
bash tools/run_gdn_recurrent_cp_tests.sh 4
bash tools/run_gdn_recurrent_cp_tests.sh 2
```

The runner uses `--noconftest` to avoid unrelated fixture-data downloads. Coverage
includes both state modes, convolution widths 1/2/4, TP/SP regression, packed
boundaries and isolation, GPT and HybridStack integration, checkpoint replay,
all parameter gradients, and bit-exact core replay. Recurrent core tests compare
outputs and q/k/v/g/beta gradients with native full-sequence FLA. A weak-decay,
final-token-only loss verifies that gradients reach the first rank without
vanishing. Kernel replay coverage is registered in the determinism manifest.

## Local model validation

A pretrained 40-layer Qwen3.5-35B-A3B text model was tested in BF16 with
TP1/PP2/CP4/EP4, microbatch size 1, and four microbatches. With recurrent state
propagation, all 147,360 next-token argmax positions matched headwise execution
across sequence lengths 32, 64, 128, 248, 256, 264, 512, 520, 2048, and 32768.
Sampled layer outputs and MoE routing also matched. Whole-model parameter
gradients were not bitwise identical: sampled relative L2 differences were
0.873% at 2K and 0.759% at 32K, compared with headwise replay differences of
0.850% and 0.614% respectively.

Median timings from blocked runs, excluding loading, compilation and warmup:

| Measurement | Headwise | Recurrent | Speedup |
|---|---:|---:|---:|
| GDN module forward + backward, 32K, including GPT layout conversion | 13.226 ms | 9.208 ms | 1.436x |
| Full model forward + backward + gradient synchronization, 2K | 2928.483 ms | 2932.267 ms | 0.999x |
| Full model forward + backward + gradient synchronization, 32K | 5825.016 ms | 5391.698 ms | 1.080x |

The module uses eight timed samples per mode after three warmup steps per block;
the full model uses six samples after two warmup steps per block. Full-model
training uses whole-layer activation recomputation and excludes optimizer updates.
Short unaligned shards can be slower because of redistribution. These experiments
were run on the implementation based on `e4294782fb94`, before the publication
rebase; the PR records the checks rerun on its final base.
