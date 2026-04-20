# Attention Residuals

Attention Residuals replace fixed residual accumulation with a learned
depth-wise attention over residual-producing hidden states. This implementation
adds Full AttnRes and Block AttnRes to Megatron Core transformer blocks.

The implementation follows the paper's convention that self-attention and MLP
are separate residual-producing sublayers. For each transformer layer, AttnRes
is applied before self-attention and before MLP, then the corresponding sublayer
output is added to the AttnRes state. A final AttnRes aggregation is applied at
the transformer block output before the final layer norm.

## Variants

Full AttnRes attends over all preceding residual-producing values:

```bash
--attention-residuals \
--attention-residual-type full
```

Block AttnRes groups sublayer outputs into depth blocks and attends over
completed block summaries plus the current partial block:

```bash
--attention-residuals \
--attention-residual-type block \
--attention-residual-num-blocks 8
```

For a decoder stack with `L` transformer layers, there are `2L` residual
sublayers. With 16 transformer layers and 8 AttnRes blocks, each block contains
4 residual sublayers.

## Implementations

The depth-attention scorer supports four implementation modes:

```bash
--attention-residual-implementation torch
--attention-residual-implementation checkpointed
--attention-residual-implementation triton
--attention-residual-implementation triton_bwd
```

`torch` is the eager PyTorch reference. It is useful for correctness checks but
saves more forward intermediates.

`checkpointed` uses custom autograd and recomputes AttnRes internals in
backward. This reduces saved forward intermediates and is a good memory-saving
reference.

`triton` uses Triton kernels for the forward reduction and accumulation path,
with checkpointed PyTorch recomputation in backward.

`triton_bwd` uses Triton kernels for both forward and backward recomputation.
This is the recommended implementation for performance experiments when Triton
is available.

## RMSNorm

By default, AttnRes applies RMSNorm to keys before depth-wise scoring:

```bash
--attention-residuals
```

Disable it with:

```bash
--no-attention-residual-rmsnorm
```

## Diagnostic Logging

For short smoke tests, query gradient diagnostics can be enabled with:

```bash
--attention-residual-log-weights
```

This prints gradient norms for the AttnRes query parameters. It introduces GPU
synchronization overhead and should stay disabled for timing comparisons.

## Current Scope

Supported and exercised:

- Decoder-only dense GPT/Llama-style transformer layers.
- Tensor parallelism and sequence parallelism.
- Transformer Engine FP8 training path in the provided examples.
- Full and Block AttnRes with `torch`, `checkpointed`, `triton`, and
  `triton_bwd` implementations.

Not yet supported or not yet validated:

- MoE MLP layers.
- Cross-attention layers.
- Pipeline parallel configurations greater than one stage.
- Full-layer activation recomputation with AttnRes.
- Inference/KV-cache paths.

## Example

The example scripts under `examples/attention_residuals/` show how to launch
baseline, Full AttnRes, and Block AttnRes runs with the same model/data
configuration.
