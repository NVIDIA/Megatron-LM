# GDN Elementwise Fusion

Gated DeltaNet (GDN) has two independent, opt-in `TransformerConfig` options:

| Option | Fused operations | Default |
| --- | --- | --- |
| `gdn_pre_gated_delta_rule_fusion` | Causal convolution, SiLU, layout transforms, Q/K L2 normalization, head expansion, beta and decay preparation | `False` |
| `gdn_gated_output_norm_fusion` | Output RMSNorm and SiLU gating after the gated delta rule | `False` |

Both options support the `gdn` attention variant, including the deprecated
`gated_delta_net` alias. They can be enabled independently or together. The
linear-attention recurrence and output projection retain their existing
implementations. The post-GDR implementation is adapted from Layali Rashid's
[output-gating fusion in PR #7368](https://github.com/NVIDIA/Megatron-LM/pull/7368).
Its post-GDR kernels operate on local tensor shapes and strides, including
multi-batch projection views and heads redistributed by context parallelism.

For example, configure a supported BF16 GDN model with:

```python
config = TransformerConfig(
    # ... model dimensions and other training options ...
    experimental_attention_variant="gdn",
    gdn_pre_gated_delta_rule_fusion=True,
    gdn_gated_output_norm_fusion=True,
)
```

## Post-GDR requirements

The post-GDR fusion checks its requirements on **every forward**, including
selective output-norm recomputation. When enabled, unsupported inputs raise
`ValueError`; there is no silent fallback. Disable the option to use the
existing unfused path.

The supported configuration requires:

- `deterministic_mode=False`;
- SiLU/Swish activation and an `RMSNorm` output normalization module;
- nonempty CUDA BF16 or FP16 recurrence output;
- matching output and gate shapes `[batch, sequence_length, local_heads, head_dim]`
  with a power-of-two head dimension;
- a gate in the activation dtype or FP32 and a contiguous BF16, FP16 or FP32
  RMSNorm weight of length `head_dim`, all on the same CUDA device.

The gate may be a strided view into the input projection. Fusion consumes
that view directly, including separate batch and sequence strides; it does
not materialize a contiguous copy. Strided recurrence outputs are supported
as well. Outputs and returned input gradients are contiguous in logical order.
Context parallelism retains the existing communication before and after this
local operation, and the existing GDN/TP/CP shape constraints still apply. The checks
inspect tensor metadata without synchronizing CUDA. Post-GDR fusion can
process packed sequences because normalization and gating operate per token;
the existing GDN packed-sequence checks still apply.

`MCORE_GDN_FUSION` is not used. Select the feature through
`TransformerConfig.gdn_gated_output_norm_fusion` (or the corresponding
`--gdn-gated-output-norm-fusion` training argument).

## Execution and numerical behavior

The training `forward` runs pre-GDR preprocessing, the recurrence, gated output
normalization, and the output projection in `_forward_compute`. Inference keeps
its existing dispatch. Selective `gdn_norm_out` recomputation retains its existing
checkpoint lifecycle.

The post-GDR kernels preserve the activation-dtype RMSNorm materialization boundary
before the FP32 SiLU-gating multiply. They support first-order autograd only.
Floating-point operation ordering can differ from the unfused path;
correctness tests do not establish bitwise equivalence or training
convergence. Enabling either fusion is incompatible with deterministic mode.
