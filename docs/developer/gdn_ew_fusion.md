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
Its Triton kernels and autograd implementation are unchanged.

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

- `deterministic_mode=False` and context-parallel size 1;
- SiLU/Swish activation and an `RMSNorm` output normalization module;
- nonempty, contiguous CUDA BF16 recurrence output with head dimension 128;
- gate shape `[1, sequence_length, 16, 128]` with contiguous elements within
  each token (last two strides 128 and 1);
- matching output/gate element counts, a contiguous 128-element norm weight,
  and all tensors on the same CUDA device.

The gate may be a strided view into the input projection. Fusion consumes
that view directly; it does not materialize a contiguous copy. The checks
inspect tensor metadata without synchronizing CUDA. Post-GDR fusion can
process packed sequences because normalization and gating operate per token;
the existing GDN packed-sequence checks still apply.

`MCORE_GDN_FUSION` is not used. Select the feature through
`TransformerConfig.gdn_gated_output_norm_fusion` (or the corresponding
`--gdn-gated-output-norm-fusion` training argument).

## Execution and numerical behavior

Pre-GDR fusion runs inside `forward_pre_attn_and_core_attn`, which returns
the normalized recurrence result. `forward_post_core_attn` applies the
output projection. The ordinary `forward` preserves inference dispatch and
uses these stages for training. Selective `gdn_norm_out` recomputation keeps
its existing checkpoint lifecycle.

The post-GDR kernels preserve the BF16 RMSNorm materialization boundary
before the FP32 SiLU-gating multiply. They support first-order autograd only.
Floating-point operation ordering can differ from the unfused path;
correctness tests do not establish bitwise equivalence or training
convergence. Enabling either fusion is incompatible with deterministic mode.
