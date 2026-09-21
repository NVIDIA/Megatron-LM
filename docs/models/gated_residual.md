# Gated Residual (Qwen4-Exp hyper-connection variant)

`mhc_connection_variant="gated_residual"` selects the 4-stream residual
mechanism of Qwen4-Exp (HF `model_type: qwen4_exp`) as an alternative to the
default mHC (Manifold-Constrained Hyper-Connections) mathematics, reusing the
entire mHC engineering pipeline: per-sublayer hook points, block-level
expand/contract, MTP interop, selective recompute, and SP gradient marking.

## Mathematics

Each sublayer (attention and MLP each) owns one `GatedResidualModule` with
`n = mhc_num_residual_streams`, `C = hidden_size`, `r = hc_lowrank`:

```
x̃      = GroupedRMSNorm(x, group_size=C)          # per-stream RMSNorm, zero-centered gamma
u      = silu(W_down @ x̃ / n)                     # [.., r]
g_read = sigmoid(W_up @ u)                         # [.., n*C] per-channel read gate
mixed  = mean_n(g_read ⊙ x̃)                       # [.., C]   -> sublayer input
g_write= 2 sigmoid(W_inj @ x̃ / n)                 # [.., n]   per-stream write gate
x_out  = x + flatten(g_write ⊗ (F(mixed) + bias))  # identity residual, no cross-stream mixing
```

The block exit contract (and the MTP head's) is the same module with
`use_combine=False` — only the n→1 `mixed` contraction, no write gate. It is
also the **last normalization**: HF `Qwen4ExpTextModel` feeds the mixer output
straight into `lm_head`, and the released checkpoint has no `model.norm` /
`mtp.norm` tensor, so the gated-residual variant builds no `final_layernorm`
/ `final_norm` (GPT `TransformerBlock`, `HybridStack`, and MTP alike).

Differences from mHC: per-channel low-rank read gate instead of a per-stream
scalar; `mean` aggregation instead of `sum`; identity residual instead of the
Sinkhorn doubly-stochastic mixing matrix (`h_res` in the 4-tuple API is always
`None`); an explicit learnable group norm; **no pre-sublayer norms** — the
layer specs set `input_layernorm`/`pre_mlp_layernorm` to `IdentityOp` and use
unfused QKV/`in_proj`/`linear_fc1` projections.

## Configuration

| Knob | Meaning |
|---|---|
| `--enable-mhc-connections` | master switch (shared with mHC) |
| `--mhc-connection-variant gated_residual` | select this variant (default `mhc`) |
| `--mhc-num-residual-streams 4` | n |
| `--hc-lowrank 320` | read-gate bottleneck r |
| `--recompute-granularity selective --recompute-modules mhc` | GR recompute (shared knob with mHC) |

The layer-spec builders take `mhc_connection_variant` as an argument; it must
match `config.mhc_connection_variant` (TransformerBlock verifies this at
build time). On the hybrid path, an explicitly provided `--spec` overrides the
auto-selected norm-free spec — the builder warns, since a spec that keeps the
fused input layernorms would normalize the streams twice.

Constraints enforced at config validation: requires
`transformer_impl='transformer_engine'`; CUDA graphs are rejected (the
partial-MoE capture paths pack the 4-tuple's `h_res` slot as a graph output,
which this variant returns as `None` — lifting this is a separate fused-kernel
workstream); `use_fused_mhc` and `mhc_recompute_attn_cuda_graph_split` are
mHC-only; TP>1 without `sequence_parallel` warns (the replicated GR compute,
~10% of forward FLOPs at real scale, would be duplicated per TP rank).

## dtype policy

| Tensor | dtype |
|---|---|
| `hc_norm.weight`, normalization math | fp32 (`mark_keep_in_fp32`) |
| `input_mix_weight_down/up` weights + GEMMs | params dtype (bf16) |
| silu / sigmoid / mean aggregation | fp32 |
| `block_inject_weight` weight + GEMM | fp32 (`mark_keep_in_fp32`) |
| residual streams `[s, b, n*C]` | activation dtype, never quantized |

The two low-rank GEMMs stay out of the TE fp8 path (plain `nn.Linear`) — one
dimension is only `r` wide, and their outputs generate per-channel gates whose
quantization noise would propagate into the whole residual path.

The compute dtype of the gate GEMMs is the gate *weight* dtype (the params
dtype), not the activation dtype, so `fp32_residual_connection=True` is
supported: fp32 residual streams are normed in fp32, the gates are computed in
the params dtype (feeding the bf16 sublayer GEMMs), and the write-back
preserves the fp32 stream.

mxfp8: `mixed` feeds the first mxfp8 GEMM with **no pre-norm** (faithful to
the reference; adding one would break pretrain parity). Measured on a
hidden-2560 proxy after short training, per-32-element-block amax statistics
of `mixed` are benign — cross-block amax max/median ≤ 2.8, within-block
amax/median(|x|) p99 ≤ 7 — well inside e4m3 range with per-block e8m0 scales,
so no per-layer bf16 fallback is currently wired. If a trained model shows bad
layers, the fallback design is a per-layer first-GEMM bf16 override.

## Parallelism

- **TP**: GR weights are replicated (non-TP-aware `nn.Linear`). With SP, every
  parameter carries `sequence_parallel=True` so gradients all-reduce over TP.
- **PP**: the inter-stage tensor is the n-stream `[s, b, n*C]` (mHC-shared
  `hc_mult` machinery); p2p volume is ×n.
- **CP**: needs no GR-side code — GR is per-token, each CP rank processes its
  sequence chunks independently, and the replicated parameters' gradients are
  summed by the regular DDP reduction over the DP×CP group. Verified by a CP2
  gradient-parity unit test and a CP1-vs-CP2 loss-consistency run.
- **MTP**: the decoder saves the pre-contraction streams (`mhc_multistream`),
  MTP consumes them and applies its own GR exit contract; the nested MTP stack
  skips `input_expand` and builds no exit-contract params (`not is_mtp_layer`),
  avoiding orphaned parameters under DDP.
- **Recompute**: with `recompute_modules=["mhc"]`, the whole gate computation
  is one `CheckpointWithoutOutput` regenerating `(mixed, g_write)` from the
  residual (which is alive anyway); the write-back is checkpointed like mHC's.

## HF ↔ Megatron weight mapping (specification for the whole-model converter)

Per-sublayer modules (HF `model.layers.<L>.attn_hyper_connection` /
`.mlp_hyper_connection` → Megatron `decoder.layers.<L>.self_attention_hyper_connection`
/ `.mlp_hyper_connection`; hybrid path: `decoder.layers.<i>.hyper_connection`
of the wrapper that owns the sublayer):

| HF tensor | Megatron tensor | Shape | Notes |
|---|---|---|---|
| `hc_norm.weight` | `hc_norm.weight` | `[n*C]` | **zero-centered gamma both sides** (stored value is γ−1, applied as `1+w`); copy verbatim, no ±1 offset needed as long as `layernorm_zero_centered_gamma=True` |
| `input_mix_weight_down.weight` | `input_mix_weight_down.weight` | `[r, n*C]` | verbatim |
| `input_mix_weight_up.weight` | `input_mix_weight_up.weight` | `[n*C, r]` | verbatim |
| `block_inject_weight.weight` | `block_inject_weight.weight` | `[n, n*C]` | verbatim; absent on exit-contract instances |

Exit contracts: HF `model.hyper_connection_mixer.*` → Megatron
`decoder.hc_exit_contract.*` (GPT/hybrid decoder) and MTP's
`mtp.layers.<d>.hc_exit_contract.*`. All GR tensors are TP-replicated, so the
converter never splits them; dist-checkpoint reshard across TP/PP changes is
handled by the standard replicated-tensor path (verified by training TP1-saved
checkpoints under TP2/PP2/TP2PP2).

Layout caveats for the *surrounding* model (not GR tensors, but consequences
of the norm-free specs): HF fuses nothing into QKV/`in_proj`/`fc1`, and
neither do the GR layer specs — there are **no `*.layer_norm_weight` keys** in
GR checkpoints, so converter rules written for mHC/fused-LN layouts must not
be applied. Qwen's fused `key_proj (C -> n*C)`-style per-stream fusions (PLE
side) are out of GR's scope.

## Tests

- `tests/unit_tests/transformer/test_gated_residual.py` — module parity vs the
  ported HF oracle (fwd+bwd), write-back formula, recompute-path equivalence,
  dtype marking, config validation.
- `tests/unit_tests/transformer/test_gated_residual_wiring.py` — norm-free
  layer shape, end-to-end 4-stream pipeline arithmetic on a TransformerBlock,
  hybrid spec/dispatch.
- `tests/unit_tests/transformer/test_gated_residual_distributed.py` — TP2+SP
  gradient all-reduce parity; GR+MTP end-to-end on GPTModel (TP1/TP2).
