# DeepSeek-V4.1 training on Megatron-Core: design notes

Status: milestone M0 (skeleton). Sections marked *M1* describe the planned context-parallel
implementation and are not yet code.

## 1. Model layout

DeepSeek-V4.1-Flash is 40 model layers; each is one attention sub-layer and one MoE
sub-layer, both wrapped in manifold-constrained hyper-connections (mHC) with four residual
streams. On HybridModel a model layer is the two pattern symbols `W`/`D` (attention) and
`E` (MoE):

```
WEWE DEDE...DE | DEDE...DE        # 2 window layers, 18 ratio-2 layers | 20 ratio-1 layers
```

Megatron's `layer_number` counts pattern positions (1-based). The model layer id is
`(layer_number - 1) // 2`; the V4.1 stack refuses patterns that do not alternate
attention / MoE or whose pipeline boundaries fall inside a model layer.

Configuration fields (`MLATransformerConfig`):

| Field | Released value |
| --- | --- |
| `dsv4_version` | `"v4.1"` |
| `csa_compress_ratios` (pattern positions, 0 at `E`) | `[0,0, 2×18, 1×20]` interleaved with zeros |
| `csa2_kv_source_layers` | `[2, 8, 14, 20]` |
| `csa2_index_source_layers` | `[2, 8, 14, 20, 24, 28, 32, 36]` |
| `csa2_candidate_source_layer` / `_topk_blocks` / `_block_size` | `20` / `2048` / `8` |
| `engram_layer_ids`, `engram_num_embeddings` | `[1, 14]`, `[384006168, 384016682]` |
| `engram_max_ngram_size`, `engram_bucket_size`, `engram_n_heads`, `engram_head_dim` | `4`, `16000000`, `8`, `256` |
| `engram_compressed_vocab_size`, `engram_pad_token_id` | `99092`, `2` |

## 2. CSA2 roles

`csa2/roles.py` resolves one role per model layer from the three lists above:

| Role | Condition | Computes | Publishes | Reads |
| --- | --- | --- | --- | --- |
| window | ratio 0 | sliding-window attention | nothing | nothing |
| full | in `kv_source_layers` | compressor, indexer keys, indexer, attention | compressed KV, index keys, top-k, (candidates at layer 20) | nothing |
| reindex | in `index_source_layers` only | indexer queries + top-k, attention | top-k | compressed KV, index keys, (candidates) |
| reuse | otherwise | attention | nothing | compressed KV, top-k |

Static rules enforced at configuration time: sources strictly increasing and inside the
model; a source is never a window layer; every KV source is an index source (index keys are
derived from the compressor output); a layer's ratio equals its KV source's ratio; the
candidate source is the last KV source; candidate sizes are positive exactly when a
candidate source is set.

## 3. Shared state

`DSv41SharedState` is created per forward by `DSv41HybridStack` and passed to every layer
as the keyword argument `dsv41_state` (through a small hook in `HybridStack` and in
`checkpointed_forward`). Entries are addressed by the *producer's* static id (model layer of
the KV / index / candidate source; pattern position for the mHC handoff). Consequences:

* A layer re-executed under activation recomputation reads the same entries as in the
  original forward, also with several microbatches in flight (each has its own state object
  captured in the checkpoint arguments).
* Producer tensors stay in the autograd graph; gradients from all consumers flow into the
  source layer. This is the intended memory profile: the compressed KV and index keys of the
  four source layers stay resident until backward.
* The attention module receives the state through a slot owned by its wrapper layer, set
  only for the duration of the inner-layer call.

Pipeline split: the boundary after model layer 19 keeps every source with its consumers
(sources 2, 8, 14 serve layers up to 19; source 20 serves 20 to 39). A boundary between a
source and a consumer raises at the first forward.

## 4. Single-pass hyper-connections

Parameterisation is unchanged from V4 and matches upstream `HyperConnectionModule`
(`mapping_proj.weight` = `hc_*_fn`, `bias` = `hc_*_base`, `alpha_*` = `hc_*_scale`,
Sinkhorn identical). What changes is the data flow: the projection at sub-layer `k`
yields `H_post`, `H_res` for `k` and `H_pre` for `k + 1`. `SinglePassHyperConnectionHybridLayer`
aggregates with the `H_pre` published by its predecessor (identity mix for the first
sub-layer), publishes its own, and the stack aggregates the head input with the last
sub-layer's `H_pre`; there are no separate head mixing parameters in the V4.1 checkpoint.

Across pipeline stages the `H_pre` of the last sub-layer of a stage (`[s, b, 4]`) is
appended to the residual-stream tensor (`[s, b, 4·hidden + 4]`); `schedules.py` sizes the
transfer accordingly. The handoff is carried in the activation dtype.

## 5. Engram (frozen)

`NgramHasher` maps input ids to table rows once per forward (stage 0), following the
official hashing exactly: compressed token map, per-layer odd multipliers from
`numpy.random.default_rng(10007·layer)`, XOR rolling hash over the 2- to 4-gram history,
prime bucket per (n-gram size, head), padding at sequence start. `EngramMemory` looks the
rows up in a table sharded by rows over the expert-parallel group (all-to-all of ids and
rows), projects them to one key per stream plus one value, and adds the value through the
normalised-dot-product gate. All Engram parameters are frozen in this phase.

Context parallelism (M1): each rank needs the `max_ngram - 1` tokens preceding its chunk;
this is a halo exchange of input ids, not an all-gather.

## 6. Attention module

`DSv41SelfAttention` subclasses the merged DeepSeek-V4 self-attention (projections, RoPE,
grouped output) and corrects two V4 behaviours: the rotary selection (every layer with
ratio ≥ 1 uses the compressed base (160000) with YaRN, window layers use the plain base) and
the query normalisation (V4 applies a weight-free per-head RMS norm after `linear_q_up_proj`;
the official V4.1 `Attention.forward` rotates `wq_b(q_norm(wq_a(x)))` directly, so
`query_head_rms_norm = False`; found by the alignment probe). `CSA2Attention` is
the core attention. M0 runs the framework-free reference path (`csa2/reference.py`): gather-based
sparse attention with the per-head sink over `[window keys | compressed keys]`, SBHD layout,
CP 1. The reference path is also the oracle for the fused and CP paths.

## 7. Context parallelism (M1, planned)

Follows the CSA v1 scheme in `csa_utils/cp_utils.py`: contiguous CP partition of THD rows,
raw KV rank-local with a left halo covering the window and the compressor stride, all-gather
of the *compressed* KV and index keys only (rank-major buffer + sequence-major row map),
local queries against global compressed keys, reduce-scatter in backward. V4.1 differences:

* only the four KV sources gather; reindex and reuse layers read gathered tensors from the
  shared state, so communication drops from "every layer" to four gathers per forward;
* the candidate mask is computed on global compressed positions at layer 20 and shared;
* ratio 1 means the compressed axis equals the token axis, so gathered compressed KV for the
  decoder half is one full 128K × 512 buffer per source (one source, layer 20).

## 8. M0 execution contract and known deviations

Enforced in `MLATransformerConfig._validate_dsv41` until the corresponding milestone lands:
no `recompute_granularity='full'` and no selective mHC recomputation (the shared state is
not re-published under checkpoint replay and its tensors would bypass the checkpoint
boundary; M1 makes the shared tensors explicit checkpoint inputs/outputs), no CUDA graphs,
`hidden_dropout=0`, `qk_layernorm=True`, RMSNorm, no dense mode, no indexer loss. The
indexer parameters are frozen (`csa2_indexer_frozen`) because only integer selections leave
the indexer; unfreezing needs the distillation loss. Engram layers must sit on the first
pipeline stage (the only stage with `input_ids`). Explicit attention masks are rejected:
visibility is derived from positions (one sequence per batch row in SBHD; packed sequences
arrive with THD in M1). Two rules added by the M1 alignment work: experts must be SwiGLU
(`activation_func=F.silu`, `gated_linear_unit=True`; the `TransformerConfig` default GELU
gives GeGLU experts that are numerically close enough to pass loss-curve checks), and
`csa2_candidate_topk_blocks * csa2_candidate_block_size` must cover `dsa_indexer_topk`
(with fewer candidates the official indexer back-fills its top-k with non-candidate
positions in an unspecified order; the released 2048 × 8 blocks for top-512 never hit this).
Four guards added by the 2026-09-12 design review: `activation_func_clamp_value` must be set (the released
SwiGLU clamps the gate from above and the up projection on both sides at 10; the alignment
tool sets it itself and therefore cannot catch a launch that omits it), `attention_dropout`
must be 0 (CSA2 never applies it), `attention_latent_norm_epsilon` must equal
`layernorm_epsilon` (compressor and indexer norms read the latter), and
`csa2_indexer_frozen` must stay True while there is no indexer distillation loss.

Precision rules confirmed or tightened by the same review: the head contraction and every
per-layer mHC mix run the `H_pre`-weighted stream sum in fp32 (rounding the sigmoid weights
to bf16 first cost ~3 significant digits on the logits); indexer head weights are scaled in
fp32 (`(d_i·h_i)^-0.5` is not a power of two); the reference sparse-attention path gathers
keys from an fp32 copy so the key / compressed-KV gradients are scatter-added in fp32 (the
fused cuDNN backward was never affected). Top-k ties are left to `torch.topk` as in the
official code (a stable sort would only change bit-reproducibility of tied selections
across CP or chunk sizes). The pipeline handoff still carries `H_pre` in the
activation dtype (see §5).

Alignment status (2026-09-12, `tools/dsv41/align_with_reference.py`, random-weight tiny
model, official fake quantisation disabled, PyTorch transcriptions of the two bf16
TileLang kernels): mean full-vocabulary KL 1.9e-5, top-1 agreement 97.7 %, all six
residual-stream cosines ≥ 0.9998, identical for the reference and fused attention paths.

Approved numerical deviations from the released inference code, to be revisited by the M2
alignment work: no FP8 quantisation of the window KV, no FP4 quantisation of the compressed
KV and of the indexer query/keys, and fp32 (instead of bf16-rounded) indexer scores and
probability-value products. Training runs in bf16 without these inference-side
quantisations; the alignment test in M2 quantifies the effect against the official
implementation.

## 9. Out of scope for this phase

Vision encoder, MTP / DSpark heads, Engram training, indexer distillation loss (to be
evaluated for SFT), inference paths.
