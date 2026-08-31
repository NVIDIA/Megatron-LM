---
orphan: true
---

<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Experimental MLA with Latent Context Parallelism

## Status

This document proposes a deliberately narrow first implementation. It is not a compatibility plan
for the existing MLA attention path. The attention algorithm and all backend code remain isolated
in the feature-owned module. That module also owns its local attention spec, non-mutating GPT and
Hybrid spec rewrites, and one explicit backend-plan preprocessing entry point. GPTModel and
HybridModel each make one conditional spec-initialization call; ordinary `TransformerBlock` and
`HybridStack` each make one guarded preprocessing call before their layer loop. Generic dynamic-CP
metadata consistency belongs to the data scheduler rather than the attention feature. Transformer
Engine (TE) and all existing MLA tests remain unchanged. One shared MLA file has a deliberately
small compatibility change: it honors the existing per-layer `no_rope_freq` flag by not
constructing or applying a rotary embedding for that layer. The latent-CP module owns all CP,
backend, merge, and transport logic.

P2P context parallelism (CP) circulates normalized MLA latent KV and the positional key component,
rather than expanded K and V. Each receiving CP rank reconstructs K/V immediately before its
attention phase. Expanded K/V remain temporary: backward recomputes the latent-KV up projection,
while the cuDNN path retains local O/LSE and calls its backward graph without replaying SDPA.

## Goals and initial scope

The first version will:

- support training-only MLA self-attention in THD format;
- support static and per-microbatch dynamic P2P CP with the zigzag partition, including the exact
  no-ring CP=1 degeneration;
- require sequence parallelism whenever tensor parallelism is greater than one;
- support the unfused `rope` and `yarn` rotary modes and Kimi-style no-RoPE layers selected by the
  existing `no_rope_freq` flag;
- preserve the latest MLA output gate in both `elementwise` and `headwise` modes;
- select a direct attention adapter with the existing
  `TransformerConfig.attention_backend`: `AttnBackend.fused` selects the cuDNN Frontend Graph API
  and `AttnBackend.flash` selects FlashAttention-4 (FA4);
- communicate latent KV plus the shared raw-or-rotated positional key, never head-expanded K/V;
- preserve existing MLA projection parameter names and the output contract;
- use only public PyTorch, cuDNN Frontend, FA4, and MCore APIs at the attention boundary; and
- expose narrow layout and transport interfaces for later contiguous-to-zigzag and A2A+P2P work.

The first version intentionally does not support inference/KV cache, arbitrary masks, dropout,
FP16, FP8/FP4, CUDA graphs, padded THD storage, fused MLA RoPE, GQA, outer layer
recompute, fine-grained activation offload, CPU offloading, frozen output-projection weights, or
the absorbed-MLA algorithm. It does not change standard
attention, the legacy MLA path, TE, or TE's CP implementation.

Before merge, MTP, FP8/FP4 (including MXFP8), outer/selective recompute, and fine-grained
activation offload remain explicit implementation TODOs. Performance recipes must disable those
features rather than relying on an implicit fallback. CUDA graph execution is simply outside the
v1 contract and is disabled without a separate pre-merge TODO.

## Existing code reused, and the dependency boundary

The new module subclasses `MLASelfAttention` from
`megatron/core/transformer/multi_latent_attention.py` only to reuse projection-module construction,
the optional rotary embedding module, the output-gate projection/application helper, checkpoint
names, and sharded-state-dict behavior. It preserves the caller's complete supported projection
stack and parameter objects. For the local MCore stack it does not call
`RowParallelLinear.forward`, whose current implementation passes a null TP group to its linear
primitive; for the TE stack it calls the preserved public `TERowParallelLinear` module exactly as
the old path does. It overrides `forward`, stops KV projection before `linear_kv_up_proj`, and never
calls inherited `get_query_key_value_tensors` or `_run_core_attention`, because those functions
construct full K/V before the current CP wrapper.

The new module uses a short, feature-local `_latent_cp_down_projection` helper and, only for layers
that enable RoPE, the public MCore `apply_rotary_pos_emb` export. It must not call the inherited
`_qkv_down_projection`: that protected helper gathers Q with an omitted group and can therefore
resolve the global tensor-parallel group. The feature helper calls the preserved projection
*modules*, not an inherited projection helper. It distinguishes feature-sharded local down outputs
from duplicated TE down outputs and performs only the TP redistribution required by that stack.
The rotated shared positional key is returned to the same per-TP sequence lane. The local
`_explicit_output_projection` fallback uses the public MCore linear/reduction exports with the
injected TP group; the TE profile uses its original public TE module. No private TE function is
called.

The construction helper replaces the base MLA `core_attention` submodule with `IdentityOp`.
`MLASelfAttention.__init__` therefore retains projection construction and state-dict keys without
constructing TE `DotProductAttention`; the overridden forward never executes the identity module.

The following existing MCore surfaces are used because they are exported classes/functions or form
the current module-spec/config contract:

| Dependency | Source | Reason for use |
| --- | --- | --- |
| `MLASelfAttention`, `MLASelfAttentionSubmodules` | `megatron/core/transformer/multi_latent_attention.py` | Projection and checkpoint compatibility; no legacy attention execution. |
| `TELinear`, `TELayerNormColumnParallelLinear`, `TEColumnParallelLinear`, `TERowParallelLinear` | `megatron/core/extensions/transformer_engine.py` | Public old-path projection stack preserved by model-spec conversion; no TE attention wrapper is used. |
| `AttnBackend` | `megatron/core/transformer/enums.py` | Existing backend selection contract. |
| `PackedSeqParams` | `megatron/core/packed_seq_params.py` | Existing THD metadata contract. |
| `ProcessGroupCollection` | `megatron/core/process_groups_config.py` | Explicit CP/TP group injection. |
| `ModuleSpec`, `IdentityOp` | `megatron/core/transformer/spec_utils.py`, `megatron/core/transformer/identity_op.py` | Explicit opt-in construction without a legacy attention wrapper. |
| `apply_rotary_pos_emb` | `megatron/core/models/common/embeddings/__init__.py` | Existing differentiable packed MLA RoPE behavior. |
| `gather_from_tensor_model_parallel_region` | `megatron/core/tensor_parallel/mappings.py` | Public autograd-aware Q/KV output gather, always called with the injected TP group. |
| `scatter_to_sequence_parallel_region` | `megatron/core/tensor_parallel/mappings.py` | Public first-dimension scatter for complete Q/KV latents and the raw-or-rotated K positional branch, always called with the injected TP group. |
| `gather_from_sequence_parallel_region` | `megatron/core/tensor_parallel/mappings.py` | Public first-dimension gather for a received K positional shard; its backward reduce-scatter sums shared-key gradients across TP heads. |
| `linear_with_grad_accumulation_and_async_allreduce` | `megatron/core/tensor_parallel/__init__.py`, `layers.py` | Local-stack output fallback; preserves weight-gradient/`main_grad` semantics and receives the injected TP group. |
| `reduce_scatter_to_sequence_parallel_region`, `reduce_from_tensor_model_parallel_region` | `megatron/core/tensor_parallel/__init__.py`, `mappings.py` | Local-stack row-parallel output reductions, always called with the injected TP group. |
| `apply_module` | `megatron/core/typed_torch.py` | Public typed-module invocation used for the preserved TE output module. |
| `CpPartitionModeConverter` and related exports | `megatron/core/context_parallel_layout/__init__.py` | Future layout adapter only; v1 validates an already-zigzag input. |

No new code imports or reads a process group from `parallel_state`. The caller injects a populated
`ProcessGroupCollection`; rank, size, and peer global ranks are derived from its `tp` group and the
effective CP group with public `torch.distributed` APIs and an explicit group argument. The base constructor
receives the same collection and constructs every accepted local or TE projection with the same
group contract (duplicated TE down projections do not communicate). V1 validates every
communicating projection's stored TP group against that injected
object before forward. Static forwards use `pg_collection.cp`. Dynamic forwards require
scheduler-populated `PackedSeqParams.local_cp_size` and `PackedSeqParams.cp_group`. The data
scheduler validates that the size is a positive Python integer, the group is present, and its size
matches. The feature resolves that group with the shared `resolve_cp_group` helper, threads it
through layout, RoPE, preprocessing, and transport, and never mutates the module's stored
collection. Every collective names the injected TP or effective CP group; omission is a test
failure, not a fallback.

### Supported projection spec and exact TP path

V1 supports two exact projection profiles and rejects mixtures of them:

- **Local MCore:** the existing `ColumnParallelLinear` Q/KV projections and optional gate,
  `RowParallelLinear` output, and standalone `WrappedTorchNorm` Q/KV norms from
  `get_gpt_layer_local_spec(..., qk_layernorm=True)`. The converter replaces only the two norm
  builders with the feature-local equivalent needed for explicit SP parameter-gradient ownership.
- **Transformer Engine:** the ordinary old-path stack from
  `get_gpt_layer_with_transformer_engine_spec(..., qk_layernorm=True)`: duplicated `TELinear`
  Q/KV down projections, `TELayerNormColumnParallelLinear` fused norm+up projections, optional
  `TEColumnParallelLinear` gate, `TERowParallelLinear` output, and `IdentityOp` standalone norms.
  Every one of these module specs is preserved by identity.

For either profile, `make_mla_with_latent_cp_spec` replaces only `core_attention` with `IdentityOp`
and changes the attention module class. This makes the non-CP projection, norm, gate, and output
timeline identical to the corresponding old path; only the latent payload, phase KV expansion,
direct backend attention, merge, and CP transport are feature-owned. V1 still requires a nonzero
`q_lora_rank` and rejects fused QKV down projection, inference/modelopt profiles, and any unqualified
hybrid projection combination.

TP=1 uses the ordinary non-SP path. TP size `N>1` requires `sequence_parallel=True`, so each TP
rank enters with a contiguous `[T_r/N, 1, hidden_size]` sequence shard. The exact TP>1 path is:

1. Local column-parallel down projections gather the SP input and produce feature shards; the
   feature helper gathers those shards with the explicit TP group and scatters the complete Q/KV
   latents back to `[T_r/N,...]`. Duplicated TE down projections instead consume and return the
   existing sequence shard directly, so no feature gather/scatter is inserted.
2. The local profile applies its standalone norms before the ring. The TE profile communicates raw
   latent KV and leaves its standalone norms as identity because normalization remains fused into
   each preserved up projection. Both communicate one `[T_r/N,C+D_r]` payload per TP lane.

3. The preserved sequence-parallel Q up projection gathers Q shards and produces full local query
   rows `[T_r,H_tp,D_qk]`; on TE it also performs the fused RMSNorm. For duplicated TE down
   projections, the positional K sequence shard is explicitly gathered before owner-local RoPE and
   scattered afterward. The local feature-sharded path already has full rows before RoPE and performs
   the same final scatter. A no-RoPE layer concatenates the raw branches unchanged.
4. For every received shard, the preserved sequence-parallel KV up projection gathers latent KV
   over TP before producing local K-content/V heads; on TE it also performs fused RMSNorm. The
   module independently gathers the received K positional shard
   with `gather_from_sequence_parallel_region(..., tensor_parallel_output_grad=True, group=TP)`.
   Because splitting the contiguous ring payload along its channel dimension produces strided views,
   both the latent and positional slices are materialized as contiguous tensors before either TP
   collective; the collective boundary never relies on backend acceptance of noncontiguous inputs.
   Phase Q/KV rows use first-dimension views when the planner recorded a contiguous span, including
   the single-packed-sequence benchmark path, and retain `index_select` for general packed rows.
   The gather's backward reduce-scatter is the single TP sum for the shared positional-key gradient;
   the KV up-projection supplies the corresponding latent-input reduce-scatter.
5. The merged `[T_r, H_tp, D_v]` output is flattened and cast once to BF16. If
   `attention_output_gate=True`, the inherited public MLA helper projects `hidden_states` through
   the accepted TP-sharded `linear_gate` and applies an FP32 sigmoid followed by BF16 multiplication;
   `elementwise` supplies one gate per local output element and `headwise` one gate per local head.
   The gated or ungated tensor is then passed to the new-file-local `_explicit_output_projection`.
   The module and state-dict/sharded-state-dict names are unchanged. For the TE profile the helper
   calls the preserved `TERowParallelLinear`, matching the old path exactly. For the local profile it
   calls the public MCore linear primitive with `bias=None`, the stored
   `gradient_accumulation_fusion` flag, `allreduce_dgrad=False`, `sequence_parallel=False`, and
   `tp_group=self.pg_collection.tp`, then calls
   `reduce_scatter_to_sequence_parallel_region(..., group=TP)`. The returned layer output is
   `[T_r/N, 1, hidden_size]`; backward all-gathers dInput and leaves the local weight/main gradient
   on its normal row shard. V1 rejects output-projection bias, frozen output weights, and CPU
   offloading rather than silently changing those semantics.

`pg_collection.cp` contains ranks with the same TP lane, and `pg_collection.tp` contains ranks with
the same CP ownership. Each CP ring therefore carries a distinct sequence shard rather than a
replicated payload. Future A2A work may fuse the TP sequence redistribution with CP layout changes.

## Tensor and RoPE contract

### Notation

For one tensor-parallel rank:

| Symbol | Meaning |
| --- | --- |
| `P`, `r` | CP group size and local CP rank. |
| `T_r` | Total physical THD tokens owned by one CP rank across the packed batch. |
| `H_tp` | `num_attention_heads / tensor_model_parallel_size`. |
| `C` | `config.kv_lora_rank`. |
| `D_c` | `config.qk_head_dim`, non-positional Q/K content dimension. |
| `D_r` | `config.qk_pos_emb_head_dim`, positional-branch width (rotated only when enabled). |
| `D_qk` | Backend Q/K head dimension, `D_c + D_r`. |
| `D_v` | `config.v_head_dim`. |

For the targeted MLA shape, `D_c=128`, `D_r=64`, `D_qk=192`, and `D_v=128`. V1 allows any head
count divisible by TP, but requires those head dimensions and `H_q=H_kv`.

After the exact TP path above, the module forms:

- compressed KV `Z_kv`: `[T_r/N, C]` per TP lane, normalized for local MCore and raw for TE;
- raw shared positional key `K_pos_raw`: `[T_r, D_r]`;
- query `Q`: `[T_r, H_tp, D_qk]`, after Q up-projection and optional Q RoPE; and
- shared positional key `K_pos`: `[T_r/N, D_r]` per TP lane, after optional owner-local RoPE and
  sequence scatter.

The exact per-lane communicated payload is

```text
Z = concat(Z_kv, K_pos, dim=-1)        # [T_r/N, C + D_r], BF16
```

`K_pos` is required: MLA expands it over heads and concatenates it with the content key.
Communicating only `Z_kv` leaves a remote rank without that Q/K branch. When enabled, RoPE is
applied on the owner before communication so the receiver needs no position metadata. In no-RoPE
mode the same payload carries the raw branch. After each ring hop, the TP
group gathers the `N` sequence shards before phase indexing. The K-positional gather uses
`tensor_parallel_output_grad=True`, so its backward reduce-scatter sums the shared-key gradients
from all TP-local head shards. Sequence-parallel `linear_kv_up_proj` provides the same ownership
semantics for the latent branch.

For a received payload, the phase callable performs:

```text
KV_content = linear_kv_up_proj(sequence_shard(Z_kv))
           -> [T_r, H_tp, D_c + D_v]
K_pos = gather_from_sequence_parallel_region(sequence_shard(K_pos), group=TP)
       -> [T_r, D_r]
K_content, V = split(KV_content, [D_c, D_v], dim=-1)
K = concat(K_content, expand_heads(K_pos), dim=-1)
  -> [T_r, H_tp, D_qk]
V -> [T_r, H_tp, D_v]
```

The legacy full-K/V payload per TP lane has `T_r * H_tp * (D_qk + D_v)` elements. The latent
payload per TP lane has `(T_r/N) * (C + D_r)` elements, for the communication ratio

```text
(C + D_r) / (N * H_tp * (D_qk + D_v)).
```

This excludes any extra V padding in another implementation and therefore does not overstate the
reduction.

### Global packed positions are owner metadata when RoPE is enabled

Q and `K_rope` are rotated exactly once, before the ring, with the **original global packed
metadata**:

- `cu_global = packed_seq_params.cu_seqlens_q == packed_seq_params.cu_seqlens_kv`;
- `max_global = packed_seq_params.max_seqlen_q == packed_seq_params.max_seqlen_kv`, or the maximum
  adjacent difference of `cu_global` when the fields are absent; and
- the explicitly injected CP group, so MCore maps each local zigzag row to its original per-sequence
  global position.

The module passes `cu_global` and `max_global` to the public MCore rotary path for both Q and K. It
never passes `cu_full` or `cu_half` from the attention phase planner to RoPE. Those derived arrays
describe compact backend matrices only and would rotate tokens at incorrect positions if reused.

RoPE-enabled layers support `config.rope_type in {"rope", "yarn"}` with
`apply_rope_fusion=False`. `rope` uses the
standard MCore frequencies. `yarn` uses the current MCore Yarn parameters and scale, numerically
checked against the pinned official DeepSeek formula. Packed sequences reset positions at every
entry in `cu_global`; zigzag sharding changes storage ownership but not position numbers.

When the existing normalized per-layer `no_rope_freq[layer_number-1]` entry is true, the shared MLA
constructor sets `rotary_pos_emb=None` and uses the standard `1/sqrt(D_qk)` scale. The latent path
does not call `get_rotary_seq_len`, construct frequencies, or call `apply_rotary_pos_emb`; it
concatenates the raw Q/K positional-width branches exactly as the pinned Kimi-K3 MLA implementation
does. `apply_rope_fusion` is irrelevant for such a layer because no rotary operation exists.

## Zigzag ownership and phase plan

For `P>1`, each packed sequence of global physical length `S` requires `S % (2P) == 0`. Let `L=S/P` be the
rank-local length and split the global sequence into `2P` chronological chunks of length `L/2`.
Rank `r` owns

```text
F_r = chunk[r]
B_r = chunk[2P - 1 - r]
local sequence storage = concat(F_r, B_r)       # length L
```

The payload moves clockwise: rank `r` sends to `(r+1) mod P` and receives from `(r-1) mod P`. At
phase `i`, rank `r` holds the payload owned by `j=(r-i) mod P`.

| Phase on query rank `r` | Source owner | Q rows | KV rows | Kernel shape per sequence | Mask |
| --- | --- | --- | --- | --- | --- |
| `i = 0` | `j = r` | `F_r + B_r` | `F_r + B_r` | `L x L` | causal |
| `1 <= i <= r` | `j = r-i < r` | `F_r + B_r` | `F_j` (first KV half) | `L x (L/2)` | non-causal |
| `i > r` | `j = P+r-i > r` | `B_r` (second Q half) | `F_j + B_j` | `(L/2) x L` | non-causal |

On the diagonal, compact causal attention handles within-chunk causality and lets `B_r` attend
`F_r`, while preventing `F_r` from seeing `B_r`. For `j<r`, only `F_j` is earlier than both local
query chunks. For `j>r`, neither source chunk is visible to `F_r`, while both are earlier than
`B_r`. Every globally causal Q/K pair appears exactly once; rectangular phases need no dense mask.

For `P=1`, arbitrary positive packed lengths are accepted. There is one full/full causal diagonal
phase, `cu_half == cu_full`, `front_indices` names every row, `back_indices` is empty, and the
transport yields the local payload once without constructing a P2P operation.

### Backend phase metadata

Let the original global cumulative lengths be `cu_global=[0,S_0,S_0+S_1,...]`. V1 requires equal
Q/KV metadata and, for `P>1`, every `S_n` divisible by `2P`. The phase planner derives CUDA int32 metadata:

```text
cu_full = cumulative([S_n / P])       # compact rank-local full side
cu_half = cumulative([S_n / (2P)])    # compact front/back half side
max_full = max(S_n / P)
max_half = max(S_n / (2P))
```

For `P=1`, `cu_half=cu_full` and `max_half=max_full`; the half metadata is retained only to keep the
layout object uniform and is not used by its sole phase.

THD packs sequences consecutively and each local sequence is `[F_r,B_r]`. Every derived
cumulative tensor is contiguous `torch.int32` on the same device as `cu_global`. The implementation
passes `dtype=torch.int32` explicitly to `torch.cumsum`; integral `torch.cumsum` otherwise promotes
the result to `torch.int64`, which the public FA4 varlen API rejects. Sequence lengths and tensor
capacity validation bound the cumulative values before this derivation. The planner derives
per-sequence front/back indices and uses:

- diagonal: `cu_seqlens_q=cu_full`, `cu_seqlens_kv=cu_full`;
- lower phase: `cu_seqlens_q=cu_full`, `cu_seqlens_kv=cu_half`; and
- upper phase: `cu_seqlens_q=cu_half`, `cu_seqlens_kv=cu_full`.

These values are backend attention metadata only. They are never used to generate or apply RoPE.
The planner additionally records optional host `(start, stop)` spans for each Q, KV, and upper
scatter row map. A span is present only when its exact tensor indices form one contiguous interval;
otherwise the original index tensor remains the authoritative general packed representation.

V1 rejects inter-sequence and tail padding. The data scheduler sets `pad_between_seqs` from the
actual valid-versus-physical cumulative boundaries instead of conservatively claiming gaps for every
THD batch. Padded cumulative lengths must be absent or exactly equal to valid cumulative lengths, and
`T_r == cu_full[-1]`. This keeps P2P message sizes identical
and makes all phase descriptors unambiguous.

## Numerical and online-softmax contract

Both kernels consume BF16 Q/K/V. Their raw phase output is BF16 and their LSE/stats are FP32. Each
adapter immediately returns canonical FP32 `O_i` and FP32 `E_i` with shapes
`[T_q,H_tp,D_v]` and `[T_q,H_tp]`.

When an upper phase's back rows form a contiguous interval, only that interval of the accumulated
output/LSE enters the merge and the prefix/suffix are concatenated unchanged. This avoids allocating
full-size zero/`-inf` scatter buffers. General multi-sequence layouts retain the functional
`index_copy` fallback below, never an in-place write:

```text
O_i_full = zeros([T_r,H_tp,D_v], FP32).index_copy(0, back_indices, O_i)
E_i_full = full([T_r,H_tp], -inf, FP32).index_copy(0, back_indices, E_i)
```

Merge all selected rows in FP32:

```text
E = logaddexp(E_a, E_b)
w_a, w_b = exp(E_a - E), exp(E_b - E)
O = O_a * w_a + O_b * w_b
```

The custom autograd boundary saves `O_a`, `O_b`, `O`, and the two weights and returns the
analytical gradients:

```text
dO_a = w_a * dO
dE_a = w_a * (dE + sum(dO * (O_a - O), dim=-1))
```

with the symmetric equations for side `b`. This removes the generic elementwise autograd graph
without changing the FP32 online-softmax math or its LSE-gradient contract. Partial outputs and LSE
remain deliberately retained even though expanded K/V are not. After the last phase there is exactly
one `O.to(torch.bfloat16)` before reshape and `linear_proj`; there are no intermediate BF16 merges.

FA4 documents that LSE returned by `return_lse=True` supports `dLSE`. The FP32 merger returns FP32
`G_i` and `gE_i`; autograd casts `G_i` to the raw BF16 FA4 output dtype at the adapter boundary and
passes FP32 `gE_i` to public FA4 backward.

cuDNN Graph `sdpa(generate_stats=True)` returns FP32 stats, but public `sdpa_backward` has no
`dStats` input. For global FP32 output gradient `G`:

```text
w_i  = exp(E_i - E_global)                         # FP32
G_i  = w_i[..., None] * G                          # FP32
gE_i = sum(G_i * (O_i - O_global), dim=-1)         # FP32
```

In real arithmetic, the direct corrected cuDNN call would use local stats, `dO=G_i`, and
`o=O_global`, making its row scalar `D=dot(G_i,O_global)`. The phase adapter receives `(G_i,gE_i)`
and constructs the equivalent FP32 proxy:

```text
norm2 = sum(G_i * G_i, dim=-1)                     # FP32
safe  = isfinite(norm2) & (norm2 >= sqrt(finfo(FP32).tiny))
denom = where(safe, norm2, ones_like(norm2))
corr  = where(safe[...,None], (gE_i/denom)[...,None] * G_i, 0)
O_corr = O_i - corr                                # FP32
```

For safe rows, `dot(G_i,O_corr)=dot(G_i,O_global)` exactly in real arithmetic. Zero/tiny rows use
zero correction to avoid invalid or unstable division; their rounded behavior is part of numerical
qualification. Immediately before public `graph.sdpa_backward`, only `G_i` and `O_corr` are cast to
BF16. Q/K/V and returned dQ/dK/dV use BF16 graph I/O; stats and graph compute/intermediate types are
FP32. The two BF16 casts mean the identity is not claimed to be bitwise exact in floating-point.

Qualification tests cover exactly zero gradients, tiny norms on both sides of the mask threshold,
extreme phase weights, and very negative LSE. An uncorrected cuDNN backward is never accepted.

## Differentiable P2P ring and recomputation

### Autograd topology and gradient ownership

`_LatentRingExchange`, local to `transport.py`, is a `torch.autograd.Function` over one payload hop.
Forward and backward are:

```text
forward:  Y_r  = X_(r-1)    # send X_r to r+1, receive from r-1
backward: dX_r = dY_(r+1)   # send dY_r to r-1, receive from r+1
```

The attention loop has `P` compute phases and `P-1` exchange nodes. Autograd addition at each node
accumulates the local phase contribution with the gradient arriving from downstream nodes. Every
owner's payload gradient therefore contains all query-rank contributions when it reaches the
owner's original `Z`; no extra hop or latent all-reduce is needed. All ranks build the same graph,
so backward traverses ring hops in reverse phase order.

Gradient ownership is:

- Q gradients accumulate over all local phases and flow through the local Q projection;
- payload gradients follow the reverse ring on the same TP sequence lane. The latent component is
  TP-reduce-scattered by sequence-parallel `linear_kv_up_proj`; the shared K-positional component is
  TP-reduce-scattered by its explicit sequence gather. The earlier sequence scatters then
  all-gather both gradients before owner-local RoPE/KV norm and KV down projection;
- `linear_kv_up_proj` parameter gradients sum its phase uses locally; and
- normal MCore distributed parameter buffers reduce replicated parameter gradients over DP-CP.
  The module does not add another parameter all-reduce.

The constructor requires `pg_collection` and retains its `cp`/`tp` groups. Static forwards use the
stored CP group. For dynamic CP, the data scheduler owns creation and generic consistency checks for
the paired `PackedSeqParams.local_cp_size/cp_group` fields. A dynamic forward resolves the
per-microbatch group through the shared helper and passes that object to every layout, RoPE,
preprocessing, and transport operation without assigning it back to `pg_collection`. Peers are
resolved from the effective group with
`torch.distributed.get_process_group_ranks`. Consecutive microbatches may therefore use CP=1 or a
larger initialized subgroup on one immutable module.

### Phase checkpoint and saved-state scope

Each phase takes a Q view, one latent payload view, and immutable metadata. It performs KV
up-projection, key construction, direct backend attention, and returns canonical FP32 `(O_i,E_i)`.
Dropout is zero.

The cuDNN adapter uses a feature-owned custom autograd boundary. Forward saves Q, latent payload,
every trainable parameter of the preserved up-projection module, raw BF16 local output, and FP32
LSE, but not expanded K/V. Backward re-executes only KV expansion, applies the public cuDNN backward
graph to retained O/LSE state, then differentiates reconstructed K/V through the preserved local or
TE fused norm+up module. This preserves
gradient-accumulation-fusion side effects and parameter hooks while removing the second SDPA
forward. The FA4 adapter retains the non-reentrant phase-checkpoint fallback until its qualified
public API exposes an equally narrow split forward/backward ownership boundary.

Full K/V exist only during the initial phase forward and the short projection replay; they are
never sent or stored in ring state.

The intentionally retained activation classes are:

- one latent tensor per differentiable ring node, `O(P*T_r*(C+D_r))` elements;
- every canonical partial `O_i` in FP32 and `E_i` in FP32, plus final merge state; and
- local Q/projection inputs required by normal MCore autograd.

No expanded remote K/V is physically retained outside phase execution. It is absent from saved
autograd inputs/outputs and module state. An outer `saved_tensors_hooks` recorder enumerates the
shape, element count, dtype, Python tensor class, and semantic state class of every tensor physically
packed into the surrounding autograd graph after forward. The pack hook returns a holder while the
recorder keeps only its weak reference, so only holders still owned by live autograd state are
enumerated. It permits phase Q/latent inputs and FP32 partial O/LSE/merge state, while rejecting
expanded remote K/V shapes. The cuDNN custom autograd function owns saved Q/latent/O/LSE tensors
directly; FA4 checkpoint replay uses the same recorder contract.

The lifetime test includes a sensitivity control: its test-only native backend passes K/V through a
numerically identity custom autograd sentinel whose `save_for_backward` names the exact THD K/V
state, mirroring the public cuDNN adapter's custom-function contract. It temporarily replaces
checkpoint with direct phase execution and requires the same recorder to detect expanded value and
Q/K-shaped attention state. This avoids depending on private `einsum`/BMM saved-view layouts.
Tensor-wrapper weak references remain a supplemental check only. This is saved-state evidence, not
an allocator-wide memory snapshot claim. The cuDNN selective-recompute test also compares output,
Q, payload, projection-weight, and norm-weight gradients against an independent native reference and
asserts one SDPA forward plus one KV replay.

To avoid ambiguous nested checkpoint/offload behavior, v1 requires
`recompute_granularity is None` and `fine_grained_activation_offloading=False`. The inactive
`recompute_modules` value is deliberately ignored because `TransformerConfig` normalizes an
unspecified list to `["core_attn"]` even when recompute is disabled. The variant rejects outer
full-layer recompute, selective `mla_up_proj`/`core_attn` recompute, and fine-grained activation
offload at construction. Supporting
any of them later requires explicit nested-checkpoint and saved-tensor/offload tests.

### Pipelined transport, lifetimes, and deadlock ordering

Before yielding phase `i`, every rank submits the exchange for phase `i+1` on one process/device
communication stream. The public `torch.distributed.batch_isend_irecv` batch contains `P2POp`
objects ordered as `[isend(next), irecv(previous)]`, and every operation sets
`group=effective_cp_group`. The communication stream first waits for the current payload producer;
each returned `Work.wait` is issued while that stream is current, followed by a CUDA readiness event.
The generator then yields phase `i` on the ordinary attention stream. When it resumes for phase
`i+1`, that consumer stream waits on the event, so the intervening attention kernels can overlap the
one-hop receive without exposing an unready tensor.

Backward submits `[isend(previous), irecv(next)]` on the same communication stream and inserts the
corresponding event dependency before returning `dX_r`. Send and receive tensors are recorded on the
communication stream, and the pending lease retains send storage until the consumer dependency is
installed. CP=1 creates no stream and submits no P2P. Fixed payload shapes, peer order, operation
lists, and phase counts remain identical across ranks, preventing mismatched-message and
parity-order deadlocks.

Feature-static config, package, runtime, and capability checks run in the layer constructor, using
the configured maximum CP/TP groups where group properties are needed. Cheap activation checks run
in `forward`. Before the layer loop, block-level preprocessing derives one immutable microbatch
layout shared by every latent-CP layer, then builds or reuses expensive cuDNN Graph plans and
canonical ragged bindings. It stores no attention result, performs no rank consensus, and introduces
no decoder scope or wrapper class. During a phase, the cuDNN adapter only looks up a previously
prepared binding; it cannot build a graph or derive metadata in the ring. A direct/custom layer call
that bypasses block preprocessing therefore fails in phase zero
before the first ring hop. Once P2P starts, the module makes no claim to recover from an arbitrary
Python, CUDA, or NCCL exception; failures propagate through normal PyTorch/NCCL error handling.

`LatentCPTransport` remains an extension seam. `PayloadLease.tensor` is ordered for use on the
consumer stream before it is yielded; the readiness event is transport-private. A future explicitly
configured transport may preserve the same lease contract while changing the collective topology.

## Direct backend adapters

Both adapters implement:

```text
forward_phase(q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale)
    -> (output_fp32, lse_fp32)
```

They canonicalize backend outputs but do not know CP transport or ownership. Planner-owned
`cu_q` and `cu_kv` cross the backend boundary unchanged: each must be a contiguous `torch.int32`
tensor colocated with its Q or K tensor. Invalid cumulative metadata is never normalized by an
implicit cast or copy.

### FA4 adapter

The only attention callable is public `flash_attn.cute.flash_attn_varlen_func`, imported directly.
The call supplies `q`, `k`, `v`, the exact planner-owned contiguous-int32 `cu_seqlens_q` and
`cu_seqlens_k` tensors, `max_seqlen_q`, `max_seqlen_k`, `softmax_scale`, `causal`, and
`return_lse=True`. The adapter validates dtype, contiguity, and Q/K device colocation before the
public call and performs no metadata conversion. It does not call an MCore/TE attention wrapper.

Public FA4 source establishes rank-3 THD Q/K/V, distinct Q/K and V head dimensions, rectangular
lengths, causal selection, returned LSE, and differentiable LSE. Development qualification still
checks the installed package, canonical LSE layout, all three packed phase shapes, and target
`D_qk=192,D_v=128`. V1 enables FA4 only on the qualified SM100 tuple.

### cuDNN adapter

The adapter imports public `cudnn` and uses only:

- `cudnn.pygraph`;
- public graph tensor creation/configuration, including ragged offsets;
- `graph.sdpa(..., generate_stats=True)` and `graph.sdpa_backward(...)`;
- `graph.validate`, `graph.build_operation_graph`, `graph.create_execution_plans`,
  `graph.check_support`, `graph.build_plans`, `graph.get_workspace_size`, and `graph.execute`; and
- public `cudnn.data_type` and `cudnn.heur_mode` enums.

One cuDNN adapter, public handle, plan cache, and bounded ragged-binding cache are shared by all
latent-CP layers in one process on one CUDA device. Forward/backward graphs are keyed by
process/device identity, exact
frontend/runtime versions, dtype and capability, heads/dimensions, phase shape, causal flag, scale,
maximum packed lengths, and metadata capacity. Each binding retains exact cumulative tensors,
canonical rank-4 length/offset buffers, actual token totals, and its prepared plan. Handle mutation,
plan construction, and execution are lock-protected. Workspaces are persistent per
`(plan, CUDA stream, direction)`, so same-stream reuse remains ordered while different streams never
share scratch storage. Variant packs and retained O/LSE remain invocation-owned. When an exact
64-token-aligned capacity is already contiguous, Q/K/V/O/dO/stats bind directly without a staging
copy; non-aligned phases retain the padded fallback. `generate_stats=True` supplies local FP32 LSE
for the corrected backward.

The public Attention descriptor contract is used literally for all ragged metadata. `SEQ_Q` and
`SEQ_KV` are contiguous INT32 graph tensors and bound buffers with dimensions `(B, 1, 1, 1)` and
strides `(1, 1, 1, 1)`. `Q_OFFSET`, `K_OFFSET`, `V_OFFSET`, `O_OFFSET`, and `STATS_OFFSET` are
contiguous INT64 graph tensors and bound buffers with dimensions `(B + 1, 1, 1, 1)` and the same
unit strides. Their flattened values remain the per-sequence lengths and element offsets described by
the ragged SDPA contract. V1 never relies on implicit rank-1 descriptor promotion: that behavior is
frontend/runtime-dependent and some otherwise supported Graph implementations reject rank-1
sequence-length descriptors while building the operation graph.

The source pin for implementation and qualification is
[`NVIDIA/cudnn-frontend@0a14b7181d129d30e7bad34b8c3ed0a0c995e23d`](https://github.com/NVIDIA/cudnn-frontend/commit/0a14b7181d129d30e7bad34b8c3ed0a0c995e23d).
No TE wrapper, `transformer_engine_torch`, `tex.*`, or underscore-prefixed TE helper is allowed.

Public references:

- [FA4 CuTe public interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py)
- [FA4 CuTe usage](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md)
- [cuDNN attention operations](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)
- [cuDNN Frontend releases](https://github.com/NVIDIA/cudnn-frontend/releases)

### Qualified runtime matrix and runtime checks

Numerical qualification is a development/CI activity, never a constructor or first-training-step
self-test. The new production module owns the source pin and exact runtime gate as immutable
module-local constants:

```python
CUDNN_FRONTEND_SOURCE_REV: Final[str] = (
    "0a14b7181d129d30e7bad34b8c3ed0a0c995e23d"
)
QualifiedBackendTuple: TypeAlias = tuple[
    AttnBackend, str, str, tuple[int, int]
]
QUALIFIED_BACKEND_CONFIGS: Final[tuple[QualifiedBackendTuple, ...]] = (
    (AttnBackend.fused, "1.22.1", "9.21.0", (9, 0)),
    (AttnBackend.fused, "1.26.0", "9.25.0", (10, 0)),
    (AttnBackend.flash, "4.0.0b11", "flash-attn-4==4.0.0b11", (10, 0)),
)
```

The tuple fields are exactly `(backend, frontend_or_package_version,
linked_cudnn_runtime_or_fa4_distribution, compute_capability)`. For fused, the third field is the
linked cuDNN runtime version. For flash, it is the exact distribution identity. Compute capability
is the exact `(major, minor)` pair. There are no wildcards, ranges, environment/config overrides, or
runtime mutation.

The completed sanitized qualification matrix is:

| Hardware | Backend | Frontend/package | Runtime/distribution identity | Evidence epsilon |
| --- | --- | --- | --- | --- |
| H100 / SM90 | `AttnBackend.fused` | `1.22.1` | cuDNN `9.21.0` | `4.561878810305231e-05` |
| SM100 | `AttnBackend.fused` | `1.26.0` | cuDNN `9.25.0` | `4.423665134356547e-05` |
| SM100 | `AttnBackend.flash` | `4.0.0b11` | `flash-attn-4==4.0.0b11` | `4.3095951884009054e-05` |

The test file independently spells the exact tuple sequence in
`EXPECTED_QUALIFIED_BACKEND_CONFIGS` and the exact tuple-to-epsilon mapping in
`EXPECTED_QUALIFICATION_EPS`. Tests require exact production/test/source-revision equality, exact
mapping keys, and no extra entry. The epsilon is a parity assertion threshold, not a production
runtime knob.

Qualified full-parity tests do not assign an adapter or mutate the allow-list. They resolve the
installed tuple without constructing an adapter, require exact membership, construct the layer, and
require its constructor-populated adapter and runtime tuple to match. They call the same explicit
block preprocessing entry point before forward. A real-backend test on an installed but unqualified
tuple skips with the detected tuple and complete exact allow-list in its reason before layer or
backend construction. Production itself remains fail-closed and raises; it never skips, probes
numerically, or falls back. Direct cuDNN mathematical diagnostics use the same resolution gate
before obtaining the public qualified adapter.

Sanitized qualification evidence records the source revision, exact distribution/runtime and
compute capability, configuration/seed, per-tensor cosine and tensor-similarity minima, derived
epsilon, and descriptor/plan status. Environment-specific paths, system names, and raw launch
identifiers remain outside this document.

At `MLAWithLatentCP.__init__`, runtime constructs the four-field tuple from the selected backend,
exact installed frontend/package version, exact linked cuDNN runtime or FA4 distribution identity,
and `torch.cuda.get_device_capability()`. It requires exact membership in
`QUALIFIED_BACKEND_CONFIGS`; missing metadata fails closed. The constructor immediately creates the
qualified adapter. These checks are feature-specific and independent of a microbatch.

Cheap activation, dtype, head-dimension, packed-metadata, and effective-group checks stay in
`forward`. `preprocess_mla_latent_cp(block, hidden_states, packed_seq_params)` is called by both
`TransformerBlock` and `HybridStack` after input scheduling and before the layer loop. It derives one
exact microbatch phase layout with the effective CP group, installs that immutable layout on every
latent-CP layer, and calls each adapter's `prepare` method. FA4 preparation is a no-op; the shared
cuDNN adapter builds or reuses public forward/backward Graph plans and canonical ragged buffers.

The generic data scheduler already transfers THD boundaries to the host while building the
zigzag/contiguous partition route. That route retains the compact host boundary tuple, effective CP
size/rank, and source metadata identity; finalization also resolves the generic inter-sequence
padding flag once. The latent layout adapter consumes that scheduler-owned tuple instead of
synchronizing CUDA scalars or repeating padding comparisons in every layer. A bounded 16-entry
semantic LRU, keyed by the host boundaries, effective CP geometry, token count, device, and maximum
length, reuses derived
`cu_full`/`cu_half`, front/back indices, and immutable phase specs across microbatches. A hit replaces
only `cu_global` with the current microbatch tensor used by RoPE. CP=1 routes are identity routes and
retain arbitrary positive odd sequence lengths for dynamic-CP degeneration.

The preprocessing function stores no attention result, creates no decoder context or wrapper class,
and launches no collective. `forward` validates a host-only identity key and reuses the prepared
layout without reading CUDA scalars. Stable derived phase-tensor identities make the bounded cuDNN
ragged-binding cache hit across semantically identical microbatches, so phase execution never
rebuilds a key, ragged buffer, or Graph inside the ring. A
custom caller that invokes a latent-CP layer outside a normal block must call the public
preprocessing function first or fail before the first phase executes.

The adapter caches deterministic Graph artifacts by runtime/shape contract and up to 128 immutable
microbatch bindings by metadata object identity. It is not an availability protocol and never runs
reference inputs, compares numbers, mutates the allow-list, or provides a numerical probe.

## Feature-owned architecture and config-driven construction

The algorithm lives in a same-name feature package, while its dedicated tests remain in one file:

```text
megatron/core/transformer/experimental_attention_variant/mla_with_latent_cp/
├── __init__.py
├── backend.py
├── cudnn_backend.py
├── fa4_backend.py
├── layout.py
├── mla_with_latent_cp.py
├── specs.py
├── transport.py
└── utils.py
tests/unit_tests/transformer/experimental_attention_variant/test_mla_with_latent_cp.py
```

The package boundaries follow runtime ownership: `mla_with_latent_cp.py` contains the MLA subclass,
projection/forward path, and block preprocessing entry point; `layout.py` owns packed-zigzag
validation and deterministic phase planning; `transport.py` owns the differentiable synchronous P2P
ring; `fa4_backend.py` and `cudnn_backend.py` contain only their respective public backend adapters;
`backend.py` owns exact runtime qualification and dispatch; `specs.py` owns non-mutating GPT/Hybrid
spec transformations; and `utils.py` owns shared errors, immutable qualification constants, and
backend-independent FP32 merge/correction helpers. `__init__.py` preserves the original package
import surface without registration side effects.

Dependencies point from the MLA module toward layout, transport, backend, and utilities; backend
dispatch points toward the concrete adapters; spec integration points toward the MLA module.
Concrete backends never import the MLA class or model-spec integration, preventing import cycles.

Small integration changes add `TransformerConfig.mla_latent_cp`, generic dynamic-group consistency
checks in the data scheduler, and one feature-spec initialization call in each of GPTModel and
HybridModel. GPT/Hybrid layer-spec builders and training builders do not carry latent-CP branches.
The feature initializer rewrites only compatible attention slots and preserves the ordinary block
class. `TransformerBlock` and `HybridStack` each call the feature-owned preprocessing function once
before their layer loop. The shared `multi_latent_attention.py` change is limited to the existing
`no_rope_freq` contract: it avoids rotary construction/application and selects the standard scale
on flagged layers. It contains no latent-CP algorithm.

`get_gpt_layer_local_spec(...)` returns a whole transformer-layer `ModuleSpec`; the attention
factory does not accept that outer object. `make_mla_with_latent_cp_spec(base_mla_spec)` accepts
exactly `layer_spec.submodules.self_attention`, validates that it is one of the two supported MLA
projection profiles, and returns a new `ModuleSpec` whose module is `MLAWithLatentCP`. It constructs a new
`MLASelfAttentionSubmodules` with `core_attention=IdentityOp` and copies `params`/`metainfo` mappings;
it never mutates `base_mla_spec`, its submodules, or the outer layer spec.

The lower-level manual construction pattern remains available to spec authors:

```python
from dataclasses import replace

layer_spec = get_gpt_layer_local_spec(
    num_experts=None,
    moe_grouped_gemm=False,
    qk_layernorm=True,
    multi_latent_attention=True,
    normalization="RMSNorm",
)
base_mla_spec = layer_spec.submodules.self_attention
latent_mla_spec = make_mla_with_latent_cp_spec(base_mla_spec)

# `replace` returns fresh dataclass instances; the original `layer_spec` is still usable unchanged.
latent_layer_spec = replace(
    layer_spec,
    params=dict(layer_spec.params),
    metainfo=dict(layer_spec.metainfo),
    submodules=replace(
        layer_spec.submodules,
        self_attention=latent_mla_spec,
    ),
)
```

The factory internally uses the same `dataclasses.replace` pattern for `base_mla_spec` and
`base_mla_spec.submodules`; it does not assign to either input object. Normal model construction
passes each existing MLA attention spec through `make_mla_with_latent_cp_spec`, preserving its
projection stack. `configure_mla_latent_cp_decoder()` accepts either one transformer-layer `ModuleSpec` or a
`TransformerBlockSubmodules`, replaces only exact ordinary `MLASelfAttention` slots, and leaves GDN,
KDA, DSA/CSA/HCA, Mamba, MLP, and MoE slots untouched, and returns only the fresh spec.
`configure_mla_latent_cp_hybrid_stack()` derives its ordinary attention slot from the Hybrid stack's
own `mla_layer` template, replaces that template's attention, and returns a fresh root spec while
preserving the original stack module. Neither path mutates its input spec, nests the decoder under a
wrapper module, changes state-dict keys, or imports GPT code from the Hybrid layer-spec file.
Ordinary `TransformerBlock` and `HybridStack` call `preprocess_mla_latent_cp` before entering their
layer loop; the feature does not synthesize a decoder subclass or maintain forward-scoped state.

Model construction opts in through configuration:

```yaml
multi_latent_attention: true
mla_latent_cp: true
attention_backend: fused
cp_comm_type: p2p
cp_partition_mode: zigzag
```

The argument factory also exposes `--mla-latent-cp`. Ordinary GPT replaces self-attention in both
dense and MoE layers; gated-delta GPT and `HybridModel` replace only their standard-attention (`*`)
slots. Mamba/GDN, D, CSA/HCA, window-attention, MLP, and MoE slot selection stays unchanged.
Structurally compatible explicit layer/stack specs are transformed non-mutatingly; specs without an
ordinary MLA template, inference/modelopt, MTP, and other unsupported combinations fail rather than
silently ignoring the flag. `attention_backend` remains the fused-versus-FA4 selector. The manual
factory above remains useful to developers building custom specs.

The reused projection state-dict names stay compatible, while unsupported specs fail during factory
construction rather than halfway through forward.

## Validation placement and early errors

Responsibility follows ownership and cost:

- the DCP data scheduler validates that dynamic `local_cp_size` is positive, its `cp_group` exists,
  and the group size matches before publishing `PackedSeqParams`;
- `MLAWithLatentCP.__init__` validates feature-static config, projection/group wiring, package and
  runtime versions, hardware capability, and exact allow-list membership;
- `forward` validates cheap activation, dtype, and effective-group identity for the current
  microbatch; and
- block preprocessing owns layout validation plus microbatch-specific cuDNN bindings and plans.

The supported contract is:

- `mla_latent_cp == multi_latent_attention == True` when selected through model configuration;
- explicit `ProcessGroupCollection` with usable maximum `cp` and `tp` groups;
- `cp_comm_type == "p2p"` for CP greater than one;
- one exact local-MCore or TE projection/norm profile and nonzero Q LoRA described above;
- `PackedSeqParams`, `qkv_format == "thd"`, and `cp_partition_mode == "zigzag"`;
- equal CUDA-int32 Q/K cumulative lengths, monotonic self-attention metadata, original global max
  lengths, and every physical sequence divisible by `2P` when `P>1`; CP=1 accepts arbitrary
  positive lengths;
- padded cumulative lengths absent or equal to valid lengths, with local tokens matching
  `cu_full[-1]`;
- causal logical mask, `attention_mask is None`, and self-attention;
- BF16 activations and weights, a trainable bias-free output projection, zero dropout, and
  FP16/FP8/FP4 disabled;
- TP greater than one only with `sequence_parallel=True`; the physical input/output tokens are the
  corresponding first-dimension TP shard while packed metadata retains the pre-SP CP-local count;
- training without inference context/cache, CUDA graph, outer/selective recompute, fine-grained
  activation offload, or CPU offloading;
- either a `rope`/`yarn` layer with fused RoPE disabled, or a `no_rope_freq`-selected layer that
  constructs/applies no rotary embedding and uses the standard attention scale;
- when `attention_output_gate=True`, an accepted bias-free TP-sharded `linear_gate` and either
  `elementwise` or `headwise` granularity;
- heads divisible by TP, `H_q=H_kv`, `D_c+D_r=192`, and `D_v=128`;
- backend exactly `AttnBackend.fused` or `AttnBackend.flash`; and
- valid public descriptors and prepared cuDNN plans before phase execution.

`AttnBackend.auto`, `unfused`, and `local` are not remapped. `AttnBackend.flash` means FA4, not
FA2/FA3, and is rejected below SM100. Any installed tuple outside the exact matrix remains
unqualified; it does not trigger a runtime numerical test or TE fallback.

## Extension interfaces

Two small protocols keep future collectives out of attention math:

```text
LatentCPLayoutAdapter.prepare(local_hidden, packed_params, cp_group) -> LayoutView
LatentCPTransport.iter_payloads(local_payload, phase_plan) -> Iterator[PayloadLease]
```

`LayoutView` owns front/back indices, phase cumulative lengths, owner global-position mapping, and
output restoration. V1's `AlreadyZigZagTHDAdapter` validates and returns views. A future
`ContiguousToZigZagAdapter` can use public `CpPartitionModeConverter`, but remains separate rather
than hiding an eager conversion in the attention module.

`P2PRingTransport` yields an owner plus a consumer-stream-ordered payload while prefetching the next
hop. A future `HierarchicalA2AP2PTransport` can consume a layout plan and combine
contiguous-to-zigzag permutation with low-level A2A, then expose the same owner order and readiness
contract. Its process groups must be injected, for
example through `pg_collection.hcp`; no global `parallel_state` read is introduced.

These are extension seams, not placeholder implementations in v1.

## Independent native reference contract

The primary correctness reference is the official
[`deepseek-ai/DeepSeek-V3` repository at `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`](https://github.com/deepseek-ai/DeepSeek-V3/tree/9b4e9788e4a3a731f7567338ed15d3ec549ce03b),
specifically [`inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py):

- `precompute_freqs_cis` and `apply_rotary_emb` define Rope/Yarn frequencies and rotation;
- `MLA.forward` defines Q low-rank projection/norm/up-projection, KV down split, KV norm/up-projection,
  shared positional K expansion, scaling, FP32 softmax, V reduction, and output projection; and
- its `naive` attention branch is the formula source, not its cache mutation or distributed wrappers.

The test implements a local `NaiveMLA` using only standard `torch`, `torch.nn`, and
`torch.nn.functional` operations. It does not import MCore projection, RoPE, attention, CP helpers,
FA4, cuDNN, or TE. It loops over original packed sequences, computes full global causal attention,
and only then applies an independently written zigzag index mapping for comparison.

The no-RoPE branch is separately pinned to
[`moonshotai/Kimi-K3` at `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py).
It retains the positional-width Q/K branches but concatenates them without frequency construction
or rotation and uses `D_qk**-0.5`. The independent reference also implements the latest MLA output
gate directly with standard `nn.Linear`, an FP32 sigmoid, and elementwise or per-head BF16
multiplication.

The parameter map is exhaustive and bijective for the required Q-LoRA/no-bias configuration:

| Official reference parameter | MCore parameter | TP reconstruction before comparison |
| --- | --- | --- |
| `wq_a.weight` | `linear_q_down_proj.weight` | concatenate output shards on dim 0 |
| `q_norm.weight` | `q_layernorm.weight` | replicated; assert equality across TP lanes |
| `wq_b.weight` | `linear_q_up_proj.weight` | concatenate head/output shards on dim 0 |
| `wkv_a.weight` | `linear_kv_down_proj.weight` | concatenate output shards on dim 0 |
| `kv_norm.weight` | `kv_layernorm.weight` | replicated; assert equality across TP lanes |
| `wkv_b.weight` | `linear_kv_up_proj.weight` | concatenate head/output shards on dim 0 |
| `wo.weight` | `linear_proj.weight` | concatenate input/head shards on dim 1 |
| optional `gate.weight` | `linear_gate.weight` | concatenate output/head shards on dim 0 |

The test asserts both parameter-name sets are exhausted, every shape/slice is exact, and no trainable
parameter is ignored. For gradients, it first sums CP-replicated contributions as MCore DP-CP would,
then reconstructs TP shards by the table. It compares output, local zigzag hidden-state gradient,
and every reconstructed parameter gradient.

The required feasible production-shape case uses hidden size 7168, `H=96`, Q-LoRA rank 1536,
KV-LoRA rank 512, `D_c=128`, `D_r=64`, `D_qk=192`, `D_v=128`, TP=2, CP=2, and packed lengths
`[128,8,8,8,8]`. It preserves the requested 80%/four-5% packing while keeping attention memory
small. Both `rope` and `yarn` run. The pinned official oracle computes its long-context scale from
`config.mscale`, the sole source field in `inference/model.py`. Yarn qualification explicitly sets
both MCore `mscale=1.0` and `mscale_all_dim=1.0`; the latter is an MCore compatibility field that
makes its scale equivalent to the official formula and is not read by the oracle. The test lowers
`original_seq_len` so at least one packed sequence exercises extrapolation.

Similarity uses both double-precision cosine similarity and tensor similarity for every compared
tensor. Qualification records

```text
observed_error = max(1-cosine, 1-tensor_similarity)
eps = max(1e-5, 2*max_observed_error)
```

over all qualification seeds/tensors for a backend/architecture tuple. Each parity run retains both
metrics for output, input gradient, and all seven base parameter gradients plus the optional gate
gradient. It reduces their
minima over the injected TP-CP group and one deterministic rank per TP-CP grid prints one stable
JSON line containing backend, RoPE mode, seed, exact runtime tuple, exact `qualified_eps`,
per-tensor metrics, `max_observed_error`, and newly observed candidate epsilon. For a qualified
full-backend run, every metric must exceed `1-qualified_eps` and the newly observed candidate must
be no larger than that tuple's frozen evidence epsilon. All tuple epsilons also remain below the
global `1e-3` hard ceiling. The backend-independent Torch-phase diagnostic retains the global
ceiling. Zero tensors require exact zero/nonfinite checks before the similarity convention is
applied. CPU FP64 online-merge tests supplement this reference but are not the main parity oracle.

## New-file-only test plan

All tests live in the new experimental test file; existing MLA tests remain untouched.

1. **Static phase-plan tests.** For CP 1/2/4 and multiple packed sequences, assert owner order,
   front/back indices, exact three-shape schedule, metadata, and one-time causal-pair coverage.
   CP=1 additionally uses odd packed lengths, one full/full phase, equal full/half metadata, and no
   back rows. Single-sequence layouts require exact diagonal/lower/upper slice spans; multi-sequence
   noncontiguous front/back maps retain the index fallback.
   Require every planner-derived diagonal/lower/upper Q/KV cumulative tensor to be contiguous
   `torch.int32`, catching the default integral-`cumsum` promotion to `torch.int64`.
2. **Global-position parity.** For multi-sequence THD, independently construct original per-sequence
   positions, zigzag-shard them, and compare owner Q/K rotation for both `rope` and `yarn`. Assert
   original `cu_global/max_global` reach RoPE and derived `cu_full/cu_half` reach only backends.
   A Kimi-style no-RoPE case makes both rotary constructors and the apply function raise if called,
   requires `rotary_pos_emb is None`, and compares the raw Q/K positional-width branches through the
   independent Kimi-pinned full-chain oracle.
3. **Projection and TP contract.** Assert both accepted profiles. The local profile requires exact
   Q/KV last-dimension gathers and three first-dimension scatters (Q latent, KV latent, and K
   positional), all with `group is pg_collection.tp`; the TE profile preserves duplicated down,
   fused norm+up, gate, and output specs by identity and adds only the K-positional SP gather/scatter.
   TP>1 with SP disabled must fail at construction. Gather the
   per-lane BF16 payloads along the sequence dimension and require the pre-SP token count. For a
   real phase, spy on the explicit K-positional sequence gather and require
   `tensor_parallel_output_grad=True`; the obsolete non-SP copy mapping and every default-group
   resolver must raise. Assert every accepted projection stores the injected TP group. For output
   projection, make inherited `RowParallelLinear.forward` and default resolution raise, require the
   explicit TP group on the public linear and sequence reduce-scatter calls, and compare BF16
   forward, dInput, and local weight gradient against independent `F.linear` plus an explicit
   reduce-scatter contract. Separately require model integration to preserve every TE projection
   module by identity while replacing only core attention.
   The local fallback is tested for both gradient-accumulation flag values.
   Output-gate parity exhaustively maps the optional gate weight and compares output, input
   gradient, gate gradient, and every base parameter gradient for elementwise and headwise modes.
4. **Payload and ring tests.** Assert each forward P2P tensor has `T_r*(C+D_r)` elements, never the
   full-K/V size. For CP=2/4, prove forward owner routing, reverse gradient routing, fixed peer order,
   that every recorded `P2POp` constructor receives the effective CP group, and that the next
   exchange is submitted on the dedicated stream before the current lease is yielded. Every returned
   work must be waited exactly once. After construction, patch
   `parallel_state.get_tensor_model_parallel_group`,
   `get_context_parallel_group`, and `get_tensor_and_context_parallel_group` to raise throughout
   the complete TP=2 x CP=2 production forward/backward; test-harness collectives remain explicitly
   bound to injected groups outside that guard.
   CP=1 yields the local payload once and must never invoke `_LatentRingExchange`.
5. **Independent multi-rank parity.** Run the pinned `NaiveMLA` reference against TP=2 x CP=2 with
   `H=96,D_qk=192,D_v=128`. Compare output, input gradient, and every mapped parameter gradient with
   both similarity metrics after explicit CP reduction/TP reconstruction. The H100/SM90 fused and
   SM100 fused/FA4 cases use constructor qualification plus explicit block preprocessing, assert the
   exact tuple epsilon for every metric, and require the candidate epsilon not to exceed it.
   Separately construct the unchanged `MLASelfAttention + TEDotProductAttention` P2P CP path and the
   latent-CP path from the same real TE projection spec, with identical TP=2 x CP=2 SP-sharded input,
   weights, and upstream gradient. For rope and YARN, compare output and SP-sharded input gradient
   directly, then compare every parameter gradient after the same explicit CP reduction and TP
   reconstruction. This accounts for latent recomputation and legacy pre-expansion assigning KV-up
   work to different CP ranks while also protecting the non-CP TE module topology. The
   independent pinned `NaiveMLA` parity remains the backend-independent oracle.
   A separate dynamic-CP run constructs one module with static CP=2, then compares effective CP=1
   (including odd lengths) and CP=2 forwards/backwards against the same reference. It requires the
   module's static group object to remain unchanged and reconstructs parameter gradients over the
   effective group.
6. **Backend dispatch, qualification, and preprocessing ownership.** Assert fused construction
   creates only the shared direct cuDNN adapter and flash only FA4; TE `DotProductAttention` is
   neither built nor called. The fake public FA4 callable receives exact planner-owned int32 tensors
   by identity; invalid Q or KV dtype, contiguity, or device colocation fails without conversion.
   Assert the production source pin and immutable allow-list equal the independently spelled
   evidence tuples exactly, and that the epsilon mapping has the same keys and no extras. An
   installed unqualified tuple skips a real-backend test before layer/adapter/Graph construction.
   A qualified layer must hold its matching adapter/runtime immediately after construction. The
   explicit block function derives one layout and shares it across latent-CP layers. Fake cuDNN tests
   call `prepare` twice and prove the second call reuses the same plan and canonical metadata binding;
   phase execution only performs a binding lookup. Generic route tests cover host boundary ownership
   and CP=1 odd-length identity routes; a focused cache test uses distinct microbatch tensors with the
   same boundaries and proves phase metadata identity is stable without tensor-scalar reads. CPU
   guards require exact-capacity staging to
   preserve tensor identity and workspace reuse to remain isolated by direction and CUDA stream.
   Feature-initialization tests require non-mutating GPT/Hybrid spec rewrites that preserve ordinary
   block classes. Existing data-scheduler tests own dynamic `local_cp_size/cp_group` consistency;
   full dynamic parity owns per-microbatch effective-group behavior.
7. **cuDNN merge backward.** Compare all phase shapes against standard PyTorch, including `G_i`,
   `gE_i`, `O_corr`, zero/tiny norm rows, extreme phase weights/LSE, and BF16 boundary casts.
8. **Dtype and functional merge.** Assert raw BF16 backend output, canonical/merged FP32 output+LSE,
   analytical output/LSE gradients against direct softmax, contiguous subset merge parity against
   the functional upper-scatter fallback, and exactly one final BF16 cast before
   `_explicit_output_projection`.
9. **Recompute/lifetime evidence.** Count `P` KV up-projections in forward and `P` in backward replay,
   while cuDNN executes exactly `P` SDPA forwards total. Outer saved-tensor hooks enumerate retained
   shape/numel/dtype/Python class and classify Q/latent plus partial O/LSE state; expanded K/V is
   forbidden. Independent native math checks Q, payload, and all up-projection parameter gradients. The
   FA4 checkpoint-disabled sensitivity control must still expose expanded value and Q/K-shaped
   saved state, proving the recorder would catch a lifetime regression.
10. **Negative validation.** Cover unsupported projection specs, SBHD, contiguous/A2A modes,
    padding, non-divisible CP>1 lengths, malformed/mismatched dynamic metadata,
    non-causal/explicit masks, FP16/FP8/FP4, dropout,
    inference, TP>1 without sequence parallel, fused RoPE, unsupported head dims, missing groups,
    outer/selective recompute, fine-grained offload, CPU offloading, frozen output weights, unqualified
    versions/hardware, and unsupported backend enums.
11. **Hardware policy.** H100/SM90 runs the exact qualified fused/cuDNN tuple and keeps the FA4
    architecture skip. SM100 runs both exact qualified backends. A different installed runtime
    skips only real-backend tests before Graph/kernel construction with the detected tuple in the
    reason; static, Torch-phase, planner, transport, and negative tests continue normally.

## Post-qualification validation and rerun contract

1. Run import/static checks, Ruff lint/format, isort, compile checks, and the full CPU-safe new-file
   suite after any source change.
2. Assert the source revision, exact three-entry production allow-list, independently spelled test
   tuple list, and independently spelled tuple-to-epsilon mapping remain equal with no extras.
3. On H100/SM90 with Frontend `1.22.1` and cuDNN `9.21.0`, run both gated direct-cuDNN backward
   diagnostics and fused `rope`/`yarn` TP=2 x CP=2 full parity through construction and preprocessing.
4. On SM100, run fused Frontend `1.26.0` with cuDNN `9.25.0` and FA4
   `flash-attn-4==4.0.0b11`, for both `rope` and `yarn`, then run the complete new test file.
5. For every qualified full parity, archive the stable JSON line, require every per-tensor metric
   against the exact tuple epsilon, and require the newly observed candidate epsilon not to exceed
   that frozen value. Run relevant existing transformer/MLA tests without changing their backend
   selection or adding a feature-specific workaround.
6. On an installed tuple outside the exact matrix, verify real-backend tests skip before adapter,
   Graph, or kernel construction with the detected tuple in the reason; all backend-independent
   coverage must still run. A new package/runtime/capability requires new evidence and a new exact
   source tuple rather than widening an existing entry.

Remote job IDs, environment-specific paths, commands, logs, and system aliases remain in the local
worklog. The upstream document and GitHub-bound commit contain no internal cluster or host aliases.

## Known risks and deliberate non-goals

- **cuDNN true THD:** the pinned Frontend revision must support zero-padding-free ragged descriptors
  and each rectangular phase on the qualified H100/SM90 and SM100 tuples.
- **cuDNN corrected backward:** the real-arithmetic correction crosses BF16 graph boundaries and is
  supported only after independent tensor-by-tensor evidence covers ordinary and extreme rows.
- **FA4 packaging:** source/distribution identity, import precedence, LSE layout, and `dLSE` remain
  tuple-specific qualification inputs.
- **Memory/compute:** the cuDNN path retains raw local O/LSE and recomputes KV expansion, not SDPA;
  FA4 retains full phase replay. Contiguous phase views and subset merge remove benchmark-path
  gather/scatter temporaries, while general packed layouts retain their indexed fallback. The feature
  claims removal of remote full K/V, not O(1) activation memory or free recomputation.
- **Collectives:** all ranks must construct an identical autograd graph and phase order. The one-hop
  pipeline overlaps only adjacent P2P/attention work; it does not alter the collective topology or
  promise full communication hiding.
- **Projection scope:** the exact local MCore and ordinary TE MLA profiles described above, a
  trainable bias-free output weight, TP=1 non-SP or TP>1 with SP, and no CPU offloading are
  supported. The output helper preserves the inherited module/weight/state dict; only the local
  `RowParallelLinear` profile bypasses its implicit-group forward. Mixed/fused-down/inference
  projection profiles remain unsupported.
- **Recompute/offload:** outer/selective recompute and fine-grained activation offload are rejected,
  rather than assumed safe with nested phase checkpoints.
- **Layout/transport:** padded THD, contiguous input, and A2A+P2P remain behind explicit future
  adapters. Dynamic CP supports only already-zigzag groups initialized by MCore.

No qualification failure may be bypassed with a TE attention fallback, private backend call,
blanket skip, or relaxed unexplained tolerance.
