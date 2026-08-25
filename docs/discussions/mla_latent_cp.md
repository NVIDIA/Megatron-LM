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
in the feature-owned module, while narrow config and model-spec plumbing makes the feature directly
selectable by GPT and HybridModel builders. Transformer Engine (TE),
`megatron/core/transformer/multi_latent_attention.py`, and all existing MLA tests remain unchanged.

P2P context parallelism (CP) circulates normalized MLA latent KV and the positional key component,
rather than expanded K and V. Each receiving CP rank reconstructs K/V immediately before its
attention phase. The phase is checkpointed, so remote full K/V are temporary and are recomputed in
backward.

## Goals and initial scope

The first version will:

- support training-only MLA self-attention in THD format;
- support static P2P CP with the zigzag partition;
- require sequence parallelism whenever tensor parallelism is greater than one;
- support the unfused `rope` and `yarn` rotary modes;
- select a direct attention adapter with the existing
  `TransformerConfig.attention_backend`: `AttnBackend.fused` selects the cuDNN Frontend Graph API
  and `AttnBackend.flash` selects FlashAttention-4 (FA4);
- communicate latent KV plus the shared rotated RoPE key, never head-expanded K/V;
- preserve existing MLA projection parameter names and the output contract;
- use only public PyTorch, cuDNN Frontend, FA4, and MCore APIs at the attention boundary; and
- expose narrow layout and transport interfaces for later contiguous-to-zigzag and A2A+P2P work.

The first version intentionally does not support inference/KV cache, arbitrary masks, dropout,
FP16, FP8/FP4, CUDA graphs, dynamic CP groups, padded THD storage, fused MLA RoPE, GQA, outer layer
recompute, fine-grained activation offload, CPU offloading, frozen output-projection weights, or
the absorbed-MLA algorithm. It does not change standard
attention, the legacy MLA path, TE, or TE's CP implementation.

## Existing code reused, and the dependency boundary

The new module subclasses `MLASelfAttention` from
`megatron/core/transformer/multi_latent_attention.py` only to reuse projection-module construction,
the rotary embedding module, checkpoint names, and sharded-state-dict behavior. It
retains the inherited `linear_proj` module and parameter objects but
does not call `RowParallelLinear.forward`, whose current implementation passes a null TP group to its
linear primitive. It overrides `forward`, stops KV projection before `linear_kv_up_proj`, and never
calls inherited `get_query_key_value_tensors` or `_run_core_attention`, because those functions
construct full K/V before the current CP wrapper.

The new module uses a short, new-file-local `_latent_cp_down_projection` helper and the public MCore
`apply_rotary_pos_emb` export. It must not call the inherited `_qkv_down_projection`: that protected
helper gathers Q with an omitted group and can therefore resolve the global tensor-parallel group.
The local down-projection helper calls the inherited projection *modules*, not any inherited
projection helper, and passes the injected TP group to every mapping operation. For TP greater than
one, the complete compressed Q/KV outputs are scattered over the sequence dimension before their
norms. The rotated shared positional key is scattered into the same per-TP sequence lane. A second
local `_explicit_output_projection` helper applies the inherited row-sharded weight through the
public `linear_with_grad_accumulation_and_async_allreduce` export with the injected TP group and
then calls the public sequence reduce-scatter (or the ordinary row reduction for TP=1). No private
TE symbol is imported.

The construction helper replaces the base MLA `core_attention` submodule with `IdentityOp`.
`MLASelfAttention.__init__` therefore retains projection construction and state-dict keys without
constructing TE `DotProductAttention`; the overridden forward never executes the identity module.

The following existing MCore surfaces are used because they are exported classes/functions or form
the current module-spec/config contract:

| Dependency | Source | Reason for use |
| --- | --- | --- |
| `MLASelfAttention`, `MLASelfAttentionSubmodules` | `megatron/core/transformer/multi_latent_attention.py` | Projection and checkpoint compatibility; no legacy attention execution. |
| `AttnBackend` | `megatron/core/transformer/enums.py` | Existing backend selection contract. |
| `PackedSeqParams` | `megatron/core/packed_seq_params.py` | Existing THD metadata contract. |
| `ProcessGroupCollection` | `megatron/core/process_groups_config.py` | Explicit CP/TP group injection. |
| `ModuleSpec`, `IdentityOp` | `megatron/core/transformer/spec_utils.py`, `megatron/core/transformer/identity_op.py` | Explicit opt-in construction without a legacy attention wrapper. |
| `apply_rotary_pos_emb` | `megatron/core/models/common/embeddings/__init__.py` | Existing differentiable packed MLA RoPE behavior. |
| `gather_from_tensor_model_parallel_region` | `megatron/core/tensor_parallel/mappings.py` | Public autograd-aware Q/KV output gather, always called with the injected TP group. |
| `scatter_to_sequence_parallel_region` | `megatron/core/tensor_parallel/mappings.py` | Public first-dimension scatter for complete Q/KV latents and rotated K-RoPE, always called with the injected TP group. |
| `gather_from_sequence_parallel_region` | `megatron/core/tensor_parallel/mappings.py` | Public first-dimension gather for a received K-RoPE shard; its backward reduce-scatter sums shared-key gradients across TP heads. |
| `linear_with_grad_accumulation_and_async_allreduce` | `megatron/core/tensor_parallel/__init__.py`, `layers.py` | Public exported linear autograd primitive; preserves local weight-gradient/`main_grad` accumulation semantics and receives the injected TP group. |
| `reduce_scatter_to_sequence_parallel_region`, `reduce_from_tensor_model_parallel_region` | `megatron/core/tensor_parallel/__init__.py`, `mappings.py` | Public row-parallel output reductions for TP>1 SP and TP=1 respectively, always called with the injected TP group. |
| `CpPartitionModeConverter` and related exports | `megatron/core/context_parallel_layout/__init__.py` | Future layout adapter only; v1 validates an already-zigzag input. |

No new code imports or reads a process group from `parallel_state`. The caller injects a populated
`ProcessGroupCollection`; rank, size, and peer global ranks are derived from its `cp` and `tp`
groups with public `torch.distributed` APIs and an explicit group argument. The base constructor
receives the same collection and constructs every accepted Column/RowParallel projection with
`tp_group=pg_collection.tp`; v1 validates each projection's stored `tp_group` against that injected
object before forward. Every collective launched by the new module or transport likewise names
`pg_collection.tp` or `pg_collection.cp`; omission is a test failure, not a fallback.

### Supported projection spec and exact TP path

V1 supports TP, but deliberately supports only the non-fused MCore MLA projection spec returned by
`get_gpt_layer_local_spec(..., multi_latent_attention=True, qk_layernorm=True,
normalization="RMSNorm")`. The accepted submodules are:

- `linear_q_down_proj`, `linear_q_up_proj`, `linear_kv_down_proj`, and
  `linear_kv_up_proj`: `megatron.core.tensor_parallel.layers.ColumnParallelLinear`;
- `linear_proj`: `megatron.core.tensor_parallel.layers.RowParallelLinear`; and
- standalone MCore RMSNorm Q/KV norms returned by the local spec, not a norm fused into an
  up-projection.

V1 requires a nonzero `q_lora_rank`. It rejects `FusedMLASelfAttention`, inference-optimized specs,
`TELinear`, `TEColumnParallelLinear`, `TELayerNormColumnParallelLinear`, fused down projections, and
an identity/fused KV norm. Supporting those projection variants later requires its own parity matrix;
they are not accepted merely because the parent class can construct them.

TP=1 uses the ordinary non-SP path. TP size `N>1` requires `sequence_parallel=True`, so each TP
rank enters with a contiguous `[T_r/N, 1, hidden_size]` sequence shard. The exact TP>1 path is:

1. The sequence-parallel `linear_q_down_proj` and `linear_kv_down_proj` first gather the input
   sequence and produce last-dimension shards
   `[T_r, 1, q_lora_rank/N]` and `[T_r, 1, (C+D_r)/N]`.
2. `_latent_cp_down_projection` invokes both projection modules directly. When the resulting last
   dimensions are sharded, it gathers Q and KV with
   `gather_from_tensor_model_parallel_region(tensor, group=self.pg_collection.tp)`. It never calls
   `_qkv_down_projection` and never omits `group`. The complete compressed tensors are then split
   over their first dimension with `scatter_to_sequence_parallel_region(..., group=TP)` before Q/KV
   norm. Thus each norm consumes `[T_r/N, q_lora_rank]` or `[T_r/N, C]`, as in ordinary SP.

3. Sequence-parallel `linear_q_up_proj` gathers the normalized Q shards and produces the full local
   query rows `[T_r, H_tp, D_qk]`. K-RoPE is applied once to full owner-local rows, then scattered
   over the sequence dimension. Each TP lane communicates only its corresponding
   `[T_r/N, C+D_r]` latent/RoPE shard around the CP ring.
4. For every received shard, sequence-parallel `linear_kv_up_proj` gathers latent KV over TP before
   producing local K-content/V heads. The module independently gathers the received K-RoPE shard
   with `gather_from_sequence_parallel_region(..., tensor_parallel_output_grad=True, group=TP)`.
   It then applies the phase KV indices to both full-row tensors. The gather's backward
   reduce-scatter is the single TP sum for the shared positional-key gradient; the KV up-projection
   supplies the corresponding latent-input reduce-scatter.
5. The merged `[T_r, H_tp, D_v]` output is flattened, cast once to BF16, and passed to the
   new-file-local `_explicit_output_projection`. That helper reuses `linear_proj.weight` without
   replacing the module or changing state-dict/sharded-state-dict names. It calls the public MCore
   linear primitive with `bias=None`, the stored `gradient_accumulation_fusion` flag,
   `allreduce_dgrad=False`, `sequence_parallel=False`, and `tp_group=self.pg_collection.tp`; it then
   calls `reduce_scatter_to_sequence_parallel_region(..., group=TP)`. The returned layer output is
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
| `D_r` | `config.qk_pos_emb_head_dim`, RoPE dimension. |
| `D_qk` | Backend Q/K head dimension, `D_c + D_r`. |
| `D_v` | `config.v_head_dim`. |

For the targeted MLA shape, `D_c=128`, `D_r=64`, `D_qk=192`, and `D_v=128`. V1 allows any head
count divisible by TP, but requires those head dimensions and `H_q=H_kv`.

After the exact TP path above, the module forms:

- normalized compressed KV `Z_kv`: `[T_r/N, C]` per TP lane;
- raw shared positional key `K_rope_raw`: `[T_r, D_r]`;
- query `Q`: `[T_r, H_tp, D_qk]`, after Q up-projection and Q RoPE; and
- rotated shared positional key `K_rope`: `[T_r/N, D_r]` per TP lane, after owner-local RoPE and
  sequence scatter.

The exact per-lane communicated payload is

```text
Z = concat(Z_kv, K_rope, dim=-1)       # [T_r/N, C + D_r], BF16
```

`K_rope` is required: current MLA expands it over heads and concatenates it with the content key.
Communicating only `Z_kv` leaves a remote rank without the positional key. RoPE is applied on the
owner before communication so the receiver needs no position metadata. After each ring hop, the TP
group gathers the `N` sequence shards before phase indexing. The K-RoPE gather uses
`tensor_parallel_output_grad=True`, so its backward reduce-scatter sums the shared-key gradients
from all TP-local head shards. Sequence-parallel `linear_kv_up_proj` provides the same ownership
semantics for the latent branch.

For a received payload, the phase callable performs:

```text
KV_content = linear_kv_up_proj(sequence_shard(Z_kv))
           -> [T_r, H_tp, D_c + D_v]
K_rope = gather_from_sequence_parallel_region(sequence_shard(K_rope), group=TP)
       -> [T_r, D_r]
K_content, V = split(KV_content, [D_c, D_v], dim=-1)
K = concat(K_content, expand_heads(K_rope), dim=-1)
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

### Global packed positions are owner metadata

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

V1 supports `config.rope_type in {"rope", "yarn"}` with `apply_rope_fusion=False`. `rope` uses the
standard MCore frequencies. `yarn` uses the current MCore Yarn parameters and scale, numerically
checked against the pinned official DeepSeek formula. Packed sequences reset positions at every
entry in `cu_global`; zigzag sharding changes storage ownership but not position numbers.

## Zigzag ownership and phase plan

For each packed sequence of global physical length `S`, require `S % (2P) == 0`. Let `L=S/P` be the
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

### Backend phase metadata

Let the original global cumulative lengths be `cu_global=[0,S_0,S_0+S_1,...]`. V1 requires equal
Q/KV metadata and every `S_n` divisible by `2P`. The phase planner derives CUDA int32 metadata:

```text
cu_full = cumulative([S_n / P])       # compact rank-local full side
cu_half = cumulative([S_n / (2P)])    # compact front/back half side
max_full = max(S_n / P)
max_half = max(S_n / (2P))
```

THD packs sequences consecutively and each local sequence is `[F_r,B_r]`. Every derived
cumulative tensor is contiguous `torch.int32` on the same device as `cu_global`. The implementation
passes `dtype=torch.int32` explicitly to `torch.cumsum`; integral `torch.cumsum` otherwise promotes
the result to `torch.int64`, which the public FA4 varlen API rejects. Sequence lengths and tensor
capacity validation bound the cumulative values before this derivation. The planner caches
per-sequence front/back indices and uses:

- diagonal: `cu_seqlens_q=cu_full`, `cu_seqlens_kv=cu_full`;
- lower phase: `cu_seqlens_q=cu_full`, `cu_seqlens_kv=cu_half`; and
- upper phase: `cu_seqlens_q=cu_half`, `cu_seqlens_kv=cu_full`.

These values are backend attention metadata only. They are never used to generate or apply RoPE.

V1 rejects inter-sequence and tail padding. Padded cumulative lengths must be absent or exactly
equal to valid cumulative lengths, and `T_r == cu_full[-1]`. This keeps P2P message sizes identical
and makes all phase descriptors unambiguous.

## Numerical and online-softmax contract

Both kernels consume BF16 Q/K/V. Their raw phase output is BF16 and their LSE/stats are FP32. Each
adapter immediately returns canonical FP32 `O_i` and FP32 `E_i` with shapes
`[T_q,H_tp,D_v]` and `[T_q,H_tp]`.

An upper phase is expanded without in-place writes:

```text
O_i_full = zeros([T_r,H_tp,D_v], FP32).index_copy(0, back_indices, O_i)
E_i_full = full([T_r,H_tp], -inf, FP32).index_copy(0, back_indices, E_i)
```

The method is functional `index_copy`, never `index_copy_` or mutation of a tensor saved for
backward. Merge all phases in FP32:

```text
E = logaddexp(E_a, E_b)
O = O_a * exp(E_a - E) + O_b * exp(E_b - E)
```

After the last phase there is exactly one `O.to(torch.bfloat16)` before reshape and
`linear_proj`. There are no intermediate BF16 merges.

The merge autograd boundary saves each canonical partial `(O_i,E_i)`, the final `(O_global,
E_global)`, and row-validity metadata; it recomputes phase weights in backward. Thus partial outputs
and LSE are deliberately retained in v1 even though expanded K/V are not.

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

`_LatentRingExchange`, local to the new file, is a `torch.autograd.Function` over one payload hop.
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
  TP-reduce-scattered by sequence-parallel `linear_kv_up_proj`; the shared K-RoPE component is
  TP-reduce-scattered by its explicit sequence gather. The earlier sequence scatters then
  all-gather both gradients before owner-local RoPE/KV norm and KV down projection;
- `linear_kv_up_proj` parameter gradients sum its phase uses locally; and
- normal MCore distributed parameter buffers reduce replicated parameter gradients over DP-CP.
  The module does not add another parameter all-reduce.

The constructor requires `pg_collection` and uses its `cp`/`tp` groups. Peers are resolved once with
`torch.distributed.get_process_group_ranks(self.pg_collection.cp)`. Packed metadata cannot replace
the group in static v1.

### Phase checkpoint and saved-state scope

Each phase callable takes a Q view, one latent payload view, and immutable metadata. It performs KV
up-projection, key construction, direct backend attention, and returns canonical FP32 `(O_i,E_i)`.
It is wrapped with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False,
preserve_rng_state=False)`. Dropout is zero.

Checkpoint keeps its Q/latent inputs and returned partial outputs, but discards tensors saved inside
KV expansion and backend attention. Backward re-executes KV up-projection and, in v1, attention
forward before backend backward. Full K/V exist only during that phase call and are never sent or
stored in ring state.

The intentionally retained activation classes are:

- one latent tensor per differentiable ring node, `O(P*T_r*(C+D_r))` elements;
- every canonical partial `O_i` in FP32 and `E_i` in FP32, plus final merge state; and
- local Q/projection inputs required by normal MCore autograd.

No expanded remote K/V is physically retained outside the checkpointed phase. It is absent from
checkpoint inputs/outputs and module state. An outer `saved_tensors_hooks` recorder enumerates the
shape, element count, dtype, Python tensor class, and semantic state class of every tensor physically
packed into the surrounding autograd graph after forward. The pack hook returns a holder while the
recorder keeps only its weak reference, so only holders still owned by live autograd state are
enumerated. It permits the phase Q/latent checkpoint inputs and FP32 partial O/LSE/merge state, while
rejecting expanded remote K/V shapes. A backend
custom autograd function may call `ctx.save_for_backward` inside the phase as part of its ordinary
implementation, but non-reentrant checkpointing replaces those inner saves with replay holders, so
they do not survive as physical K/V storage in the outer graph.

The lifetime test includes a sensitivity control: its test-only native backend passes K/V through a
numerically identity custom autograd sentinel whose `save_for_backward` names the exact THD K/V
state, mirroring the public cuDNN adapter's custom-function contract. It temporarily replaces
checkpoint with direct phase execution and requires the same recorder to detect expanded value and
Q/K-shaped attention state. This avoids depending on private `einsum`/BMM saved-view layouts.
Tensor-wrapper weak references remain a supplemental check only. This is saved-state evidence, not
an allocator-wide memory snapshot claim. A future backend-specific
backward may remove attention-forward recompute or reduce partial-output storage, but that is not a
v1 claim.

To avoid ambiguous nested checkpoint/offload behavior, v1 requires
`recompute_granularity is None`,
`recompute_modules is None or recompute_modules == []`, and
`fine_grained_activation_offloading=False`. It rejects outer full-layer recompute, selective
`mla_up_proj`/`core_attn` recompute, and fine-grained activation offload at construction. Supporting
any of them later requires explicit nested-checkpoint and saved-tensor/offload tests.

### V1 transport, lifetimes, and deadlock ordering

V1 chooses fixed **wait-at-each-hop** execution. It does not overlap P2P with attention. After phase
`i`, every rank submits one public `torch.distributed.batch_isend_irecv` batch of `P2POp` objects
ordered as `[isend(next), irecv(previous)]`. Every `P2POp` sets
`group=self.pg_collection.cp`; the rank waits every returned `Work` before computing phase `i+1`.
Backward submits `[isend(previous), irecv(next)]` with the same explicit CP group and likewise waits
before consuming `dX_r`.

P2P uses the current CUDA stream. Send storage remains live through `Work.wait`; receive storage is a
fresh tensor for the autograd node and is never overwritten before backward. Fixed payload shapes,
identical operation lists, and identical phase counts prevent mismatched-message and parity-order
deadlocks.

All config, metadata, dtype, peer, package, capability, and descriptor/plan checks complete before
the first ring hop. Runtime-tuple resolution, direct-adapter creation, and preparation of every
rank-local phase class execute inside one ordinary-`Exception`-to-status boundary. Every rank then
performs CUDA-int32 `MIN` consensus first on the injected CP group and then on the injected TP group;
singleton dimensions are the trivial consensus and launch no collective. Only after both reductions
succeed may a rank proceed or populate the successful-preflight cache. If either reduction reports
failure, all TP x CP peers raise before P2P and the locally failing rank chains its original error.
A local failure or partial consensus never creates a success entry. Once P2P starts, the module makes
no claim to recover from an arbitrary Python, CUDA, or NCCL exception; failures propagate through
normal PyTorch/NCCL error handling.

`LatentCPTransport` remains an extension seam. A later explicitly configured transport may return a
`PayloadLease` plus readiness event from a communication stream, but no overlap setting or fallback
exists in v1.

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

One cuDNN adapter, public handle, and plan cache are shared by all latent-CP layers in one process on
one CUDA device. Forward/backward graphs are keyed by process/device identity, exact
frontend/runtime versions, dtype and capability, heads/dimensions, phase shape, causal flag, scale,
maximum packed lengths, and metadata capacity. Handle stream mutation, plan construction, and
execution are lock-protected. Workspace tensors and variant packs are invocation-local and are never
adapter-owned or reused across concurrent calls. `generate_stats=True` supplies local FP32 LSE for
the corrected backward.

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
| H100 / SM90 | `AttnBackend.fused` | `1.22.1` | cuDNN `9.21.0` | `4.536757133166702e-05` |
| SM100 | `AttnBackend.fused` | `1.26.0` | cuDNN `9.25.0` | `4.423665134356547e-05` |
| SM100 | `AttnBackend.flash` | `4.0.0b11` | `flash-attn-4==4.0.0b11` | `4.3095951884009054e-05` |

The test file independently spells the exact tuple sequence in
`EXPECTED_QUALIFIED_BACKEND_CONFIGS` and the exact tuple-to-epsilon mapping in
`EXPECTED_QUALIFICATION_EPS`. Tests require exact production/test/source-revision equality, exact
mapping keys, and no extra entry. The epsilon is a parity assertion threshold, not a production
runtime knob.

Qualified full-parity tests do not assign an adapter or mutate the allow-list. They resolve the
installed tuple without constructing an adapter, require exact membership, construct the layer with
empty backend state, and exercise the normal module `forward -> _get_backend` qualification and
CP-then-TP preflight path. A real-backend test on an installed but unqualified tuple skips with the
detected tuple and complete exact allow-list in its reason before any cuDNN Graph or FA4 kernel is
constructed. Production itself remains fail-closed and raises; it never skips, probes numerically,
or falls back. Direct cuDNN mathematical diagnostics use the same resolution gate before obtaining
the public qualified adapter.

Sanitized qualification evidence records the source revision, exact distribution/runtime and
compute capability, configuration/seed, per-tensor cosine and tensor-similarity minima, derived
epsilon, and descriptor/plan status. Environment-specific paths, system names, and raw launch
identifiers remain outside this document.

Runtime constructs the same four-field tuple from the selected backend, exact installed
frontend/package version, exact linked cuDNN runtime or FA4 distribution identity, and
`torch.cuda.get_device_capability()`, then requires exact membership in
`QUALIFIED_BACKEND_CONFIGS`. Missing version metadata fails closed. Runtime additionally checks
dtype/head dimensions, public descriptor validation, support query, and execution-plan
construction/cache lookup for the three phase classes. All plan classes are prepared before P2P.

A separate process-local **successful-preflight cache** removes steady-state consensus cost. Its
rank-common signature contains the exact runtime tuple and device properties, global packed lengths
and maximum, TP/CP sizes, heads/dimensions/scale/backend, and the complete ordered global-rank tuples
for the injected TP rows and CP columns. PID and CUDA device ordinal scope the local entry but do not
enter the cross-rank signature. The ordered grid is resolved once from the injected groups and is
cached with strong group references. The first layer for a new signature prepares its local phase
classes and completes CP-then-TP `MIN`; only that success stores the shared adapter. Later layers and
steps with the same signature reuse its already-prepared process/device adapter with zero consensus
collectives. Failure never writes an entry, and CP1 x TP1 performs neither consensus all-reduce.
Read-only counter snapshots expose first-use, hit, success, failure, group-consensus, and grid-identity
counts for tests; production exposes no reset or mutable override.

This cache records successful deterministic availability/plan preflight, not numerical qualification:
runtime does not execute random/reference inputs, compare numbers, mutate the allow-list tuple, or
provide an opt-in numerical probe.

## Feature-owned architecture and config-driven construction

The algorithm and its dedicated tests live in:

```text
megatron/core/transformer/experimental_attention_variant/mla_with_latent_cp.py
tests/unit_tests/transformer/experimental_attention_variant/test_mla_with_latent_cp.py
```

The production file contains the module, phase planner, FP32 merger, direct adapters,
differentiable synchronous ring transport, no-op zigzag layout adapter, and spec factory.
Small integration changes add `TransformerConfig.mla_latent_cp` and route only GPT/HybridModel
attention slots through that factory; no attention algorithm is moved into an existing MLA file.

`get_gpt_layer_local_spec(...)` returns a whole transformer-layer `ModuleSpec`; the attention
factory does not accept that outer object. `make_mla_with_latent_cp_spec(base_mla_spec)` accepts
exactly `layer_spec.submodules.self_attention`, validates that it is the supported MLA attention
spec, and returns a new `ModuleSpec` whose module is `MLAWithLatentCP`. It constructs a new
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
instead opts in through configuration:

```yaml
multi_latent_attention: true
mla_latent_cp: true
attention_backend: fused
cp_comm_type: p2p
cp_partition_mode: zigzag
```

The argument factory also exposes `--mla-latent-cp`. Ordinary GPT replaces self-attention in both
dense and MoE layers; gated-delta GPT and `HybridModel` replace only their standard-attention (`*`)
slots. Mamba/GDN, D, CSA/HCA, window-attention, MLP, and MoE slot selection stays unchanged. Explicit
layer/stack specs, inference/modelopt, MTP, and other unsupported combinations fail rather than
silently ignoring the flag. `attention_backend` remains the fused-versus-FA4 selector. The manual
factory above remains useful to developers building custom specs.

The reused projection state-dict names stay compatible, while unsupported specs fail during factory
construction rather than halfway through forward.

## Validation and early errors

Construction or pre-collective validation requires:

- `mla_latent_cp == multi_latent_attention == True` when selected through model configuration;
- explicit `ProcessGroupCollection` with usable `cp` and `tp` groups;
- `cp_comm_type == "p2p"` for CP greater than one;
- the exact local-MCore projection/norm spec and nonzero Q LoRA described above;
- `PackedSeqParams`, `qkv_format == "thd"`, and `cp_partition_mode == "zigzag"`;
- static CP: no different packed `cp_group` and no dynamic `local_cp_size`;
- equal CUDA-int32 Q/K cumulative lengths, monotonic self-attention metadata, original global max
  lengths, and every physical sequence divisible by `2P`;
- padded cumulative lengths absent or equal to valid lengths, with local tokens matching
  `cu_full[-1]`;
- causal logical mask, `attention_mask is None`, and self-attention;
- BF16 activations and weights, a trainable bias-free output projection, zero dropout, and
  FP16/FP8/FP4 disabled;
- TP greater than one only with `sequence_parallel=True`; the physical input/output tokens are the
  corresponding first-dimension TP shard while packed metadata retains the pre-SP CP-local count;
- training without inference context/cache, CUDA graph, outer/selective recompute, fine-grained
  activation offload, or CPU offloading;
- `rope_type` equal to `rope` or `yarn`, with fused RoPE disabled;
- heads divisible by TP, `H_q=H_kv`, `D_c+D_r=192`, and `D_v=128`;
- backend exactly `AttnBackend.fused` or `AttnBackend.flash`; and
- a deterministic qualified package/hardware allow-list match and valid descriptors/plans.

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

V1's `P2PRingTransport` yields owner and a ready, synchronously received payload. A future
`HierarchicalA2AP2PTransport` can consume a layout plan and combine contiguous-to-zigzag permutation
with low-level A2A, then expose the same owner order. Its process groups must be injected, for
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
metrics for output, input gradient, and all seven mapped parameter gradients. It reduces their
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
   Require every planner-derived diagonal/lower/upper Q/KV cumulative tensor to be contiguous
   `torch.int32`, catching the default integral-`cumsum` promotion to `torch.int64`.
2. **Global-position parity.** For multi-sequence THD, independently construct original per-sequence
   positions, zigzag-shard them, and compare owner Q/K rotation for both `rope` and `yarn`. Assert
   original `cu_global/max_global` reach RoPE and derived `cu_full/cu_half` reach only backends.
3. **Projection and TP contract.** Assert the accepted spec classes, exact Q/KV last-dimension
   gathers, and three first-dimension scatters (Q latent, KV latent, and rotated K-RoPE), all with
   `group is pg_collection.tp`. TP>1 with SP disabled must fail at construction. Gather the
   per-lane BF16 payloads along the sequence dimension and require the pre-SP token count. For a
   real phase, spy on the explicit K-RoPE sequence gather and require
   `tensor_parallel_output_grad=True`; the obsolete non-SP copy mapping and every default-group
   resolver must raise. Assert every accepted projection stores the injected TP group. For output
   projection, make inherited `RowParallelLinear.forward` and default resolution raise, require the
   explicit TP group on the public linear and sequence reduce-scatter calls, and compare BF16
   forward, dInput, and local weight gradient against independent `F.linear` plus an explicit
   reduce-scatter contract for both gradient-accumulation flag values.
4. **Payload and ring tests.** Assert each forward P2P tensor has `T_r*(C+D_r)` elements, never the
   full-K/V size. For CP=2/4, prove forward owner routing, reverse gradient routing, fixed peer order,
   that every recorded `P2POp` constructor receives `group=pg_collection.cp`, and wait-at-each-hop
   behavior. After construction, patch `parallel_state.get_tensor_model_parallel_group`,
   `get_context_parallel_group`, and `get_tensor_and_context_parallel_group` to raise throughout
   the complete TP=2 x CP=2 production forward/backward; test-harness collectives remain explicitly
   bound to injected groups outside that guard.
5. **Independent multi-rank parity.** Run the pinned `NaiveMLA` reference against TP=2 x CP=2 with
   `H=96,D_qk=192,D_v=128`. Compare output, input gradient, and every mapped parameter gradient with
   both similarity metrics after explicit CP reduction/TP reconstruction. The H100/SM90 fused and
   SM100 fused/FA4 cases use normal production qualification/preflight, assert the exact tuple
   epsilon for every metric, and require the newly observed candidate epsilon not to exceed it.
   Separately construct the unchanged `MLASelfAttention + TEDotProductAttention` P2P CP path with
   identical TP=2 x CP=2 SP-sharded input, weights, and upstream gradient. For rope and YARN, compare
   latent-CP and legacy full-KV-CP output and SP-sharded input gradient directly, then compare every
   parameter gradient after the same explicit CP reduction and TP reconstruction. This accounts for
   latent recomputation and legacy pre-expansion assigning KV-up work to different CP ranks. The
   independent pinned `NaiveMLA` parity remains the backend-independent oracle.
6. **Backend dispatch, qualification, and preflight caches.** Assert fused creates only the direct
   shared cuDNN adapter and flash only FA4; TE `DotProductAttention` is neither built nor called.
   The fake public FA4 callable receives the exact planner-owned int32 tensors by identity; invalid
   Q or KV dtype, contiguity, or device colocation fails before that callable without conversion.
   Assert the production source pin and immutable allow-list equal the independently spelled
   evidence tuples exactly, and assert the independently spelled epsilon mapping has exactly the
   same keys and no extras. An installed unqualified tuple must skip a real-backend test before
   adapter/Graph construction with an exact-tuple reason; a qualified full parity must start with
   empty adapter state and populate it only through normal production preflight. Counter deltas prove
   first-signature CP-then-TP consensus, same-signature zero-collective
   hits, local and remote-only failure-without-insertion, successful retry, custom-group grid
   identity, and no CP1 x TP1 consensus collective. The remote-only case begins locally successful,
   injects failure into CP `MIN`, still requires the following TP `MIN`, and asserts exact
   `[CP, TP]` group/op order.
7. **cuDNN merge backward.** Compare all phase shapes against standard PyTorch, including `G_i`,
   `gE_i`, `O_corr`, zero/tiny norm rows, extreme phase weights/LSE, and BF16 boundary casts.
8. **Dtype and functional merge.** Assert raw BF16 backend output, canonical/merged FP32 output+LSE,
   functional upper scatter, and exactly one final BF16 cast before `_explicit_output_projection`.
9. **Recompute/lifetime evidence.** Count `P` KV up-projections in forward and `P` in checkpoint
   replay. Outer saved-tensor hooks enumerate retained shape/numel/dtype/Python class and classify
   checkpoint Q/latent plus FP32 partial O/LSE state; expanded K/V is forbidden. Weakrefs supplement
   this evidence. A checkpoint-disabled sensitivity control must expose expanded value and
   Q/K-shaped saved state, proving the recorder would catch the regression.
10. **Negative validation.** Cover unsupported projection specs, SBHD, contiguous/A2A modes,
    padding, non-divisible lengths, dynamic CP, non-causal/explicit masks, FP16/FP8/FP4, dropout,
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
   diagnostics and fused `rope`/`yarn` TP=2 x CP=2 full parity through normal production preflight.
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
- **Memory/compute:** v1 deliberately retains FP32 partial O/LSE and recomputes phase attention. It
  claims removal of remote full K/V, not O(1) activation memory or free recomputation.
- **Collectives:** all ranks must construct an identical autograd graph; v1 sacrifices overlap to
  make ordering and lifetime explicit.
- **Projection scope:** only the local MCore Column/RowParallel projection spec, trainable bias-free
  output weight, TP=1 non-SP or TP>1 with SP, and no CPU offloading are supported. The output helper
  deliberately preserves the inherited module/weight/state dict while bypassing only its
  implicit-group forward. TE/fused projection specs are future work.
- **Recompute/offload:** outer/selective recompute and fine-grained activation offload are rejected,
  rather than assumed safe with nested phase checkpoints.
- **Layout/transport:** padded THD, dynamic CP, contiguous input, and A2A+P2P remain behind explicit
  future adapters.

No qualification failure may be bypassed with a TE attention fallback, private backend call,
blanket skip, or relaxed unexplained tolerance.
