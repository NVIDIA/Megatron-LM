# QSA (Qwen Sparse Attention)

Experimental attention variant (`experimental_attention_variant="qsa"`, hybrid layer
symbol `Q`) implementing the block-sparse GQA attention of Qwen3.8-Flash-Next
(HF `model_type: qwen4_exp`). Reference semantics: HF `modeling_qwen4_exp.py`
(`Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextAttention`); the pure-torch parity oracle lives in
`tests/unit_tests/transformer/experimental_attention_variant/qsa_reference.py`.

## Structure

- **Main attention** is a stock `SelfAttention` composed from existing knobs:
  GQA (`num_query_groups`), `attention_output_gate` (query|gate interleaved per head in
  `q_proj`), `rotary_percent` (partial RoPE), `qk_layernorm`, explicit `kv_channels`.
  Qwen3.8-Flash-Next: 24 q / 2 kv heads, head_dim 256, rotary 0.25.
- **Indexer** (`QSAIndexer`): `index_qk_proj: hidden -> (n_heads + 1) * head_dim`
  (4 MQA query heads + 1 shared key head, dim 128), per-head q/k RMSNorm, the main
  attention's rotary frequencies applied to queries at token positions and to
  **parameter-free fp32 mean-pooled** 4-token block keys at each block's first-token
  position, `relu(q@k).sum(heads)/sqrt(head_dim)` scoring in fp32, **deterministic**
  top-`budget/compress_ratio` block selection under the `(score desc, block_id asc)`
  lexicographic order, and unconditional retention of the ragged causal tail. The
  indexer consumes the post-input-layernorm hidden states, its input is detached from
  the trunk, and its parameters are replicated across TP
  (`average_gradients_across_tp_domain`) and kept out of FP8 (`index_qk_proj` runs
  under an FP8-disabled context).
- Blocking is **causal-prefix aligned** (fixed 4-token grid from the sequence — or,
  under packed THD, from each document's — start). This is provably equivalent to the
  HF per-query visible-prefix blocking for unpadded causal inputs; note the
  HF-faithful caveat that at positions where `(t+1) % 4 == 0` the query's own block is
  not force-kept and must win the top-k.

## Config

| Knob | Default | Meaning |
|---|---|---|
| `qsa_indexer_n_heads` | 4 | indexer query heads (MQA) |
| `qsa_indexer_head_dim` | 128 | indexer head dim (q heads and the shared key head) |
| `qsa_indexer_budget` | 2048 | token budget; selected blocks = budget / compress_ratio |
| `qsa_indexer_compress_ratio` | 4 | block size in tokens |
| `qsa_indexer_loss_coeff` | None | sparse-KL indexer distillation loss coefficient |
| `qsa_use_sparse_attention` | False | sparse Triton kernel path vs the dense-mask bridge |
| `qsa_cp_packing_layout` | per_document | how packed THD batches are zigzag-sharded across CP ranks (`per_document` / `per_sequence`); a caller contract that cannot be inferred from cu_seqlens |

Sequences no longer than the budget degenerate to dense causal attention — tests that
exercise selection must use per-document lengths > budget (production budget 2048 →
seq > 2.5k).

## Execution paths

1. **Dense-mask bridge** (default): the selection is materialized as a `[b, s, s]` bool
   mask and fed to dense core attention with `AttnMaskType.arbitrary`. Functional
   reference; no sparsity speedup; BSHD only.
2. **Sparse kernel path** (`qsa_use_sparse_attention=True`): self-contained Triton
   kernels (`experimental_attention_variant/ops/triton_qsa.py`, SGLang-prefill-style
   design): a fused index-score kernel (scoring + visibility + int64 sort-key
   encoding, no `[s, chunk, heads]` intermediate), the deterministic int64-key
   `torch.topk` streaming merge, and **query-tile shared-superset gather** sparse GQA
   forward/backward — each program serves `BQ` consecutive queries against the UNION
   of their selected blocks with per-query membership bits, so gathered K/V tiles are
   reused across the tile. `dk`/`dv` run one program per 4-token block over a
   per-query CSR (sequential, no atomics — deterministic gradients). Any GQA group
   size works (no head padding); packed THD is encoded as per-block token
   `[base, end)` ranges, so the kernels never see `cu_seqlens`.
   **Performance status (GB300, bf16, 24q/2kv/d256, budget 2048)**: the shared-superset
   win scales with selection *locality*. Kernel-level, with window-local selections
   (the trained-indexer regime) the forward reaches parity with dense flash at seq 32k
   (12.6 vs 11.5 ms) and is **1.5x faster at 64k** (25.0 vs 37.5 ms); with fully
   random selections (zero locality, the untrained floor) it is ~2.3x slower than the
   window case and loses to dense. End-to-end with an untrained indexer (random-like
   selections) the module is still slower than dense at 16k–64k, so the path stays
   **off by default**; backward remains gather-bound (~0.7x dense at 64k even with
   locality) — per-block query-run reuse in dkv is the identified follow-up. See the
   artifact report in the development repo for the measured tables.

## Indexer training (sparse KL)

`qsa_indexer_loss_coeff > 0` enables block-granularity sparse KL distillation
(tech-report Eq. 17/20): the teacher is the main-attention distribution over the
**selected blocks' tokens only** (bounded memory), computed from detached re-projected
q/k in fp32, softmaxed per head, summed over heads with a TP all-reduce, **max-pooled**
to blocks (Eq. 17 — a block is worth its most salient token; the max is nonlinear, so
the TP reduce happens on token-level probs *before* pooling) and L1-normalized over the
selected set (the report's token-level L1 is a per-query scalar absorbed by this final
normalization); the student is the masked log-softmax of the indexer's fp32 block
scores on the same set. The loss rides the `DSAIndexerLossAutoScaler` aux-loss pipeline
(loss-scale hook in `schedules.py` covers `'qsa'`) and the DSA indexer-loss logging
tracker. Deterministic selection makes recompute replay bit-identical (verified: full
recompute vs none trains bit-identically in deterministic mode).

## Hybrid usage

`Q` is a standard-attention-family symbol (NOT in `Symbols.MLA_ATTENTION` — it can
coexist with `*`). Wire via
`--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_qsa_stack_spec`; the
`'Q' -> qsa` variant inference sets `experimental_attention_variant` from the pattern.
Qwen3.8-Flash-Next slice: `GEGEGEQE` (3x GDN+MoE, 1x QSA+MoE). MTP segments may use `Q`
(`.../QE`), giving the QSA+MoE MTP head. Uniform (non-hybrid) GPT models select the
variant directly via `--experimental-attention-variant qsa`.

## Checkpoint mapping (HF `qwen4_exp` -> Megatron), per QSA layer

Let `h = hidden_size (2560)`, `np = 24`, `ng = 2`, `d = head_dim (256)`,
`hpg = np / ng = 12`, `di = indexer head_dim (128)`, `ni = indexer heads (4)`.

| HF tensor | Shape | Megatron destination and layout rule |
|---|---|---|
| `q_proj.weight` | `[np*d*2, h]` | Rows are packed **per head** as `[query(d), gate(d)]` chunks (HF views `[np, 2d]` and chunks in half). Megatron `linear_qkv.weight` packs **per query group** as `[Q(hpg*d), GATE(hpg*d), K(d), V(d)]`. For group g, head j (global head i = g*hpg + j): Megatron Q rows `<- q_proj[ i*2d : i*2d+d ]`; Megatron GATE rows `<- q_proj[ i*2d+d : (i+1)*2d ]`. |
| `k_proj.weight` | `[ng*d, h]` | group g's K rows `<- k_proj[ g*d : (g+1)*d ]` |
| `v_proj.weight` | `[ng*d, h]` | group g's V rows `<- v_proj[ g*d : (g+1)*d ]` |
| `o_proj.weight` | `[h, np*d]` | `linear_proj.weight`, direct copy (column order = head-major, matches) |
| `q_norm.weight` / `k_norm.weight` | `[d]` | `q_layernorm.weight` / `k_layernorm.weight`. HF RMSNorm is zero-centered (`1 + w`); use `layernorm_zero_centered_gamma=True` and copy `w` verbatim. |
| `indexer.index_qk_proj.weight` | `[(ni+1)*di, h]` | `indexer.index_qk_proj.weight`, direct copy — rows split `[ni*di | di]` into query heads and the shared key head exactly as in HF (`torch.split`). |
| `indexer.q_layernorm.weight` / `indexer.k_layernorm.weight` | `[di]` | `indexer.q_layernorm.weight` / `indexer.k_layernorm.weight` (zero-centered, copy verbatim) |

Under TP, `linear_qkv`/`linear_proj` shard per query group as any GQA attention; the
indexer parameters are replicated on every TP rank (copy the full tensor to each).
Distributed checkpointing: all QSA parameters ride the standard sharded-state-dict
machinery (`linear_qkv` group-sharded; indexer tensors replicated) — save/reload
across a parallelism change (TP1PP1 -> TP2/PP2) is covered by the QSA parallel
consistency tests.

## Context parallelism (allgather CP)

`context_parallel_size > 1` is supported on the sparse kernel path with
`cp_comm_type='all_gather'` (ring/p2p attention is not implemented; the bridge is
single-rank only). Design (adapted from the community QSA implementation's CP
blueprint): queries stay local at their zigzag positions; the indexer all-gathers
only the single shared raw index-key head (differentiable — backward is a
reduce-scatter), reorders it to global token order and pools 4-token blocks
strictly *after* the reorder (pooling rank-local neighbours would combine
non-adjacent tokens); the attention all-gathers K/V the same way. The kernels take
local queries against global K/V (`TQ != TK`) with an explicit per-row
`q_positions` array for causal masking. All RoPE under CP is applied at explicit
positions by indexing the frequency table (layout-exact; this deliberately bypasses
the THD+CP rope helper, whose per-document assumption is wrong for per-sequence
packing). Packed THD requires declaring the zigzag via `qsa_cp_packing_layout`
(see `dsa_layout.CPPackingLayout`); both `per_document` and `per_sequence` are
supported and CP2-tested against the oracle.

## Limitations

- Pretrain / continue-pretrain only: no inference or KV-cache path (asserted).
- CP is allgather-only (K/V memory per rank is O(global sequence) for QSA layers);
  `cp_comm_type` must be `all_gather` or unset (QSA performs its own gathers and
  ignores the knob; an explicit p2p/a2a value is rejected at config validation, as
  is CP with the dense-mask bridge). TPxCP composition is untested.
- The dense-mask bridge is BSHD-only; packed THD requires the sparse kernel path.
- Padded batches are not supported: bool attention masks are OR-merged into the bridge
  mask but assumed causal/per-document; the sparse path derives causality from the
  selection and ignores bool masks entirely (pad tokens would be attended and
  selectable). Additive float masks are rejected on both paths. Use unpadded BSHD or
  packed THD inputs.
