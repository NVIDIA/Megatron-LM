<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# DeepSeek Sparse Attention

DeepSeek Sparse Attention (DSA), introduced with DeepSeek-V3.2, adds a lightweight *indexer* to
every attention layer. For each query the indexer scores the causal keys, keeps the
`dsa_indexer_topk` highest-scoring ones, and the main attention attends only to those keys. The
indexer is trained with a KL loss against the main attention distribution, or kept frozen.

Megatron Core selects DSA with `experimental_attention_variant: dsa`. All options on this page are
`TransformerConfig` fields. On the command line they use the same names in `--hyphen-case`
(`dsa_indexer_topk_backend` is `--dsa-indexer-topk-backend`); a Boolean field that defaults to
`True` is turned off with `--no-<name>`.

The page first gives an overview of DSA support in Megatron Core, then describes its features one
section at a time. Each feature section is self-contained: what the feature does, its options, the
rules that configuration validation enforces, and who benefits. A dependency matrix, reference
measurements and the known limitations close the page.

## Overview

| Area | What is available | Main options |
| --- | --- | --- |
| Indexer | Standard multi-head indexer (ReLU scoring, optional Hadamard rotation) and a simplified single-head indexer | `dsa_indexer_n_heads`, `dsa_indexer_head_dim`, `dsa_indexer_topk`, `dsa_indexer_mode` |
| Indexer loss | Dense or sparse (selected keys only) KL loss; a coefficient of `0` freezes the indexer projections | `dsa_indexer_loss_coeff`, `dsa_indexer_use_sparse_loss` |
| Kernels | `cudnn`: fused cuDNN frontend indexer, top-k and sparse-attention backward, with the FlashMLA sparse forward. `tilelang`: fused TileLang kernels. `none`: PyTorch fallback. These three are DSA over MLA. `min-memory-triton`, `min-memory-torch` and `reference` are DSA over GQA. | `dsa_kernel_backend` |
| DSA over GQA | Simplified indexer on a GQA attention layer with chunked min-memory kernels, wired into the hybrid stack by `dsa_layer_specs.dsa_stack_spec` (and `gdp_dsa_stack_spec`) | `dsa_indexer_mode: simplified`, `dsa_kernel_backend` |
| Hybrid models | `D` in `hybrid_layer_pattern` builds a DSA-over-MLA layer (DSv4 attention under `dsv4_hybrid`) | `hybrid_layer_pattern` |
| KPool | Selection of softmax-weighted key pools instead of single keys; needs no indexer loss (`dsa_indexer_loss_coeff` 0 or unset) and `dsa_indexer_topk` divisible by the pool size | `dsa_indexer_kpool`, `dsa_indexer_kpool_use_quantization`, `dsa_indexer_kpool_always_select_tail` |
| IndexShare | Cross-layer top-k sharing: layers up to `dsa_indexer_skip_topk_offset` (at least layer 1) compute their own top-k, then every `dsa_indexer_topk_freq`-th layer does, and the layers between reuse the most recent computing layer's indices. Layers are numbered globally, MTP layers after the decoder layers. Sharing does not cross pipeline stages. | `dsa_indexer_topk_freq`, `dsa_indexer_skip_topk_offset` |
| Context parallelism | All-gather CP only (`cp_comm_type: all_gather`), static or dynamic; packed THD sequences with host-built CP layout metadata | `context_parallel_size`, `cp_comm_type`, `dynamic_context_parallel` |

DeepSeek-V4's compact CSA indexer (`experimental_attention_variant: dsv4_hybrid`) shares some of
these options but is a separate path and is not covered here. For DSA with context parallelism see
also [Context Parallelism](context_parallel.md); for repeated MTP layers see
[Multi-Token Prediction](multi_token_prediction.md).

## Features

Unless a section says otherwise, the features below apply to `experimental_attention_variant: dsa`
over MLA with `dsa_kernel_backend: cudnn`. Where an upstream pull request carries a feature, its
section names it.

### Repeated-MTP Sharing of DSA State

Pull requests: [#6472](https://github.com/NVIDIA/Megatron-LM/pull/6472) (sharing) and
[#7107](https://github.com/NVIDIA/Megatron-LM/pull/7107) (activation recompute and full-iteration
CUDA graphs).

**What it does.** With a repeated MTP layer, later MTP depths reuse what depth 0 computed in the
same forward and microbatch: its latent KV, its sparse-attention indices, or both. Queries and
sparse attention still run at every depth. Shared KV stays attached to autograd, so the consumer
gradients accumulate into the depth-0 KV projection. Index sharing skips the indexer, and with it
the indexer loss, at later depths.

Sharing travels through a generic per-forward sharing state
(`megatron/core/transformer/forward_sharing.py`), which also carries ordinary IndexShare indices.
`GPTModel`, `TransformerBlock` and the MTP block scope it to one forward; activation checkpoints
snapshot it per invocation, so each replay sees the payloads of its own forward.

| Option | Default | Values |
| --- | --- | --- |
| `mtp_repeated_layer_shared_components` | `None` (off) | A list of `latent_kv` and/or `sparse_attention_index` |

**Rules enforced by configuration validation.**

- Requires `experimental_attention_variant: dsa`, `mtp_use_repeated_layer: true` and
  `mtp_num_layers > 1`; entries must be known and unique.
- `latent_kv` is rejected with selective recompute that includes `core_attn`. The default selective
  module list is `[core_attn]`, so set `recompute_modules` explicitly (for example `mla_up_proj`) or
  use full recompute.
- Rejected when a per-layer CUDA graph scope captures attention (`cuda_graph_impl` `local` or
  `transformer_engine` with attention in the scope): such a graph would freeze one depth's shared
  state while the producer or a consumer runs outside it. MoE-only scopes are accepted, and so is
  `cuda_graph_impl: full_iteration`, whose single graph contains the producer and every consumer
  depth in their eager order.

**Who benefits.** Models with several repeated MTP depths: later depths skip the KV projection
and/or the indexer. The recompute compatibility list and the interaction with ordinary IndexShare
are in [Multi-Token Prediction](multi_token_prediction.md#repeated-layer-sharing).

### DSA Layers on MTP-Only Pipeline Stages

Pull request: [#8535](https://github.com/NVIDIA/Megatron-LM/pull/8535).

**What it does.** When the last (virtual) pipeline stage holds only MTP layers, its local decoder
block spec is empty, and the MTP spec builders' "no decoder layers" branch falls back to the
standard-attention layer spec. With `experimental_attention_variant` set, both builders take the
MTP layer spec from the global experimental-attention pattern before that branch, so the MTP layer
uses the configured attention variant on every stage.

**Options.** None.

**Who benefits.** DSA models whose pipeline layout puts the MTP layers in a stage of their own
(for example `...|t*8|m*7L`).

### KPool Robustness

Pull request: [#8536](https://github.com/NVIDIA/Megatron-LM/pull/8536).

**What it does.** Two fixes for KPool selection (`dsa_indexer_kpool > 1`):

- A top-k-computing layer that can prove from its layout that every query row selects at least one
  key skips the empty-row masking around sparse attention. KPool selection without
  `dsa_indexer_kpool_always_select_tail` leaves the rows before a query's first complete pool empty,
  which the layout check cannot see, so such a layer does not claim that proof.
- A rank with no query rows (possible with packed context-parallel splits) gets an empty selection
  of the usual width and dtypes.

**Options.** None. Layers without KPool, and KPool with tail selection (the default), are
unchanged.

### KPool Scoring Matching the Hugging Face Reference

**What it does.** Two opt-in options reproduce the arithmetic of the Hugging Face Transformers
KPool indexer (`Glm5NextTextIndexer`, transformers v5.16.1), and explicit-mask pools start where
that reference starts them.

| Option | Default | Effect |
| --- | --- | --- |
| `dsa_indexer_kpool_key_dtype_compression` | `False` | Cast the FP32 pool softmax to the key dtype before the weighted reduction. `False` keeps the FP32 weighted reduction. |
| `dsa_indexer_hf_scoring` | `False` | Upcast the BF16 projected head weights before the head-count scale, scale the FP32 `q @ k` before the ReLU, and reduce heads with an FP32 matmul, for materialized and chunked scores. |

Pool origin (no option): with an explicit attention mask whose padding is not a multiple of the
pool size, pools start at each sample's first valid key, as in the reference, rather than on a
global grid. Otherwise this is an exact identity, so unmasked, varlen, packed and pool-aligned
inputs keep their selections bit for bit. Left padding at the model level remains unsupported,
because causal DSA layers ignore the attention mask.

**Rules enforced by configuration validation.** `dsa_indexer_hf_scoring` requires
`dsa_indexer_kpool > 1`, `dsa_indexer_weights_proj_output_dtype: bf16` and
`dsa_indexer_scoring_relu: true`. Compression and score-input quantization
(`dsa_indexer_kpool_use_quantization`) are independent.

**Who benefits.** KPool models that need numerical parity with the Hugging Face reference.

### Context-Parallel Key Reorder and Prebuilt Packed-CP Layouts

**What it does.** Two improvements for all-gather context parallelism, both on by default.

- *Inverse-permutation backward.* DSA reorders the gathered latent K/V and indexer K into
  full-sequence order with a row permutation. Its backward is a plain gather with the inverse
  permutation, not autograd's index-select backward (a zero fill plus an atomic `index_add` on the
  critical path before the dKV reduce-scatter); results match up to the sign of zero.
- *Prebuilt packed-CP layout tables.* For DSA models, `prepare_packed_seq_params` builds the
  packed-CP layout (query positions, the gathered-key reorder and its inverse) at batch
  construction from the host `cu_seqlens` and attaches it to the `PackedSeqParams`. The fallback,
  a lazy build in the first DSA layer's forward memoized on the per-microbatch `PackedSeqParams`,
  cannot run inside a CUDA graph capture, and a memo entry from an eager warm-up would be baked into
  a graph and go stale on replay. DSA consumes the prebuilt tables when their geometry and device
  match, and warns once per kind of mismatch before falling back to the lazy build.
  `copy_prebuilt_packed_cp_layout_()` refreshes a static `PackedSeqParams` in place for graph
  replay with another packing.

**Options.** None.

**Rules.** Inside a CUDA graph capture, a packed DSA layer with context parallelism raises unless a
matching prebuilt layout is attached; it never builds tables there. Middle pipeline stages that
receive raw boundaries (whose physical row count is unknown at batch construction) detach the
layout and keep the lazy build, which works eagerly only. A graph implementation that refreshes the
tables in place must not do so before the backward of every microbatch that read them has run.
Under dynamic CP the table sizes follow the realized CP size. Configuration validation still warns
that CUDA graph capture of packed DSA with context parallelism is experimental.

**Who benefits.** Long-context packed THD training with all-gather CP, and in particular runs that
capture it in CUDA graphs. The host build runs at batch construction, not in the first DSA layer.

### Checkpoint Top-k Reuse Under Activation Recompute

*This is an opt-in feature.*

**What it does.** Under activation recompute, the first call of a checkpointed top-k-computing
layer runs without grad and skips the indexer loss, so its backward replay reruns the dense indexer
and the top-k only to recover the selected scores for the sparse indexer loss. With
`dsa_indexer_reuse_checkpoint_topk`, the first call keeps the selected FP32 scores next to its
published indices, and the replay computes the sparse indexer loss from them instead of rerunning
the indexer. With a frozen indexer (`dsa_indexer_loss_coeff` 0 or unset) no scores are kept: the
replay takes the indices the first call published and skips the indexer projections, the dense
scorer and the top-k, so it attends exactly the first call's selection.

With the indexer loss on, the stash costs an FP32 `[batch, sequence, dsa_indexer_topk]` tensor per
top-k-computing layer per in-flight microbatch. A missing stash falls back to recomputation with a
per-rank warning; a key-length or row mismatch, or a checkpoint that runs a layer twice, raises.

| Option | Default | Values |
| --- | --- | --- |
| `dsa_indexer_reuse_checkpoint_topk` | `False` (opt-in) | `True` / `False` |

**Rules enforced by configuration validation.**

- Requires `dsa_kernel_backend: cudnn`, the fused DSA kernels (`attention_backend` other than
  `unfused`) and IndexShare (`dsa_indexer_topk_freq > 1`), through whose sharing state the indices
  are reused.
- With the indexer loss on, requires `dsa_indexer_use_sparse_loss: true`.
- Not supported with KPool selection (`dsa_indexer_kpool > 1`).
- For a frozen indexer, rejected with quantized indexer projections
  (`dsa_indexer_qk_proj_use_quantization` or `dsa_indexer_weights_proj_use_quantization`) under
  `fp8_recipe` `delayed` or `custom` or `fp4_recipe` `custom`: the replay skips those projections,
  so Transformer Engine's delayed-scaling recompute stash would never be consumed.
- Rejected with full recompute and `overlap_moe_expert_parallel_comm`, whose schedule replays
  layers without the checkpoint wrapper.
- Takes effect only with full recompute or selective recompute of `core_attn`; other recompute
  settings leave it inert.

**Who benefits.** Long-context runs with full recompute, where the replayed dense indexer is
expensive. The DeepGEMM scorer requires it under such recompute (see below).

### IndexShare Pipeline-Layout Validation and Shared Non-Empty Certificate

**What it does.**

- *Layout validation.* A layer that reuses top-k needs its source layer earlier in the same
  (pipeline rank, virtual stage) chunk. With `dsa_indexer_topk_freq > 1` and a custom
  `pipeline_model_parallel_layout`, configuration validation checks every chunk when the layout
  is parsed: decoder layers are numbered in execution order, MTP layers continue from
  `num_layers + 1`, and each chunk's first DSA layer (its first decoder layer, or the first MTP layer
  of an MTP-only chunk) must compute its own top-k. A violating layout raises a `ValueError` naming
  the chunk, the layer and its source, on every rank and without a GPU, instead of failing at model
  build or the first forward on the ranks that own the chunk. For example, with 78 decoder layers,
  7 MTP layers, `dsa_indexer_skip_topk_offset: 3` and `dsa_indexer_topk_freq: 4`, the layout
  `Et*22|t*24|t*24|t*8m*7L` is valid (its chunks start at layers 1, 23, 47 and 71), and stage
  boundaries can only move in steps of four layers.
- *Non-empty certificate.* A top-k-computing layer that proves every query row selects at least
  one key skips the empty-row masking around sparse attention. It publishes that certificate
  next to its indices, so the layers that reuse them (IndexShare, repeated-MTP sharing, and the
  frozen replay of checkpoint top-k reuse) skip the masking as well. Consumers receive the
  producer's tensors, so results are bitwise identical. A KPool producer without tail selection
  publishes no certificate.

**Options.** None. Pipeline splits without a custom `pipeline_model_parallel_layout` (even splits,
or `num_layers_in_first_pipeline_stage` / `num_layers_in_last_pipeline_stage`) are not checked at
parse time; they keep the model-build checks.

**Who benefits.** IndexShare models with custom pipeline layouts, and every IndexShare consumer
layer.

### cuDNN Frontend 1.26 Call-Site Semantics

Pull request: [#8534](https://github.com/NVIDIA/Megatron-LM/pull/8534).

**What it does.** cuDNN frontend 1.26
([NVIDIA/cudnn-frontend#316](https://github.com/NVIDIA/cudnn-frontend/pull/316)) changed two
defaults that the DSA call sites rely on:

- `indexer_backward_wrapper` treats top-k ids as local to each batch unless
  `topk_indices_global=True`. Megatron Core's sparse-loss path passes global ids, so the call sets
  `topk_indices_global=True` whenever the wrapper accepts it; without it, BSHD micro-batches with
  more than one sequence would lose or misplace the sparse indexer gradient of every batch after
  the first.
- The dense indexer forward, dense attention score recompute and dense indexer backward kernels
  default to a top-left causal offset instead of the bottom-right alignment. Megatron Core's BSHD
  call sites, including the single-sequence packed CP scorer, rely on bottom-right alignment, so on
  frontend 1.26 and newer the call passes the bottom-right `q_causal_offsets` explicitly when
  the offset is non-zero. Older frontends and square calls keep their exact call; a top-left
  frontend whose wrapper lacks `q_causal_offsets` raises.

**Options.** None.

**Who benefits.** Every `cudnn` DSA run on cuDNN frontend 1.26 or newer, including the 1.29.0 pin
of Megatron Core's `dev` extra.

### Deterministic cuDNN DSA

**What it does.** With `deterministic_mode: true` (or
`torch.use_deterministic_algorithms(True)`) the `cudnn` DSA path runs deterministic top-k,
indexer-backward and sparse-attention-backward implementations, so repeated runs are bitwise
identical:

- The top-k membership is canonical: every key above the cutoff plus the lowest-index copies of
  the cutoff value. `+inf` is selectable, NaN and `-inf` never are.
- The indexer backward and the sparse-attention backward run fixed-order implementations; the
  cuDNN deterministic sparse-attention backward keeps one workspace per device and shape for eager
  and graph execution.

*Fast canonical top-k.* With cuDNN frontend 1.31 or newer, whose `indexer_top_k_wrapper` names
`tie_break`, the canonical set comes from one `tie_break=1` launch per row chunk at the
non-deterministic chunk size, sorted by index. Zero cutoffs, where the radix kernel ranks `+0.0`
above `-0.0`, are repaired on just those rows by a Triton kernel. The path keeps fixed shapes and
needs no host synchronization, so it can be captured in a CUDA graph. Older frontends keep the
full-row canonicalization in smaller chunks, which is much slower at long context. Which path runs
is logged once (a warning for the full-row path and for missing Triton).

**Options.** `deterministic_mode` (default `False`).

**Rules.** The deterministic fused sparse-attention backward needs an SM 10.x device, a frontend
whose `sparse_attention_backward_wrapper` accepts `deterministic` and `workspace`, and 16, 32, 64,
96 or 128 attention heads; otherwise that layer uses the PyTorch absorbed sparse attention.
Deterministic calls never use the DeepSelect top-k, and the DeepGEMM scorer rejects
`deterministic_mode`.

**Who benefits.** Bitwise run-to-run comparisons and numerics debugging, including at long
context.

### cuDNN Top-k Launch Sizing and Scratch Budget

**What it does.**

- The cuDNN radix top-k allocates an int32 scratch of `rows x 2 x columns` per launch, so Megatron
  Core splits the rows of a top-k call into launches under a scratch budget. Launches are also
  capped at the frontend's limit of `2**31` scratch elements (8 GiB), and chunk starts stay 32-byte
  aligned.
- Each launch writes into `[rows, top-k]` outputs allocated before the first launch, and its result
  is dropped before the next launch, so a CUDA graph pool holds one scratch-sized block. Results
  kept for a final concatenation would make later launches carve their outputs out of the scratch
  block just freed and need a second scratch-sized block.
- Every cuDNN DSA entry point Megatron Core calls is resolved once, followed by one garbage
  collection. The frontend imports its CuTe DSL kernel modules on first attribute access. Under
  CUTLASS DSL 4.6.2 each kernel decoration leaves a self-referencing frame whose `f_back` chain
  reaches the frame that triggered the import, so an import inside a training step would pin that
  step's activations and gradients until the next cyclic collection.

| Option | Default | Effect |
| --- | --- | --- |
| `dsa_cudnn_topk_scratch_gib` | `2.0` | Scratch budget, in GiB, per cuDNN top-k launch; must be positive and finite |

**Trade-off.** A call's full scratch is twice the size of its FP32 score buffer, and the call takes
the smaller of that and the budget as transient memory, which must be free eagerly and inside CUDA
graph pools. A larger budget therefore gives fewer, fuller launches for more memory. At 1M keys per
row the default 2 GiB gives 256 rows per launch, and 8 GiB gives 1024 rows per launch, exactly the
frontend's per-launch element limit; budgets above 8 GiB do not enlarge launches. Budgets above
2 GiB change only the calls that hand the top-k a whole score buffer: the general packed THD scorer
(cuDNN scores or DeepGEMM logits) and the dense indexer loss (`dsa_indexer_use_sparse_loss: false`
with the loss on). The plain causal, unfused bounds and single-sequence packed CP paths score in
chunks of at most 1 GiB of FP32 scores, whose top-k the 2 GiB default already runs in one launch.
The budget bounds every cuDNN top-k launch of the DSA path, deterministic ones included (the
deterministic full-row fallback launches still smaller chunks), except that a launch always takes
at least one 32-byte-aligned row group (up to 8 FP32 or 16 BF16 rows). The DeepSelect top-k needs
no scratch, and the DeepSeek-V4 CSA paths size their own launches.

**Who benefits.** Long-context runs (wide score rows) whose top-k runs on cuDNN and that have
memory headroom.

### Skipping the `-inf` Score Prefill

**What it does.** cuDNN frontend fills the whole dense indexer score buffer with `-inf` before every
scorer launch, although on the general packed THD path every reader consumes only the columns the
kernel writes. When the installed frontend's `indexer_forward_wrapper` names a `clean_logits`
parameter and the frontend is 1.29 or newer (whose varlen radix top-k carries the out-of-bounds
lane guard of [NVIDIA/cudnn-frontend#410](https://github.com/NVIDIA/cudnn-frontend/pull/410)), the
general packed THD scorer passes `clean_logits=False` and derives each row's column limit on the
device, so that no reader goes past it. Every other scorer path keeps the prefill: the dense
indexer loss reads whole rows and the BSHD paths rely on it for masking.

| Option | Default | Effect |
| --- | --- | --- |
| `dsa_indexer_skip_masked_score_fill` | `True` | Skip the prefill where supported; `--no-dsa-indexer-skip-masked-score-fill` keeps it, for A/B runs |

**Rules.** No released cuDNN frontend has `clean_logits` yet; it is proposed in
[NVIDIA/cudnn-frontend#1761](https://github.com/NVIDIA/cudnn-frontend/pull/1761). Until a release
carries it, the option is inert: the prefill stays, and that is logged once. KPool selection and the
DeepGEMM scorer do not use this scorer, so the option has no effect there.

**Who benefits.** Long-context packed THD runs that score with the cuDNN scorer, where the prefill
writes the whole score buffer on every call.

### MXFP8 Indexer Precision

*This is an experimental, opt-in feature: MXFP8 changes which keys are selected, and it is not yet
validated for training quality.*

**What it does.** `dsa_indexer_precision: mxfp8` quantizes the indexer Q and K to MXFP8 (E4M3 with
E8M0 1x32 scales) for the cuDNN dense scorer on the general packed THD path. MXFP8 decides only
which keys are selected: the indexer weights, loss and backward stay BF16, and with the indexer
loss on, the scores of the selected keys that feed the sparse KL (and that checkpoint top-k reuse
stashes) are recomputed in FP32 from the BF16 indexer inputs by cuDNN's
`sparse_indexer_score_recompute_wrapper`. Given the selection, the KL gradient is therefore the BF16
path's. The KL value is too, except that a selected key whose indexer probability falls below the
smallest normal FP32 number (a logit more than 87.3 below the log-sum-exp of the row's selected
logits) contributes at most about 87.3 nats, where the BF16 path's term keeps growing; compare
both arms of a loss A/B with the same clamp. Frozen-indexer and selection-only calls never
recompute.

| Option | Default | Values |
| --- | --- | --- |
| `dsa_indexer_precision` | `bf16` | `bf16`, `mxfp8` |

**Rules enforced by configuration validation** (for `dsa`): `dsa_kernel_backend: cudnn`, the fused
DSA kernels (`attention_backend` other than `unfused`), `dsa_indexer_head_dim: 128` with 32 or 64
indexer heads, `dsa_indexer_use_sparse_loss: true`, `dsa_indexer_weights_proj_output_dtype: bf16`,
and no KPool selection (`dsa_indexer_kpool > 1` bypasses the dense scorer; use
`dsa_indexer_kpool_use_quantization` for quantized KPool scoring).

**Rules checked at the first scorer call.** An SM100 or newer device, a cuDNN frontend with the
MXFP8 dense scorer (1.29 or newer), Transformer Engine MXFP8 quantization and Triton. One error
lists every missing piece. Every other scorer path, and every fallback to the BF16 reference
indexer, raises instead of silently scoring in BF16, so an A/B run never reports BF16 numerics or
throughput as MXFP8.

In the [reference measurements](#reference-measurements), MXFP8 selection (through the DeepGEMM
scorer) reached recall@2048 of 0.94-0.98 against BF16 on real rows.

**Who benefits.** Long-context runs where indexer scoring is a large share of the step, once
selection quality has been validated for the model (for example with a loss A/B against `bf16`).

### DeepSelect Top-k Backend

*This is an experimental, opt-in feature. It depends on an external build that has run on SM 10.3
(GB300) only.*

**What it does.** With `dsa_indexer_topk_backend: deepselect`, an eligible non-deterministic indexer
top-k runs as one [DeepSelect](https://github.com/deepseek-ai/DeepSelect) launch over all rows
instead of the scratch-bounded cuDNN radix top-k. It reads FP32 scores (the cuDNN scorers) and BF16
scores (the DeepGEMM scorer). Every top-k caller of the `cudnn` path takes the backend from the
configuration. The output contract equals the cuDNN wrapper's (int32 indices in unspecified slot
order, the selected values in the scores' dtype, short rows padded with `-1` / `-inf`); only which
equal scores fill the last slots may differ, which cuDNN leaves unspecified too. DeepSelect is
deterministic run to run but does not keep the lowest-index ties, so deterministic mode never uses
it. The launch needs no scratch and no host synchronization, and is CUDA graph safe.

Two builds are accepted, the first that imports wins: `ds_tune`, DeepSelect `bfa4507` with tuning
patches for top-k 2048, and upstream `deep_select`. A call uses DeepSelect only when all of the
following hold; any other call runs the cuDNN top-k, and the outcome is logged once per reason and
score dtype (a warning when the backend is unavailable, an info line otherwise):

- a build imports and exposes the arguments used, on an SM 10.3 device (the builds also ship
  SM 10.0 kernels, which have not been validated, so SM 10.0 keeps cuDNN with a warning);
- FP32 or BF16 scores with unit column stride, a 16-byte aligned start and a row stride that is a
  multiple of 1024 bytes (where needed, the packed THD scorer pads its FP32 score rows for
  DeepSelect when cuDNN frontend 1.31 or newer can write into the padded buffer directly);
- top-k at most 4096, a multiple of 8 for FP32 and of 16 for BF16, smaller than the row width, with
  a compiled configuration for the score dtype;
- a row width below `2**23` and at least the build's threshold below, where it measured faster
  than cuDNN on GB300 (the thresholds affect speed only, never results):

| Build | FP32 scores | BF16 scores |
| --- | --- | --- |
| `ds_tune` | 65536 columns; 81920 for calls of more than 4096 rows | 16384 columns |
| `deep_select` | 524288 columns | 524288 columns |

NaN rows: a row longer than top-k that sees a NaN score selects its first top-k in-row columns with
NaN values. With the sparse indexer loss (selected scores kept) that row ends with no key; without
it, the row attends to those first columns. The cuDNN top-k instead spends one slot on the NaN and
keeps top-k - 1 real keys. Both are garbage-in rows.

| Option | Default | Values |
| --- | --- | --- |
| `dsa_indexer_topk_backend` | `cudnn` | `cudnn`, `deepselect` |

**Rules enforced by configuration validation.** `deepselect` requires
`experimental_attention_variant: dsa` and `dsa_kernel_backend: cudnn`. KPool selection does not run
this top-k, so the option has no effect there.

**Who benefits.** Long-context runs on GB300, whose wide score rows make the cuDNN top-k a large
share of the indexer time. A `ds_tune` build is compiled against one PyTorch version and does not
import under another; that also falls back with a warning, so check the log for "DSA indexer top-k
uses DeepSelect".

### DeepGEMM MXFP8 Selection Scorer

*This is an experimental, opt-in feature: it selects with MXFP8 (see
[MXFP8 Indexer Precision](#mxfp8-indexer-precision)) and depends on an external build that has
run on SM 10.3 (GB300) only.*

**What it does.** With `dsa_indexer_scorer_backend: deepgemm` and `dsa_indexer_precision: mxfp8`,
the general packed THD path (any CP size) scores with
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)'s `fp8_fp4_mqa_logits` instead of the cuDNN
MXFP8 dense scorer. Each packed query row attends to a contiguous range of the original, globally
ordered K, which DeepGEMM takes directly as per-row key ranges, so no duplicated-prefix key pool is
built. The BF16 logits (half the size of the FP32 scores) are read in place by the DeepSelect BF16
top-k, the intended pairing, or by cuDNN's BF16 radix top-k on calls DeepSelect declines or with
`dsa_indexer_topk_backend: cudnn`. Calls larger than one DeepGEMM call takes are split by rows. The
indexer loss never uses DeepGEMM's values: as for any MXFP8 scorer, the selected keys' scores are
recomputed in FP32 from the BF16 inputs, so this option changes only which keys are selected.

| Option | Default | Values |
| --- | --- | --- |
| `dsa_indexer_scorer_backend` | `cudnn` | `cudnn`, `deepgemm` |

**Rules enforced by configuration validation.**

- Requires `experimental_attention_variant: dsa`, `dsa_kernel_backend: cudnn` and
  `dsa_indexer_precision: mxfp8`, with every MXFP8 requirement.
- Rejected with `deterministic_mode`.
- With full recompute, or selective recompute that includes `core_attn` (the default module list),
  requires `dsa_indexer_reuse_checkpoint_topk` (which in turn requires `dsa_indexer_topk_freq > 1`),
  with the indexer loss on or frozen. A replay that selects again from BF16 logits could pick other
  tied keys than the first call (cuDNN's BF16 radix top-k, which takes every call DeepSelect
  declines, does not repeat its choice among ties), and the recomputed attention gradients must
  belong to the first call's output. Selective recompute of other modules (for example `layernorm`)
  does not replay the indexer and is accepted.

**Fallback.** A call uses the cuDNN MXFP8 scorer instead, logged once per reason (a warning, or an
info line for determinism), when: `deep_gemm` is missing, older than 2.8.1 or fails to import (for
example a build for another PyTorch); the device is not SM 10.3 (SM 10.0 is not validated yet) or
not the process's DeepGEMM device; heads, head dim or dtypes do not fit (32 or 64 indexer heads of
dim 128, BF16 weights); Transformer Engine MXFP8 is missing; the cuDNN frontend is older than 1.29
(DeepGEMM leaves the logits past each row's length unwritten, and the radix top-k never reads them
only from 1.29 on); `torch.use_deterministic_algorithms` is on; or DeepGEMM's JIT warm-up fails. A
CUDA out-of-memory error propagates instead. The confirmation "DSA indexer scorer uses DeepGEMM"
logs at info level.

**Operational notes.**

- A fallback scores in FP32, so its score buffer is twice DeepGEMM's BF16 one; memory budgeted for
  DeepGEMM may not hold it. A fallback on some ranks only (for example one node without the build)
  would select different keys on different tensor-parallel ranks, so install the build on every
  node.
- DeepGEMM JIT-compiles its kernels at the first eager call per device and head count. Set
  `DG_JIT_CACHE_DIR` to a persistent directory shared by all ranks (pre-populated with
  `dsa_deepgemm_scorer.warmup()`), keep `nvcc` available at run time, and run at least one eager
  step before CUDA graph capture (`cuda_graph_warmup_steps >= 1`); capturing a cold kernel raises.
- One process runs DeepGEMM on one device, the first that calls it, because its JIT keeps one
  kernel handle per process. Megatron Core runs one device per rank, so this does not trigger in
  training; call `warmup()` after `torch.cuda.set_device`.
- Batch-invariant mode also uses `deep_gemm` when it is importable, so put a DeepGEMM build on
  `PYTHONPATH` only for runs that use this scorer.

**Who benefits.** Long-context runs on GB300 with MXFP8 selection, paired with the DeepSelect
top-k.

## Configuration Rules

The cross-option rules that configuration validation (`TransformerConfig.__post_init__`) enforces
for the features above:

| Option | Requires | Rejected with |
| --- | --- | --- |
| `mtp_repeated_layer_shared_components` | `experimental_attention_variant: dsa`, `mtp_use_repeated_layer`, `mtp_num_layers > 1` | `latent_kv` with selective `core_attn` recompute; per-layer CUDA graph scopes that capture attention (`full_iteration` is accepted) |
| `pipeline_model_parallel_layout` with `dsa_indexer_topk_freq > 1` | every chunk starts on a top-k-computing layer | - |
| `dsa_indexer_hf_scoring` | `dsa_indexer_kpool > 1`, BF16 weights projection output, ReLU scoring | - |
| `dsa_indexer_reuse_checkpoint_topk` | `dsa_kernel_backend: cudnn`, fused kernels, `dsa_indexer_topk_freq > 1`; the sparse indexer loss when the loss is on | KPool; quantized indexer projections under delayed-scaling recipes (frozen indexer); full recompute with `overlap_moe_expert_parallel_comm` |
| `dsa_indexer_precision: mxfp8` | `dsa_kernel_backend: cudnn`, fused kernels, indexer head dim 128 with 32 or 64 heads, sparse indexer loss, BF16 indexer weights | KPool |
| `dsa_indexer_topk_backend: deepselect` | `experimental_attention_variant: dsa`, `dsa_kernel_backend: cudnn` | - |
| `dsa_indexer_scorer_backend: deepgemm` | `experimental_attention_variant: dsa`, `dsa_kernel_backend: cudnn`, `dsa_indexer_precision: mxfp8`; `dsa_indexer_reuse_checkpoint_topk` under full or `core_attn` recompute (which in turn requires `dsa_indexer_topk_freq > 1`) | `deterministic_mode` |
| KPool (`dsa_indexer_kpool > 1`) | no indexer loss (`dsa_indexer_loss_coeff` 0 or unset), `dsa_indexer_topk` divisible by the pool size | `dsa_indexer_precision: mxfp8`, `dsa_indexer_reuse_checkpoint_topk` |

At run time, a CUDA graph capture of packed DSA with context parallelism requires the prebuilt
packed-CP layout tables (see
[Context-Parallel Key Reorder and Prebuilt Packed-CP Layouts](#context-parallel-key-reorder-and-prebuilt-packed-cp-layouts)).

## Dependency Matrix

| Dependency | Version | Used for | Older or missing |
| --- | --- | --- | --- |
| cuDNN frontend | 1.26 or newer | Explicit bottom-right causal offsets and global sparse top-k ids | Frontends up to 1.25 keep the historical calls |
| cuDNN frontend | 1.29 or newer (the `dev` extra pins `nvidia-cudnn-frontend[cutedsl]==1.29.0`) | MXFP8 dense scorer; the radix top-k out-of-bounds lane guard that the prefill skip and the DeepGEMM scorer need | MXFP8 precision raises; DeepGEMM falls back to the cuDNN MXFP8 scorer, which then raises; the prefill stays |
| cuDNN frontend | 1.31 or newer | `tie_break=1` for the fast deterministic top-k; direct binding of row-padded score buffers for DeepSelect | Deterministic mode uses the full-row canonicalization (warning); widths that need a padded buffer keep the cuDNN top-k |
| cuDNN frontend | `clean_logits` ([NVIDIA/cudnn-frontend#1761](https://github.com/NVIDIA/cudnn-frontend/pull/1761), not in a release yet) | Skipping the `-inf` score prefill | The prefill stays (logged once) |
| CUTLASS DSL | Installed with the frontend's `cutedsl` extra (measured with 4.6.2) | cuDNN frontend CuTe DSL kernels | The frontend cannot run its DSA kernels. 4.6.2 retains caller frames on kernel import, so Megatron Core resolves every entry point once, outside the training step |
| Transformer Engine | MXFP8 quantization | MXFP8 precision, DeepGEMM scorer inputs | MXFP8 raises; DeepGEMM falls back |
| Triton | - | MXFP8 scale packing, DeepSelect NaN-row repair, deterministic zero-tie repair | MXFP8 raises; the repairs use PyTorch implementations |
| DeepSelect | Upstream [`deep_select`](https://github.com/deepseek-ai/DeepSelect), or the `ds_tune` build (`0.1.0+bfa4507.p3`: DeepSelect `bfa4507` with three tuning patches; sources and build script at [buptzyb/DeepSelect@ee633040be](https://github.com/buptzyb/DeepSelect/commit/ee633040be536d3a2b590c5ceb9fad200ba537bd)); SM 10.3 | DeepSelect top-k | The cuDNN top-k runs (warning) |
| DeepGEMM | 2.8.1 or newer, installed from source (measured with `2.8.2+5081f85`); SM 10.3; `nvcc` at run time | DeepGEMM scorer | The cuDNN MXFP8 scorer runs (warning). The `batch_invariant` dependency group pins 2.8.0, which falls back |

Neither DeepSelect nor DeepGEMM is a hard dependency. The DSA paths import them at their first call;
batch-invariant kernels also import `deep_gemm`, when present, at Megatron import time.
Both are experimental and used from untagged builds, identified by commit in their local version;
both have been validated on GB300 only.

## Reference Measurements

The measurements below are taken from the public GLM-5.2 integration tracker
[#6944](https://github.com/NVIDIA/Megatron-LM/pull/6944) (body as of head `edebb6b2d9`), which runs
the features on this page together with other model features. They are end-to-end numbers for that
one configuration, not per-feature speedups, and they make no convergence claim.

**Setup.** 256 GB300 GPUs. GLM-5.2 with 78 decoder layers (3 dense, 75 MoE with 256 experts),
TP1/PP4/EP64, packed THD with dynamic CP (effective CP 1, 32 and 64 at 4K, 128K and 1M tokens),
repeated MTP with 7 depths sharing latent KV and sparse-attention indices, IndexShare every fourth
layer, full-iteration CUDA graphs, mock data with forced-balance routing. Every run sets
`dsa_indexer_precision: mxfp8`, `dsa_indexer_scorer_backend: deepgemm` and
`dsa_indexer_topk_backend: deepselect`; the 1M runs (full recompute) also set
`dsa_indexer_reuse_checkpoint_topk`. Runtime: cuDNN frontend 1.31.0 with the `clean_logits` opt-out
of [NVIDIA/cudnn-frontend#1761](https://github.com/NVIDIA/cudnn-frontend/pull/1761) applied (not
exercised, see below), CUTLASS DSL 4.6.2, DeepSelect `bfa4507` with tuning patches, DeepGEMM
`2.8.2+5081f85`. Each length runs with the indexer trained (`dsa_indexer_loss_coeff: 1e-2`) and
frozen (`0`). Medians over steps 13-32 of one run per configuration:

| Sequence | Indexer | Median step (s) | Tokens/s/GPU |
| --- | --- | ---: | ---: |
| 4K | trained | 13.505 | 2,350.5 |
| 4K | frozen | 12.593 | 2,520.8 |
| 128K | trained | 15.728 | 2,018.3 |
| 128K | frozen | 14.504 | 2,188.6 |
| 1M | trained | 25.023 | 1,268.6 |
| 1M | frozen | 23.580 | 1,346.2 |

**Observations stated in the tracker.**

- In a 128K stage ablation (one allocation, stages added in a fixed order, so each delta depends on
  the stages before it), adding the prebuilt packed-CP layout tables together with the
  full-iteration CUDA graph took the step from 27.553 s to 19.428 s (-29.49%, which also removes
  eager-only host costs). Adding the inverse-permutation key-reorder backward, the shared non-empty
  certificate, the MXFP8 indexer with the DeepGEMM scorer, the DeepSelect top-k and dev's
  invalid-slot handling in the sparse indexer backward (#6166, disabled in the earlier stages) took
  it from 19.428 s to 16.735 s (-13.86%).
- Freezing the indexer shortened the step by 0.912 s (6.76%) at 4K, 1.224 s (7.78%) at 128K and
  1.443 s (5.77%) at 1M.
- The MXFP8 DeepGEMM scorer selects different keys from BF16: recall@2048 was 0.94-0.98 on real
  rows.
- DeepSelect took the top-k at 128K and 1M. At 4K every row is narrower than the BF16 threshold,
  so the cuDNN top-k ran, as designed.
- The `clean_logits` frontend patch in the runtime was not exercised: DeepGEMM scored every call.

## Known Limitations and Open Items

- **Overlapping upstream work.** [#6793](https://github.com/NVIDIA/Megatron-LM/pull/6793) (main)
  isolates the non-packed IndexShare holder per activation-checkpointed forward with a private
  carrier; the forward sharing state of #6472 and #7107 snapshots the same state per checkpoint
  invocation. The two approaches need to be reconciled when both reach the same branch.
  [#7813](https://github.com/NVIDIA/Megatron-LM/pull/7813) (dev, draft) exposes configurable cuDNN
  top-k tie breaking and overlaps the `tie_break=1` deterministic top-k.
  [#8403](https://github.com/NVIDIA/Megatron-LM/pull/8403) (main, draft) changes the same cuDNN
  indexer call sites and adds low-precision indexer scoring, and
  [#8154](https://github.com/NVIDIA/Megatron-LM/pull/8154) (main) passes its own causal offsets in
  the score-chunk function that the frontend 1.26 adaptation also changes.
- **Scope of the forward sharing state.** #6472 replaces the per-layer IndexShare holder with the
  forward sharing state and opens its per-forward lifetime in `GPTModel`, `TransformerBlock` and the
  MTP block. `HybridModel` does not open one. Downstream code that overrides
  `DSAttention._get_index_share_topk_holder`, which #6472 removes (Megatron-Bridge does), needs to
  move to the forward sharing state.
- **Repeated-MTP sharing under CUDA graphs.** The `full_iteration` exemption assumes one graph that
  contains the whole step; if chunk-wise CUDA graphs
  ([#5258](https://github.com/NVIDIA/Megatron-LM/pull/5258)) land, it needs to be revisited for
  them.
- **Experimental kernels.** MXFP8 selection has not been validated for training quality. DeepSelect
  and DeepGEMM have run on SM 10.3 only and are used from untagged builds; SM 10.0 falls back until
  their GPU tests pass there.
- **Prefill skip.** Inert until a cuDNN frontend release carries `clean_logits`.
- **Prebuilt packed-CP layouts.** Wiring the tables into a graph implementation that changes the
  packing between replays is left to that implementation, which must refresh a slot only after the
  backward of every microbatch that read it has run.
- **Deterministic mode.** The fast top-k needs cuDNN frontend 1.31 or newer, and the fused
  deterministic sparse-attention backward is limited to the head counts listed above.
