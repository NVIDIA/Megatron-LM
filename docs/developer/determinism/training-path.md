# Determinism Branch Map — A Walk Through One Training Step

This is a forward→backward→optimizer walk through a Megatron training step,
calling out **every place where determinism enters or is decided**. Each entry
links to the file:line and to the row in [`op-catalog.md`](./op-catalog.md). Read
[`status.md`](./status.md) first for the control plane and definitions.

Legend for the "Determinism" column:
- 🟢 deterministic (by formula, or pinned reduction order)
- 🔵 has an explicit deterministic branch (selected by `deterministic_mode` /
  `torch.are_deterministic_algorithms_enabled()`)
- 🟡 deterministic only under a precondition (env var / fixed NCCL algo / unique
  indices) — **verify** at scale
- 🔴 non-deterministic with no in-repo deterministic path (a gap)

---

## Stage 0 — Data & RNG (before the model)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Sample order / shuffle | `megatron/training/datasets/data_samplers.py`, `megatron/core/datasets/` | 🟡 | Reproducible iff seed + data path + DP layout are fixed. Part of the "fixed input" contract, not a kernel concern. |
| Global RNG seeding | `megatron/training/initialize.py`, `tensor_parallel/random.py` (`CudaRNGStatesTracker`) | 🟢 | Per-rank RNG is tracked and restorable — this is exactly what `BitExactRunner` snapshots/restores between its two runs. |
| Dropout masks | attention/MLP dropout layers | 🟡 | Reproducible under fixed RNG state; set dropout=0 for hero runs to remove a source entirely. |

## Stage 1 — Forward

### 1a. Embedding

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Vocab-parallel embedding lookup | `tensor_parallel/layers.py:299-303` | 🔵 | det: direct index `self.weight[masked_input]`; non-det: `F.embedding` (whose **backward** scatters grads with atomics → non-deterministic with duplicate tokens). Branch on `self.deterministic_mode`. |
| Rotary / position embeddings (RoPE) | `models/common/embeddings/rotary_pos_embedding.py` | 🟢 | Pure trig formula, no scatter/atomics. |

### 1b. Attention

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| QKV / output projection (TP linear) | `tensor_parallel/layers.py` (Column/RowParallelLinear), TE linear in `extensions/transformer_engine.py` | 🟡 | GEMM forward is deterministic given fixed cuBLAS workspace (`CUBLAS_WORKSPACE_CONFIG=:4096:8`). TP all-reduce/all-gather reproducible under fixed `NCCL_ALGO`. |
| Core attention (fwd) | TE `DotProductAttention` (`extensions/transformer_engine.py:1697`) | 🔵🟡 | When `deterministic_mode` on, asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` so TE picks deterministic kernels. The **backward** is the real concern (see Stage 3). |
| Softmax / scale / mask | `transformer/dot_product_attention.py` | 🟢 | Elementwise. |

### 1c. Normalization

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| LayerNorm / RMSNorm (fwd) | `transformer/torch_norm.py`, TE norm in `extensions/transformer_engine.py` | 🟢 | Forward reduction is deterministic. Backward weight-grad reduction is the concern (Stage 3). |

### 1d. MLP / MoE (the determinism hot zone)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Router gating + softmax/sigmoid | `transformer/moe/moe_utils.py:789-818` | 🟢 | Computed in fp32; elementwise. |
| Top-k expert selection | `moe_utils.py:777` (`torch.topk(..., sorted=torch.is_grad_enabled())`) | 🟡 | `sorted=True` during training pins order; this `topk` is the source of the `cub::DeviceRadixSort` that shows a large det-vs-nondet delta in profiling (**hotspot**). |
| Group-limited (node-limited) top-k | `moe_utils.py:617-634` (`group_mask.scatter_`) | 🟡 | `scatter_` writes 1s at unique group indices → deterministic in forward; no explicit det branch (**verify**). |
| Routing map / probs construction | `moe_utils.py:823-837` | 🔵 | det: `index_put_(accumulate=False)`; non-det: `scatter`. Also `compute_routing_scores_for_aux_loss:890` and capacity masks `946/951` use `scatter` with **no** det branch (**verify**). |
| Capacity-factor drop | `moe_utils.py:940-951` | 🟡 | `scatter` of capacity mask; unique indices. |
| Token permute (dispatch sort) | `moe_utils.py:299-431` (`argsort(stable=True)` + `index_select`, or fused TE permute) | 🟢 | Stable sort + gather is deterministic. |
| EP all-to-all dispatch | `transformer/moe/token_dispatcher.py` | 🟡 | Collective itself is ordered; reproducible under fixed NCCL algo. Order of tokens is set by (deterministic) routing. |
| Grouped GEMM (expert FFN) | `extensions/transformer_engine.py` `TEGroupedLinear` | 🟡 | Forward deterministic; backward weight-grad accumulation order is the concern + a perf target (Longcat "optimized grouped GEMM"). |
| Token unpermute (combine) | `moe_utils.py:513-531` | 🔵 | det: `index_add_` (CUDA-graph safe); non-det: `scatter_add_`. This is the `aten::fill_`/`empty`/`index_put` **hotspot**. |
| Router replay (optional) | `transformer/moe/router_replay.py` | 🟢 | Records top-k indices once and replays them — forces identical routing across runs (a determinism *tool*, not on the default path). |

### 1e. SSM / Mamba (hybrid models: nemotron-3-ultra)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Triton kernel autotune | `ssm/ops/determinism.py:81-103` | 🔵 | det: pick cheapest config (avoids autotune-driven kernel variance); else run-all autotune. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | 🔵 | det: allocate `zeros(..., tile_dim)` and `.sum(-1)` (ordered reduction); non-det: `empty(...)`. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | 🔵 | det: torch `chunk_gated_delta_rule`; non-det: FLA fused kernel. |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | 🔵 | det: `F.conv1d` (+ transposes); non-det: FLA `causal_conv1d`. |
| Packed sequence (`thd`) | `ssm/gated_delta_net.py:314` | 🔴 | `assert not deterministic_mode` — **no deterministic packed-seq path** (gap). |

## Stage 2 — Loss

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py:119-156` | 🟡 | 3 all-reduces (MAX, SUM, SUM) across TP. Deterministic under fixed `NCCL_ALGO` (ring order pins FP reduction order). fp32 intermediates. |
| Fused CE | `cross_entropy_loss_fusion` | 🔴→forbidden | Non-deterministic; **asserted off** in deterministic mode (`arguments.py:1502`). |
| MoE aux loss | `moe_utils.py:842-890` | 🟡 | `scatter` for routing map; aux-loss scalar reduction. |

## Stage 3 — Backward

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Autograd traversal | PyTorch engine | 🟢 | Order is deterministic for a fixed graph. |
| FlashAttention backward | TE, gated by `NVTE_ALLOW_NONDETERMINISTIC_ALGO` (`extensions/transformer_engine.py:1697`) | 🔵🟡 | **The classic non-determinism source** — atomic dQ/dK/dV accumulation. Deterministic TE path exists but is slower (Longcat "deterministic FAG" = independent accumulation buffers + global deterministic sum) — a top perf target. The 0.1% tolerance comments in `param_and_grad_buffer.py:326/334/345` exist for the *non-deterministic* default mode. |
| LayerNorm/RMSNorm backward | TE / torch norm | 🟡 | Weight-grad reduction across the sequence; deterministic via TE deterministic kernels under the env var. |
| Embedding backward | see 1a | 🔵 | det path = direct-index `index_put(accumulate=True)` (deterministic under `use_deterministic_algorithms`), not `F.embedding`'s atomic scatter. |
| MoE unpermute/permute backward | see 1d | 🔵 | Mirror of forward `index_add_`/`scatter_add_` branch. |
| Grouped-GEMM backward | TE `TEGroupedLinear` | 🟡 | wgrad accumulation order; perf target. |

## Stage 4 — Gradient reduction (DP / FSDP)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py:201-231` | 🟡 | bf16 collective has FP non-associativity but a **fixed NCCL ring order makes it reproducible** run-to-run. |
| fp32-accumulation reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` (enabled via `ddp_config.reduce_scatter_with_fp32_accumulation`) | 🟢 | All-to-all then **ordered `torch.sum(..., dtype=fp32)`** — a deterministic, higher-precision reduction. Primarily an accuracy feature; also determinism-friendly. |
| Async param gather / grad reduce overlap | `param_and_grad_buffer.py:349+`, `tensor_parallel/layers.py:544/565/577` (`async_op=True`) | 🟡 | Async completion order can vary; determinism relies on `wait()` barriers re-imposing order before use. `tp_comm_overlap` is force-disabled in det mode. |

## Stage 5 — Optimizer

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Distributed optimizer param sharding/order | `optimizer/distrib_optimizer.py:1094` | 🟢 | Param→shard mapping "preserves deterministic ordering across ranks". |
| Adam / Muon update | `optimizer/` | 🟢 | Elementwise; deterministic given deterministic grads. |
| Grad clipping (global norm) | `optimizer/clip_grads.py` | 🟡 | Global-norm all-reduce; reproducible under fixed NCCL algo. |

## Cross-cutting (affect every stage)

| Concern | Control | Determinism | Notes |
| --- | --- | --- | --- |
| cuBLAS GEMM workspace | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 🟡 | Required for deterministic GEMM algo selection. |
| NCCL collective algorithm | `NCCL_ALGO=Ring` | 🟡 | Pins reduction topology/order. `Tree` excluded by PR #5041. |
| TE non-deterministic algos | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 🟡 | Forces TE deterministic attention/norm kernels. |
| CUDA caching allocator | (none) | 🟡 | Allocation pattern can influence kernel autotuning/selection; flagged in the roadmap as worth investigating. |
| PP / VPP microbatch interleave | schedules in `core/pipeline_parallel/` | 🟢→🟡 | Schedule is deterministic, but interleaving **scrambles observed event order** — the key reason a naive "first divergence" hook is hard (Workstream 4). |

---

## How to read this against the catalog

Each 🟡/🔴 above is a candidate for either a **test** (Workstream 2 — does it stay
bit-exact at scale?) or an **optimization** (Workstream 3 — is its deterministic
path the perf bottleneck?). The 🔵 rows already have a deterministic branch and are
the *template* for adding new ones (follow the `moe_utils.py:517` pattern:
`if torch.are_deterministic_algorithms_enabled(): <det> else: <fast>`).
