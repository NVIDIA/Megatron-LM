# Determinism Op Catalog

The per-operation catalog of determinism in Megatron-Core. For the narrative walk
see [`training-path.md`](./training-path.md); for the control plane and targets see
[`status.md`](./status.md).

## How this catalog is maintained (evidence, not guesses)

Every row is classified by one of three evidence sources. **Do not estimate** perf
numbers — fill the "Perf Δ" column from the PR #5041 nsys leaderboard
(`tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py`)
on a real recipe, and the determinism verdict from one of:

- **doc** — guaranteed by PyTorch / TE / NCCL / cuBLAS documentation (e.g.
  `index_add`, `index_put(accumulate)` are documented deterministic under
  `torch.use_deterministic_algorithms(True)`; `scatter_add` / `F.embedding`
  backward / FA backward are documented non-deterministic).
- **code** — an explicit deterministic branch exists in the source.
- **test** — confirmed by `BitExactRunner` (toggle `deterministic_mode`; same input
  must give bit-identical out+grad). Rows still needing this are marked **⚠ verify**.

Status legend (matches `training-path.md`): 🟢 deterministic · 🔵 has det branch ·
🟡 conditional (verify) · 🔴 gap (no det path).

---

## Control plane

| Item | File:line | Effect |
| --- | --- | --- |
| `--deterministic-mode` (current) | `megatron/training/arguments.py:1499-1508` | asserts no flash-attn, no CE-fusion; validates `NCCL_ALGO`; `torch.use_deterministic_algorithms(True)` |
| `set_determinism_env_vars()` (PR #5041) | `megatron/training/determinism.py` | setdefault `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| `apply_determinism_to_args()` (PR #5041) | `megatron/training/determinism.py` | assert no CE-fusion; validate `NCCL_ALGO` (excl. Tree); force `tp_comm_overlap=False`; permit flash-attn; `use_deterministic_algorithms(True)` |
| `deterministic_mode` config flag | `megatron/core/model_parallel_config.py:153` | threaded into `TransformerConfig`; read by library branches below |

---

## Catalog table

> Columns: **Op** · **File:line** · **Primitive** · **Det?** · **Det path** ·
> **Non-det path** · **Selected by** · **Evidence** · **Perf Δ** · **Gap / TODO**

### MoE (the determinism hot zone)

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Token unpermute (combine) | `transformer/moe/moe_utils.py:513-531` | scatter-accumulate | 🔵 | `index_add_` (CUDA-graph safe) | `scatter_add_` | `are_deterministic_algorithms_enabled()` | doc+code | TBD (leaderboard) | **Hotspot**: `aten::fill_`/`empty`/`index_put`. Target a fused deterministic ScatterAdd (Longcat). |
| Routing map/probs | `moe_utils.py:823-837` | scatter vs index_put | 🔵 | `index_put_(accumulate=False)` ×2 | `scatter` ×2 | `are_deterministic_algorithms_enabled()` | code | TBD | Comment `TODO: element-wise instead of scatter?` |
| Top-k expert select | `moe_utils.py:777` | `torch.topk(sorted=is_grad_enabled())` | 🟡 | `sorted=True` in training pins order | `sorted=False` (inference) | grad-enabled | doc | **Hotspot** (`cub::DeviceRadixSort`, large det/nondet Δ) | Investigate deterministic top-k avoiding radix-sort blowup; check fused TE topk parity. |
| Group-limited top-k mask | `moe_utils.py:617-634` | `scatter_(1, group_idx, 1)` | 🟢 | unique group indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** bit-exact by `test_deepseek_model.py` (DSV3 group routing: `num_groups`/`group_topk`) across EP≤4/TP/FSDP/PP/VPP. ⚠ still unverified at EP>16. |
| Aux-loss routing map | `moe_utils.py:890` | `scatter` | 🟢 | unique indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** via DSV3 proxy (`seq_aux_loss` enabled) — bit-exact. |
| Capacity-drop mask | `moe_utils.py:940-951` | `scatter` | 🟡 | unique indices | — (no branch) | always | — | small | ⚠ verify under capacity factor. |
| Router map (router.py) | `transformer/moe/router.py:261` | `scatter` | 🟡 | unique indices | — | always | — | small | ⚠ verify. |
| Pad routing map | `moe_utils.py:637-669` / `fusions/fused_pad_routing_map.py` | `cumsum` + mask write | 🟢 | ordered cumsum | — | always | doc | small | cumsum is deterministic. |
| Token permute (dispatch) | `moe_utils.py:299-431` | `argsort(stable=True)` + `index_select` / fused TE | 🟢 | stable sort + gather | — | always | doc | — | — |
| EP all-to-all dispatch/combine | `transformer/moe/token_dispatcher.py` | `all_to_all` | 🟢🟡 | fixed NCCL algo | — | env | doc+**test** | — | Bit-exact at EP≤4 (DSV3 + nemotron proxies). ⚠ EP>16 at scale unverified (DSV3 only diverges there). |
| Grouped GEMM (experts) | `extensions/transformer_engine.py` `TEGroupedLinear` | grouped matmul | 🟢🟡 | TE deterministic kernels | TE fast kernels | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | doc+**test** | TBD | Bit-exact in DSV3 + nemotron proxies (`moe_grouped_gemm=True`). wgrad order is a perf target ("optimized grouped GEMM", "fused GemmAdd"). |
| Router replay | `transformer/moe/router_replay.py` | record/replay top-k | 🟢 | replay recorded indices | — | opt-in | code | — | A determinism *tool*, not default path. |

### Attention

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlashAttention backward | TE; `extensions/transformer_engine.py:1697-1703` | atomic dQ/dK/dV accumulation | 🔵🟡 | TE deterministic FA kernels | TE atomic FA | `deterministic_mode` asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | doc+code | TBD (top perf target) | **Deterministic FAG** (Longcat/DSV4): independent accum buffers + global deterministic sum. |
| Core attention fwd | TE `DotProductAttention` | fused attn | 🟢 | — | — | — | doc | — | Forward deterministic. |
| Multi-Latent Attention (MLA) | `transformer/multi_latent_attention.py` (q/kv low-rank proj + YaRN rope) | low-rank GEMMs + rope | 🟢 | — | — | — | doc+**test** | — | **Verified** bit-exact by `test_deepseek_model.py` (DSV3 MLA + qk_layernorm + YaRN) across EP/TP/FSDP/PP/VPP. |
| Attention dropout | `transformer/dot_product_attention.py:114-220` | RNG mask | 🟡 | fixed RNG state | — | RNG | doc | — | Set dropout=0 for hero runs to eliminate. |

### Normalization

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LayerNorm/RMSNorm fwd | `transformer/torch_norm.py`, TE norm | reduction | 🟢 | — | — | — | doc | — | — |
| LayerNorm/RMSNorm bwd | TE / torch norm | weight-grad reduction | 🟡 | TE deterministic kernels | torch/TE fast | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | doc | TBD | ⚠ verify torch-norm fallback backward. |

### Embedding & TP linear

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel embedding | `tensor_parallel/layers.py:299-303` | indexing vs `F.embedding` | 🔵 | direct `weight[idx]` (det bwd) | `F.embedding` (atomic bwd) | `self.deterministic_mode` | doc+code | TBD | Comment: "F.embedding has non-deterministic backward". |
| RoPE / position emb | `models/common/embeddings/rotary_pos_embedding.py` | trig | 🟢 | — | — | — | doc | — | — |
| Column/Row parallel linear (GEMM) | `tensor_parallel/layers.py` | cuBLAS GEMM | 🟡 | fixed cuBLAS workspace | — | `CUBLAS_WORKSPACE_CONFIG` | doc | — | — |
| Async TP all-gather/all-reduce | `tensor_parallel/layers.py:544/565/577`, `mappings.py:452` | `async_op=True` | 🟡 | `wait()` re-imposes order | overlap | `tp_comm_overlap` | doc | — | `tp_comm_overlap` force-off in det mode. |

### Loss

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py:119-156` | 3× all-reduce + fp32 | 🟡 | fixed NCCL algo, fp32 intermediates | — | env | doc | — | Native CE is deterministic under `NCCL_ALGO`. |
| Fused cross-entropy | `cross_entropy_loss_fusion` | fused kernel | 🔴 | — (forbidden) | fused | asserted off | code | — | Open: can it be made deterministic? (perf opportunity). |

### Backward / grad reduction / optimizer

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py:201-231` | NCCL collective | 🟡 | fixed NCCL ring order | — | env | doc | — | bf16 FP non-assoc but reproducible run-to-run. |
| fp32-accum reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` | all-to-all + ordered `sum(fp32)` | 🟢 | ordered fp32 sum | bf16 RS | `ddp_config.reduce_scatter_with_fp32_accumulation` | code | small | Accuracy + determinism friendly; 1-bucket only. |
| Distributed optimizer param order | `optimizer/distrib_optimizer.py:1094` | shard mapping | 🟢 | "preserving deterministic ordering across ranks" | — | always | code | — | — |
| Grad clip global norm | `optimizer/clip_grads.py` | all-reduce | 🟡 | fixed NCCL algo | — | env | doc | — | — |

### SSM / Mamba (hybrid models)

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Triton autotune | `ssm/ops/determinism.py:81-103` | autotune config select | 🔵 | cheapest config | run-all autotune | `use_deterministic_mode()` | code | TBD | Avoids autotune-driven kernel variance. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | `zeros`+`sum` vs `empty` | 🔵 | ordered tile sum | unordered | `use_deterministic_mode()` | code | mem+ | Extra memory for tiled reduction. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | torch vs FLA fused | 🔵 | `torch_chunk_gated_delta_rule` | FLA fused | `deterministic_mode` | code | TBD | — |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | `F.conv1d` vs FLA | 🔵 | `F.conv1d` (+transpose) | `causal_conv1d` | `deterministic_mode` | code | TBD | — |
| Packed sequence (`thd`) | `ssm/gated_delta_net.py:314` | — | 🔴 | — | thd | asserted off | code | — | **Gap**: no deterministic packed-seq SSM path. |

### Inference / RL (not training-loop, listed for completeness)

| Op | File:line | Det? | Notes |
| --- | --- | --- | --- |
| DP inference coordinator scheduling | `inference/.../dynamic_engine.py:607`, `data_parallel_inference_coordinator.py:181` | 🔵 | sorted rank identities vs completion order |
| RL rollout ordering | `rl/rl_utils.py:678` | 🔵 | sort by `problem_id` vs completion order |
| Inference token sampling | `inference/sampling/torch_sampling.py:61-69` | 🟡 | `cumsum`/`scatter` in top-p; RNG-driven |
| DSA sparse-attention masks | `transformer/experimental_attention_variant/dsa.py:214/385/950` | 🟡 | `scatter_` index masks; ⚠ verify for DSV3.x sparse attention |

---

## Hotspots (perf priority — fill Δ from the leaderboard)

From the det-vs-nondet nsys leaderboard, the overhead concentrates here. These are
the Workstream 3 targets, roughly in priority order:

1. **MoE unpermute** `index_add_` path (`moe_utils.py:517`) — `aten::fill_` /
   `aten::empty` / `aten::index_put_` blowup. → fused deterministic ScatterAdd.
2. **MoE top-k** `cub::DeviceRadixSort` (`moe_utils.py:777`) — large det/nondet Δ.
3. **FlashAttention backward** — deterministic FAG (independent accum + global sum).
4. **Grouped GEMM** wgrad — optimized grouped GEMM / fused GemmAdd.
5. **Lift limitations** where feasible — CE fusion, tp_comm_overlap.

## Verification backlog (the ⚠ rows)

Run `BitExactRunner` (PR #5041) with `deterministic_mode` toggled, plus the
`RacingStreams` / `CudaSleepJitter` stressors, to confirm/deny each row.

**Verified (WS2 proxies, draco 8×H100, fw-final pipe.50600619):** the
`test_deepseek_model.py` (MLA + group-limited MoE routing) and
`test_nemotron_hybrid_model.py` (MoE-in-hybrid) proxies pass bit-exact across
EP≤4 / TP / FSDP / PP / VPP (12/12 cells). This promotes: group-limited top-k,
aux-loss routing map, grouped GEMM, EP all-to-all (small EP), MLA, and
MoE-inside-hybrid.

**Still open:** capacity-drop mask, `router.py:261` scatter, EP all-to-all at
**EP>16** (proxies cap at EP4 — needs the Tier-B mbridge e2e recipe to reach the
scale where DSV3 empirically diverges), torch-norm backward fallback, DSA sparse
masks. Promote each to 🟢/🔵 or open a gap with a fix following the
`moe_utils.py:517` det-branch pattern.
