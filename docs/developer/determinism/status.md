# Determinism in Megatron-LM — Current Status

> **Audience:** Megatron-LM / Megatron-Core developers working on deterministic
> training. For the *user-facing* "how do I turn it on" guide, see
> `docs/user-guide/deterministic-training.md` (added by PR #5041).
>
> **Companion docs in this directory**
> - [`training-path.md`](./training-path.md) — the forward+backward branch map:
>   every point in the training step where the code (or an underlying kernel)
>   chooses a deterministic vs non-deterministic path.
> - [`op-catalog.md`](./op-catalog.md) — the per-op catalog table (det? / det path /
>   non-det path / how selected / perf Δ / gap).

---

## 1. What "deterministic" means here

**Bitwise determinism:** given a fixed input and seed, the model yields an
*identical* loss/grad curve across every run, with everything else held fixed —
input data + order + seed, model recipe + config, parallelism layout
(TP/PP/DP/EP/CP/VPP), container image, library versions (MCore, CUDA, cuDNN,
NCCL, Transformer Engine, PyTorch, driver), NCCL env/runtime settings, and
hardware type + cluster topology.

The definition is about **numerical results, not the computation graph.** What is
explicitly *allowed* to vary between runs:

- The computation graph may differ slightly if final numerics are bit-identical.
- Execution ordering may change, **as long as reduction orderings are the same**
  (floating-point addition is non-associative — this is the root cause of nearly
  all non-determinism; see §2).
- Kernel scheduling / implementation may vary if outputs/grads are unchanged.
- Deterministic mode may select different (slower / more memory-hungry) kernels or
  fallback paths than non-deterministic mode — this is the design we use.

## 2. Why it is hard

The primary enemy is **floating-point non-associativity**: `(a+b)+c ≠ a+(b+c)` in
bf16/fp16/fp32. Any kernel whose result depends on the *order* of a parallel
reduction (atomic adds, multi-block reductions, async collectives whose
completion order varies, allocator-driven kernel autotuning) can produce
different round-off between runs. Determinism therefore means pinning reduction
order everywhere it matters.

## 3. Targets and motivation

Customers are moving determinism from a debug/CI nicety to a **production / hero-run
requirement** (deterministic replay of loss spikes, checkpoint-resume that follows
the original trajectory, and a proof point that the GPU stack produces correct
math — mitigating SDC/SDE concerns). The blocker to enabling it by default is the
performance penalty.

| Milestone | Determinism overhead (step time) |
| --- | --- |
| Current baseline (mcore) | **~15%** overall; Nemotron-3-Ultra **+17.4%**, DSV3 **+8.5%** |
| Adoption target | **< 10%** (would let us default `deterministic_mode=True`) |
| Aggressive target | **≈ 5%** (per Meituan Longcat: deterministic FAG + ScatterAdd + grouped GEMM + fused GemmAdd) |

Profiling attributes most of the gap to a handful of ops — see
[`op-catalog.md`](./op-catalog.md) "Hotspots". The headline offenders from the
det-vs-nondet nsys leaderboard are `aten::fill_`, `aten::empty`,
`aten::index_put_`, `scatter_add`, and `cub::DeviceRadixSort` (MoE top-k path).

## 4. Control plane — how determinism is turned on

### 4.1 Today (on `main`)

`--deterministic-mode` is handled inline in `validate_args`
(`megatron/training/arguments.py:1499-1508`):

1. Asserts `not use_flash_attn` — **flash-attn is forbidden** today.
2. Asserts `not cross_entropy_loss_fusion` — fused CE is non-deterministic.
3. Validates `NCCL_ALGO ∈ {Tree, Ring, CollnetDirect, CollnetChain, ^NVLS}`
   (the env var must already be exported by the launcher).
4. Calls `torch.use_deterministic_algorithms(True)`.

It does **not** set env vars for you and does **not** force `tp_comm_overlap` off.

The config flag is `model_parallel_config.py:153` `deterministic_mode: bool = False`,
threaded into `TransformerConfig`; library code reads either that flag or
`torch.are_deterministic_algorithms_enabled()`.

### 4.2 After PR #5041 (`megatron/training/determinism.py`)

PR #5041 extracts the logic into a reusable module (mirroring Megatron-Bridge) so
tests and profiling scripts can opt in without an `args` Namespace:

- **`set_determinism_env_vars()`** — `os.environ.setdefault` of:
  `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Uses `setdefault` so a launcher-exported
  value wins (these are captured by NCCL/cuBLAS/TE at *first use*, so they must be
  set before the first kernel).
- **`apply_determinism_to_args(args)`** — asserts CE fusion off, validates
  `NCCL_ALGO` (**now excludes `Tree`** — its reduction order is not
  user-controllable), forces `tp_comm_overlap=False` (with a `warn_rank_0`),
  calls `torch.use_deterministic_algorithms(True)`.

Two behavior changes vs. today worth flagging:

- **Flash-attn is now permitted** under `--deterministic-mode`. TE FlashAttention
  is deterministic on supported configs when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`;
  the bit-exact suite covers it. (Old code rejected it outright.)
- **`tp_comm_overlap` is force-disabled** rather than left to the user.

## 5. Enforced limitations in deterministic mode

These features are currently **incompatible** with deterministic mode:

| Feature | Where enforced | Reason |
| --- | --- | --- |
| `cross_entropy_loss_fusion` | `arguments.py:1502` / `determinism.py` assert | Fused CE kernel is non-deterministic |
| `tp_comm_overlap` (async TP) | `determinism.py` (forces off) | Async NCCL collective ordering varies |
| Flash-attn (today only) | `arguments.py:1501` assert | Relaxed by PR #5041 (TE FA is deterministic) |
| Packed sequence (`thd`) in gated-delta-net | `ssm/gated_delta_net.py:314` assert | No deterministic packed-seq SSM path |

> **Open question (tracked in the roadmap docs):** it is not fully established
> *whether* CE-fusion and tp_comm_overlap can be made deterministic. Resolving
> this is part of Workstream 3 (perf) — lifting a limitation is often cheaper than
> a slow fallback.

## 6. Surface area — the det/non-det branches that exist

The kernel-level determinism surface in `megatron/core` is **small**: there are
only ~5 `torch.are_deterministic_algorithms_enabled()` call sites plus a handful
of `config.deterministic_mode` branches. They are fully enumerated in
[`op-catalog.md`](./op-catalog.md); the load-bearing ones:

- **MoE unpermute** (`moe_utils.py:517`): `index_add_` (det, CUDA-graph safe) vs
  `scatter_add_` (fast).
- **MoE routing map/probs** (`moe_utils.py:823`): `index_put_(accumulate=False)`
  vs `scatter`.
- **Vocab embedding fwd** (`tensor_parallel/layers.py:299`): direct `weight[idx]`
  (det backward) vs `F.embedding` (non-det backward).
- **Mamba/SSM** (`ssm/ops/determinism.py`, `ssm/gated_delta_net.py:213/314/430`):
  cheapest-autotune + tiled-workspace reduction; torch `chunk_gated_delta_rule` /
  `F.conv1d` fallbacks; packed-seq forbidden.
- **TE attention** (`extensions/transformer_engine.py:1697`): asserts
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` when `deterministic_mode` is on.
- **Inference/RL scheduling** (`dynamic_engine.py:607`,
  `data_parallel_inference_coordinator.py:181`, `rl/rl_utils.py:678`): sort by a
  stable key instead of completion order to remove timing jitter.

## 7. Validation status (tests)

- **Module-level (PR #5041, OPEN):** `tests/unit_tests/determinism/` —
  `BitExactRunner` runs a "cell" twice (restoring RNG between runs) and asserts
  bit-identical outputs **and** gradients across
  GPTModel × HybridModel × TransformerBlock, crossed with TP/EP/FSDP/PP/VPP
  composites and FP8 (tensorwise/delayed/mxfp8) / FP4 (nvfp4) recipes. Stress
  utilities `RacingStreams` and `CudaSleepJitter` perturb scheduling to surface
  latent collective-ordering races.
- **Perf gate (PR #5041, OPEN):**
  `tests/performance_tests/shell_test_utils/determinism/` runs `pretrain_gpt.py`
  det + nondet under nsys, prints a per-NVTX-range leaderboard
  (`print_nsys_leaderboard.py`), and **fails if det/nondet step-time ratio >
  1.25×**. Routed via `recipes/h100/determinism-perf.yaml`.
- **Model-level proxies (WS2 Tier A — added):** `BitExactRunner`-based proxies
  for the two target architectures:
  - `tests/unit_tests/determinism/correctness/test_deepseek_model.py` — DSV3-style
    MLA + fine-grained MoE with **group-limited routing** (the `group_limited_topk`
    path) + sigmoid/expert-bias + grouped GEMM.
  - `tests/unit_tests/determinism/correctness/test_nemotron_hybrid_model.py` —
    Nemotron-3-Ultra-style **MoE-inside-hybrid** (Mamba + attention + MoE) under EP.

  Both pass bit-exact across EP≤4 / TP / FSDP / PP / VPP (12/12 cells, draco
  8×H100). CI auto-discovers them via the `determinism/correctness/**/*.py` glob
  in `unit-tests.yaml`. **Note:** the branch pins `nvidia-resiliency-ext==0.6.0`;
  images older than that need an in-container upgrade to import mcore.
- **E2E full-recipe (WS2 Tier B — pending):** the real nemotron-3-ultra recipe
  lives in **Megatron-Bridge** (`zhiyul/nemotron-3-ultra-perf-recipe`); the
  weekly multi-node e2e + wandb dashboard is the remaining tier — it is the only
  way to reach **EP>16**, where DSV3 non-determinism empirically appears.

## 8. Known gaps (feeding the roadmap)

1. **No e2e determinism coverage** → Workstream 2: scaled CI-gating proxies in
   mcore (nemotron-3-ultra, DSV3) + full recipes in Megatron-Bridge (weekly).
2. **~15% perf overhead** concentrated in MoE scatter/unpermute, FlashAttention
   backward (FAG), grouped GEMM, and top-k radix sort → Workstream 3.
3. **No first-divergence debugging tool.** Non-determinism only appears at scale
   under specific parallelism combos (DSV3 diverges only at EP>16 + PP/VPP). Plain
   `nn.Module` hooks miss collectives, optimizer math, recompute, and
   allocator-driven kernel selection; PP/VPP scramble event order so "first" is
   ill-defined → Workstream 4: typed instrumentation API + first-divergence finder.
4. **Scatter/cumsum sites without an explicit det branch** (e.g.
   `moe_utils.py:622, 836-837, 890, 946, 951`, `router.py:261`) — likely safe in
   forward (unique indices) but their backward and large-scale behavior must be
   confirmed with `BitExactRunner` (see catalog "⚠ verify" rows).

## 9. References

- Determinism roadmap & meeting notes (internal Google Docs).
- Customer technical reports: Meituan Longcat (deterministic FAG / ScatterAdd /
  grouped GEMM / fused GemmAdd), MSFT, DeepSeek-V4.
- Thinking Machines, "Defeating nondeterminism in LLM inference."
- PyTorch reproducibility & `torch.use_deterministic_algorithms` docs.
