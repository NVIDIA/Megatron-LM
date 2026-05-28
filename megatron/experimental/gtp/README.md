# Generalized Tensor Parallelism (GTP) — Key Features

> ⚠️ **Experimental.** GTP is an experimental feature and its API, configuration, and behavior may change in future versions without notice.

**Scope**: this doc is a high-level summary of GTP.

Core implementation: `megatron/experimental/gtp/generalized_tensor_parallelism.py`. The public surface is re-exported from `megatron/experimental/gtp/__init__.py`, which also owns the `HAVE_GTP` fallback used by callers that need to remain importable when GTP is unavailable. Low-precision tensor primitives (FP8 / MXFP8 / NVFP4) remain in TransformerEngine and are imported by `generalized_tensor_parallelism.py`.

---

## 1. Fine-Grained, Per-Weight Materialization & Gradient Reduction

Each weight is sharded 1/N across an GTP group along `out_features`, stored as an `GTPShardedParam` subclass of `nn.Parameter`. Materialization and gradient reduction are both **per-weight, per-call** — not per-model or per-module:

- **Independent state per param**: each has its own AG state (`state`) and RS state (`rs_state`) machines, both cycling `NONE → ASYNC_WAIT → DATA_READY → NONE` and tracked separately so fwd and bwd async ops don't interfere.
- **Prefetch chain for AG** (doubly-linked `prev_w` / `next_w`): during fwd, each weight's `all_gather_and_prefetch` issues async AG for `next_w`; during bwd, `all_gather_and_prefetch_bwd` issues async AG for `prev_w`. Layer *i*'s AG overlaps with layer *i−1*'s GEMM. For an L-layer model, L−1 all-gathers are fully hidden behind compute.
- **Deferred RS finalize for wgrad**: `wgrad_reduce_scatter` on param *i* launches an **async** reduce-scatter (handle stashed in `_wgrad_rs_handle`) and returns `None` to autograd — the wgrad is NOT finalized into `main_grad` yet. Finalization is **deferred one step**: the next bwd step (param *i−1*'s `wgrad_reduce_scatter`) calls `self.next_w._wait_reduce_scatter()` + `_finalize_wgrad()`, which waits on the stashed handle, accumulates the reduced wgrad into `main_grad`, and fires the DDP `register_grad_ready` hook. The chain's head (first-in-fwd, last-in-bwd) uses a synchronous RS since nothing follows it. This one-step deferral is what lets layer *i*'s RS overlap with layer *i−1*'s bwd GEMMs.
- **Cold start only**: every weight's very first AG is synchronous (`DATA_READY_SYNC`, no prefetch has run yet); the async prefetch chain kicks in from the second forward onward.

Contrast with FSDP: FSDP gathers at module-group granularity in full precision with PyTorch-managed lifecycle. GTP works at individual-weight granularity, in quantized form, with its own explicit ticket-based buffer pool and a one-step-deferred RS finalizer.

> **FSDP can't shrink into GTP because FSDP's overlap is bucket-grained by design** — bucket granularity exists *to avoid* paying NCCL launch latency on tiny params (LayerNorm γ/β, biases, Mamba `dt_bias`/`D`/`A_log`) and *to avoid* the per-weight scheduling state that GTP relies on (per-param prefetch chain, ticket-based buffer cache, stream choreography). Removing buckets doesn't make FSDP faster; it makes FSDP into GTP, with all the engineering that entails — selective wrapping (only large GEMM weights), per-weight prefetch chain, per-param buffer ticket, and explicit AG/RS stream coreography on a side stream so external drains have something meaningful to wait on.

## 2. CUDA Graph Compatibility

CG compatibility is designed-in from day one, not retrofitted. The entire sync / buffer / chain architecture is shaped around making **captured fwd/bwd replays produce identical bit-for-bit behavior** — without the usual capture-vs-eager pitfalls that force other weight-sharding schemes to either disable CG or require special handling.

- **Two chains, never cross-linked** (`GTPChain.GRAPHED` / `GTPChain.UNGRAPHED`). `prev_w` / `next_w` only connect same-chain params, so a captured traversal never reaches into eager Python and vice-versa.
- **`torch.cuda.Event(external=True)`** for `ag_event` / `rs_event` — the events survive CG capture boundaries and can be waited on from replay-time streams.
- **Idempotent ticket cache**: `GTPWeightCache.get(ticket)` keeps `slot.buf` set even after `release()`, so replays read the same buffer address as capture. `clear()` drops buffers while keeping tickets valid → supports CG re-capture with lazy re-allocation.
- **`reallocate_to_mempool(device, mempool)`** pre-migrates GRAPHED-chain buffers into the CG memory pool *before* capture, so no CUDA allocations happen inside the captured graph. UNGRAPHED buffers stay in regular allocator memory.
- **Lazy, one-shot chain linking**: `prefetch_initialized` is flipped during the first fwd (warmup), so the chain-construction Python side-effects never execute inside a captured graph. The link table is buffered and flushed atomically at the second forward.
- **DDP hook manual triggering**: `register_grad_accum_hook` stores the DDP hook on the param; `_CudagraphReplayNode.backward` calls it manually after replay (since `AccumulateGrad` hooks are silenced by replay). This is also how the `assert self.grad_reduce_handle is not None` failure from partial-CG + overlap-grad-reduce is resolved.
- **Drains at CG / eager boundary**: `_drain_gtp_side_streams()` before eager MoE expert compute. Inside bwd capture, two-phase drain: Phase 1 joins the within-graph cascade and records `bwd_completion_event` (next runner unblocks); Phase 2 calls `wait_async_comms(GRAPHED)` to drain the chain-tail handle and re-joins side streams (queued after the event so it doesn't delay the next runner).
- **Side-stream registration**: the `(GRAPHED, gtp_group)` ag/rs streams are materialized at runner init so `_register_side_stream` captures them before the first forward.

## 3. Low-Precision Quantize-Then-Gather

Wire bandwidth scales with the **quantized** size, not BF16 size — GTP composes with low-precision training rather than fighting it.

- **FP8 / MXFP8**: quantize kernel runs per microbatch on the local shard with no GTP-group amax reduction (FP8 amax allreduce is the standard DP-group one in `reduce_and_update_fp8_tensors`, unchanged by GTP). On subsequent microbatches, `skip_weight_cast=True` reuses the quantized buffer.
- **NVFP4** (4-bit, block-scaled): amax reduced across the GTP group before scaling so ranks share a consistent scale for the full weight; custom `_all_gather_nvfp4` handles rowwise + columnwise views and interleaved layout. Post-processing (re-assemble interleaved data, re-pad `scale_inv`, transition to `GEMM_READY`) is deferred into `_NVFP4AllGatherAsyncHandle.wait()` so it stays off the critical path.
- **Coalesced NCCL**: `grouped_gather_along_first_dim` uses `torch.distributed._coalescing_manager` to batch E experts' AGs into a single NCCL op. `BatchedNVFP4AllGatherAsyncHandle` wraps per-expert post-processing.
- **Padding**: at construction the **full tensor** is padded along dim0 to a multiple of `pad_for_alignment × gtp_size`, then sharded equally across the group. After all-gather, the padding ends up contiguous at the tail, so stripping is a single trailing slice (`tensor[:-pad_length]`) — no per-shard reshuffle, and the design naturally supports `pad_length` large enough to span multiple ranks' shards when the unpadded dim0 is small.

### Per-microbatch schedule

```
Steady-state fwd (NVFP4):
    default: ──GEMM(W_0)──quant+amax(W_1)──GEMM(W_1)──quant+amax(W_2)──GEMM(W_2)──...
    ag_str:                       [AG_issue W_1]            [AG_issue W_2]

Steady-state fwd (FP8 / MXFP8):
    default: ──GEMM(W_0)────quant(W_1)─────GEMM(W_1)────quant(W_2)─────GEMM(W_2)──...
    ag_str:                       [AG_issue W_1]            [AG_issue W_2]
                              (no GTP-group amax allreduce)

Steady-state bwd (all recipes):
    default: ──bwd GEMMs(W_i)──...
    ag_str:               [AG_issue W_{i-1}]
                          (bwd reuses fwd's quantized buffer; no quant, no amax)
```

quant+amax run sequentially with surrounding compute on the default stream; only the `dist.all_gather` issue is wrapped in `with torch.cuda.stream(ag_stream)`. The NCCL kernel runs on c10d's private ncclStream and overlaps with the next GEMM until it reaches its wait.

For NVFP4 the per-microbatch prefetch cost is **two** NCCL ops on the GTP ncclStream (amax allreduce + AG) serialized on the same communicator. FP8 and MXFP8 incur only the AG; their standard DP-group amax allreduce in `reduce_and_update_fp8_tensors` is unchanged by GTP. BF16 skips quant entirely.

### Communication volume breakdown

Per-microbatch per-weight comm budget (assuming bf16 wgrad reduce-scatter):

| Format | Block | Data B/elem | Scale_inv B/elem | Per-elem | Fwd AR(amax)                   | Fwd AG | Bwd AG | Wgrad RS (bf16) | Total B/elem | vs BF16        |
|--------|-------|-------------|------------------|----------|--------------------------------|--------|--------|-----------------|--------------|----------------|
| BF16   | n/a   | 2.0000      | —                | 2.0000   | —                              | 2.0000 | 2.0000 | 2.0000          | 6.0000       | 1.00× (baseline) |
| MXFP8  | 32    | 1.0000      | 1/32 = 0.0313    | 1.0313   | — (microscale, no global amax) | 1.0313 | 1.0313 | 2.0000          | 4.0626       | 0.68× (–32%)   |
| NVFP4  | 16    | 0.5000      | 1/16 = 0.0625    | 0.5625   | ≈0 in volume (latency-bound)   | 0.5625 | 0.5625 | 2.0000          | 3.1250       | 0.52× (–48%)   |

How to read the columns:
- `Per-elem` = `Data B/elem + Scale_inv B/elem` — wire cost of one quantized weight buffer (data + scale_inv together).
- `Fwd AG` and `Bwd AG` each carry the quantized buffer once, so they equal `Per-elem`. Bwd reuses fwd's `self.quantized` buffer — no re-quantize, no AR(amax).
- `Wgrad RS (bf16)` = 2.0 B/elem — gradient is reduce-scattered in bf16 regardless of weight precision.
- `Fwd AR(amax)` is a separate NCCL collective: NVFP4 needs it (one fp32 scalar per tensor → ~0 B/elem volume but a fixed launch latency); MXFP8 doesn't (microscale-only).
- `Total B/elem` = `Fwd AG + Bwd AG + Wgrad RS` — amax AR is omitted because its volume is essentially 0.

Concrete numbers for one weight of shape `[16384, 4096]` (67.1M params), per microbatch:

| Format | Per-microbatch volume |
|--------|-----------------------|
| BF16   | 403 MB                |
| MXFP8  | 273 MB (–32%)         |
| NVFP4  | 210 MB (–48%)         |

Quantize-then-gather attacks AG only: AG portion shrinks ~72% from BF16 → NVFP4, but RS is untouched, so the wgrad RS becomes the dominant comm path in NVFP4 (~64% of the budget at bf16 RS, ~78% at fp32 RS).

### Insights from the comm-volume table

1. **AG savings ≠ end-to-end comm savings.** The headline "NVFP4 → 4× smaller wire" applies to AG only. End-to-end per-weight comm goes 6.0 → 3.125 B/elem, which is **2×, not 4×**. The other half of the savings is left on the table because RS doesn't shrink.

2. **Wgrad RS becomes the new bottleneck under NVFP4.** RS share of the budget jumps from ~33% (BF16) to ~64% (NVFP4) at bf16 RS dtype. Future optimization should target the RS path (RS dtype reduction, coalesced gradient reduction), not more aggressive AG quantization.

3. **Diminishing returns from precision halving.** BF16 → MXFP8 saves 1.94 B/elem (–32% of baseline); MXFP8 → NVFP4 saves only an additional 0.94 B/elem (–16% of baseline). Each step saves a smaller share of the original budget — there's no point chasing more aggressive AG quantization until RS shrinks.

4. **AR(amax) is latency-only, not volume.** One fp32 scalar per tensor → effectively 0 B/elem in a bandwidth budget. It belongs in a launch-overhead model, not in the bytes table.

5. **Bwd inherits AG savings for free.** The quantize + amax + cast cost is paid once per microbatch in fwd. Bwd reuses the cached `self.quantized` buffer with no re-quantize and no AR — but still gets the small-AG payload. The cost-benefit of quantize-then-gather is asymmetric: fwd pays once, both fwd-AG and bwd-AG benefit.

**Net takeaway:** "NVFP4 → 4× smaller wire" is half-true. GTP+NVFP4 cuts AG ~4× but leaves RS untouched, so end-to-end comm is only ~2× faster. The next lever isn't more aggressive quantization — it's RS dtype reduction or coalesced gradient reduction.

## 4. Buffer / Memory Management

Two distinct pools with explicit lifecycle rules:

- **`GTPWeightCache`** (AG/RS output buffers) — ticket-based, keyed on `(shape, dtype, fwd, expert_idx, reduce_scatter)`. Same-shape buffers across layers are shared. Tickets persistent; buffer allocated lazily on first `get()`; addresses stable across iterations for CG replay.
- **`_wgrad_buf_pool`** (UNGRAPHED wgrad input recycling) — tagged with `_from_gtp_wgrad_pool=True` at `_wgrad_pool_get`. `_wgrad_pool_put` no-ops on foreign buffers (fresh allocs from Megatron `layers.py` or aten F.embedding bwd) → caching allocator handles those. Prevents the pool from accumulating untagged buffers each iter.

## 5. Composability with TP / SP / EP / DDP

- **TP** (intra-layer): orthogonal axis — GTP shards `out_features` regardless of TP's parallel mode (column or row). 2D grid naturally formed via `tp_group × gtp_group`.
- **SP** (sequence-parallel): transparent — GTP operates at weight dim, SP at sequence dim.
- **EP** (MoE): `GroupedLinear` with GTP → each routed expert sharded across `EXPERT_GENERALIZED_TENSOR_PARALLEL_REMAT_GROUP`, independent of EP. MoE AllToAll (HybridEP/NVLink) runs independently of GTP AG/RS (NCCL/IB).
- **DDP**: GTP bypasses autograd's grad accumulator (async RS returns `None`; `_finalize_wgrad` accumulates directly into `main_grad`). `register_grad_accum_hook` + manual invocation from `_finalize_wgrad` (eager path) and `_CudagraphReplayNode.backward` (captured path) serializes DDP RS strictly after GTP RS — critical at IB scale to avoid deadlock between DDP and GTP on the same NIC.

## 6. Opt-in, Minimally Invasive Integration

- Drop-in `gtp_group` kwarg on `Linear` / `LayerNormLinear` / `LayerNormMLP` / `GroupedLinear`; no framework-level refactor required.
- `classify_gtp_chains(model)` walks `named_parameters()` once at init and sets `chain_id` on every `GTPShardedParam` based on the current `cuda_graph_modules`.
- Turning it off is a no-op: when `gtp_group.size() == 1`, `wrap_module_params_gtp` short-circuits; when `generalized_tensor_parallel_remat_size == 1`, the GTP path in `layers.py` is skipped entirely.
- User-tunable knobs (`GTPConfig.pad_for_alignment`, `weight_prefetch`, `check_param_states`) plus a debug-name tagger (`tag_gtp_params_with_names`) for readable link-table output.

## 7. Overlap Design Summary

```
fwd:  AG(W_{i+1}) ∥ GEMM(W_i)                              ∥ CG replay of captured layers
bwd:  AG(W_{i-1}) ∥ dgrad(W_i) ∥ wgrad(W_i) ∥ RS(wgrad_i)  ∥ [finalize wgrad_{i+1} + DDP hook]
```

At bwd step *i* the step is launching *RS of wgrad_i* while finalizing the *previous* iter's wgrad (`wgrad_{i+1}` in bwd order = the next-one-over in fwd order). That one-step deferral is what makes the RS run concurrent with the next layer's dgrad/wgrad GEMMs instead of blocking after every layer.

Communication never blocks compute except at the very first layer of each direction (cold start) and at enforced serialization points (CG/eager drains, finalize-grads barrier).

### 7a. wgrad-before-dgrad schedule  *(deferred to a follow-up MR)*

Current behavior: backward always runs dgrad GEMM, then wgrad GEMM, then issues the GTP wgrad RS — the RS overlaps with the *next* layer's bwd GEMMs (the one-step deferral above).

A future MR will add an opt-in wgrad-before-dgrad schedule on `_Linear` / `_LayerNormLinear` so the GTP wgrad RS NCCL overlaps with the dgrad GEMM of the **same** layer (best for the GTP + no-TP case). Until that MR lands, attempting to set `GTPConfig.wgrad_before_dgrad = True` raises `NotImplementedError`.

## 8. Scaling

Effective per-GPU weight size = `W / (TP × GTP)`. Example: TP=4 + GTP=8 with NVFP4 → 32× weight-memory reduction and 128× wire-bandwidth reduction vs full BF16 replication, before data parallelism.

## 9. Usage

GTP is enabled through two CLI flags on Megatron's training launcher; everything else (process-group construction, parameter slicing, prefetch chain wiring, optimizer routing) is automatic once the flags are set.

### Required flags

```bash
# Shard dense weights (attention, mamba, MLP linears) 1/N along out_features.
--generalized-tensor-parallel-remat-size <N>

# Shard MoE routed-expert weights 1/M along out_features. Independent from
# `--generalized-tensor-parallel-remat-size`; can be 1 for non-MoE models.
--expert-generalized-tensor-parallel-remat-size <M>
```

### High-priority streams (Blackwell and later)

Required on GB200 / B100 so the GTP comm streams get the SM priority needed for AG/RS overlap with compute:

```bash
--high-priority-stream-groups ep gtp expt_gtp tp
```

The launcher also exports `CUDA_GRAPHS_USE_NODE_PRIORITY=1` so captured CUDA graphs respect the inherited stream priority.

### Minimal end-to-end example

```bash
# 4 ranks, GTP=2 across out_features, no TP, BF16 weights.
torchrun --nproc-per-node 4 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --generalized-tensor-parallel-remat-size 2 \
    --expert-generalized-tensor-parallel-remat-size 1 \
    --bf16 \
    --num-layers 12 --hidden-size 1024 --num-attention-heads 16 \
    --seq-length 1024 --max-position-embeddings 1024 \
    --micro-batch-size 1 --global-batch-size 4 \
    --train-iters 10 \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tokenizer-type NullTokenizer --vocab-size 32000 \
    --data-path <data> --split 99,1,0
```

At iter-0 you'll see one rank-0 log line confirming the active config:

```
GTP enabled. GTPConfig(pad_for_alignment=16, check_param_states=False,
  weight_prefetch=True, async_reduction=True, wgrad_before_dgrad=False,
  fp8_param_gather=False, coalesce_amax_allreduce=False)
```

### What the flags do under the hood

1. `parallel_state.initialize_model_parallel(...)` builds two new groups: `_GENERALIZED_TENSOR_PARALLEL_REMAT_GROUP` (size = `--generalized-tensor-parallel-remat-size`) and `_EXPERT_GENERALIZED_TENSOR_PARALLEL_REMAT_GROUP` (size = `--expert-generalized-tensor-parallel-remat-size`), plus the corresponding DP-with-GTP carve-outs (`_DATA_PARALLEL_GROUP_WITH_GTP`, `_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP`).
2. Megatron's `extensions/transformer_engine.py` reads `pg_collection.gtp` / `pg_collection.expt_gtp` and forwards them as the `gtp_group=` kwarg to `te.Linear` / `te.LayerNormLinear` / `te.GroupedLinear`. TE's `module/base.py` calls back into `megatron.experimental.gtp` via the hook registry (`register_gtp_hooks`) to slice each weight at `reset_parameters` time.
3. DDP carves out GTP shards into a separate bucket pool (`gtp_buffer_groups`) reduced over `intra_dp_cp_with_gtp_group` rather than full DP — the wgrad RS already reduced over the GTP axis.
4. Optimizer state is sharded across the same `with_gtp` subgroup; clip-by-global-norm sums squared norms over `model_parallel × with_gtp` so the reduction count matches the actual replica count.
5. `classify_gtp_chains(model)` runs once after model build (in `training.py`'s `get_model`) and wires each `GTPShardedParam` into a `GRAPHED` or `UNGRAPHED` prefetch chain based on the active `cuda_graph_modules`.

### Tuning knobs

Set via `from megatron.experimental.gtp import GTP_CONFIG, update_gtp_config`:

```python
update_gtp_config(
    pad_for_alignment=16,         # NVFP4: 16, MXFP8: 32, BF16: any; auto-set in training.py
    weight_prefetch=True,         # Disable to debug the cold-start path
    async_reduction=True,         # Wheter perform GTP gradient reduction asynchronously
    # wgrad_before_dgrad: deferred — setting True currently raises NotImplementedError
    fp8_param_gather=False,       # Companion to Megatron's --fp8-param-gather; currently asserted off
    # coalesce_amax_allreduce: deferred — setting True logs an info and falls back to per-weight
)
```

`training.py` auto-tunes `pad_for_alignment` based on the quantization recipe (`--fp4`, `--fp8-recipe=mxfp8`, etc.) before model construction. The other knobs are usually left at defaults.
