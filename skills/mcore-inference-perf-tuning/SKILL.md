---
name: mcore-inference-perf-tuning
description: Performance tuning for Megatron-LM dynamic-batching inference. Covers CUDA graphs, chunked prefill, prefix caching (including Mamba/GDP state caching), parallelism layout, MXFP8 numerics, reasoning/tool parsers, and how to read the inference step log to decide what to change next.
license: Apache-2.0
when_to_use: Tuning or debugging inference throughput/latency; choosing TP/EP/ETP/SP for a serving job; enabling CUDA graphs, chunked prefill, or prefix caching; sizing the KV buffer, max-tokens, or max-requests; interpreting `--inference-logging-step-interval` output; low prefix cache hit rate, frequent evictions, queued requests, or OOM; 'inference is slow', 'tune the server', 'why are requests waiting', 'prefix cache not hitting', 'enable CUDA graphs for inference'.
---

# Inference Performance Tuning

Applies to the **dynamic batching** engine (`--inference-dynamic-batching`),
as launched by `tools/run_dynamic_text_generation_server.py` or
`examples/inference/launch_inference_server.py`.

---

## Answer-First Baseline

Start from this and only deviate with a reason. `<N>` = GPUs per node group.

```bash
torchrun --nproc_per_node=8 -m tools.run_dynamic_text_generation_server \
    --load "$CHECKPOINT" --use-checkpoint-args --bf16 \
    \
    `# --- parallelism ---` \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --transformer-impl inference_optimized \
    --attention-backend flash \
    \
    `# --- CUDA graphs ---` \
    --cuda-graph-impl local \
    --inference-cuda-graph-scope block \
    --inference-dynamic-batching-num-cuda-graphs -1 \
    \
    `# --- batching / chunked prefill ---` \
    --inference-max-seq-length 32768 \
    --inference-dynamic-batching-max-tokens 4096 \
    --enable-chunked-prefill \
    --inference-dynamic-batching-buffer-size-gb 40 \
    --inference-dynamic-batching-max-requests 256 \
    \
    `# --- prefix caching ---` \
    --inference-dynamic-batching-prefix-caching \
    --inference-dynamic-batching-prefix-caching-eviction-policy lru \
    --inference-dynamic-batching-prefix-caching-coordinator-policy longest_prefix \
    --inference-dynamic-batching-prefix-caching-mamba-gb 20 \
    \
    `# --- numerics (hybrid/Mamba) ---` \
    --mamba-inference-ssm-states-dtype fp32 \
    \
    `# --- misc ---` \
    --skip-prompt-log-probs \
    --parsers nemotron-v3-reasoning qwen3-coder-tool \
    --inference-logging-step-interval 100 \
    --logging-level 20
```

Flag-name gotchas that cost real time:

- The sequence-length flag is **`--inference-max-seq-length`**, not
  `--inference-max-sequence-length`. There is no such flag as the latter;
  argparse will reject it.
- `--cuda-graph-scope` and `--cuda-graph-modules` are **deprecated**. Use
  `--cuda-graph-impl` + `--inference-cuda-graph-scope`. In particular, older
  scripts carrying `--cuda-graph-scope full_iteration_inference` should be
  migrated to `--cuda-graph-impl local --inference-cuda-graph-scope block`.
- `--inference-dynamic-batching-prefix-caching` is a
  `BooleanOptionalAction`, so `--no-inference-dynamic-batching-prefix-caching`
  turns it off.

---

## 1. CUDA Graphs

### Why they dominate decode performance

Decode is one token per request per step. The GPU work per step is tiny —
tens of microseconds of kernels per layer — while the CPU work to launch
those kernels is not. Without graphs, the Python/ATen launch path becomes the
critical path: the GPU idles between kernels waiting for the CPU to enqueue
the next one, and step time is dominated by launch overhead that does not
shrink as the model gets faster. On a small model at small batch this can be
**most of the step time**.

A CUDA graph captures the whole kernel sequence once and replays it as a
single launch. That removes per-kernel dispatch, per-step Python, and most of
the CPU/GPU sync points. Graphs are the single highest-leverage inference
flag — enable them before tuning anything else, because every other
measurement is polluted by launch overhead when they are off.

Graphs require **static shapes**, so the engine captures a set of graphs at
discrete batch/token sizes and pads the real batch up to the nearest captured
size. Padding is the cost you trade for the launch savings; sizing the graph
set (below) is about keeping that padding small.

### Enabling

```bash
--cuda-graph-impl local
--inference-cuda-graph-scope block
```

- `--cuda-graph-impl` selects the capture mechanism:
  `none` (default, eager), `local` (Megatron's own capture — **this is the
  inference path**), `transformer_engine`, `full_iteration` (training).
- `--inference-cuda-graph-scope` selects the ownership boundary
  (`megatron/core/transformer/enums.py:101`):
  - `none` — eager.
  - `layer` — one graph per `TransformerLayer` / `MambaLayer`. This is the
    **default** when `--cuda-graph-impl local` is set and the scope is
    unspecified.
  - `block` — one graph for the enclosing `TransformerBlock` / `HybridBlock`.
    **Prefer this.** A single replay for the whole block beats N per-layer
    replays: fewer launches, and inter-layer work (norms, residuals, TP
    collectives) lands inside the graph instead of between graphs.

Only `local` permits a non-`none` inference scope
(`ALLOWED_INFERENCE_SCOPES`, `megatron/core/transformer/cuda_graph_config.py:17`);
combining `block` with any other impl asserts at startup.

### Sizing the graph set

```bash
--inference-dynamic-batching-num-cuda-graphs -1
```

The default is `16` graphs spread over batch sizes `1..max_requests`. Passing
**`-1`** lets the engine derive the count from `max_requests` instead of
holding it fixed. This is what you want: with a fixed 16 and a large
`max_requests`, the spacing between captured sizes is wide, so a batch of 33
pads to 64 and you burn ~half the step on padding. Autoderived counts keep
relative padding bounded.

Related knobs, in the order you should reach for them:

- `--inference-dynamic-batching-cuda-graph-sizing-distribution`
  (`hybrid` default / `exponential` / `linear`) — `hybrid` uses exponential
  spacing for prefill/mixed graphs and linear spacing for decode-only graphs
  (whose token counts are capped at `max_requests` and are too small for
  halving to cover well). Leave it alone unless you have measured padding
  waste in the log's `real config` vs `cuda graph` numbers.
- `--inference-cuda-graph-max-tokens` (default 512) — token ceiling for the
  largest prefill/mixed graph. Raise it if prefill steps are frequently
  running eager (`cuda graph [...]: OFF` in the log on P steps).
- `--inference-cuda-graph-all-prefills` — extends prefill capture all the way
  to `max_tokens`. More capture time and memory; use when prefill steps
  matter more than startup time.
- `--decode-only-cuda-graphs` — captures **only** decode-only steps. Cuts
  capture time and memory at the cost of eager prefill/mixed steps.

Capture happens at startup and costs both wall time and memory. If startup is
slow or you OOM during capture, that is the graph set — shrink it before
shrinking the KV buffer.

---

## 2. Chunked Prefill, and the two "max" knobs

### `--inference-max-seq-length` vs `--inference-dynamic-batching-max-tokens`

These are constantly confused. They measure different things.

| | `--inference-max-seq-length` | `--inference-dynamic-batching-max-tokens` |
|---|---|---|
| Unit | tokens **in one request** | tokens **in one engine step**, summed over all requests |
| Bounds | prompt + generated output for a single request | the per-step token budget of the forward pass |
| Default | 2560 | 16384 (`DynamicInferenceContext.DEFAULT_MAX_TOKENS`) |
| Drives | `max_kv_block_count` per request — how many KV blocks one request may ever hold | the size of every per-token buffer in the context (`token_to_input_ids`, `token_to_pos_ids`, position/block index tensors, Mamba `seq_idx` buffers, …) and the ceiling on CUDA graph token counts |
| Too small ⇒ | long requests get rejected/truncated | long prompts cannot be prefilled at all (without chunked prefill) |
| Too large ⇒ | more blocks reserved per request; fewer concurrent requests | large per-step buffers, wide CUDA graph range, wasted memory |

Mental model: **`max-seq-length` is a per-request length limit;
`max-tokens` is a per-step width limit.** A 32K-token prompt with
`--inference-max-seq-length 32768 --inference-dynamic-batching-max-tokens 4096`
is legal *only* with chunked prefill, which is what makes the pairing work.

There is one hard coupling asserted at startup
(`dynamic_context.py:717`): `max_tokens >= max_requests`. Each active request
contributes at least one token per decode step, so the step budget can never
be narrower than the request count.

### Enabling chunked prefill

```bash
--enable-chunked-prefill
--inference-dynamic-batching-max-tokens 4096
```

Chunked prefill splits one long prompt's prefill across multiple steps
instead of demanding that it fit in a single step's token budget. That
decouples the two knobs above: you can serve long contexts without sizing
every per-step buffer for the longest possible prompt.

It also fixes a latency problem. Without chunking, a single long prefill
monopolizes a step, and every decoding request stalls behind it — a classic
TPOT spike / head-of-line block. With chunking, prefill is metered into
`max_tokens`-sized slices that share each step with ongoing decodes (mixed
steps), so inter-token latency stays flat under prefill load.

**Sizing `max-tokens`: 4096 is the right default.** The tradeoff:

- Too small (≤1024): prefill is chopped into many steps, each paying full
  per-step overhead, and prefill throughput drops. It also caps how much
  decode batching the mixed step can absorb.
- Too large (16384 default): per-step token buffers grow linearly, prefill
  chunks are long enough to stall decodes again, and the CUDA graph token
  range widens (more graphs, more padding, more capture memory).
- 4096 keeps the GEMMs comfortably compute-bound while bounding the stall a
  decode request can experience behind a prefill chunk.

Two model-specific constraints:

- **Hybrid/Mamba in batch-invariant mode** requires
  `max_tokens > ssm_chunk_alignment` (`dynamic_context.py:858`).
- `max_tokens` also feeds the Mamba prefix-cache **scratch** sizing — see §3.

---

## 3. Prefix Caching

### What it is

Requests that share a prompt prefix compute identical KV for that prefix.
Prefix caching hashes prompt content per KV block and lets a new request
**adopt existing blocks** rather than recompute them. The prefill for the
matched span is skipped outright.

The workloads where this is worth the most:

- **Agentic / multi-turn.** Turn N's prompt is turn N−1's prompt plus the
  model's reply plus a tool result. The shared prefix is nearly the entire
  context and grows every turn.
- **Long shared system prompts / few-shot preambles** across many requests.
- **Beam-search-ish or fan-out sampling** from a common prompt.

The payoff is superlinear in conversation depth: without caching, an
N-turn conversation re-prefills O(N²) tokens; with caching, O(N).

### Enabling

```bash
--inference-dynamic-batching-prefix-caching
```

Off by default. When disabled, blocks are never shared between requests even
with byte-identical prompts.

### Eviction policy — use `lru`

```bash
--inference-dynamic-batching-prefix-caching-eviction-policy lru
```

Two policies (`arguments.py:2253`):

- `ref_zero` — a block returns to the free pool the instant its refcount hits
  zero. The moment a conversation's request finishes, its blocks are gone.
- `lru` (**default, and what you want**) — finished blocks stay cached and are
  evicted only under allocation pressure, oldest-touched first.

For agentic multi-turn this is the difference between a working cache and no
cache at all. Between turns there is *think time*: the client is running a
tool, or a human is typing. During that window the conversation has no live
request, so under `ref_zero` its entire KV context is freed — and the next
turn re-prefills the whole conversation. `lru` keeps that context resident
across the gap and evicts it only if the GPU actually needs the blocks. This
is the single most important prefix-caching flag for agent workloads.

`lru` costs nothing when the pool is not under pressure: blocks that would
have been freed simply sit in the free pool marked evictable, and are
reclaimed on demand.

### Mamba / GDP models: budget the state cache explicitly

```bash
--inference-dynamic-batching-prefix-caching-mamba-gb 20
```

**Hybrid models need this flag or prefix caching will barely help them.**

Attention KV is positional: cache the blocks and any request with the same
prefix can attend over them. Mamba/GDP layers are recurrent — the layer's
contribution to position *i* is a **sequential state**, not a per-position
tensor. To skip prefill for a prefix on a hybrid model you must have both the
KV blocks *and* the Mamba state at that prefix's block boundary. Miss the
state and the prefill runs anyway, no matter how many KV blocks matched.

This budget covers two things, and the split matters
(`arguments.py:2313`):

1. **Durable slots** — the `ssm_states`/`conv_states` snapshots reused across
   requests. This is the actual cache.
2. **Per-step extraction scratch** — `intermediate_ssm_out` /
   `intermediate_conv_out`, sized to
   `min(ceil(max_tokens / block_size), 3 * max_requests)` slots.

**Scratch is reserved first.** So a large `max_tokens` or `max_requests` eats
into the budget before any durable slot is allocated. If durable slots come
out low, lowering `max_tokens` (another reason 4096 beats 16384) or
`max_requests` buys them back without raising the GB budget.

20 GB is a reasonable starting point for an 8-GPU node. Verify against the
log's `mamba N/M durable slots` field (§6) — that number is the ground truth
for whether the budget is doing anything.

Related: `--inference-dynamic-batching-mamba-memory-ratio` controls how much
of the *main* KV buffer goes to Mamba state tensors. That is a separate
allocation from the prefix-cache budget; don't conflate them.

### Coordinator routing policy (multi-DP)

With DP > 1, each rank has an **independent** cache. Routing decides which
rank sees a request, and therefore whether its prefix is on that rank at all.
Route round-robin and you get a ~1/DP hit rate on a workload that should be
hitting ~100%.

Three policies (`arguments.py:2261`):

| Policy | Behavior |
|---|---|
| `load_balanced` | Fewest in-flight requests. Ignores prefix affinity entirely. **Safe fallback, worst hit rate.** |
| `first_prefix_block` | Routes on the first block hash only. Cheap; keeps a conversation pinned to one rank. Combines affinity with load balancing. |
| `longest_prefix` | **Current default. Use this.** Routes to the rank with the longest matching prefix, combined with load balancing. |

To answer the standing question directly: `longest_prefix` is the newer,
better-optimized policy and is now the default — you no longer need to fall
back to `load_balanced` for safety. `longest_prefix` and `first_prefix_block`
both **automatically degrade to load-balanced routing** when prefix caching is
off or no prefix matches, so there is no lopsided-fleet failure mode to guard
against. Prefer:

```bash
--inference-dynamic-batching-prefix-caching-coordinator-policy longest_prefix
```

Only drop to `load_balanced` if you have measured that affinity routing is
producing a hot rank that affinity-vs-load tuning cannot fix.

Two tuning knobs on top:

- `--inference-dynamic-batching-prefix-caching-routing-alpha` — how hard load
  is penalized: `score = cache_score - alpha * relative_load`. `0` is pure
  affinity; larger diverts to idle ranks more readily. Dimensionless, not
  capped at 1. **Actual default is `1.0`** (the help text's "Default: 0.5" is
  stale — see `arguments.py:2274`). Lower it toward 0.25–0.5 when hit rate
  matters more than tail latency; raise it when one rank is hot.
- `--inference-dynamic-batching-prefix-cache-ttl-seconds` (default 300) — how
  long the coordinator *assumes* an engine still holds a routed block. The
  coordinator never observes evictions, so this is its only staleness
  mechanism. If your engines are under heavy eviction pressure, the
  coordinator's model of the fleet is optimistic and a lower TTL routes more
  honestly.

---

## 4. Parallelism

### The dimensions

- **TP** (`--tensor-model-parallel-size`) — splits attention heads and MLP
  weights across ranks. Every layer pays an all-reduce (or
  reduce-scatter/all-gather with SP). Reduces per-GPU weight memory and
  per-GPU FLOPs; adds per-layer latency. In decode, TP collectives are
  latency-bound, not bandwidth-bound.
- **EP** (`--expert-model-parallel-size`) — splits MoE **experts** across
  ranks. Each rank owns a disjoint expert subset; tokens are dispatched
  all-to-all to their experts and combined back. Expert weights dominate MoE
  parameter count, so EP is the cheapest way to fit an MoE model: it shards
  the big weights without splitting any individual matmul.
- **ETP** (`--expert-tensor-parallel-size`) — TP applied *within* each
  expert's weights, on top of EP.
- **SP** (`--sequence-parallel`) — shards activations along the sequence
  dimension in the regions between TP collectives (LayerNorms, dropout,
  residuals), converting the TP all-reduce into reduce-scatter + all-gather.
  Same communication volume, less activation memory, fewer redundant
  elementwise FLOPs.
- **PP** (`--pipeline-model-parallel-size`) — generally **leave at 1** for
  latency-sensitive serving. Pipeline bubbles are brutal at decode batch
  sizes.

### Recommended layout

Work in this order:

1. **Maximize EP for the DP size.** For an MoE model, set
   `--expert-model-parallel-size` as large as the layout allows (typically
   the full local world size, e.g. 8 on a node). Sharding experts is nearly
   free relative to splitting individual GEMMs.
2. **Force `--expert-tensor-parallel-size 1`.** No TP on the expert weights.
   EP already sharded them; adding ETP shrinks each expert's GEMM below
   efficient tile sizes and adds a collective inside the expert. Expert GEMMs
   at decode batch sizes are already small — don't cut them further.
3. **Then choose TP, and the choice depends on prefix caching:**
   - **If you do not care about prefix caching:** *minimize* TP. Every TP
     rank added is another per-layer collective on the decode critical path.
     Use the smallest TP that fits weights + KV cache in memory.
   - **If you do care about prefix caching:** *raise* TP to shrink DP.
     Inference DP replicas each own a **separate, non-shared prefix cache**;
     `DP = world_size / (TP × PP)`. Going TP=1 → TP=2 on 8 GPUs halves the
     number of independent caches, doubling the effective cache size per
     replica and roughly doubling the fraction of the working set resident.
     For agentic workloads, the prefill you skip usually dwarfs the extra
     per-layer collective latency. Pick the TP that makes the aggregate KV
     working set fit in one replica's buffer.

   This is a genuine tradeoff, not a default — measure `% skipped` in the log
   (§6) at each TP before committing.
4. **Always enable `--sequence-parallel`.** With TP > 1 there is no reason not
   to: same wire bytes, strictly less activation memory and elementwise work.
5. **Always use `--transformer-impl inference_optimized`.** This selects the
   inference-optimized layer implementations
   (`megatron/core/tensor_parallel/inference_layers.py`) and is a hard
   prerequisite for several other optimizations:
   - MXFP8 inference (§5),
   - `--inference-cuda-graph-scope block` with fp8,
   - `--inference-fuse-tp-communication`,
   - `--inference-disable-triton-nvls-kernels`.

   Each of those asserts `transformer_impl == "inference_optimized"` at
   startup.

The baseline script's `TP=2, ETP=1, EP=8, SP` on 8 GPUs is the concrete
instance of this recipe for a hybrid MoE model that cares about prefix cache
hits.

### NVLink domain boundaries and NVLS collectives

The inference-optimized layers use **NVLS (multicast) collectives** through
`torch.distributed._symmetric_memory` and Triton kernels
(`megatron/core/inference/communication/torch_symm_triton/`,
`megatron/core/inference/symmetric_memory.py`). These are substantially
faster than NCCL ring collectives at decode message sizes because the switch
does the reduction in-network.

Requirements (`are_tensors_nvls_eligible`,
`torch_symm_triton/utils.py:26`):

- **Hopper or newer** (SM ≥ 9).
- Tensor byte sizes divisible by 16 — the kernels move 128-bit chunks.
- A successful `symm_mem.rendezvous()` on the process group.

**The constraint that bites: symmetric memory requires all ranks in the group
to be in the same NVLink/NVSwitch domain.** Inside a node (or inside an NVL72
domain on GB200) that holds. The moment a TP or EP group **crosses the NVL
boundary** — TP spanning two DGX nodes over InfiniBand, or an EP group
straddling two NVL domains — the rendezvous fails or the multicast path is
unavailable, and you fall back to NCCL over the slow link. The fallback is
graceful (`SymmetricMemoryBuffer` records `init_failure_reason` and the
call sites check `are_tensors_nvls_eligible` before using the NVLS path), so
**you will not get an error — you will silently get much worse decode
latency.**

Guidance:

- **Keep TP inside one NVLink domain.** Never let a TP group cross nodes for
  a latency-sensitive serving job. TP collectives are on the per-layer
  critical path; over IB they will dominate step time.
- **Keep EP inside one NVLink domain** where possible. The MoE all-to-all is
  per-layer too.
- To scale past one domain, scale **DP** (independent replicas, no
  cross-domain collectives at all) rather than widening TP or EP. Note the
  prefix-cache cost of more DP replicas from §4.3 and set the coordinator
  policy accordingly.
- If NVLS misbehaves (Triton/driver issues, correctness suspicion), disable
  it explicitly with `--inference-disable-triton-nvls-kernels` rather than
  guessing — it requires `--transformer-impl inference_optimized`.
- Confirm your job is actually getting the fast path rather than assuming it:
  check the startup logs for symmetric-memory init failure reasons, and
  compare decode step time against a single-node run at the same TP.

---

## 5. Numerics

### MXFP8 inference

MXFP8 (microscaled FP8, 32-element blocks) roughly halves weight memory and
bandwidth for the linear layers. Weight-bandwidth is the decode bottleneck,
so this is a direct throughput win, and the microscale blocks keep accuracy
much closer to BF16 than per-tensor FP8.

Enable with:

```bash
--transformer-impl inference_optimized
--fp8-format e4m3
--fp8-recipe mxfp8
```

The quantization happens post-load: `megatron/inference/utils.py:114` checks
`transformer_impl == "inference_optimized" and fp8_recipe == "mxfp8"`, then
calls `quantize_model_to_mxfp8()` on the loaded checkpoint. **You load a BF16
checkpoint and quantize at startup** — no separately-quantized checkpoint
needed.

Requirements and interactions:

- **FlashInfer ≥ 0.6.4** with inference-optimized layers, else startup raises
  (`arguments.py:1119`).
- With `--inference-cuda-graph-scope block`, fp8 is **only** supported as
  `mxfp8` and **only** with `inference_optimized`
  (`arguments.py:1255-1260`). Both asserts fire at startup.
- `--inference-grouped-gemm-backend` selects the MoE kernel and interacts
  with MXFP8 (`transformer_config.py:1301`):
  - `vllm` (default) — vLLM Triton fused MoE, **BF16 only**.
  - `torch` — `grouped_mm`; supports BF16 **and** MXFP8.
  - `flashinfer` — TRT-LLM block-scale MoE for MXFP8. Fastest MXFP8 MoE path,
    but it keeps *both* canonical expert weights (for refit) and a padded
    TRT-LLM Major-K copy, so **expert-weight memory goes up** relative to
    `torch`. Budget for that before switching.

  For an MXFP8 MoE model, pick `flashinfer` if you have the memory headroom,
  `torch` otherwise. The default `vllm` will not give you MXFP8 experts.

### Mamba / GDP state precision

```bash
--mamba-inference-ssm-states-dtype fp32
```

Default is `bf16`. **Set this to `fp32` on any Mamba/GDP/hybrid model.**

The SSM state is a *recurrent accumulator*: it is updated in place across
every token in the sequence. Unlike attention KV — where each position's
error stays local — SSM rounding error compounds multiplicatively down the
sequence. At bf16's ~8 bits of mantissa, long-context generation drifts, and
the failure mode is nasty: no error, just degraded output quality that gets
worse the longer the conversation runs. Exactly the regime agentic workloads
live in.

The cost is small: the state is per-request-per-layer, not per-token, so
fp32 doubles a modest allocation rather than the KV cache. Note that it *does*
double the per-slot cost of the Mamba prefix cache, so recheck durable slots
in the log after flipping it.

`--mamba-inference-conv-states-dtype` (also `bf16` by default) covers the
short convolution state. That one is a fixed-width window, not an
accumulator, so bf16 is fine; raise it only if you are chasing a specific
numerical discrepancy.

---

## 6. Reading the Inference Log

### Enabling

```bash
--inference-logging-step-interval 100
--logging-level 20
```

`--inference-logging-step-interval` defaults to `0` (disabled). It counts
engine steps, so at 100 a decode-heavy server prints every few hundred ms.
Set it lower (10–50) while tuning, raise it for production. `--logging-level
20` is `logging.INFO`, which the step line is emitted at.

Related: `--inference-wandb-logging` mirrors these as `inference/*` metrics;
`--inference-text-gen-server-logging` adds per-request logs.

### Anatomy of a step line

Emitted from `megatron/core/inference/engines/dynamic_engine.py:3163`.

```
INFO:root:* rank 1 | step 8400 | 23:44:22 ... time: 6.547 ms
  [decode + real config [4]: 0 P + 4 D + cuda graph [4]: 0 P + 4 D]
  ... reqs: a 4/4, p 0, w 23, f 247, e 0
  ... blocks: occupied 1521/43671, allocatable 43388, active-used 283, paused-used 0/0
  ... mem: tensors 550, alloc 97.2 gb, res 99.0 gb.
  ... prefix cache (cumul): 250 hits, 11475 blocks matched
  ... prefill (cumul): computed 423697, skipped 2936064 (87.4% skipped)
  ... prefix cache util: KV 1515/43672 blocks cached (1238 evictable), mamba 253/15184 durable slots
```

| Field | Meaning |
|---|---|
| `time: 6.547 ms` | wall time for this engine step |
| `[decode + ...]` | step type; `real config` = actual batch dims, `cuda graph` = padded dims replayed. `cuda graph [...]: OFF` means this step ran **eager** |
| `0 P + 4 D` | 0 prefill requests + 4 decode requests in the step |
| `reqs: a 4/4` | **active / max_requests** |
| `p 0` | paused |
| `w 23` | **waiting** (queued, not admitted) |
| `f 247` | finished (cumulative) |
| `e 0` | **evicted** (cumulative) |
| `blocks: occupied 1521/43671` | occupied / usable KV blocks |
| `allocatable 43388` | blocks currently available to allocate |
| `active-used 283` | blocks held by active requests |
| `paused-used 0/0` | blocks held by paused requests / paused block budget |
| `mem: alloc / res` | torch allocated / reserved GB |
| `prefix cache (cumul)` | cumulative hits and blocks matched |
| `prefill (cumul): ... (87.4% skipped)` | **the headline prefix-caching metric** — fraction of prompt tokens whose prefill was reused |
| `prefix cache util: KV 1515/43672 (1238 evictable)` | blocks currently holding cached prefixes; evictable = cached with refcount 0 |
| `mamba 253/15184 durable slots` | Mamba state slots in use / available |

Reading the example line: `a 4/4` with `w 23` and `occupied 1521/43671` is
the textbook **`max_requests` bottleneck** — 23 requests are queued while 97%
of the KV pool sits idle, purely because `max_requests` is 4.

### Diagnostic table

| Symptom in the log | Diagnosis | Fix |
|---|---|---|
| `reqs: a N/N` pinned at max **and** `w` > 0 **and** `occupied` ≪ pool | Request-slot starved, not memory starved. Concurrency is capped artificially. | **Raise `--inference-dynamic-batching-max-requests`.** Push it until `occupied/pool` climbs or `w` drains. Keep `max_tokens >= max_requests`. |
| `w` > 0 and `occupied` ≈ pool, `allocatable` small | Genuinely memory-bound. | Raise `--inference-dynamic-batching-buffer-size-gb`; or shrink per-request footprint via `--inference-max-seq-length`; or add DP. |
| `occupied` ≪ pool and `alloc`/`res` far below GPU capacity | **Underutilizing GPU memory.** The buffer is smaller than it could be. | Raise `--inference-dynamic-batching-buffer-size-gb` until `res` sits ~5–8 GB under device capacity. More blocks = more concurrency *and* a bigger prefix cache. |
| `e` (evicted) climbing steadily | Requests are being evicted under block pressure — work is being thrown away and redone. | First raise `buffer-size-gb`. If capped, raise `--inference-dynamic-batching-paused-buffer-size-gb` so paused requests retain blocks instead of being dropped; consider `--inference-dynamic-batching-unified-memory-level 1` to spill to host memory. Lowering `max_requests` also helps — fewer admitted requests fight for blocks. |
| `p` (paused) persistently high | Requests admitted then parked for lack of blocks. Same root cause as eviction, one stage earlier. | Same fixes; also check `paused-used X/Y` — `X` at `Y` means the paused budget is the binding constraint. |
| `% skipped` low (< 30%) on a workload that should share prefixes | **Low prefix cache hit rate.** | Confirm prefix caching is on; confirm eviction policy is `lru` not `ref_zero`; with DP > 1 switch the coordinator policy to `longest_prefix` and lower `routing-alpha` toward 0.25–0.5; on hybrid models check `mamba durable slots` — if saturated, the KV matched but the Mamba state didn't and prefill ran anyway. |
| `% skipped` high but step `time` still growing | **Not a caching problem.** With a long resident context, attention over the growing KV dominates. (The engine's own comment at `dynamic_engine.py:3234` calls this out.) | Nothing to fix in the cache. Look at attention backend, TP, and whether `max-seq-length` is larger than the workload needs. |
| `KV X/Y blocks cached` with `evictable` near 0 | Cache is full of *live* blocks; nothing can be reclaimed. New prefixes will evict aggressively. | Raise the buffer, or lower `max_requests` so fewer blocks are pinned live. |
| `mamba N/M durable slots` with N ≈ M | **Mamba state cache saturated.** Cached prefixes will start LRU-evicting their states, and hybrid prefill can only be skipped where state survives. | Raise `--inference-dynamic-batching-prefix-caching-mamba-gb`; or shrink the reserved scratch by lowering `--inference-dynamic-batching-max-tokens` or `max_requests` (scratch is reserved before durable slots — see §3). |
| `cuda graph [...]: OFF` on decode steps | Decode running eager. Worst case for latency. | Check `--cuda-graph-impl local` is set and the batch is within the captured range; set `--inference-dynamic-batching-num-cuda-graphs -1`. |
| `cuda graph` dims much larger than `real config` dims | Excessive graph padding — you're paying for tokens you don't have. | `--inference-dynamic-batching-num-cuda-graphs -1`; consider the sizing distribution. |
| `cuda graph [...]: OFF` on prefill/mixed steps only | Prefill exceeds the largest captured prefill graph. | Raise `--inference-cuda-graph-max-tokens` (default 512), or set `--inference-cuda-graph-all-prefills`. |

**Tuning loop:** raise `max_requests` until `w` stops draining or memory
binds → raise `buffer-size-gb` until `res` approaches capacity → confirm
`% skipped` for the caching config → confirm graphs are on and padding is
small. Change one thing per run; every one of these knobs moves the others.

---

## 7. Miscellaneous Performance Flags

### `--skip-prompt-log-probs` — set this

```bash
--skip-prompt-log-probs
```

By default, if a client requests log probs the engine must materialize logits
for **every prompt token**, not just the generated one. That tensor is
`[prompt_tokens, vocab_size]` — with a 32K prompt and a 128K vocab it is
gigabytes per request, in the KV buffer's memory, for data almost nobody
reads.

`--skip-prompt-log-probs` says "return log probs for generated tokens only,"
which lets the engine keep `materialize_only_last_token_logits=True` even
when `--return-log-probs` is set
(`megatron/training/config/inference_config.py:264`). The memory reduction is
dramatic and it frees buffer space that becomes KV blocks and prefix cache.

Set it unless you are specifically doing prompt scoring / perplexity
evaluation. Megatron's own perf test harness pins it
(`tests/performance_tests/shell_test_utils/run_perf_test.sh:206`).

### Reasoning and tool-call parsers

```bash
--parsers nemotron-v3-reasoning qwen3-coder-tool
```

`--parsers` takes a space-separated list, resolved through `PARSER_MAPPING`
(`megatron/core/tokenizers/text/parsers/__init__.py`). Registered names:

- `nemotron-v3-reasoning` — strips/structures Nemotron v3 reasoning traces.
- `qwen3-coder-tool` — parses Qwen3-Coder tool-call syntax into structured
  tool calls.
- `deepseek-r1-reasoning` — DeepSeek-R1 reasoning traces.

Listing both `nemotron-v3-reasoning` and `qwen3-coder-tool` is the standard
pairing for Nemotron-family agentic serving: one handles the reasoning
channel, the other the tool-call channel, and they are complementary rather
than exclusive. Without them the client receives raw text containing
reasoning markers and unparsed tool syntax, and every downstream consumer
reimplements the parsing.

Parsers run in the server frontend, not the engine — no step-time cost.

### Other knobs worth knowing

- `--inference-dynamic-batching-async-sched-mode async` (default) — overlaps
  scheduling phases by reordering to prepare-before-resolve. Keep it;
  `legacy` exists to bisect scheduling bugs.
- `--inference-dynamic-batching-sampling-backend flashinfer` — faster
  sampling kernels. Falls back to `torch` with a warning if FlashInfer is
  absent, so it is safe to pass.
- `--inference-use-synchronous-zmq-collectives` — reduces performance
  *variability* for MoE models. Try it when p99 is noisy but median is fine.
- `--inference-dynamic-batching-unified-memory-level 1` — allocates the
  memory buffer in unified memory so paused requests can spill to host.
  Trades bandwidth for capacity; useful against eviction pressure.
- `--inference-dynamic-batching-block-size` (default 256, must be a multiple
  of 256) — smaller blocks reduce internal fragmentation and give finer
  prefix-match granularity; larger blocks reduce block-table overhead. The
  default is nearly always right.
- `--num-speculative-tokens` — speculative decoding. When enabled, the log
  gains a `spec (cumul): accept X%` field with per-position rates; an
  acceptance rate below ~50% usually means the draft is not earning its cost.

---

## Quick Reference

| Goal | Flags |
|---|---|
| Kill launch overhead | `--cuda-graph-impl local --inference-cuda-graph-scope block --inference-dynamic-batching-num-cuda-graphs -1` |
| Long contexts without stalling decode | `--enable-chunked-prefill --inference-dynamic-batching-max-tokens 4096` + large `--inference-max-seq-length` |
| Reuse work across turns | `--inference-dynamic-batching-prefix-caching --...-eviction-policy lru --...-coordinator-policy longest_prefix` |
| Prefix caching on hybrid models | add `--inference-dynamic-batching-prefix-caching-mamba-gb 20` |
| MoE parallelism | `--expert-model-parallel-size <world> --expert-tensor-parallel-size 1 --sequence-parallel` |
| Fast layers + NVLS | `--transformer-impl inference_optimized`, keep TP/EP inside one NVLink domain |
| Halve weight bandwidth | `--transformer-impl inference_optimized --fp8-format e4m3 --fp8-recipe mxfp8` (+ `--inference-grouped-gemm-backend flashinfer\|torch` for MoE) |
| Robust Mamba numerics | `--mamba-inference-ssm-states-dtype fp32` |
| Cut logits memory | `--skip-prompt-log-probs` |
| Structured agent output | `--parsers nemotron-v3-reasoning qwen3-coder-tool` |
| See what's happening | `--inference-logging-step-interval 100 --logging-level 20` |

### Source map

| Topic | File |
|---|---|
| All inference CLI args | `megatron/training/arguments.py` (`_add_inference_args`, ~line 2155) |
| CUDA graph impl/scope validation | `megatron/core/transformer/cuda_graph_config.py`, `megatron/core/transformer/enums.py:101` |
| Context sizing, `max_tokens`/`max_requests` | `megatron/core/inference/contexts/dynamic_context.py:680-730` |
| Step logging | `megatron/core/inference/engines/dynamic_engine.py:3155-3260` |
| MXFP8 quantization at load | `megatron/inference/utils.py:114`, `megatron/core/inference/quantization/` |
| NVLS / symmetric memory | `megatron/core/inference/symmetric_memory.py`, `megatron/core/inference/communication/torch_symm_triton/utils.py:26` |
| Inference-optimized layers | `megatron/core/tensor_parallel/inference_layers.py` |
| Parser registry | `megatron/core/tokenizers/text/parsers/__init__.py` |
| Server entrypoint | `tools/run_dynamic_text_generation_server.py` |
| Perf test harness | `tests/performance_tests/` |
