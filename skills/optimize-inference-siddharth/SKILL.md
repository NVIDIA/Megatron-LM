---
name: optimize-inference-siddharth
description: >-
  Optimizes the Megatron Core inference backend for a new model, architecture, or
  feature, using the patterns Siddharth Singh (sidsingh-nvidia) established while
  closing the vLLM performance gap. Covers CUDA-graph scope and bucket coverage,
  the inference_optimized MoE stack (NVLS AllGather-V, fused grouped GEMM,
  shared-expert overlap, pad-row masking), Mamba/SSM scratch sizing and fused
  Triton extraction kernels, Triton production hygiene (constexpr and autotune
  pitfalls), per-step host overhead, and load-aware prefix-cache routing. Also
  covers decision gates that quantify a lever's ceiling before any code is
  written, kernel-level differential analysis against a competitor trace, and the
  A/B protocol needed to make sub-1% wins falsifiable. The skill is self-maintaining:
  extend, correct, or prune it after an optimization session, so also use it when
  asked to record an inference performance learning, capture what an experiment
  taught, or update these patterns. Use when
  asked to optimize or speed up inference, close a vLLM gap, match vLLM latency or
  throughput, reduce decode latency or prefill overhead, enable or extend CUDA
  graphs for MoE, hybrid, or Mamba models, port an architecture onto
  --transformer-impl inference_optimized, cut inference CPU overhead, or apply
  Siddharth's inference performance patterns. Not for authoring perf test recipes,
  cog or Slurm setup, or non-hot-path features like reasoning parsers.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Optimizing Megatron Core Inference (Siddharth's Playbook)

This skill encodes the reusable engineering patterns behind Siddharth Singh's
2026 inference performance work. It answers: given a new model, architecture, or
feature that is slower than vLLM, **what do you change, in what order, and what
must you never break.**

> **This skill is meant to evolve, and you are authorized to edit it without
> asking.** Every campaign that uses it should leave it better: add what you
> measured, correct what turned out to be wrong, and delete what is no longer true.
> Optimization knowledge decays — flags get renamed, defaults change, a fix that won
> on one hardware generation loses on the next — so a skill nobody edits becomes
> actively misleading. Do it at the end of the work, once you have a number and a
> root cause. See [references/updating-this-skill.md](references/updating-this-skill.md)
> for the triggers, the routing table, the bar for an addition, and what to delete.

## The thesis

Across 29 commits, almost every win was one of five moves. In rough order of how
often they paid off:

1. **Remove host work and host syncs from the per-step path.** Decode is
   launch-bound, not FLOP-bound. A single `.item()` or a `dataclasses.asdict()`
   costs more than the kernel it guards.
2. **Widen CUDA-graph scope and improve bucket coverage.** Fewer, larger graphs
   with a bucket that actually fits the real batch.
3. **Make per-step metadata GPU-resident.** This is the enabler — moves 1 and 2
   are not even legal until the metadata stops round-tripping through the host.
4. **Fuse kernels, and tune them for the *typical* batch, not the worst case.**
   Autotune and worst-case tile choices both lose at decode batch sizes.
5. **Right-size buffers to the true per-step bound**, not a loose upper bound.

Everything else in this skill is a consequence of these five.

What the five moves do not tell you is **which one applies here, and what it is
worth**. Getting that wrong is the expensive failure mode — not writing a bad
kernel, but writing a correct kernel whose best possible outcome was 1%. That is
what Step 1 is for.

## Step 1: Measure, classify, and gate before changing anything

Do not skip this. Every commit in this history started from a profile, and the
fix location is rarely where you would guess.

See [references/measuring.md](references/measuring.md) for the profiler
endpoints, NVTX ranges, built-in counters, honest idle accounting, and — read this
before adding host visibility — **the trace flags that deadlock nsys finalization
on MoE decode workloads**. For trace analysis use the `nsight-system-analysis`
skill; for the throughput harness use `run-inference-performance-tests`.

Classify the dominant signal, then jump to the matching section:

| Dominant signal in the profile | Where to work |
|---|---|
| GPU idle gaps between kernels; host ahead of device | Step 2, CUDA graphs |
| Host time after the forward (serialize, detokenize, ZMQ) | Step 5, host path |
| Exposed NCCL, or EP ranks waiting on each other | Step 3, NVLS AllGather-V and `ep_consensus_interval` |
| Periodic multi-ms stalls, or slow first steps | Step 4, Triton recompilation and autotune |
| Grouped GEMM inefficient at small batch | Step 3, grouped-GEMM backend and tile heuristic |
| OOM, or a prefix cache far smaller than expected | Step 4, scratch sizing math |
| Many sub-microsecond kernels in a row | Step 3, fusion — but gate it first |

### Then gate the lever: share is not headroom

A category's share of device time is not what you can win back, for two reasons that
have each burned real weeks. A category can be 33% of device time with **2% of
headroom**, because it is already moving the bytes it has to move. And **device time
is not wall time** — kernels overlap across streams, so the step's wall clock is the
critical path through the per-layer dependency chain; work on a side stream
contributes less than its share, while a tiny kernel on the serial chain costs its
duration *plus* the dispatch gap behind it, every layer.

So before writing anything that takes more than about a day, compute the ceiling:
establish the floor from measured machine constants, measure the current cost under
**graph replay** at the decode token count, take the ratio, subtract what the fix
itself costs (added launches, grid syncs, atomics, extra passes), and express the
result as a fraction of the step. Then write down **proceed** or **gated out**.

Full method, the per-launch fixed costs you need, and three case studies where a
gate killed a multi-week effort — including a hand-written grouped GEMM whose
entire ceiling was 1.45×, most of it reachable by tile tuning alone — are in
[references/decision-gates.md](references/decision-gates.md).

Skip the gate for cheap reversible changes: flag flips, tile retunes, backend enum
swaps. Gating those costs more than trying them.

### If the target is a competitor, diff against its trace

When the goal is "match vLLM," its trace is a specification, not just a scoreboard.
Take matched windows of one forward block from each and answer one question: **is
any individual kernel slower, or does mcore just launch more of them?** The fixes
are opposite. On Qwen3-30B the answer was ~467 kernels against ~1784 for the same
block, which pointed at fusion and away from kernel rewrites.

Method, window anchoring, and the traps in adopting a competitor's kernel are in
[references/vllm-differential.md](references/vllm-differential.md).

### Open a ledger before the first change

For anything spanning more than one session, keep an append-only ledger: a fixed
protocol table (hardware, model, batch, output length, parallelism, warmup/timed
counts), one row per experiment **including every rejection with its root cause**,
a running distance-to-target, and a next-levers list re-derived after each profile.
Never edit a recorded result; supersede it. See
[references/measuring.md](references/measuring.md). The negative results are the
higher-value half — they are what stops the next person re-deriving a dead end.

## Step 2: CUDA-graph the largest region that is safe

Full detail in [references/cuda-graphs.md](references/cuda-graphs.md).

Set `cuda_graph_impl="local"` and `inference_cuda_graph_scope=block` to capture
the whole decoder block in one graph. For hybrid models, graph ownership lives on
`HybridModel`, not the stack, so the embedding and output layers land inside the
same capture — that widening was itself a measurable win.

Then fix coverage. A step whose shape matches no captured bucket silently falls
back to eager, so a "CUDA graphs enabled" run can still be launch-bound. Check
`num_cuda_graphs` (`-1` auto-sizes), `cuda_graph_max_tokens` (512 by default, so
prefill and mixed steps up to 512 tokens get a graph), and
`cuda_graph_sizing_distribution`.

Know what the wide capture costs, though — it is a trade, not a free win. Under
full-iteration inference capture, the flashinfer sampling backend cannot run, async
scheduling is guarded off for EP (and deadlocks if you open the guard), and
multi-stream comm/compute overlap is bounded by the graph's structure rather than by
hardware queues, so `CUDA_DEVICE_MAX_CONNECTIONS` does nothing. Details and error
signatures are under *What a wide capture costs you* in
[references/cuda-graphs.md](references/cuda-graphs.md).

## Step 3: Enable the model-path stack

**MoE** — see [references/moe-inference.md](references/moe-inference.md).
Use `--transformer-impl inference_optimized`, which swaps in
`InferenceTopKRouter` and `InferenceGroupedMLP` and picks a dispatcher from
`inference_moe_token_dispatcher_type`. Prefer `nvls` (the default): the NCCL
fallback requires equal token counts across EP ranks, which forces decode-only
graphs. Then check the grouped-GEMM backend, shared-expert overlap, and that
padding rows route to no expert.

**Hybrid / Mamba** — see
[references/mamba-and-triton.md](references/mamba-and-triton.md).
Size the extraction scratch to the real per-step bound before the durable prefix
cache claims the rest, and use fused gather-plus-scatter kernels gated on a
runtime count instead of materializing intermediates.

**Dense GPT** — Steps 2, 4, and 5 usually cover it. Still apply the padding and
dtype contracts to any fused kernel you add.

## Step 4: Kernel and Triton hygiene

Full detail in [references/mamba-and-triton.md](references/mamba-and-triton.md).
The two rules that cause the most damage when violated are the `tl.constexpr`
specialization rule and the no-autotune-in-production rule, both in the hard
rules below.

## Step 5: Trim the host path

Full detail in [references/host-path.md](references/host-path.md). The per-step
host critical section is the `bookkeeping` to `detokenization` to
`coordinator_communication` span in
[dynamic_engine.py](megatron/core/inference/engines/dynamic_engine.py). Anything
there runs once per step and blocks the next one.

## Step 6: Re-measure, keep a kill switch, and feed the result back

Re-run the same measurement from Step 1 with the identical config. Then confirm
correctness with `run-inference-functional-tests`, and regenerate golden values
if you changed bucketization (padding changes shift outputs — expected, not a
bug).

Re-measure **in the same allocation, back to back, arms alternating** — identical
configs drifted by up to 1.6% between sessions on this workload, which is larger
than most individual wins, so a cross-session comparison will mislead you in both
directions. Accept on **distribution separation** (slowest ON beats fastest OFF),
not on mean delta. Protocol and worked examples: *The noise floor is bigger than
your win* in [references/measuring.md](references/measuring.md).

Expect the end-to-end result to differ from the kernel-level result, often by
several times in either direction. Work on the serial dependency chain converts at
more than 1:1 (a ~1% microbench ceiling delivered +2.9%, because removing a launch
also removes a graph node and a dispatch gap ×48 layers); already-overlapped work
converts at a third to a half; work off the critical path converts at roughly zero
(a 1.25× kernel win delivered a wash). If the measured conversion is far from your
prediction, you mis-identified where the work sits.

Every optimization here shipped with a way to turn it off
(`inference_disable_triton_nvls_kernels`,
`inference_moe_disable_fused_quant_kernels`, the backend enums). Add one. It is
how the next person A/Bs your change instead of reverting it.

Then close the loop: append the result to the campaign ledger — **including
rejections, with their root cause** — and promote whatever generalizes into this
skill. The promotion test is whether it would change what someone does on a
*different* model; if it only describes this one, it stays in the ledger. Mechanics
in [references/updating-this-skill.md](references/updating-this-skill.md).

---

## Hard rules

These are non-negotiable. Each one exists because violating it broke something.

### 1. No host sync on the per-step path

No `.item()`, `.tolist()`, `.cpu()`, or data-dependent Python branch in code that
runs every step. To publish a per-step scalar, `fill_` it into a **preallocated,
fixed-address** GPU tensor and read it inside the kernel:

```python
# megatron/core/inference/contexts/attention_context/mamba_metadata.py
# fill_ is async (no host sync) and keeps the tensor at the same address
# captured graphs reference.
self._intermediate_real_count_buffer.fill_(self.intermediate_count)
```

The fixed address is the whole point: a captured graph records pointers, so the
*value* may change between replays but the *address* may not.

### 2. `tl.constexpr` only for values fixed for the process lifetime

A `tl.constexpr` parameter is baked into the compiled kernel, so **every distinct
value triggers a fresh JIT compile.** Marking a per-step batch or token count
`constexpr` means recompiling on every step. That was the entirety of commit
`f29c747` — a four-line fix worth a large latency spike.

Constexpr is correct for block and tile sizes, `tl.arange` bounds, and unroll
counts. It is wrong for anything that varies per step. Note that even plain
`int` arguments get specialized on divisibility-by-16 and `== 1`; if such a value
varies, use `@triton.jit(do_not_specialize=["batch"])`.

### 3. Graph-safe kernel shape: worst-case grid, data-conditional body

Size the launch grid at capture time to the maximum, then let padded programs
exit immediately:

```python
# megatron/core/ssm/ops/intermediate_extraction.py
real_count = tl.load(real_count_ptr).to(tl.int32)
if pid_slot >= real_count:
    return
```

This keeps the grid static for replay while making padded slots nearly free.

### 4. Size per-step work to the matched bucket, not the global max

Once a step matches a CUDA-graph bucket, all metadata updates and scratch writes
should be bounded by that bucket, never by global `max_tokens` or `max_requests`.
Commit `9b4074b` was largely this one change applied to Mamba prefill.

### 5. Padding must not do real work

Three concrete forms:

- CUDA-graph pad rows get routing index `-1` (via `mask_routing_padding`) so they
  activate no expert.
- Pad tokens point at a reserved `dummy_block_idx` so KV-append writes somewhere
  valid and throwaway.
- Do not *zero* rows past `valid_tokens`. Downstream only reads the valid prefix,
  so a zeroing pass is pure wasted bandwidth — just don't write them.

### 6. Tune for the expected batch; never autotune in production

Pick tile, warp, and stage counts from a host-side heuristic keyed on the typical
token count. Commit `20f09364` replaced 25 `@triton.autotune` configs with
vLLM's `_get_default_config` heuristic, which cut compile time and picked better
tiles for decode-sized batches. Route any remaining autotuning through
`autotune_configs` in
[determinism.py](megatron/core/ssm/ops/determinism.py).

### 7. Preserve Transformer Engine `nn.Parameter` identity

Grouped GEMM wants one stacked `[num_experts, out, in]` weight; TE stores
per-expert parameters. Redirect `param.data` to a view into the stacked buffer
rather than replacing the `Parameter` object, and build it **lazily on first
forward**, after checkpoint load:

```python
# megatron/core/transformer/moe/experts.py
# Redirect param.data to view into contiguous buffer.
# The nn.Parameter object stays the same - TE's internal state is preserved.
fc1_param.data = _fc1_weight[i]
```

Replacing the object corrupts TE's FP8 and bookkeeping state; building eagerly at
`__init__` reads weights that do not exist yet. Commit `905c0e38` fixed both
after they broke the RL integration.

### 8. Hold the dtype contract at every kernel boundary

Token ids are `int64` throughout the pipeline. FlashInfer's sampling kernels
return `int32`, so their results are cast with `.long()` at the boundary.
Normalize dtype where the external kernel enters, not by letting it propagate —
the cast is free once, a mismatch inserts conversion kernels on every step.

Related, and a sharper trap: **not everything belongs in a graph.** FlashInfer
sampling is deliberately left eager, because its kernel choice is data-dependent
*and* FlashInfer bakes the philox RNG state into a graph as a by-value constant at
capture — so a captured sampler replays identical random numbers every step. Check
for baked-in state before widening a graph over anything stateful or random.

### 9. Guard correctness with a reference implementation

The testing pattern throughout: compare the fast kernel against an obvious
PyTorch or plain-loop reference with `torch.testing.assert_close`; assert the
gating behavior explicitly by prefilling the output with a sentinel and checking
padded slots still hold it; and re-derive any sizing formula independently in the
test so a changed bound fails loudly. See `add-inference-unit-tests`.

### 10. A sub-1% claim needs same-session, back-to-back, separated arms

Session-to-session drift on identical configs reached 1.6% — larger than most
individual wins once the easy ones are gone. So a number compared against last
session's baseline is not evidence.

Re-baseline the current best config **in the allocation you will test in**, run the
OFF and ON arms **back to back**, repeat the pair, and accept only if the arms do
not overlap:

```
min(ON iterations) > max(OFF iterations)
```

Report pairwise deltas rather than one average, and treat the first timed iteration
as a suspected cold outlier. A +1% mean with overlapping arms is not a result; a
+0.9% with separated arms is.

### 11. If it is not bit-exact, say so and justify it differently

Bit-exactness is achievable more often than assumed — retuning tiles while holding
`BLOCK_SIZE_K` fixed preserves the fp32 reduction order, and a masked slot adding an
exact 0.0 keeps a fused reduction bit-exact. **Check whether a bit-exact formulation
exists before accepting drift.**

When none does — norm and reduction fusions frequently land one ulp off, since TE's
internal rsqrt and reduction order differ — the acceptance argument changes shape and
must be made explicitly: bound the deviation in ulps against the reference across
several token counts and seeds (`max_rel ≤ 7.9e-3` is about one bf16 ulp); diff
fixed temperature-0 coherence output against the gate-OFF arm; and for any prompt
that diverges, **inspect where**. A divergence at a genuinely low-confidence branch
with both continuations fluent and factually correct is acceptable; one that degrades
fluency or correctness is not. Record which prompts diverged — "not bit-exact" must
be lookup-able, not rediscovered.

Never let an ulp-level deviation ride on `assert_close` alone: the tolerance that
passes it also passes a real bug.

---

## Code entry points

| Area | Paths |
|---|---|
| MoE dispatchers | [token_dispatcher_inference.py](megatron/core/transformer/moe/token_dispatcher_inference.py) — `NCCLAllGatherDispatcher`, `NVLSAllGatherVDispatcher` |
| MoE experts / router | [experts.py](megatron/core/transformer/moe/experts.py) `InferenceGroupedMLP`, [router.py](megatron/core/transformer/moe/router.py) `InferenceTopKRouter`, [moe_layer.py](megatron/core/transformer/moe/moe_layer.py) |
| Fused MoE kernels | [megatron/core/inference/moe/](megatron/core/inference/moe/) — `fused_moe.py`, `vllm_fused_moe.py`, `permute.py`, `activations.py`, `metadata.py` |
| Pad-row masking | [inference_routing_mask_kernel.py](megatron/core/transformer/moe/inference_routing_mask_kernel.py) |
| Backend selection | [backends.py](megatron/core/models/backends.py), [moe_module_specs.py](megatron/core/models/gpt/moe_module_specs.py) |
| CUDA-graph buckets | [batch_dimensions_utils.py](megatron/core/inference/batch_dimensions_utils.py) |
| Graph scope hooks | [enums.py](megatron/core/transformer/enums.py), [transformer_block.py](megatron/core/transformer/transformer_block.py), [hybrid_model.py](megatron/core/models/hybrid/hybrid_model.py) |
| Per-step context state | [dynamic_context.py](megatron/core/inference/contexts/dynamic_context.py), [gpu_view.py](megatron/core/inference/contexts/gpu_view.py) |
| Mamba / SSM | [mamba_mixer.py](megatron/core/ssm/mamba_mixer.py), [intermediate_extraction.py](megatron/core/ssm/ops/intermediate_extraction.py), [mamba_metadata.py](megatron/core/inference/contexts/attention_context/mamba_metadata.py), [mamba_slot_allocator.py](megatron/core/inference/contexts/mamba_slot_allocator.py) |
| Symmetric memory / NVLS | [symmetric_memory.py](megatron/core/inference/symmetric_memory.py), [torch_symm_triton/](megatron/core/inference/communication/torch_symm_triton/), [inference_layers.py](megatron/core/tensor_parallel/inference_layers.py) |
| Engine / host path | [dynamic_engine.py](megatron/core/inference/engines/dynamic_engine.py), [inference_request.py](megatron/core/inference/inference_request.py), [data_parallel_inference_coordinator/](megatron/core/inference/data_parallel_inference_coordinator/) |

## Flags

Verified against the current tree. Defaults matter — several are already the
tuned value, so the useful move is often *checking* rather than changing them.

| Flag | Default | Effect |
|---|---|---|
| `transformer_impl` | `transformer_engine` | `inference_optimized` swaps in the whole inference MoE stack |
| `inference_moe_token_dispatcher_type` | `'nvls'` | `nccl` fallback forces equal EP token counts and decode-only graphs |
| `inference_grouped_gemm_backend` | `"vllm"` | `vllm`, `flashinfer`, or `torch`; MXFP8 needs `torch` |
| `inference_disable_triton_nvls_kernels` | `False` | Kill switch: fall back to NCCL collectives |
| `inference_moe_disable_fused_quant_kernels` | `False` | Kill switch: unfuse activation + quantize |
| `moe_router_dtype` | — | Must be `fp32` for `inference_optimized`, to avoid per-decode dtype conversion |
| `cuda_graph_impl` | `"none"` | Inference graphs need `local` |
| `inference_cuda_graph_scope` | derived | `none`, `layer`, or `block`; `local` derives `layer`, so set `block` explicitly |
| `cuda_graph_max_tokens` | `512` | Token ceiling for prefill and mixed graphs |
| `num_cuda_graphs` | `16` | `-1` auto-sizes from `log2(max_tokens)` |
| `cuda_graph_sizing_distribution` | `EXPONENTIAL` | `LINEAR` gives the dense small-batch ladder |
| `ep_consensus_interval` | `20` | Skip EP consensus all-reduces while busy |
| `prefix_caching_routing_alpha` | `0.5` | 0 is pure load balance, 1 is pure cache affinity |
| `SamplingParams.return_prompt_tokens` | `False` | Opt in to echoing prompt ids over the wire |
| `moe_enable_routing_replay` | `False` | Record per-token expert choices for imbalance analysis |

### Flags that look like free wins and are not

Measured and rejected on Qwen3-30B-A3B EP4 under `inference_optimized` + block
scope. Each is the sort of thing you would reasonably try first.

| Flag | Result |
|---|---|
| `--moe-router-fusion`, `--moe-permute-fusion` | **Crash**: `AssertionError: hidden_size mismatch: 128 vs 8`. Wired to the training MoE path; TE's fused router emits a dense `num_experts` map, not the dense top-k the inference dispatcher needs. |
| `--inference-dynamic-batching-sampling-backend flashinfer` | **Crash** under graph capture: `Generator not registered with the capturing graph`. |
| `--inference-dynamic-batching-async-sched-mode serial` | Guarded off for EP; opening the guard deadlocks the first single-request decode step. Gains +0.85% at batch — the post-sampling chain is a genuine data dependency, not a scheduling artifact. |
| `CUDA_DEVICE_MAX_CONNECTIONS=8` | Flat. Overlap is bounded by graph structure, not queue count. |
| `--inference-moe-token-dispatcher-type nccl` | **0.66×**. Pads to the worst-case per-rank count, roughly doubling comm volume. Correctness fallback only. |
| `--inference-grouped-gemm-backend torch` | **0.82×** for BF16. Use only when MXFP8 forces it. |

## References

- [references/decision-gates.md](references/decision-gates.md) — quantify a lever's ceiling before building it; per-launch fixed costs; three gates that killed multi-week efforts
- [references/vllm-differential.md](references/vllm-differential.md) — kernel-level comparison against a competitor trace; slower kernels vs more kernels
- [references/cuda-graphs.md](references/cuda-graphs.md) — scope, bucketing, graph-safety invariants, idle EP ranks, what a wide capture forecloses
- [references/moe-inference.md](references/moe-inference.md) — dispatchers, grouped GEMM, overlap, pad masking
- [references/mamba-and-triton.md](references/mamba-and-triton.md) — Triton rules, fused extraction, scratch sizing
- [references/host-path.md](references/host-path.md) — serialization, IPC payloads, DP routing
- [references/measuring.md](references/measuring.md) — profiler endpoints, NVTX, built-in counters, nsys flags that deadlock, union-busy idle accounting, A/B protocol, kernel-to-e2e conversion, ledger
- [references/commit-log.md](references/commit-log.md) — every commit mapped to its pattern, plus superseded APIs
- [references/updating-this-skill.md](references/updating-this-skill.md) — how to extend, correct, and prune this skill as you learn
- [assets/review-checklist.md](assets/review-checklist.md) — pre-PR checklist

## Out of scope

- Authoring perf or functional test recipes — use `add-inference-performance-test`
- Cluster and container bootstrap — use `cog-setup-and-help`
- Non-hot-path product features (reasoning parsers, chat-template retention)

## Keeping this skill current

Edit this skill at the end of a piece of work, once you have a number and a root
cause — never mid-experiment on a hypothesis. Triggers: an accepted optimization, a
**rejection** (the highest-value entry type), a negative decision gate, a tooling
failure that cost over an hour, a measurement that contradicts something here, a
flag or default that differs from what is documented, or an invariant whose
violation broke something.

Route by subject into the reference that owns it — only new invariants, flag
behavior, and routing lines belong in this file, because `SKILL.md` is the router
and it stops working once it needs its own router. An addition must be measured,
root-caused, scoped, and actionable. Delete what is false; never delete a measured
negative result.

Triggers, routing table, house style, deletion policy, size budgets, and the
revision log: [references/updating-this-skill.md](references/updating-this-skill.md).
