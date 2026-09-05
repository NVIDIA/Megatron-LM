# Measuring Inference Performance

Measure first, then change one thing, then measure again. Every commit in this
history started from a profile, and the bottleneck was frequently not where it
seemed. This document covers the observability that exists specifically so
inference changes can be attributed.

Source commits: `eadbaa61` (#5611, profiler endpoints), `edd45620` (#5609,
cache-hit reporting), `faced5128` (#3034, routing instrumentation), `53a2b19a`
(#2920, NVTX ranges).

## Bracketing an nsys capture with the profile endpoints

Profiling a server is awkward: you want the steady state, not startup, warmup, or
CUDA-graph capture. The profile endpoints solve this by relaying
`cudaProfilerStart` / `cudaProfilerStop` to every engine, so an outer nsys process
records only the window you ask for.

From
[profile.py](megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/profile.py):

```
POST /start_profile and /stop_profile relay a control signal through the
InferenceClient -> data-parallel coordinator -> every connected EP/DP engine,
which calls cudaProfilerStart()/cudaProfilerStop(). Pair with an outer
`nsys profile --capture-range=cudaProfilerApi` to bracket a capture window.
```

Workflow:

1. Launch the server under nsys with the capture range armed:

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o inference_profile \
    <your server launch command>
```

2. Send warmup load until the engine reaches steady state, including enough steps
   that all CUDA graphs are captured.
3. Start recording, drive the load you care about, stop recording:

```bash
curl -X POST http://localhost:5000/start_profile
# ... drive the workload ...
curl -X POST http://localhost:5000/stop_profile
```

Both `/start_profile` and `/v1/start_profile` are routed, likewise for stop. The
signal reaches **every rank**, so a multi-GPU trace covers all of them.

Then analyze the trace with the `nsight-system-analysis` skill.

### Flags that deadlock finalization — read before adding host visibility

On a MoE decode workload under full-iteration CUDA graphs, nsys **hangs in
finalization** and `QdstrmImporter` then rejects the `.qdstrm` if any of the
following is enabled. This was bisected one flag at a time over several sessions
on Qwen3-30B / 4×GB200, and it is independent of `--cuda-graph-trace` level:

| Trigger | Symptom |
|---|---|
| `osrt` in `--trace` | Finalization deadlock; unusable qdstrm |
| `--sample=process-tree` (CPU / Python sampling) | Same |
| NVTX ranges active under graph capture | Same — capture inflates NVTX event volume enormously |

The only set that reliably finalizes on this workload:

```bash
nsys profile --trace=cuda,nvtx --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o inference_profile <launch command>
```

This is a real constraint on method, not a nuisance: **the documented way to
attribute host time is exactly the way that breaks.** Two consequences, both with
working substitutes below — attribute host-side idle from the CUDA-API rows in a
clean GPU-only trace (see *Idle is not one thing*), and name host phases with
`perf_counter` instead of NVTX (see *When NVTX is unavailable*).

### Validate that your instrumentation is not the measurement

`--cuda-graph-trace=node` resolves individual kernels inside a graph, which is
what makes per-kernel analysis possible, but it instruments every node — 1158 of
them on this model — so it is fair to suspect it inflates the step.

Check rather than assume. Capturing the same steady-state window both ways gave a
step period of **8882.5 µs (node) against 8945.1 µs (graph)** — 0.71% apart and in
the *wrong direction* for an instrumentation artifact — and host `cudaGraphLaunch`
of 190.0 against 184.7 µs. Node mode also reported 1169 kernels/step, matching the
independent count. Only after that control did a 900 µs/step graph-machinery
attribution become safe to act on.

Run this control once per workload before trusting node-mode wall-time
attributions. Use `--cuda-graph-trace=graph` when you only need step periods, since
it produces far fewer events.

## NVTX ranges label the host critical path

Device timelines show kernels; they do not show why the host fell behind. The
per-step host critical section is annotated, so the CPU side of the trace is
readable without guessing:

| Range | What it covers |
|---|---|
| `bookkeeping` | Per-step request state updates |
| `detokenization` | Only when no coordinator is present |
| `coordinator_communication` | msgpack serialize + ZMQ send |
| `serialize_tensor` | Individual tensor to list conversion |
| `drain_zmq_socket`, `add_request` | Inbound request handling |

If you add host work to the step loop, add an NVTX range around it. `nvtx_range_push`
/ `nvtx_range_pop` come from `megatron/core/utils.py`. The precedent is that even
`serialize_tensor` — a two-line function — is annotated, because unmeasured host
cost is invisible cost.

Two gotchas before you rely on these ranges. The engine's own helpers are **inert
unless NVTX profiling is enabled**, which only the training loop does by default —
a whole 187.9 µs/step host phase (`post_process_requests` via `async_bookkeep`) was
invisible in one host trace for exactly this reason, and had to be found by
elimination instead. And enabling NVTX under graph capture is one of the
finalization-deadlock triggers above, so on this workload the ranges you most want
are the ones you cannot record.

### When NVTX is unavailable: `perf_counter` phase timing

Since the range names are the useful part and nsys is the broken part, keep the
names and drop nsys. Monkeypatch the same push/pop symbols with
`time.perf_counter_ns` self-timers over a shared nesting stack — each label then
reports self-time excluding its children — and print a per-step breakdown every N
steps. No profiler, no CUDA-graph interaction, and it names precisely the
between-step Python phases a GPU-only trace can only localize.

Gate it behind an env var and make `install()` a no-op by default, the same way
every other diagnostic here ships.

## Built-in counters, no profiler needed

Two things are exposed through the API and are often enough to confirm or reject a
hypothesis in seconds.

### Prefix cache hit rate

`DynamicInferenceRequest.num_cached_tokens` accumulates
`num_matched_blocks * block_size_tokens` across prefill chunks, and surfaces in the
chat-completions response as `usage.prompt_tokens_details.cached_tokens`
(OpenAI-compatible).

Use it to check that a prefix-caching or routing change actually improved cache
hits, rather than inferring it from end-to-end latency. Particularly relevant when
tuning `prefix_caching_routing_alpha` — see
[host-path.md](host-path.md).

### MoE expert load imbalance

Set `moe_enable_routing_replay` (requires `num_moe_experts`). The context then
records, per request, which experts every token was routed to, in a static
CUDA-graph-safe buffer of shape
`[max_tokens, num_moe_layers, moe_router_topk]`, exposed as
`DynamicInferenceRequest.routing_indices` with shape
`[total_tokens, num_layers, topk]`.

Aggregate a histogram across many requests. A few hot experts with the rest idle
means wasted expert capacity and straggler experts gating every step — a
throughput problem no kernel tuning will fix.

Implementation: [routing_metadata.py](megatron/core/inference/contexts/routing_metadata.py),
[router_replay.py](megatron/core/transformer/moe/router_replay.py), collected in
`TextGenerationController._router_record_bookkeeping()`. Note that the recording
itself follows the rules in this skill: a preallocated static buffer, toggled by
`using_cuda_graph_this_step()`, sliced to `active_token_count` so padding is
excluded.

## Benchmark harnesses

| Tool | Use |
|---|---|
| [tools/run_inference_performance_test.py](tools/run_inference_performance_test.py) | Standalone driver: builds the model and engine, sweeps requests, times steps. Fastest local loop. |
| `tests/performance_tests/shell_test_utils/run_perf_test.sh` | The harness CI perf recipes invoke. |
| `tests/performance_tests/shell_test_utils/compare_to_baseline.py` | Compares a run against checked-in `baseline_values.json`. |
| `tests/test_utils/recipes/h100/*inference*.yaml` | Registered perf and functional recipes. |
| [examples/inference/](examples/inference/) | Server launch and offline inference examples. |

For cluster runs use the `run-inference-performance-tests` skill (throughput and
latency versus baseline) and `run-inference-functional-tests` (correctness,
including the CUDA-graph paths). Neither is duplicated here.

## Interpreting the profile

### Pick a clean window, and measure busy as a union

Two arithmetic mistakes will make every category number wrong.

**Do not take a fraction of the capture span.** A span fraction mixes prefill,
graph capture, and ramp into the "steady state." Instead search for the **densest
window** of the length you want — kernel starts per unit time — and verify the
kernel sequence repeats with a stable period inside it. A 1-2 s window is enough.

**Do not sum per-stream kernel time.** Kernels overlap across streams, so summing
double-counts and can exceed the wall clock. Compute the **union of kernel
intervals** across all streams on the device; that is busy time, and
`window − union` is real idle.

On the twelve-gate Qwen config a clean 2 s window gave union-busy **81.6%** and
idle **18.4%**, with device-time by category of expert GEMM 587 ms, NVLS comm
247 ms, dense GEMM 236 ms, attention 211 ms, and then routing 69, norm 67,
elementwise 66, `moe_sum` 65, rope/KV 15. Those small buckets being ~65-70 ms each
*and partly overlapped on other streams* is what established that further
small-kernel fusion was worth only ~0.5-1% apiece — a conclusion that a
per-stream sum would have hidden.

### Idle is not one thing

Total idle is not an actionable number, because a large fraction of it is not
attackable. Split it by gap size first:

| Gap size | Share (Qwen, 178 ms idle in 1 s) | What it is |
|---|---:|---|
| < 10 µs | 59.6 ms (33%) | Intra-graph kernel scheduling — unavoidable |
| ≥ 10 µs | 118.8 ms (67%) | Host chain between steps — attackable, median 37 µs |

Then attribute the large gaps **without a host trace**, using the CUDA-API
(`CUPTI_ACTIVITY_KIND_RUNTIME`) rows that a clean `cuda,nvtx` capture already
contains. For each gap, ask which host API calls fall inside it:

| Inside the gap | Share of large-gap idle |
|---|---:|
| Nothing — no CUDA call at all (Python/CPU compute) | **73.9%** |
| Token D2H memcpy | 8.9% |
| Graph launch | 7.5% |
| Kernel launch | 6.9% |
| Sync wait | 1.6% |

So ~74% of attackable idle — about 8-9% of wall time — was **Python on the
critical path**, and only 1.6% was waiting on the GPU. Localize it further by
bracketing each uncovered region with the CUDA APIs immediately before and after
it: 47.6% sat between `cudaMemcpyAsync` and `cuKernelGetName`, another 15.7%
between `cudaLaunchKernel` and `cudaMemcpyAsync`. That pins the cost to
**between-step host orchestration after the sampled-token D2H copy** — post-sampling
bookkeeping, scheduling, attention-state prep — which is where the host-path wins
in [host-path.md](host-path.md) came from.

This RUNTIME-row method is the recommended one, not a fallback. It works on the
only capture configuration that finalizes.

### Map the signal to a section

Map the dominant signal to the section of this skill that addresses it:

| Signal | Reading | Go to |
|---|---|---|
| GPU idle gaps, host running ahead | Launch-bound: too many launches, or no matching graph bucket | [cuda-graphs.md](cuda-graphs.md) |
| Long host span after the last kernel | Post-processing on the critical path | [host-path.md](host-path.md) |
| NCCL kernels not overlapped | Exposed collective, or EP ranks out of step | [moe-inference.md](moe-inference.md) |
| Multi-ms stalls, or slow early steps | Triton recompilation or autotune | [mamba-and-triton.md](mamba-and-triton.md) |
| Grouped GEMM wide but shallow | Tiles tuned for the wrong batch size | [moe-inference.md](moe-inference.md) |
| Many sub-microsecond kernels in a row | Fusion candidate | [moe-inference.md](moe-inference.md) |

## Measurement discipline

- **A/B against a matched config.** Same batch size, sequence length, and
  parallelism as the vLLM or baseline number you are comparing to. A latency
  comparison across different `max_requests` values means nothing.
- **Exclude capture from the measurement.** Graph capture runs a full forward per
  bucket. Warm up until capture is done before you start recording.
- **Change one thing.** Most of these commits are single-mechanism precisely so the
  measurement attributes cleanly.
- **Keep the kill switch and measure both ways.** `inference_disable_triton_nvls_kernels`,
  `inference_moe_disable_fused_quant_kernels`, and the backend enums exist so a
  regression can be bisected by config rather than by revert.
- **Re-measure per model.** The shared-expert CTA cap was a win on one model and a
  loss on another (see [moe-inference.md](moe-inference.md)). Overlap and tile
  tuning do not transfer.

## The noise floor is bigger than your win

Once the easy multi-percent wins are gone, the remaining levers are worth well
under 1% each, and at that point the above discipline is necessary but **not
sufficient**. Across the Qwen campaign, throughput for an *identical* config
measured in different allocations drifted by **−1.61%, −0.57%, and +1.22%** on
three separate occasions — larger than most of the individual wins that followed
(+0.94%, +0.90%, +0.98%, ~+0.5%). Comparing against last session's number will
produce confident, wrong conclusions in both directions.

Four rules make a sub-1% claim falsifiable.

**1. Re-baseline in the same session.** Before each lever, run the current best
config in the *same allocation* you will test in, and use that as the reference.
Record it as its own ledger entry. This is the single highest-value habit here.

**2. Run the arms back to back.** Same allocation, same warm process where
possible, alternating OFF and ON. An OFF arm collected an hour earlier is a
different measurement.

**3. Judge by distribution separation, not mean delta.** With N timed iterations
per arm, the acceptance criterion is that the **arms do not overlap** — the
slowest ON iteration beats the fastest OFF iteration. Examples that met it:

```
QWEN-026:  min ON 27,102.9  >  max OFF 26,425.7        (fully separated)
QWEN-025:  13 of 15 ON iterations beat all 15 OFF      (near-separated)
QWEN-023:  slowest ON 25,992.1 > fastest OFF 25,893.9  (fully separated)
```

A +1% mean delta with overlapping arms is not a result. A +0.9% delta with
separated arms is, even though it is smaller than the session drift — because
drift is between sessions and separation is measured within one.

**4. Repeat the pair, and report pairwise deltas.** Two or three OFF/ON pairs, each
delta stated separately (`+0.99% / +1.26% / +0.68%`), tells the reader the spread.
A single averaged number hides it.

Also discard the first timed iteration or check it explicitly — cold-start outliers
of ~1.3% appeared repeatedly while iterations 2-5 spanned 0.23%.

## Predict how much of a kernel win will convert

The most common surprise in this campaign was the ratio between a kernel-level
speedup and the end-to-end result. It went both ways, by large factors, and it is
predictable enough to estimate before you build.

| Where the work sits | Conversion | Evidence |
|---|---|---|
| Serial per-layer dependency chain | **more than 1:1** | Fused QK-norm: ~1% microbench ceiling → **+2.9% e2e**, ~3×. Removing a launch also removes a graph node and the host dispatch gap behind it, ×48 layers. |
| Overlapped with other work | **well under 1:1** | A 183.9 µs/step host saving converted at only **1/3 to 1/2**, because that phase was already partly overlapped with GPU work. |
| Off the critical path | **~0** | A 1.25× speedup of the MoE activation path delivered **+0.55%, then +0.13%** — a wash. The op was never the wall. |

The rule: **estimate against the critical path, never against device-time share.**
Before building, establish whether the target is (a) on the serial chain, (b)
already overlapped, or (c) off it — the same determination the
[decision-gates.md](decision-gates.md) ceiling calculation needs.

The corollary is that a small kernel on the serial chain can be worth more than a
large kernel off it, which inverts the usual ranking. Two ~1.5-4 µs norm fusions
delivered +2.9% and +1.37%, while the largest device-time category in the profile
had 2.3% of headroom left in total.

## Keep an append-only ledger

For anything longer than a single session, the ledger *is* the method. Sessions get
preempted, context is lost, and without a durable record the same dead end gets
re-explored — a routing-kernel rewrite that measured a wash was nearly repeated for
exactly this reason.

Requirements that earned their place:

- **A fixed protocol table at the top** — cluster, hardware, model, batch, output
  length, parallelism, warmup/timed counts, correctness gate, primary metric — plus
  the explicit rule that results are not comparable when any of them differ.
- **One row per experiment, including every rejection**, with its root cause. The
  negative results are the higher-value half: they are what stops the next person,
  and three of them killed multi-week efforts here.
- **Append-only. Never edit a recorded result.** Supersede it with a new entry.
- **A running distance-to-target number** so priorities stay honest.
- **A ranked next-levers list**, re-derived after each profile rather than carried
  forward.

Record the baselines in a fixed order — competitor first, then yours, then the
gap — and only then change code.
