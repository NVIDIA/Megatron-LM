# Decision Gates: Prove the Ceiling Before You Build

Profiling tells you where time is spent. It does not tell you whether that time is
**recoverable**. Those are different questions, and the gap between them is where
whole sessions get lost — writing a kernel whose best possible outcome was never
worth the week.

A decision gate is a short, measurement-backed answer to one question: *if this
optimization worked perfectly, how much would the step actually get faster?* If
the answer is small, or if the mechanism you plan to attack is not the mechanism
that is slow, you stop before writing production code.

Source: the Qwen3-30B-A3B EP4 campaign ledger
(`skills/run-qwen-model/EXPERIMENTS.md`), where three gates each killed a
multi-session effort, and a fourth picked the right candidate out of six.

## Why share is not headroom

The instinct is to rank categories by their share of device time and attack the
top one. That instinct is wrong twice over on this stack.

**Some categories are already at a hardware floor.** A category can be 33% of
device time and have 2% of headroom, because it is moving the bytes it has to
move. Ranking by share puts it first; ranking by headroom puts it last.

**Device time is not wall time.** Kernels overlap across streams, so the step's
wall clock is the *critical path* through the per-layer dependency chain, not the
sum of kernel durations. A category with a large device-time share that sits on a
side stream, already overlapped with compute, contributes far less to wall time
than its share suggests. Conversely, a tiny kernel on the serial chain costs its
duration *plus* the dispatch gap behind it, every layer.

So the gate has to produce a **wall-time ceiling**, not a device-time share.

## The gate, in four steps

### 1. Establish the floor

Compute what the operation could not go faster than, from first principles, using
*measured* machine constants rather than datasheet peaks.

For a memory-bound op, the floor is bytes / achievable bandwidth. Measure the
achievable bandwidth yourself with a streaming-read microbenchmark on the same
device — datasheet HBM numbers overstate it, and being wrong here invalidates the
whole gate.

For a latency-bound op, decompose the measured time into launch, synchronization,
and transfer, and identify which term dominates. An empty-kernel launch and a
barrier-only kernel are the two calibration points you need.

### 2. Measure the current cost the same way

Under **CUDA-graph replay**, not eager, and at the token count decode actually
runs at. Eager per-kernel timings and replay timings differ by more than the
effect sizes you are chasing.

### 3. Ceiling = current / floor, then subtract what the fix costs

The gross ceiling is the ratio. The net ceiling subtracts the machinery the fix
introduces: an added zeroing launch, a grid sync, extra atomics, a second
streaming pass. Fusions frequently give back most of the gross win this way, and
occasionally more than all of it.

Then convert to a fraction of the step, because that is the number that decides
whether to proceed.

### 4. Write down a verdict with a threshold

State the gross ceiling, the net ceiling, the fraction of the step, and either
**proceed** or **gated out**, with the reason. Record it in the ledger even when
the answer is "don't build this" — especially then. A recorded negative gate is
what stops the next person re-deriving it.

## Calibrate the fixed cost of a launch

Several gates reduce to "is removing this launch worth it," so measure the two
constants once per platform and reuse them.

On GB200 under full-iteration graph capture, the campaign measured:

| Constant | Value | How |
|---|---|---|
| Empty-kernel launch | 0.72 µs | Time a no-op kernel under replay |
| Inter-kernel dispatch gap | 0.55 µs | Median gap between consecutive kernels in the trace |
| Floor per launch | 1.27 µs | Sum of the two |
| Graph node cost | ~0.17 µs | `cudaGraphLaunch` 199.1 µs / 1158 nodes |

The consequence: **a removed launch is worth its own duration plus the dispatch
gap behind it**, so eliminating a 1.5 µs kernel buys ~2.05 µs of wall time, not
1.5. And no kernel can be optimized below 1.27 µs — a 1.33 µs kernel has no
headroom left, however inefficient its body looks.

## Diagnose the mechanism, not just the magnitude

A gate that identifies the wrong *cause* is as expensive as no gate. The routing
token-count kernel measured 7.52 µs per layer — clearly anomalous for its byte
count — and the first attempt rewrote its reduction, replacing per-pair
`atomic_add` with a `tl.histogram` variant. It measured **0.96×, a wash**, and
the reason was that atomic contention was never the problem: at `BLOCK_SIZE=1024`
only **2 of the kernel's 152 CTAs received any work**. The cost was launch
overhead and CTA underutilization, which an in-kernel reduction rewrite cannot
touch.

Before attacking a slow kernel, establish *why* it is slow: bytes, FLOPs,
occupancy, CTA utilization, a dependent load serializing a branch, or pure launch
overhead. The fix follows from the mechanism, and the wrong mechanism produces a
technically correct kernel that changes nothing.

## Case study 1: the grouped GEMM that could not win

**The proposal.** MoE expert GEMM was the largest single category at ~33-40% of
decode device time. The obvious plan was a hand-written CUTLASS/CuTe grouped GEMM
to replace the Triton path — a multi-week effort.

**The gate.** Weight traffic is 302 MB per layer per rank. A measured streaming-read
ceiling of 6.081 TB/s gives a **49.66 µs floor**. Production FC1+FC2 measured
**72.13 µs**. So the entire ceiling for *any* implementation is **1.45×** — and
achieved throughput on valid FLOPs was 63-83 TFLOP/s, about 3% of the device's
BF16 peak, confirming the op is nowhere near compute-bound.

Padding waste was real but nearly free: 74.3% dead rows at `BLOCK_M=64`, yet
cutting the padding ratio from 3.89× to 1.49× buys only ~1.2×, because the dead
rows re-read weights that are already resident.

**The verdict.** A hand-written GEMM's whole ceiling was 1.45×, and **1.26× of it
was reachable by retuning Triton tiles alone**. The kernel effort was gated out;
the tile retune shipped instead and delivered **+4.32% end-to-end** for a
day's work. A later profile confirmed the gate independently: expert GEMM ended up
only 229 µs/step — 2.3% of the step — above the weight-bandwidth floor.

**Generalizable:** roofline the category *before* proposing an implementation
swap. A memory-bound op with a 1.45× ceiling does not deserve a new kernel; it
deserves better tiles.

## Case study 2: the barrier that was a hardware floor

**The proposal.** Exposed NVLS EP communication was ~11.7% of device time and
100% exposed (the interval union equalled the sum). Plans on the table: pipeline
the collective in chunks, or fuse ReduceScatter into the FC2 epilogue.

**The gate.** Decomposing the collectives showed they are **latency-bound, not
bandwidth-bound**:

```
AllGather-V   6.57 µs = 0.72 launch + 5.08 barrier + 0.77 transfer
ReduceScatter 7.88 µs = 0.72 launch + 5.03 barrier + 2.13 transfer
```

Bytes accounted for only 127 µs of a ~693 µs/step total, and the
ReduceScatter transfer was already at 82% of the NVLink floor — so batching bytes
cannot help. A follow-up microbenchmark isolated the barrier further: a
barrier-only kernel costs 5.75-5.83 µs against a 0.72 µs empty kernel, so the
5.05 µs **is the four-way system-scope flag round trip itself**, not polling
granularity. Rewriting the spin (`atom.cas` → `ld.acquire.sys` poll) measured a
**2.3% regression**, exactly as the gate predicted.

Of 1060 µs/step, 632 µs was intrinsic and 428 µs was **inter-rank skew** — ranks
waiting on the slowest rank's expert GEMM, which is a routing-balance problem, not
a communication problem.

**The verdict.** Recoverable critical path: **6.4-7.0% of the step**, and each
candidate gave most of it back. Epilogue fusion keeps the barrier (≈1.4% left);
2-chunk pipelining *adds* a barrier per chunk and needs concurrent streams under
graph capture, which full-iteration capture forbids. CTA count was already at its
optimum. Lever abandoned.

**Generalizable:** decompose a collective into launch + barrier + transfer before
optimizing it. If the barrier dominates, you are looking at a fabric round trip
and no kernel rewrite will move it. Check whether the residual is really *skew*,
in which case the fix is load balance somewhere else entirely.

## Case study 3: pricing six fusions before writing one

**The proposal.** The routing/permute category was 1251 µs/step across 242
kernels — visibly a launch storm, with several plausible fusions.

**The gate.** Break the category down per kernel name — launches, device µs,
dispatch gap — then price every candidate against the 1.27 µs per-launch floor:

| Kernel | µs | × per step |
|---|---:|---:|
| `_moe_sum` | 7.79 | 48 |
| `_count_local_tokens` | 7.52 | 48 |
| `gatherTopK` | 6.05 | 48 |
| `_scatter_token_indices` | 2.66 | 48 |
| router softmax | 1.94 | 48 |
| `_prefix_fill_init` | 1.33 | 48 |

Candidate ceilings, gross and net:

| Candidate | Gross | Net after costs | Verdict |
|---|---:|---|---|
| `_moe_sum` → FC2 epilogue | 400 µs (4.03%) | ~3.3%, possibly negative — needs cross-CTA fp32 atomics **plus** a zeroing launch | gated out |
| Cooperative-grid merge of count→fill→scatter | 122 µs (1.23%) | less, after two grid syncs | gated out |
| Fold count + its `zeros` fill into `_prefix_fill_init` | 449 µs (4.52%) | no grid sync, integer-exact | **chosen** |

**The verdict.** The chosen fusion shipped at **+3.01% end-to-end**. The
`_moe_sum` epilogue candidate — superficially the most attractive, since
`_moe_sum` was the single most expensive routing kernel — had the exact failure
shape of an earlier rejected mega-fusion: cross-CTA atomics plus a zeroing pass
eating the gain. It was never built.

Notably, `_moe_sum` was later improved anyway, but by a different mechanism the
gate exposed: its per-topk-slot locality test was a uniform scalar branch on a
*dependent* load, serializing the topk walk. Predicating it into the load mask
gave 1.36× and **+1.83%** end-to-end, bit-exact, with no fusion at all.

**Generalizable:** when a category has several fusion candidates, price all of
them before building any. Rank by *net* ceiling, and prefer candidates that need
no grid sync and no atomics — those two costs are what turn a 4% gross win into a
0% net win.

## Verdict template

Record this in the ledger for every gate, including the negative ones:

```
GATE <id> — <lever name>
Question:      <what would be built, and what it would cost to build>
Floor:         <bytes / bandwidth, or launch+barrier+transfer decomposition>
Current:       <measured under graph replay, at decode token count>
Gross ceiling: <ratio, and µs/step, and % of step>
Fix costs:     <added launches, grid syncs, atomics, extra passes>
Net ceiling:   <% of step after costs>
Mechanism:     <why it is slow: bytes / occupancy / CTA util / dependent load / launch>
Verdict:       PROCEED | GATED OUT — <reason>
```

## When to skip the gate

Gates cost time too. Skip straight to measurement when the change is cheap and
reversible: flipping an existing flag, retuning tile sizes, swapping a backend
enum. Those are their own experiments — a gate on a config flip costs more than
the flip.

Gate anything that would take more than about a day to build, anything requiring
a new kernel, and anything whose category you have not yet rooflined.
