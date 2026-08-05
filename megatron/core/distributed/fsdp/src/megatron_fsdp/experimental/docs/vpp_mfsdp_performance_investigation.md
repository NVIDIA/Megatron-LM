# VPP + M-FSDP v2 + Combined 1F1B Performance Investigation

**Status:** Candidate issues identified; investigation not yet started

This document tracks potential performance problems exposed by the PP=3,
VPP=2 example. They are hypotheses until measurements confirm them. The goal
is to investigate one issue at a time without mixing performance work with the
correctness and interface design.

The central question is:

> Can we schedule compute, EP, PP, parameter gathering/release, and gradient
> reduction so communication is hidden without exceeding the memory budget or
> creating contention on the critical path?

The operation vocabulary and concrete cases are defined in
[Compute and communication scheduling scenarios](vpp_mfsdp_compute_communication_scenarios.md).
Longer production-shaped cases are defined in
[DSv4 Pro PP3/VPP2 scheduling scenarios](dsv4_pro_pp3_vpp2_scheduling_scenarios.md).

## 1. Facts from the baseline example

The example has six microbatches and twelve virtual operations per rank. With
the extra warmup forward required for EP overlap, its phase counts are:

| Rank | Warmup forwards | Combined F+B pairs | Cooldown backwards |
|---|---:|---:|---:|
| P0 | 8 | 4 | 8 |
| P1 | 6 | 6 | 6 |
| P2 | 4 | 8 | 4 |

The highlighted same-chunk operation on P0 is:

```text
forward m3/C1:   L12 -> L13 -> L14 -> L15
backward m1/C1:  L15 -> L14 -> L13 -> L12

combined order:
  F L12 + B L15
  F L13 + B L14
  F L14 + B L13
  F L15 + B L12
```

Each layer pair may involve parameter all-gather, forward or backward compute,
EP dispatch/combine, parameter release, delayed wgrad, and gradient reduction.
Pipeline P2P is active around the combined operation as well.

## 2. Candidate issue register

| Priority | Candidate issue | Primary symptom |
|---:|---|---|
| P1 | Unified compute and communication scheduling | Exposed communication, contention, or excessive parameter residency |
| P2 | Too little combined steady state | Warmup/cooldown dominate iteration time |
| P3 | Exposed warmup/cooldown communication | Large startup or drain tail, especially on P0 |
| P4 | Fine-grained launch overhead | Many small collectives, events, waits, and allocations |
| P5 | MoE and stage imbalance | Pipeline backpressure and rank-to-rank idle gaps |

P1 consolidates the former prefetch-order, cross-domain contention, and
parameter-residency issues. These three concerns are coupled decisions in one
schedule rather than independent optimizations. Priority represents
investigation order, not proven impact.

## 3. P1 — unified compute and communication scheduling

The combined step traverses layers forward and backward at the same time. It
must coordinate three inseparable choices:

1. **Readiness:** gather the next forward and backward parameters before their
   consumers would block.
2. **Concurrency:** overlap EP, PP, gather, reduction, and compute only when
   doing so shortens the critical path rather than creating resource
   contention or communicator serialization.
3. **Residency:** release parameters after their true last consumer while
   keeping the current and prefetched working set within the memory budget.

The schedule cannot be derived from one linear module traversal. For
`F L12 + B L15`, the next pair is `F L13 + B L14`: forward advances while
backward retreats. Same-chunk pairs share a root; different-chunk pairs can
hold state from two roots concurrently.

**Question:** Can one dependency schedule keep required parameters ready,
reduce completed gradients, and make useful EP/PP progress without exposing
communication or exceeding the permitted live working set?

**Evidence needed:** A per-rank timeline containing operation identity,
gather/reduction enqueue and completion, EP and PP activity, compute intervals,
waits, parameter residency, stream/communicator identity, and peak memory.

The first production-shaped analysis target is DSv4 Pro scenario R3, centered
on `F L56 + B L56` for two different microbatches. Follow it with R1's
different-chunk pair, R2's unequal tail, and R4's final drain.

## 4. P2 — too little combined steady state

Combined forward/backward execution exists only in steady state. In the
example, P0 has four combined pairs but eight warmup forwards and eight
cooldown backwards. P2 has twice as many combined pairs as P0.

This does not prove that P0 is slower, because operation durations can differ.
It does show that the available overlap window is asymmetric and that a small
microbatch count may limit the total benefit.

**Question:** How does the fraction of iteration time spent in useful combined
steady state change with microbatch count and VP microbatch group size?

**Evidence needed:** Warmup, steady, and cooldown duration per PP rank, plus
iteration time for a controlled sweep of microbatch count.

## 5. P3 — exposed warmup and cooldown communication

Warmup has no paired backward work; cooldown has no new paired forward work.
P0 has the longest warmup and cooldown in the example. Backward gathers,
delayed wgrad, gradient reductions, or final synchronization may form an
especially visible drain tail.

**Question:** Which communication and waits remain exposed before the first
combined pair and after the last one?

**Evidence needed:** A phase-colored timeline and the time from the last
combined pair through completion of root gradient finalization.

## 6. P4 — fine-grained launch overhead

Fine-grained parameter gathering creates more overlap opportunities and can
lower transient memory, but it also increases the number of collectives,
events, stream waits, callbacks, and buffer-management operations. With small
microbatches, layer compute may be too short to amortize these costs.

**Question:** Is the communication unit large enough to be bandwidth-efficient
and the compute interval long enough to hide it?

**Evidence needed:** Collective size distribution, launch count, host gaps,
CUDA event/wait count, and a controlled communication-unit-size sweep.

## 7. P5 — MoE and stage imbalance

Expert token counts can vary between microbatches and ranks. A slow EP
dispatch, expert computation, or combine on either side of a layer pair can
delay the entire combined operation. P2P dependencies then propagate that
delay to other pipeline stages.

The equal four-layer chunks in the example describe static balance only; they
do not guarantee equal runtime.

**Question:** Are rank idle gaps correlated with expert token imbalance or
with a consistently slower model chunk?

**Evidence needed:** Tokens per expert, layer-pair duration, P2P wait time, and
per-rank idle time for the same iteration.

## 8. Common measurement baseline

Use the same model, batch, topology, precision, recompute policy, and number of
measured optimizer steps for these four configurations:

1. VPP 1F1B without combined EP overlap or M-FSDP overlap.
2. Combined 1F1B with EP overlap and conservative M-FSDP synchronization.
3. VPP with M-FSDP overlap but without combined 1F1B.
4. VPP with combined 1F1B, EP overlap, and M-FSDP overlap.

For every configuration, collect:

- end-to-end iteration time and throughput;
- warmup, steady, and cooldown duration by PP rank;
- exposed all-gather, gradient-reduction, EP, and P2P wait time;
- collective count, size, duration, group, and stream;
- current and prefetched parameter residency;
- peak allocated and reserved GPU memory; and
- the final synchronization tail before optimizer step.

Report both the median and a tail percentile after warmup iterations. A faster
isolated kernel or greater overlap percentage is not a success unless
end-to-end iteration time improves within the memory limit.

## 9. Investigation order

Start with **P1: unified compute and communication scheduling** and analyze the
scenario inventory in this order:

1. DSv4 R3: same-unit center crossing at the final virtual stage.
2. DSv4 R1: same-root and two-root 10-layer traversals.
3. DSv4 R2: unequal forward/backward tail.
4. DSv4 R4: backward-only drain and finalization.
5. Small S1/S4/S5: phase, wraparound, and endpoint variants.

For each issue, follow the same sequence:

1. Write the dependency graph.
2. Capture the relevant timeline and counters.
3. Confirm or reject the performance hypothesis.
4. Quantify its contribution to iteration time and memory.
5. Only then evaluate scheduling changes and record the trade-off.
