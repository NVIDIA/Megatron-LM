# VPP + M-FSDP v2 Compute and Communication Scheduling Scenarios

**Status:** Scenario inventory; no scheduling policy selected

This document consolidates parameter prefetch ordering, communication
contention, and parameter residency into one problem:

> Given the next PP/VPP operation, schedule compute, EP communication, PP
> communication, M-FSDP parameter gathering/release, and gradient reduction so
> that dependencies are satisfied, useful work overlaps, and live memory stays
> within budget.

The scenarios come from the PP=3, VPP=2, six-microbatch baseline. They define
the cases a scheduling design must handle before we choose stream placement,
prefetch distance, or buffer policy.

The corresponding production-shaped DSv4 Pro sequences are in
[DSv4 Pro PP3/VPP2 scheduling scenarios](dsv4_pro_pp3_vpp2_scheduling_scenarios.md).
That document expands the four-layer examples into 10–11-unit chunk plans,
including MTP, unequal forward/backward tails, and final gradient reduction.

## 1. Operation identity

Every scheduled operation needs this identity:

```text
(physical PP rank, VP chunk, microbatch, layer/FSDP unit, direction, phase)
```

For example:

```text
(P0, C1, m3, L12, forward, steady)
(P0, C1, m1, L15, backward, steady)
```

The microbatch or layer number alone is insufficient because forward and
backward can operate on different microbatches, chunks, and layers at the same
local scheduling step.

## 2. Key operations

The notation below names logical operations. An implementation may split one
operation into several kernels, copies, events, or collectives.

### 2.1 Pipeline-parallel operations

| Operation | Meaning |
|---|---|
| `PP.RECV_F` | Receive the activation required by a forward virtual stage |
| `PP.SEND_F` | Send the produced activation to the next virtual stage |
| `PP.RECV_B` | Receive the output gradient required by a backward virtual stage |
| `PP.SEND_B` | Send the input gradient to the previous virtual stage |
| `PP.WAIT_*` | Wait until an asynchronous receive or send is safe to consume or release |

The logical PP neighbors are virtual stages, not always adjacent physical
ranks. In this example, forward crosses `P2/C0 -> P0/C1`, and backward crosses
`P0/C1 -> P2/C0`.

### 2.2 Expert-parallel and compute operations

| Operation | Domain | Meaning |
|---|---|---|
| `F.PRE` | Compute | Forward attention, normalization, router, and dispatch preparation |
| `EP.DISPATCH_F` | EP | Forward token dispatch all-to-all |
| `F.EXPERT` | Compute | Forward expert MLP computation |
| `EP.COMBINE_F` | EP | Forward token combine all-to-all |
| `EP.COMBINE_B` | EP | Backward through token combine |
| `B.EXPERT_DGRAD` | Compute | Expert backward activation-gradient computation |
| `B.EXPERT_WGRAD` | Compute | Expert weight-gradient computation, possibly delayed |
| `EP.DISPATCH_B` | EP | Backward through token dispatch |
| `B.PRE_DGRAD` | Compute | Backward activation-gradient work before dispatch |
| `B.PRE_WGRAD` | Compute | Weight-gradient work before dispatch, possibly delayed |

`EP.DISPATCH_*` and `EP.COMBINE_*` use the EP communication domain. Their
duration depends on the routed token distribution, not only tensor shape.

### 2.3 M-FSDP operations

| Operation | Meaning |
|---|---|
| `FSDP.AG_F(U)` | All-gather/unshard FSDP unit `U` for forward use |
| `FSDP.READY_F(U)` | Make forward compute wait for the gathered parameter if necessary |
| `FSDP.RELEASE_F(U)` | Reshard and release forward's full-parameter storage after its last consumer |
| `FSDP.AG_B(U)` | All-gather/unshard `U` for backward use |
| `FSDP.READY_B(U)` | Make backward compute wait for the gathered parameter if necessary |
| `FSDP.RELEASE_B(U)` | Reshard and release backward's full-parameter storage |
| `FSDP.REDUCE_GRAD(U)` | Reduce-scatter a completed sharded gradient, or all-reduce in a non-sharded domain |
| `FSDP.FINALIZE` | Wait for required reductions and install/finalize gradients for optimizer use |

Dense units communicate over dense-DP4 in the example. Expert units use their
expert-DP2 domain. `FSDP.REDUCE_GRAD` must wait for every required dgrad/wgrad
producer but should not wait for unrelated work.

## 3. One combined layer pair

At model-chunk scope, a combined operation has this envelope:

```text
PP.WAIT_RECV_F -> forward preprocess
PP.WAIT_RECV_B -> backward postprocess or local loss gradient
paired forward/backward layers
PP.SEND_F + PP.RECV_F(next)
PP.SEND_B + PP.RECV_B(next)
delayed B.PRE_WGRAD for the lowest layer in the chunk
```

The forward PP transfer is placed on the communication stream so it can
overlap late backward compute. The backward PP transfer is intended to overlap
the delayed wgrad. First and last virtual stages omit PP operations that have
no logical neighbor.

For `F Li + B Lj`, the combined executor uses this conceptual enqueue order:

```text
backward: EP.COMBINE_B(Lj)
forward:  F.PRE(Li)
backward: B.EXPERT_DGRAD(Lj)
forward:  EP.DISPATCH_F(Li)
backward: B.EXPERT_WGRAD(Lj), EP.DISPATCH_B(Lj)
backward: B.PRE_DGRAD(Lj)                    # may move earlier by policy
forward:  F.EXPERT(Li)
forward:  EP.COMBINE_F(Li)
backward: B.PRE_DGRAD(Lj)                    # if not issued earlier
backward: B.PRE_WGRAD(Lj)                    # lowest model layer may be delayed past PP
```

EP operations use a communication stream and compute operations use a compute
stream, with events establishing dependencies. This is enqueue/dependency
order, not a claim that all operations execute serially.

M-FSDP surrounds the relevant unit's first and last consumers:

```text
FSDP.AG_F(Li) -> FSDP.READY_F(Li) -> forward consumers
                                      -> FSDP.RELEASE_F(Li)

FSDP.AG_B(Lj) -> FSDP.READY_B(Lj) -> backward dgrad/wgrad consumers
                                      -> FSDP.RELEASE_B(Lj)
                                      -> FSDP.REDUCE_GRAD(Lj)
```

A scheduling solution must place gathers early enough to hide them, releases
late enough to preserve correctness, and reductions early enough to overlap
without blocking a gather needed on the critical path.

## 4. Scenario inventory

| ID | Example | What makes it distinct |
|---|---|---|
| S1 | P0 warmup `F m4/C0` | Forward only; activation and parameter residency grow |
| S2 | P0 `F m5/C0 + B m0/C1` | Different chunks and two FSDP roots |
| S3 | P0 `F m3/C1 + B m1/C1` | Same chunk, opposite layer traversals, different microbatches |
| S4 | `P2/C0 <-> P0/C1` | PP crosses the virtual-stage wraparound |
| S5 | P2 `F m1/C1 + B m0/C1` | Last virtual stage; loss begins backward locally |
| S6 | P0 cooldown ending at `B m5/C0` | Backward-only drain, gradient reduction, and finalization |

## 5. S1 — forward-only warmup

Representative operation:

```text
P0: F m4/C0
```

Per layer/FSDP unit:

```text
[PP.RECV_F if required]
FSDP.AG_F -> FSDP.READY_F
F.PRE -> EP.DISPATCH_F -> F.EXPERT -> EP.COMBINE_F
FSDP.RELEASE_F
[PP.SEND_F]
```

There is no backward computation or gradient reduction available to pair with
this forward. The scheduler must decide how far to prefetch later forward
units and whether retaining any full parameters until backward is cheaper than
gathering them again. Meanwhile, saved activations accumulate for eight P0
warmup forwards.

## 6. S2 — different-chunk combined operation

Representative operation:

```text
P0: F m5/C0 + B m0/C1
```

The forward uses the C0 FSDP root while backward uses the C1 root:

```text
PP inputs:       forward input/data + PP.RECV_B(C1,m0)
M-FSDP inputs:   AG_F(C0,next forward unit) + AG_B(C1,next backward unit)
layer work:      F(C0,L0->L3) paired with B(C1,L15->L12)
PP outputs:      PP.SEND_F(P0/C0 -> P1/C0)
                 PP.SEND_B(P0/C1 -> P2/C0)
```

This case permits independent chunk state but can keep parameter buffers from
two roots live concurrently. It also mixes ordinary forward PP traffic with a
backward transfer across the VPP chunk boundary.

The scheduling questions are which root may prefetch first, whether their
gathers share a stream or communicator, and how many forward/backward units
may remain resident simultaneously.

## 7. S3 — same-chunk combined operation

Representative operation:

```text
P0: F m3/C1 + B m1/C1

F L12 + B L15
F L13 + B L14
F L14 + B L13
F L15 + B L12
```

Both directions share the C1 root but belong to different autograd graphs and
move through its layers in opposite directions. At a layer-pair boundary the
desired working set may include:

```text
current forward unit
current backward unit
prefetched next-forward unit
prefetched next-backward unit
completed gradients awaiting reduction
```

The next-forward and next-backward gathers cannot be derived from one linear
module traversal. Release must also be tied to each direction's last consumer,
not merely to the end of the combined call.

This is the primary scenario for designing the unified compute/communication
schedule.

## 8. S4 — virtual-stage PP wraparound

The boundary between C0 and C1 crosses physical pipeline ranks:

```text
forward activation:  P2/C0 -> P0/C1
backward gradient:    P0/C1 -> P2/C0
```

P0/C1 cannot start forward compute until the activation produced by P2/C0 is
ready. P2/C0 cannot start the corresponding backward until the gradient from
P0/C1 arrives. These transfers may overlap with EP and M-FSDP communication on
both ranks.

The scheduler must treat the wraparound as an ordinary logical PP dependency
while accounting for the fact that its physical direction differs from the
usual `P0 -> P1 -> P2` forward flow.

## 9. S5 — last virtual stage and loss boundary

Representative operation:

```text
P2: F m1/C1 + B m0/C1
```

`P2/C1` is the final virtual stage. It receives a forward activation but does
not send one onward. Backward for the older microbatch begins from the local
loss rather than `PP.RECV_B`:

```text
PP.RECV_F -> forward/loss compute
local loss gradient -> backward compute
PP.SEND_B(P2/C1 -> P1/C1)
```

This rank has fewer PP dependencies and eight combined pairs, versus four on
P0. A scheduling policy must handle missing sends/receives without assuming
that the same overlap opportunity exists on every PP rank.

## 10. S6 — backward-only cooldown and final drain

Representative tail:

```text
P0: ... -> B m3/C0 -> B m4/C0 -> B m5/C0 -> FSDP.FINALIZE
```

Per layer/FSDP unit:

```text
[PP.RECV_B]
FSDP.AG_B -> FSDP.READY_B
EP.COMBINE_B -> B.EXPERT_DGRAD/WGRAD -> EP.DISPATCH_B
-> B.PRE_DGRAD/WGRAD
FSDP.RELEASE_B -> FSDP.REDUCE_GRAD
[PP.SEND_B]
```

There is no new forward work to hide backward gathers or reductions. The last
microbatch also determines when accumulated gradients become reducible and
when `FSDP.FINALIZE` may complete. A policy that performs well in S2/S3 can
still leave a large exposed tail here.

## 11. Questions every scheduling solution must answer

For each scenario, a proposed schedule must state:

1. Which forward and backward FSDP units are current and next?
2. When is each parameter gather enqueued, and what work should hide it?
3. Which event makes compute wait for parameter readiness?
4. What is the last consumer that permits forward or backward release?
5. When is each gradient complete, including delayed wgrad?
6. When and on which communication domain is its reduction issued?
7. Which EP, PP, gather, and reduction operations may execute concurrently?
8. What ordering is required when operations share a communicator or stream?
9. What is the maximum number and size of simultaneously resident units?
10. What changes at a first/last virtual stage or a phase boundary?

The schedule should be expressible as a dependency graph first. Stream and
communicator assignment should follow from that graph rather than define
correctness implicitly.

## 12. Starting point

Begin with S3 and construct a dependency graph for one layer pair,
`F L12 + B L15`. Then extend the graph to the transition
`F L13 + B L14`, where both forward and backward prefetch decisions become
visible. After that, test the same abstraction against S2's two roots and S6's
backward-only drain.

After validating the notation on this small case, use the DSv4 Pro R3 center
transition, `F L55 + B L57 -> F L56 + B L56 -> F L57 + B L55`, as the primary
design case. It includes simultaneous forward/backward use of the same FSDP
unit on different microbatches.
