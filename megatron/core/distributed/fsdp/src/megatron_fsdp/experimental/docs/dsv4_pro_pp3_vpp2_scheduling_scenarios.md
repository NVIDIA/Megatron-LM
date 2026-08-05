# DSv4 Pro PP3/VPP2 Compute and Communication Scenarios

**Status:** Production-shaped working case; VPP layout and batch assumptions
must be confirmed

This document applies the VPP + M-FSDP v2 + combined 1F1B scheduling problem
to a DSv4 Pro-sized model. It provides longer operation sequences for design
discussion. It does not claim that this exact configuration is already a
deployed or runnable recipe.

## 1. Reality boundary

No single local configuration currently combines all of the following:

- the full 61-layer DSv4 Pro shape;
- PP3 and VPP2;
- combined forward/backward EP overlap;
- the 3,072-rank production topology; and
- M-FSDP v2 with dense HFSDP outer128 × inner8.

The available inputs establish different parts of the case:

- The full-shape model definition provides 61 transformer layers, hidden size
  7,168, 128 attention heads, 384 experts, top-6 routing, DSv4 hybrid
  attention, mHC, and one MTP layer.
- The production topology provides world3072, PP3, EP64, dense HFSDP
  outer128 × inner8, and pure expert-DP16 outside EP.
- The runnable PP3 compatibility proxy provides a six-microbatch PP3 schedule,
  but VPP and combined EP overlap are disabled.
- The runnable M-FSDP v2 overlap proxy provides VPP2 and combined 1F1B, but it
  uses PP2 and a smaller model.

Current M-FSDP v2 also rejects HSDP in the runnable PP3 proxy. Therefore the
HFSDP operations below describe the intended production scheduling domain,
not a capability already demonstrated by that proxy.

## 2. Confirmed facts and working assumptions

| Property | Value | Status |
|---|---:|---|
| Transformer layers | 61 | Confirmed by model definition |
| Hidden size | 7,168 | Confirmed by model definition |
| Experts / router top-k | 384 / 6 | Confirmed by model definition |
| MTP layers | 1 | Confirmed by model definition |
| Sequence length | 4,096 | Confirmed by model and PP3 proxy |
| World / PP / EP | 3,072 / 3 / 64 | Confirmed target topology |
| Ranks per PP stage | 1,024 | Derived from world / PP |
| Dense sharding | HFSDP outer128 × inner8 | Confirmed target topology |
| Expert placement | EP64 × pure expert-DP16 | Confirmed target topology |
| TP / CP / ETP | 1 / 1 / 1 | Working assumption |
| VPP | 2 | Working assumption required by this design |
| Microbatches `M` | 6 | Working assumption matching the PP3 proxy |
| MBS / GBS | 1 / 6,144 | GBS derived from MBS × dense-DP1024 × M6 |
| VP microbatch group | 3 | Working assumption using the PP-size default |
| FSDP unit granularity | One transformer or MTP layer | Working assumption for the operation sequences |

If the production batch size, VPP size, or layout differs, the virtual
microbatch sequence must be regenerated. The layer-pair cases below remain
useful, but their surrounding microbatch labels may change.

## 3. Proposed PP3/VPP2 layer layout

Use this provisional layout:

```text
Et*10|t*10|t*10|t*10|t*11|t*10mL
```

It expands into six virtual stages:

| Virtual stage | Physical owner | Contents |
|---|---|---|
| VS0 | P0/C0 | Embedding + L0–L9 |
| VS1 | P1/C0 | L10–L19 |
| VS2 | P2/C0 | L20–L29 |
| VS3 | P0/C1 | L30–L39 |
| VS4 | P1/C1 | L40–L50 |
| VS5 | P2/C1 | L51–L60 + MTP + loss |

The layout contains all 61 transformer layers and one MTP layer. It gives the
last virtual stage ten transformer layers plus MTP instead of eleven
transformer layers, but it is not yet compute-balanced. Embedding, MTP, loss,
hybrid-attention patterns, and routed-token imbalance can all change stage
time.

Every microbatch follows:

```text
P0/C0 -> P1/C0 -> P2/C0 -> P0/C1 -> P1/C1 -> P2/C1
```

Backward follows the reverse path. The C0/C1 boundary therefore includes the
physical wraparound `P2/C0 -> P0/C1` in forward and `P0/C1 -> P2/C0` in
backward.

## 4. Exact M=6 phase counts

With PP3, VPP2, VP group size three, and the extra forward required by
combined EP overlap:

| Rank | Warmup forwards | Combined pairs | Cooldown backwards |
|---|---:|---:|---:|
| P0 | 8 | 4 | 8 |
| P1 | 6 | 6 | 6 |
| P2 | 4 | 8 | 4 |

The rows below are local scheduler order, not a shared clock.

## 5. R1 — P0's complete steady-state window

P0 reaches steady state after this warmup tail:

```text
F m0/C1
F m1/C1
F m2/C1
F m3/C0
F m4/C0
```

Its complete steady-state window is:

```text
T0: F m5/C0 + B m0/C1     # different chunks, two roots
T1: F m3/C1 + B m1/C1     # same chunk
T2: F m4/C1 + B m2/C1     # same chunk
T3: F m5/C1 + B m0/C0     # different chunks, reverse direction
```

### R1a — different chunks and two roots

T0 expands to ten layer pairs:

```text
forward m5/C0:  L0  -> L1  -> L2  -> L3  -> L4
                L5  -> L6  -> L7  -> L8  -> L9

backward m0/C1: L39 -> L38 -> L37 -> L36 -> L35
                L34 -> L33 -> L32 -> L31 -> L30

paired:
  F L0 + B L39
  F L1 + B L38
  F L2 + B L37
  F L3 + B L36
  F L4 + B L35
  F L5 + B L34
  F L6 + B L33
  F L7 + B L32
  F L8 + B L31
  F L9 + B L30
```

Forward uses the C0 FSDP root and backward uses C1. Both roots can have current
and prefetched parameter storage live. At the PP boundary, P0 sends the C0
activation to P1/C0 and sends the C1 input gradient across the VPP boundary to
P2/C0.

### R1b — same chunk and opposing traversal

T1 expands to:

```text
forward m3/C1:  L30 -> L31 -> L32 -> L33 -> L34
                L35 -> L36 -> L37 -> L38 -> L39

backward m1/C1: L39 -> L38 -> L37 -> L36 -> L35
                L34 -> L33 -> L32 -> L31 -> L30

paired:
  F L30 + B L39
  F L31 + B L38
  F L32 + B L37
  F L33 + B L36
  F L34 + B L35
  F L35 + B L34
  F L36 + B L33
  F L37 + B L32
  F L38 + B L31
  F L39 + B L30
```

The two directions share the C1 root but belong to different microbatches and
autograd graphs. The forward prefetch order advances while the backward order
retreats. T2 repeats the same structural case with different microbatches.

## 6. R2 — unequal forward and backward plans on P1

P1/C0 contains ten transformer layers while P1/C1 contains eleven. A combined
operation can therefore have an unpaired tail.

For `F m3/C0 + B m0/C1`:

```text
F L10 + B L50
F L11 + B L49
F L12 + B L48
F L13 + B L47
F L14 + B L46
F L15 + B L45
F L16 + B L44
F L17 + B L43
F L18 + B L42
F L19 + B L41
        + B L40       # backward-only tail
```

The reverse chunk combination has a forward-only tail:

```text
F m3/C1 + B m0/C0

F L40 + B L19
F L41 + B L18
F L42 + B L17
F L43 + B L16
F L44 + B L15
F L45 + B L14
F L46 + B L13
F L47 + B L12
F L48 + B L11
F L49 + B L10
F L50               # forward-only tail
```

The schedule must handle unequal plan lengths without assuming that every
combined call is a rectangle of paired layers. Prefetch, release, and stream
ownership must continue correctly through the tail.

## 7. R3 — P2 endpoint with MTP and an exact-unit crossing

Consider P2's first same-chunk C1 pair:

```text
F m1/C1 + B m0/C1
```

The C1 scheduled units are ten transformer layers plus MTP:

```text
forward:   L51 L52 L53 L54 L55 L56 L57 L58 L59 L60 MTP
backward:  MTP L60 L59 L58 L57 L56 L55 L54 L53 L52 L51
```

The combined sequence is:

```text
F L51 + B MTP
F L52 + B L60
F L53 + B L59
F L54 + B L58
F L55 + B L57
F L56 + B L56       # same FSDP unit, different microbatches
F L57 + B L55
F L58 + B L54
F L59 + B L53
F L60 + B L52
F MTP + B L51
```

P2/C1 is the last virtual stage:

```text
PP.RECV_F(m1) -> forward layers and MTP -> local loss(m1)
local loss gradient(m0) -> backward MTP and layers -> PP.SEND_B(m0)
```

There is no `PP.RECV_B` for m0 and no `PP.SEND_F` for m1. Under the assumed
layer-level FSDP wrapping, the center pair requests forward and backward use
of the L56 unit in the same combined step. With formats that use different
forward and backward parameter orientations, the two consumers cannot be
assumed to share one gathered representation.

This case also pairs heterogeneous units at both ends: transformer forward
with MTP backward, then MTP forward with transformer backward.

## 8. Key operations inside one DSv4 layer pair

For `F Li + B Lj`, use this operation inventory. Ordering between the two dense
all-gathers is deliberately left for the scheduling design.

```text
dense parameter readiness:
  HFSDP.AG_B_dense(Lj) over inner-FSDP8
  HFSDP.AG_F_dense(Li) over inner-FSDP8
  READY_B(Lj), READY_F(Li)

backward EP:
  EP64.COMBINE_B(Lj)

forward dense compute:
  hybrid attention + mHC + normalization + router(Li)

backward expert compute:
  local-expert dgrad(Lj)

forward EP:
  EP64.DISPATCH_F(Li)

backward expert work:
  local-expert wgrad(Lj)
  EP64.DISPATCH_B(Lj)

forward expert work:
  six-local-expert compute(Li)
  EP64.COMBINE_F(Li)

backward dense work:
  attention/router/mHC dgrad and wgrad(Lj)

parameter release:
  HFSDP.RELEASE_F_dense(Li)
  HFSDP.RELEASE_B_dense(Lj)

gradient state:
  mark dense and expert contributions accumulated for microbatch
```

Expert parameters are replicated over expert-DP16 in this target, so they do
not have an expert-parameter all-gather. If the target instead adopts expert
M-FSDP sharding, expert parameter gather/release and reduce-scatter must be
added; that is a different scheduling case.

EP, PP, dense all-gather, dense gradient reduction, expert gradient reduction,
and compute may use different streams or communicators, but they still compete
for physical interconnect and GPU resources.

## 9. R4 — accumulation versus final backward

T1 contains `B m1/C1`, which is not the last microbatch for C1. Its gradients
normally accumulate without launching the chunk's final DP synchronization:

```text
B m1/C1
  -> finish dgrad and wgrad contributions
  -> release backward full parameters
  -> retain/accumulate gradient state
  -> no final dense/expert reduction yet
```

P0's cooldown is:

```text
B m1/C0
B m2/C0
B m3/C1
B m4/C1
B m5/C1       # final C1 microbatch: make C1 reductions eligible
B m3/C0
B m4/C0
B m5/C0       # final C0 microbatch: make C0 reductions eligible
```

At `B m5/C1`, the production target can issue:

```text
dense C1 gradients complete
  -> reduce-scatter over inner-FSDP8

expert C1 gradients complete
  -> all-reduce over expert-DP16
```

The same transition occurs for C0 at `B m5/C0`. After all chunks finish, any
remaining dense outer-HFSDP synchronization and root gradient finalization
must complete before optimizer use.

This drain has no new forward work to hide backward gathers or reductions, so
it must be analyzed separately from the steady-state layer pairs.

## 10. Initial design cases

Use these cases in order:

1. **R3 center transition:**
   `F L55 + B L57 -> F L56 + B L56 -> F L57 + B L55`.
2. **R1 same-root traversal:** ten successive pairs in P0/C1.
3. **R1 two-root traversal:** P0/C0 forward paired with P0/C1 backward.
4. **R2 unequal tail:** ten pairs followed by one unpaired operation.
5. **R4 final accumulation:** last C1 and C0 backwards followed by reductions
   and finalization.

For each case, the next step is a dependency graph that identifies:

- current and next forward/backward parameter units;
- parameter representation and readiness events;
- last consumers and release points;
- EP and PP communication dependencies;
- gradient-completion and reduction eligibility; and
- the maximum live parameter and gradient working set.

Only after those dependencies are agreed should the design assign streams,
communicators, prefetch distance, or buffer count.

## 11. Source artifacts

- [DSv4 Pro model definition](../agentic-mcore-dev/agentic-mcore-dev/configs/benchmarking/models/deepseek_v4_pro.yaml)
- [PP3 production topology](deepseek_v4_pro_pp3_hfsdp_ep64_schedule.md)
- [Runnable PP3 compatibility proxy](../agentic-mcore-dev/agentic-mcore-dev/configs/benchmarking/recipes/deepseek_v4_pro_4gpu_proxy/b200/bf16_48GPU_TP1PP3EP8_mfsdp_v2_realdata_convergence_100step.yaml)
- [Runnable PP2/VPP2 M-FSDP v2 overlap proxy](../agentic-mcore-dev/agentic-mcore-dev/configs/benchmarking/recipes/deepseek_v4_pro_4gpu_proxy/b200/bf16_8GPU_TP1PP2VPP2EP2_mfsdp_v2_1f1b_overlap.yaml)
