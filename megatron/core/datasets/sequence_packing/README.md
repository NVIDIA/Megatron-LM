# Sequence Packing and Dynamic CP Planning

This package defines an API-only planning layer for the Dynamic CP and sequence
packing redesign. It separates three concerns that are currently mixed across
the data iterator, scheduler, and forward/backward schedule:

1. Build packed units from variable-length source sequences.
2. Place those packed units onto backend-specific rank groups.
3. Execute the resulting plan by rerouting data, materializing THD tensors, and
   broadcasting metadata.

The first PR for this package is intentionally non-invasive. It adds contracts
and design documentation only. Existing runtime behavior remains unchanged.

## Current Main Baseline

Current main already has Dynamic CP support, but the responsibilities are tightly
coupled:

```text
HybridCPDataLoaderWrapper
  - reads packed samples from the data iterator
  - extracts sub-sample sequence lengths
  - gathers sequence lengths across DP
  - calls BalancedCPScheduler
  - reroutes sub-samples across DPxCP ranks

BalancedCPScheduler
  - decides local CP size for each sub-sample
  - assigns sub-samples to DPxCP ranks
  - groups assignments to avoid CP deadlock

hybrid_context_parallel_forward_backward
  - consumes the scheduler output
  - creates one-sample iterators
  - sets local_cp_size
  - inserts barriers between groups
  - drives forward/backward
```

This works, but it makes it hard to reuse sequence packing for static CP,
Dynamic CP, HybridEP, or offline planners.

## Target Architecture

```text
raw variable-length samples
        |
        v
SequencePackingScheduler
  generic, online or offline
  answers: which sequences form each pack?
        |
        v
PackDescriptor
  THD pack intent: sample IDs, lengths, cu_seqlens shape
        |
        v
PlacementScheduler
  backend-specific
  answers: which ranks execute each pack?
        |
        v
ExecutionPlan
  barrier-safe groups of pack assignments
        |
        v
PlanExecutionEngine
  future shared engine:
    - reroute samples
    - materialize THD tensors
    - build cu_seqlens / cu_seqlens_padded
    - broadcast TP/PP metadata
    - execute barriers and train/eval integration
        |
        v
PackedSeqParams / Transformer Engine / MoE backend
```

## THD's Role

THD is the common packed representation:

```text
tokens: [T, H, D]
metadata: cu_seqlens, cu_seqlens_padded, max_seqlen
```

THD is not the scheduler. It is the representation produced by packing and
consumed by downstream backends such as standard varlen attention, static CP,
Dynamic CP, HybridEP, and future placement strategies.

## Scheduler Responsibilities

### SequencePackingScheduler

Generic layer. It maps source sequence metadata to pack descriptors.

Examples:

- Online DP-balanced packing from PR #3386.
- Offline global bin packing from a precomputed plan file.
- Future HybridEP-aware packing that balances expert token pressure.

It should not create process groups, run all-to-all, call forward/backward, or
decide Dynamic CP group membership.

### PlacementScheduler

Backend-specific layer. It maps packs to logical execution resources.

Examples:

- Current main's `BalancedCPScheduler` as a Dynamic CP placement scheduler.
- Static CP placement, where every pack runs on the same CP group.
- HybridEP placement, where packs carry expert-parallel routing constraints.
- No-CP placement, where packs execute on a single data-parallel rank.

### PlanExecutionEngine

Shared future engine. It consumes `ExecutionPlan` and owns the operational
steps that should not be duplicated by every scheduler:

- sample reroute,
- THD materialization,
- padding policy,
- metadata broadcast,
- barrier execution,
- train/eval integration.

This package defines the contract only; engine implementation is future work.

## Mapping Existing Work

### Current Main

Current main's Dynamic CP path maps to the target design as:

```text
BalancedCPScheduler
  -> PlacementScheduler for Dynamic CP

HybridCPDataLoaderWrapper + hybrid_context_parallel_forward_backward
  -> prototype PlanExecutionEngine responsibilities
```

The first implementation follow-up should preserve current behavior while
teaching the current Dynamic CP path to emit an `ExecutionPlan`.

### PR #3386

PR #3386 adds an online DP-balanced sequence-packing path. In this design:

```text
DpBalancedScheduler
  -> SequencePackingScheduler implementation

data movement, packing materialization, TP/PP broadcast
  -> PlanExecutionEngine responsibilities
```

### PR #5191

PR #5191 adds packing inside the existing Dynamic CP path. In this design:

```text
Dynamic CP placement
  -> PlacementScheduler implementation

ad hoc pack materialization inside HybridCPDataLoaderWrapper
  -> shared PlanExecutionEngine responsibility
```

The long-term goal is to share one packing/materialization engine instead of
duplicating pack-building logic between PR #3386 and PR #5191.

## Planned Follow-up Sequence

```text
PR 1: API and design only
PR 2: adapt current main Dynamic CP to emit ExecutionPlan
PR 3: adapt #3386's DP-balanced packing to SequencePackingScheduler
PR 4: compose sequence packing with Dynamic CP placement
PR 5: implement shared execution engine and remove duplicate materialization
PR 6: add HybridEP-specific placement/padding extensions
```

Each implementation PR should keep tests with the code it validates. The
API-only PR can remain CPU-only because it does not create process groups or run
distributed collectives.
