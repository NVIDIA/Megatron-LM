# Async Scheduling

Async scheduling for dynamic decode overlaps CPU bookkeeping for step `N` with
GPU work for step `N + 1`. The architecture is transactional: one canonical
plan describes the candidate forward, one transaction owns its speculative
state, and participants own subsystem side effects.

```text
idle -> planned -> prepared -> launched -> resolved -> committed/rolled_back -> retired
```

The text generation controller still calls model primitives, sampling, and
context bookkeeping. Async policy and transaction state live in
`AsyncDecodeCoordinator`, `AsyncDecodePlan`, `AsyncDecodeTransaction`, and
`AsyncTransactionParticipant` implementations.

## Canonical Plan

`AsyncDecodePlan` is the immutable description of a candidate decode step. It
contains:

- request IDs and source rows for the planned active layout
- query lengths, KV offsets, token positions, token-to-request rows, and KV
  block indices
- reserved KV block metadata and finished request IDs
- CUDA graph shape, decode stride, and padded request count
- the prepared `AsyncLayoutSnapshot`
- row-map decision, graph compatibility, and layout compatibility
- EP, Mamba, MTP, logprob, and expected resource requirements

The context builds this plan while preparing the speculative next-step layout.
When the controller launches the async forward, the transaction keeps that plan
and attaches the runtime layout snapshot used for pending-forward resolution.

## Transaction Lifecycle

`AsyncDecodeTransaction` owns speculative async work for one pending forward:
state, fences/events, sample tickets, resource ledger, row map, participant
state, and diagnostics.

The coordinator owns transaction creation, pending detection, and retirement.
Controller compatibility helpers delegate to the coordinator so legacy call
sites can keep using `_pending_async_transaction`,
`_has_pending_async_forward_state`, `_begin_async_step_transaction`, and
`_retire_async_transaction`.

The lifecycle rules are:

- `planned` and `prepared`: a candidate plan and CPU layout exist, but the
  pending forward has not been launched.
- `launched`: H2D bookkeeping and the forward have been enqueued before CPU
  bookkeeping drains, and resources that may still be visible to the forward
  are marked in flight.
- `resolved`: the pending forward was matched against current rows and either
  identity reuse or row-mapped reuse was selected.
- `committed`: participant-owned side effects are accepted exactly once.
- `rolled_back`: participant-owned side effects are released or discarded
  exactly once.
- `retired`: the transaction no longer owns active pending-forward state.

The hot-path ordering is intentionally pre-commit:

```text
forward(N) -> sample(N) -> prepare plan(N+1) -> launch forward(N+1)
           -> CPU bookkeeping(N) -> resolve/commit or rollback(N+1)
```

The next-step plan is prepared and launched before CPU bookkeeping can reveal
whether it will be reused. Resolution after bookkeeping is therefore a
transaction decision, not a second scheduling path.

## Central Eligibility

`classify_async_eligibility` returns a structured `AsyncEligibilityDecision`.
It centralizes async scheduling gates such as enablement, step barriers, CUDA
graph availability, graph scope, pipeline parallelism, unsupported sampling
backends, MTP constraints, decode-only state, active request count, admission
barriers, and graph stride compatibility.

Pending-forward reuse is centralized separately in
`resolve_async_pending_forward`. It returns `AsyncPendingForwardDecision` with
the reusable flag, row map, row-mapped flag, discard reason, row-map policy, and
layout/graph compatibility. `AsyncDecodeTransaction.pending_forward_decision`
previews this decision without changing state. `resolve_against_current`
records the decision on the plan and transitions the transaction to `resolved`
or a discarded state. Controller code consumes this decision instead of
recomputing row-map or graph facts in multiple places.

## Layout And Row Maps

`AsyncLayoutSnapshot` captures the prepared request IDs, graph shape, request
lengths, KV offsets, token rows, token positions, KV block mapping, and optional
Mamba read/write indices. A pending forward can be reused only when:

- graph shape and decode stride match the current step
- every current request ID can be mapped to a pending row
- request, token, KV, and Mamba layout fields match after applying that row map

`AsyncRowMapPolicy.REUSE` is the default. It preserves PR 5202 behavior and
performance by allowing row-mapped pending-forward reuse when layouts are
compatible.

`AsyncRowMapPolicy.IDENTITY_ONLY` rejects non-identity row maps. It is intended
for exact async-off/async-on parity validation because it avoids inherited PR
5202 row-mapped numeric drift where possible. The default remains `reuse` until
Nano inference-bench data shows `identity_only` is within the configured
performance threshold.

## Participants

Each subsystem that owns speculative state is represented as an
`AsyncTransactionParticipant` with `prepare`, `validate`, `commit`, `rollback`,
and `diagnostics` hooks.

Current participants include:

- `AsyncMambaStateParticipant`: accepts async Mamba candidate banks on commit.
- `AsyncSampleReadbackParticipant`: tracks the sample readback ticket lifetime.
- `AsyncLogprobMTPParticipant`: records generated-logprob and MTP requirements.
- `AsyncResourceParticipant`: releases deferred resources on commit and drains
  all speculative resources on rollback.

Participant hooks are idempotent. A commit or rollback retry must not publish
Mamba banks twice, release a KV block twice, or consume the same sample ticket
twice.

Participants are attached through transaction helpers. If a participant is
added after the transaction has already prepared its hooks, the new participant
is prepared immediately. Attaching participants after commit, rollback, or
retirement is rejected.

## Resource Ownership

`AsyncResourceLedger` records KV reservations, deferred KV blocks, deferred
Mamba slots, in-flight state, and consumed reservations.

When a speculative forward is launched, the coordinator marks the ledger in
flight and attaches an `AsyncResourceParticipant` when the ledger owns
reservations or deferred resources. During CPU
reconciliation, matching reserved KV blocks can be consumed by request ID and
block column. Unused reservations are deferred. Request cleanup that overlaps an
in-flight forward defers KV and Mamba releases through the same ledger.

On commit, the resource participant releases deferred resources through the
context allocators and clears in-flight state. On rollback, it first moves any
unused reservations into the deferred list and then releases everything. This
keeps prepare, commit, and rollback symmetric for speculative resources.

The context borrows the active ledger reference only while the coordinator-owned
transaction is in flight. Commit and rollback release behavior remains owned by
the transaction ledger.

## Mamba, MTP, And Logprobs

Hybrid models include Mamba bank read/write indices in the layout snapshot.
Pending forwards are reusable only when the current rows match the planned
Mamba layout after row-map resolution. On commit, the Mamba participant accepts
candidate banks for the transaction plan's request IDs. On rollback, candidate
banks are not published.

MTP uses pending base logits for verification, rewinds rejected KV cache
entries, computes serial MTP logits for verified inputs, and may prepare the
next async decode after sampling. Generated logprobs and top-n logprobs use the
resolved row indices when row-mapped pending logits are consumed.

## EP Coordination

`EPAsyncStepProtocol` still owns tagged collectives for work consensus, step
completion, graph-shape sync, step-begin reuse/discard decisions, and async
handoff launch/skip decisions.

The coordinator owns per-step EP state separately from the transaction. It
previews pending-forward reuse at step begin, records the handoff decision for
the current step, and discards prepared state without launching a transaction
when EP skips an async handoff.

## Diagnostics

Async diagnostics are intentionally cheap and deterministic:

- controller counters track eligibility, prepared/launched/reused/committed
  forwards, rollback/discard, row-mapped reuse, identity reuse, graph mismatch,
  layout mismatch, and row-map policy
- transaction diagnostics include state, request IDs, row map, discard reason,
  plan diagnostics, and participant diagnostics
- resource diagnostics include reservations, deferred KV blocks, deferred
  Mamba slots, Mamba leases, and consumed reservations
- EP diagnostics include protocol counters and transaction-local EP decisions

The focused unit tests cover overlap ordering, no pre-launch host sync, reuse
and commit counters, rollback and release-once behavior, graph mismatch
discard, identity reuse, row-mapped reuse, Mamba state, MTP/logprob handling, EP
launch/skip symmetry, and ZMQ protocol compatibility.
