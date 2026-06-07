# Async Scheduling

Async scheduling for dynamic decode overlaps the CPU work for the next step with
the current GPU forward. Each speculative async forward is represented by one
`AsyncStepTransaction`:

```text
prepare -> launch -> resolve -> commit -> retire
```

The transaction is the top-level owner for async lifecycle state. It carries the
prepared layout snapshot, sample readback ticket, resource ledger, CUDA fences,
EP launch decision, resolved row map, state, and discard reason. The controller
coordinates model-specific operations, but it does not keep separate pending
forward, sample, or resource state.

## Lifecycle

`prepare` builds next-step decode metadata in `DynamicInferenceContext` without
publishing it as the live layout. The controller records this prepared layout in
an `AsyncLayoutSnapshot` when it launches the speculative forward.

`launch` publishes the prepared decode plan, transfers bookkeeping to the GPU,
launches the forward, attaches the `AsyncSampleTicket` and
`AsyncResourceLedger`, and marks resources as in flight.

`resolve` compares the pending transaction snapshot with the current context at
the next decode step. A compatible transaction can be reused directly or through
a row map if request rows were compacted.

`commit` consumes the pending forward result and accepts side effects such as
async Mamba writes. `discard` records why a transaction could not be reused.

`retire` clears the active transaction after deferred resources are released.

## Layout Resolution

`AsyncLayoutSnapshot` captures the request IDs, CUDA graph shape, request query
lengths, KV length offsets, token-to-request mapping, token positions, KV block
mapping, and optional Mamba read/write indices for the prepared forward. It is
compared against a current-context snapshot with:

- `graph_compatible_with`, which checks decode stride and padded CUDA graph
  request count.
- `row_map_to_current`, which maps current request rows to pending forward rows.
- `layout_compatible_with`, which validates request, token, KV, and Mamba layout
  fields after applying the row map.

If any check fails, the transaction is discarded and the controller falls back to
a serial forward for the current step.

## Sample Readback

`AsyncSampleReadback` owns the stable CUDA sample slots, pinned CPU sample
slots, source-ready events, copy-done events, and copy stream. A transfer returns
an `AsyncSampleTicket` for one transaction, including the slot number, active
request count, CUDA and CPU token buffers, optional MTP token buffers, and fences.

The ticket keeps sample lifetime explicit: request bookkeeping waits on the
copy-done event when it needs CPU samples, while the next async launch can
continue using the default stream.

## Resource Retirement

`AsyncResourceLedger` owns speculative resource lifetime for a transaction. It
records reserved KV blocks from the prepared decode plan, lets the context
consume matching reservations during CPU reconciliation, and moves unused
reservations into deferred release.

When a speculative forward is in flight, request cleanup defers KV blocks and
Mamba slots rather than returning them to allocators immediately. On transaction
retirement, `release_deferred` returns all deferred resources through the context
allocators and clears the in-flight flag.

## Mamba Ownership

Hybrid models include Mamba slot and bank state in the layout snapshot. Pending
forwards are reusable only when the current rows can be mapped to the snapshot's
Mamba read/write indices. When a pending forward commits, the context accepts the
async Mamba state for the active request IDs. Slot frees that overlap an
in-flight forward are deferred through the resource ledger.

## MTP And Logprobs

MTP uses the pending base logits for verification, rewinds rejected KV cache
entries, computes serial MTP logits for the verified inputs, and may launch the
next async decode after sampling. Accepted-token tensors are reset after each
returned step.

Generated logprobs and top-n logprobs use row indices from the resolved
transaction when a compacted pending forward is reused. Sampling bookkeeping is
skipped only when the async path has already collected everything needed for
sampling and logprob results.

## EP Coordination

`EPAsyncStepProtocol` is phase-centered. Tagged collectives are grouped by phase
and step ID:

- work consensus and step completion
- step-begin pending-forward reuse, discard, and row-map decisions
- async handoff launch or skip decisions

Dummy EP ranks join the same phases as real ranks. A real-rank launch only
proceeds when the EP handoff phase agrees that all participating ranks can
launch; otherwise the prepared plan is discarded and the step records an
explicit skip reason.

## Diagnostics

Async diagnostics are intentionally tied to transaction components:

- `AsyncStepTransaction.state`, `row_map`, and `discard_reason`
- `AsyncSampleTicket.slot`, `active_request_count`, and fences
- `AsyncResourceLedger` reservation, deferral, and consumption counters
- eligibility decisions from `classify_async_eligibility`
- EP phase counters from `EPAsyncStepProtocol.diagnostics()`

These counters support focused tests for pending-forward reuse, row-mapped
reuse, graph mismatch discard, sample slot ownership, deferred KV and Mamba
release, MTP, logprobs, stop words, and EP launch/skip symmetry.
