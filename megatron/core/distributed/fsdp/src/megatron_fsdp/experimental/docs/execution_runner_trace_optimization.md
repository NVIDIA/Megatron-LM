# FsdpExecutionRunner: Trace Path and Optimization Path

**Status:** Implemented design for the M-FSDP v2 execution-order runner.

**Audience:** Distributed-training developers working on Megatron-FSDP v2 with
pipeline (PP/VPP) and expert-parallel combined-1F1B schedules.

## 1. Problem

Under the combined-1F1B + VPP schedule, parameter consumption is
occurrence-based rather than a single traversal of the module tree:

- The same FSDP unit can be consumed in forward and backward (e.g. `F L56`
  and `B L56` for different microbatches).
- Model chunks interleave, and warmup/steady/cooldown differ per pipeline
  rank.
- The schedule fires one fine-grained hook per sub-module (dense layer,
  experts), so the same `FsdpModule` can be touched several times per pass.

The static `forward_order` / `backward_order` sequences cannot express this
runtime path, so M-FSDP v2 uses a per-context `FsdpExecutionRunner` that
**traces** the real execution and **replays** it to drive prefetch. This
document defines two cooperating paths inside the runner:

1. **Trace path** — records the real op stream (consume and reshard events).
2. **Optimization path** — during replay, translates the real ops into an
   optimized plan (e.g. skip a reshard + all-gather pair when the traced
   schedule re-consumes the same module immediately).

## 2. Design: two paths in one runner

```text
                 FsdpContext (one per rank, shared across VPP chunks)
                              |
                     FsdpExecutionRunner
                    /                    \
            Trace path                 Optimization path
   records real ops during      replay validates against the
   the first global batch       trace and returns directives
        (consume, reshard)      (prefetch target, skip reshard)
                              |
                    FsdpModule entry points
              pre_forward / pre_backward / unshard_parameters
              _reshard_parameter_groups
```

### 2.1 Trace path (global batch 1)

The runner records every fine-grained execution event as a `RunnerEvent`:

```python
class EventKind(Enum):
    UNSHARD = auto()   # module params are unsharded for compute
    RESHARD = auto()   # module params are released after compute

@dataclasses.dataclass(frozen=True)
class RunnerEvent:
    kind: EventKind
    module: FsdpModule
    orientation: str | None   # rowwise/colwise; None for reshard
```

The trace is the ordered list of events observed during the first global
batch:

```text
[UNSHARD(L2, rowwise), RESHARD(L2), UNSHARD(L2, colwise), RESHARD(L2),
 UNSHARD(L0, rowwise), RESHARD(L0), UNSHARD(L0, colwise), RESHARD(L0), ...]
```

During tracing no prefetch is issued (demand-only) and no reshard is
optimized away. The training loop calls `complete_trace()` at every
global-batch boundary (via the optimizer step); the first non-empty trace
compiles into the replay cycle.

### 2.2 Optimization path (global batch 2+)

During replay, each real op is validated against the traced event at the
current position (`_replay_index`), and the runner returns an optimization
directive:

- `record_unshard(module, orientation)` — validates the unshard against
  the traced event, advances the cursor, and
  `suggest_prefetch_plan(module, orientation, depth=N)` then returns the
  Nth future **unshard** occurrence (skipping reshard events).
- `record_reshard(module)` — validates the reshard and advances the cursor;
  `suggest_skip_reshard(module)` then returns whether the actual reshard can
  be **skipped** so the storage stays resident.

Lookahead never wraps across the global-batch boundary. If a deep target has
an earlier physical occurrence, the plan reserves that materialization at its
last intervening reshard instead of issuing an all-gather that would be
released before the target. Immediate and reserved prefetches are tracked
until their exact target occurrence and flushed before optimizer work.

A mismatch (wrong event kind, module, or orientation) is a divergence:
the runner clears the trace, re-traces from that event, and degrades to
demand-only execution until a full cycle matches again.

## 3. Optimization: skip reshard + unshard on immediate reuse

### 3.1 Rule

During replay, when a reshard for module `M` is about to execute, the runner
looks at the traced event that follows the reshard:

```text
... RESHARD(M)  UNSHARD(M, orient) ...
```

If the next traced unshard is the **same module with the same orientation**,
the storage is re-consumed immediately, so the reshard is unnecessary.
`suggest_skip_reshard(M)` returns `True` and the module keeps its unsharded
storage resident. The following consume then finds the storage already
materialized and skips the all-gather.

### 3.2 Orientation and deep retention

The immediate reshard-skip rule remains conservative and requires the same
orientation. A deep reservation may target the other phase: ordinary groups
ignore orientation, while an MXFP8 unshard materializes and binds both the
row-wise and column-wise payloads.

### 3.3 When not applied

- Different orientation on the immediate re-consume.
- Another module's consume intervenes between the reshard and the re-consume.
- Default mode (`use_trace_replay=False`): the runner stays idle and every
  reshard is executed normally.
- Tracing phase or after a divergence.
- A target would cross the optimizer/global-batch boundary.

### 3.4 Example

Traced cycle (forward-only pass over two layers):

```text
[UNSHARD(L0,row), RESHARD(L0), UNSHARD(L0,row), RESHARD(L0),
 UNSHARD(L1,row), RESHARD(L1), UNSHARD(L1,row), RESHARD(L1)]
```

Replay:

| Real op | Runner directive |
|---|---|
| `record_unshard(L0,row)` | prefetch `(L0,row)` (next unshard) |
| `record_reshard(L0)` | `suggest_skip_reshard` → **skip** — next unshard is `(L0,row)` |
| `record_unshard(L0,row)` | storage resident, no all-gather |
| `record_unshard(L1,row)` | prefetch `(L1,row)` |
| `record_reshard(L1)` | **skip** |
| `record_unshard(L1,row)` | storage resident, no all-gather |

Saves one all-gather and one reshard per module per pass.

## 4. Interface

Public API of `FsdpExecutionRunner` (owned by `FsdpContext`):

| Method | Path | Purpose |
|---|---|---|
| `record_unshard(module, orientation)` | trace | record/validate an unshard event |
| `record_reshard(module)` | trace | record/validate a reshard event; clears the module's unshard round |
| `suggest_prefetch_plan(module, orientation, depth)` | optimization | depth-N target and any intervening reshard gate |
| `defer_prefetch(plan)` / `track_prefetch(plan)` | optimization | track gated or immediately submitted speculative gathers |
| `suggest_skip_reshard(module) -> bool` | optimization | whether to keep storage resident |
| `release_speculative_prefetches()` | lifecycle | release unconsumed full parameters before the optimizer |
| `complete_trace()` | trace | compile the cycle at the batch boundary |
| `report()` | diagnostics | replay statistics |
| `phase`, `is_tracing`, `use_trace_replay` | — | runner state |

`FsdpModule` integration:

```python
# unshard_parameters (unshard entry point)
runner = self.context.runner
if runner.record_unshard(self, orientation):
    prefetch = runner.suggest_prefetch_plan(
        self, orientation, depth=self.context.prefetch_depth
    )
    if prefetch is not None:
        if prefetch.release_after_reshard_index is not None:
            runner.defer_prefetch(prefetch)
        else:
            prefetch.module._unshard_parameter_groups(prefetch.orientation)
            runner.track_prefetch(prefetch)

# _reshard_parameter_groups (release entry point)
runner = self.context.runner
reshard_index = runner.record_reshard(self)
if runner.suggest_skip_reshard(self):
    return
if runner.retain_prefetches_across_reshard(self, reshard_index):
    return  # storage stays resident
for group in self._parameter_groups:
    group.reshard_parameters()
... # release storage on the all-gather stream
```

## 5. Correctness arguments

- **Consume validation** ensures the real schedule still matches the traced
  cycle; divergence falls back to demand-only, never skipping a collective.
- **Reshard skip** is safe only for an immediate same-module,
  same-orientation re-consume, so the materialized payload is always the one
  the next compute reads.
- **Optimizer boundary** lookahead does not wrap, all speculative gathers are
  released before optimizer work, and an incomplete replay forces re-tracing.
- **Dedup** (`_consumed_this_round`) keeps the trace at one unshard per module
  per round despite per-sub-module hooks; `record_reshard` clears the module's
  dedup entry so the next round records a fresh unshard.
- **Memory** is explicit: prefetch depth determines the number of future
  full-parameter lifetimes, and every lifetime is tied to an exact trace
  occurrence or intervening reshard gate.

## 6. Open questions

1. Should the reshard-skip policy be extended to a *window* (keep resident if
   re-consumed within N events) instead of strictly immediate? This trades
   memory for fewer all-gathers and needs a residency budget.
2. Should a future version add a byte budget in addition to the occurrence
   depth, without reintroducing policy machinery on the default hot path?

## 7. Sources

- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/execution_runner.py`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py`
