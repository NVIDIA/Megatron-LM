# FsdpExecutionRunner: Trace Path and Optimization Path

**Status:** Design proposal for the M-FSDP v2 execution-order runner.

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
  `suggest_prefetch(module, orientation)` then returns the next **unshard**
  event (skipping intervening reshard events) as the prefetch target.
- `record_reshard(module)` — validates the reshard and advances the cursor;
  `suggest_skip_reshard(module)` then returns whether the actual reshard can
  be **skipped** so the storage stays resident.

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

### 3.2 Why same orientation?

M-FSDP v2 MXFP8 parameter groups keep separate row-wise (forward GEMM) and
column-wise (backward GEMM) payloads. Keeping storage resident across an
orientation change would leave the wrong payload materialized, so the
optimization only applies when the immediate re-consume uses the same
orientation.

### 3.3 When not applied

- Different orientation on the immediate re-consume.
- Another module's consume intervenes between the reshard and the re-consume.
- Default mode (`use_trace_replay=False`): the runner stays idle and every
  reshard is executed normally.
- Tracing phase or after a divergence.

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
| `suggest_prefetch(module, orientation)` | optimization | next module to all-gather ahead |
| `suggest_skip_reshard(module) -> bool` | optimization | whether to keep storage resident |
| `complete_trace()` | trace | compile the cycle at the batch boundary |
| `report()` | diagnostics | replay statistics |
| `phase`, `is_tracing`, `use_trace_replay` | — | runner state |

`FsdpModule` integration:

```python
# unshard_parameters (unshard entry point)
self.context.runner.record_unshard(self, orientation)
prefetch = self.context.runner.suggest_prefetch(self, orientation)
if prefetch is not None:
    next_module, next_orientation = prefetch
    next_module._unshard_parameter_groups(next_orientation)

# _reshard_parameter_groups (release entry point)
self.context.runner.record_reshard(self)
if self.context.runner.suggest_skip_reshard(self):
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
- **Dedup** (`_consumed_this_round`) keeps the trace at one unshard per module
  per round despite per-sub-module hooks; `record_reshard` clears the module's
  dedup entry so the next round records a fresh unshard.
- **Memory** is bounded: skipping a reshard keeps at most one extra module's
  storage resident, and only while it is immediately reused.

## 6. Open questions

1. Should the reshard-skip policy be extended to a *window* (keep resident if
   re-consumed within N events) instead of strictly immediate? This trades
   memory for fewer all-gathers and needs a residency budget.
2. Should the optimization path also skip the all-gather for a module that is
   resident but whose reshard was *not* skipped (e.g. prefetched modules)?
3. How should the optimization path interact with the MXFP8 scale-inverse
   grids when a payload is kept resident across optimizer steps?
4. Should `complete_trace()` compile a more elaborate plan with residency
   windows and configurable prefetch distances instead of the event cursor?

## 7. Sources

- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/execution_runner.py`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py`
