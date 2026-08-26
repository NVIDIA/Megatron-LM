# Deprecating `parallel_state`

`megatron.core.parallel_state` holds the process groups for a **single, global** parallel grid.
Megatron-Core is migrating to explicit process-group passing via `ProcessGroupCollection`.

This page answers the question that keeps coming up in review: *is this deprecated, and what do I
use instead?*

## Why it matters

For a job with one parallel grid, reading a group from `parallel_state` is merely deprecated —
it returns the right answer.

For a job that builds **independent** grids — a vision encoder and an LLM with different
parallelism, generalized tensor parallelism, MIMO — it returns a group belonging to the *wrong*
grid. Nothing raises. The collective runs on the wrong ranks, or a rank is seeded with the wrong
RNG offset, and the job produces wrong numbers.

That is the reason this migration exists. It is a correctness problem before it is a tidiness
problem.

## What is deprecated

| Tier | What | Replacement | Status |
|---|---|---|---|
| **1** | Group accessors — `get_*_group()` | `pg_collection.<field>` | Deprecated. Banned in new `megatron/core` code. |
| **2** | Rank / size accessors — `get_*_rank()`, `get_*_world_size()` | `pg.rank()` / `pg.size()` on the group you were passed | Deprecated. Banned in new `megatron/core` code. |
| **3** | Pipeline-stage predicates — `is_pipeline_first_stage()`, `is_pipeline_last_stage()` | A process-group-based equivalent exists at `megatron/core/inference/communication_utils.py` | Being designed. These encode virtual-pipeline semantics, not just a group. |
| **4** | Non-group global state — virtual-pipeline rank/size, `GlobalMemoryBuffer`, gloo groups, NCCL options | **None yet.** `ProcessGroupCollection` does not cover these. | Needs design. Do not migrate ad hoc. |

**Not deprecated:** `initialize_model_parallel`, `destroy_model_parallel`, `is_initialized`.
These are the intended long-term surface — bootstrap, and nothing else.

## `use_mpu_process_groups()` is not a migration target

`ProcessGroupCollection.use_mpu_process_groups()` builds a collection *by reading the same global
state*. It exists to keep already-migrated call sites working during the transition.

```python
# Not progress -- both read the global grid.
- tp_group = parallel_state.get_tensor_model_parallel_group()
+ pg_collection = ProcessGroupCollection.use_mpu_process_groups()
```

New code must accept a `ProcessGroupCollection` from its caller. Existing call sites are being
removed; do not add more.

## What this means for your change

**Writing a new feature?** Accept a `ProcessGroupCollection` or an explicit
`torch.distributed.ProcessGroup` and pass it through. Do not add a `None` default that falls back
to `parallel_state` — that fallback is the bug, not the convenience.

```python
# Don't: silently correct for one grid, silently wrong for two.
def my_op(x, tp_group=None):
    if tp_group is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()

# Do: the caller knows which grid it is on.
def my_op(x, tp_group: torch.distributed.ProcessGroup):
    ...
```

**Fixing a bug?** Leave existing `parallel_state` calls as they are. Changing process-group
plumbing has its own blast radius and belongs in its own PR — bundling it into a fix makes the fix
harder to review and harder to revert.

**Reviewing?** Flag new tier-1/tier-2 accessor use in `megatron/core`. Allowed exceptions:
`parallel_state.py` itself, `process_groups_config.py`, bootstrap code that materializes a
collection from the globals, tests, and explicitly-commented migration fallbacks.

This guidance applies to `megatron/core`. It does not apply to `megatron/training` or other
training-loop code unless a change explicitly opts in.
