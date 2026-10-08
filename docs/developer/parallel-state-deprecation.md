# Deprecating `parallel_state`

`megatron.core.parallel_state` holds the process groups for a **single, global** parallel grid.
Megatron-Core is migrating to explicit process-group passing via `ProcessGroupCollection`.

## Why it matters

A global accessor can return the intended group when the model uses the global parallel grid.
It cannot select a different model's grid. For example, an encoder and a language model with
independent parallelism can need different groups on the same rank. Reading globals in either
model can select unrelated ranks or fail if those globals were never initialized. Incorrect
collectives or rank-dependent RNG offsets can then produce incorrect results or hang.

The caller knows which model owns an operation and should supply its groups explicitly.

## What is deprecated

| Tier | What | Replacement | Status |
|---|---|---|---|
| **1** | Group accessors — `get_*_group()` | The matching `pg_collection` field, or an explicit group where no field exists | Deprecated for new model components. See compatibility exceptions below. |
| **2** | Group-local rank / size accessors | `pg.rank()` / `pg.size()` on the supplied group | Deprecated for new model components. Preserve optional-group and initialization semantics. |
| **2** | Global source / peer rank and rank-list accessors | `torch.distributed.get_global_rank(pg, group_rank)` or `get_process_group_ranks(pg)` | Deprecated for new model components. Preserve rank ordering. |
| **3** | Pipeline-stage predicates — `is_pipeline_first_stage()`, `is_pipeline_last_stage()` | Explicit pipeline-group and virtual-stage information | Migration needs virtual-pipeline semantics as well as a group. |
| **4** | Non-group global state — virtual-pipeline rank/size, `GlobalMemoryBuffer` | No general `ProcessGroupCollection` replacement | Needs separate design. Do not migrate ad hoc. |

`initialize_model_parallel`, `destroy_model_parallel`, and `is_initialized` are not deprecated
by this migration. `get_nccl_options` constructs options from caller-supplied configuration;
it does not read global process-group state and is outside this deprecation.

The helpers in `megatron/core/inference/communication_utils.py` accept an explicit pipeline
group, but only check physical stages; they do not replace virtual-pipeline-aware predicates.
Gloo and dynamic (formerly hybrid) data/context groups are still tier-1 group accessors,
although the collection has no corresponding fields. Pass those groups explicitly when migrating
a caller.

## Select the matching group

Collection field names are not always a direct abbreviation of the accessor name. In particular,
expert tensor parallelism uses `expt_tp`, which may differ from the dense `tp` group.

| Accessor | Field in the caller's collection |
|---|---|
| `get_tensor_model_parallel_group()` | `tp` |
| `get_pipeline_model_parallel_group()` | `pp` |
| `get_context_parallel_group()` | `cp` |
| `get_expert_model_parallel_group()` | `ep` |
| `get_expert_tensor_parallel_group()` | `expt_tp` |
| `get_expert_tensor_and_model_parallel_group()` | `tp_ep` |
| `get_expert_tensor_model_pipeline_parallel_group()` | `tp_ep_pp`; use `tp_ep_pp_with_egtp_remat` when `with_egtp_remat=True` |
| `get_data_parallel_group()` | `dp_gtp_remat`; use `dp` when `with_gtp_remat=False` |
| `get_data_parallel_group(with_context_parallel=True)` | `dp_cp_gtp_remat`; use `dp_cp` when `with_gtp_remat=False` |
| `get_expert_data_parallel_group()` | `expt_dp_gtp_remat`; use `expt_dp` when `with_gtp_remat=False` |

These data-parallel mappings assume the partial-group flag is false. For partial groups,
`intra_dp_cp` and `intra_expt_dp` exclude GTP-remat; the collection does not cover every flag
combination. Select a matching explicit group when there is no corresponding field. Preserve
all accessor flags and the communication purpose when migrating a call.

A source or peer rank is a **global** rank, while `pg.rank()` is local to its group.
For example, the first global rank in `pg` is `torch.distributed.get_global_rank(pg, 0)`.
The data-parallel source-rank accessor uses the replicate `dp` / `dp_cp` group, even though the
data-parallel group accessor defaults to the GTP-remat-inclusive group.

The replacement assumes a valid group for a participating rank. Some legacy accessors also
support cached rank/size overrides, uninitialized distributed state, or absent optional groups.
Preserve the caller's intended behavior explicitly in those cases; an unconditional `.rank()`
or `.size()` is not equivalent. In particular, passing `None` to PyTorch distributed APIs can
select the default world group.

## `use_mpu_process_groups()` is a compatibility shim

`ProcessGroupCollection.use_mpu_process_groups()` builds a collection **from the same global
state**. Calling it inside a model component preserves that component's global dependency.
New components should accept the owning model's collection or an explicit group from the caller.
The shim remains useful at bootstrap boundaries and in explicitly commented migration fallbacks
for existing callers.

## What this means for your change

**Writing a new feature?** Accept a `ProcessGroupCollection` or an explicit
`torch.distributed.ProcessGroup` and pass it through. Do not introduce a `None` default that
falls back to `parallel_state` for new features.

```python
# The caller supplies the owning model's tensor-parallel group.
def my_op(x, tp_group: torch.distributed.ProcessGroup):
    ...
```

**Fixing a bug?** Existing `parallel_state` calls may remain. Keep unrelated process-group
plumbing changes in their own PR so the fix can be reviewed and reverted independently.

**Reviewing?** Flag new tier-1/tier-2 accessor use in `megatron/core`. Allowed exceptions:
`parallel_state.py` itself, `process_groups_config.py`, bootstrap code that materializes a
collection from the globals, tests, and explicitly commented migration fallbacks.

**Using Megatron Core with your own parallel grid?** Build a `ProcessGroupCollection` for that
grid and pass it in. See
[Building a `ProcessGroupCollection` for your own grid](process-group-collection.md).

This guidance applies to `megatron/core`. It does not apply to `megatron/training` or other
training-loop code unless a change explicitly opts in.

## The CI check

The "Process group usage check" CI job runs `tools/check_process_group_usage.py`. It fails a PR
that adds a read of the global grid anywhere in `megatron/core`.

Existing reads are listed in `tools/process_group_usage_allowlist.json`. An entry records the
file, the enclosing function or class, and the accessor, but not the line number, so code can move
within its function without changing the allowlist. Each call has its own entry: a second
identical call in the same function is a new read.

The check counts calls to:

- tier-1 and tier-2 accessors: `parallel_state` functions whose names start with `get_` and end
  in `_group`, `_groups`, `_gloo`, `_rank`, `_ranks`, or `_world_size`. It follows calls made
  through `parallel_state`, `mpu`, an `import ... as` alias, or a direct
  `from megatron.core.parallel_state import ...`.
- `ProcessGroupCollection.use_mpu_process_groups()`.

It does not count:

- functions that are not deprecated, such as `initialize_model_parallel`,
  `destroy_model_parallel`, `is_initialized`, and `get_nccl_options` (it also skips
  `get_all_ranks` by name);
- tier-3 predicates and tier-4 state, such as `is_pipeline_first_stage()`, the virtual-pipeline
  rank and size, and `get_global_memory_buffer()`;
- code in `parallel_state.py` and `process_groups_config.py`, or outside `megatron/core`;
- reads through an assigned alias (`grid = parallel_state`) or a dynamic lookup
  (`getattr(parallel_state, name)`).

The check is syntactic. Reviewers still look for reads it cannot see.

### When the check fails

The job lists each new read with its location, the enclosing scope, and the accessor, for
example `megatron/core/foo.py:12  my_func:accessor:parallel_state.get_tensor_model_parallel_group`.

- **New code:** take the group from the caller and pass it through. Switching to
  `use_mpu_process_groups()` does not help; the check counts the shim too.
- **Moved code:** a read that moves to another function or file, or whose function is renamed,
  is reported as new. If the read cannot be removed in the same PR, edit the allowlist by hand:
  replace the old entry with the new one. Point it out in the PR description so that a reviewer
  approves the entry. `--update` cannot do this; it refuses to run while there is a new read.

### When you remove reads

The check also fails when an allowlisted read no longer exists. Refresh the allowlist and commit
it with your change:

```bash
python tools/check_process_group_usage.py --update
```

`--update` only removes stale entries. It never adds one.

### Run it locally

The checker needs only the Python standard library:

```bash
python tools/check_process_group_usage.py           # the CI check
python tools/check_process_group_usage.py --stats   # count reads without checking the allowlist
python tests/unit_tests/test_check_process_group_usage.py   # the checker's own tests
```

## Where this is going

The goal is that code in `megatron/core` receives its process groups from the caller, and
`parallel_state` only sets up and tears down the standard grid. Removing the `parallel_state`
module itself is not planned.

The steps are:

1. **Migrate the readers.** Components in `megatron/core` take their groups from the caller. The
   CI check keeps new reads out while this happens.
2. **Add a builder.** A function built on `HyperCommGrid` returns the `ProcessGroupCollection` for
   the standard grid without setting any globals.
3. **Make `initialize_model_parallel` a thin wrapper.** It calls the builder and registers the
   result in `parallel_state`, so existing callers keep working.
4. **Move the callers.** Training loops, Megatron Bridge, examples, and tests build or receive a
   collection and pass it in.
5. **Retire the global registration.** The module-level groups and their `get_*()` accessors
   are deprecated, then removed.

Steps 1 to 4 do not change how `initialize_model_parallel`, `destroy_model_parallel`, and
`is_initialized` behave for their callers. Step 5 is not part of this migration; it is a later
change that will be announced separately.

A component that stops reading the globals may keep a fallback for existing callers that do not
pass a collection yet. Such a fallback emits a deprecation warning for at least one release and is
then removed. After that, the component requires the collection or group from its caller.

For current status, see the tracking issue
[#6307](https://github.com/NVIDIA/Megatron-LM/issues/6307).
