# Deprecating `parallel_state`

`megatron.core.parallel_state` holds the process groups of a **single, global** parallel grid.
Megatron Core components take their process groups from the `ProcessGroupCollection` that their
caller passes instead. The functions that read the global grid are deprecated in Megatron Core
0.21 and removed in 0.23. The module itself stays: it keeps the functions that build, register
and tear down the standard grid.

This page describes what is deprecated, how the warnings work, how a framework built on Megatron
Core migrates, and how the repository keeps new reads of the global grid out.

## Why it matters

A global accessor can return the intended group when the model uses the global parallel grid.
It cannot select a different model's grid. For example, an encoder and a language model with
independent parallelism can need different groups on the same rank. Reading globals in either
model can select unrelated ranks or fail if those globals were never initialized. Incorrect
collectives or rank-dependent RNG offsets can then produce incorrect results or hang.

The caller knows which model owns an operation and should supply its groups explicitly.

[Process-group failure patterns](process-group-failure-patterns.md) shows the ways this goes wrong
in practice, and how to write each case instead.

## What is deprecated

Called from outside Megatron Core, each of these functions emits a
`ParallelStateDeprecationWarning` that names its replacement. They are deprecated in 0.21 and
removed in 0.23, under the N+2 policy in
[API backward compatibility checking](../api-backwards-compatibility-check.md).

| Deprecated | Replacement |
|---|---|
| Group accessors: `get_*_group()`, `get_*_groups()`, the Gloo groups `get_*_group_gloo()`, and the dynamic context-parallel group maps | The matching field of the model's collection; see [Select the matching group](#select-the-matching-group) |
| Rank and size accessors: `get_*_rank()` and `get_*_world_size()` | `rank()` and `size()` of that field, or `get_pg_rank()` and `get_pg_size()` from `megatron.core.utils` for a group that can be `None` |
| Global-rank accessors: `get_*_src_rank()`, `get_*_global_ranks()`, `get_pipeline_model_parallel_first_rank()` and its `last`, `next` and `prev` siblings, and `get_all_ranks()` | `torch.distributed.get_global_rank(group, 0)`, `torch.distributed.get_process_group_ranks(group)`, and `get_pp_first_rank(pp_group)` and its siblings in `megatron.core.pipeline_parallel.utils` |
| Stage predicates: `is_pipeline_first_stage()`, `is_pipeline_last_stage()`, `is_rank_in_embedding_group()` and `is_rank_in_position_embedding_group()` | `is_pp_first_stage(pp_group)`, `is_pp_last_stage(pp_group)`, `is_vp_first_stage(vp_stage, vp_size)` and `is_vp_last_stage(vp_stage, vp_size)` in `megatron.core.pipeline_parallel.utils`; membership in `pg_collection.embd` or `pg_collection.pos_embd` |
| Virtual-pipeline rank and size, and their setters | The `vp_stage` of the model chunk, passed explicitly, and `virtual_pipeline_model_parallel_size` in the model's configuration |
| Rank and size overrides, which set the `_MPU_*` module variables: `set_tensor_model_parallel_rank()`, `set_expert_model_parallel_world_size()` and the other `set_*_rank()` and `set_*_world_size()` functions | A collection whose groups have that rank and size, passed to the code that needs it or registered with `register_process_groups()` |
| Memory-buffer aliases: `get_global_memory_buffer()`, `destroy_global_memory_buffer()` and `_set_global_memory_buffer()` | The functions of the same name in `megatron.core.utils`, which create the buffer on first use |
| `ProcessGroupCollection.use_mpu_process_groups()` | The collection that the caller passes, or `parallel_state.get_registered_process_groups()`; see [The registry](#the-registry) |

Bootstrap, teardown and the registry are not deprecated: `initialize_model_parallel()`,
`build_process_group_collection()`, `destroy_model_parallel()`, `is_initialized()`,
`model_parallel_is_initialized()`, `register_process_groups()` and
`get_registered_process_groups()`. Neither are the helpers that create groups or configure
their communicators, such as `create_group()`, `create_all_gather_groups()` and
`get_nccl_options()`.

### How the warning behaves

- `ParallelStateDeprecationWarning` is a `FutureWarning`, so Python shows it by default.
- Each call site warns once per process: a deprecated function called from a given file and line.
  Under an `error` filter, every call raises.
- Calls from code under `megatron/core` do not warn. Megatron Core's compatibility fallbacks emit
  a `ProcessGroupFallbackWarning` for the caller that omitted its groups instead (see
  [Fallbacks](#fallbacks-processgroupfallbackwarning)), and its remaining reads are counted by the
  [checker](#the-ci-check).
- Calls in the block of `global_process_group_fallback()` do not warn: that block has already
  emitted a `ProcessGroupFallbackWarning` for the fallback they belong to.
- Calls that `torch.compile` traces do not warn, so they do not break the graph.
- The test suite ignores the warning through `filterwarnings` in `pyproject.toml`, because tests
  set up and read the global grid on purpose.

## Select the matching group

Collection field names are not always a direct abbreviation of the accessor name. In particular,
expert tensor parallelism uses `expt_tp`, which may differ from the dense `tp` group.

| Accessor | Field in the caller's collection |
|---|---|
| `get_tensor_model_parallel_group()` | `tp` |
| `get_pipeline_model_parallel_group()` | `pp` |
| `get_model_parallel_group()` | `mp` |
| `get_context_parallel_group()` | `cp` |
| `get_tensor_and_context_parallel_group()` | `tp_cp` |
| `get_hierarchical_context_parallel_groups()` | `hcp` |
| `get_embedding_group()`, `get_position_embedding_group()` | `embd`, `pos_embd` |
| `get_gtp_weight_remat_group()`, `get_expert_gtp_weight_remat_group()` | `gtp_remat`, `expt_gtp_remat` |
| `get_expert_model_parallel_group()` | `ep` |
| `get_expert_tensor_parallel_group()` | `expt_tp` |
| `get_expert_tensor_and_model_parallel_group()` | `tp_ep` |
| `get_expert_tensor_model_pipeline_parallel_group()` | `tp_ep_pp`; `tp_ep_pp_with_egtp_remat` when `with_egtp_remat=True` |
| `get_data_parallel_group()` | `dp_gtp_remat`; `dp` when `with_gtp_remat=False` |
| `get_data_parallel_group(with_context_parallel=True)` | `dp_cp_gtp_remat`; `dp_cp` when `with_gtp_remat=False` |
| `get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True, with_gtp_remat=False)` | `intra_dp_cp` |
| `get_expert_data_parallel_group()` | `expt_dp_gtp_remat`; `expt_dp` when `with_gtp_remat=False` |
| `get_expert_data_parallel_group(partial_expert_data_parallel=True, with_gtp_remat=False)` | `intra_expt_dp` |
| `get_data_parallel_group_gloo(with_context_parallel=True, partial_data_parallel=True)` | `intra_dp_cp_gloo` |
| `get_expert_data_parallel_group_gloo(partial_expert_data_parallel=True)` | `intra_expt_dp_gloo` |
| `get_tensor_and_data_parallel_group(with_context_parallel=True)` | `tp_dp_cp` |
| `get_amax_reduction_group(with_context_parallel=True, tp_only_amax_red=...)` | `amax_reduction_group(pg_collection, tp_only_amax_red)` from `megatron.core.process_groups_config` |
| `get_dynamic_data_context_parallel_group_map()`, `get_dynamic_data_context_parallel_groups(group_size=n)` | `dynamic_cp`, `dynamic_cp[n]` |
| `get_dynamic_tensor_data_context_parallel_group_map()`, `get_dynamic_tensor_data_context_parallel_group(group_size=n)` | `dynamic_tp_cp`, `dynamic_tp_cp[n]` |
| `get_intra_distributed_optimizer_instance_group()`, `get_inter_distributed_optimizer_instance_group()` | `intra_dist_opt`, `inter_dist_opt` |

The collection does not cover every flag combination. The other Gloo groups, the partial
data-parallel groups that span the GTP-remat axis, and the tensor × data-parallel group without
context parallelism have no field. `build_process_group_collection()` returns them in its
second result, a `StandardGridExtras`; otherwise pass such a group explicitly. Preserve all
accessor flags and the communication purpose when migrating a call.

A source or peer rank is a **global** rank, while `pg.rank()` is local to its group.
For example, the first global rank in `pg` is `torch.distributed.get_global_rank(pg, 0)`.
The data-parallel source-rank accessor uses the replicate `dp` / `dp_cp` group, even though the
data-parallel group accessor defaults to the GTP-remat-inclusive group.

The replacement assumes a valid group for a participating rank. Some legacy accessors also
support cached rank/size overrides, uninitialized distributed state, or absent optional groups.
Preserve the caller's intended behavior explicitly in those cases; an unconditional `.rank()`
or `.size()` is not equivalent. `get_pg_rank(None)` is 0 and `get_pg_size(None)` is 1, while
passing `None` to a `torch.distributed` API selects the default (world) group.

## Fallbacks: `ProcessGroupFallbackWarning`

Many Megatron Core components still accept a call without their process groups, and then fall
back to the global grid. Each such fallback emits a `ProcessGroupFallbackWarning` from
`megatron.core.process_groups_config`:

- It is a `FutureWarning`, separate from `ParallelStateDeprecationWarning`.
- It names the component and the omitted argument, and says that the fallback is deprecated in
  0.21 and removed in 0.23. Once it is removed, the argument is required.
- It warns once per component and argument in a process, on every rank that reaches it. It points
  at the first caller outside Megatron Core and `torch`, which is the code that omitted the
  argument.
- If the global grid is not initialized, a separate warning names the groups that came back
  `None`; `torch.distributed` treats them as the default (world) group.

Inside Megatron Core, a fallback goes through `resolve_process_groups()`, which warns before it
reads the grid and checks that the fields the component reads are set:

```python
from megatron.core.process_groups_config import resolve_process_groups

pg_collection = resolve_process_groups(
    pg_collection,
    owner="MyLayer",
    required=("tp", "pp"),
    deprecated_in="0.21",
    removed_in="0.23",
)
```

New code does not add a fallback: it requires its groups. The checker counts every call of
`resolve_process_groups()` in Megatron Core as a read of the global grid.

Code outside Megatron Core that must keep a fallback for its own callers during the deprecation
period reads the grid in a `global_process_group_fallback()` block. The block emits one
`ProcessGroupFallbackWarning`, and the deprecated functions it calls do not also emit a
`ParallelStateDeprecationWarning`:

```python
from megatron.core import parallel_state
from megatron.core.process_groups_config import global_process_group_fallback

if tp_group is None:
    with global_process_group_fallback(
        "my_function", "tp_group", deprecated_in="0.21", removed_in="0.23"
    ):
        tp_group = parallel_state.get_tensor_model_parallel_group()
```

Keep the block to the reads of the fallback: no deprecated function called anywhere in the block
warns, at any depth.

## The registry

`initialize_model_parallel()` builds the standard grid with `build_process_group_collection()`,
registers the result with `register_process_groups()`, and returns the collection.
`get_registered_process_groups()` returns that collection, or `None` if none is registered.
`destroy_model_parallel()` clears the registration.

A framework that builds its own grid, for example with `HyperCommGrid`, registers its collection
so that code which still reads `parallel_state` uses that grid:

```python
from megatron.core import parallel_state

parallel_state.register_process_groups(pg_collection)
```

- One collection can be registered per process. A second call raises `RuntimeError` until
  `destroy_model_parallel()` runs.
- The groups are registered when the function is called. Setting a field of the collection
  afterwards does not change what the accessors return.
- A field that is `None` or not set registers no group; its accessor then reports the group as
  not initialized.
- The groups and global rank lists that the collection has no field for come from the `extras`
  argument, which `build_process_group_collection()` returns. Without it, the rank lists are
  derived from the collection's groups, and the groups that a collection cannot describe, such as
  the full data-parallel Gloo groups and the tensor × data-parallel group, are not registered.

`get_registered_process_groups()` returns the registered collection itself, not a copy, and
every reader shares it. To add fields, such as the all-gather groups `dp_cp_ag` and
`expt_dp_ag`, set them on a `copy.copy()` of it. `copy.deepcopy()` returns the same collection,
because a collection holds process-group handles, and `dataclasses.replace()` returns one
without groups.

```python
import copy

from megatron.core import parallel_state

pg_collection = copy.copy(parallel_state.get_registered_process_groups())
pg_collection.dp_cp_ag = dp_cp_ag_group
```

## Strict mode

Strict mode turns every `ProcessGroupFallbackWarning` into an error, on the rank that reaches the
fallback:

- **Training:** `--strict-process-group-fallbacks`. It applies from before distributed
  initialization and model construction.
- **Other frameworks:** call `enable_strict_process_group_fallbacks()` from
  `megatron.core.process_groups_config` after their own warning setup. A filter installed
  later that also matches the class, such as `warnings.simplefilter("ignore")`, takes precedence
  again.
- **Unit tests:** `pytest -W error::megatron.core.process_groups_config.ProcessGroupFallbackWarning`,
  or the same value in a `filterwarnings` marker or ini option. `python -W` and `PYTHONWARNINGS`
  cannot name the class: Python resolves them before Megatron Core can be imported.

Each fallback warns once per process. A fallback that already warned, for example under
`pytest.warns` in an earlier test, stays silent under a later error filter.
`enable_strict_process_group_fallbacks()` forgets the fallbacks that already warned, so they
raise again.

Strict mode leaves `ParallelStateDeprecationWarning` a warning, by design: a run can be strict
about Megatron Core's fallbacks while its own code still calls the accessors. To find those
calls, turn the accessor warning into an error as well:

```bash
pytest -W error::megatron.core.parallel_state.ParallelStateDeprecationWarning tests/my_tests
```

A command-line `-W` takes precedence over the `filterwarnings` entry of `pyproject.toml` that
ignores the warning. In code, use a filter:

```python
import warnings

from megatron.core.parallel_state import ParallelStateDeprecationWarning

warnings.filterwarnings("error", category=ParallelStateDeprecationWarning)
```

## `use_mpu_process_groups()` is a compatibility shim

`ProcessGroupCollection.use_mpu_process_groups()` builds a **new** collection from the
deprecated accessors, so it carries the same global dependency. Called from outside Megatron
Core it emits a `ParallelStateDeprecationWarning`.

- It reads the groups without checking that the grid is initialized. On an uninitialized grid
  most fields come back `None`, which `torch.distributed` treats as the default (world) group,
  and the accessors that need a grid, such as the one for `dp`, raise.
- It builds every field unless `required_pgs` names a subset. `dp_cp_ag` and `expt_dp_ag` are
  always `None`.
- Replace it with the collection that the caller passes. Where there is no caller to pass one,
  such as an entry point, use the collection that `initialize_model_parallel()` returned or
  `get_registered_process_groups()`.

## Migrating a framework

A framework that trains or serves models with Megatron Core, such as Megatron-Bridge or
NeMo-RL, migrates in these steps. Run its tests with strict mode and with
`ParallelStateDeprecationWarning` as an error to find what is left.

**1. Hold one collection per grid.** Keep the collection that `initialize_model_parallel()`
returns:

```python
from megatron.core import parallel_state

pg_collection = parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2
)
```

Or build one for your own grid, as
[Building a `ProcessGroupCollection` for your own grid](process-group-collection.md) shows with
`HyperCommGrid`, and register it, so that code you have not migrated yet reads the same grid:

```python
from megatron.core import parallel_state

# build_dense_pg_collection is the HyperCommGrid example of the page linked above.
pg_collection = build_dense_pg_collection(tp=2, pp=2, dp=2)
parallel_state.register_process_groups(pg_collection)
```

A process registers one collection. A job with several grids passes each model its own
collection.

**2. Pass the collection to every Megatron Core API that takes one.** The model, RNG seeding,
DDP, the optimizer, the schedule and gradient finalization each fall back to the global grid
when they do not get it:

```python
from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

config.finalize_model_grads_func = finalize_model_grads  # called with the collection
model_parallel_cuda_manual_seed(1234, pg_collection=pg_collection)
model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=vocab_size,
    max_sequence_length=seq_length,
    pre_process=is_pp_first_stage(pg_collection.pp),
    post_process=is_pp_last_stage(pg_collection.pp),
    pg_collection=pg_collection,
)
model = DistributedDataParallel(config, ddp_config, model, pg_collection=pg_collection)
optimizer = get_megatron_optimizer(optimizer_config, [model], pg_collection=pg_collection)
forward_backward_func = get_forward_backward_func(pp_size=pg_collection.pp.size(), vp_size=None)
losses = forward_backward_func(
    forward_step_func=forward_step,
    data_iterator=data_iterator,
    model=[model],
    num_microbatches=num_microbatches,
    seq_length=seq_length,
    micro_batch_size=micro_batch_size,
    p2p_communicator=P2PCommunicator(pp_group=pg_collection.pp, config=config),
    pg_collection=pg_collection,
)
```

[Passing the collection](process-group-collection.md#passing-the-collection) lists more details.

For inference, the dynamic engine takes the groups of the model it serves. Pass the collection
to `megatron.inference.utils.get_dynamic_inference_engine()`, which builds and loads the model
when it is not given one:

```python
from megatron.core import parallel_state
from megatron.inference.utils import get_dynamic_inference_engine

engine = get_dynamic_inference_engine(
    pg_collection=parallel_state.get_registered_process_groups()
)
```

A framework that builds the engine itself with
`megatron.core.inference.engine_factory.build_dynamic_inference_engine()` sets
`InferenceConfig(pg_collection=...)`. Left unset, inference uses the collection that the model
was built with, when it sets every group that inference reads (`tp`, `pp`, `dp`, `mp`, `ep`,
`cp`, `expt_tp` and `expt_dp`), and otherwise falls back to the global grid with a warning.

**3. Read sizes, ranks and stages from the collection.** Replace each accessor with the field
from [Select the matching group](#select-the-matching-group):

```python
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.utils import get_pg_rank, get_pg_size

tp_size = get_pg_size(pg_collection.tp)  # was get_tensor_model_parallel_world_size()
dp_rank = get_pg_rank(pg_collection.dp_gtp_remat)  # was get_data_parallel_rank()
if is_pp_last_stage(pg_collection.pp):  # was is_pipeline_last_stage()
    loss = reduce_loss(output)
```

**4. Stop using `use_mpu_process_groups()`.** Use the collection from step 1, or
`get_registered_process_groups()` where nothing passes one. Copy it with `copy.copy()` before
adding fields.

**5. Remove your own fallbacks to the global grid.** Make the groups a required argument. A
fallback that must stay during the deprecation period reads the grid in a
`global_process_group_fallback()` block.

## What this means for your change

**Writing a new feature in Megatron Core?** Accept a `ProcessGroupCollection` or an explicit
`torch.distributed.ProcessGroup` and pass it through. Do not add a `None` default that falls
back to `parallel_state`.

```python
# The caller supplies the owning model's tensor-parallel group.
def my_op(x, tp_group: torch.distributed.ProcessGroup):
    ...
```

**Fixing a bug?** Existing `parallel_state` calls may remain. Keep unrelated process-group
plumbing changes in their own PR so the fix can be reviewed and reverted independently.

**Changing code outside Megatron Core in this repository?** Take the groups from the model, the
caller or the registered collection. A fallback reads the grid in a
`global_process_group_fallback()` block, and bootstrap code that reads the grid on purpose
ignores `ParallelStateDeprecationWarning` in a `warnings.catch_warnings()` block, so a stock run
emits no `ParallelStateDeprecationWarning`.

**Reviewing?** Flag new calls of the deprecated functions and of `use_mpu_process_groups()` in
`megatron/core`. Allowed exceptions: `parallel_state.py` itself, `process_groups_config.py`,
bootstrap code that materializes a collection from the globals, tests, and warned compatibility
fallbacks.

**Using Megatron Core with your own parallel grid?** Build a `ProcessGroupCollection` for that
grid and pass it in. See
[Building a `ProcessGroupCollection` for your own grid](process-group-collection.md).

## The CI check

Two checks keep new reads of the global grid out of the repository. CI runs both.

`tools/check_process_group_usage.py` is a ratchet over Megatron Core. It parses the files under
`megatron/core`, except `parallel_state.py` and `process_groups_config.py`, and counts calls of:

- the deprecated `parallel_state` functions: the group, rank and size accessors (kind
  `accessor`), and the stage predicates, virtual-pipeline state, rank and size overrides,
  dynamic context-parallel group maps, `get_all_ranks()` and memory-buffer aliases (kind
  `function`);
- `ProcessGroupCollection.use_mpu_process_groups()` (kind `shim`);
- `resolve_process_groups()`, the warned fallback (kind `guarded`).

`tools/process_group_usage_allowlist.json` records the existing calls. An entry records the
file, the enclosing function or class, and the function called, but not the line number, so code
can move within its function without changing the allowlist. Each call has its own entry: a second
identical call in the same function is a new read. The checker exits with an error when it finds
a call that the allowlist does not hold, or an entry whose call is gone.

The check is syntactic. It follows calls made through `parallel_state`, `mpu`, an
`import ... as` alias, or a direct `from megatron.core.parallel_state import ...`, but not
through an assigned alias (`grid = parallel_state`) or a dynamic lookup
(`getattr(parallel_state, name)`). Reviewers still look for reads it cannot see.

`tests/unit_tests/test_parallel_state_accessor_uses.py` guards the code outside Megatron Core:
the non-test code under `megatron/`, `examples/`, `tools/` and `tasks/`. It allows a call of a
deprecated function or of `use_mpu_process_groups()` only in a
`global_process_group_fallback()` block or in a bootstrap block that ignores
`ParallelStateDeprecationWarning`.

### When the check reports a new read

The checker lists each new read with its location, the enclosing scope, the kind and the
function, for example
`megatron/core/foo.py:12  my_func:accessor:parallel_state.get_tensor_model_parallel_group`.

- **New code:** take the group from the caller and pass it through. Switching to
  `use_mpu_process_groups()` or `resolve_process_groups()` does not help; the checker counts
  both.
- **Moved code:** a read that moves to another function or file, or whose function is renamed,
  is reported as new. If the read cannot be removed in the same PR, edit the allowlist by hand:
  replace the old entry with the new one. Point it out in the PR description so that a reviewer
  approves the entry. `--update` cannot do this; it refuses to run while there is a new read.

### When you remove reads

The checker also reports allowlisted reads that no longer exist. Refresh the allowlist and commit
it with your change:

```bash
python tools/check_process_group_usage.py --update
```

`--update` removes stale entries. When a `resolve_process_groups()` call replaces a removed
direct read in the same file and scope, it swaps the entry. It never adds any other entry.

### Run it locally

The checker and both tests need only the Python standard library:

```bash
python tools/check_process_group_usage.py           # the ratchet
python tools/check_process_group_usage.py --stats   # count reads by kind
python tests/unit_tests/test_check_process_group_usage.py   # the checker's own tests
python tests/unit_tests/test_parallel_state_accessor_uses.py   # the guard outside megatron/core
```

## Removal schedule

| Release | What happens |
|---|---|
| 0.21 | The deprecated functions and `use_mpu_process_groups()` warn when called from outside Megatron Core. The fallbacks of Megatron Core components warn with `ProcessGroupFallbackWarning`. Both warnings name 0.23. |
| 0.22 | Both warnings stay, the second release in which callers can migrate. |
| 0.23 | The deprecated functions and `use_mpu_process_groups()` are removed, and so are the fallbacks: the process-group arguments that warned become required. |

After 0.23, `parallel_state` keeps bootstrap (`initialize_model_parallel()` and
`build_process_group_collection()`), teardown (`destroy_model_parallel()`), the
initialization checks, and the registry (`register_process_groups()` and
`get_registered_process_groups()`). Code that needs process groups receives the owning model's
collection from its caller.

For current status, see the tracking issue
[#6307](https://github.com/NVIDIA/Megatron-LM/issues/6307).
