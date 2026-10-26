# Process-group failure patterns

Code that reads the global parallel grid usually works when every model in the job uses that
grid. It goes wrong when a component runs on a grid of its own, for example:

- an encoder and a language model with different parallelism;
- a separate inference layout;
- a second model in the same job.

This page lists the patterns behind those failures. For each one it shows what the pattern looks
like, what goes wrong, and how to write it instead.

For why the global grid is being deprecated, see
[Deprecating `parallel_state`](parallel-state-deprecation.md). For how to build a collection for
your own grid, see
[Building a `ProcessGroupCollection` for your own grid](process-group-collection.md).

## Two facts behind most failures

**An unset field reads as `None`.** `ProcessGroupCollection` returns `None` for any field that was
never set:

```python
pgs = ProcessGroupCollection(tp=tp_group)
hasattr(pgs, "pp")  # True
pgs.pp              # None
```

A missing field therefore looks the same as a field set to `None`, and a `hasattr` check never
fails. `vars(pgs)` contains only the fields that were set.

**`None` means different things to different callees:**

- Megatron Core helpers treat `None` as a group of one rank. `get_pg_size(None)` is 1,
  `get_pg_rank(None)` is 0, and `is_pp_first_stage(None)` and `is_pp_last_stage(None)` are both
  true.
- `torch.distributed` treats `group=None` as the default group, which contains every rank in the
  job.

A component that runs on its own grid then fails in one of two ways:

- **Another grid is initialized globally.** The global read succeeds and returns that grid's ranks.
  Nothing raises; the results are wrong, or the job hangs.
- **No global grid exists.** The read raises, or falls through to the default group.

[Testing on a grid of its own](#testing-on-a-grid-of-its-own) shows how to catch both.

## 1. A silent default collection

```python
class MyLayer(MegatronModule):
    def __init__(self, config, pg_collection=None):
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.tp_group = pg_collection.tp
```

**What goes wrong.** A caller that forgets to pass its collection gets the global grid's groups,
with no signal. If the global grid belongs to another model, the layer runs on that model's ranks.
With no global grid, `use_mpu_process_groups()` raises even for groups the layer never uses,
because by default it builds every field.

**Write instead.** Take the collection from the caller. A fallback for existing callers may stay
while they migrate. It must do four things:

- carry a comment that marks it as a fallback;
- warn, naming the component and the release that removes the fallback;
- build only the fields the component reads;
- check that those fields are set.

```python
        if pg_collection is None:
            # Compatibility fallback for callers that do not pass a collection yet.
            warnings.warn(
                f"{type(self).__name__} was built without a pg_collection and uses the global "
                "parallel grid. Pass the owning model's ProcessGroupCollection.",
                FutureWarning,
                stacklevel=2,
            )
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
```

A hand-built collection that copies `parallel_state` accessors into the fields one by one is the
same pattern, and needs the same treatment.

## 2. A missing group that looks present

```python
        pp_group = pg_collection.pp if hasattr(pg_collection, "pp") else None  # always true
        pp_rank = get_pg_rank(pp_group)  # an unset pp gives rank 0 on every stage
```

```python
        assert hasattr(pg_collection, "dp_cp")  # never fails
        torch.distributed.all_reduce(num_tokens, group=pg_collection.dp_cp)  # unset: every rank
```

**What goes wrong:**

- A collection without `pp` makes every pipeline stage build the first stage's layers.
- A reduction over an unset group runs over every rank in the job. For example, take a token count
  that should be summed over the data-parallel ranks. On a TP=2 × PP=2 grid with one data-parallel
  replica it is summed over 4 ranks instead, so the per-token loss and its gradients come out 4×
  too small. A job with several grids hangs instead, because ranks outside the model never join the
  collective.
- An unset tensor-parallel group in inference makes `get_world_size(None)` return the size of the
  whole job. An unset data-parallel group gives tensor-parallel peers different sampling seeds.

**Write instead.** Check the fields once, when the component is built. Use `vars()`, so that an
absent field raises and an explicit `None` keeps its meaning of "this axis is not used". Raise
`ValueError` rather than asserting, because `python -O` removes asserts.

```python
        groups = vars(pg_collection)
        missing = [name for name in ("tp", "pp") if name not in groups]
        if missing:
            raise ValueError(f"{type(self).__name__} needs {missing} in its pg_collection")
        self.tp_group = groups["tp"]
        self.pp_group = groups["pp"]  # None means no pipeline parallelism
```

## 3. A collective without the owner's group

```python
        fc1_input = copy_to_tensor_model_parallel_region(hidden_states)  # no group
        ...
        output = reduce_from_tensor_model_parallel_region(fc2_output)  # no group
```

The tensor-parallel mappings fall back to the global tensor-parallel group when `group` is `None`.
They warn only on global rank 0, and only with a `DeprecationWarning`, which Python hides by
default.

**What goes wrong.** The weights are sharded over the module's tensor-parallel group, but the
gather or reduction runs over the global one. Depending on the layouts, this sums unrelated
replicas, gathers the wrong number of shards (a shape error), or leaves each rank with another
rank's sequence chunk.

**Write instead.** Pass the group the module owns to every mapping and collective:

```python
        fc1_input = copy_to_tensor_model_parallel_region(hidden_states, group=self.tp_group)
        ...
        output = reduce_from_tensor_model_parallel_region(fc2_output, group=self.tp_group)
```

## 4. A global size or rank used in arithmetic

```python
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        ep_size = parallel_state.get_expert_model_parallel_world_size()
```

**What goes wrong.** The number is right for the global grid and wrong for the component. Sizes
and ranks feed shard shapes, padding multiples, RNG seed offsets, summation orders, and the choice
of pipeline schedule:

- A fused layer that checks the global size can take a single-rank path on a sharded grid.
- An inference context pads to the global grid's multiple.
- A pipeline-parallel model can be given the schedule for no pipelining.

**Write instead.** Take the size or rank from the group or communicator the component already
holds:

```python
        tp_size = get_pg_size(self.tp_group)
        ep_size = get_pg_size(self.ep_group)
```

## 5. A pipeline rank that is not passed on

```python
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage  # pp_rank is not passed
        )
```

**What goes wrong.** Without `pp_rank`, the layout uses the global pipeline rank. Under a global
grid with one pipeline stage, a model on its own two-stage grid builds the first stage's layers,
with the first stage's layer numbers, on every rank. Its checkpoint keys and its multi-token
prediction placement are wrong too. A builder that holds the collection but does not pass the rank
to the spec functions has the same effect.

**Write instead.** Pass the rank from the owner's pipeline group through every layer of the call
chain:

```python
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=get_pg_rank(pg_collection.pp)
        )
```

## 6. Checkpoint metadata from the global grid

```python
    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)  # global DP x CP group
        ...
```

**What goes wrong.** Replica ids decide which rank writes each shard. When the caller passes no
metadata, which is common in tools, converters, exports and tests, the ids are computed over the
global data-parallel ranks. On another grid, several ranks or no rank write a shard. Expert
offsets computed from the global expert-parallel rank and size save optimizer state under another
expert's name.

**Write instead.** Fill the metadata from the owner's collection, with the same group that the
global default selects (the GTP-remat-inclusive DP × CP group):

```python
        metadata = dict(metadata or {})
        metadata.setdefault(
            "dp_cp_group", self.pg_collection.dp_cp_gtp_remat or self.pg_collection.dp_cp
        )
```

Pass expert ranks and sizes from the owner's expert groups in the same way.

## 7. A collective over every rank

```python
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)  # no group
```

The same happens when the group argument is a field that was never set, as described in
[pattern 2](#2-a-missing-group-that-looks-present).

**What goes wrong.** The collective includes every rank in the job. If some ranks never reach this
code, for example encoder-only ranks or ranks serving rollouts, the job hangs. If they all reach it,
statistics from different models are mixed. Reductions of FP8 scaling statistics, gradient norms
and agreement flags are common cases.

**Write instead.** Name the group whose ranks all run the code. If a collective really must span
the whole job, say so in a comment, so that reviewers can tell it apart from a missing group.

## 8. Changing shared state

```python
        if getattr(pg_collection, "dp", None) is None:
            # Writes into the caller's collection.
            pg_collection.dp = parallel_state.get_data_parallel_group()
```

```python
        if tp_group is None:
            tp_group = torch.distributed.new_group(ranks=[torch.distributed.get_rank()])  # per rank
```

**What goes wrong:**

- Writing into a collection changes it for every later reader. A collection is usually shared by
  every component of a model.
- `torch.distributed.new_group` is collective. Every rank must call it with the same ranks in the
  same order. Creating a different group on each rank desynchronizes group creation, and can hang.
- State that only `initialize_model_parallel` creates, such as the global memory buffer, is missing
  on a grid that was built without it.
- Teardown that does not destroy every group it created leaks communicators on each
  initialize/destroy cycle.

**Write instead:**

- Keep derived groups in local variables, or raise when a required field is missing.
- Create groups collectively. For one group per rank, use
  `torch.distributed.new_subgroups_by_enumeration([[r] for r in range(world_size)])`.
- Create shared buffers on first use instead of at grid initialization.
- Track every group that initialization creates, and destroy each one at teardown.

## 9. A training loop that gives the collection only to the model

```python
    model = model_provider(pg_collection=pgs)
    model = DistributedDataParallel(config, ddp_config, model)  # global groups
    optimizer = get_megatron_optimizer(optimizer_config, [model])  # global groups
    forward_backward_func = get_forward_backward_func()  # global pipeline size
```

**What goes wrong.** The model runs on its collection, but gradient synchronization, optimizer
sharding, gradient norms, the pipeline schedule and RNG seeding use the global grid. The model then
trains with gradients reduced over another grid's ranks.

**Write instead.** Build one collection, then pass it to every component that runs on the grid, as
in [Passing the collection](process-group-collection.md#passing-the-collection).

## 10. Stage predicates and other non-group state

```python
    if parallel_state.is_pipeline_first_stage():
        ...
    vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
```

**What goes wrong.** Pipeline-stage predicates, the virtual-pipeline size and the global memory
buffer have no collection field, so code that uses them only works on the global grid. Offline
tools sometimes assign fake groups to `parallel_state`'s private module variables so that models
can be built. A component that reads its own collection never sees those groups.

**Write instead:**

- Check stages against the owner's pipeline group, with the virtual-pipeline information passed
  explicitly: `is_pp_first_stage(pp_group)` and `is_vp_first_stage(vp_stage, vp_size)` from
  `megatron.core.pipeline_parallel.utils`.
- Read the virtual-pipeline size from `config.virtual_pipeline_model_parallel_size`.
- In tools, put the fake groups in a `ProcessGroupCollection` and pass it to the models, instead of
  writing module variables.

## 11. A fallback nobody sees

```python
        if tp_group is None:
            if torch.distributed.get_rank() == 0:
                warnings.warn("tp_group is None, using default tp group", DeprecationWarning)
            tp_group = parallel_state.get_tensor_model_parallel_group()
```

**What goes wrong:**

- Python hides `DeprecationWarning` outside `__main__`, so this warning is usually never shown.
- Many fallbacks do not warn at all.
- Training scripts often silence warnings on ranks other than 0. A fallback that only happens on
  those ranks, for example on a pipeline stage that does not include rank 0, is then invisible.

Users never learn that they depend on the global grid, so removing a fallback later breaks them
without notice.

**Write instead:**

- Use a `FutureWarning` subclass, which Python shows by default.
- Warn on every rank where the fallback happens, once per component.
- Name the release that removes the fallback.
- Make sure filters that quiet the other ranks let this category through.

Keep the category specific, so that tests and strict runs can turn exactly these warnings into
errors:

```python
with warnings.catch_warnings():
    warnings.simplefilter("error", MyFallbackWarning)
    build_model(pg_collection=pgs)  # raises if anything falls back to the global grid
```

## Testing on a grid of its own

A test on the standard grid does not catch any of these patterns, because the global groups and
the component's groups are the same. Instead:

- **Use a different layout.** Initialize a global grid, then build the component on a collection
  whose layout differs from it, for example TP=2 for the component under a global TP=1, or PP=2
  under a global PP=1. Compare the result with the same component built on a matching global grid.
- **Use no global grid.** Build the collection with `HyperCommGrid` without calling
  `initialize_model_parallel`, and check that the component runs.
- **Forbid global reads.** Patch the accessors the component should not call so that they raise:

```python
def _no_global_grid(*args, **kwargs):
    raise AssertionError("read the global parallel grid")


monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", _no_global_grid)
```

Patch the name the component actually calls. A module that imported an accessor with
`from megatron.core.parallel_state import ...` holds its own reference, so patch it in that module.
