# Megatron FSDP v2 checkpoint design

This document describes the current Megatron FSDP v2 checkpoint path. It is not
a migration proposal and should only claim support that is implemented or covered
by active tests.

Megatron FSDP v2 uses the `fsdp_dtensor` checkpoint format. Model parameters are
saved as `DTensor` values and I/O is performed by PyTorch Distributed Checkpoint
(DCP). Optimizer state is still produced by Megatron's distributed optimizer
checkpoint API, then adapted so DCP can save and load it.

## Current support matrix

| Source checkpoint | Target | Model | Optimizer | Status |
|---|---|---:|---:|---|
| Megatron FSDP v2 `fsdp_dtensor` | Megatron FSDP v2 | Yes | Yes | Active round-trip coverage for `optim_grads_params`. |
| ND-parallel `torch_dist` with `fully_reshardable` optimizer state | Megatron FSDP v2 | Yes | Yes | Active online-conversion coverage. |
| ND-parallel `torch_dist` with `dp_reshardable` optimizer state | Megatron FSDP v2 | Partial | No | Optimizer load is rejected because the bucket layout is incompatible with the v2 name-keyed optimizer layout. |
| Megatron FSDP v1 `fsdp_dtensor` | Megatron FSDP v2 | No | No | Test cases are present but skipped because the v1 checkpoint format is not available. |
| Legacy `torch` | Megatron FSDP v2 | No | No | Not supported by the online conversion path. |

Known gaps are listed at the end of this document.

## Main training checkpoint path

The Megatron training loop still builds checkpoints through
`megatron/training/checkpointing.py::generate_state_dict()`. For
`--ckpt-format fsdp_dtensor` and Megatron FSDP v2, the important steps are:

1. `model[i].state_dict_for_save_checkpoint()` returns the FSDP v2 model state
   dict. The adapter wires this method to `module.state_dict()`.
2. `propagate_chunk_metadata_to_state_dict(model[i], model_sd)` copies uneven
   DTensor chunk metadata from model parameters to the state-dict DTensors.
3. `optimizer.sharded_state_dict(..., metadata={"distrib_optim_sharding_type":
   "fsdp_dtensor"})` returns a name-keyed optimizer state dict:

   ```python
   {
       "state": {
           "<param_name>": {
               "exp_avg": <plain tensor or DTensor>,
               "exp_avg_sq": <plain tensor or DTensor>,
               ...
           },
           ...
       },
       "param_to_group_meta": {
           "<param_name>": {"lr": ..., "weight_decay": ..., ...},
           ...
       },
   }
   ```

4. Before DCP save or load, `preprocess_fsdp_dtensor_state_dict()` detects
   Megatron FSDP v2 and delegates to `preprocess_mcore_fsdp_v2_state_dict()`,
   implemented in `megatron/core/distributed/fsdp/checkpoint.py`.
5. `preprocess_mcore_fsdp_v2_state_dict()` applies MCore-specific postprocessing
   through `_apply_mcore_postprocess()`.

This is the production training path. The `MegatronFSDPStateful` wrapper in
`checkpoint.py` is a standalone DCP `Stateful` helper; it is not the path used by
the Megatron training loop.

## Save flow

```text
save_checkpoint()
  └─ generate_state_dict()
       ├─ model.state_dict_for_save_checkpoint()
       ├─ propagate_chunk_metadata_to_state_dict()
       └─ optimizer.sharded_state_dict(..., fsdp_dtensor)
  └─ preprocess_fsdp_dtensor_state_dict()
       └─ preprocess_mcore_fsdp_v2_state_dict()
            └─ _apply_mcore_postprocess()
                 ├─ copy model chunk metadata into state-dict DTensors
                 ├─ wrap optimizer state tensors as DTensors when needed
                 ├─ remove FP8 extra-state artifacts
                 ├─ split fused SwiGLU / GDN / MambaMixer parameters
                 ├─ remap expert keys when `num_experts` is set
                 └─ verify DTensor chunk metadata
  └─ torch.distributed.checkpoint.save()
```

Optimizer state from FusedAdam and similar optimizers may be plain tensors
matching each parameter's local DTensor shard. DCP needs distributed values to be
represented as DTensors, so `_build_dtensor_optim_sd()` returns a copy of the
optimizer state dict with matching plain tensors wrapped as uneven DTensors.

## Load flow

```text
load_checkpoint()
  └─ generate_state_dict(..., is_loading=True)
       └─ optimizer.sharded_state_dict(..., is_loading=True)
            └─ initialize placeholder optimizer states
  └─ _load_base_checkpoint(..., ckpt_format="fsdp_dtensor")
       ├─ keep shallow copies of raw model and optimizer dicts
       ├─ preprocess_fsdp_dtensor_state_dict()
       ├─ DCP load into the preprocessed DTensor skeleton
       └─ restore raw model and optimizer dicts
  └─ ddp_model.load_state_dict()
  └─ sync_module_states_after_load()
  └─ optimizer.load_state_dict()
```

The load path intentionally preserves the raw optimizer state dict before
preprocessing. The preprocessed dict gives DCP DTensor objects to load into, and
those DTensors share storage with the original plain tensors. After DCP load, the
raw optimizer dict is restored so `optimizer.load_state_dict()` receives the
plain-tensor format expected by the underlying optimizer.

## Key modules and responsibilities

| Module | Responsibility |
|---|---|
| `megatron/training/checkpointing.py` | Builds save/load skeletons, dispatches v2 preprocessing, performs DCP save/load, and calls model/optimizer `load_state_dict()`. |
| `megatron/core/distributed/fsdp/checkpoint.py` | Implements Megatron FSDP v2 postprocessing, DTensor optimizer wrapping, chunk metadata verification, prefix helpers, and `torch_dist` online conversion. |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py` | Creates and propagates uneven-DTensor chunk metadata used by DCP. |
| `megatron/core/optimizer/distrib_optimizer.py` | Produces and consumes the `fsdp_dtensor` optimizer state dict with name-keyed parameter state. |
| `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` | Routes the FSDP v2 adapter state-dict calls and synchronizes module state after load. |

## MCore postprocessing

`_apply_mcore_postprocess(raw_state_dict, args, model)` is the v2 entry point for
checkpoint-specific state-dict rewrites. It copies the input top-level dict and
then applies the following transformations:

- Propagate uneven-DTensor chunk metadata from live model parameters to model
  state-dict values.
- Wrap matching optimizer state tensors as uneven DTensors for DCP.
- Remove FP8 `._extra_state` entries from the model state dict.
- Split fused model parameters for SwiGLU, GDN, and MambaMixer layouts. Optimizer
  state is split when the helper can map it safely.
- Remap expert keys when `args.num_experts` is set.
- Verify that every DTensor has valid chunk metadata before DCP sees the dict.

The postprocessing layer is format-specific glue. It should not silently claim
support for model structures that are not covered by tests.

## Uneven DTensor chunk metadata

DCP needs correct chunk metadata for uneven sharding, where local shard sizes may
differ across ranks. Megatron FSDP v2 uses helpers from `uneven_dtensor.py` to
attach two methods to each DTensor local tensor:

- `__create_chunk_list__`
- `__create_write_items__`

`_verify_chunk_metadata()` checks every DTensor in the flattened state dict:

1. The local tensor has `__create_chunk_list__`.
2. The total number of elements described by the chunk list equals the local
   tensor's number of elements.

On failure, the verifier logs the key, global/local shape, chunk offsets and
sizes, metadata source tag, and device mesh before raising.

Useful metadata source tags include:

| Source tag | Meaning |
|---|---|
| `init` | Metadata created during FSDP v2 initialization. |
| `preprocess` | Metadata recomputed by uneven-DTensor preprocessing. |
| `propagate:<src>` | Metadata copied from a model parameter. |
| `split` | Metadata derived while splitting a fused DTensor. |
| `make_uneven` | Metadata created while wrapping a local tensor as an uneven DTensor. |

## Prefix alignment

Megatron checkpoints conventionally store model keys with a `module.` prefix.
Megatron FSDP v2 applies `fully_shard()` directly to the model, so raw model
state-dict keys may not contain that prefix. `checkpoint.py` contains
`add_module_prefix()`, `strip_module_prefix()`, and `get_model_state_dict()` for
this alignment.

In the current training path, parameter names may already include the adapter's
`module.` prefix depending on which object is used to build the state dict.
Prefix handling should therefore be treated as normalization logic, not as proof
that every caller sees the same raw key form.

## Online conversion from `torch_dist`

When loading a global `torch_dist` checkpoint into Megatron FSDP v2,
`checkpointing.py` first builds a v2 `fsdp_dtensor` skeleton, then calls
`load_torch_dist_checkpoint_into_megatron_fsdp_v2()`.

The implementation in `checkpoint.py::_load_torch_dist_into_megatron_fsdp_v2()`
does five things:

1. Preprocess and verify the v2 skeleton. Optimizer plain tensors are wrapped in
   shadow DTensors that share storage with the originals.
2. Read DCP metadata from the `torch_dist` checkpoint and map checkpoint keys to
   v2 canonical parameter names.
3. DCP-load one-to-one model weights, high-precision optimizer parameter copies,
   and optimizer states.
4. Load fused multi-layer or expert tensors and slice them into per-layer or
   per-expert v2 DTensors.
5. In strict mode, error if any required v2 model parameter or optimizer state was
   not loaded.

Only `fully_reshardable` distributed-optimizer checkpoints are documented as
optimizer-load compatible with this conversion path. `dp_reshardable` optimizer
state uses a bucket-based layout and is rejected for Megatron FSDP v2 optimizer
conversion.

## Loading into the live model

After DCP fills the state dict, the training loop calls `load_state_dict()` on the
DDP/FSDP wrapper. For Megatron FSDP v2, the adapter delegates to the parent module
load implementation and then `sync_module_states_after_load()` walks FSDP v2
modules and calls the root module's state synchronization helper.

Optimizer load accepts the name-keyed `fsdp_dtensor` optimizer state. If
`param_to_group_meta` is present, `DistributedOptimizer.load_state_dict()` rebuilds
standard PyTorch `param_groups` before delegating to the inner optimizer.

## Active test coverage

The current unit coverage is centered in
`tests/unit_tests/distributed/megatron_fsdp/v2/test_mcore_checkpoint.py` and
related FSDP v2 tests.

Covered paths include:

- ND-parallel `torch_dist` with `fully_reshardable` optimizer state to Megatron
  FSDP v2.
- Megatron FSDP v2 `fsdp_dtensor` round-trip with `optim_grads_params`.
- Optimizer state verification for the covered conversion/round-trip cases.
- `get_state_dict` helper coverage in `test_fully_shard.py`.
- Format-transform helper coverage in `test_fsdp_dtensor_checkpoint.py`.

The v1-to-v2 parameterized cases in `test_mcore_checkpoint.py` are skipped with
`pytest.skip("v1 checkpoint format not available")`.

## Known gaps

- Megatron FSDP v2 checkpoint round-trip coverage for sharding strategies other
  than `optim_grads_params`.
- Megatron FSDP v1 `fsdp_dtensor` to Megatron FSDP v2 conversion.
- Pipeline-parallel checkpoint coverage for Megatron FSDP v2. The current
  preprocessing path assumes a top-level `model` entry and does not document
  `model0`, `model1`, ... handling as supported.
- Multi-optimizer / `ChainedOptimizer` checkpoint coverage.
- Frozen-parameter and stub-optimizer coverage.
- Cross-topology resharding across different DP layouts.
- Async checkpoint save/load for Megatron FSDP v2.
