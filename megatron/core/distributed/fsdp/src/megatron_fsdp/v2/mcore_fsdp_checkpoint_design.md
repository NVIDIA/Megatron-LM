# Megatron FSDP v2 Checkpoint Design

## 1. Overview

Megatron FSDP v2 (`use_megatron_fsdp_v2=True`) wraps model parameters as PyTorch
`DTensor` objects sharded across the data-parallel dimension. This document describes
the checkpoint save/load architecture and how it integrates with MCore's
`DistributedOptimizer` and DCP (`torch.distributed.checkpoint`).

### Goals

- Save and load Megatron FSDP v2 model + optimizer state via DCP.
- Support `fsdp_dtensor` as the canonical checkpoint format.
- Enable online checkpoint conversion from legacy Megatron formats (ND-parallel,
  Megatron FSDP v1 baseline) to Megatron FSDP v2.
- Handle uneven DTensor sharding (parameters not evenly divisible by DP size).

### Non-Goals (current scope)

- Async checkpoint save/load for Megatron FSDP v2.
- Cross-node checkpoint resharding (different DP topology on load).

### Feature Support Matrix

| Source Format | Target Format | Model | Optimizer | Notes |
|---------------|--------------|-------|-----------|-------|
| MFSDP v2 | MFSDP v2 | ✓ | ✓ | Full round-trip and cross-setting (`optim_grads` ↔ `optim_grads_params`) |
| MFSDP v1 baseline | MFSDP v2 | ✗ | ✗ | Currently skipped in tests (``pytest.skip("v1 checkpoint format not available")``).  Both use ``fsdp_dtensor`` format but key names may differ.  Planned for future support. |
| ND-parallel (`torch_dist`) | MFSDP v2 | ✓ | ✓ (``fully_reshardable`` only) | Online conversion via `_load_torch_dist_into_megatron_fsdp_v2` in `checkpointing.py`. Expert weights are split from flattened multi-expert tensors. Optimizer states (`exp_avg`, `exp_avg_sq`) are loaded into V2's name-based format. Hi-precision FP32 optimizer param copies are used for model weights when available. ``dp_reshardable`` format is not supported (bucket-based layout incompatible with name-based V2 format). |
| ND-parallel (`torch`) | MFSDP v2 | ✗ | ✗ | Not supported (different serialization) |

---

## 2. Background: Two FSDP Paths

MCore wraps FSDP through `FullyShardedDataParallel` (in `mcore_fsdp_adapter.py`).
There are two code paths:

| Path | Flag | Inner module | Model state dict |
|------|------|-------------|------------------|
| **Legacy Megatron FSDP** | `use_megatron_fsdp=True`, `use_megatron_fsdp_v2=False` | `MegatronFSDP` | `state_dict()` with DTensor hooks |
| **Megatron FSDP v2** | `use_megatron_fsdp=True`, `use_megatron_fsdp_v2=True` | `FSDPModule` (DTensor-native) | `state_dict()` (DTensors natively) |

The `fsdp_dtensor` checkpoint format (`--ckpt-format fsdp_dtensor`) is the required
format for both paths. It uses DCP directly, storing each parameter as a `DTensor`.

---

## 3. Architecture

### 3.1 Component Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| `uneven_dtensor.py` | `megatron/core/distributed/fsdp/src/megatron_fsdp/` | `get_state_dict`, `preprocess_state_dict_for_uneven_dtensor`, chunk metadata for uneven DTensors |
| `fsdp_dtensor_checkpoint.py` | `megatron/core/transformer/` | (v1 only) SWiGLU split, GDN split, expert key remapping, FP8 cleanup.  v2 equivalents live in ``checkpoint.py``.
| `distrib_optimizer.py` | `megatron/core/optimizer/` | `state_dict()`, `load_state_dict()`, `sharded_state_dict()`, `sharded_param_state_fsdp_dtensor()` |
| `checkpointing.py` | `megatron/training/` | High-level save/load orchestration, `_build_megatron_fsdp_v2_state_dict`, `preprocess_fsdp_dtensor_state_dict()` |
| `checkpoint.py` | `megatron/core/distributed/fsdp/` | `MegatronFSDPStateful`, `_apply_mcore_postprocess`, `_verify_chunk_metadata`, `_propagate_chunk_metadata_to_state_dict`, `_split_fused_params_v2` (SwiGLU+GDN+MambaMixer), `_build_dtensor_optim_sd`, `_preprocess_and_verify_v2_state_dict`, `_build_torch_dist_to_v2_map`, `load_torch_dist_into_fsdp_v2` |
| `mcore_fsdp_adapter.py` | `megatron/core/distributed/fsdp/` | Routes to v1 `MegatronFSDP` or Megatron FSDP v2 `fully_shard` |

### 3.1.1 `checkpoint.py` Functions (Megatron FSDP v2)

| Function | Description |
|----------|-------------|
| `MegatronFSDPStateful` | ``Stateful`` wrapper implementing DCP protocol. ``state_dict()`` calls ``_get_state_dict`` from ``uneven_dtensor`` (attaches uneven DTensor chunk metadata), then ``_apply_mcore_postprocess`` (SwiGLU/GDN split, FP8 cleanup, expert remapping). ``load_state_dict()`` uses PyTorch's ``set_state_dict``. |
| `_preprocess_and_verify_v2_state_dict` | Build a shadow optimizer state dict with DTensors sharing storage with original plain tensors (dual-dict pattern). Verifies ``__create_chunk_list__`` and ``__create_write_items__`` metadata on all model and optimizer DTensors. Returns canonical ``(v2_by_canonical, v2_optim_state)`` maps. |
| `_build_torch_dist_to_v2_map` | Build mapping from torch_dist metadata keys to v2 DTensors. Returns ``(regular_model, hi_prec_model, optim_keys, optim_matched)``. |
| `_build_dtensor_optim_sd(raw_opt_sd, model)` | Wrap plain-tensor optimizer states as uneven DTensors. Returns a copy (does not mutate original). Uses ``_maybe_wrap_as_uneven_dtensor`` internally. |
| `_maybe_wrap_as_uneven_dtensor(tensor, dist_param)` | Wrap a single plain tensor as an uneven DTensor if it matches the parameter's local shard shape; otherwise return unchanged. Shared by ``_build_dtensor_optim_sd`` and ``_preprocess_and_verify_v2_state_dict``. |
| `load_torch_dist_into_fsdp_v2` | Entry point for online checkpoint conversion from legacy ``torch_dist`` format to ``fsdp_dtensor``. Five-phase pipeline: preprocess/verify, build name mapping, DCP load, expert params, verify. |
| `add_module_prefix(state_dict)` | Add ``module.`` prefix to all state dict keys. Megatron FSDP v2 lacks ``MegatronFSDP`` wrapper so ``model.state_dict()`` keys have no prefix; this aligns with Megatron's checkpoint format. |
| `strip_module_prefix(state_dict)` | Remove ``module.`` prefix from state dict keys. Inverse of ``add_module_prefix``, used when loading checkpoint back into FSDP v2 model. |
| `get_model_state_dict(model)` | Get model state dict with ``module.`` prefix. Auto-detects whether prefix is already present; adds it if missing. |
| `get_optimizer_state_dict(optimizer, is_loading)` | Get optimizer state dict via Path A (``sharded_param_state_fsdp_dtensor``). Delegates to ``optimizer.sharded_state_dict()`` with ``fsdp_dtensor`` sharding type. Returns ``{"state": ..., "param_to_group_meta": ...}``. |
| `handle_fp8_extra_state_case(model_sd)` | Remove ``._extra_state`` keys from model state dict (FP8 artifact cleanup). |
| `handle_experts_in_state_dict(model_sd, num_experts)` | Rename expert parameter keys for expert-parallel sharding. |
| `handle_swiglu_in_state_dict_v2(model, model_sd, opt_sd)` | Thin wrapper: precomputes ``layer_glu`` map, delegates to ``_split_fused_params_v2`` with SwiGLU detector and ``_w``/``_v`` suffix format. |
| `handle_gdn_in_state_dict_v2(model, model_sd, opt_sd)` | Thin wrapper: delegates to ``_split_fused_params_v2`` with ``_match_gdn_key`` detector and ``.query``/``.key``/``.value`` suffix format. |
| `handle_mamba_in_state_dict_v2(model, model_sd, opt_sd)` | Thin wrapper: delegates to ``_split_fused_params_v2`` with ``_mamba_mixer_detector`` and MambaMixer suffix format. |
| `_split_fused_params_v2(...)` | Unified fused-parameter splitting skeleton. Iterates model and optimizer state dicts, calls the ``detector(key, dtensor, model) -> (sizes, names, dim)`` callback to identify fused tensors, splits via ``split_dtensor``, and renames keys via ``key_fmt(key, sub_name)``. Shared by SwiGLU, GDN, and MambaMixer. |
| `_verify_chunk_metadata(flattened_sd)` | Final verification: checks every DTensor has ``__create_chunk_list__`` AND that ``chunks_total_numel == local_numel``. On failure, logs key, shape, chunk details, source tag, and device mesh before raising ``AssertionError``. |
| `_propagate_chunk_metadata_to_state_dict(model, state_dict)` | Copy ``__create_chunk_list__`` / ``__create_write_items__`` from model parameters (which have them from init) to state dict DTensors (fresh from ``model.state_dict()``). Matches by key name. Logs unmatched DTensors at rank 0 for diagnostic tracing. |
| `get_chunk_meta_source(dtensor)` | Return the source tag (e.g. ``"init"``, ``"preprocess"``, ``"split"``, ``"propagate:init"``) stored on ``dtensor._local_tensor._chunk_meta_source``. |
| `_is_swiglu_key(key)` | Check whether a key matches any SwiGLU fc1 pattern. |
| `_match_gdn_key(key, dtensor)` | GDN detector: returns ``(sizes, names, dim)`` for fused QKV projections, or ``None``. |
| `_detect_glu_layers(model)` | Return ``{layer_path: gated_linear_unit}`` for all TransformerLayers. |
| `_model_has_module_prefix(model)` | Detect whether model's ``named_parameters()`` keys already carry ``module.`` prefix. |
| `normalize_torch_dist_key(key)` | Normalize a torch_dist checkpoint key to v2 canonical form. Maps ``transformer_layer`` → ``mtp_model_layer``. |
| `reverse_normalize_torch_dist_key(key)` | Reverse the v2 canonical key back to torch_dist naming (``mtp_model_layer`` → ``transformer_layer``). Used when constructing DCP load paths that must match torch_dist storage paths. |


### 3.2 Current Save Flow

> **Note:** This describes the current `generate_state_dict`-based flow. For FSDP v2,
> ``_build_megatron_fsdp_v2_state_dict`` (Section 12) replaces this in the v2 code path.

```
save_checkpoint()
  |
  +-- 1. generate_state_dict()
  |     |
  |     +-- model[i].state_dict_for_save_checkpoint()
  |     |     Returns DTensor state dict.
  |     |
  |     +-- optimizer.sharded_state_dict(state_dict, metadata={'distrib_optim_sharding_type': 'fsdp_dtensor'})
  |     |     Returns {"state": {name: opt_state}, "param_to_group_meta": {...}}
  |     |     via sharded_param_state_fsdp_dtensor().
  |     |
  |     +-- rng_state, scheduler state, etc.
  |
  +-- 2. preprocess_fsdp_dtensor_state_dict()
  |     - handle_fp8_extra_state_case
  |     - handle_swiglu_in_state_dict (model + optimizer)
  |     - handle_experts_in_state_dict (EP key remapping)
  |     - preprocess_state_dict_for_uneven_dtensor
  |
  +-- 3. torch.distributed.checkpoint.save(state_dict, storage_writer)
```

### 3.3 Current Load Flow

> **Note:** This describes the current `generate_state_dict`-based flow. For FSDP v2,
> ``_build_megatron_fsdp_v2_state_dict`` (Section 12) replaces this in the v2 code path.

```
load_checkpoint()
  |
  +-- 1. Build sharded_state_dict via generate_state_dict()
  |     Same structure as save. For optimizer: is_loading=True triggers
  |     _init_optimizer_states_with_dummy_values() to create placeholder states.
  |
  +-- 2. _load_base_checkpoint()
  |     - preprocess_fsdp_dtensor_state_dict()
  |     - torch.distributed.checkpoint.load_state_dict(state_dict, reader)
  |
  +-- 3. Post-load application
        - ddp_model[i].load_state_dict(state_dict['model'], strict)
        - optimizer.load_state_dict(state_dict['optimizer'])
        - RNG states, scheduler, etc.
```

---

## 4. Model State Dict

Megatron FSDP v2 uses `model.state_dict()` which returns a dict of `DTensor` values.
The keys follow the Megatron `module.` prefix convention
(e.g., `module.embedding.word_embeddings.weight`).

### 4.1 `state_dict_for_save_checkpoint` — Adapter (Phase 1 Bridge)

**Current status:** Set to `not_implemented_op` for v2 (line 384 of `mcore_fsdp_adapter.py`).

**Phase 1 fix:** Wire to `model.state_dict()`. This is a **temporary bridge** until
Phase 2 replaces `generate_state_dict` with `get_state_dict` from `uneven_dtensor.py`.
Once Path B is the primary path, this wiring is no longer needed and will be removed
in Phase 3.

```python
# In _init_with_fully_shard(), replace:
self.module.state_dict_for_save_checkpoint = not_implemented_op
self.state_dict_for_save_checkpoint = not_implemented_op
# With:
self.module.state_dict_for_save_checkpoint = lambda *args, **kwargs: module.state_dict()
self.state_dict_for_save_checkpoint = lambda *args, **kwargs: module.state_dict()
```

### 4.2 `load_state_dict` — Adapter

**Current:** Calls `super().load_state_dict()` at line 398-399 which delegates through
the FSDP framework. **Status: OK.**

```python
if self.ddp_config.use_megatron_fsdp_v2:
    super().load_state_dict(state_dict, strict=strict)
    return
```

---

## 5. Optimizer State Dict

There are **two** paths for obtaining the optimizer state dict, depending on context.

### 5.1 Path A: Megatron Training Loop (`checkpointing.py`)

Uses `optimizer.sharded_state_dict(model_state_dict, sharding_type="fsdp_dtensor")`,
which calls `sharded_param_state_fsdp_dtensor()`. Returns:

```python
{
    "state": {
        "<param_name>": {"exp_avg": torch.Tensor, "exp_avg_sq": torch.Tensor, ...},
        ...
    },
    "param_to_group_meta": {
        "<param_name>": {"lr": ..., "weight_decay": ...},
        ...
    }
}
```

**Important:** ``FusedAdam`` (and similar NVIDIA optimizers) store ``exp_avg`` /
``exp_avg_sq`` as **plain tensors** matching the parameter's local DTensor shard,
NOT as DTensors. This is because the optimizer kernel operates on the local data
directly. Path A returns these plain tensors as-is.

For DCP compatibility, these plain tensors must be wrapped as DTensors via
``_wrap_optim_states_as_dtensors`` before DCP save (see Section 5.10), and
unwrapped back via ``_unwrap_optim_states_from_dtensors`` after DCP load.

This is the primary path for Megatron-integrated training.

### 5.2 Path B: Standalone `get_state_dict()` (`uneven_dtensor.py`)

Uses PyTorch's native `torch.distributed.checkpoint.state_dict.get_state_dict()`,
which calls `optimizer.state_dict()` internally. For `DistributedOptimizer` with
Megatron FSDP, `state_dict()` returns the inner optimizer's full state dict directly.

This path is used by:
- `checkpoint.py` (``MegatronFSDPStateful`` wrapper and MCore post-processing helpers)
- `test_mcore_checkpoint.py` (checkpoint save/load and online format conversion tests)

### 5.3 `DistributedOptimizer.__init__` — FSDP Short-Circuit

When `use_megatron_fsdp=True` or `use_megatron_fsdp_v2=True`, `__init__()` returns
early (line 543) without setting up buffer ranges, gbuf mappings, or shard slicing.
Megatron FSDP manages weight/gradient memory directly.

### 5.4 `DistributedOptimizer.state_dict()` — FSDP Branch

Returns the **full** inner optimizer state dict because:
- FSDP manages parameter state as DTensors (no separate `save_parameter_state`).
- PyTorch's `get_state_dict()` expects `optimizer.state_dict()` to include state.
- The `sharded_param_state_fsdp_dtensor()` path handles Megatron-specific key
  remapping separately.

```python
if self.ddp_config.use_megatron_fsdp or self.ddp_config.use_megatron_fsdp_v2:
    return self.optimizer.state_dict()
```

**Why not strip `"state"` like the non-FSDP path?** The legacy non-FSDP path strips
the `"state"` key from `state_dict()` and stores optimizer parameter states in a
separate `param_state` checkpoint. This is necessary because the non-FSDP path manages
parameters in contiguous gradient buffers requiring manual sharding. For Megatron FSDP,
DCP handles DTensor sharding natively, so splitting is unnecessary.

### 5.5 `DistributedOptimizer.load_state_dict()` — FSDP Branch

Converts name-based `param_to_group_meta` back to tensor-based `param_groups`, then
delegates to the inner optimizer:

```python
if self.ddp_config.use_megatron_fsdp or self.ddp_config.use_megatron_fsdp_v2:
    if "param_to_group_meta" in state_dict:
        state_dict["param_groups"] = self._param2group_meta_to_param_groups(...)
    self.optimizer.load_state_dict(state_dict)
    return
```

### 5.6 `DistributedOptimizer.sharded_state_dict()` — FSDP Branch

Only `sharding_type="fsdp_dtensor"` is supported. Calls
`sharded_param_state_fsdp_dtensor()` which:
1. Optionally initializes optimizer states with dummy values (for loading).
2. Maps tensor keys to parameter name strings via `_param_name()`.
3. Returns `{"state": ..., "param_to_group_meta": ...}`.

**Why this works for Megatron FSDP v2:** The v2 path uses the same `self.optimizer`
(a standard Torch optimizer like AdamW) and the same `_param_name` mapping. DTensor
parameters are correctly identified by name. The state dict keys are the same as the
checkpoint's model state dict keys, so DCP can match them.

### 5.8 `sharding_type="fsdp_dtensor"` Rationale

The `fsdp_dtensor` checkpoint format signals to MCore's `checkpointing.py` to:
1. Use DCP for all I/O.
2. Store parameters as DTensors (each carries its own sharding metadata).
3. Store optimizer state as a flat `{param_name: state}` dict.

This is the only format compatible with Megatron FSDP because the non-DCP formats
(`torch`, `torch_dist`) use gather/scatter patterns that assume contiguous gradient
buffers, which don't exist in the FSDP path.

### 5.10 DTensor Wrapping / Unwrapping Layer (Path A)

Path A must add a wrapping/unwrapping layer to bridge the gap between
FusedAdam's plain-tensor optimizer states and DCP's DTensor requirement.

**Why wrapping is needed on save:**

DCP builds a global save plan by inspecting the state dict. Sharded data must
be DTensors so DCP knows about the sharding (mesh, placements, per-rank chunk
metadata). Plain tensors with the same FQN on every rank are treated as
non-sharded (replicated) data, which causes ``"item.index.fqn not in md"``
errors when ranks hold different local shards.

**Why unwrapping is needed on load:**

After DCP loads DTensors into the skeleton, the optimizer state dict contains
DTensor values. But FusedAdam's ``load_state_dict`` → ``set_scaled_state``
calls ``state[state_name].copy_(unscaled_state)`` where ``state[state_name]``
is a plain tensor (FusedAdam's internal storage). Passing a DTensor causes
``RuntimeError: aten.copy_.default got mixed torch.Tensor and DTensor``.

**The wrapping functions (in ``checkpoint.py``):**

- ``_build_dtensor_optim_sd(raw_opt_sd, model)`` — wraps every plain-tensor optimizer
  state that matches a parameter's local shard as an uneven DTensor. Returns a copy;
  original is not mutated.
- ``_maybe_wrap_as_uneven_dtensor(tensor, dist_param)`` — wraps a single tensor. Shared
  by ``_build_dtensor_optim_sd`` and ``_preprocess_and_verify_v2_state_dict``. Returns
  the tensor unchanged if it is already a DTensor or the shape doesn't match.

**Placement in the flow:**

::

   Save:  get_optim_state_dict (plain) → _wrap_optim_states_as_dtensors → DCP save
   Load:  DCP load (DTensors) → _unwrap_optim_states_from_dtensors → optimizer.load_state_dict

**Note on the baseline (v1) path:**

The baseline path does NOT need these functions because the existing conversion
tests (``test_mcore_checkpoint.py``) never save optimizer states to DCP — they
save only model state dicts via ``dcp_save({"model": source_sd}, ...)``. If
optimizer state saving were added to the baseline path, the same wrapping/
unwrapping layer would be required.

### 5.11 Dual-Dict Pattern: Plain Tensors → DTensor Wrapping → Restore

The main training loop (``checkpointing.py``) uses a **dual-dict pattern** to
avoid the need for ``_unwrap_optim_states_from_dtensors``:

::

  _build_megatron_fsdp_v2_state_dict()
    |
    +-- state_dict["optimizer"] = plain tensors (from get_optimizer_state_dict)
    |
  save_checkpoint() / _load_base_checkpoint()
    |
    +-- (load only) raw_optimizer_state_dict = state_dict["optimizer"].copy()
    |       captures plain-tensor dict before wrapping
    |
    +-- _wrap_optim_states_as_dtensors(state_dict, model)
    |       wraps in-place: plain → DTensor (shared storage via make_uneven_dtensor)
    |
    +-- DCP save / DCP load
    |       writes through DTensors → plain tensors auto-updated via shared storage
    |
    +-- (load only) state_dict["optimizer"] = raw_optimizer_state_dict
            restores plain-tensor dict for optimizer.load_state_dict

Key properties:

* **Wrapping is deferred** — ``_build_megatron_fsdp_v2_state_dict`` keeps
  optimizer states as plain tensors. Wrapping as DTensors happens later, in
  ``save_checkpoint`` (before DCP save) or ``_load_base_checkpoint`` (after
  raw copy, before DCP load).
* **No unwrapping needed** — the ``raw_optimizer_state_dict.copy()`` at
  line 1526 of ``checkpointing.py`` (existing code) captures the plain-tensor
  state dict. After DCP load writes through the DTensors (which share storage
  with the plain tensors), the raw dict is restored. The raw dict already has
  the correct loaded values — no ``.to_local()`` conversion needed.
* **split_optimizer=False** — ``_apply_mcore_postprocess`` only splits model
  keys (SwiGLU ``_w``/``_v``). Optimizer keys stay as canonical parameter
  names, which ``DistributedOptimizer.load_state_dict`` can match.

---

## 6. Proposed: `uneven_dtensor.get_state_dict` as Primary Path

### 6.1 Motivation

The current `generate_state_dict()` in `checkpointing.py` mixes three concerns:
1. Building state dict structure per PP chunk
2. Format-specific branching (`torch_dist` vs `fsdp_dtensor` vs legacy)
3. Dispatching to optimizer-specific APIs (`sharded_state_dict` with sharding type)

For Megatron FSDP v2, we can do better. `get_state_dict()` from `uneven_dtensor.py`
wraps PyTorch's native `torch.distributed.checkpoint.state_dict.get_state_dict()`
with uneven DTensor preprocessing. It produces correct DTensor state dicts for both
model and optimizer in a single call. MCore-specific post-processing (FP8 cleanup,
SwiGLU split, expert key remapping) can be applied as a separate layer.

### 6.2 Proposed Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: Megatron FSDP State Dict                           │
│  uneven_dtensor.get_state_dict(model, optimizer)             │
│  → model_state_dict: {name: DTensor}                         │
│  → optimizer_state_dict: {state: ..., param_groups: ...}     │
│  Internally calls PyTorch's get_state_dict + preprocesses    │
│  uneven DTensor chunk metadata for DCP serialization.        │
│  Handles: DTensor serialization, model/optimizer state dict  │
│           separation, uneven DTensor chunk metadata          │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│  Layer 2: MCore Post-Processing                              │
│  _apply_mcore_postprocess(state_dict, args, model)           │
│  → _propagate_chunk_metadata_to_state_dict  (copy metadata)  │
│  → _build_dtensor_optim_sd                  (wrap as DTensor)│
│  → handle_fp8_extra_state_case              (FP8 cleanup)    │
│  → handle_swiglu_in_state_dict_v2           (split fc1)      │
│  → handle_gdn_in_state_dict_v2              (split QKV)      │
│  → handle_mamba_in_state_dict_v2            (split Mamba)    │
│  → handle_experts_in_state_dict             (EP key remap)   │
│  → _verify_chunk_metadata                   (consistency ✓)  │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│  Layer 3: DCP I/O                                            │
│  torch.distributed.checkpoint.save(state_dict, writer)       │
│  torch.distributed.checkpoint.load_state_dict(state_dict,    │
│                                              reader)         │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 Save Flow (Proposed)

```python
def save_checkpoint_fsdp_v2(model_chunks, optimizer, ...):
    # Layer 1: State dicts via uneven_dtensor.get_state_dict
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        get_state_dict,
    )

    for pp_rank, model_chunk in enumerate(model_chunks):
        model_sd, optim_sd = get_state_dict(
            model=model_chunk,
            optimizers=optimizer,
        )
        # model_sd keys: "module.embedding.weight", "module.layers.0.fc.weight", ...
        # optim_sd keys: "state", "param_groups"
        # Uneven DTensor chunk metadata is already attached by get_state_dict.

        state_dict = {
            f"model{pp_rank}" if len(model_chunks) > 1 else "model": model_sd,
            "optimizer": optim_sd,
            "rng_state": ...,
        }

        # Layer 2: MCore post-processing (FP8, SwiGLU, expert keys)
        preprocess_fsdp_dtensor_state_dict(args, state_dict, model_chunks[0])

        # Layer 3: DCP save
        torch.distributed.checkpoint.save(state_dict, checkpoint_id=str(ckpt_dir))
```

### 6.4 Load Flow (Proposed)

> **Important:** `get_state_dict` is designed for save — it gathers current
> state into a dict. For loading, we need a **skeleton** (dict with correct keys and
> tensor shapes but empty/placeholder values) that DCP fills in-place. The optimizer
> states must be pre-allocated BEFORE building the skeleton, so the skeleton has the
> right structure for DCP to write into.

```python
def load_checkpoint_fsdp_v2(model_chunks, optimizer, ...):
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        get_state_dict,
    )

    # Pre-allocate: optimizer states must exist before get_state_dict
    # can produce a skeleton with the right structure.
    # Equivalent to the current is_loading=True path in sharded_param_state_fsdp_dtensor.
    optimizer._init_optimizer_states_with_dummy_values()

    for pp_rank, model_chunk in enumerate(model_chunks):
        # Build skeleton — get_state_dict produces a dict with the right keys
        # and tensor shapes, plus uneven DTensor chunk metadata for DCP.
        # Values are stale (skeleton only needs structure).
        model_sd, optim_sd = get_state_dict(
            model=model_chunk,
            optimizers=optimizer,
        )

        state_dict = {
            f"model{pp_rank}" if len(model_chunks) > 1 else "model": model_sd,
            "optimizer": optim_sd,
        }

        # MCore post-processing (on skeleton, before DCP load)
        state_dict["_model"] = model_chunks[0]
        preprocess_fsdp_dtensor_state_dict(args, state_dict, model_chunks[0])

        # DCP load fills DTensors in-place.
        torch.distributed.checkpoint.load_state_dict(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_id),
            planner=DefaultLoadPlanner(allow_partial_load=not strict),
        )

        # After DCP load, model DTensors are already updated in-place.
        # For v2, load_state_dict is a no-op for DTensor params but required
        # for non-DTensor state (buffers, extra_state) and strict key checking.
        model_chunk.load_state_dict(state_dict["model"], strict=False)
        optimizer.load_state_dict(state_dict["optimizer"])
```

### 6.5 Key Design Decisions

**Why `uneven_dtensor.get_state_dict` instead of `model.state_dict_for_save_checkpoint` + `optimizer.sharded_state_dict`?**

| Aspect | Current (Path A) | Proposed (Path B) |
|--------|-----------------|-------------------|
| Model state dict | `model.state_dict_for_save_checkpoint()` — `not_implemented_op` for v2 | `get_state_dict(model, optimizer)` — works natively with DTensors |
| Optimizer state dict | `optimizer.sharded_state_dict(state_dict, sharding_type="fsdp_dtensor")` → `sharded_param_state_fsdp_dtensor()` | `get_state_dict(model, optimizer)` → calls `optimizer.state_dict()` internally |
| PP handling | `checkpointing.py` iterates model chunks manually | `get_state_dict` handles via `submodules`; caller iterates PP chunks |
| Loading skeleton | `is_loading=True` → `_init_optimizer_states_with_dummy_values()` | `get_state_dict` with pre-allocated optimizer states |
| Uneven DTensor | Done inside `preprocess_fsdp_dtensor_state_dict` | Done inside `get_state_dict` (via `preprocess_state_dict_for_uneven_dtensor`) — no separate step needed |

**Why use `uneven_dtensor.get_state_dict` instead of raw `torch.distributed.checkpoint.state_dict.get_state_dict`?**

`uneven_dtensor.get_state_dict` adds uneven DTensor chunk metadata in the same call.
With the raw PyTorch version, `preprocess_state_dict_for_uneven_dtensor` must be called
once by `preprocess_state_dict_for_uneven_dtensor`). Using `uneven_dtensor.get_state_dict`
consolidates this into a single well-tested entry point.

**Why separate the layers?**

1. **Testability** — Each layer can be tested independently. State dict correctness
   can be verified with simple models. MCore post-processing can be tested with
   synthetic state dicts.

2. **Reduced coupling** — `checkpointing.py` no longer needs to know about
   `sharding_type` or `sharded_param_state_fsdp_dtensor`. It just calls
   `get_state_dict`, applies post-processing, and hands to DCP.

3. **Online conversion is trivial** — The same `get_state_dict` + post-processing
   pipeline works for both save and online conversion (loading a legacy checkpoint into
   a v2 model). The only difference is the DCP operation (save vs load).

### 6.6 Prerequisites

For this approach to work, the following must be in place:

1. **`DistributedOptimizer.state_dict()` returns inner state dict** — Already implemented
   (Section 5.4). `get_state_dict` calls `optimizer.state_dict()` internally via
   PyTorch's native `get_state_dict`.

2. **`DistributedOptimizer.load_state_dict()` handles inner state dict** — Already
   implemented (Section 5.5). After DCP loads into the skeleton, we call
   `optimizer.load_state_dict()`.

3. **Optimizer states pre-allocated before load** — DCP does in-place loading into
   the skeleton's tensor buffers. The skeleton must have correctly-sized optimizer
   state tensors (exp_avg, exp_avg_sq) before DCP writes into them. Currently this
   is done by `_init_optimizer_states_with_dummy_values()` (called by
   `sharded_param_state_fsdp_dtensor` with `is_loading=True`). With Path B, this
   method must be called explicitly before `get_state_dict`.

4. **Model `load_state_dict` works** — Already implemented (Section 4.2).

5. **Post-processing functions are FSDP-agnostic** — `handle_fp8_extra_state_case`,
   `handle_swiglu_in_state_dict`, `handle_experts_in_state_dict` operate on dicts, not
   on model internals. They work for both DTensor and regular tensor state dicts.

6. **Uneven DTensor preprocessing handled by `get_state_dict`** — `uneven_dtensor.get_state_dict`
   calls `preprocess_state_dict_for_uneven_dtensor` on both model and optimizer state
   dicts. `preprocess_fsdp_dtensor_state_dict` also calls it. To avoid double-calling,
   `preprocess_fsdp_dtensor_state_dict` should skip the uneven DTensor step when the
   state dict already went through `get_state_dict`.

### 6.7 Migration Path

| Phase | Description |
|-------|-------------|
| **Phase 1** (now) | Keep current `generate_state_dict` path. Add `use_megatron_fsdp_v2` guards to `distrib_optimizer.py`. Wire `state_dict_for_save_checkpoint` in adapter. |
| **Phase 2** | Add `get_state_dict`-based path (from `uneven_dtensor.py`) as an alternative code path in `checkpointing.py`, gated by `use_megatron_fsdp_v2`. Verify round-trip correctness. |
| **Phase 3** | Deprecate `sharded_param_state_fsdp_dtensor` and the `is_loading` skeleton pattern for v2. Remove `state_dict_for_save_checkpoint` wiring from adapter. |

---

## 7. Online Checkpoint Conversion

### 7.1 Problem

Users may have checkpoints from legacy Megatron formats (ND-parallel with
`DistributedOptimizer`, or Megatron FSDP v1 baseline) and want to load them into a
Megatron FSDP v2 model. Key structures differ:

| Format | Model Keys | Optimizer Keys |
|--------|-----------|----------------|
| ND-parallel | `module.layer.weight` | Tensor-keyed (by param tensor id) |
| Megatron FSDP v1 baseline | `module.layer.weight` | Tensor-keyed |
| Megatron FSDP v2 | `module.layer.weight` | String-keyed (by param name) |

### 7.2 Solution: `_load_torch_dist_into_megatron_fsdp_v2`

The function `_load_torch_dist_into_megatron_fsdp_v2` in `checkpointing.py` is the entry
point for online conversion from `torch_dist` to `fsdp_dtensor` format. It is called
from `_load_global_dist_base_checkpoint` when `use_megatron_fsdp_v2` is set and the
source checkpoint uses `torch_dist` format.

The conversion proceeds in five phases (implemented in `load_torch_dist_into_fsdp_v2`):

#### Phase 1 — Preprocess & Verify v2 State Dict

``_preprocess_and_verify_v2_state_dict`` builds canonical maps of v2 model and
optimizer state entries.  Plain-tensor optimizer states are wrapped as uneven
DTensors sharing storage with the originals (dual-dict pattern, see Section 5.11).
Both ``__create_chunk_list__`` and ``__create_write_items__`` metadata are verified
on all model and optimizer DTensors.

#### Phase 2 — DCP Key Mapping

``_build_torch_dist_to_v2_map`` iterates torch_dist metadata keys and matches them
against canonical v2 entries: regular model weights, hi-precision (``param``)
optimizer copies, and optimizer state tensors (``exp_avg``, ``exp_avg_sq``).

Metadata keys from the torch_dist checkpoint are canonicalized for
**matching** against v2 entries, while the original torch_dist storage paths are kept
as DCP state-dict keys so they match the checkpoint metadata verbatim:

- **Model weights** (``model.<param_name>``): the ``model.`` prefix is stripped and
  shard suffixes (``/shard_X_Y`` on ``_extra_state`` entries) are removed.  The
  remaining name is canonicalized via ``normalize_torch_dist_key``
  (``transformer_layer`` → ``mtp_model_layer``) **only for matching**
  against the v2 model's canonical keys.  The **original** torch_dist name
  (without the ``model.`` prefix) is stored as the DCP load key.

- **Hi-precision optimizer copies** (``optimizer.state.param.<param_name>``): the
  ``optimizer.state.param.`` prefix is stripped.  The param name is canonicalized
  and ``module.`` prefix is stripped for matching; the original torch_dist name is
  used as the DCP key.  When both a regular model weight and a hi-prec optimizer
  copy map to the same v2 DTensor, the hi-prec copy takes priority.

- **Optimizer state tensors** (``optimizer.state.exp_avg.<param_name>``,
  ``optimizer.state.exp_avg_sq.<param_name>``): the state key and param name are
  extracted; the param name is canonicalized and ``module.`` prefix stripped for
  matching against the v2 optimizer's ``state`` dict
  (``v2_optim_state[canonical_name][state_key]``).  The full original
  torch_dist key (``optimizer.state.exp_avg.original_name``) is used as the DCP
  load key.

#### Phase 3 — Single DCP Load

A single ``dcp.load`` call loads all matched tensors:

```python
mapped_sd = {
    "model": {
        "decoder.layers.0.weight": v2_dtensor,    # regular weights
        ...
    },
    "optimizer": {
        "state": {
            "exp_avg": {
                "decoder.layers.0.weight": v2_opt_dtensor,
                ...
            },
            "exp_avg_sq": { ... },
            "param": {                              # hi-prec model copies
                "decoder.layers.0.weight": v2_dtensor,
                ...
            },
        }
    }
}
dcp.load(state_dict=mapped_sd, checkpoint_id=..., planner=DefaultLoadPlanner(allow_partial_load=True))
```

The DCP state dict mirrors the torch_dist checkpoint directory structure:
regular weights under ``model.``, and all optimizer-related tensors (including
hi-precision parameter copies) under ``optimizer.state.*``.

After loading, hi-precision model copies are merged back into the `model` subtree
so the model state dict is complete.

#### Phase 4 — Load Fused Layer / Expert Params by Slicing

Torch_dist stores multi-layer and multi-expert tensors as a single fused tensor (e.g.,
``decoder.layers.self_attention.linear_qkv.weight`` of shape ``(num_layers, H, 3*W)``).
FSDP v2 stores each layer as an individual DTensor
(``decoder.layers.0.self_attention.linear_qkv.weight`` of shape ``(H, 3*W)``).
DCP cannot split one tensor into many, so Phase 4 handles this in `load_torch_dist_into_fsdp_v2`.

Three tensor formats are supported:

1. **Regular fused** — shape ``(num_layers, ...)`` → sliced per layer index.
2. **GroupedMLP experts** — shape ``(num_layers, num_experts, ...)`` → sliced per
   ``(layer_idx, expert_idx)``.
3. **SequentialMLP experts** — shape ``(num_global_experts, ...)`` → sliced per global
   expert index (EP-aware, mapped to local expert via
   ``ep_rank * num_local + local_expert_idx``).

To avoid OOM when loading a large fused tensor (e.g., GroupedMLP with many layers and
experts), the fused buffer is created as a **Shard(0) DTensor** across the DP device
mesh:

```python
device_mesh = example_val.device_mesh
flat = torch.distributed.tensor.empty(
    fused_shape, dtype=example_val.dtype,
    device_mesh=device_mesh, placements=[Shard(0)],
)
dcp.load(state_dict={td_key: flat}, ...)
flat = redistribute_uneven_dtensor_to_replicated(flat).to_local()
```

- ``torch.distributed.tensor.empty(..., placements=[Shard(0)])`` distributes memory
  across DP ranks so no single rank allocates the full fused tensor.
- ``redistribute_uneven_dtensor_to_replicated`` gathers shards into a full local
  tensor for per-layer/per-expert slicing.
- After slicing, ``__create_chunk_list__`` metadata is used to copy each rank's
  DP-shard chunk with correct local offsets into the destination v2 DTensor.

#### Phase 5 — Strictness Verification

After all four loading phases complete, the v2 model's parameter names are compared
against the union of all loaded entries.  ``_extra_state`` entries are excluded (they
are FP8/FP4 metadata that reinitializes on load).  Optimizer state entries not matched
to torch_dist data are also reported.  Any unmatched parameter triggers a
``RuntimeError``, ensuring no weights are silently skipped.

### 7.3 Key Normalization Helpers

Located in `megatron/core/distributed/fsdp/checkpoint.py`:

| Function | Direction | Transforms |
|----------|-----------|------------|
| `normalize_torch_dist_key` | torch_dist → v2 | ``transformer_layer`` → ``mtp_model_layer`` |
| `reverse_normalize_torch_dist_key` | v2 → torch_dist | ``mtp_model_layer`` → ``transformer_layer`` |

These are used both in `_load_torch_dist_into_megatron_fsdp_v2` (for matching DCP
keys) and in Phase 4 of `load_torch_dist_into_fsdp_v2` (for constructing storage paths
that match the torch_dist checkpoint).

### 7.4 Strictness Checks

After all phases complete (see Phase 5 above), two strictness checks run:

1. **Model parameter coverage:** every v2 model canonical parameter name must appear
   in the set of loaded entries (regular model + hi-prec + expert params).
   ``._extra_state`` entries are excluded (FP8/FP4 metadata that reinitializes).

2. **Optimizer state coverage:** every v2 optimizer state parameter must have been
   matched to at least one torch_dist state tensor.  Unmatched entries trigger a
   ``RuntimeError``.

---

## 8. Edge Cases

### 8.1 Multi-Optimizer (MoE: Expert + Non-Expert)

MoE models use a `ChainedOptimizer` wrapping two `DistributedOptimizer` instances
(one for expert params, one for non-expert params). The proposed Path B must handle
this by building separate state dicts per sub-optimizer:

```python
# For ChainedOptimizer (expert + non-expert):
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import get_state_dict

state_dict["optimizer_expert"] = get_state_dict(model, optimizers=optimizer.expert)
state_dict["optimizer_non_expert"] = get_state_dict(model, optimizers=optimizer.non_expert)
```

The current `generate_state_dict` path handles this by calling `optimizer.sharded_state_dict()`
on the `ChainedOptimizer`, which internally dispatches to each sub-optimizer.

### 8.2 Frozen Sub-Models (`optimizer is None`)

When sub-models are frozen, the `DistributedOptimizer` is a stub (`is_stub_optimizer=True`).
`state_dict()` and `load_state_dict()` should be no-ops. The checkpoint should only contain
model weights, not optimizer state. This needs to be tested.

### 8.3 `param_to_group_meta` Under Path B

**Current Path A**: `sharded_param_state_fsdp_dtensor()` converts `self.optimizer.param_groups`
(tensor-keyed) to `param_to_group_meta` (name-keyed) for the checkpoint, and back on load.

**Path B**: `get_state_dict` calls `optimizer.state_dict()` which returns the inner
optimizer's param groups. These are **already in the standard PyTorch format** — no
conversion needed. The `param_to_group_meta` key format is a Megatron artifact that exists
only because Path A manually constructs the state dict. With Path B, `get_state_dict`
handles this natively, so `param_to_group_meta` is no longer needed.

This means `DistributedOptimizer.load_state_dict()` for Path B should accept the raw
`self.optimizer.state_dict()` format directly, bypassing the `param_to_group_meta`
conversion path. The current FSDP branch in `load_state_dict()` already handles this:
if `"param_to_group_meta"` is absent, it just calls `self.optimizer.load_state_dict(state_dict)`.

---

## 9. Uneven DTensor Handling

### 9.1 Problem

When a parameter's size is not evenly divisible by the DP world size, each rank
holds a different-sized local shard. Standard DCP assumes uniform shard sizes.

### 9.2 Solution: Chunk Metadata Patching

`preprocess_state_dict_for_uneven_dtensor()` walks the state dict, finds all
`DTensor` values, and calls `update_uneven_dtensor_chunk_metadata()` on each.
This function:

1. Gathers chunk metadata from all ranks via `all_gather_object`.
2. Computes global offsets and sizes for each rank's shard.
3. Patches the DTensor with `__create_chunk_list__` and `__create_write_items__`
   closures that DCP uses for serialization.

This is called on **both** model and optimizer state dicts.

### 9.3 Chunk Metadata Source Tag

Each DTensor's local tensor carries a `_chunk_meta_source` attribute that records
where its `__create_chunk_list__` metadata was set. This enables tracing the
provenance of chunk metadata when debugging size mismatches.

| Source Tag | Set By | When |
|---|---|---|
| `"init"` | `update_uneven_dtensor_chunk_metadata` (default) | FSDP v2 construction (`param_group.py`) |
| `"preprocess"` | `preprocess_state_dict_for_uneven_dtensor` | Save Phase 1 (re-computes via all_gather) |
| `"propagate:<src>"` | `copy_chunk_metadata` | Save Phase 2 (copied from model param; `<src>` is the original tag) |
| `"split"` | `split_dtensor` | Save Phase 3 (locally derived from parent's metadata) |
| `"make_uneven"` | `make_uneven_dtensor(chunk_metadata=...)` | Optimizer state wrapping as DTensors |

Use `get_chunk_meta_source(dtensor)` to read the tag (returns `"none"` if unset).

### 9.4 Final Consistency Check (`_verify_chunk_metadata`)

At the end of `_apply_mcore_postprocess`, `_verify_chunk_metadata` validates
every DTensor in the flattened state dict:

1. **Existence check** — every DTensor must have `__create_chunk_list__`.
2. **Numel consistency check** — `sum(prod(chunk.sizes)) == local_tensor.numel()`.
   Each chunk's sizes may be multi-dimensional (e.g., `(256, 512)`), so the
   element count is the product of all dimensions. A mismatch (e.g., 40960 vs
   2560) indicates chunk metadata was computed with the wrong device mesh
   (DP instead of EDP) or against stale state.
3. **On failure** — logs the key, global/local shapes, chunk list with offsets
   and sizes, source tag, and device mesh before raising `AssertionError`.
   This pinpoints which phase produced incorrect metadata and what mesh was used.

---

## 10. DTensor Attribute Propagation

When loading a checkpoint into a Megatron FSDP v2 model, parameters are DTensors.
Certain attributes (e.g., `is_embedding_parameter`, `allreduce`) set on the original
`nn.Parameter` objects by upstream layers (e.g., TE) must be propagated to the
DTensor wrappers. This is handled in `mcore_fsdp_adapter.py` at lines 307-331.
Missing attributes cause optimizer misclassification (`_get_param_groups`) and wrong
gradient scaling.

See `design.md` (Pitfall: Attribute Propagation) for the full list of attributes
and their consumers.

---

## 11. Key Differences from Legacy Path

| Aspect | Legacy Megatron FSDP | Megatron FSDP v2 |
|--------|---------------------|-------------------|
| Parameter representation | `MegatronFSDP`-managed DTensors | Native `FSDPModule` DTensors |
| Model wrapper | `MegatronFSDP` (adds `module.` prefix in state dict) | No `MegatronFSDP` wrap — `fully_shard` applied directly |
| Model state dict keys | Keys have `module.` prefix from `MegatronFSDP` wrapper | Keys lack `module.` prefix; `add_module_prefix()` used for checkpoint alignment |
| Model `state_dict_for_save_checkpoint` | `model.state_dict()` with `state_dict_pre_hook` | `model.state_dict()` (already DTensors) |
| Optimizer buffer management | Megatron FSDP managed | Standard Torch optimizer managed |
| Gradient buffer | Megatron FSDP `param_and_grad_buffer` | None (FSDP v2 handles internally) |
| Load model state dict | `module.load_state_dict(custom, strict)` with `_load_state_dict_post_hook` | `super().load_state_dict(state_dict, strict)` — after `strip_module_prefix()` |
| Zero gradient | `model_chunk.zero_grad_buffer()` | `model_chunk._zero_grad_buffer()` |

### 11.1 Model State Dict ``module.`` Prefix Alignment

**Problem:** Legacy Megatron FSDP wraps the model in a ``MegatronFSDP`` class which
stores the model as ``self.module``. This means ``MegatronFSDP.state_dict()`` produces
keys with a ``module.`` prefix (e.g., ``module.embedding.word_embeddings.weight``).
Megatron FSDP v2 applies ``fully_shard`` directly without a ``MegatronFSDP`` wrapper,
so the raw model's ``state_dict()`` produces keys **without** the prefix (e.g.,
``embedding.word_embeddings.weight``).

**Solution:** ``checkpoint.py`` provides two post-processing functions:

- ``add_module_prefix(state_dict)`` — adds ``module.`` prefix to all keys before
  saving, aligning with Megatron's checkpoint format.
- ``strip_module_prefix(state_dict)`` — removes ``module.`` prefix after loading,
  aligning with the FSDP v2 model's expected key format.

``get_model_state_dict(model)`` auto-detects whether the prefix is present and adds
it if missing. ``load_checkpoint`` strips the prefix back before calling
``model.load_state_dict()``.

**Note on optimizer keys:** ``sharded_param_state_fsdp_dtensor()`` uses
``model_chunk.named_parameters()`` to derive parameter names. When
``model_chunks`` are ``FullyShardedDataParallel`` instances (which store the model
as ``self.module``), the returned names already carry the ``module.`` prefix. This
ensures model and optimizer state dict keys are consistent in the checkpoint.

---

## 12. Implementation Checklist

### Phase 1: `distrib_optimizer.py` — FSDP v2 Guard Propagation

- [x] `__init__`: Guard early return with `use_megatron_fsdp or use_megatron_fsdp_v2`
- [x] `state_dict()`: Return `self.optimizer.state_dict()` for FSDP paths
- [x] `load_state_dict()`: Guard direct load path with `use_megatron_fsdp_v2`
- [x] `sharded_state_dict()`: Guard `fsdp_dtensor` enforcement with `use_megatron_fsdp_v2`
- [x] `sharded_param_state_fsdp_dtensor()`: Update assertion to accept both flags

### Phase 1: `mcore_fsdp_adapter.py` — Model State Dict

- [x] Wire `state_dict_for_save_checkpoint` to `module.state_dict()` in
      `_init_with_fully_shard()` (replace `not_implemented_op` at line 383-384)
- [x] Wire `self.state_dict_for_save_checkpoint` to `self.state_dict()` in
      `_init_with_fully_shard()` (replace `not_implemented_op` at line 383-384)

### Phase 2: Torch-Native `get_state_dict` Path (Path B — `MegatronFSDPStateful`)

- [x] Add `_is_megatron_fsdp_v2()` helper — checks ``ddp_config.use_megatron_fsdp_v2``
      on the ``FullyShardedDataParallel`` adapter, with fallback to ``isinstance(model[0], FSDPModule)``
      for models that went through ``fully_shard()`` without the adapter wrapper
- [x] Add `_build_megatron_fsdp_v2_state_dict()` to `checkpointing.py` using
      ``MegatronFSDPStateful`` (which internally uses ``get_state_dict`` from
      ``uneven_dtensor`` for both model and optimizer, then applies MCore
      post-processing)
- [x] Wire save path: when `_is_megatron_fsdp_v2(model)`, use `_build_megatron_fsdp_v2_state_dict`
- [x] Wire load path: same condition, pre-allocate optimizer states via
      `_init_optimizer_states_with_dummy_values()`, then use `_build_megatron_fsdp_v2_state_dict`
- [x] Skip `preprocess_fsdp_dtensor_state_dict` for v2 in both save and load paths
      (post-processing already handled by ``_apply_mcore_postprocess`` inside
      ``MegatronFSDPStateful.state_dict()``)
- [ ] Handle PP: iterate model chunks, build per-chunk state dicts (not yet required — all current tests use PP=1)
- [ ] Handle multi-optimizer (ChainedOptimizer: expert + non-expert optimizers) (not yet required — all current tests use single optimizer)

### Phase 2: Path A — Standalone Save/Load (`checkpoint.py`)

- [ ] Implement ``save_checkpoint(model, ckpt_dir, optimizer, args)`` — standalone DCP save
      helper for external usage (not currently needed by the main training loop).

### Path A Save/Load Flow (Final)

::

   _build_megatron_fsdp_v2_state_dict()
     |
     +-- get_model_state_dict(model[0])             # model DTensors
     +-- preprocess_state_dict_for_uneven_dtensor(model_sd)  # chunk metadata on params
     +-- get_optimizer_state_dict(optimizer)         # plain tensors (NO wrapping here)
     +-- _apply_mcore_postprocess(split_optimizer=False)  # model split, optimizer canonical
     +-- preprocess_state_dict_for_uneven_dtensor(full_sd)  # metadata on split model DTensors
     +-- Scheduler, Rerun, RNG
     |
   save_checkpoint()                        # SAVE path
     +-- _wrap_optim_states_as_dtensors(...)       # wrap plain → DTensor for DCP save plan
     +-- DCP save
     |
   _load_base_checkpoint()                  # LOAD path
     +-- raw_optimizer_state_dict = state_dict["optimizer"].copy()  # capture plain dict
     +-- _wrap_optim_states_as_dtensors(...)       # wrap plain → DTensor for DCP load
     +-- DCP load (fills DTensors → plain tensors updated via shared storage)
     +-- state_dict["optimizer"] = raw_optimizer_state_dict    # restore plain dict
     +-- optimizer.load_state_dict(state_dict["optimizer"])    # receives plain tensors

### Testing

- [x] Round-trip test: Save via MCore `save_checkpoint`, load via `setup_model_and_optimizer` → `load_checkpoint`
- [x] Optimizer state dict round-trip: save → load → verify values
      (``test_megatron_fsdp_v2_round_trip`` now validates both model and optimizer)
- [ ] Cross-format optimizer state conversion: ND-parallel → Megatron FSDP v2
- [ ] Unskip and fix hanging `get_state_dict` tests in `test_fully_shard.py`
- [ ] Uneven DTensor sharding with non-divisible parameter sizes
- [ ] Frozen parameter handling in `get_state_dict`
- [ ] PP + v2 checkpoint round-trip test
- [ ] Online conversion test with torch-native Path B

---

## 13. Testing Strategy

### Active Tests

| Test | File | Status |
|------|------|--------|
| Online convert: ND-parallel → Megatron FSDP v2 | `test_mcore_checkpoint.py` | Active |
| Online convert: FSDP v1 baseline → Megatron FSDP v2 | `test_mcore_checkpoint.py` | Active |
| Megatron FSDP v2 → v2 round-trip | `test_mcore_checkpoint.py` | Active |
| `MegatronFSDPStateful` wrapper | `checkpoint.py` | Active |
| `get_state_dict` strict DTensor assert | `test_fully_shard.py` | Active |
| SWiGLU/expert key transforms | `test_fsdp_dtensor_checkpoint.py` | Active |
| Megatron FSDP v2 E2E training | `test_mcore_nd_parallel.py` | Active |

### Known-Gap Tests

| Test | File | Status |
|------|------|--------|
| `get_state_dict` basic | `test_fully_shard.py` | Skipped (hangs) |
| `get_state_dict` nested FSDP | `test_fully_shard.py` | Skipped (hangs) |
| `get_state_dict` LLM scenario | `test_fully_shard.py` | Skipped (hangs) |
| `get_state_dict` frozen params | `test_fully_shard.py` | Skipped (hangs) |

### Test Gaps

- Optimizer state dict round-trip for Megatron FSDP v2.
- Cross-format optimizer state conversion (ND-parallel optimizer → Megatron FSDP v2).
- Uneven DTensor sharding with non-divisible parameter sizes.
- Frozen parameter handling in `get_state_dict`.

---

## 14. Debugging Checklist

1. **Model state dict has DTensors?** — All parameters should be `DTensor` after
   `fully_shard()`:
   ```python
   from torch.distributed.tensor import DTensor
   for name, p in model.named_parameters():
       assert isinstance(p, DTensor), f"{name}: expected DTensor, got {type(p)}"
   ```

2. **Checkpoint save succeeds?** — DCP writes to disk. Check for
   `FileSystemWriter` or `FileSystemWriterAsync` in logs.

3. **Checkpoint load succeeds?** — Verify that `dcp_load()` returns without errors.
   Set `strict_fsdp_dtensor_load=True` for strict key matching.

4. **Model parameters match after load?** — Compare source and loaded state dicts
   using `_state_dict_to_full_tensor` helper.

5. **Optimizer state restored?** — Check that `optimizer.state` is non-empty and
   contains expected keys (`exp_avg`, `exp_avg_sq`, `step`).

6. **Chunk metadata present?** — After save, verify `__create_chunk_list__` exists
   on all DTensors and the state dict passed `_verify_chunk_metadata` without error.
   Check the log for `[chunk_metadata_diag]` warnings about unmatched DTensors.

7. **Chunk metadata consistent?** — If DCP reports `invalid fill tensor-volume`,
   inspect the `[chunk_metadata_verify]` error log. It shows global/local shapes,
   chunk offsets/sizes, the source tag (which phase set the metadata), and the
   device mesh. Common causes:
   - `source=preprocess` + wrong device mesh → `all_gather_object` computed
     offsets against DP mesh when parameter is sharded across EDP mesh.
   - `source=propagate:init` + key mismatch → metadata was copied from a
     different parameter (or not copied at all, falling back to Phase 1 metadata).
   - `source=split` + wrong split → fused param split derived wrong local
     offsets from parent's chunk metadata.

8. **NaN after load?** — Common causes:
   - Missing `allreduce` attribute propagation (expert params misclassified)
   - Missing `overwrite_main_grad=True` for wgrad fusion (gradient doubling)
   - Wrong `gradient_accumulation_fusion` setting

---

## 15. Future Work

- **Async checkpoint**: Integrate `FileSystemWriterAsync` for non-blocking saves.
- **Cross-topology resharding**: Support loading checkpoints saved with a different
  DP world size (requires DCP's resharding planner).
- **Unified `set_state_dict`**: Add a `set_state_dict()` wrapper in
  `uneven_dtensor.py` that handles preprocessing inverse.
- **Complete skipped tests**: Debug and unskip the `test_fully_shard.py` checkpoint
  tests that currently hang.
