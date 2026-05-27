# NVFP4 Support in Megatron FSDP v2 — Design Document

## 1. Overview

This document proposes adding NVFP4 primary-weights support to Megatron FSDP v2
(the `fully_shard` API path under `megatron_fsdp/v2/`). The existing
non-FSDP DDP path (`DistributedDataParallel` + `DistributedOptimizer`) already
supports `--fp4-param-gather` — this design mirrors that support into the v2
FSDP path, following the same patterns used for FP8 support.

## 2. Current State

### 2.1 What already works (non-FSDP DDP path)

- Model weights initialised as `NVFP4Tensor` (packed uint8: 2 FP4 values/byte, `numel_packed = numel_logical // 2`).
- `_ParamAndGradBuffer` maintains dual index maps:
  - `param_index_map` — full-numel offsets for grads
  - `nvfp4_packed_param_index_map` — packed offsets for the param buffer
- All-gather communicates packed uint8 bytes; gradients use full-numel FP32.
- After optimizer step: `quantize_nvfp4_param_shard()` calls TE's
  `quantize_master_weights` to cast FP32 → NVFP4 in-place in the packed buffer.

### 2.2 What already works (FSDP v2 FP8 path)

- `FullyShardMixedPrecisionPolicy` (in `v2/mixed_precision.py`) handles
  FP8 / MXFP8 detection, buffer dtype, raw-data extraction, post-unshard
  processing, post-reshard cache invalidation, and main→model quantization.
- `ParameterGroup` creates 3–4 `DataParallelBuffer` instances:
  `model_weight_buffer` (uint8 for FP8), optional `transpose_weight_buffer`,
  `main_weight_buffer` (FP32), and `main_grad_buffer` (FP32).
- All buffers share the same `BufferIndex` layout (same params, same shapes).

### 2.3 What is missing

No NVFP4 path exists in any file under `megatron_fsdp/`. The v1 `MegatronFSDP`
wrapper also lacks it. The `DistributedDataParallelConfig` for FSDP
(`megatron_fsdp/distributed_data_parallel_config.py`) has no `fp4_param_gather`
field, and `mcore_fsdp_adapter.py` has no translation of `fp4_param_gather` into
the v2 mixed-precision policy.

## 3. Key Difference from FP8: Packed Storage

| Aspect | FP8 (Float8Tensor) | NVFP4 (NVFP4Tensor) |
|--------|--------------------|----------------------|
| Storage dtype | `uint8` | `uint8` (packed) |
| Bytes per logical element | 1 | 0.5 |
| Packed numel | `N` (same as logical) | `N // 2` |
| Internal raw attr | `_rowwise_data` / `_data` | `_rowwise_data` |
| Transpose/columnwise | MXFP8 has `_columnwise_data` | None |
| Transpose cache | Yes (needs invalidation) | No |
| Quantization API | `cast_master_weights_to_fp8` | `quantize_master_weights` (same TE function, different tensor type) |

The **packed storage** is the central design challenge: the `model_weight_buffer`
needs a `BufferIndex` with **packed shapes** (`shape[-1] // 2`), while
`main_weight_buffer` and `main_grad_buffer` need **full shapes**.

## 4. Design Decisions

### 4.1 Separate BufferIndex per buffer (no shared layout)

Each `DataParallelBuffer` already owns its own `BufferIndex`. For NVFP4
param groups, the model-weight buffer will use packed shapes in its index,
while main-weight and main-grad buffers use the standard full shapes.

**Why this is clean**: `param_idx` (mapping `param → position_in_list`) stays
consistent across all three buffers because it only depends on parameter
ordering. Each buffer's `BufferIndex` translates the same `param_idx` into
different byte offsets appropriate for that buffer's dtype/shape.

### 4.2 Policy-driven shape transform

The `FullyShardMixedPrecisionPolicy` already owns dtype and raw-data decisions.
We extend it with an `nvfp4` sub-policy (analogous to `fp8`) that:

- Returns `torch.uint8` from `model_weight_buffer_dtype()`.
- Returns packed shapes from a new `get_param_storage_shapes()`.
- Returns `_rowwise_data` from `get_param_data()`.
- Calls `post_all_gather_processing()` in `post_unshard()`.
- Invokes `quantize_master_weights` (via `quantize_nvfp4_param_shard`) in
  `copy_main_weights_to_model_weights()`.

### 4.3 No transpose/cache management needed

NVFP4 has no columnwise data or transpose cache. `needs_transpose_weight_buffer()`
already returns `False` for non-MXFP8 tensors. No changes needed there.

## 5. Detailed Implementation Plan

### 5.1 `megatron_fsdp/v2/mixed_precision.py`

**Add NVFP4 detection** (following the FP8 pattern at lines 28–63):

```python
try:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor as NVFP4_TENSOR_CLASS
    HAVE_TE_NVFP4 = True
except Exception:
    NVFP4_TENSOR_CLASS = None
    HAVE_TE_NVFP4 = False
```

**Add `FullyShardNVFP4Policy` dataclass** (analogous to `FullyShardFP8Policy`):

```python
@dataclass(frozen=True)
class FullyShardNVFP4Policy:
    enabled: bool = False
    recipe: Optional[str] = None
```

**Extend `FullyShardMixedPrecisionPolicy`** with `nvfp4` field:

```python
@dataclass(frozen=True)
class FullyShardMixedPrecisionPolicy:
    ...
    fp4_param_gather: bool = False
    fp4_recipe: Optional[str] = None
    nvfp4: Optional[FullyShardNVFP4Policy] = field(default=None, repr=False)
    ...
```

**`__post_init__`**: initialize `nvfp4` from flat fields or vice versa (same pattern
as FP8, lines 160–172).

**New / modified methods**:

| Method | Change |
|--------|--------|
| `model_init_context()` | Enter `fp8_model_init(NVFP4BlockScaling)` when NVFP4 enabled |
| `group_key_dtype()` | Return `("quantized", "NVFP4Tensor", recipe)` for NVFP4 params |
| `is_nvfp4_param()` | New: `isinstance(tensor, NVFP4_TENSOR_CLASS)` |
| `model_weight_buffer_dtype()` | Return `torch.uint8` for NVFP4 (same as FP8) |
| `get_param_storage_shapes()` | **New**: Return packed shapes for NVFP4, original shapes otherwise |
| `get_param_data()` | Return `tensor._rowwise_data` (packed) for NVFP4 |
| `bind_unsharded_param()` | Set `_rowwise_data` to all-gathered buffer view (same pattern as FP8 `_data`) |
| `get_high_precision_value()` | Use `dequantize()` or preserved init val |
| `post_unshard()` | Call `post_all_gather_processing()` for NVFP4 params (same as FP8 rowwise) |
| `post_reshard()` | No-op for NVFP4 (no transpose cache to clear) |
| `copy_main_weights_to_model_weights()` | Call `quantize_nvfp4_param_shard()` (from `fp4_utils.py`) |
| `storage_tensors_to_free()` | Free `_rowwise_data` for NVFP4 (no transpose) |
| `needs_transpose_weight_buffer()` | Already returns `False` for non-MXFP8 — no change |
| `fine_grained_forward_hooks_required()` | Return `True` if any NVFP4 params present |
| `weight_buffers_for_unshard()` | Return `[model_weight_buffer]` (no transpose) |
| `validate_param_group()` | Works as-is (grouping key already differentiates NVFP4) |

### 5.2 `megatron_fsdp/v2/param_group.py`

**`_init_buffers()`** — unchanged.  Buffer creation passes the shared logical
`chunk_size_factor` through `_create_buffer()`.  Shape resolution is handled
in `DataParallelBuffer.__init__()` and `BufferIndex.compact()`.

### 5.3 `megatron_fsdp/v2/dp_buffer.py`

**`BufferIndex.compact(factor, compact_shapes)`** — **new method** that proportionally
scales all indices for packed storage:

```python
def compact(self, factor: float, compact_shapes: List[torch.Size]) -> None:
    for item_id, item in self.item_index_map.items():
        new_map[item_id] = ItemIndex(
            global_data_index=int(item.global_data_index * factor),
            size=int(item.size * factor),
            item_id=item.item_id,
            shape=compact_shapes[item_id],
        )
    self.item_index_map = new_map
    self.bucket_meta = BucketMeta(
        global_data_index=0,
        size=int(self.bucket_meta.size * factor),
        items=list(new_map.values()),
    )
    self.shard_meta = self._build_shard_meta(...)
```

For NVFP4, ``factor = 0.5`` and ``compact_shapes`` come from
``get_param_storage_shapes()``.  This preserves the proportional item-offset
mapping between buffers while eliminating fragment-binning waste.

**`DataParallelBuffer.__init__()`** — always builds with logical shapes, then compacts
NVFP4 weight buffers:

```python
_logical_shapes = [p.shape for p in params]
self.buffer_index = BufferIndex(
    param_shapes=_logical_shapes,
    chunk_size_factor=chunk_size_factor,
    ...,
)
if buffer_role in ("model_weight", "transpose_weight") and any(mp_policy.is_nvfp4_param(p) for p in params):
    self.buffer_index.compact(0.5, mp_policy.get_param_storage_shapes(params))
```

Main-weight and main-grad buffers are never compacted — they keep the
logical layout and the original ``chunk_size_factor``.

### 5.4 `megatron_fsdp/v2/fully_shard.py`

No changes needed — parameter replacement and hook registration are dtype-agnostic.

### 5.5 `megatron_fsdp/v2/hooks.py`

No changes needed.

### 5.6 `megatron_fsdp/v2/fsdp_module.py`

No changes to `FSDPModule`. The parameter grouping logic in
`_get_module_fsdp_param_groups()` already calls `mp_policy.group_key_dtype()`
which will differentiate NVFP4 params from others.

### 5.7 `megatron_fsdp/distributed_data_parallel_config.py`

Add `fp4_param_gather: bool = False` field (mirroring `fp8_param_gather` at line 37):

```python
fp4_param_gather: bool = False
"""If true, keep the compute param in fp4 (do not use any other intermediate dtype) and
   perform the param all-gather in fp4."""
```

### 5.8 `megatron/core/distributed/distributed_data_parallel_config.py`

Already has `fp4_param_gather: bool = False` at line 75. No change needed.

### 5.9 `mcore_fsdp_adapter.py`

**`_init_with_fully_shard()`** — pass FP4 configuration into the policy:

```python
fully_shard_mp_policy = FullyShardMixedPrecisionPolicy(
    ...
    nvfp4=FullyShardNVFP4Policy(
        enabled=ddp_config.fp4_param_gather,
        recipe=config.fp4_recipe,
    ),
)
```

Also pass `fp4_param_gather` to the v1 `MegatronFSDP` path if desired (similar
to FP8 wiring at line 257–261).

### 5.10 `megatron/training/arguments.py`

The `--fp4-param-gather` argument already exists (line 1816). Verify it is
propagated into the FSDP config when `--use-megatron-fsdp` is set. Currently
`DistributedDataParallelConfig` is auto-populated from `args` fields (line 1457
in `training.py`), so adding `fp4_param_gather` to the FSDP sub-config dataclass
(step 5.7) will make it flow automatically.

### 5.11 `megatron/core/fp4_utils.py`

No changes needed. Existing functions (`is_nvfp4tensor`, `quantize_nvfp4_param_shard`,
`get_nvfp4_rowwise_packed_shape`) will be imported by the modified mixed_precision.py.

## 6. Data Flow Summary

```
[Model Init] fp8_model_init(NVFP4BlockScaling)
  → TE creates NVFP4Tensor params (packed uint8, shape[-1] // 2)
       │
       ▼
[FSDP v2 ParameterGroup._init_buffers()]
  model_weight_buffer:  DataParallelBuffer(uint8, packed shapes, distributed)
  main_weight_buffer:   DataParallelBuffer(fp32, full shapes, distributed)
  main_grad_buffer:     DataParallelBuffer(fp32, full shapes, distributed)
       │
       ▼
[Forward: unshard()]
  1. All-gather packed uint8 from model_weight_buffer shards
  2. bind_unsharded_param(): rebind NVFP4Tensor._rowwise_data → full buffer view
  3. post_unshard(): post_all_gather_processing() rebuilds TE caches
       │
       ▼
[Backward: reduce_grad()]
  1. Copy param.grad → main_grad_buffer (fp32, full numel)
  2. reduce_scatter_tensor() on main_grad_buffer
       │
       ▼
[Optimizer step]
  Optimizer updates main_weight_buffer (fp32)
       │
       ▼
[copy_main_weights_to_model_weights()]
  quantize_nvfp4_param_shard(): fp32 main → packed NVFP4 model buffer
  (same TE quantize_master_weights API used by non-FSDP path)
```

## 7. Files to Modify

| File | Change |
|------|--------|
| `megatron_fsdp/v2/mixed_precision.py` | Add `FullyShardNVFP4Policy`, NVFP4 detection, all policy methods |
| `megatron_fsdp/v2/param_group.py` | Pass logical `chunk_size_factor` (computed from ``p.shape``) to all buffers |
| `megatron_fsdp/v2/dp_buffer.py` | Add `BufferIndex.compact()`; all buffers build with logical shapes, only NVFP4 weight buffers are compacted |
| `megatron_fsdp/distributed_data_parallel_config.py` | Add `fp4_param_gather` field |
| `mcore_fsdp_adapter.py` | Wire `fp4_param_gather` + `fp4_recipe` into policy |
| `megatron_fsdp/v2/__init__.py` | Export new NVFP4 policy class |

Files that do **not** need changes: `fp4_utils.py`, `hooks.py`, `fully_shard.py`,
`fsdp_module.py` (except possibly fine-grained hook gating).

## 8. Risks and Edge Cases

1. **Packed buffer alignment**: The `BufferIndex._build_layout()` pads to
   `dp_world_size * chunk_size_factor`.  `chunk_size_factor` is computed from
   logical `p.shape[1:].numel()` so that it works correctly for all buffers
   (model-weight, main-weight, main-grad).  Only the model-weight buffer uses
   packed shapes in its `BufferIndex`; the main buffers use logical shapes.

2. **Empty shards**: NVFP4 params with odd inner dimensions are rejected by
   TE (assertion in `get_nvfp4_rowwise_packed_shape`). No special handling needed.

3. **Mixed NVFP4 + non-NVFP4 params in same group**: Prevented by
   `group_key_dtype()` returning different keys.

4. **Gradient accumulation fusion**: The `overwrite_main_grad` and
   `grad_added_to_main_grad` flags set in `pre_backward_hook` work on
   `ParameterGroup` params regardless of their storage dtype.

5. **Checkpoint save/load**: The `state_dict` hooks and DTensor-based checkpoint
    path in `mcore_fsdp_checkpoint_design.md` should work transparently since
    the DTensor views point into the main_weight_buffer (fp32), not the packed
    model-weight buffer.

## 9. BufferIndex API (Standardized)

The `BufferIndex` class provides three coordinate domains for querying item
positions.  All method names use a consistent `_range` suffix and clear domain
prefixes:

| Method | Domain | Returns | Description |
|--------|--------|---------|-------------|
| `_get_item_self_range(item_id)` | **Item-self** | `(start, end)` | Offset within the item itself (0 = start of item). What portion of this item falls in the current rank's shard. |
| `_get_item_local_range(item_id, *, as_shard=False)` | **Local buffer** | `(start, end)` | Byte offsets within `self.data` (the local GPU buffer). Where to read/write. The `as_shard` flag forces shard-intersection computation even for non-distributed buffers. |
| `_get_item_global_range(item_id)` | **Global buffer** | `(global_offset, size)` | Position and size in the full logical (unsharded) buffer. Same on all ranks. |

`_get_item_local_range` absorbs the old `_get_item_local_index` and
`_get_item_local_shard_index`.  The `as_shard` kwarg unifies what was
previously the `only_shard` flag on `get_item()`:

```python
def get_item(self, item_id, *, as_shard=False):
    start, end = self.buffer_index._get_item_local_range(item_id, as_shard=as_shard)
    return self.data[start:end]
```

## 10. DataParallelBuffer.summon_full_params (planned)

> **Not yet implemented.** Context manager that temporarily unshards a
> buffer, then automatically reshards on exit:

```python
@contextmanager
def summon_full_params(self, async_op=False):
    need_unshard = self.is_distributed and not self.is_unsharded()
    if need_unshard:
        self.unshard(async_op=async_op)
    try:
        yield
    finally:
        if need_unshard:
            self.reshard()
```

Only unshards/reshards if the buffer is actually distributed and not
already unsharded.  Non-distributed buffers are a transparent no-op.
