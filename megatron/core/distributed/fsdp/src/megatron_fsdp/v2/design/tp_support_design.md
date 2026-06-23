# Tensor Parallelism Support in Megatron FSDP v2 — Design Document

## 1. Overview

This document proposes adding Tensor Parallelism (TP) support to Megatron FSDP v2.
Currently v2 operates on a 1D DP-only DeviceMesh and does not handle parameters
that are already sharded by TP layers (e.g., `ColumnParallelLinear`,
`RowParallelLinear`). This document describes how to compose TP and FSDP sharding
in v2, referencing the proven patterns from v1.

## 2. Background

### 2.1 How v1 Handles TP

Megatron FSDP v1 composes TP and FSDP using a **two-step DTensor construction**
in `make_fsdp_dtensor()` (`param_and_grad_buffer.py:4553–4712`):

**Step 1 — TP-sharded DTensor.** If a parameter has a `_tensor_parallel_mode`
attribute (`"column"` or `"row"`), a TP-only DTensor is constructed first:

```python
tp_mesh = dist_index.get_submesh("tp")
placements = [Shard(partition_dim)]  # dim=0 for column, dim=1 for row
global_shape[partition_dim] *= tp_mesh.mesh.numel()
param = DTensor.from_local(local_tensor, tp_mesh, placements, shape=global_shape, ...)
```

**Step 2 — FSDP-sharded DTensor.** The TP DTensor is then wrapped in an
additional FSDP placement, composing a multi-dimensional DTensor:

```python
device_mesh = dist_index.get_submesh(("dp_cp", "tp"))   # 2D mesh
placements = [Shard(0), Shard(tp_dim)]                   # DP shard + TP shard
param = DTensor.from_local(param.to_local(), device_mesh, placements, ...)
```

The key insight: **FSDP v1 communicates gradients ONLY on the DP group**.
TP communication (all-reduce, all-gather, reduce-scatter) is handled
independently by the TP layers themselves (`tensor_parallel/layers.py`).
The 2D DTensor representation is purely for sharding semantics — the
communication operations are decoupled.

### 2.2 Current v2 State

v2 currently has no TP awareness:

| Component | v2 Status |
|-----------|-----------|
| DeviceMesh | 1D DP only (`v2/utils.py:67–82`) |
| Placements | `[Shard(0)]` or `[Replicate()]` only (`v2/param_group.py:306`) |
| `fully_shard()` API | No TP mesh parameter; `shard_placement_fn` is TODO |
| Adapter mesh | `_init_dp_mesh()` creates DP-only mesh (`mcore_fsdp_adapter.py:688`) |
| TP annotation | Not consumed; `_annotate_tensor_parallelism` exists in adapter |
| Gradient ops | Reduce-scatter on DP group only (correct, no change needed) |

### 2.3 Why TP + FSDP Composition Matters

When TP is active, each TP rank holds a shard of the weight. For example, with
`tensor_model_parallel_size=4`:

- `ColumnParallelLinear.weight`: shape `[out_per_partition, in]` — sharded along dim 0
- `RowParallelLinear.weight`: shape `[out, in_per_partition]` — sharded along dim 1

v2 currently wraps these **partial** tensors with `[Shard(0)]` on a DP mesh,
which misrepresents the true global shape: the DTensor reports the TP-partial
shape as "global," and checkpoint save/load produces incorrect shard metadata.

Correct behavior: the DTensor should carry a 2D `(dp, tp)` mesh with placements
`[Shard(0), Shard(tp_dim)]` and the full global shape.

## 3. Proposed Design

### 3.1 High-Level Approach

Follow v1's two-step DTensor construction, adapted for v2's `ParameterGroup` and
`DataParallelBuffer` architecture:

1. **Annotate:** Reuse `_annotate_tensor_parallelism` from v1 to mark params
2. **Construct 2D mesh:** Build `(dp, tp)` DeviceMesh in the adapter or directly
   in `fully_shard()`
3. **Two-step DTensor:** In `ParameterGroup._init_buffers()`, wrap TP-sharded
   params first, then apply FSDP shard
4. **No gradient changes:** FSDP reduce-scatter remains DP-only (correct)
5. **Checkpoint metadata:** Leverage existing uneven DTensor infrastructure

### 3.2 Phase 1: Standalone `fully_shard()` with TP Mesh (Minimal Viable)

Add a `tp_mesh` parameter to `fully_shard()`:

```python
def fully_shard(
    module,
    *,
    mesh: Optional[DeviceMesh] = None,
    tp_mesh: Optional[DeviceMesh] = None,   # NEW
    ...
) -> nn.Module:
```

When `tp_mesh` is provided:
- **In `_init_named_param_groups()`:** For each parameter, if it belongs to a
  TP-sharded module (detected via `_tensor_parallel_mode` or module type),
  construct placements as `[Shard(0), Shard(tp_dim)]` on a combined 2D mesh
- **In `dp_buffer.py`:** `BufferIndex.shard_meta` must account for the TP-local
  size (the "logical" size visible to this rank after TP sharding) vs. the
  "storage" size (the FSDP shard size). The `unshard()` all-gather returns the
  TP-local full tensor, not the true global tensor — this is correct because
  TP layers operate on their local TP shard during forward/backward.

**TP-local vs. global semantics:**

```
True global shape:    [out, in]
TP-local shape:       [out/tp, in]  (column-parallel)
FSDP shard shape:     [out/tp/dp, in]  (sharded across DP ranks)

DTensor shape:        [out, in]       ← global shape annotated
DTensor mesh:         (dp, tp)
DTensor placements:   [Shard(0), Shard(0)]  ← DP shard then TP shard
```

The `DTensor.to_local()` returns the TP-local+FSDP-local shard — exactly the
data that PyTorch's autograd operates on during forward/backward.

### 3.3 Phase 2: Integration with MCore Adapter

In `mcore_fsdp_adapter.py`, extend `_init_with_fully_shard()` to:

1. Call `_annotate_tensor_parallelism()` (already exists at line 434–502)
2. Build a 2D `(dp, tp)` DeviceMesh via `_get_dp_tp_mesh()` (already exists at
   line 783) instead of the current `_init_dp_mesh()`
3. Pass the 2D mesh + tp_mesh to `fully_shard()`
4. The adapter's `_init_distributed_params()` path for v2 should use the
   same two-step DTensor construction

### 3.4 Phase 3: Bucketing with TP

v1's `BucketingPolicy` does NOT consider TP when computing buffer sizes — it
only uses DP shard sizes. v2 should do the same: buffers are sized for the
TP-local parameter shape (after TP sharding), not the global shape. This is
correct because:

- `unshard()` all-gathers to the TP-local full tensor (not global)
- `reduce_grad()` reduce-scatters from the TP-local full gradient (not global)
- TP layers independently handle their own communication

### 3.5 Gradient Communication (No Change)

v2's gradient reduce-scatter already operates on the DP group only — this is
correct and should not change for TP support. TP parameters participate in
TP-level all-reduce/reduce-scatter within the TP layer's forward/backward, and
in DP-level reduce-scatter within v2's `reduce_grad()`. The two are independent
and naturally compose.

### 3.6 Checkpointing with TP

The uneven DTensor infrastructure (`uneven_dtensor.py`) already supports
multi-dimensional shard metadata via `_shard_order`. v2's checkpoint path
(through `get_state_dict()` → `preprocess_state_dict_for_uneven_dtensor()`)
should work correctly once DTensors carry the 2D mesh, because:

- Each DTensor knows its own shard placement and global shape
- DCP (`torch.distributed.checkpoint`) natively handles multi-dimensional
  DTensor serialization
- Chunk metadata computation (`gather_and_compute_chunk_metadata()`) supports
  arbitrary `_shard_order` values

## 4. Implementation Checklist

### Phase 1: Standalone TP (3–5 days)

- [ ] Add `tp_mesh` parameter to `fully_shard()` signature
- [ ] In `ParameterGroup._init_buffers()`: detect TP params and construct 2D mesh + placements
- [ ] In `ParameterGroup`: add `_tp_partition_dim()` helper (delegate to
  `get_mcore_tensor_parallel_partition_dim`)
- [ ] In `BufferIndex`: ensure `shard_meta` correctly reflects TP-local size
- [ ] Unit test: `test_tp_column_parallel()` — ColumnParallelLinear + fully_shard + forward/backward
- [ ] Unit test: `test_tp_row_parallel()` — RowParallelLinear + fully_shard + forward/backward
- [ ] Unit test: `test_tp_fsdp_loss_identity()` — compare FSDP+TP loss vs DDP baseline

### Phase 2: MCore Adapter Integration (2–3 days)

- [ ] Wire `_annotate_tensor_parallelism()` in adapter v2 path
- [ ] Build 2D `(dp, tp)` mesh in `_init_with_fully_shard()`
- [ ] Pass TP mesh and annotations through `fully_shard()` calls
- [ ] Integration test: `gpt3_mcore_tp2_pp1` config with `--use-megatron-fsdp-v2`

### Phase 3: Checkpoint + Polish (2–3 days)

- [ ] Verify `get_state_dict()` returns correct global shapes for TP+FSDP params
- [ ] Test checkpoint save/load round-trip with TP active
- [ ] Test online format conversion: torch_dist → fsdp_dtensor with TP
- [ ] Handle `force_sync_tp_duplicated_param` for replicated TP params (e.g.,
  RowParallel bias, LayerNorm) — broadcast from tp_rank=0
- [ ] Update README with TP usage example

## 5. Replicated TP Parameters

Some parameters in TP modules are not sharded (e.g., `RowParallelLinear.bias`,
`LayerNorm.weight`). These have `_tensor_parallel_mode = "replicated"` in v1's
annotation system. For these params:

- **DTensor type:** `Replicate()` on the TP dimension
- **Synchronization:** v1 broadcasts from `tp_rank=0` during init to ensure
  consistency. v2 should do the same via `force_sync_tp_duplicated_param`
- **FSDP handling:** Normal FSDP shard — these params have the full unpartitioned
  shape, so v2's default logic applies correctly

Combined placements for replicated TP + FSDP: `[Shard(0), Replicate()]`

## 6. Risks and Open Questions

1. **`DTensor.full_tensor()` behavior with 2D mesh.** Calling `.full_tensor()`
   on a `(dp, tp)` DTensor all-gathers across BOTH dimensions. v2's `unshard()`
   should only all-gather across the DP dimension — never across TP. Solution:
   use `DTensor.redistribute(placements=[Replicate(), Shard(tp_dim)])` instead
   of `.full_tensor()`, or construct a 1D DP submesh for the all-gather.

2. **Stream ordering with TP.** TP layers perform their own communication on the
   default stream. v2's `ag_stream` / `rs_stream` operations must be correctly
   ordered relative to TP comms. v1 handles this with `stream.wait_stream()`
   barriers; v2's `_FSDPRootContext` already has this mechanism.

3. **HSDP + TP (3D mesh).** v1 supports `(outer_dp, inner_dp, tp)` 3D meshes.
   v2 should defer HSDP+TP support to a follow-up; the design is extensible.

4. **Expert parallelism with TP.** MoE experts may have their own EP mesh
   dimension. v1 handles this via `is_expert_parallel` parameter in
   `get_submesh()`. v2 should similarly extend its param group logic.

## 7. v1 Code References

| v1 Component | File | Lines |
|-------------|------|-------|
| `make_fsdp_dtensor()` | `param_and_grad_buffer.py` | 4553–4712 |
| `_get_fsdp_tensor_spec()` | `param_and_grad_buffer.py` | 4477–4550 |
| `FSDPDistributedIndex` | `utils.py` | 475–673 |
| TP helper functions | `utils.py` | 819–850 |
| `_annotate_tensor_parallelism()` | `mcore_fsdp_adapter.py` | 434–502 |
| `_get_dp_tp_mesh()` | `mcore_fsdp_adapter.py` | 783 |
| 2D/3D mesh construction | `mcore_fsdp_adapter.py` | 504–633 |
| Gradient communication (DP-only) | `param_and_grad_buffer.py` | 3168–3268 |
| `_init_optimizer_named_parameters` (TP attr propagation) | `param_and_grad_buffer.py` | 2872–2926 |
| Chunk metadata with `_shard_order` | `uneven_dtensor.py` | 32–97 |
