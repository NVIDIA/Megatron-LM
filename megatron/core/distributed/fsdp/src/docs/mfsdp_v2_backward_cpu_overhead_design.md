# MFSDP v2 Backward Gradient Path — CPU Overhead Reduction Design

## 1. Problem

nsys profile of the M-FSDP v2 + MXFP8 PP3/VPP2/EP8 run (Lyris GB200, pd1)
shows ~10 ms of CPU time per backward under the `MFSDP <root> backward`
NVTX tag, spent in:

- **`copy_`** — packing per-parameter gradients into the reduce-scatter input
  buffer (`copy_gradients_to_partial_buffer`, `parameter_group.py`)
- **`_FromTorchTensor`** — DTensor construction per parameter per backward
  (`get_dtensor` / `DBuffer.from_local` in the grad-reduction path and the
  `.grad` rebind in `reduce_partial_gradients`)

These are **host-side** costs: per-microbatch per-parameter tensor ops that
serialize the backward and sit on the critical path of the 1F1B schedule.
Each contributes `O(num_params_in_group)` Python/ATen dispatches per
microbatch.

## 2. Root-cause analysis

### 2.1 `copy_` — gradient packing (parameter_group.py)

```python
def copy_gradients_to_partial_buffer(self, partial_grad: DBuffer) -> None:
    for index, fsdp_parameter in enumerate(self.fsdp_parameters):
        parameter = self._get_unsharded_parameter(index)
        partial_grad.get_local_tensor(index).copy_(parameter.grad)
        parameter.grad = None
```

Per group per backward: one `copy_` per parameter. With a transformer layer
grouped by `(dtype, requires_grad, is_fp8)`, an MoE layer has ~10-20
parameters per group → 10-20 `copy_` dispatches per backward per group,
each with its own kernel launch, stream sync, and tensor-arg validation.

### 2.2 `_FromTorchTensor` — DTensor creation (dbuffer.py)

`reduce_partial_gradients` (parameter_group.py) rebinds every sharded
parameter's `.grad`:

```python
for index, fsdp_parameter in enumerate(self.fsdp_parameters):
    fsdp_parameter.sharded.grad = self.main_grad.get_dtensor(index)
```

`get_dtensor` calls `DTensor.from_local(...)` → `_FromTorchTensor` per
parameter per backward. The DTensor shell is rebuilt every microbatch even
though the backing storage is reused, so the host-side `_FromTorchTensor`
cost is paid `O(params)` times per backward.

### 2.3 `get_local_tensor` — per-call `narrow`/`view` (dbuffer.py)

`copy_gradients_to_partial_buffer` and `get_dtensor` both resolve each
parameter's local shard via `local_buffer.narrow(...).view(...)` every call.
The view is stable within a buffer lifetime (storage is preserved on
resize), so rebuilding it is pure host overhead.

## 3. Design

### Step 1 (PR #121): Cache the sharded-grad DTensors

**Where**: `FsdpParameterGroup`, new `_grad_dtensor_cache` keyed on the
`main_grad` DBuffer identity.

- On the first `reduce_partial_gradients` of a group, build the DTensors via
  `get_dtensor` and cache them by parameter index.
- On later calls, rebind the cached DTensor's local tensor in place instead
  of rebuilding; when storage and shape are unchanged, reuse the cached
  view entirely (no rebind, no device sync).
- Invalidate the cache when `main_grad` is replaced (redistribute) or
  changes dtype.

This removes `_FromTorchTensor` from the per-microbatch hot path.

### Step 2 (PR #122): Fused foreach grad copy

**Where**: `copy_gradients_to_partial_buffer` (parameter_group.py).

Replace the per-parameter `.copy_()` loop with `torch._foreach_copy_()` over
batched destination/source lists — one kernel launch instead of
`O(params)` launches, with `parameter.grad = None` still applied per
parameter after the fused copy.

### Step 3 (PR #122): Cache DBuffer local-tensor views

**Where**: `DBuffer.get_local_tensor` (dbuffer.py).

Cache the `narrow(...).view(...)` result keyed by tensor index; the cache is
cleared on `_resize_storage` (release/reallocate), since cached views alias
the resized storage. Empty shards are not cached, so indices never share
the same (empty) tensor object.

### Step 4 (PR #122): NVTX ranges for MFSDP v2 comm

**Where**: `FsdpModule` (module.py).

Add `MFSDP <name> allgather` and `MFSDP <name> reduce_grad` nvtx ranges so
M-FSDP v2 communication is visible in nsys traces, and the CPU overhead of
each phase can be measured directly.

### Step 5: Validate

- Unit test: `test_fully_shard.py` — assert DTensor identity is stable
  across microbatches (`grad is grad` for the same sharded parameter),
  and that losses still match the baseline.
- nsys re-profile: measure CPU time under `MFSDP <root> backward`;
  target: `_FromTorchTensor` → 0 in the hot path, `copy_` count → 1
  (fused foreach).

## 4. Risks / constraints

- **DTensor cache validity**: `main_grad` storage can be reallocated by
  `redistribute` (HSDP outer reduction) or dtype change. The cache must
  key on `(dtype, numel, placements)` and be invalidated on
  `_main_grad` replacement.
- **DBuffer view cache validity**: cached views alias `local_buffer`
  storage, so `_resize_storage` must clear the cache; `from_local_buffer`
  (rebuild) must reset it too.
- **foreach copy shape matching**: `torch._foreach_copy_` requires
  destination/source shapes to match; empty-shard parameters are handled by
  skipping cached empty views.
- **MXFP8 groups**: `Fp8ParameterGroup` shares the base class grad path;
  DTensor caching applies to both.
- **delayed wgrad (`delay_wgrad_compute`)**: the post-accumulate hook
  ordering must stay compatible with `skip_backward_callback`.

## 5. Expected outcome

| Metric | Before | After |
|---|---|---|
| `_FromTorchTensor` per backward | O(params) | O(1) (cached) |
| `copy_` per backward | O(params) | 1 (fused foreach) |
| `get_local_tensor` per backward | O(params) narrow/view | O(1) (cached views) |
| CPU time under `MFSDP <root> backward` | ~10 ms | target < 3 ms |
