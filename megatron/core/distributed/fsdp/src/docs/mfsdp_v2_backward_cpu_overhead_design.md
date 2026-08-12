# MFSDP v2 Backward Gradient Path — CPU Overhead Reduction Design

## 1. Problem

nsys profile of the M-FSDP v2 + MXFP8 PP3/VPP2/EP8 run (Lyris GB200, pd1)
shows ~10 ms of CPU time per backward under the `MFSDP <root> backward`
NVTX tag, spent in:

- **`copy_`** — packing per-parameter gradients into the reduce-scatter input
  buffer (`copy_gradients_to_partial_buffer`, `parameter_group.py:391`)
- **`_FromTorchTensor`** — DTensor construction per parameter per backward
  (`get_dtensor` / `DBuffer.from_local` in the grad-reduction path and the
  `.grad` rebind in `reduce_partial_gradients`)

These are **host-side** costs: per-microbatch per-parameter tensor ops that
serialize the backward and sit on the critical path of the 1F1B schedule.
Each contributes `O(num_params_in_group)` Python/ATen dispatches per
microbatch.

## 2. Root-cause analysis

### 2.1 `copy_` — gradient packing (parameter_group.py:386-392)

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

**The wgrad already lands in `parameter.grad` (an autograd-managed buffer).
A fused path can make the MLP/attention wgrad accumulate directly into the
partial buffer views, skipping the copy entirely** (PR #5032's
"gradient-accumulation-fusion" concept).

### 2.2 `_FromTorchTensor` — DTensor creation (dbuffer.py:466-496)

`reduce_partial_gradients` (parameter_group.py:471-472) rebinds every
sharded parameter's `.grad`:

```python
for index, fsdp_parameter in enumerate(self.fsdp_parameters):
    fsdp_parameter.sharded.grad = self.main_grad.get_dtensor(index)
```

`get_dtensor` calls `DTensor.from_local(...)` → `_FromTorchTensor` per
parameter per backward. PR #5032 caches these DTensors (`_dist_grad_cache`)
and rebinds only the `_local_tensor.data` (via
`rebind_uneven_dtensor_local_tensor`) — the DTensor shell is created once
and its storage pointer is updated in place, eliminating the
`_FromTorchTensor` CPU cost on subsequent microbatches.

## 3. Design (step by step)

### Step 1: Cache the sharded-grad DTensors (`get_dtensor` reuse)

**Where**: `FsdpParameterGroup`, new `_grad_dtensor_cache: list[DTensor | None]`.

- On the first `reduce_partial_gradients` of a group, build the DTensors via
  `get_dtensor` and cache them keyed by parameter index.
- On later calls, instead of `DTensor.from_local`, **rebind the cached
  DTensor's `_local_tensor.data`** to the new `main_grad` local view
  (`object.__setattr__(dt._local_tensor, "data", new_view)`).
- Invalidate the cache when `main_grad` is re-allocated (dtype/size change)
  or when the group is rebuilt.

This removes `_FromTorchTensor` from the per-microbatch hot path.

### Step 2: Gradient-accumulation fusion for the copy_

**Where**: `_reduce_gradient_groups` (module.py:520-538) +
`copy_gradients_to_partial_buffer` (parameter_group.py:386).

**2a. (rejected) Reuse the partial buffer across microbatches.** A first
attempt cached `allocate_partial_grad_buffer` on the group. nsys showed
`cudaEventSynchronize` ~24x and `cudaMemcpyAsync` ~37x worse: with the NCCL
symmetric-memory pool the reused buffer stays registered across microbatches,
so the next backward's `copy_` write forces a device sync. **Reverted** — a
fresh buffer per backward preserves the allocate-on-reduce-scatter-stream +
release invariant that avoids allocator/symm-mem serialization.

**2b. (tested — regression) Fuse the wgrad write into the partial buffer.**
Implemented the TE gradient-accumulation-fusion path for M-FSDP v2: attached
`get_main_grad()` + `__fsdp_param__` + `main_grad` to each unsharded
parameter so TE's `layers.py` writes the wgrad GEMM output directly into the
reduce-scatter input buffer, and skipped the `copy_` in
`copy_gradients_to_partial_buffer` via `grad_added_to_main_grad`.

**A/B result on Lyris GB200 (24xGB200, MBS=2, mock data, nsys):**

| Metric | Baseline (no fusion) | Wgrad fusion | Delta |
|---|---|---|---|
| Throughput | 456 TFLOP/s/GPU | ~79 TFLOP/s/GPU | **5.8x worse** |
| cudaEventSynchronize | 1288 ms | 17319 ms | **13.4x worse** |
| KernelLaunch CPU | 353 ms | 5828 ms | 16.5x worse |
| MemsetAsync | 28 ms | 317 ms | 11x worse |

**Why it regressed**: the fused wgrad writes into a *Partial* reduce-scatter
input buffer allocated **lazily during the layer backward on the current
stream**. The cross-stream handoff (buffer allocated on current stream, RS on
the reduce-scatter stream) plus TE's `te_general_gemm(out=...)` into the
fp32 partial view forces a `cudaEventSynchronize` per parameter per backward
(13x worse). The buffer is also zeroed per backward (MemsetAsync 11x).

**The fix that did NOT work here (unlike PR #5032's v2 path)**:
PR #5032 writes into a *Replicate* main-grad buffer that is **pre-allocated
per backward with stable storage**, not a lazily-allocated Partial buffer.
Retrying 2b requires:
  1. Pre-allocate the fused-wgrad target **before** the layer's backward
     (not lazily inside `get_main_grad`), with storage bound to one stream.
  2. Use a Replicate (or same-stream) placement so TE's `out=` write does
     not cross streams.
  3. Keep the buffer dtype equal to the wgrad dtype (bf16), not fp32.

**Status**: reverted. PR #121 keeps only the gradient-DTensor cache (the
`_FromTorchTensor` elimination), which is a clean win with no regression.

### Step 3: Validate

- Unit test: `test_fully_shard.py` — assert DTensor identity is stable
  across microbatches (`grad is grad` for the same sharded parameter),
  and that losses still match the baseline.
- nsys re-profile: measure CPU time under `MFSDP <root> backward`;
  target: `_FromTorchTensor` → 0 in the hot path, `copy_` count halved.

## 4. Risks / constraints

- **DTensor cache validity**: `main_grad` storage can be reallocated by
  `redistribute` (HSDP outer reduction) or dtype change. The cache must
  key on `(dtype, numel, placements)` and be invalidated on
  `_main_grad` replacement.
- **partial buffer reuse**: `allocate_partial_grad_buffer` is currently
  called inside `with torch.cuda.stream(reduce_scatter_stream)` — reusing
  the buffer across microbatches must preserve the stream-ordering
  guarantees (the wait_stream edges in `_reduce_gradient_groups`).
- **MXFP8 groups**: `Fp8ParameterGroup` shares the base class grad path;
  DTensor caching applies to both.
- **delayed wgrad (`delay_wgrad_compute`)**: the post-accumulate hook
  ordering must stay compatible with `skip_backward_callback`.

## 5. Expected outcome

| Metric | Before | After |
|---|---|---|
| `_FromTorchTensor` per backward | O(params) | O(1) (cached) |
| `copy_` per backward | O(params) | unchanged (fusion reverted — see 2b) |
| CPU time under `MFSDP <root> backward` | ~10 ms | target < 3 ms (DTensor cache) |
