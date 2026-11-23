# GLU model and optimizer checkpoint layout

Native `torch_dist` checkpoints store GLU FC1 rows in canonical gate/up order.
The Core factory converts the runtime row layout on save and restores the target
runtime layout on load. The factory retains the original parameter identity, so
model-space optimizer factories apply the same conversion to FP32 masters and
Adam moments. Routed TEGroupedMLP, fused shared experts and the grouped dense MLP
declare their runtime layout at the factory call site.

The existing content metadata records three semantic fields:

| Field | Meaning |
| --- | --- |
| `glu_checkpoint_layout` | `contiguous`: model rows use canonical gate/up order. |
| `glu_optimizer_layout` | `contiguous` for model-space state; `runtime` for raw optimizer buffers. |
| `glu_runtime_layout` | Saving run's routed/shared/dense block sizes and TP/ETP sizes. |

There is no checkpoint-version bump or alternative tensor loader. Native model
factory merges return runtime tensors, so training loops must not permute them
again. Local checkpoints retain the same fields in common state. MLM's legacy
`torch` format keeps its existing model conversion and raw optimizer layout.

To change GLU interleave size or TP/ETP while retaining optimizer state, first
save a model-space checkpoint with `--dist-ckpt-optim-fully-reshardable`.
The memory-efficient gather option uses the same layout contract. Raw
`dp_reshardable`/gather-scatter buffers retain runtime order and require the saved
interleave and TP/ETP settings when restoring the optimizer. Model-only loads
from marked canonical checkpoints may choose a different layout.

Older checkpoints from known noninterleaved runs remain supported, including
layout changes for their `fully_reshardable` state. Unmarked interleaved
checkpoints are rejected, even for model-only loading: historical tensor and
factory conversion differed, so runtime configuration cannot establish storage
order. Recover them with their original writer and verify model, master and
moment row order before conversion and metadata annotation. Adding a marker
alone does not repair a checkpoint.

New interleaved model-space optimizer checkpoints require a layout-aware Core
reader. Older readers can ignore the new metadata and misinterpret canonical
moments as runtime tensors. Bridge must also adopt the matching change that
removes its post-load/save permutation before using this Core implementation.

Tests cover BF16 distributed Adam on two GPUs, both gather modes, masters and
moments, three bit-exact subsequent updates after a block-size change, canonical
disk round-trips and TP=1 to TP=2 factory resharding. The latter is a factory/state
test, not a complete TP-changing training run. Interleaved FSDP/DTensor and GTP
weights are rejected. This does not validate FP8/FP4 fused kernels or checkpoint
memory at production scale. Custom GLU modules must declare their layout through
`apply_swiglu_sharded_factory(..., glu_interleave_size=...)`.
