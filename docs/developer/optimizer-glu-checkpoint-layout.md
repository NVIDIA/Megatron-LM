# GLU model and optimizer checkpoint layout

Native `torch_dist` checkpoints store GLU FC1 rows in canonical gate/up order.
The Core factory converts the runtime row layout on save and restores the target
runtime layout on load. The factory retains the original parameter identity, so
model-space optimizer factories apply the same conversion to FP32 masters and
Adam moments, including ordinary non-distributed optimizers. BF16/FP16 grouped
parameters retain their model dtype while optimizer shards retain their own FP32 dtype. Routed TEGroupedMLP, fused shared experts and the grouped dense MLP
declare their runtime layout at the factory call site.

The existing content metadata records these semantic fields:

| Field | Meaning |
| --- | --- |
| `glu_checkpoint_layout` | `contiguous`: model rows use canonical gate/up order. |
| `glu_optimizer_layout` | `contiguous` for model-space state; `runtime` for raw optimizer buffers. |
| `glu_runtime_layout` | Built modules' routed/shared/dense block sizes and TP/ETP sizes. |
| `glu_tensor_layouts` | Per-factory row layout and TP sharding, recorded by Core save itself. |

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
order. In particular, the direct Core writer before this change split an
interleaved FC1 tensor in half without first undoing interleaving. For block size
1, runtime rows `[0,4,1,5,2,6,3,7]` could reach disk in that order. Treating those
rows as canonical produces `[0,2,4,6,1,3,5,7]` after the new factory merge.
MLM's training-level tensor conversion introduced in #7606 and Bridge's
plain-tensor conversion can produce canonical model rows, but a saved runtime
configuration does not prove that either conversion actually ran. Neither a
matching interleave setting nor a model-only request establishes provenance. Recover them with their original writer and verify model, master and
moment row order before conversion and metadata annotation. Adding a marker
alone does not repair a checkpoint.

New interleaved model-space optimizer checkpoints require a layout-aware Core
reader. Older readers can ignore the new metadata and misinterpret canonical
moments as runtime tensors. Bridge must also adopt the matching change that
removes its post-load/save permutation before using this Core implementation.

Tests cover BF16 distributed Adam on two GPUs, both gather modes, masters and
moments, three bit-exact subsequent updates after a block-size change, canonical
disk round-trips and TP=1 to TP=2 factory resharding. The latter is a factory/state
test, not a complete TP-changing training run. MLM's training entrypoint still rejects interleaved FSDP and GTP.
Bridge retains its FSDP DTensor model conversion through the shared Core row
converter; it records semantic metadata in that backend and keeps optimizer
rows in their original layout. Native interleaved FSDP2 factories remain unsupported. This does not validate FP8/FP4 fused kernels or checkpoint
memory at production scale. Custom GLU modules must declare their layout through
`apply_swiglu_sharded_factory(..., glu_interleave_size=...)`.


Core `save` collects GLU factory declarations across ranks and writes the model
layout marker even when called directly, as AutoBridge does. Automatically generated
metadata stays out of the returned model/optimizer state dictionary, so strict
module loading and chained-optimizer loading retain their original keys. It is
available through `load_content_metadata`; explicitly supplied content metadata
retains its existing return behavior. Core `load` checks
the marker before factory expansion or tensor loading. The factory stores the
registered parameter, including single grouped expert weights; expert splitting
happens inside its build function so optimizer identity lookup sees that same
parameter. HF import/export converts the declared runtime layout before splitting
or joining projections. Legacy tensor conversion selects exact names from built
GLU modules and leaves non-GLU vision FC1 tensors alone.

For an older interleaved checkpoint, recover with its original code and format,
verify its model rows against a known output, and save with this layout-aware
writer. If only model provenance is established, start a fresh optimizer.
A blanket metadata annotation or generic row shuffle cannot establish unknown
master/moment layouts, so this change does not provide an automatic conversion
of ambiguous historical optimizer state.
