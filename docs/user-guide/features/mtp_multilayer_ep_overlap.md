<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Multi-Layer MTP with Expert-Parallel A2A Overlap

Megatron-LM supports training with multiple Multi-Token Prediction (MTP) layers
(`mtp_num_layers > 1`) combined with expert-parallel all-to-all overlap
(`overlap_moe_expert_parallel_comm`). This combination enables deep MTP stacks —
such as the two-layer MTP used in DeepSeek-V3 — to coexist with the fine-grained
pipeline schedule required for EP communication overlap.

## Background

When `overlap_moe_expert_parallel_comm=True` is set, the training loop replaces
the standard 1F1B schedule with a fine-grained, layer-wise schedule managed by
`TransformerModelChunkSchedulePlan`. Each layer is an independent schedule node;
its forward and backward callables are invoked one at a time so that dispatch and
combine all-to-all collectives can be overlapped with adjacent computation.

MTP layers participate in this schedule through a set of dedicated callables:

```
submodule_mtp_attn_forward
  → dispatch_forward
  → moe_forward
  → combine_forward
  → submodule_mtp_postprocess_forward
```

With a single MTP layer (`mtp_num_layers=1`), this schedule is straightforward.
With multiple independent MTP layers (`mtp_num_layers>1`,
`mtp_use_repeated_layer=False`), the last MTP postprocess node concatenates the
hidden states from all MTP depths into a single tensor before loss computation.
That concatenation introduced an autograd-graph sharing bug that caused training
to crash on the first backward pass.

## The Backward Bug and Its Fix

### Root Cause

In the fine-grained schedule, each node calls
`default_backward_func(..., keep_graph=False)`. The `keep_graph=False` flag
releases all saved autograd state (such as `ctx.tensor_objects` inside
`ViewlessTensor`) immediately after one traversal.

With `mtp_num_layers=2` the postprocess outputs are:

- `mtp_0_post`: output of MTP layer 0's `_postprocess`, referenced both by
  layer 0's schedule node (`self.output`) and as an input to the
  `torch.cat([decoder_out, mtp_0_post, mtp_1_post])` in layer 1's node.
- `mtp_1_post`: output of MTP layer 1's `_postprocess`, referenced only by
  layer 1's node.

During backward, layer 1's node runs first. Its `default_backward_func` traverses
the cat tensor, which reaches `mtp_0_post` and consumes `ctx_0`
(`ViewlessTensor`'s saved state). When layer 0's node runs next, it attempts to
traverse `mtp_0_post` through `ctx_0` again—but the saved tensors are already
gone, causing:

```
AttributeError: ctx must have .tensor_objects to restore saved tensors
```

### Fix

The fix breaks the shared autograd graph by detaching intermediate MTP hidden
states before the final concatenation, then accumulating the resulting gradients
back into the correct schedule node's backward pass.

**In `submodule_mtp_postprocess_forward`** (last layer branch), intermediate
hidden states (all depths except the first decoder output and the last MTP
output) are replaced with detached leaves before the cat:

```python
if node.is_last_layer:
    node.chunk_state.mtp_postprocess_extra_grads = {}
    cat_tensors = []
    for i, hs in enumerate(node.chunk_state.mtp_hidden_states):
        if 0 < i < len(node.chunk_state.mtp_hidden_states) - 1:
            hs_d = hs.detach().requires_grad_(hs.requires_grad)
            node.chunk_state.mtp_postprocess_extra_grads[i - 1] = hs_d
            cat_tensors.append(hs_d)
        else:
            cat_tensors.append(hs)
    hidden_states = torch.cat(cat_tensors, dim=0)
```

**In `TransformerLayerNode.backward_impl`**, non-last MTP postprocess nodes
read back the gradient that accumulated on their detached leaf and add it to
`output_grad` before calling `default_backward_func`:

```python
if (self.is_mtp and not self.is_last_layer
        and hasattr(self, 'mtp_postprocess_idx')
        and hasattr(self.chunk_state, 'mtp_postprocess_extra_grads')):
    idx = self.mtp_postprocess_idx
    srcs = self.chunk_state.mtp_postprocess_extra_grads
    if idx in srcs and srcs[idx] is not None:
        extra_grad = srcs[idx].grad
        if extra_grad is not None:
            grads = (
                (grads[0] + extra_grad if grads[0] is not None else extra_grad),
            ) + grads[1:]
```

This ensures each `ctx_k` is traversed exactly once, in its own schedule node's
backward pass, while the full gradient from both the MTP loss path and the
downstream layer path is correctly accumulated.

## Related Arguments

| Argument | Description |
| --- | --- |
| `mtp_num_layers` | Number of MTP layers. Values greater than 1 are supported with `overlap_moe_expert_parallel_comm=True` when `mtp_use_repeated_layer=False`. |
| `mtp_loss_scaling_factor` | Weight for the MTP auxiliary loss. Default: `0.1`. |
| `overlap_moe_expert_parallel_comm` | Enable fine-grained EP A2A overlap. When set with `mtp_num_layers > 1`, the multi-layer MTP backward fix described above is automatically active. |

## Usage

Enable multi-layer MTP with EP overlap by setting both flags together:

```bash
--mtp-num-layers 2
--mtp-loss-scaling-factor 0.1
--overlap-moe-expert-parallel-comm
```

No additional flags are required to activate the backward fix; it is applied
automatically whenever `mtp_num_layers > 1` and `mtp_use_repeated_layer=False`
(the default).

## Implementation Notes

- **CUDA graph compatibility.** With `mtp_use_repeated_layer=False` each MTP
  layer has an independent `mtp_model_layer` object. The CUDA graph build loop
  iterates over all physical layers, so each depth is captured and replayed
  independently. The `_get_embeddings` and `_postprocess` calls run eagerly
  outside the graph, which avoids static-tensor constraints.

- **VPP layout.** When `pipeline_model_parallel_layout` places MTP layers in a
  dedicated virtual pipeline stage (`mtp_standalone=True`), the
  `get_mtp_layer_offset` function returns the correct offset for each depth.
  The number of chunks assembled by `submodule_mtp_postprocess_forward` matches
  the `num_recv_chunks` expected by `process_mtp_loss`.

- **Gradient correctness.** The detach-and-accumulate pattern preserves the
  full gradient path: the gradient flowing from the MTP loss through the
  detached leaf and the gradient from the downstream MTP layer's attention input
  are both accumulated onto `combined_k_d` before traversing `ctx_k`.

## Unsupported Combinations

- `mtp_use_repeated_layer=True` with `overlap_moe_expert_parallel_comm=True`
  and `mtp_num_layers > 1` is **not yet supported**. With repeated layers the
  fine-grained schedule plan builder sees only one physical layer and generates
  a single schedule node regardless of `mtp_num_layers`, causing incorrect loss
  computation and shape mismatches. Support for this combination requires
  separate changes to `model_chunk_schedule_plan.py` and `cuda_graphs.py` to
  expand the single physical layer into the correct number of logical nodes.
