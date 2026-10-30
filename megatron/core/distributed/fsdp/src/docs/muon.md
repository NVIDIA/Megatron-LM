<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# Muon with MFSDP v2

## Background

MFSDP shards one **module** at a time. By default, parameters owned by that
module with the same dtype and `requires_grad` form one parameter group.
Each group packs its weights and gradients into separate contiguous
buffers. Ranks store local segments of these buffers. Weights are
all-gathered for module computation; gradients are reduce-scattered after
backward.

The [sharding placements](../megatron_fsdp/experimental/placement.py)
determine how the buffers are partitioned:

- `RowAtomic`: keeps rows intact, with equal-sized rank segments. This is
  the default for ordinary floating-point parameters.
- `BlockAtomic(k)`: keeps blocks of `k` consecutive rows intact, with
  equal-sized rank segments. Keeping each quantization block on one rank
  simplifies blockwise quantization. MXFP8 uses `k=32`.
- `TensorAtomic`: assigns each complete parameter to one rank. Rank
  segments may differ in size or be empty. Keeping weights, gradients,
  and momentum on the same owner makes Muon's optimizer step
  communication-free.

By contrast,
[PyTorch FSDP2](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html#communication-grouping-and-scheduling)
shards **each parameter independently along dim 0**. Both systems
can group parameters for communication; the difference is how they form
the shards.

### Example

Five trainable FP32 parameters owned by one module are packed as follows.
This RowAtomic layout comes from
[`test_compute_layout_fills_lcm_padding_gaps`](../../../../../../tests/unit_tests/distributed/mfsdp_v2/test_dbuffer.py#L60)
and the illustration in
[`build_for_row_atomic`](../megatron_fsdp/experimental/layout_builder.py#L39).

```text
Shapes: P0=(2,6), P1=(4,4), P2=(4,4), P3=(1,2), P4=(1,6)
One group buffer, 12 elements per rank (including padding)

rank 0 [ 0,12): | P0 row 0              | P0 row 1               |
rank 1 [12,24): | P1 row 0      | P1 row 1      | P1 row 2       |
rank 2 [24,36): | P1 row 3      | P3    | gap   | P2 row 0       |
rank 3 [36,48): | P2 row 1      | P2 row 2      | P2 row 3       |
rank 4 [48,60): | P4                    | pad                    |
```

All-gather assembles these segments into a full group buffer on every
rank. Gradient reduce-scatter reduces the full buffers and returns the
corresponding segment to each rank. Afterward, `P0`, `P3`, and `P4` are
already whole on one rank. Only `P1` and `P2` need reconstruction for local
full-matrix Muon orthogonalization.

## Challenge

**RowAtomic and BlockAtomic.** Muon orthogonalizes the momentum-derived
input for each complete logical matrix using Newton–Schulz. Applying it
independently to row or block shards, or to the concatenated group buffer,
would change the update. The challenge is to provide whole-matrix inputs
to Muon while retaining module-level sharding.

**TensorAtomic.** Whole-matrix inputs are already local. The challenge is
to enforce the layout requested by Muon without substantially slowing
forward/backward. Block-atomic sharding can distribute parameters unevenly
while keeping the total data per rank largely balanced. Assigning whole
matrices can make that distribution unbalanced. If gathering weights or
reducing gradients takes longer, that extra time could cancel out the time
saved in the optimizer step.

## Algorithm options

Both options need a separate optimizer for excluded parameters, such as
biases and embeddings.

### Gather and scatter

**Implemented in [`FsdpMuon`](../megatron_fsdp/experimental/muon.py).** Keep
weights, gradients, and momentum in the existing sharded layout. For each
parameter group, after gradient reduction:

1. Update local momentum and construct the orthogonalization inputs.
2. Gather each split matrix's input to one compute owner.
3. Run full-matrix Newton–Schulz once on that owner.
4. Scatter update slices back and apply them to local main weight shards.
   `fully_shard_optimizer` (with `precision_aware=True`) or MCore's
   `FullyShardedOptimizer` then refreshes the model weights.

The [owner planner](../megatron_fsdp/experimental/owner_planning.py)
balances estimated compute across groups, choosing owners from ranks that
already hold a fragment. [P2P gather/scatter](../megatron_fsdp/experimental/gather_scatter.py)
moves only remote fragments. In the example, `P0`, `P3`, and `P4` require
no optimizer-step network traffic. Persistent ownership stays unchanged;
full inputs and updates exist temporarily on compute owners.

Gathers and scatters return their P2P works instead of waiting, so communication
overlaps Newton–Schulz. After a group is orthogonalized, the gathers of the next
`lookahead` groups that need communication are posted, and such a group's updates are
applied once `lookahead` later ones have been scattered, so each scatter overlaps their
Newton–Schulz. The groups whose tensors all sit on one rank (no communication) are split
evenly between both ends, with the other groups in between. They compute their momentum
when their turn comes and apply their updates right after, so they hold no buffers
meanwhile, and their Newton–Schulz hides the first gathers and the last scatters.
Each group in flight holds its momentum inputs, full tensors, and received
updates, so `lookahead` trades memory for overlap; zero gives the
non-overlapped order.

A simpler alternative in [issue #5533](https://github.com/NVIDIA/Megatron-LM/issues/5533)
all-gathers boundary-matrix inputs and duplicates orthogonalization. The
current path avoids that redundant computation.

### TensorAtomic placement

**Alternative based on [MatrixFSDP](https://arxiv.org/abs/2607.05895), tracked
in [issue #6161](https://github.com/NVIDIA/Megatron-LM/issues/6161).** Assign
each whole matrix, its gradient, and its momentum to one persistent owner;
other ranks hold empty shards for that matrix.

Using the same five tensors as the background example, one possible owner
assignment is:

```text
One group buffer, 52 elements total, no padding

rank 0 [ 0,12): | P0 row 0              | P0 row 1               |
rank 1 [12,28): | P1 row 0      | P1 row 1      | P1 row 2      | P1 row 3       |
rank 2 [28,28): empty
rank 3 [28,44): | P2 row 0      | P2 row 1      | P2 row 2      | P2 row 3       |
rank 4 [44,52): | P3    | P4                    |
```

Every matrix is whole on its owner. Rank 2 owns no tensors in this group.

1. Materialize the module's weights from owners for forward/backward.
2. Reduce each matrix's gradient to its owner.
3. Update momentum, orthogonalize, and update weights locally on the owner.

This removes optimizer-step matrix communication. Forward/backward should
[communicate uneven owner segments without padding](https://github.com/NVIDIA/Megatron-LM/issues/6161#issuecomment-5229334204).
MFSDP already provides the [layouts](../megatron_fsdp/experimental/placement.py)
and [collectives](../megatron_fsdp/experimental/dbuffer.py) for this.

## Tradeoffs

- **Gather/scatter favors forward/backward** through more balanced sharding.
- **TensorAtomic favors the optimizer** by avoiding
  gather/scatter communication and temporary buffers.

## Next step

1. **Optimize performance.** Batch Newton–Schulz across FSDP units (it
   currently runs one unit at a time), and make parameter placement aware of
   hardware topology to keep traffic on fast links, as discussed in [#6368](https://github.com/NVIDIA/Megatron-LM/issues/6368).
2. **Evaluate TensorAtomic at larger scale.** Compare it with gather/scatter
   on larger models across multiple nodes, measuring forward/backward time,
   full-step time, and peak memory.
