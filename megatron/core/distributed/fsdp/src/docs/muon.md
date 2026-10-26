<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# Muon with MFSDP v2

## Background

MFSDP shards one **module** at a time. By default, parameters owned by that
module with the same dtype and `requires_grad` form one parameter group.
They are packed into a shared buffer, which is all-gathered for module
computation and gradient-reduced together after backward. The default
`RowAtomic` layout divides that buffer into equal-sized rank segments,
keeping rows intact while allowing matrices to cross rank boundaries.

[PyTorch FSDP2](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html#communication-grouping-and-scheduling)
instead shards **each parameter independently along dim 0**. Both systems
can group parameters for communication; the difference is how they form
the shards.

Muon orthogonalizes the momentum-derived input for **each complete logical
matrix** using Newton–Schulz. Applying it independently to row shards, or
to the concatenated group buffer, would change the update. The challenge
is to provide whole-matrix inputs while retaining module-level sharding.

### Example

Five trainable FP32 parameters in one module share the following buffer.
This layout comes from
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

## Algorithm options

### Gather and scatter

**Implemented in [`FsdpMuon`](../megatron_fsdp/experimental/muon.py).** Keep
weights, gradients, and momentum in the existing sharded layout. For each
parameter group, after gradient reduction:

1. Update local momentum and construct the orthogonalization inputs.
2. Gather each split matrix's input to one compute owner.
3. Run full-matrix Newton–Schulz once on that owner.
4. Scatter update slices back, apply them to local weight shards, and
   refresh model weights.

The [owner planner](../megatron_fsdp/experimental/owner_planning.py)
balances estimated compute across groups, choosing owners from ranks that
already hold a fragment. [P2P gather/scatter](../megatron_fsdp/experimental/gather_scatter.py)
moves only remote fragments. In the example, `P0`, `P3`, and `P4` require
no optimizer-step network traffic. Persistent ownership stays unchanged;
full inputs and updates exist temporarily on compute owners.

A simpler alternative in [issue #5533](https://github.com/NVIDIA/Megatron-LM/issues/5533)
all-gathers boundary-matrix inputs and duplicates orthogonalization. The
current path avoids that redundant computation.

### TensorAtomic placement

**Alternative based on [MatrixFSDP](https://arxiv.org/abs/2607.05895), tracked
in [issue #6161](https://github.com/NVIDIA/Megatron-LM/issues/6161).** Assign
each whole matrix, its gradient, and its momentum to one persistent owner;
other ranks hold empty shards for that matrix.

1. Materialize the module's weights from owners for forward/backward.
2. Reduce each matrix's gradient to its owner.
3. Update momentum, orthogonalize, and update weights locally on the owner.

This removes optimizer-step matrix communication. Forward/backward still
communicate at the module-group level, but owner segments can differ in
size or be empty. Balancing ownership across the model does not ensure
balance within every module. Communication must handle these uneven
segments without padding every rank to the largest contribution, as
explained in the [authors' discussion](https://github.com/NVIDIA/Megatron-LM/issues/6161#issuecomment-5229334204).

MFSDP already supports [`TensorAtomic`](../megatron_fsdp/experimental/placement.py)
layouts and [uneven DBuffer collectives](../megatron_fsdp/experimental/dbuffer.py).
The next step is a direct Muon path that skips gather/scatter staging and
local copies. Both approaches need a companion optimizer for excluded
parameters, such as biases and embeddings.

## Tradeoffs

| | Gather/scatter | TensorAtomic |
| --- | --- | --- |
| Persistent storage | Equal-sized group shards, with alignment padding | Whole tensors packed into uneven owner segments |
| Optimizer communication | Gather/scatter remote fragments of split matrices | None for the matrix update |
| Forward/backward | Equal-sized group collectives | Uneven group materialization and reduction |
| Memory | Sharded state plus temporary full matrices | Whole-matrix state and workspace must fit on each owner |
| Load balance | Compute owners restricted to existing fragment holders | Balance persistent memory and compute across modules |

The benefit depends on how much boundary-matrix traffic is removed relative
to the cost of uneven forward/backward communication.

## Next step

Keep gather/scatter as the baseline and prototype the TensorAtomic fast
path. Compare full-step time and peak memory with identical module
boundaries, precision, and optimizer settings; verify updates against
full-matrix Muon. Include single-node and multi-node runs.
