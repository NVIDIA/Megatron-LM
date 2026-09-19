<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Engram conditional memory

Engram compresses token IDs, hashes suffix n-grams into embedding tables, and adds
retrieved memory to selected Hybrid attention inputs through query/key gating and
causal ShortConv. `EngramHybridProvider` composes `EngramAttentionLayer`; each
attention layer owns its memory parameters. GPTModel and the MTP branch are unchanged.

## Placement and inputs

`engram_layer_ids` are stable nonnegative memory IDs. `engram_target_layer_indices`
optionally selects corresponding zero-based `*` positions in the main Hybrid pattern.
Without explicit positions, memory ID `i` selects the `i`-th attention occurrence.
Both lists must be unique and equally sized. Pipeline separators and the MTP suffix
are excluded when resolving positions and hash capacities.

For eight logical blocks followed by MTP1, use `*-*E*E*E*E*E*E*E/*E`.
`--engram-layer-ids 1 --engram-target-layer-indices 2` inserts memory before the
second attention's normalization. Other Hybrid layer families may coexist between
selected standard attentions.

The provider passes explicit raw tokens for each microbatch, including recomputation
and PP/VPP consumers without an embedding layer. It does not cache a mutable current
batch. Engram restores SP/CP sequence partitions before hashing and convolution,
then returns hidden states in the consumer attention layout.

## Tables and native optimization

The `local` backend shards table width over TP and uses ordinary dense gradients.
The `row_a2a` backend partitions rows over an E/R grid within each pipeline stage:
E ranks exchange requested rows through native A2A, and R ranks replicate each
partition. `--engram-row-parallel-size` selects E, which must divide the stage's
TP × DP × CP participant count; the default uses all stage participants as E.
VPP chunks reuse the stage's process groups.

Lookup backward accumulates owner gradients into FP32 `main_grad`. Native DDP
buffers reduce replicas once after microbatch accumulation. DDP groups table
parameters by their explicit optimizer-sharding group when constructing buffers;
ordinary, expert, and table buffers share the native DDP lifecycle.
Their `DistributedOptimizer` owns parameter gathers and FP32 Adam state.

Tables use native Adam with a 5× learning rate and zero weight decay. Fusion and
backbone parameters follow the selected native optimizer policy, including Muon.
Parameters may be FP32 or BF16; master values and moments remain FP32. The normal
training loop owns gradient clearing, normalization, global clipping, the common
nonfinite-gradient decision, updates, scheduling, and checkpointing. Table-free
pipeline stages participate in the same statistical and checkpoint collectives.

## Checkpoints and supported execution

Native `torch_dist` checkpoints identify table tensors by logical table coordinates
and retain FP32 optimizer state. Same-topology resume restores model, optimizer,
scheduler, RNG, and data progress. Resharding preserves logical state; subsequent
floating-point trajectories need not be bitwise identical across topologies.
The supported optimizer checkpoint format is native fully sharded model space.

The training scope includes FP32/BF16, TP/SP, PP/VPP, CP, EP, MTP1, and ordinary/full
recomputation. Local table head width must divide TP. Row tables require ordinary
Megatron DDP and unity loss scaling. Inference decoding, packed sequences, mHC,
FP8/FP4, FSDP, ModelOpt, CUDA graphs, CPU offload, MoE shortcut connections, and fine-grained expert-overlap
schedules remain outside this feature. Restrictions apply only when Engram is enabled.

Tests distinguish addressing/fusion equivalence, native Adam mathematics, E/R and
pipeline checkpoint state, token routing, and real Hybrid execution. Optional backend
skips do not establish support. Full-model BF16 CP comparisons retain separate loss
and gradient tolerances; fixed-input comparisons isolate Engram layout correctness.

See [the public training example](../../../examples/engram/README.md) for the paired
MoE and MoE+Engram configuration. A short run establishes training and resume behavior,
not a convergence benefit.
