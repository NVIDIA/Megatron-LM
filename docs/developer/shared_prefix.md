# Shared-prefix execution for hybrid models

This experimental, opt-in path avoids executing an identical prompt separately for
several completions. A caller supplies `shared_prefix_layout` to `HybridModel.forward`.
Without that argument, the ordinary model path, input-ID handling, quantization
initialization, and default execution settings remain in place.

Reusable packing lives in `megatron.rl`, alongside this model execution path:
row/star/forest planning, group sharding and slots, tensor materialization,
TP/CP geometry, real-row alignment, and dense-bin reconstruction. See the
[packing API and integration contract](../../megatron/rl/shared_prefix.md).
Callers retain their configuration, batch transport, distributed coordination,
and RL objective. NeMo RL uses adapters to this implementation.

## Layout and attention

A star stores `[prompt, completion_1, ..., completion_G]` once. A forest packs
several independent stars into one forward. Every completion attends to its prompt
and its own causal history. It cannot attend to a sibling completion or another
star. RoPE positions restart at the original prompt length for each completion.
The layout also records physical padding, logical lengths, and CP token ownership.
Backward accumulates the completion contributions into shared prompt activations.

`PackedTreeLayout` also describes deeper trees and supplies ancestry-aware reference
masks. The fused hybrid attention/Mamba adapter still lowers only contiguous stars
and forests with unpadded prompt roots; deeper layouts fail explicitly. See the
[tree descriptor contracts](../../tests/unit_tests/rl/test_tree_layout.py).

TP sequence parallelism and CP zigzag ownership use the caller's process groups.
The fused attention path exchanges the required sequence/head shards and retains
an independent causal domain for each completion. This is an execution layout;
logical sample weights, loss masks, and group normalization remain the caller's
responsibility. Log-probability and training forwards must use the same assignment
when comparing their outputs at unchanged weights.

## Mamba state forking

The recurrent prefix is evaluated up to a scan-chunk boundary. Its SSM state is
forked into independent completion branches. The remaining prompt tail is replayed
with each branch, and convolution carries the required prompt halo. This preserves
the kernel's chunk alignment and boundary context while eliminating most duplicated
prompt work. Ragged branches track their own lengths and boundaries rather than
turning the longest completion into useful work for every branch. Backward combines
branch state, halo, and shared-prefix contributions.

The implementation therefore does not promise that every prompt token executes
exactly once in every Mamba sub-operation. The shared aligned prefix, residual-tail
replay, and convolution halo are distinct parts of the contract.

## MoE, recomputation, and MTP

Shared prompt rows carry their logical multiplicity for expert-bias statistics.
Both boolean routing maps and upstream dense top-k expert-index maps are supported;
padding and invalid dense routes contribute zero. Ordinary routing counts follow
the upstream path when multiplicity metadata is absent. Hash MoE remains supported
by the ordinary path and is explicitly rejected for shared execution.

The router may run fixed row blocks inside a scoped shared forward. Activation
recomputation restores that scope while suppressing tensor-observation callbacks
inside the restored context, so a logical forward is observed only once. Frozen
router parameters skip unused parameter gradients while preserving gradients into
hidden states.

MTP uses dense branch inputs and its existing heads. It does not share the MTP
prefix. When multiple independently normalized groups share one forward, optional
`loss_group_lengths` preserves their token-count correction. Ordinary MTP still
accepts precomputed decoder embeddings and upstream CP layout preparation.

## Supported scope and explicit guards

The shared adapter currently targets complete PP1 hybrid models with fp16/bf16,
zero dropout, RoPE or no positional embedding, ordinary self-attention and supported
Mamba/MoE layers. TP greater than one requires sequence parallelism. Full activation
recomputation supports the uniform method. Unsupported combinations fail explicitly,
including hash routing, quantization recipes, fp8/fp4, wide residual streams, mHC,
attention-logit softcapping, external attention/padding masks, inference contexts,
fine-grained activation offloading, CUDA graphs, sliding-window attention, auxiliary
router losses, and expert-capacity token dropping. See the validation functions for
the complete runtime contract.

`sequence_relative_kernels` and `deterministic_tp_reduce_scatter` are separate,
default-off experimental controls. The former changes the ordinary attention/Mamba
numerical backend and must not be treated as a requirement for enabling the shared
layout. Neither is qualified merely because the shared-prefix model runs.

The deferred Triton variants `NRL_SP_FUSED_KV_GATHER`,
`NRL_SP_FUSED_BACKWARD_GLUE`, and `NRL_SP_FUSED_DQ_ASSEMBLY` are unsupported
opt-ins. They default to disabled; enabling any of them raises an explicit error.

## Validation status

Four-rank portable checks pass for packing/tensors, NeMo AST planner contracts,
five CPU MTP/branch fixtures, 300 packing replays, and registry import contracts;
see the [tree contracts](../../tests/unit_tests/rl/test_tree_layout.py) and
[tensor contracts](../../tests/unit_tests/rl/test_shared_prefix_tensors.py).
CUDA reference attention passes 40 multi-level cases. Standalone CP1 BF16/FP16
FlashAttention checks pass 128 deterministic cases and 96 default comparisons per
rank, with byte-identical outputs and Q/K/V gradients. This evidence does not
qualify full hybrid-model or distributed CP/TP/EP execution.

Full current-PR GRPO has no completed optimizer update, and matched dense/shared
gradient, full-model MTP/MoE, and bounded training/evaluation qualification remain
outstanding. A Mamba recurrence simplification remains excluded because 11 of 36
native gradient cases fail the required numerical gate. The registered attention replay tests pass 126 cases per rank on four GB200 GPUs,
including real NCCL CP1/2/4, stream contention, numerical references, and import
guards. This qualifies standalone attention, not full hybrid-model training.
Full PR quality checks and coverage/native replay for the 12 remaining kernel
determinism obligations remain draft gates; see the
[kernel manifest](../../tests/unit_tests/determinism/kernels/manifest.py) and
[determinism testing requirements](determinism/testing.md).
