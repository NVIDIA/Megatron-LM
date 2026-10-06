# Shared-prefix execution for hybrid models

This experimental, opt-in path avoids executing an identical prompt separately for
several completions. A caller supplies `shared_prefix_layout` to `HybridModel.forward`.
Without that argument, the ordinary model path, input-ID handling, quantization
initialization, and default execution settings remain in place.

## Layout and attention

A star stores `[prompt, completion_1, ..., completion_G]` once. A forest packs
several independent stars into one forward. Every completion attends to its prompt
and its own causal history. It cannot attend to a sibling completion or another
star. RoPE positions restart at the original prompt length for each completion.
The layout also records physical padding, logical lengths, and CP token ownership.
Backward accumulates the completion contributions into shared prompt activations.

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

## Validation status

This current-main port has syntax, static-interface, and isolated CPU contract
checks for layout accounting, branch copy gradients, frozen-router gradients, and
recompute observation scope. These checks do not execute CUDA, Triton,
FlashAttention, Transformer Engine, distributed CP/TP/EP, or the full model.

A previous production revision completed training and checkpoint evaluations.
That is evidence for that revision and configuration, not GPU qualification of this
port. Within-implementation forward agreement, cross-implementation gradients,
and long-run evaluation quality are separate acceptance criteria. The controlled
backbone-gradient discrepancy remains open; no gradient-identity or general
numerical-equivalence claim is made here.

Before promotion: run native distributed tests and kernel replay coverage, compare
same-weight dense/shared outputs and gradients against repeated dense and repeated
shared controls, exercise matched packing and loss masks, validate MTP and expert
statistics, and run a bounded training/evaluation qualification. New kernel files
also require registration and replay tests in the upstream determinism manifest.
