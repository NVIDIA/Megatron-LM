# Shared-prefix execution for hybrid models

This experimental, opt-in path avoids executing an identical prompt separately for
several completions. A caller supplies `shared_prefix_layout` to `HybridModel.forward`.
Without that argument, the ordinary model path, input-ID handling, quantization
initialization, and default execution settings remain in place, and importing
`HybridModel` does not load the shared-prefix adapter.

Model execution is split by owner. `megatron.core.datasets.shared_prefix.layout`
describes the packed layout.
`megatron.core.models.hybrid.shared_prefix` validates a model configuration and a
`HybridStack`, and runs the stack's layers in order. Each layer type keeps its own
execution: `megatron.core.ssm.shared_prefix.mamba_shared_prefix` and the ragged
Mamba kernels under `megatron.core.ssm.shared_prefix` fork the prefix state,
`megatron.core.ops.attention.shared_prefix` runs the forest attention, and
`megatron.core.transformer.shared_prefix_attention` checks that a self-attention
layer can run it. Callers build the layout and own their configuration, batch
transport, distributed coordination, and RL objective.

The stack passes each layer its shared-prefix input as a keyword argument of the
layer call. Self-attention layers receive the layout as `shared_prefix_forest`.
Mamba layers receive a `shared_prefix_call` request, which `MambaLayer` passes on
to `MambaMixer`. With expert bias enabled, MoE layers receive the logical token
counts as `shared_prefix_token_multiplicities`. No module state is set, so
activation recomputation replays each layer call with its own arguments.

## Layout and attention

A tree is one prompt with its completions. `SharedPrefixTree` stores its tokens as
`[prompt, completion_1, ..., completion_G]`, with the prompt once. A pack is several
trees in one forward. `SharedPrefixPack` stores its trees one after another, and the
trees are independent of each other. Every completion attends to its prompt
and its own causal history. It cannot attend to a sibling completion or another
tree. RoPE positions restart at the original prompt length for each completion.
The layout also records physical padding, logical lengths, and CP token ownership.
Backward accumulates the completion contributions into shared prompt activations.

These two types are the only core layouts. A prompt carries no padding, and a
completion does not branch again. A caller with deeper trees must lower them to
these types first. The attention and Mamba kernels call a pack a forest: the
self-attention keyword is `shared_prefix_forest`, and the kernel names contain
`forest`.

The padded length of a forward must be a multiple of the topology multiple, which
`shared_prefix_topology_multiple` returns: 1 at TP1/CP1, 2·CP at TP1 with CP
greater than one, and 2·TP·CP whenever TP is greater than one. At CP1 this is
2·TP, which is stricter than the TP alignment of ordinary sequence packing. Stack
validation applies this rule to the layout of every forward.

TP sequence parallelism and CP zigzag ownership use the caller's process groups.
The forest attention path exchanges the required sequence/head shards and retains
an independent causal domain for each completion. This is an execution layout;
logical sample weights, loss masks, and group normalization remain the caller's
responsibility. Log-probability and training forwards must use the same assignment
when comparing their outputs at unchanged weights.

Attention runs every tree of a pack in at most two FlashAttention varlen passes,
a causal self pass and a non-causal pass from completions to their prompt, and
merges them by log-sum-exp. Forward and backward are exact. The backward requires
flash-attn 2.7.0 or later, which stack validation checks before any layer runs.
The passes always call FlashAttention 2, whatever `flash_attention_version` pins for
Transformer Engine and the inference engine.
Under `torch.use_deterministic_algorithms(True)`, which `--deterministic-mode`
enables, the backward uses FlashAttention's deterministic kernel; no other switch
is needed for bit-exact attention replay.

## Mamba state forking

The recurrent prefix is evaluated up to a scan-chunk boundary. Its SSM state is
forked into independent completion branches. The remaining prompt tail is replayed
with each branch, and convolution carries the required prompt halo. This preserves
the kernel's chunk alignment and boundary context while eliminating most duplicated
prompt work. Backward combines branch state, halo, and shared-prefix contributions.

The implementation therefore does not promise that every prompt token executes
exactly once in every Mamba sub-operation. The shared aligned prefix, residual-tail
replay, and convolution halo are distinct parts of the contract. Every shared path
passes channel-last input to `causal_conv1d`, as `MambaMixer` does, because its
channel-first backward is wrong at some sequence lengths.

Each Mamba layer runs through `MambaLayer.__call__` and `MambaMixer.__call__`, as
attention and MoE layers run through their module calls, so module hooks such as
Megatron FSDP's parameter gathers fire as on the dense path. Shared-prefix training
under Megatron FSDP has not been qualified.

`NRL_SP_MAMBA_IMPL` selects the Mamba backend. The default, `ragged_state_fork`,
applies to every topology and tree count, including a single tree at TP1/CP1; it pads
each branch only to its own chunk boundary. The other values are explicit opt-ins:
`state_fork` uses only the public mamba-ssm API and pads every branch to its
longest sibling, `replay_prefix` rescans the prompt for each branch as a parity
baseline, and `packed_recurrence` and `packed_fused` are diagnostic oracles. The
`ragged_state_fork_training`, `replay_prefix_training` and
`packed_recurrence_training` variants use `state_fork` in evaluation mode. An
unknown value fails during stack validation.

`ragged_state_fork` calls private mamba-ssm SSD kernels (see the `ssm` extra in
`pyproject.toml`) and requires a power-of-two `chunk_size` and the default SSM
state dtype. If the kernels cannot be imported or either requirement is not met,
stack validation fails before any layer runs and names `NRL_SP_MAMBA_IMPL=state_fork`
as the workaround. `NRL_SP_MAMBA_SAVE_INTERMEDIATES=1` keeps four scan
intermediates for backward instead of recomputing them; the default is `0`.

The ragged gather and forest state kernels accumulate in a fixed order without
atomics. They are registered in the
[kernel manifest](../../tests/unit_tests/determinism/kernels/manifest.py) and
replayed bit-exactly in
`tests/unit_tests/determinism/kernels/test_shared_prefix_mamba_kernels.py`. The
reused mamba-ssm SSD backward and the `causal_conv1d` weight gradient take their
ordered reductions with `MAMBA_DETERMINISTIC=1` and `CAUSAL_CONV1D_DETERMINISTIC=1`,
which that test sets; `--deterministic-mode` rejects explicit non-deterministic
values of either.

## MoE, recomputation, and MTP

Shared prompt rows carry their logical multiplicity for expert-bias statistics.
Both boolean routing maps and upstream dense top-k expert-index maps are supported;
padding and invalid dense routes contribute zero, and logical counts accumulate
exactly in the int64 expert-bias buffer. Ordinary routing counts follow the
upstream path when multiplicity metadata is absent. Hash MoE remains supported by
the ordinary path and is explicitly rejected for shared execution.

The caller states whether per-branch padding rows count toward expert bias with
`shared_prefix_exclude_sequence_padding_from_expert_bias` on `HybridModel.forward`
(`exclude_sequence_padding_from_expert_bias` on `forward_hybrid_stack_shared_prefix`).
`True` excludes them, as a dense forward with the packed-sequence padding mask does.
`False` counts them, as a dense forward with `padding_mask=None` does. When the
argument is unset, padding rows are excluded if `moe_token_dispatcher_type` is
`flex` and `moe_flex_dispatcher_backend` is `hybridep`, and counted otherwise.
The MTP routers follow the same convention: when padding is excluded, the dense
MTP branches receive their per-sequence padding as `padding_mask`. Passing the
argument without `shared_prefix_layout` raises `ValueError`. Trailing topology
padding never counts.

The router runs fixed row blocks inside a scoped shared forward, so its GEMM does
not depend on how trees are packed. Activation recomputation restores that scope
while suppressing tensor-observation callbacks inside the restored context, so a
logical forward is observed only once. Frozen router parameters skip unused
parameter gradients while preserving gradients into hidden states.

MTP uses dense branch inputs and its existing heads. It does not share the MTP
prefix. Shared-prefix MTP runs only in training; an evaluation-mode shared forward
skips it, as `compute_mtp_loss=False` does. A training forward rebuilds the dense
branches from `input_ids`, requires `loss_mask`, and rejects `decoder_input` and
`mtp_input_mask`. A pack, with or without explicit `mtp_loss_group_root_counts`,
also requires `calculate_per_token_loss=True`. With `compute_mtp_loss=False`,
these MTP-only checks are skipped. Each rank sums its prompt copies' gradients in
a fixed FP32 order over the prompt rows only; at CP>1 the per-rank partials are
combined by the CP reduce-scatter. A shared rerun is therefore as reproducible as a
dense one. The backward peak memory of this gather exceeds that of a plain
`index_select` only by an FP32 and a BF16 copy of the prompt rows. MTP MoE layers
run under the same fixed router row-block scope as the backbone.

When several independently normalized groups share one forward, `process_mtp_loss`
takes their CP-local `loss_group_lengths` and normalizes each group as a packed
dense forward of that group alone would. A pack keeps one group per tree unless
`mtp_loss_group_root_counts` merges consecutive trees. Ordinary MTP still accepts
precomputed decoder embeddings and upstream CP layout preparation.

## Supported scope and explicit guards

The shared adapter currently targets complete PP1 hybrid models with fp16/bf16,
zero dropout, RoPE or no positional embedding, ordinary self-attention with
vanilla softmax, MambaMixer layers with gated RMSNorm, and MLP or MoE layers.
TP greater than one requires sequence parallelism and TP-sharded output logits.
Full activation recomputation supports the uniform method.

`validate_shared_prefix_config(config)` in
`megatron.core.models.hybrid.shared_prefix` holds the rules that need only the
model configuration. An integration can call it at setup, before it builds a
model. `HybridModel.forward` validates a request once, before the embedding. It
runs the same configuration rules. It adds the rules that need the built model
or the forward: the process groups, the layer types, the checks that each Mamba
and attention layer owns (kernel requirements among them), the arguments and
the global physical length of the forward, and the `HybridModel` constructor
arguments `position_embedding_type`, `hybrid_layer_pattern` (for its MTP
layers) and `hybrid_layer_config_list`.

The configuration does not hold the layer pattern. The configuration rules on
attention settings, such as selective core-attention recomputation and the
QK-clip and max-logit statistics, therefore apply to every model, including one
without attention layers.

Unsupported combinations fail explicitly, including `hybrid_layer_config_list`
(use `hybrid_layer_pattern`), custom hybrid layers, dynamic context parallelism,
hash routing, quantization recipes, fp8/fp4, wide
residual streams, mHC, multi-latent attention, attention-logit softcapping,
QK-clip statistics, selective core-attention recomputation, sliding-window
attention, external attention/padding masks, inference contexts, fine-grained
activation offloading, CUDA graphs, and batch-invariant mode. At CP greater than
one, Mamba requires `linear_cp_layout='zigzag'`. For MoE, the adapter rejects
load balancing that replaces top-k routing (such as sinkhorn), nonzero
auxiliary-loss and z-loss coefficients, input jitter, randomized forced routing,
expert-capacity token dropping, per-rank capacity token dropping
(`moe_expert_rank_capacity_factor`), the flex dispatcher's `ncclep` backend,
routing replay, and training MLP chunking. Auxiliary-loss balancing types with
zero coefficients, the MCore default, are accepted because they leave top-k
routing unchanged. See the validation functions for the complete runtime
contract.

## Validation status

GPU tests compare shared execution with the dense rows `[prompt, completion_g]` it
replaces:

- `tests/unit_tests/shared_prefix/hybrid/test_shared_prefix_model_parity.py`:
  HybridModel logits, parameter gradients and expert-bias counts for trees and
  packs at TP1/CP1, TP2/SP/CP2 and TP1/CP4; and full recomputation against the plain
  shared forward.
- `tests/unit_tests/shared_prefix/hybrid/test_shared_prefix_moe_mtp.py` and
  `test_shared_prefix_mtp_branches.py`: expert-bias counts through a real router,
  MTP parity and grouped normalization at TP1/CP1 and TP2/SP/CP2, the fixed-order
  prompt-row gather, and the MTP input guards.
- `tests/unit_tests/shared_prefix/test_multi_token_prediction_loss_groups.py`: one
  grouped `process_mtp_loss` call against one call per loss group, on CPU.
- `tests/unit_tests/shared_prefix/ssm/test_shared_prefix_mamba_numerics.py` and
  `test_shared_prefix_mamba_backends.py`: the four differentiable Mamba backends
  against dense rows, including branch lengths in the defective `causal_conv1d`
  length classes, and backend selection.
- `tests/unit_tests/shared_prefix/test_router_shared_prefix.py` and
  `tests/unit_tests/tensor_parallel/test_random.py`: routing arguments, logical
  expert counts, fixed-row router blocks, the frozen-router backward, and the
  forward context restored during recomputation.
- `tests/unit_tests/shared_prefix/hybrid/test_shared_prefix_guards.py` and
  `tests/unit_tests/shared_prefix/test_shared_prefix_layout.py`: the explicit
  guards, with the configuration rules checked both without a stack and through
  the stack validation, the layout metadata, and the order of Mamba module hooks,
  on CPU.

In FP32, shared and dense execution differ only in summation order. The layer
tests bound that difference at 1e-5 relative L2 for MoE and 5e-5 for Mamba. Their
FP32 runs use local PyTorch linear layers with TF32 disabled. Transformer Engine
runs FP32 GEMMs in TF32 whatever PyTorch's TF32 flags say, which leaves a relative
error floor of about 1e-4 to 3e-4. An FP32 reference built from Transformer Engine
modules needs `NVIDIA_TF32_OVERRIDE=0`, and FP32 Triton kernels need
`TRITON_F32_DEFAULT=ieee`.

BF16 shared and dense runs do not produce identical gradients. The two layouts
add the same terms in different orders, so each rounds differently. In an MoE
model a rounding difference can flip a top-k expert choice, and a flipped choice
moves the gradient far more than the rounding itself. Two dense runs with
different packings disagree for the same reason. Repeating one dense run with the
same packing (an A/A comparison) measures only run-to-run nondeterminism, so it
is the wrong null for a shared-versus-dense comparison. The right null is dense
against dense with a different packing, or each run's distance from a
high-precision reference with routing held fixed.

The BF16 tests use that reference: an FP32 dense run of the same
BF16-representable weights, with MoE top-k choices replayed from a fixed
per-token table so that rounding cannot flip them. Shared BF16 execution must stay
within 1.25x of the dense BF16 error against that reference (1.5x for a single
Mamba layer). At TP/CP, where no FP32 reference exists, the shared-versus-dense
gap must stay within 1.5x of the same model's TP1/CP1 gap. Expert-bias counts
must match exactly. The model tests use an init std of 0.1: at Megatron's default
of 0.02 attention is nearly uniform, and wrong RoPE positions stay below BF16
rounding.

The attention and Mamba kernels are registered in the
[kernel manifest](../../tests/unit_tests/determinism/kernels/manifest.py) and
replayed bit-exactly in `test_shared_prefix_attention.py` and
`test_shared_prefix_mamba_kernels.py` under `tests/unit_tests/determinism/kernels/`;
see the [determinism testing requirements](determinism/testing.md). The attention
tests also compare outputs and gradients with an FP64 reference, within twice the
error of ordinary FlashAttention over the dense branches, use real NCCL exchanges
at CP1/2/4, and drive the shared-prefix branch of `SelfAttention.forward`,
checking that the forest kernel replaces core attention.

These tests qualify layer- and model-level numerics and determinism on small
models. They include no whole-model gradient comparison at a large training
topology, such as TP2/CP4 with sequence parallelism and MTP. They do not qualify
RL training quality, which the integrating framework must establish.
