# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from types import SimpleNamespace

from megatron.training.theoretical_memory_usage import compute_activation_memory


def _make_args(**overrides):
    args = SimpleNamespace(
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        ffn_hidden_size=16,
        hidden_size=4,
        micro_batch_size=2,
        moe_ffn_hidden_size=8,
        moe_latent_size=None,
        moe_layer_freq=[0, 1],
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=4,
        num_experts=4,
        num_layers=2,
        padded_vocab_size=32,
        pipeline_model_parallel_size=1,
        recompute_granularity="selective",
        recompute_modules=["core_attn"],
        seq_length=8,
        swiglu=False,
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


# s * b = 16 and s * b * h = 64 for the shared configuration above.
SEQ_MICRO_BATCH = 8 * 2
SEQ_MICRO_BATCH_HIDDEN = SEQ_MICRO_BATCH * 4
# Attention block, both LayerNorms, residuals and dropout masks.
ATTENTION_AND_NORM = 18 * SEQ_MICRO_BATCH_HIDDEN
# Input token ids and the embedding dropout mask, both with pp_size microbatches in flight.
EMBEDDING = 8 * 8 * 2 + SEQ_MICRO_BATCH_HIDDEN
# Inputs to the output layer and the CE loss, only present when pp_size == 1.
OUTPUT_LAYER = SEQ_MICRO_BATCH_HIDDEN * 4 * (1 + (32 / 4))


def _mlp_intermediate(ffn_hidden_size, gated=False, recompute_activation=False):
    """FC1 output (doubled by a gated linear unit) plus the activation output, 2 bytes each."""
    return 2 * ffn_hidden_size * (2 if gated else 1) + (
        0 if recompute_activation else 2 * ffn_hidden_size
    )


def test_dense_model_matches_paper_formula():
    """A non-gated dense model with ffn == 4 * hidden must reproduce 34 * s * b * h."""
    args = _make_args(num_experts=None, moe_shared_expert_intermediate_size=None)

    per_layer_memory = 34 * SEQ_MICRO_BATCH_HIDDEN
    expected_memory = per_layer_memory * args.num_layers + EMBEDDING + OUTPUT_LAYER

    assert math.isclose(compute_activation_memory(args, num_microbatches=1), expected_memory)


def test_moe_layer_uses_expert_ffn_hidden_size_and_router_topk():
    """The MoE layer's FFN activations follow moe_ffn_hidden_size, not ffn_hidden_size."""
    args = _make_args()

    dense_layer = ATTENTION_AND_NORM + SEQ_MICRO_BATCH * _mlp_intermediate(16)
    # The shared expert still runs at hidden_size over every token.
    moe_layer = ATTENTION_AND_NORM + SEQ_MICRO_BATCH * _mlp_intermediate(4)
    # Every token is dispatched to topk=2 experts; etp=1 here so no extra gather factor.
    routed_token_copies = 2 * SEQ_MICRO_BATCH
    # Permuted tokens plus the expert outputs held for the combine.
    dispatched = 4 * routed_token_copies * 4
    # FC1 output and activation output inside each expert.
    expert_intermediate = routed_token_copies * _mlp_intermediate(8)

    expected_memory = (
        dense_layer + moe_layer + dispatched + expert_intermediate + EMBEDDING + OUTPUT_LAYER
    )

    assert math.isclose(compute_activation_memory(args, num_microbatches=1), expected_memory)


def test_moe_activation_memory_ignored_without_experts():
    """MoE arguments must not affect a model that has no experts."""
    baseline = compute_activation_memory(
        _make_args(num_experts=None, moe_shared_expert_intermediate_size=None), num_microbatches=1
    )
    with_unused_moe_args = compute_activation_memory(
        _make_args(
            num_experts=None,
            moe_shared_expert_intermediate_size=None,
            moe_ffn_hidden_size=4096,
            moe_router_topk=8,
        ),
        num_microbatches=1,
    )

    assert math.isclose(baseline, with_unused_moe_args)


def test_moe_activation_memory_scales_linearly_with_router_topk():
    """Each extra routed copy of the batch adds the same dispatch and expert-FFN footprint."""
    topk_one = compute_activation_memory(_make_args(moe_router_topk=1), num_microbatches=1)
    topk_two = compute_activation_memory(_make_args(moe_router_topk=2), num_microbatches=1)
    topk_four = compute_activation_memory(_make_args(moe_router_topk=4), num_microbatches=1)

    per_topk_cost = 4 * SEQ_MICRO_BATCH * 4 + SEQ_MICRO_BATCH * _mlp_intermediate(8)

    assert math.isclose(topk_two - topk_one, per_topk_cost)
    assert math.isclose(topk_four - topk_two, 2 * per_topk_cost)


def test_expert_activations_are_invariant_to_expert_model_parallelism():
    """EP is carved out of the data-parallel dimension, so every EP rank keeps its own
    microbatch and dispatch leaves the per-rank token count unchanged."""
    memories = [
        compute_activation_memory(
            _make_args(expert_model_parallel_size=ep_size), num_microbatches=1
        )
        for ep_size in (1, 2, 8, 64)
    ]

    assert all(math.isclose(m, memories[0]) for m in memories)


def test_expert_tensor_parallelism_gathers_dispatched_tokens():
    """ETP replicates the dispatched tokens while sharding the expert FFN width, so raising
    ETP grows only the dispatch buffer."""
    etp_one = compute_activation_memory(
        _make_args(expert_tensor_parallel_size=1), num_microbatches=1
    )
    etp_two = compute_activation_memory(
        _make_args(expert_tensor_parallel_size=2), num_microbatches=1
    )

    routed_token_copies = 2 * SEQ_MICRO_BATCH
    assert math.isclose(etp_two - etp_one, 4 * routed_token_copies * 4)


def test_tensor_parallelism_shards_every_term():
    """All activations, routed-expert terms included, are partitioned by TP."""
    tp_one = compute_activation_memory(_make_args(tensor_model_parallel_size=1), num_microbatches=1)
    tp_four = compute_activation_memory(
        _make_args(tensor_model_parallel_size=4), num_microbatches=1
    )

    assert math.isclose(tp_four, tp_one / 4)


def test_gated_linear_unit_widens_every_mlp_intermediate():
    """SwiGLU doubles each FC1 output, so the dense, shared-expert and routed-expert
    intermediates all grow by half."""
    plain = _make_args(swiglu=False)
    gated = _make_args(swiglu=True)

    routed_token_copies = 2 * SEQ_MICRO_BATCH
    extra = (
        SEQ_MICRO_BATCH * 16  # dense FC1 output doubles
        + SEQ_MICRO_BATCH * 4  # shared expert FC1 output doubles
        + routed_token_copies * 8  # routed expert FC1 output doubles
    ) * 2  # 2 bytes per element

    assert math.isclose(
        compute_activation_memory(gated, num_microbatches=1) - extra,
        compute_activation_memory(plain, num_microbatches=1),
    )


def test_recompute_modules_drop_the_terms_they_recompute():
    """Selective recompute of a submodule removes exactly that submodule's saved tensors."""
    baseline = compute_activation_memory(_make_args(), num_microbatches=1)
    routed_token_copies = 2 * SEQ_MICRO_BATCH

    shared_dropped = compute_activation_memory(
        _make_args(recompute_modules=["core_attn", "shared_experts"]), num_microbatches=1
    )
    assert math.isclose(baseline - shared_dropped, SEQ_MICRO_BATCH * _mlp_intermediate(4))

    act_dropped = compute_activation_memory(
        _make_args(recompute_modules=["core_attn", "moe_act"]), num_microbatches=1
    )
    assert math.isclose(baseline - act_dropped, routed_token_copies * 2 * 8)

    moe_dropped = compute_activation_memory(
        _make_args(recompute_modules=["core_attn", "moe"]), num_microbatches=1
    )
    expected_drop = (
        SEQ_MICRO_BATCH * _mlp_intermediate(4)
        + 4 * routed_token_copies * 4
        + routed_token_copies * _mlp_intermediate(8)
    )
    assert math.isclose(baseline - moe_dropped, expected_drop)

    mlp_dropped = compute_activation_memory(
        _make_args(recompute_modules=["core_attn", "mlp"]), num_microbatches=1
    )
    assert math.isclose(baseline - mlp_dropped, SEQ_MICRO_BATCH * _mlp_intermediate(16))


def test_latent_moe_routes_experts_through_latent_size():
    """With moe_latent_size set, the dispatched tokens are held at the latent width."""
    full = compute_activation_memory(_make_args(), num_microbatches=1)
    latent = compute_activation_memory(_make_args(moe_latent_size=2), num_microbatches=1)

    routed_token_copies = 2 * SEQ_MICRO_BATCH
    assert math.isclose(full - latent, 4 * routed_token_copies * (4 - 2))
