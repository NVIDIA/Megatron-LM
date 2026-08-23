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
        seq_length=8,
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


# s * b = 16, s * b * h = 64 for the shared configuration above.
SEQ_MICRO_BATCH = 8 * 2
SEQ_MICRO_BATCH_HIDDEN = SEQ_MICRO_BATCH * 4
# Attention block, both LayerNorms, residuals and dropout masks.
ATTENTION_AND_NORM = 18 * SEQ_MICRO_BATCH_HIDDEN
# Input token ids and the embedding dropout mask, both with pp_size microbatches in flight.
EMBEDDING = 8 * 8 * 2 + SEQ_MICRO_BATCH_HIDDEN
# Inputs to the output layer and the CE loss, only present when pp_size == 1.
OUTPUT_LAYER = SEQ_MICRO_BATCH_HIDDEN * 4 * (1 + (32 / 4))


def test_dense_model_matches_paper_formula():
    """A dense model with ffn_hidden_size == 4 * hidden_size must reproduce 34 * s * b * h."""
    args = _make_args(num_experts=None, moe_shared_expert_intermediate_size=None)

    per_layer_memory = 34 * SEQ_MICRO_BATCH_HIDDEN
    expected_memory = per_layer_memory * args.num_layers + EMBEDDING + OUTPUT_LAYER

    assert math.isclose(compute_activation_memory(args, num_microbatches=1), expected_memory)


def test_moe_layer_uses_expert_ffn_hidden_size_and_router_topk():
    """The MoE layer's FFN activations follow moe_ffn_hidden_size, not ffn_hidden_size."""
    args = _make_args()

    dense_layer_memory = ATTENTION_AND_NORM + 4 * SEQ_MICRO_BATCH * 16
    # The shared expert still runs at hidden_size over every token.
    moe_layer_memory = ATTENTION_AND_NORM + 4 * SEQ_MICRO_BATCH * 4
    # Every token is dispatched to topk=2 experts.
    dispatched_tokens = 2 * SEQ_MICRO_BATCH
    # Permuted tokens plus the expert outputs held for the combine.
    dispatched_token_memory = 4 * dispatched_tokens * 4
    # FC1 output and activation output inside each expert.
    expert_intermediate_memory = 4 * dispatched_tokens * 8

    expected_memory = (
        dense_layer_memory
        + moe_layer_memory
        + dispatched_token_memory
        + expert_intermediate_memory
        + EMBEDDING
        + OUTPUT_LAYER
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

    # Permuted tokens (at hidden_size) plus the FC1 and activation outputs (at moe_ffn).
    per_topk_cost = 4 * SEQ_MICRO_BATCH * 4 + 4 * SEQ_MICRO_BATCH * 8

    assert math.isclose(topk_two - topk_one, per_topk_cost)
    assert math.isclose(topk_four - topk_two, 2 * per_topk_cost)


def test_expert_activations_shard_over_expert_parallelism():
    """Expert activations shrink with EP, and the expert FFN term additionally with ETP."""
    memories = [
        compute_activation_memory(
            _make_args(expert_model_parallel_size=ep_size, expert_tensor_parallel_size=etp_size),
            num_microbatches=1,
        )
        for ep_size, etp_size in ((1, 1), (2, 1), (2, 2), (4, 2))
    ]

    assert memories[0] > memories[1] > memories[2] > memories[3]


def test_expert_tensor_parallelism_does_not_shard_dispatched_tokens():
    """ETP shards the expert FFN intermediates but not the permuted token buffers."""
    etp_one = compute_activation_memory(
        _make_args(expert_tensor_parallel_size=1), num_microbatches=1
    )
    etp_two = compute_activation_memory(
        _make_args(expert_tensor_parallel_size=2), num_microbatches=1
    )

    dispatched_tokens = 2 * SEQ_MICRO_BATCH
    expert_intermediate_memory = 4 * dispatched_tokens * 8

    assert math.isclose(etp_one - etp_two, expert_intermediate_memory / 2)
