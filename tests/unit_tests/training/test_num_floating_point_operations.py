# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLOPs must use the decoder's final attention type for every MTP depth."""

from types import SimpleNamespace

import pytest

from megatron.training.training import num_floating_point_operations


def _make_args(**overrides):
    args = SimpleNamespace(
        attention_output_gate=True,
        experimental_attention_variant="gdn",
        ffn_hidden_size=16,
        group_query_attention=True,
        hidden_size=8,
        hybrid_layer_pattern=None,
        kv_channels=4,
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=3,
        moe_ffn_hidden_size=10,
        moe_latent_size=None,
        moe_layer_freq=[0, 1] * 20,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=4,
        mtp_num_layers=None,
        multi_latent_attention=False,
        num_attention_heads=2,
        num_experts=4,
        num_layers=40,
        num_query_groups=1,
        padded_vocab_size=32,
        seq_length=8,
        swiglu=True,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _expected_flops(tokens, squared_lengths, layer_counts, mtp_depth):
    """Hand-calculated coefficients for the fixed small model in _make_args."""
    gdn, full, dense, moe = layer_counts
    # All coefficients include forward/backward (3) and multiply/add (2).
    # Full: Q/K/V/gate widths 8/4/4/8, output 8 -> hidden 8.
    full_projections = 1536  # 6 * (8 * (8 + 4 + 4 + 8) + 8 * 8)
    full_core = 48  # 6 * query width 8, including causal masking.
    # GDN: input width 20, convolution 40, recurrence 72, output projection 48.
    gdn_attention = 1920  # 6 * (8 * 20 + 40 + 72 + 48)
    dense_mlp = 2304  # SwiGLU: 6 * hidden 8 * intermediate 16 * 3.
    moe_mlp = 3456  # Routed and shared: 6 * 8 * 3 * (10 * topk 2 + 4).
    mtp_norms_and_projection = 912  # 6 * (3 * 8 + 2 * 8 * 8).
    logits = 1536  # 6 * hidden 8 * vocabulary 32.
    return (
        tokens
        * (
            gdn * gdn_attention
            + full * full_projections
            + dense * dense_mlp
            + moe * moe_mlp
            + mtp_depth * mtp_norms_and_projection
            + (1 + mtp_depth) * logits
        )
        + squared_lengths * full * full_core
    )


def _batch_stats(layout):
    if layout == "bshd":
        # Three ordinary sequences of length eight, using the default arguments.
        return {}, 24, 192
    # Unequal packed subsequences [3, 5, 7], shorter than the BSHD allocation.
    return {"total_real_tokens_in_batch": 15, "seqlen_squared_sum_in_batch": 83}, 15, 83


@pytest.mark.parametrize("layout", ["bshd", "thd"])
@pytest.mark.parametrize("pattern_form", ["int", "list"])
@pytest.mark.parametrize(
    "num_layers,mtp_depth,expected_counts",
    [
        # Counts are (GDN, full attention, dense MLP, MoE MLP).
        # Forty decoder layers end in full attention and MoE.
        (40, None, (30, 10, 20, 20)),
        (40, 0, (30, 10, 20, 20)),
        (40, 1, (30, 11, 20, 21)),
        (40, 2, (30, 12, 20, 22)),
        # Three decoder layers end in GDN and a dense MLP. MTP stays GDN,
        # even though extending the integer frequency would next produce full attention.
        (3, None, (3, 0, 2, 1)),
        (3, 0, (3, 0, 2, 1)),
        (3, 1, (4, 0, 3, 1)),
        (3, 2, (5, 0, 4, 1)),
    ],
)
def test_mtp_repeats_final_decoder_attention_type(
    layout, pattern_form, num_layers, mtp_depth, expected_counts
):
    pattern = [1, 1, 1, 0] * 10 if num_layers == 40 else [1, 1, 1]
    original_pattern = pattern.copy()
    args = _make_args(
        num_layers=num_layers,
        mtp_num_layers=mtp_depth,
        linear_attention_freq=4 if pattern_form == "int" else pattern,
        moe_layer_freq=[0, 1] * 20 if num_layers == 40 else [0, 1, 0],
    )
    stats, tokens, squared_lengths = _batch_stats(layout)

    actual = num_floating_point_operations(args, batch_size=3, **stats)

    assert actual == _expected_flops(tokens, squared_lengths, expected_counts, mtp_depth or 0)
    # Keep T fixed and change only Q to isolate the number of full-attention layers.
    merged = num_floating_point_operations(
        args, batch_size=3, total_real_tokens_in_batch=tokens, seqlen_squared_sum_in_batch=tokens**2
    )
    assert merged - actual == 48 * expected_counts[1] * (tokens**2 - squared_lengths)
    # A list describes decoder layers only; counting MTP must not append to it.
    assert pattern == original_pattern


@pytest.mark.parametrize("layout", ["bshd", "thd"])
@pytest.mark.parametrize("mtp_depth", [None, 0, 1, 2])
def test_plain_attention_and_dense_mlp_flops_are_unchanged(layout, mtp_depth):
    args = _make_args(
        experimental_attention_variant=None,
        linear_attention_freq=None,
        num_layers=3,
        num_experts=None,
        mtp_num_layers=mtp_depth,
    )
    stats, tokens, squared_lengths = _batch_stats(layout)
    depth = mtp_depth or 0

    actual = num_floating_point_operations(args, batch_size=3, **stats)

    assert actual == _expected_flops(tokens, squared_lengths, (0, 3 + depth, 3 + depth, 0), depth)
