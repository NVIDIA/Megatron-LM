# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.training import (
    build_train_valid_test_data_iterators,
    num_floating_point_operations,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


class _LenDataloader:
    """Fake dataloader with __len__ (required by the full_validation path)
    and __iter__ (consumed via cyclic_iter)."""

    def __init__(self, data):
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def mock_multi_valid_full_datasets_provider(train_val_test_num_samples):
    return (iter([1]), [_LenDataloader([2, 2]), _LenDataloader([20, 20, 20])], iter([3]))


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.start_eval_at_iter = None
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def create_test_flops_args(**overrides):
    args = SimpleNamespace()
    args.hybrid_layer_pattern = None
    args.group_query_attention = True
    args.num_attention_heads = 4
    args.num_query_groups = 2
    args.num_experts = None
    args.num_layers = 3
    args.mtp_num_layers = None
    args.moe_ffn_hidden_size = None
    args.moe_layer_freq = None
    args.moe_router_topk = 1
    args.ffn_hidden_size = 32
    args.moe_latent_size = None
    args.moe_shared_expert_intermediate_size = None
    args.swiglu = False
    args.multi_latent_attention = False
    args.experimental_attention_variant = None
    args.linear_attention_freq = None
    args.linear_key_head_dim = None
    args.linear_value_head_dim = None
    args.linear_num_key_heads = None
    args.linear_num_value_heads = None
    args.linear_conv_kernel_dim = None
    args.kv_channels = 4
    args.q_lora_rank = None
    args.kv_lora_rank = None
    args.qk_head_dim = None
    args.qk_pos_emb_head_dim = None
    args.v_head_dim = None
    args.csa_compress_ratios = None
    args.csa_window_size = None
    args.seq_length = None
    args.dsa_indexer_n_heads = None
    args.dsa_indexer_head_dim = None
    args.dsa_indexer_topk = None
    args.o_lora_rank = None
    args.o_groups = None
    args.attention_output_gate = False
    args.hidden_size = 16
    args.padded_vocab_size = 64
    args.mamba_state_dim = 3
    args.mamba_head_dim = 2
    args.mamba_num_groups = 2
    args.mamba_num_heads = 4

    for key, value in overrides.items():
        setattr(args, key, value)

    return args


def sequence_statistics(sequence_lengths):
    return (sum(sequence_lengths), sum(seqlen * seqlen for seqlen in sequence_lengths))


def expected_attention_layer_flops(args, seqlen_sum, seqlen_squared_sum):
    p = args.kv_channels * args.num_attention_heads / args.hidden_size if args.kv_channels else 1
    num_query_groups = (
        args.num_query_groups if args.group_query_attention else args.num_attention_heads
    )
    return (
        4
        * args.hidden_size
        * p
        * (
            args.hidden_size * seqlen_sum
            + args.hidden_size * (num_query_groups / args.num_attention_heads) * seqlen_sum
            + seqlen_squared_sum / 2
        )
    )


def expected_mlp_layer_flops(args, seqlen_sum):
    scale_factor = 3.0 / 2.0 if args.swiglu else 1.0
    expansion = args.ffn_hidden_size / args.hidden_size
    return 4 * expansion * scale_factor * seqlen_sum * args.hidden_size**2


def expected_moe_layer_flops(args, seqlen_sum):
    scale_factor = 3.0 / 2.0 if args.swiglu else 1.0
    moe_ffn_hidden_size = (
        args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    )
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    if args.moe_latent_size is None:
        routed_flops = (
            4
            * seqlen_sum
            * args.hidden_size
            * moe_ffn_hidden_size
            * args.moe_router_topk
            * scale_factor
        )
    else:
        routed_flops = (
            4
            * seqlen_sum
            * args.moe_latent_size
            * moe_ffn_hidden_size
            * args.moe_router_topk
            * scale_factor
        )
        routed_flops += 4 * seqlen_sum * args.hidden_size * args.moe_latent_size
    shared_flops = 4 * seqlen_sum * args.hidden_size * shared_expert_ffn_hidden_size * scale_factor
    return routed_flops + shared_flops


def expected_mamba_layer_flops(args, seqlen_sum):
    d_in = 2 * args.hidden_size
    nheads = args.mamba_num_heads if args.mamba_num_heads else d_in // args.mamba_head_dim
    return (
        2
        * seqlen_sum
        * args.hidden_size
        * (2 * d_in + 2 * args.mamba_num_groups * args.mamba_state_dim + nheads)
        + 7 * seqlen_sum * d_in * args.mamba_state_dim
        + 2 * seqlen_sum * d_in * args.hidden_size
    )


def expected_gdn_layer_flops(args, seqlen_sum):
    qk_head_dim = args.linear_key_head_dim or 128
    v_head_dim = args.linear_value_head_dim or 128
    num_qk_heads = args.linear_num_key_heads or 16
    num_v_heads = args.linear_num_value_heads or 32
    conv_kernel_dim = args.linear_conv_kernel_dim or 4
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    return (
        2
        * seqlen_sum
        * (
            args.hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
            + conv_kernel_dim * (2 * qk_dim + v_dim)
            + num_v_heads * (v_head_dim**2) * 4
            + args.hidden_size * v_dim
        )
    )


def expected_mha_or_gqa_self_attention_terms(args):
    num_query_groups = (
        args.num_query_groups if args.group_query_attention else args.num_attention_heads
    )
    query_projection_size = args.kv_channels * args.num_attention_heads
    key_projection_size = args.kv_channels * num_query_groups
    value_projection_size = args.kv_channels * num_query_groups
    gate_projection_size = query_projection_size if args.attention_output_gate else 0
    standard_self_attn_term = (
        3
        * 2
        * (
            args.hidden_size
            * (
                query_projection_size
                + key_projection_size
                + value_projection_size
                + gate_projection_size
            )
            + query_projection_size * args.hidden_size
        )
    )
    standard_self_attn_core_term = 3 * 2 * query_projection_size / 2 * 2
    return standard_self_attn_term, standard_self_attn_core_term


def expected_mla_self_attention_terms(args):
    if args.experimental_attention_variant == "dsv4_hybrid":
        q_term = args.q_lora_rank * (
            args.hidden_size
            + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
            + 1
        )
        kv_term = args.hidden_size * args.v_head_dim + args.v_head_dim
        o_term = (
            args.num_attention_heads * args.v_head_dim * args.o_lora_rank
            + args.o_groups * args.o_lora_rank * args.hidden_size
        )
        return 3 * 2 * (q_term + kv_term + o_term), 0

    if args.q_lora_rank is None:
        q_term = (
            args.hidden_size
            * args.num_attention_heads
            * (args.qk_head_dim + args.qk_pos_emb_head_dim)
        )
    else:
        q_term = args.q_lora_rank * (
            args.hidden_size
            + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
            + 1
        )
    standard_self_attn_term = (
        3
        * 2
        * (
            q_term
            + args.kv_lora_rank
            * (
                args.hidden_size
                + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                + 1
            )
            + args.hidden_size * args.qk_pos_emb_head_dim
            + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
        )
    )
    standard_self_attn_core_term = (
        3
        * 2
        * (
            args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
            + args.num_attention_heads * args.v_head_dim / 2
        )
    )
    return standard_self_attn_term, standard_self_attn_core_term


def expected_transformer_layer_counts(args):
    if args.num_experts is None:
        num_dense_layers = args.num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
        last_layer_is_moe = 0
    else:
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        else:
            moe_layer_pattern = args.moe_layer_freq
        num_moe_layers = sum(moe_layer_pattern)
        num_dense_layers = args.num_layers - num_moe_layers
        num_experts_routed_to = args.moe_router_topk
        last_layer_is_moe = moe_layer_pattern[-1]

    mtp_num_layers = args.mtp_num_layers or 0
    num_moe_layers += last_layer_is_moe * mtp_num_layers
    num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
    return num_dense_layers, num_moe_layers, num_experts_routed_to, args.num_layers + mtp_num_layers


def expected_dsv4_hybrid_extra_term(args, num_layers):
    n_layers_r0 = sum(1 for ratio in args.csa_compress_ratios if ratio == 0)
    n_layers_r4 = sum(1 for ratio in args.csa_compress_ratios if ratio == 4)
    n_layers_r128 = sum(1 for ratio in args.csa_compress_ratios if ratio == 128)

    sparse_attn_r0 = (
        n_layers_r0 * args.num_attention_heads * args.csa_window_size * args.v_head_dim * 2
    )
    avg_comp_128 = (args.seq_length // 128) / 2
    sparse_attn_r128 = (
        n_layers_r128
        * args.num_attention_heads
        * (args.csa_window_size + avg_comp_128)
        * args.v_head_dim
        * 2
    )
    main_compressor_term = (
        n_layers_r4 * args.hidden_size * (2 * args.v_head_dim) * 2
        + n_layers_r128 * args.hidden_size * args.v_head_dim * 2
    )

    if n_layers_r4 > 0:
        effective_topk_4 = min(args.dsa_indexer_topk, args.seq_length // 4)
        avg_comp_4 = effective_topk_4 * (1 - effective_topk_4 * 4 / (2 * args.seq_length))
        sparse_attn_r4 = (
            n_layers_r4
            * args.num_attention_heads
            * (args.csa_window_size + avg_comp_4)
            * args.v_head_dim
            * 2
        )
        indexer_term = (
            n_layers_r4 * args.hidden_size * (2 * args.dsa_indexer_head_dim) * 2
            + n_layers_r4 * args.q_lora_rank * args.dsa_indexer_n_heads * args.dsa_indexer_head_dim
            + n_layers_r4 * args.hidden_size * args.dsa_indexer_n_heads
            + n_layers_r4
            * args.dsa_indexer_n_heads
            * args.dsa_indexer_head_dim
            * (args.seq_length // 4)
        )
    else:
        sparse_attn_r4 = 0
        indexer_term = 0

    sparse_attn_term = sparse_attn_r0 + sparse_attn_r4 + sparse_attn_r128
    assert len(args.csa_compress_ratios) == num_layers
    return 3 * 2 * (sparse_attn_term + main_compressor_term + indexer_term)


def expected_transformer_flops(args, sequence_lengths):
    seqlen_sum, seqlen_squared_sum = sequence_statistics(sequence_lengths)
    mtp_num_layers = args.mtp_num_layers or 0
    ffn_expansion_factor = 3 if args.swiglu else 2
    num_dense_layers, num_moe_layers, num_experts_routed_to, num_layers = (
        expected_transformer_layer_counts(args)
    )
    moe_ffn_hidden_size = (
        args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    )
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )

    if args.multi_latent_attention:
        standard_self_attn_term, standard_self_attn_core_term = expected_mla_self_attention_terms(
            args
        )
    else:
        standard_self_attn_term, standard_self_attn_core_term = (
            expected_mha_or_gqa_self_attention_terms(args)
        )

    dsv4_hybrid_extra_term = 0
    if args.experimental_attention_variant == "gated_delta_net":
        if isinstance(args.linear_attention_freq, int):
            linear_attention_pattern = [
                0 if ((i + 1) % args.linear_attention_freq == 0) else 1 for i in range(num_layers)
            ]
        else:
            linear_attention_pattern = args.linear_attention_freq
        num_linear_attention_layers = sum(linear_attention_pattern)
        num_standard_attention_layers = num_layers - num_linear_attention_layers
        qk_dim = args.linear_key_head_dim * args.linear_num_key_heads
        v_dim = args.linear_value_head_dim * args.linear_num_value_heads
        linear_self_attn_term = (
            3
            * 2
            * (
                args.hidden_size * (2 * qk_dim + 2 * v_dim + 2 * args.linear_num_value_heads)
                + args.linear_conv_kernel_dim * (2 * qk_dim + v_dim)
                + args.linear_num_value_heads * (args.linear_value_head_dim**2) * 4
                + args.hidden_size * v_dim
            )
        )
    elif args.experimental_attention_variant == "dsv4_hybrid":
        num_linear_attention_layers = 0
        linear_self_attn_term = 0
        num_standard_attention_layers = num_layers
        dsv4_hybrid_extra_term = expected_dsv4_hybrid_extra_term(args, num_layers)
    else:
        num_linear_attention_layers = 0
        linear_self_attn_term = 0
        num_standard_attention_layers = num_layers

    self_attn_term = (
        linear_self_attn_term * num_linear_attention_layers
        + standard_self_attn_term * num_standard_attention_layers
        + dsv4_hybrid_extra_term
    )
    self_attn_core_term = standard_self_attn_core_term * num_standard_attention_layers
    moe_term = (
        (moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor)
        if args.moe_latent_size is None
        else (
            moe_ffn_hidden_size
            * num_experts_routed_to
            * ffn_expansion_factor
            * args.moe_latent_size
            / args.hidden_size
            + 2 * args.moe_latent_size
        )
    )
    token_linear_flops = (
        3
        * 2
        * args.hidden_size
        * (
            (args.ffn_hidden_size * ffn_expansion_factor) * num_dense_layers
            + moe_term * num_moe_layers
            + (shared_expert_ffn_hidden_size * ffn_expansion_factor) * num_moe_layers
        )
        + self_attn_term
        + 3 * 2 * mtp_num_layers * (3 * args.hidden_size + 2 * args.hidden_size * args.hidden_size)
        + 3 * 2 * args.hidden_size * args.padded_vocab_size * (mtp_num_layers + 1)
    )
    return seqlen_sum * token_linear_flops + seqlen_squared_sum * self_attn_core_term


def expected_hybrid_flops(args, sequence_lengths):
    seqlen_sum, seqlen_squared_sum = sequence_statistics(sequence_lengths)
    pattern = args.hybrid_layer_pattern.replace("|", "").replace("/", "")
    flops_fwd = (
        pattern.count("*") * expected_attention_layer_flops(args, seqlen_sum, seqlen_squared_sum)
        + pattern.count("-") * expected_mlp_layer_flops(args, seqlen_sum)
        + pattern.count("M") * expected_mamba_layer_flops(args, seqlen_sum)
        + pattern.count("E") * expected_moe_layer_flops(args, seqlen_sum)
        + pattern.count("G") * expected_gdn_layer_flops(args, seqlen_sum)
        + 2 * seqlen_sum * args.hidden_size * args.padded_vocab_size * (1 + args.mtp_num_layers)
    )
    return flops_fwd * 3


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_build_train_valid_test_data_iterators_multi_full_validation(self):
        """multiple_validation_sets + full_validation builds a list of iterators
        (one per validation set) and sets args.eval_iters to the per-loader
        lengths MAX-reduced across DP ranks."""
        args = create_test_args()
        args.multiple_validation_sets = True
        args.full_validation = True
        set_args(args)
        _, valid_iters, _ = build_train_valid_test_data_iterators(
            mock_multi_valid_full_datasets_provider
        )
        assert isinstance(valid_iters, list)
        assert len(valid_iters) == 2
        assert next(valid_iters[0]) == 2
        assert next(valid_iters[1]) == 20
        # data_parallel_size=1, so MAX across DP ranks equals the local lengths
        assert args.eval_iters == [2, 3]

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def test_num_floating_point_operations_uses_sequence_statistics_api(self):
        parameters = list(inspect.signature(num_floating_point_operations).parameters)
        assert parameters == [
            "args",
            "seqlen_sum_this_global_batch",
            "seqlen_squared_sum_this_global_batch",
        ]

    def test_num_floating_point_operations_matches_sequence_statistics_formula(self):
        cases = [
            (
                "dense_gqa_even_lengths",
                create_test_flops_args(),
                [8, 8],
                expected_transformer_flops,
            ),
            (
                "dense_gqa_packed_lengths",
                create_test_flops_args(),
                [4, 12],
                expected_transformer_flops,
            ),
            (
                "dense_mha_mtp_swiglu_gate",
                create_test_flops_args(
                    group_query_attention=False,
                    num_query_groups=None,
                    attention_output_gate=True,
                    mtp_num_layers=1,
                    swiglu=True,
                ),
                [3, 5, 2, 6],
                expected_transformer_flops,
            ),
            (
                "moe_latent_shared_expert",
                create_test_flops_args(
                    num_experts=4,
                    moe_layer_freq=[1, 0, 1],
                    moe_router_topk=2,
                    moe_ffn_hidden_size=24,
                    moe_latent_size=6,
                    moe_shared_expert_intermediate_size=8,
                    mtp_num_layers=1,
                    swiglu=True,
                ),
                [7, 1, 4],
                expected_transformer_flops,
            ),
            (
                "mla_without_q_lora",
                create_test_flops_args(
                    multi_latent_attention=True,
                    group_query_attention=False,
                    num_query_groups=None,
                    q_lora_rank=None,
                    kv_lora_rank=5,
                    qk_head_dim=3,
                    qk_pos_emb_head_dim=1,
                    v_head_dim=4,
                ),
                [6, 3],
                expected_transformer_flops,
            ),
            (
                "mla_with_q_lora",
                create_test_flops_args(
                    multi_latent_attention=True,
                    group_query_attention=False,
                    num_query_groups=None,
                    q_lora_rank=7,
                    kv_lora_rank=5,
                    qk_head_dim=3,
                    qk_pos_emb_head_dim=1,
                    v_head_dim=4,
                ),
                [2, 5, 4],
                expected_transformer_flops,
            ),
            (
                "gated_delta_net",
                create_test_flops_args(
                    num_layers=4,
                    experimental_attention_variant="gated_delta_net",
                    linear_attention_freq=2,
                    linear_key_head_dim=3,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=3,
                    linear_conv_kernel_dim=5,
                ),
                [5, 1, 6],
                expected_transformer_flops,
            ),
            (
                "dsv4_hybrid",
                create_test_flops_args(
                    num_layers=3,
                    multi_latent_attention=True,
                    group_query_attention=False,
                    num_query_groups=None,
                    experimental_attention_variant="dsv4_hybrid",
                    q_lora_rank=7,
                    qk_head_dim=3,
                    qk_pos_emb_head_dim=1,
                    v_head_dim=4,
                    o_lora_rank=5,
                    o_groups=2,
                    csa_compress_ratios=[0, 4, 128],
                    csa_window_size=8,
                    seq_length=256,
                    dsa_indexer_n_heads=2,
                    dsa_indexer_head_dim=3,
                    dsa_indexer_topk=5,
                ),
                [4, 12],
                expected_transformer_flops,
            ),
            (
                "hybrid_all_layer_types",
                create_test_flops_args(
                    hybrid_layer_pattern="M*G-E/ME",
                    mtp_num_layers=1,
                    moe_router_topk=2,
                    moe_ffn_hidden_size=24,
                    moe_latent_size=6,
                    moe_shared_expert_intermediate_size=8,
                    swiglu=True,
                    linear_key_head_dim=3,
                    linear_value_head_dim=4,
                    linear_num_key_heads=2,
                    linear_num_value_heads=3,
                    linear_conv_kernel_dim=5,
                ),
                [5, 3, 4],
                expected_hybrid_flops,
            ),
        ]

        for name, args, sequence_lengths, expected_flops_fn in cases:
            expected_flops = expected_flops_fn(args, sequence_lengths)
            seqlen_sum_this_global_batch, seqlen_squared_sum_this_global_batch = (
                sequence_statistics(sequence_lengths)
            )

            actual_flops = num_floating_point_operations(
                SimpleNamespace(**vars(args)),
                seqlen_sum_this_global_batch,
                seqlen_squared_sum_this_global_batch,
            )

            assert actual_flops == pytest.approx(expected_flops), name

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )
