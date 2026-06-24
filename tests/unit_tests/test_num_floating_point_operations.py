# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``num_floating_point_operations`` and the packed-sequence
``sum(L_i ** 2)`` accumulator.

The TFLOPs formula for self-attention has a token-linear part (QKV / output
projections) and a core-attention L^2 part (``QK^T`` and ``softmax(QK^T) V``).
For unpacked BSHD with a full causal mask the L^2 work is exactly
``batch_size * seq_length^2``. For THD packed sequences with chunks of length
``L_i`` the work is ``sum_i(L_i^2)``, strictly less when the chunks are short.

These tests pin both code paths and the accumulator math.
"""

from types import SimpleNamespace

import pytest
import torch

import megatron.training.training as training_module
from megatron.training.training import (
    consume_seqlen_stats_in_iteration,
    num_floating_point_operations,
    update_seqlen_stats_from_cu_seqlens,
)


def _reset_seqlen_accumulator():
    """Tear down the per-iteration accumulator between tests."""
    training_module._seqlen_stats_in_iteration = None
    training_module._seqlen_stats_active = False


def _make_gpt_args(
    *,
    num_layers=4,
    hidden_size=512,
    num_attention_heads=8,
    seq_length=1024,
    swiglu=True,
    ffn_hidden_size=None,
    padded_vocab_size=32000,
):
    """Minimal args for a dense MHA Transformer (no GQA, no MoE, no MLA, no MTP)."""
    args = SimpleNamespace()
    args.num_layers = num_layers
    args.hidden_size = hidden_size
    args.num_attention_heads = num_attention_heads
    args.seq_length = seq_length
    args.padded_vocab_size = padded_vocab_size
    args.swiglu = swiglu
    args.ffn_hidden_size = ffn_hidden_size if ffn_hidden_size is not None else 4 * hidden_size
    args.kv_channels = hidden_size // num_attention_heads
    args.group_query_attention = False
    args.num_query_groups = num_attention_heads
    args.attention_output_gate = False
    args.multi_latent_attention = False
    # MoE / MTP disabled.
    args.num_experts = None
    args.moe_layer_freq = 1
    args.moe_router_topk = 0
    args.moe_ffn_hidden_size = None
    args.moe_latent_size = None
    args.moe_shared_expert_intermediate_size = None
    args.mtp_num_layers = None
    # Linear attention disabled.
    args.experimental_attention_variant = None
    args.linear_attention_freq = None
    args.linear_key_head_dim = None
    args.linear_value_head_dim = None
    args.linear_num_key_heads = None
    args.linear_num_value_heads = None
    args.linear_conv_kernel_dim = None
    # MLA fields (unused but referenced).
    args.q_lora_rank = None
    args.qk_head_dim = None
    args.qk_pos_emb_head_dim = None
    args.kv_lora_rank = None
    args.v_head_dim = None
    # Not a hybrid model.
    args.hybrid_layer_pattern = None
    return args


def _make_hybrid_args(*, num_layers=4, hidden_size=512, num_attention_heads=8, seq_length=1024):
    """Minimal args for a 2-attn + 2-mamba hybrid model."""
    args = _make_gpt_args(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        seq_length=seq_length,
    )
    # ``M`` = Mamba, ``*`` = attention, ``-`` = MLP.
    args.hybrid_layer_pattern = "*M*M"
    args.mamba_state_dim = 128
    args.mamba_head_dim = 64
    args.mamba_num_groups = 8
    args.mamba_num_heads = 128
    return args


class TestBSHDBackwardCompat:
    """For unpacked BSHD, the new optional arg must not change the result."""

    def test_default_matches_explicit_bshd(self):
        args = _make_gpt_args()
        batch_size = 8

        default_flops = num_floating_point_operations(args, batch_size)
        explicit_flops = num_floating_point_operations(
            args,
            batch_size,
            seqlen_squared_sum_in_batch=batch_size * args.seq_length * args.seq_length,
        )

        assert default_flops == explicit_flops

    def test_hybrid_default_matches_explicit_bshd(self):
        args = _make_hybrid_args()
        batch_size = 4

        default_flops = num_floating_point_operations(args, batch_size)
        explicit_flops = num_floating_point_operations(
            args,
            batch_size,
            seqlen_squared_sum_in_batch=batch_size * args.seq_length * args.seq_length,
        )

        assert default_flops == explicit_flops

    def test_mla_default_matches_explicit_bshd(self):
        """MLA self-attention also splits into token-linear + L^2 parts."""
        args = _make_gpt_args(num_attention_heads=8)
        args.multi_latent_attention = True
        args.group_query_attention = False
        args.q_lora_rank = None
        args.qk_head_dim = 64
        args.qk_pos_emb_head_dim = 32
        args.kv_lora_rank = 256
        args.v_head_dim = 64
        batch_size = 4

        default_flops = num_floating_point_operations(args, batch_size)
        explicit_flops = num_floating_point_operations(
            args,
            batch_size,
            seqlen_squared_sum_in_batch=batch_size * args.seq_length * args.seq_length,
        )

        assert default_flops == explicit_flops


class TestTHDScaling:
    """Only the L^2 attention term should depend on ``seqlen_squared_sum_in_batch``."""

    def test_doubling_seqlen_squared_sum_increases_only_attention(self):
        args = _make_gpt_args()
        batch_size = 8
        bshd_sum = batch_size * args.seq_length * args.seq_length

        flops_bshd = num_floating_point_operations(
            args, batch_size, seqlen_squared_sum_in_batch=bshd_sum
        )
        flops_doubled = num_floating_point_operations(
            args, batch_size, seqlen_squared_sum_in_batch=2 * bshd_sum
        )

        delta = flops_doubled - flops_bshd
        # The delta is exactly the BSHD core-attention contribution.
        # Compute that contribution independently from the formula:
        # 4 * num_layers * h_q_proj * fwd_bwd(3) * fma(2) / 2 * 2 * sum(L^2)
        # = 6 * num_layers * (kv_channels * num_attention_heads) * sum(L^2).
        q_proj_size = args.kv_channels * args.num_attention_heads
        expected_one_bshd_core = 6 * args.num_layers * q_proj_size * bshd_sum
        assert delta == expected_one_bshd_core

    def test_thd_packed_below_bshd_when_chunks_shorter(self):
        """A packed batch with shorter chunks does less attention work."""
        args = _make_gpt_args()
        batch_size = 8
        s = args.seq_length

        # 1 packed sample of length s, sliced into 4 equal real chunks of s/4 each.
        sum_l_sq = 4 * (s // 4) ** 2  # sum(L_i^2) per sample
        thd_sum = batch_size * sum_l_sq

        bshd_sum = batch_size * s * s

        flops_thd = num_floating_point_operations(
            args, batch_size, seqlen_squared_sum_in_batch=thd_sum
        )
        flops_bshd = num_floating_point_operations(args, batch_size)

        # THD must be strictly less than BSHD (attention contribution shrinks).
        assert flops_thd < flops_bshd
        # The L^2 work is 1/4 of BSHD (4 chunks of s/4); the rest is unchanged.
        q_proj_size = args.kv_channels * args.num_attention_heads
        expected_savings = 6 * args.num_layers * q_proj_size * (bshd_sum - thd_sum)
        assert flops_bshd - flops_thd == expected_savings

    def test_thd_zero_seqlen_squared_sum_removes_core_attn(self):
        args = _make_gpt_args()
        batch_size = 8

        flops_no_core = num_floating_point_operations(
            args, batch_size, seqlen_squared_sum_in_batch=0
        )
        flops_default = num_floating_point_operations(args, batch_size)

        q_proj_size = args.kv_channels * args.num_attention_heads
        bshd_sum = batch_size * args.seq_length * args.seq_length
        expected_core = 6 * args.num_layers * q_proj_size * bshd_sum

        assert flops_default - flops_no_core == expected_core


class TestHybridTHDScaling:
    """The hybrid attn_layer_flops path also must respond to seqlen_squared_sum."""

    def test_hybrid_thd_below_bshd(self):
        args = _make_hybrid_args()
        batch_size = 4
        s = args.seq_length

        thd_sum = batch_size * 4 * (s // 4) ** 2  # 4 chunks of s/4
        flops_thd = num_floating_point_operations(
            args, batch_size, seqlen_squared_sum_in_batch=thd_sum
        )
        flops_bshd = num_floating_point_operations(args, batch_size)

        assert flops_thd < flops_bshd

    def test_hybrid_attention_layers_count(self):
        """Mamba/MLP/MoE layers are L-linear, so the L^2 delta is exactly the
        attention layers' core-attention contribution."""
        args = _make_hybrid_args()
        batch_size = 4

        bshd_sum = batch_size * args.seq_length * args.seq_length
        flops_bshd = num_floating_point_operations(args, batch_size)
        flops_doubled = num_floating_point_operations(
            args, batch_size, seqlen_squared_sum_in_batch=2 * bshd_sum
        )

        # Pattern "*M*M" -> 2 attention layers.
        num_attn_layers = 2
        # attn_layer_flops core part: 2 * sum(L^2) * h * p, with p = num_heads*kv_channels/h
        # = 2 * sum(L^2) * kv_channels * num_heads.
        # Then fwd+bwd = *3.
        h = args.hidden_size
        n = args.num_attention_heads
        kv = args.kv_channels
        expected_delta_per_layer_per_unit_sum = 2 * kv * n * 3  # *3 for fwd+bwd
        expected_delta = num_attn_layers * expected_delta_per_layer_per_unit_sum * bshd_sum
        assert flops_doubled - flops_bshd == expected_delta

    def test_mla_attention_variants_are_counted(self):
        """Regression: DSv4 MLA attention variants (CSA 'C', HCA 'H', Window 'W',
        DS_ATTENTION 'D') must contribute attention FLOPs. Previously only the
        plain ATTENTION symbol '*' was counted, so a pattern made entirely of
        C/H/W layers reported ZERO attention FLOPs -- roughly halving the
        throughput estimate vs. the equivalent gpt_model."""
        args = _make_hybrid_args()
        args.multi_latent_attention = True
        # MLA dims (DSv4-like; values only need to be self-consistent and positive).
        args.q_lora_rank = 256
        args.kv_lora_rank = 32
        args.qk_head_dim = 64
        args.qk_pos_emb_head_dim = 32
        args.v_head_dim = 64
        batch_size = 4

        # Two MLA attention layers (C, H) + two Mamba layers.
        args.hybrid_layer_pattern = "CMHM"
        flops_with_attn = num_floating_point_operations(args, batch_size)
        # Same shape but the attention layers replaced by Mamba (no attention at all).
        args.hybrid_layer_pattern = "MMMM"
        flops_no_attn = num_floating_point_operations(args, batch_size)

        # The C/H layers must add real attention FLOPs (the old bug made this 0).
        assert flops_with_attn > flops_no_attn


class TestPaddingRemoval:
    """``total_real_tokens_in_batch`` removes padding from token-linear FLOPs.

    With THD, the dataloader pads sequences for CP alignment and for
    end-of-sequence packing. The padded slot count (``batch_size *
    args.seq_length``) over-counts both kinds of padding as useful compute. By
    threading the real token count ``sum_i(L_i)`` through every token-linear
    term (MLP, MoE, projections, MTP, logits) we report only useful FLOPs.
    """

    def test_default_total_tokens_matches_bshd(self):
        """When ``total_real_tokens_in_batch`` is ``None`` the default is
        ``batch_size * args.seq_length``, recovering the old BSHD result."""
        args = _make_gpt_args()
        batch_size = 8
        default_flops = num_floating_point_operations(args, batch_size)
        explicit_flops = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=batch_size * args.seq_length,
            seqlen_squared_sum_in_batch=batch_size * args.seq_length * args.seq_length,
        )
        assert default_flops == explicit_flops

    def test_lower_total_tokens_reduces_token_linear_flops(self):
        """Halving the real token count must halve every token-linear term.
        The core-attention L^2 term is unchanged (we hold ``seqlen_sq`` fixed)."""
        args = _make_gpt_args()
        batch_size = 8
        full_tokens = batch_size * args.seq_length
        full_sum_sq = batch_size * args.seq_length * args.seq_length

        flops_full = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens,
            seqlen_squared_sum_in_batch=full_sum_sq,
        )
        flops_half = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens // 2,
            seqlen_squared_sum_in_batch=full_sum_sq,
        )

        # The token-linear part should halve; the L^2 term is the same in
        # both calls, so the difference equals 1/2 of the token-linear part.
        # In particular: flops_full > flops_half AND flops_half > full_sum_sq
        # contribution alone (because the L^2 term is unaffected).
        assert flops_half < flops_full
        # Token-linear part of ``flops_full`` is ``flops_full - L2_contrib``.
        # ``flops_half`` = (token_linear_full / 2) + L2_contrib.
        # So ``2 * flops_half - flops_full == L2_contrib``.
        q_proj_size = args.kv_channels * args.num_attention_heads
        l2_contrib = 6 * args.num_layers * q_proj_size * full_sum_sq
        assert 2 * flops_half - flops_full == l2_contrib

    def test_padding_removal_independent_of_attention(self):
        """Removing only the projection/MLP padding (``total_real_tokens``
        drops) must NOT change the core-attention contribution. Pin that the
        two parameters are independent."""
        args = _make_gpt_args()
        batch_size = 8
        full_tokens = batch_size * args.seq_length
        full_sum_sq = batch_size * args.seq_length * args.seq_length

        # Fix sum_sq (attention work); vary token count (projection work).
        flops_a = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens,
            seqlen_squared_sum_in_batch=full_sum_sq,
        )
        flops_b = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens * 3 // 4,  # 25% padding
            seqlen_squared_sum_in_batch=full_sum_sq,
        )
        # Difference comes purely from the token-linear delta.
        per_token_linear_factor = (flops_a - flops_b) / (full_tokens - full_tokens * 3 // 4)
        # Sanity check it's positive and that a 1-token swing scales linearly.
        flops_c = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens - 1,
            seqlen_squared_sum_in_batch=full_sum_sq,
        )
        assert flops_a - flops_c == pytest.approx(per_token_linear_factor)

    def test_hybrid_padding_removal(self):
        """The hybrid path also threads ``total_tokens`` through every layer
        helper (mamba, gdn, mlp, moe, attn projections, logits)."""
        args = _make_hybrid_args()
        batch_size = 4
        full_tokens = batch_size * args.seq_length
        full_sum_sq = batch_size * args.seq_length * args.seq_length

        flops_full = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens,
            seqlen_squared_sum_in_batch=full_sum_sq,
        )
        flops_half = num_floating_point_operations(
            args,
            batch_size,
            total_real_tokens_in_batch=full_tokens // 2,
            seqlen_squared_sum_in_batch=full_sum_sq,
        )
        # Token-linear contribution halves; L^2 attention term is unchanged.
        assert flops_half < flops_full


class TestAccumulator:
    """``update_seqlen_stats_from_cu_seqlens`` and ``consume_seqlen_stats_in_iteration``."""

    def setup_method(self):
        _reset_seqlen_accumulator()

    def teardown_method(self):
        _reset_seqlen_accumulator()

    def test_update_computes_both_stats(self):
        # cu_seqlens [0, 100, 250, 400] -> lengths [100, 150, 150]
        cu = torch.tensor([0, 100, 250, 400], dtype=torch.int32)
        update_seqlen_stats_from_cu_seqlens(cu)
        expected_sum = 100 + 150 + 150
        expected_sum_sq = 100**2 + 150**2 + 150**2
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == expected_sum
        assert seqlen_squared_sum == expected_sum_sq

    def test_update_accumulates_across_microbatches(self):
        cu1 = torch.tensor([0, 100, 200], dtype=torch.int32)  # sum=200, sum^2=20000
        cu2 = torch.tensor([0, 50, 250], dtype=torch.int32)  # sum=250, sum^2=42500
        update_seqlen_stats_from_cu_seqlens(cu1)
        update_seqlen_stats_from_cu_seqlens(cu2)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == 200 + 250
        assert seqlen_squared_sum == 20000 + 42500

    def test_consume_resets_accumulator(self):
        cu = torch.tensor([0, 100, 200], dtype=torch.int32)
        update_seqlen_stats_from_cu_seqlens(cu)
        _ = consume_seqlen_stats_in_iteration()
        # After draining, next consume must report BSHD (no work seen) by
        # returning ``(None, None)`` so ``num_floating_point_operations`` takes
        # the closed-form defaults.
        assert consume_seqlen_stats_in_iteration() == (None, None)

    def test_no_updates_returns_none(self):
        """BSHD path: never calling update must NOT issue a collective. The
        flag stays ``False`` and consume returns ``(None, None)``."""
        assert consume_seqlen_stats_in_iteration() == (None, None)
        # Flag stayed False -> the GPU tensor was never even allocated.
        assert training_module._seqlen_stats_in_iteration is None
        assert training_module._seqlen_stats_active is False

    def test_update_none_cu_seqlens_is_noop(self):
        update_seqlen_stats_from_cu_seqlens(None)
        # Still BSHD (no real update happened).
        assert consume_seqlen_stats_in_iteration() == (None, None)
        assert training_module._seqlen_stats_active is False

    def test_update_single_entry_cu_seqlens_is_noop(self):
        """``cu_seqlens.numel() < 2`` (no real chunks) must be ignored."""
        update_seqlen_stats_from_cu_seqlens(torch.tensor([0], dtype=torch.int32))
        assert consume_seqlen_stats_in_iteration() == (None, None)
        assert training_module._seqlen_stats_active is False

    def test_bshd_equivalent_when_chunks_fill_seq_length(self):
        """A packed batch with one chunk of length s per sample matches BSHD."""
        batch_size = 4
        s = 1024
        # Each "sample" is one packed sequence of one chunk of length s.
        for _ in range(batch_size):
            cu = torch.tensor([0, s], dtype=torch.int32)
            update_seqlen_stats_from_cu_seqlens(cu)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == batch_size * s
        assert seqlen_squared_sum == batch_size * s * s

    def test_unpadded_cu_seqlens_excludes_padding(self):
        """When the dataloader pads (cu_seqlens_padded > cu_seqlens), passing the
        REAL cu_seqlens to update() makes both stats reflect only real tokens."""
        # 2 real chunks of length 100 + 200 = 300 tokens, padded slot of 400.
        cu_real = torch.tensor([0, 100, 300], dtype=torch.int32)
        # cu_padded would be [0, 128, 400] in production (chunk pad + end pad),
        # but the accumulator must only see ``cu_real``.
        update_seqlen_stats_from_cu_seqlens(cu_real)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        # Real token count, NOT 400 (padded slot size).
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2

    def test_update_keeps_accumulator_on_gpu_when_input_on_gpu(self):
        """No per-micro-batch CPU sync: the accumulator tensor lives on the
        device of the first ``cu_seqlens`` we see. Only the final consume()
        moves data to host."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        cu = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')
        update_seqlen_stats_from_cu_seqlens(cu)
        tensor = training_module._seqlen_stats_in_iteration
        assert tensor is not None
        assert tensor.is_cuda
        assert tensor.shape == (2,)  # [sum_L, sum_L_sq]
        assert training_module._seqlen_stats_active is True
        # Drain.
        _ = consume_seqlen_stats_in_iteration()
        # Tensor stays allocated for reuse, but the flag flips back to False.
        assert training_module._seqlen_stats_active is False
        assert training_module._seqlen_stats_in_iteration is not None
        assert training_module._seqlen_stats_in_iteration.tolist() == [0.0, 0.0]


class TestAccumulatorDistributed:
    """All-reduce + ``TP*CP*PP`` deduplication.

    Each rank in a DP group sees identical ``cu_seqlens`` (broadcast across model
    parallelism). The world all-reduce therefore overcounts by ``TP * CP * PP``,
    which the consume helper divides back out. Run with at least 2 ranks via
    ``torchrun --nproc_per_node=2``.
    """

    def setup_method(self):
        _reset_seqlen_accumulator()

    def teardown_method(self):
        from tests.unit_tests.test_utilities import Utils

        _reset_seqlen_accumulator()
        Utils.destroy_model_parallel()

    def test_pure_dp_sums_across_ranks(self):
        from tests.unit_tests.test_utilities import Utils

        if Utils.world_size < 2:
            pytest.skip("requires >= 2 ranks")
        # Pure DP: TP=CP=PP=1, world = DP. No deduplication, every rank's contribution sums.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        # Each rank simulates its own micro-batch with chunks [100, 200].
        cu = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')
        update_seqlen_stats_from_cu_seqlens(cu)

        per_rank_sum = 100 + 200
        per_rank_sum_sq = 100**2 + 200**2
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == per_rank_sum * Utils.world_size
        assert seqlen_squared_sum == per_rank_sum_sq * Utils.world_size

    def test_pure_tp_deduplicates(self):
        """All TP ranks have the same cu_seqlens; deduplication divides the world sum."""
        from tests.unit_tests.test_utilities import Utils

        if Utils.world_size < 2:
            pytest.skip("requires >= 2 ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=Utils.world_size, pipeline_model_parallel_size=1
        )

        cu = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')
        update_seqlen_stats_from_cu_seqlens(cu)

        # All TP ranks updated the same value; after world all_reduce we get
        # TP * (per_rank) and divide by TP -> per_rank.
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2

    def test_bshd_path_skips_collective(self):
        """If no rank ever calls ``update_*``, ``consume_*`` must return
        ``(None, None)`` *without* issuing any collective. A spy on
        ``all_reduce`` catches a regression that would otherwise hang in
        production when one rank is in THD mode and another in BSHD (the
        current contract assumes all ranks agree)."""
        from tests.unit_tests.test_utilities import Utils

        if Utils.world_size < 2:
            pytest.skip("requires >= 2 ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        original_all_reduce = torch.distributed.all_reduce
        calls = []

        def spy(tensor, *args, **kwargs):
            calls.append(tensor)
            return original_all_reduce(tensor, *args, **kwargs)

        torch.distributed.all_reduce = spy
        try:
            result = consume_seqlen_stats_in_iteration()
        finally:
            torch.distributed.all_reduce = original_all_reduce

        assert result == (None, None)
        assert calls == [], "consume must not issue all_reduce when no update happened"


# 8-GPU topology matrix. Each tuple is ``(tp, cp, pp)`` with ``dp = 8 / (tp*cp*pp)``.
# The matrix covers every model-parallel dim in isolation and the pairwise /
# three-way combinations that fit in 8 GPUs. This pins the contract that:
#   - ``cu_seqlens`` is broadcast-replicated across the TP/CP/PP dims (every
#     rank within one DP group accumulates the same value), and
#   - ``consume_*`` recovers the global DP-summed value by all-reducing across
#     the world and dividing by ``TP * CP * PP``.
_TOPOLOGY_8GPU_PARAMS = [
    # (tp, cp, pp)
    pytest.param(1, 1, 1, id="dp8"),
    pytest.param(2, 1, 1, id="tp2_dp4"),
    pytest.param(1, 2, 1, id="cp2_dp4"),
    pytest.param(1, 1, 2, id="pp2_dp4"),
    pytest.param(2, 2, 1, id="tp2_cp2_dp2"),
    pytest.param(2, 1, 2, id="tp2_pp2_dp2"),
    pytest.param(1, 2, 2, id="cp2_pp2_dp2"),
    pytest.param(2, 2, 2, id="tp2_cp2_pp2_dp1"),
]


class TestAccumulatorTopology:
    """End-to-end correctness across the (TP, CP, PP, DP) matrix on 8 GPUs.

    Production invariant: within one DP group all ranks (TP * CP * PP of them)
    see the SAME ``cu_seqlens`` because it is broadcast across the
    model-parallel dimensions; across DP groups the data differs. The test
    simulates that by making every rank's contribution depend ONLY on its DP
    rank, and asserts the deduplicated global sum matches the closed-form
    expectation. Catches regressions where any of TP/CP/PP is dropped from the
    dedup factor.

    Skipped unless launched with ``torchrun --nproc_per_node 8``.
    """

    def setup_method(self):
        _reset_seqlen_accumulator()

    def teardown_method(self):
        from tests.unit_tests.test_utilities import Utils

        _reset_seqlen_accumulator()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp,cp,pp", _TOPOLOGY_8GPU_PARAMS)
    def test_dedup_across_topology(self, tp, cp, pp):
        from megatron.core import mpu
        from tests.unit_tests.test_utilities import Utils

        if Utils.world_size != 8:
            pytest.skip(f"requires exactly 8 ranks; got {Utils.world_size}")
        if tp * cp * pp > Utils.world_size:
            pytest.skip(f"tp*cp*pp={tp*cp*pp} > world_size={Utils.world_size}")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, context_parallel_size=cp
        )

        dp_size = Utils.world_size // (tp * cp * pp)
        assert dp_size == mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()

        # Per-DP-group ``cu_seqlens``: a 2-chunk packed sequence whose lengths
        # depend on ``dp_rank`` so that every DP group contributes a DIFFERENT
        # ``sum(L)`` AND ``sum(L^2)``. Every rank in the same DP group must
        # produce the same value -- that's what the consume() dedup unwinds.
        len_a = 100 * (dp_rank + 1)
        len_b = 200 * (dp_rank + 1)
        cu = torch.tensor([0, len_a, len_a + len_b], dtype=torch.int32, device='cuda')
        update_seqlen_stats_from_cu_seqlens(cu)

        # Closed-form expected: sum over DP groups of (len_a + len_b) and
        # (len_a^2 + len_b^2). With len_a = 100*(r+1), len_b = 200*(r+1) -->
        # sum_L per DP = 300*(r+1), sum_L_sq per DP = 50000*(r+1)^2.
        expected_total_tokens = sum(300 * (r + 1) for r in range(dp_size))
        expected_sum_sq = sum(50000 * (r + 1) ** 2 for r in range(dp_size))
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == pytest.approx(expected_total_tokens), (
            f"topology tp={tp} cp={cp} pp={pp} dp={dp_size}: "
            f"got total_real_tokens={total_real_tokens}, expected {expected_total_tokens}"
        )
        assert seqlen_squared_sum == pytest.approx(expected_sum_sq), (
            f"topology tp={tp} cp={cp} pp={pp} dp={dp_size}: "
            f"got seqlen_squared_sum={seqlen_squared_sum}, expected {expected_sum_sq}"
        )


def _make_dsv4_args():
    """Minimal args for a DSv4-hybrid MLA model with sparse attention.

    4 layers with compress_ratios [0, 4, 128, 128] (1 r0, 1 r4, 2 r128).
    No MoE / MTP to keep the golden reference simple.
    """
    args = _make_gpt_args(
        num_layers=4,
        hidden_size=512,
        num_attention_heads=8,
        seq_length=256,
        ffn_hidden_size=2048,
        padded_vocab_size=1024,
    )
    args.multi_latent_attention = True
    args.group_query_attention = False
    args.q_lora_rank = 128
    args.qk_head_dim = 64
    args.qk_pos_emb_head_dim = 32
    args.kv_lora_rank = 64
    args.v_head_dim = 64
    args.o_lora_rank = 64
    args.o_groups = 2
    args.experimental_attention_variant = "dsv4_hybrid"
    args.csa_window_size = 64
    args.csa_compress_ratios = [0, 4, 128, 128]
    args.dsa_indexer_n_heads = 4
    args.dsa_indexer_head_dim = 32
    args.dsa_indexer_topk = 16
    return args


def _dsv4_golden_flops(args, total_tokens, seqlen_squared_sum):
    """Independent golden calculator for DSv4-hybrid FLOPs.

    Reimplements the formula from ``num_floating_point_operations`` so that the
    test does not just call the same code twice. Assumes no MoE / MTP.
    """
    fwd_bwd = 3
    fma = 2
    ffn_exp = 3 if args.swiglu else 2

    # ---- MLA projections (token-linear, per layer) ----
    q_term = args.q_lora_rank * (args.hidden_size + args.num_attention_heads * args.v_head_dim + 1)
    kv_term = args.hidden_size * args.v_head_dim + args.v_head_dim
    o_term = (
        args.num_attention_heads * args.v_head_dim * args.o_lora_rank
        + args.o_groups * args.o_lora_rank * args.hidden_size
    )
    mla_proj_per_layer = fwd_bwd * fma * (q_term + kv_term + o_term)

    # ---- DSv4 sparse attention extra (token-linear + L^2) ----
    ratios = args.csa_compress_ratios
    n_r0 = sum(1 for r in ratios if r == 0)
    n_r4 = sum(1 for r in ratios if r == 4)
    n_r128 = sum(1 for r in ratios if r == 128)
    nh = args.num_attention_heads
    vhd = args.v_head_dim
    w = args.csa_window_size

    # Token-linear sparse attention
    sparse_r0 = n_r0 * nh * w * vhd * 2
    sparse_r128_win = n_r128 * nh * w * vhd * 2
    if n_r4 > 0:
        eff_topk = min(args.dsa_indexer_topk, args.seq_length // 4)
        avg_comp_4 = eff_topk * (1 - eff_topk * 4 / (2 * args.seq_length))
        sparse_r4 = n_r4 * nh * (w + avg_comp_4) * vhd * 2
        idx_tok = (
            n_r4 * args.hidden_size * (2 * args.dsa_indexer_head_dim) * 2
            + n_r4 * args.q_lora_rank * args.dsa_indexer_n_heads * args.dsa_indexer_head_dim
            + n_r4 * args.hidden_size * args.dsa_indexer_n_heads
        )
        idx_core = n_r4 * args.dsa_indexer_n_heads * args.dsa_indexer_head_dim / 4
    else:
        sparse_r4, idx_tok, idx_core = 0, 0, 0

    compressor = n_r4 * args.hidden_size * (2 * vhd) * 2 + n_r128 * args.hidden_size * (1 * vhd) * 2
    dsv4_token = fwd_bwd * fma * (sparse_r0 + sparse_r4 + sparse_r128_win + compressor + idx_tok)
    # L^2 core: r=128 compressed-KV + r=4 indexer scoring
    r128_core = n_r128 * nh * vhd / 128
    dsv4_core = fwd_bwd * fma * (r128_core + idx_core)

    # ---- Aggregation ----
    num_layers = args.num_layers
    self_attn_term = mla_proj_per_layer * num_layers + dsv4_token
    self_attn_core_term = dsv4_core  # standard core is 0 for DSv4

    mlp = fwd_bwd * fma * args.hidden_size * (args.ffn_hidden_size * ffn_exp * num_layers)
    logit = fwd_bwd * fma * args.hidden_size * args.padded_vocab_size

    return total_tokens * (mlp + self_attn_term + logit) + seqlen_squared_sum * self_attn_core_term


class TestDSv4Hybrid:
    """DSv4 hybrid sparse-attention FLOPs against an independent golden calculator."""

    def test_bshd(self):
        """BSHD (uniform sequences) must match the golden calculator."""
        args = _make_dsv4_args()
        batch_size = 2
        total_tokens = batch_size * args.seq_length
        sum_sq = batch_size * args.seq_length**2

        flops = num_floating_point_operations(args, batch_size)
        expected = _dsv4_golden_flops(args, total_tokens, sum_sq)
        assert flops == expected

    def test_thd(self):
        """THD (packed variable-length subsequences) must match the golden
        calculator and be strictly less than BSHD due to the L^2 sparse-attention
        components (r=128 compressed-KV, r=4 indexer scoring)."""
        args = _make_dsv4_args()
        batch_size = 2
        packed_lengths = [64, 64, 128, 256]
        total_tokens = sum(packed_lengths)
        thd_sum_sq = sum(L**2 for L in packed_lengths)

        flops = num_floating_point_operations(
            args,
            batch_size,
            seqlen_squared_sum_in_batch=thd_sum_sq,
            total_real_tokens_in_batch=total_tokens,
        )
        expected = _dsv4_golden_flops(args, total_tokens, thd_sum_sq)
        assert flops == expected
        # THD must be strictly less than BSHD.
        bshd_flops = num_floating_point_operations(args, batch_size)
        assert flops < bshd_flops
