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
    training_module._seqlen_stats_recording_enabled = True


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
    args.gdp_num_householder = 3
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


class TestGatedDeltaProductFlops:
    """GDP FLOPs must use the Householder count from the model configuration."""

    def test_householder_count_changes_flops(self):
        args = _make_hybrid_args()
        args.spec = ["megatron.core.models.hybrid.hybrid_layer_specs", "gdp_stack_spec"]
        batch_size = 4

        flops_m3 = num_floating_point_operations(args, batch_size)
        args.gdp_num_householder = 4
        flops_m4 = num_floating_point_operations(args, batch_size)

        total_tokens = batch_size * args.seq_length
        d_inner = args.mamba_num_heads * args.mamba_head_dim
        group_state_dim = args.mamba_num_groups * args.mamba_state_dim
        forward_delta_per_layer = (
            2
            * total_tokens
            * (
                args.hidden_size * (d_inner + group_state_dim + args.mamba_num_heads)
                + 4 * (d_inner + group_state_dim)
            )
            + 4 * total_tokens * d_inner * args.mamba_state_dim
        )
        num_gdp_layers = 2
        expected_delta = 3 * num_gdp_layers * forward_delta_per_layer

        assert flops_m4 - flops_m3 == expected_delta


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


class _FakeChunk:
    """Stand-in for one virtual model chunk. Only ``vp_stage`` matters here."""

    def __init__(self, vp_stage):
        self.vp_stage = vp_stage


class _FakeModuleWrapper:
    """Stand-in for DDP / Float16Module: exposes the chunk through ``.module``.

    ``get_attr_wrapped_model`` walks this chain, so the gate must find
    ``vp_stage`` on the wrapped chunk exactly as it does in production.
    """

    def __init__(self, module):
        self.module = module


class _ChunkWithoutVpStage:
    """A model class that defines no ``vp_stage`` and has no ``.module``.

    ``MimoModel`` is the in-tree example; it only acquires a ``vp_stage`` when
    ``Float16Module`` wraps it, which happens for fp16/bf16 runs only. The gate
    must degrade to "not interleaved" instead of raising.
    """


def _make_unguarded_forward_step(cu_seqlens, seen=None):
    """Build a user ``forward_step`` that records UNCONDITIONALLY.

    This is what ``pretrain_gpt.py`` / ``pretrain_hybrid.py`` actually ship: no
    ``vp_stage`` guard at the call site. The signature matches every shape the
    schedules use -- ``(data_iterator, model)``,
    ``(data_iterator, model, checkpoint_activations_microbatch)`` and
    ``(data_iterator, model, return_schedule_plan=True)``.
    """

    def forward_step(data_iterator, model, *args, **kwargs):
        if seen is not None:
            seen.append((model, args, kwargs))
        update_seqlen_stats_from_cu_seqlens(cu_seqlens)
        return "output_tensor", "loss_func"

    return forward_step


def _record_like_forward_step(cu_seqlens, vp_stage, wrap=True):
    """Drive the REAL gate over an unguarded user ``forward_step``.

    This exercises the shipped mechanism rather than a copy of it:
    ``train_step`` and ``evaluate`` wrap the user callable with
    ``_gate_seqlen_stats_by_vp_stage``, and the schedule then invokes it once
    per (micro-batch, model chunk) with a SINGLE chunk as the second positional
    argument. ``wrap=False`` reproduces the un-gated legacy behaviour.
    """
    forward_step = _make_unguarded_forward_step(cu_seqlens)
    if wrap:
        forward_step = training_module._gate_seqlen_stats_by_vp_stage(forward_step)
    return forward_step(iter([]), _FakeModuleWrapper(_FakeChunk(vp_stage)))


class TestAccumulatorVirtualPipeline:
    """Interleaved (virtual) pipeline parallelism must not multiply the stats.

    With ``virtual_pipeline_model_parallel_size = V``, the schedule calls the
    user ``forward_step`` once per (micro-batch, model chunk) and every chunk
    observes an identical micro-batch through its own data iterator. The
    whole-model FLOPs formula already covers all ``args.num_layers``, so only
    the primary chunk may contribute -- otherwise reported FLOPs inflate by
    exactly ``V``. The gate lives in ``megatron/training/training.py``: entry
    points keep an unguarded ``update_seqlen_stats_from_cu_seqlens`` call and
    ``train_step`` / ``evaluate`` wrap their ``forward_step_func`` with
    ``_gate_seqlen_stats_by_vp_stage``. These tests drive that real wrapper.
    """

    def setup_method(self):
        _reset_seqlen_accumulator()

    def teardown_method(self):
        _reset_seqlen_accumulator()

    def test_only_virtual_stage_zero_records(self):
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        # Simulate VPP=4: the same micro-batch is seen by four model chunks.
        for vp_stage in range(4):
            _record_like_forward_step(cu, vp_stage)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        # Counted once, not four times.
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2

    def test_none_vp_stage_means_no_interleaving(self):
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        _record_like_forward_step(cu, None)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2

    def test_non_primary_chunk_alone_records_nothing(self):
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        _record_like_forward_step(cu, 1)
        assert training_module._seqlen_stats_active is False
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens is None
        assert seqlen_squared_sum is None

    def test_multiple_microbatches_on_stage_zero_accumulate(self):
        """VPP gating must not drop legitimate per-micro-batch accumulation."""
        cu_a = torch.tensor([0, 100, 300], dtype=torch.int32)
        cu_b = torch.tensor([0, 50], dtype=torch.int32)
        for cu in (cu_a, cu_b):
            for vp_stage in range(2):  # VPP=2
                _record_like_forward_step(cu, vp_stage)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == (100 + 200) + 50
        assert seqlen_squared_sum == (100**2 + 200**2) + 50**2

    def test_unwrapped_forward_step_keeps_legacy_behaviour(self):
        """Without the wrapper the gate stays open, i.e. exactly today's ``main``.

        This pins that the module default is permissive: an entry point that
        drives the schedule itself, or a direct call from a notebook or a unit
        test, records every time. (It also reproduces the V-fold inflation this
        PR fixes, which is why ``train_step`` / ``evaluate`` install the gate.)
        """
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        for vp_stage in range(4):
            _record_like_forward_step(cu, vp_stage, wrap=False)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == 4 * (100 + 200)
        assert seqlen_squared_sum == 4 * (100**2 + 200**2)

    def test_gate_is_scoped_to_the_wrapped_call(self):
        """The override must not leak past the forward step it applies to.

        A non-primary chunk closes the gate only for the duration of its own
        forward step; a recorder running afterwards -- outside any wrapped
        callable -- must still record. Otherwise the stats would silently be
        dropped for the rest of the iteration, since the interleaved schedule
        always ends on the LAST chunk.
        """
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        _record_like_forward_step(cu, 3)  # last chunk of a VPP=4 run
        assert training_module._seqlen_stats_recording_enabled is True
        update_seqlen_stats_from_cu_seqlens(cu)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2

    def test_gate_is_restored_when_forward_step_raises(self):
        """The rerun state machine retries failed steps; a raise must not stick."""

        def exploding_forward_step(data_iterator, model, *args, **kwargs):
            raise RuntimeError("boom")

        gated = training_module._gate_seqlen_stats_by_vp_stage(exploding_forward_step)
        with pytest.raises(RuntimeError, match="boom"):
            gated(iter([]), _FakeModuleWrapper(_FakeChunk(2)))
        assert training_module._seqlen_stats_recording_enabled is True

    def test_model_without_vp_stage_does_not_raise(self):
        """``get_attr_wrapped_model`` raises when nothing in the chain has the
        attribute (e.g. an fp32 ``MimoModel``, which only gets ``vp_stage`` from
        ``Float16Module``). The gate must treat that as "not interleaved"
        instead of failing training over a reporting metric."""
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        gated = training_module._gate_seqlen_stats_by_vp_stage(_make_unguarded_forward_step(cu))
        gated(iter([]), _ChunkWithoutVpStage())
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2

    def test_vp_stage_is_read_through_the_module_chain(self):
        """DDP / FSDP / Float16Module nest the chunk under ``.module``."""
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        gated = training_module._gate_seqlen_stats_by_vp_stage(_make_unguarded_forward_step(cu))
        # Two levels of wrapping around a non-primary chunk.
        gated(iter([]), _FakeModuleWrapper(_FakeModuleWrapper(_FakeChunk(1))))
        assert training_module._seqlen_stats_active is False

    def test_wrapper_forwards_every_schedule_call_shape(self):
        """The schedules call the user step with 2 or 3 positional args, or with
        ``return_schedule_plan=True``. All must pass through untouched."""
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)
        seen = []
        gated = training_module._gate_seqlen_stats_by_vp_stage(
            _make_unguarded_forward_step(cu, seen=seen)
        )
        chunk = _FakeModuleWrapper(_FakeChunk(0))
        assert gated(iter([]), chunk) == ("output_tensor", "loss_func")
        gated(iter([]), chunk, 0.5)  # checkpoint_activations_microbatch
        gated(iter([]), chunk, return_schedule_plan=True)
        assert [(a, k) for _, a, k in seen] == [
            ((), {}),
            ((0.5,), {}),
            ((), {"return_schedule_plan": True}),
        ]

    def test_wrapper_preserves_identity_and_is_idempotent(self):
        """``functools.wraps`` keeps the user step introspectable, and wrapping
        an already-gated callable must not nest a second gate."""

        def my_forward_step(data_iterator, model):
            """Docstring."""
            return None

        gated = training_module._gate_seqlen_stats_by_vp_stage(my_forward_step)
        assert gated.__name__ == "my_forward_step"
        assert gated.__doc__ == "Docstring."
        assert training_module._gate_seqlen_stats_by_vp_stage(gated) is gated

    @pytest.mark.parametrize("vp_size", [1, 2, 4, 8])
    def test_reported_flops_are_invariant_to_virtual_pipeline_size(self, vp_size):
        """The reported FLOPs must not depend on how the model is chunked.

        The interleaved schedule runs ``forward_step`` once per (micro-batch,
        model chunk) and every chunk observes the same ``cu_seqlens`` for a
        given micro-batch, while the closed-form formula already spans all
        ``args.num_layers``. Chunking the model therefore must not change the
        answer. Without the gate this fails with exactly ``vp_size``-fold
        inflation.
        """
        args = _make_gpt_args()
        num_microbatches = 3
        cu = torch.tensor([0, 100, 300], dtype=torch.int32)

        # ``virtual_pipeline_model_parallel_size == 1`` is normalized to None in
        # arguments.py, and the non-interleaved schedule never sets a vp_stage.
        for _ in range(num_microbatches):
            for chunk in range(vp_size):
                _record_like_forward_step(cu, None if vp_size == 1 else chunk)
        total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()

        # Each micro-batch counted exactly once, whatever vp_size is.
        assert total_real_tokens == num_microbatches * (100 + 200)
        assert seqlen_squared_sum == num_microbatches * (100**2 + 200**2)

        flops = num_floating_point_operations(
            args,
            num_microbatches,
            seqlen_squared_sum_in_batch=seqlen_squared_sum,
            total_real_tokens_in_batch=total_real_tokens,
        )

        # ... and the user-visible number matches the unchunked reference.
        _reset_seqlen_accumulator()
        for _ in range(num_microbatches):
            _record_like_forward_step(cu, None)
        ref_tokens, ref_squared_sum = consume_seqlen_stats_in_iteration()
        reference_flops = num_floating_point_operations(
            args,
            num_microbatches,
            seqlen_squared_sum_in_batch=ref_squared_sum,
            total_real_tokens_in_batch=ref_tokens,
        )
        assert flops == reference_flops


class TestAccumulatorDistributed:
    """SUM all-reduce scoped to the pure data-parallel group (CP excluded).

    Each rank in a DP group sees identical ``cu_seqlens`` (broadcast across model
    parallelism), so a sum over the pure-DP group -- ``TP``/``CP``/``PP`` peers
    excluded -- visits each distinct micro-batch exactly once and yields the
    global-batch total with no divisor. CP is excluded because CP ranks share the
    same ``cu_seqlens`` and would double-count. When the DP group holds a single
    rank the collective is skipped entirely. Run with at least 2 ranks via
    ``torchrun --nproc_per_node=2``; the mixed TP x DP case needs >= 4.
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
        # Pure DP: TP=CP=PP=1, so the DP group spans the world and every rank's
        # contribution sums.
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

    def test_mixed_tp_dp_reduces_over_the_dp_group(self):
        """TP x DP: the sum spans the DP dim only, so TP replicas cannot inflate it.

        Also pins the *scope*: exactly one all-reduce, issued on the pure-DP
        group. A regression to a world reduce (even with a compensating divisor)
        or to the CP-inclusive group changes the group identity and fails here.
        """
        from megatron.core import mpu
        from tests.unit_tests.test_utilities import Utils

        if Utils.world_size < 4 or Utils.world_size % 2 != 0:
            pytest.skip("requires an even world size >= 4")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )

        dp_group = mpu.get_data_parallel_group(with_context_parallel=False)
        dp_size = Utils.world_size // 2
        assert dp_size == dp_group.size()

        # Every rank contributes the same micro-batch; the two TP peers of each
        # DP rank are duplicates and must be counted once, not twice.
        cu = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')
        update_seqlen_stats_from_cu_seqlens(cu)

        original_all_reduce = torch.distributed.all_reduce
        groups = []

        def spy(tensor, *args, **kwargs):
            groups.append(kwargs.get('group', args[1] if len(args) > 1 else None))
            return original_all_reduce(tensor, *args, **kwargs)

        torch.distributed.all_reduce = spy
        try:
            total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        finally:
            torch.distributed.all_reduce = original_all_reduce

        per_rank_sum = 100 + 200
        per_rank_sum_sq = 100**2 + 200**2
        assert groups == [dp_group], "consume must reduce exactly once, over the pure-DP group"
        assert total_real_tokens == per_rank_sum * dp_size
        assert seqlen_squared_sum == per_rank_sum_sq * dp_size

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

    def test_single_rank_dp_group_skips_collective(self):
        """With a 1-rank DP group the local value is already the global-batch
        total, so ``consume_*`` must return it *without* any collective. The
        skip predicate is a global topology property, so every rank agrees and
        nobody is stranded waiting on a peer's all-reduce."""
        from megatron.core import mpu
        from tests.unit_tests.test_utilities import Utils

        if Utils.world_size < 2:
            pytest.skip("requires >= 2 ranks")
        # TP spans the world -> the pure-DP group has exactly one member.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=Utils.world_size, pipeline_model_parallel_size=1
        )
        assert mpu.get_data_parallel_world_size(with_context_parallel=False) == 1

        cu = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')
        update_seqlen_stats_from_cu_seqlens(cu)

        original_all_reduce = torch.distributed.all_reduce
        calls = []

        def spy(tensor, *args, **kwargs):
            calls.append(tensor)
            return original_all_reduce(tensor, *args, **kwargs)

        torch.distributed.all_reduce = spy
        try:
            total_real_tokens, seqlen_squared_sum = consume_seqlen_stats_in_iteration()
        finally:
            torch.distributed.all_reduce = original_all_reduce

        assert calls == [], "consume must not issue all_reduce for a 1-rank DP group"
        assert total_real_tokens == 100 + 200
        assert seqlen_squared_sum == 100**2 + 200**2


# 8-GPU topology matrix. Each tuple is ``(tp, cp, pp)`` with ``dp = 8 / (tp*cp*pp)``.
# The matrix covers every model-parallel dim in isolation and the pairwise /
# three-way combinations that fit in 8 GPUs. This pins the contract that:
#   - ``cu_seqlens`` is broadcast-replicated across the TP/CP/PP dims (every
#     rank within one DP group accumulates the same value), and
#   - ``consume_*`` recovers the global DP-summed value by summing over the
#     pure-DP group, which visits each distinct micro-batch exactly once.
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
    rank, and asserts the global sum matches the closed-form expectation.
    Catches regressions where the reduction group picks up a replicated dim --
    folding CP in, for instance, doubles every ``cp2`` row.

    Skipped unless launched with ``torchrun --nproc_per_node 8``.
    """

    def setup_method(self):
        _reset_seqlen_accumulator()

    def teardown_method(self):
        from tests.unit_tests.test_utilities import Utils

        _reset_seqlen_accumulator()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp,cp,pp", _TOPOLOGY_8GPU_PARAMS)
    def test_dp_sum_across_topology(self, tp, cp, pp):
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
        # produce the same value -- the replication consume() must not sum.
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
