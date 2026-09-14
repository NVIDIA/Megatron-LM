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

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.training.training as training_module
from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec, hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
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


class TestHybridConfigListFlops:
    """List-defined HybridModels use each concrete config for FLOPs reporting."""

    @staticmethod
    def _base_config(num_layers=2):
        return TransformerConfig(
            num_layers=num_layers,
            hidden_size=512,
            num_attention_heads=8,
            ffn_hidden_size=2048,
            gated_linear_unit=True,
            use_cpu_initialization=True,
            is_hybrid_model=True,
        )

    @classmethod
    def _model(cls, configs, *, config=None, stack_spec=hybrid_stack_spec, vocab_size=32000):
        """Initialize only estimator metadata, without allocating decoder modules."""
        model = HybridModel.__new__(HybridModel)
        torch.nn.Module.__init__(model)
        model.config = config if config is not None else cls._base_config()
        model.hybrid_layer_config_list = tuple(configs) if configs is not None else None
        model.hybrid_stack_spec = stack_spec
        model.vocab_size = vocab_size
        return model

    @classmethod
    def _dsa_config(cls, num_layers=1):
        config = DSALayerConfig.from_config(cls._base_config(num_layers=num_layers))
        config.multi_latent_attention = True
        config.experimental_attention_variant = "dsa"
        config.q_lora_rank = 48
        config.kv_lora_rank = 96
        config.qk_head_dim = 64
        config.qk_pos_emb_head_dim = 32
        config.v_head_dim = 64
        config.dsa_indexer_n_heads = 3
        config.dsa_indexer_head_dim = 17
        config.dsa_indexer_topk = 7
        config.dsa_indexer_topk_freq = 1
        config.dsa_indexer_skip_topk_offset = 0
        return config

    def test_heterogeneous_moe_fields_change_flops(self):
        args = _make_hybrid_args(num_layers=2)
        args.hybrid_layer_pattern = None
        args.is_hybrid_model = True
        batch_size = 4
        total_tokens = batch_size * args.seq_length

        base = self._base_config()
        attention = AttentionLayerConfig.from_config(base)
        small_moe = MoELayerConfig.from_config(base)
        small_moe.moe_ffn_hidden_size = 1024
        small_moe.moe_router_topk = 2
        small_moe.moe_shared_expert_intermediate_size = None
        small_moe.moe_latent_size = None
        large_moe = MoELayerConfig.from_config(small_moe)
        large_moe.moe_ffn_hidden_size = 3072
        large_moe.moe_router_topk = 4

        small_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model([attention, small_moe]).estimate_flops,
        )
        large_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model([attention, large_moe]).estimate_flops,
        )

        scale_factor = 3 / 2  # gated MLP
        expected_forward_delta = (
            4
            * total_tokens
            * base.hidden_size
            * (
                large_moe.moe_ffn_hidden_size * large_moe.moe_router_topk
                - small_moe.moe_ffn_hidden_size * small_moe.moe_router_topk
            )
            * scale_factor
        )
        assert large_flops - small_flops == expected_forward_delta * 3

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    def test_mtp_heads_are_counted_as_executed_configs(self, mtp_use_repeated_layer):
        args = _make_hybrid_args(num_layers=1)
        args.hybrid_layer_pattern = None
        args.is_hybrid_model = True
        base = self._base_config(num_layers=1)
        base.mtp_use_repeated_layer = mtp_use_repeated_layer
        attention = AttentionLayerConfig.from_config(base)
        batch_size = 4

        decoder_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model([attention], config=base).estimate_flops,
        )
        mtp_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model(
                [attention, MTPSplit, attention, MTPSplit, attention], config=base
            ).estimate_flops,
        )

        total_tokens = batch_size * args.seq_length
        mtp_overhead = 3 * 2 * total_tokens * 2 * (3 * args.hidden_size + 2 * args.hidden_size**2)
        assert mtp_flops == 3 * decoder_flops + mtp_overhead

    def test_attention_output_gate_uses_concrete_layer_setting(self):
        args = _make_hybrid_args(num_layers=1)
        args.hybrid_layer_pattern = None
        args.is_hybrid_model = True
        base = self._base_config(num_layers=1)
        attention = AttentionLayerConfig.from_config(base)
        gated_attention = AttentionLayerConfig.from_config(attention)
        gated_attention.attention_output_gate = True
        batch_size = 4
        total_tokens = batch_size * args.seq_length

        ungated_flops = num_floating_point_operations(
            args, batch_size, model_flops_estimator=self._model([attention]).estimate_flops
        )
        gated_flops = num_floating_point_operations(
            args, batch_size, model_flops_estimator=self._model([gated_attention]).estimate_flops
        )

        query_projection_size = attention.kv_channels * attention.num_attention_heads
        expected_delta = 3 * 2 * total_tokens * attention.hidden_size * query_projection_size
        assert gated_flops - ungated_flops == expected_delta

    @pytest.mark.parametrize("args_spec_name", ["hybrid_stack_spec", "gdp_stack_spec"])
    def test_gdp_uses_model_stack_spec_instead_of_args_spec(self, args_spec_name):
        args = _make_hybrid_args(num_layers=1)
        args.hybrid_layer_pattern = None
        args.is_hybrid_model = True
        args.spec = ["megatron.core.models.hybrid.hybrid_layer_specs", args_spec_name]
        mamba = MambaLayerConfig.from_config(self._base_config(num_layers=1))

        mamba_flops = num_floating_point_operations(
            args, 4, model_flops_estimator=self._model([mamba]).estimate_flops
        )
        gdp_flops = num_floating_point_operations(
            args,
            4,
            model_flops_estimator=self._model([mamba], stack_spec=gdp_stack_spec).estimate_flops,
        )

        assert gdp_flops != mamba_flops

    @pytest.mark.parametrize("absorbed", [False, True])
    def test_dsa_topk_uses_sparse_core_dimensions(self, absorbed):
        args = _make_hybrid_args(num_layers=1)
        args.hybrid_layer_pattern = None
        args.is_hybrid_model = True
        batch_size = 4
        total_tokens = batch_size * args.seq_length
        small = self._dsa_config()
        large = DSALayerConfig.from_config(small)
        large.dsa_indexer_topk = 19
        stack_spec = deepcopy(hybrid_stack_spec)
        if not absorbed:
            stack_spec.submodules.dsa_layer.submodules.self_attention.module = MLASelfAttention

        small_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model([small], stack_spec=stack_spec).estimate_flops,
        )
        large_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model([large], stack_spec=stack_spec).estimate_flops,
        )

        core_dim = (
            2 * small.kv_lora_rank + small.qk_pos_emb_head_dim
            if absorbed
            else small.qk_head_dim + small.qk_pos_emb_head_dim + small.v_head_dim
        )
        expected_forward_delta = (
            2
            * total_tokens
            * (large.dsa_indexer_topk - small.dsa_indexer_topk)
            * small.num_attention_heads
            * core_dim
        )
        assert large_flops - small_flops == expected_forward_delta * 3

    def test_dsa_topk_frequency_uses_physical_layer_number_and_indexer_fields(self):
        args = _make_hybrid_args(num_layers=2)
        args.hybrid_layer_pattern = None
        args.is_hybrid_model = True
        batch_size = 4
        total_tokens = batch_size * args.seq_length
        first = self._dsa_config(num_layers=2)
        second = DSALayerConfig.from_config(first)
        shared_first = DSALayerConfig.from_config(first)
        shared_second = DSALayerConfig.from_config(second)
        shared_first.dsa_indexer_topk_freq = 2
        shared_second.dsa_indexer_topk_freq = 2

        every_layer_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model([first, PipelineSplit, second]).estimate_flops,
        )
        shared_flops = num_floating_point_operations(
            args,
            batch_size,
            model_flops_estimator=self._model(
                [shared_first, PipelineSplit, shared_second]
            ).estimate_flops,
        )

        indexer_projection = (
            2
            * total_tokens
            * (
                second.q_lora_rank * second.dsa_indexer_n_heads * second.dsa_indexer_head_dim
                + second.hidden_size * second.dsa_indexer_head_dim
                + second.hidden_size * second.dsa_indexer_n_heads
            )
        )
        indexer_scores = (
            batch_size
            * args.seq_length**2
            * second.dsa_indexer_n_heads
            * second.dsa_indexer_head_dim
        )
        assert every_layer_flops - shared_flops == (indexer_projection + indexer_scores) * 3

    def test_heterogeneous_flops_use_model_fields_and_vocab_not_args(self):
        args = _make_gpt_args(num_layers=99, hidden_size=128, padded_vocab_size=1024)
        base = self._base_config(num_layers=3)
        attention = AttentionLayerConfig.from_config(base)
        small_moe = MoELayerConfig.from_config(base)
        small_moe.num_moe_experts = 8
        small_moe.moe_ffn_hidden_size = 1024
        small_moe.moe_router_topk = 2
        large_moe = MoELayerConfig.from_config(base)
        large_moe.num_moe_experts = 16
        large_moe.moe_ffn_hidden_size = 3072
        large_moe.moe_router_topk = 4
        large_moe.moe_latent_size = 128
        large_moe.moe_shared_expert_intermediate_size = 512
        model = self._model(
            [
                attention,
                small_moe,
                PipelineSplit,
                large_moe,
                MTPSplit,
                attention,
                MTPSplit,
                attention,
            ],
            config=base,
            vocab_size=65536,
        )
        total_tokens = 400
        sum_squared = 20000
        # Three executed attention layers, two heterogeneous MoEs, and three logits projections.
        attention_flops = 8 * total_tokens * 512**2 + 2 * sum_squared * 512
        small_moe_flops = 6 * total_tokens * 512 * 1024 * 2
        large_moe_flops = (
            6 * total_tokens * 128 * 3072 * 4
            + 4 * total_tokens * 512 * 128
            + 6 * total_tokens * 512 * 512
        )
        logits_flops = 2 * total_tokens * 512 * 65536 * 3
        mtp_flops = 2 * total_tokens * 2 * (3 * 512 + 2 * 512**2)
        expected = 3 * (
            3 * attention_flops + small_moe_flops + large_moe_flops + logits_flops + mtp_flops
        )

        assert (
            model.estimate_flops(
                total_real_tokens_in_batch=total_tokens, seqlen_squared_sum_in_batch=sum_squared
            )
            == expected
        )
        assert (
            num_floating_point_operations(
                args,
                4,
                total_real_tokens_in_batch=total_tokens,
                seqlen_squared_sum_in_batch=sum_squared,
                model_flops_estimator=model.estimate_flops,
            )
            == expected
        )
        assert args.num_experts is None

    @pytest.mark.parametrize("q_lora_rank", [None, 48])
    def test_mla_uses_each_layers_projection_and_attention_dimensions(self, q_lora_rank):
        layers = []
        for heads, qk_dim, position_dim, kv_rank, value_dim in [
            (8, 48, 16, 96, 32),
            (4, 32, 8, 64, 24),
        ]:
            layer = MLALayerConfig.from_config(self._base_config())
            layer.multi_latent_attention = True
            layer.num_attention_heads = layer.num_query_groups = heads
            layer.q_lora_rank = q_lora_rank
            layer.qk_head_dim = qk_dim
            layer.qk_pos_emb_head_dim = position_dim
            layer.kv_lora_rank = kv_rank
            layer.v_head_dim = value_dim
            layers.append(layer)
        model = self._model(layers)
        # Sum the two layers' query projections, KV projections/norms, and output projections.
        query_projections = (
            512 * (8 * 64 + 4 * 40)
            if q_lora_rank is None
            else 48 * ((512 + 8 * 64 + 1) + (512 + 4 * 40 + 1))
        )
        other_projections = (
            96 * (512 + 8 * 80 + 1)
            + 512 * 16
            + 8 * 32 * 512
            + 64 * (512 + 4 * 56 + 1)
            + 512 * 8
            + 4 * 24 * 512
        )
        forward = (
            2 * 23 * (query_projections + other_projections)
            + 113 * (8 * 96 + 4 * 64)
            + 2 * 23 * 512 * 32000
        )
        assert (
            model.estimate_flops(total_real_tokens_in_batch=23, seqlen_squared_sum_in_batch=113)
            == 3 * forward
        )

    @pytest.mark.parametrize("variant", ["gdn", "gdn2"])
    def test_gdn_uses_each_layers_heads_dimensions_and_convolution(self, variant):
        layers = []
        for key_dim, key_heads, value_dim, value_heads, kernel in [
            (16, 3, 24, 6, 5),
            (32, 2, 16, 4, 3),
        ]:
            layer = GDNLayerConfig.from_config(self._base_config())
            layer.experimental_attention_variant = variant
            layer.linear_key_head_dim = key_dim
            layer.linear_num_key_heads = key_heads
            layer.linear_value_head_dim = value_dim
            layer.linear_num_value_heads = value_heads
            layer.linear_conv_kernel_dim = kernel
            layers.append(layer)
        model = self._model(layers)
        # Expanded key/value widths are (48, 144) and (64, 64); GDN2 changes input projections.
        input_width = (
            (4 * 48 + 3 * 144) + (4 * 64 + 3 * 64)
            if variant == "gdn2"
            else (2 * 48 + 2 * 144 + 2 * 6) + (2 * 64 + 2 * 64 + 2 * 4)
        )
        forward = (
            2
            * 23
            * (
                512 * input_width
                + 5 * (2 * 48 + 144)
                + 3 * (2 * 64 + 64)
                + 6 * 24**2 * 4
                + 4 * 16**2 * 4
                + 512 * (144 + 64)
                + 512 * 32000
            )
        )
        assert (
            model.estimate_flops(total_real_tokens_in_batch=23, seqlen_squared_sum_in_batch=113)
            == 3 * forward
        )

    def test_packed_stats_scale_layer_and_logits_work_independently(self):
        args = _make_gpt_args()
        attention = AttentionLayerConfig.from_config(self._base_config(num_layers=1))
        model = self._model([attention])
        full_tokens = 4 * args.seq_length
        sum_squared = 4 * args.seq_length**2
        full = num_floating_point_operations(args, 4, model_flops_estimator=model.estimate_flops)
        fewer_tokens = num_floating_point_operations(
            args,
            4,
            total_real_tokens_in_batch=full_tokens // 2,
            seqlen_squared_sum_in_batch=sum_squared,
            model_flops_estimator=model.estimate_flops,
        )
        shorter_chunks = num_floating_point_operations(
            args,
            4,
            total_real_tokens_in_batch=full_tokens,
            seqlen_squared_sum_in_batch=sum_squared // 4,
            model_flops_estimator=model.estimate_flops,
        )

        core_flops = 6 * sum_squared * attention.kv_channels * attention.num_attention_heads
        assert 2 * fewer_tokens - full == core_flops
        assert full - shorter_chunks == core_flops * 3 // 4

    def test_pattern_model_defers_to_legacy_estimator(self):
        args = _make_hybrid_args()
        model = self._model(None)

        assert (
            model.estimate_flops(
                total_real_tokens_in_batch=4096, seqlen_squared_sum_in_batch=4194304
            )
            is None
        )
        assert num_floating_point_operations(
            args, 4, model_flops_estimator=model.estimate_flops
        ) == num_floating_point_operations(args, 4)


class TestModelFlopsEstimatorDispatch:
    """Training passes batch statistics, but no architecture metadata, to the model."""

    @pytest.mark.parametrize("estimated_flops", [0, 1234.5])
    @pytest.mark.parametrize("token_stats", [None, (0, 0), (23, 113)])
    def test_callback_receives_resolved_stats(self, estimated_flops, token_stats):
        args = SimpleNamespace(seq_length=10)
        estimator = Mock(return_value=estimated_flops)
        kwargs = {}
        if token_stats is not None:
            kwargs = dict(
                total_real_tokens_in_batch=token_stats[0],
                seqlen_squared_sum_in_batch=token_stats[1],
            )

        assert (
            num_floating_point_operations(args, 3, model_flops_estimator=estimator, **kwargs)
            == estimated_flops
        )
        total_tokens, sum_squared = token_stats if token_stats is not None else (30, 300)
        estimator.assert_called_once_with(
            total_real_tokens_in_batch=total_tokens, seqlen_squared_sum_in_batch=sum_squared
        )

    @pytest.mark.parametrize("make_args", [_make_gpt_args, _make_hybrid_args])
    def test_none_result_preserves_legacy_estimator_and_packed_stats(self, make_args):
        args = make_args()
        estimator = Mock(return_value=None)
        stats = dict(total_real_tokens_in_batch=23, seqlen_squared_sum_in_batch=113)

        expected = num_floating_point_operations(args, 4, **stats)
        assert (
            num_floating_point_operations(args, 4, model_flops_estimator=estimator, **stats)
            == expected
        )
        estimator.assert_called_once_with(**stats)


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
