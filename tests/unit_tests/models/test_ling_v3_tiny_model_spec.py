# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from examples.ling_v3_tiny.model_spec import LING_V3_TINY_HYBRID_LAYER_PATTERN
from examples.ling_v3_tiny.model_spec import hybrid_stack_spec as ling_v3_tiny_hybrid_stack_spec
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_layer_counts,
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from tests.unit_tests.test_utilities import Utils


def _kda_submodules(spec):
    return spec.submodules.kda_layer.submodules.self_attention.submodules


def _mla_submodules(spec):
    return spec.submodules.mla_layer.submodules.self_attention.submodules


def test_ling_v3_tiny_hybrid_pattern_matches_model_topology():
    parsed = parse_hybrid_pattern(LING_V3_TINY_HYBRID_LAYER_PATTERN)
    counts = get_hybrid_layer_counts(LING_V3_TINY_HYBRID_LAYER_PATTERN)

    assert get_hybrid_total_layer_count(LING_V3_TINY_HYBRID_LAYER_PATTERN) == 48
    assert parsed.mtp_pattern == Symbols.MLA + Symbols.MOE
    assert parsed.mtp_num_depths == 1

    main_counts = {symbol: parsed.main_pattern.count(symbol) for symbol in Symbols.VALID_LAYERS}
    expected_nonzero_counts = {Symbols.KDA: 18, Symbols.MLA: 6, Symbols.MLP: 1, Symbols.MOE: 23}
    assert {
        symbol: count for symbol, count in main_counts.items() if count
    } == expected_nonzero_counts
    assert counts[Symbols.MLA] == 7
    assert counts[Symbols.MOE] == 24


def test_ling_v3_tiny_spec_selects_required_projection_backends():
    default_kda = _kda_submodules(hybrid_stack_spec)
    default_mla = _mla_submodules(hybrid_stack_spec)
    tiny_kda = _kda_submodules(ling_v3_tiny_hybrid_stack_spec)
    tiny_mla = _mla_submodules(ling_v3_tiny_hybrid_stack_spec)

    assert tiny_kda.beta_proj is ColumnParallelLinear
    assert tiny_mla.linear_q_down_proj is TEColumnParallelLinear
    assert tiny_mla.linear_kv_down_proj is TEColumnParallelLinear
    assert tiny_mla.linear_gate is ColumnParallelLinear

    assert default_kda.beta_proj is TEColumnParallelLinear
    assert default_mla.linear_q_down_proj is TELinear
    assert default_mla.linear_kv_down_proj is TELinear
    assert default_mla.linear_gate is None


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
@pytest.mark.parametrize("tp_size", [1, 2])
def test_ling_v3_tiny_reduced_hybrid_model_forward_backward(tp_size):
    """Exercise every Tiny layer type and the nested MTP recipe."""

    Utils.initialize_distributed()
    if torch.distributed.get_world_size() % tp_size != 0:
        Utils.destroy_model_parallel()
        pytest.skip("world size must be divisible by tensor parallel size")

    Utils.initialize_model_parallel(tp_size, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = MLATransformerConfig(
            num_layers=6,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=4,
            kv_channels=32,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            rope_type="rope",
            rotary_base=10000,
            rotary_percent=0.5,
            qk_layernorm=True,
            attention_output_gate=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            kda_safe_gate=True,
            kda_lower_bound=-5.0,
            num_moe_experts=2,
            moe_ffn_hidden_size=64,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_dtype="fp32",
            moe_router_topk_scaling_factor=1.0,
            moe_shared_expert_intermediate_size=64,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.1,
            normalization="RMSNorm",
            layernorm_epsilon=1.0e-6,
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            is_hybrid_model=True,
            calculate_per_token_loss=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=ling_v3_tiny_hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern="K-KE+E/+E",
            position_embedding_type="rope",
            rotary_percent=0.5,
            rotary_base=10000,
        ).cuda()

        main_layers = model.decoder.layers
        assert isinstance(main_layers[0].self_attention, KimiDeltaAttention)
        assert isinstance(main_layers[1], MLPLayer)
        assert isinstance(main_layers[2].self_attention, KimiDeltaAttention)
        assert isinstance(main_layers[3], MoETransformerLayer)
        assert isinstance(main_layers[4].self_attention, MLASelfAttention)
        assert isinstance(main_layers[5], MoETransformerLayer)

        assert isinstance(model.mtp.layers[0].mtp_model_layer, HybridStack)
        mtp_layers = model.mtp.layers[0].mtp_model_layer.layers
        assert isinstance(mtp_layers[0].self_attention, MLASelfAttention)
        assert isinstance(mtp_layers[1], MoETransformerLayer)

        sharded_state_dict = model.sharded_state_dict()
        required_sharded_keys = [
            "decoder.layers.0.self_attention.beta_proj.weight",
            "decoder.layers.4.self_attention.linear_q_down_proj.weight",
            "decoder.layers.4.self_attention.linear_kv_down_proj.weight",
            "decoder.layers.4.self_attention.linear_gate.weight",
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_gate.weight",
        ]
        for key in required_sharded_keys:
            assert key in sharded_state_dict, key

        input_ids = torch.arange(16, device="cuda", dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(16, device="cuda", dtype=torch.long).unsqueeze(0)
        labels = (input_ids + 1) % 128
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )

        assert loss.shape == input_ids.shape
        assert torch.isfinite(loss).all()
        assert "loss_sums" in MTPLossLoggingHelper.tracker
        assert torch.isfinite(MTPLossLoggingHelper.tracker["loss_sums"]).all()
        assert "num_tokens" in MTPLossLoggingHelper.tracker
        assert (MTPLossLoggingHelper.tracker["num_tokens"] > 0).all()

        loss.float().mean().backward()
        required_gradients = [
            "decoder.layers.0.self_attention.beta_proj.weight",
            "decoder.layers.4.self_attention.linear_gate.weight",
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_gate.weight",
        ]
        parameters = dict(model.named_parameters())
        for name in required_gradients:
            assert parameters[name].grad is not None, name
            assert torch.isfinite(parameters[name].grad).all(), name
    finally:
        MTPLossLoggingHelper.tracker = {}
        Utils.destroy_model_parallel()
