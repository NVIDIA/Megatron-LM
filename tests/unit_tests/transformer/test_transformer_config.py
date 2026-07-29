# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def _make_moonep_config(**overrides) -> TransformerConfig:
    kwargs = {
        "num_layers": 1,
        "hidden_size": 256,
        "ffn_hidden_size": 512,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "moe_router_topk": 2,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "moonep",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": True,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "gated_linear_unit": True,
        "activation_func": F.silu,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.fixture
def moonep_config(monkeypatch):
    """Avoid making config validation depend on the TE version in the unit-test environment."""
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda _: True
    )
    return _make_moonep_config


@pytest.mark.parametrize(
    "activation_overrides",
    [
        {"activation_func": F.silu, "gated_linear_unit": True},
        {"activation_func": quick_gelu, "gated_linear_unit": True},
        {
            "activation_func": squared_relu,
            "gated_linear_unit": False,
            "use_fused_weighted_squared_relu": True,
        },
    ],
)
def test_moonep_accepts_supported_activations(moonep_config, activation_overrides):
    config = moonep_config(**activation_overrides)

    assert config.moe_flex_dispatcher_backend == "moonep"


def test_moonep_accepts_latent_moe(moonep_config):
    config = moonep_config(moe_latent_size=128)

    assert config.moe_latent_size == 128


def test_moonep_rejects_unaligned_latent_size(moonep_config):
    with pytest.raises(ValueError, match="moe_latent_size.*divisible by 128"):
        moonep_config(moe_latent_size=64)


@pytest.mark.parametrize(
    ("override", "requirement"),
    [
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        ({"bf16": False, "params_dtype": torch.float32}, "BF16 execution"),
        ({"add_bias_linear": True}, "add_bias_linear=False"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"moe_single_grouped_weight": False}, "moe_single_grouped_weight=True"),
        ({"use_transformer_engine_op_fuser": False}, "use_transformer_engine_op_fuser=True"),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"moe_router_dtype": None}, "moe_router_dtype='fp32'"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size=1"),
        ({"moe_router_topk": 33}, "moe_router_topk<=32"),
    ],
)
def test_moonep_rejects_missing_required_flags(moonep_config, override, requirement):
    with pytest.raises(ValueError, match=requirement):
        moonep_config(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"fp8": "e4m3", "fp8_recipe": "mxfp8"},
        {"cuda_graph_impl": "local"},
        {"delay_wgrad_compute": True},
        {"overlap_dispatch_backward_with_experts_wgrad": True},
        {"overlap_moe_expert_parallel_comm": True},
        {"moe_shared_expert_overlap": True},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_pad_expert_input_to_capacity": True, "moe_expert_capacity_factor": 1.0},
        {"moe_router_padding_for_quantization": True},
        {"moe_apply_probs_on_input": True},
    ],
)
def test_moonep_rejects_unsupported_features(moonep_config, override):
    with pytest.raises(ValueError, match="MoonEP flex dispatcher configuration is unsupported"):
        moonep_config(**override)


def test_moonep_rejects_unsupported_activation(moonep_config):
    with pytest.raises(ValueError, match="weighted squared-ReLU activation"):
        moonep_config(activation_func=F.gelu, gated_linear_unit=True)


def test_moonep_manager_reports_missing_optional_package(moonep_config, monkeypatch):
    config = moonep_config()
    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.is_moonep_available", lambda: False
    )

    with pytest.raises(ImportError, match="MoonEP is not installed"):
        _MoonEPManager(
            group=None,
            num_local_experts=1,
            router_topk=config.moe_router_topk,
            num_experts=config.num_moe_experts,
            config=config,
        )
