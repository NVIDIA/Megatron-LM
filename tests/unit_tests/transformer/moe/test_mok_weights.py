# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import fp8_utils
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel.mok import backend as mok_backend
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig


def _mok_transformer_config(**overrides):
    values = {
        "num_layers": 2,
        "bf16": True,
        "moe_grouped_gemm": True,
        "hidden_size": 128,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "expert_model_parallel_size": 4,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "gradient_accumulation_fusion": True,
        "moe_megakernel_backend": "mok",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_mxfp8_shared_expert_config_expresses_bf16_module(monkeypatch, recwarn):
    original_recipe = object()
    config = SimpleNamespace(
        fp8="hybrid", fp8_param=True, quant_recipe=original_recipe, sentinel=object()
    )
    monkeypatch.setattr(weights, "_SHARED_EXPERT_BF16_WARNING_EMITTED", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    shared_config = weights.prepare_shared_expert_config(config)

    assert shared_config is not config
    assert config.fp8 == "hybrid" and config.fp8_param
    assert shared_config.fp8 is None and not shared_config.fp8_param
    assert shared_config.quant_recipe is original_recipe
    assert shared_config.sentinel is config.sentinel
    assert len(recwarn) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "fp8": "hybrid",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_modules": [],
            "moe_layer_recompute": True,
        },
    ],
)
def test_mok_accepts_key_supported_configurations(overrides):
    assert _mok_transformer_config(**overrides).moe_megakernel_backend == "mok"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"bf16": False, "fp16": True}, "FP32, FP16, and FP4 are not supported"),
        ({"fp4": "e2m1"}, "FP32, FP16, and FP4 are not supported"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not support overlap_moe_expert_parallel_comm",
        ),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        (
            {"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": False},
            "fp8_param=True",
        ),
        (
            {
                "cuda_graph_impl": "local",
                "cuda_graph_modules": [CudaGraphModule.moe_router],
            },
            "partial MoE graph replay protocol",
        ),
    ],
)
def test_mok_rejects_key_incompatible_configurations(overrides, error):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**overrides)


def test_mok_mxfp8_requires_post_all_gather_processing(monkeypatch):
    monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", None)

    with pytest.raises(RuntimeError, match="normally TE >= 2.10.0"):
        mok_backend._require_mxfp8_post_all_gather_processing()
