# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import fp8_utils
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel import factory
from megatron.core.transformer.moe.megakernel.mok import backend as mok_backend
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig


def _mok_transformer_config(**overrides):
    values = {
        "num_layers": 2,
        "bf16": True,
        "moe_grouped_gemm": True,
        "hidden_size": 128,
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


def test_bf16_shared_expert_config_reuses_original_config():
    config = SimpleNamespace(fp8_param=False)

    assert weights.prepare_shared_expert_config(config) is config


def test_mxfp8_shared_expert_config_expresses_bf16_module(monkeypatch, recwarn):
    original_recipe = object()
    config = SimpleNamespace(
        fp8="hybrid", fp8_param=True, quant_recipe=original_recipe, sentinel=object()
    )
    monkeypatch.setattr(weights, "_SHARED_EXPERT_BF16_WARNING_EMITTED", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    shared_config = weights.prepare_shared_expert_config(config)

    assert shared_config is not config
    assert config.fp8 == "hybrid"
    assert config.fp8_param
    assert config.quant_recipe is original_recipe
    assert shared_config.fp8 is None
    assert not shared_config.fp8_param
    assert shared_config.quant_recipe is original_recipe
    assert shared_config.sentinel is config.sentinel
    assert len(recwarn) == 1
    assert "shared experts in BF16" in str(recwarn[0].message)

    weights.prepare_shared_expert_config(config)
    assert len(recwarn) == 1


def test_mok_shared_expert_init_context_disables_outer_fp8(monkeypatch):
    config = SimpleNamespace(moe_megakernel_backend="mok")
    sentinel = object()
    calls = []

    def fake_disabled_context(candidate, *, is_init):
        calls.append((candidate, is_init))
        return sentinel

    monkeypatch.setattr(fp8_utils, "get_fp8_disabled_context", fake_disabled_context)

    actual = factory.megakernel_shared_expert_init_context(config)

    assert actual is sentinel
    assert calls == [(config, True)]


def test_mok_backend_accepts_bf16_routed_parameters():
    config = _mok_transformer_config()

    assert config.moe_megakernel_backend == "mok"
    assert config.fp8 is None
    assert not config.fp8_param


@pytest.mark.parametrize(
    "overrides", [{"bf16": False}, {"bf16": False, "fp16": True}, {"fp4": "e2m1"}]
)
def test_mok_backend_rejects_unsupported_routed_precision(overrides):
    with pytest.raises(ValueError, match="FP32, FP16, and FP4 are not supported"):
        _mok_transformer_config(**overrides)


def test_mok_backend_requires_grouped_mlp():
    with pytest.raises(ValueError, match="moe_grouped_gemm=True"):
        _mok_transformer_config(moe_grouped_gemm=False)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not support overlap_moe_expert_parallel_comm",
        ),
        ({"delay_wgrad_compute": True}, "does not support delay_wgrad_compute"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["shared_experts"]},
            "does not support recompute_modules",
        ),
        ({"log_moe_overload_factor": True}, "does not support log_moe_overload_factor"),
    ],
)
def test_mok_backend_rejects_bypassed_native_moe_features(overrides, error):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**overrides)


@pytest.mark.parametrize("single_grouped", [False, True])
def test_mok_backend_rejects_non_fused_gradient_accumulation(single_grouped):
    with pytest.raises(ValueError, match="gradient_accumulation_fusion=True"):
        _mok_transformer_config(
            moe_single_grouped_weight=single_grouped, gradient_accumulation_fusion=False
        )


def test_mok_backend_accepts_native_mxfp8_parameters():
    config = _mok_transformer_config(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True)

    assert config.fp8_recipe == "mxfp8"
    assert config.fp8_param


def test_mok_backend_rejects_mxfp8_without_fp8_parameters():
    with pytest.raises(ValueError, match="fp8_param=True"):
        _mok_transformer_config(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=False)


def test_megakernel_backend_config_requires_backend():
    with pytest.raises(ValueError, match="requires moe_megakernel_backend"):
        _mok_transformer_config(
            moe_megakernel_backend=None,
            moe_megakernel_backend_config={"mok_fwd_num_comm_sms": 24},
        )


def test_mok_backend_config_defaults_and_prefixed_overrides():
    defaults = mok_backend.MoKBackendConfig.from_backend_config(None)
    configured = mok_backend.MoKBackendConfig.from_backend_config(
        {"mok_fwd_num_comm_sms": 24, "mok_macrobatch_size": 32768}
    )

    assert defaults.fwd_num_comm_sms == 40
    assert defaults.macrobatch_size == 131072
    assert configured.fwd_num_comm_sms == 24
    assert configured.macrobatch_size == 32768
    assert configured.bwd_num_comm_sms == defaults.bwd_num_comm_sms


@pytest.mark.parametrize("key", ["fwd_num_comm_sms", "other_fwd_num_comm_sms"])
def test_mok_backend_config_rejects_non_mok_options(key):
    with pytest.raises(ValueError, match="Unsupported MOK entries"):
        mok_backend.MoKBackendConfig.from_backend_config({key: 24})


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ],
)
def test_mok_backend_rejects_partial_moe_cuda_graph(cuda_graph_impl, cuda_graph_modules):
    with pytest.raises(ValueError, match="partial MoE graph replay protocol"):
        _mok_transformer_config(
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=cuda_graph_modules,
        )


def test_mok_backend_accepts_full_iteration_cuda_graph():
    config = _mok_transformer_config(
        cuda_graph_impl="full_iteration",
        cuda_graph_modules=[],
    )

    assert config.cuda_graph_impl == "full_iteration"


def test_mok_mxfp8_rejects_te_without_post_all_gather_processing(monkeypatch):
    monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", None)

    with pytest.raises(RuntimeError, match="normally TE >= 2.10.0"):
        mok_backend._require_mxfp8_post_all_gather_processing()


@pytest.mark.parametrize(
    "overrides",
    [
        {"recompute_granularity": "selective", "recompute_modules": ["moe"]},
        {"moe_layer_recompute": True},
    ],
)
def test_mok_backend_accepts_whole_moe_recompute(overrides):
    config = _mok_transformer_config(**overrides)

    assert config.recompute_granularity == "selective"
    assert "moe" in config.recompute_modules
