# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer import transformer_config as transformer_config_module
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version


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
        "add_bias_linear": False,
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
        {"moe_token_dispatcher_type": "alltoall"},
        {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "hybridep"},
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "hybridep",
            "sequence_packing_scheduler": "dp_balanced",
            "max_seqlen_per_dp_cp_rank": 1024,
            "calculate_per_token_loss": True,
            "variable_seq_lengths": True,
        },
        {"moe_shared_expert_gate": True},
        {"moe_router_dtype": None},
        {"moe_router_dtype": "fp32"},
        {"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True, "moe_shared_expert_gate": True},
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


def test_native_moe_still_accepts_fp64_router():
    config = _mok_transformer_config(moe_megakernel_backend=None, moe_router_dtype="fp64")
    assert config.moe_router_dtype == "fp64"


@pytest.mark.skipif(not is_te_min_version("2.14.0"), reason="single grouped weights need TE>=2.14")
def test_native_single_grouped_weights_do_not_require_op_fuser():
    config = _mok_transformer_config(
        moe_megakernel_backend=None,
        moe_single_grouped_weight=True,
        moe_use_grouped_tensor=True,
        use_transformer_engine_op_fuser=False,
    )
    assert config.moe_single_grouped_weight
    assert not config.use_transformer_engine_op_fuser


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"bf16": False, "fp16": True}, "FP32, FP16, and FP4 are not supported"),
        ({"fp4": "e2m1"}, "FP32, FP16, and FP4 are not supported"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"moe_router_dtype": "fp64"}, "MOK does not support moe_router_dtype='fp64'"),
        ({"moe_virtual_expert_load_balance": True}, "MOK does not support MoonEP"),
        ({"use_transformer_engine_op_fuser": True}, "use_transformer_engine_op_fuser=False"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not support overlap_moe_expert_parallel_comm",
        ),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": False}, "fp8_param=True"),
        ({"moe_shortcut_connection": True}, "does not support moe_shortcut_connection"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"moe_apply_probs_on_input": True}, "moe_apply_probs_on_input"),
        ({"glu_linear_offset": 1.0}, "glu_linear_offset"),
        ({"activation_func_tanh_clamp_scale": 1.0}, "activation_func_tanh_clamp_scale"),
        ({"gtp_weight_remat_size": 2}, "GTP"),
        ({"expert_gtp_weight_remat_size": 2}, "GTP"),
        (
            {
                "transformer_impl": "inference_optimized",
                "moe_router_dtype": "fp32",
                "inference_grouped_gemm_backend": "torch",
            },
            "does not support inference_optimized",
        ),
    ],
)
def test_mok_rejects_key_incompatible_configurations(overrides, error):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**overrides)


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_rejects_per_layer_whole_layer_cuda_graph(cuda_graph_impl):
    with pytest.raises(ValueError, match="whole-layer CUDA Graph capture"):
        _mok_transformer_config(cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=[])


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [CudaGraphModule.moe_preprocess],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ],
)
def test_mok_rejects_per_layer_cuda_graph_covering_moe_preprocess(
    cuda_graph_impl, cuda_graph_modules
):
    with pytest.raises(ValueError, match="moe_preprocess"):
        _mok_transformer_config(
            cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
        )


@pytest.mark.skipif(
    not is_te_min_version("2.19.0"),
    reason="The megakernel moe CUDA graph scope requires TE 2.19 and later.",
)
@pytest.mark.parametrize(
    "cuda_graph_modules", [[CudaGraphModule.moe], [CudaGraphModule.attn, CudaGraphModule.moe]]
)
def test_mok_accepts_transformer_engine_moe_cuda_graph(cuda_graph_modules):
    config = _mok_transformer_config(
        cuda_graph_impl="transformer_engine", cuda_graph_modules=cuda_graph_modules
    )

    assert config.cuda_graph_modules == cuda_graph_modules


def test_mok_rejects_local_moe_cuda_graph():
    with pytest.raises(ValueError, match="only with cuda_graph_impl='transformer_engine'"):
        _mok_transformer_config(cuda_graph_impl="local", cuda_graph_modules=[CudaGraphModule.moe])


def test_mok_rejects_moe_cuda_graph_with_moe_recompute():
    with pytest.raises(ValueError, match="'moe' in recompute_modules"):
        _mok_transformer_config(
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe],
            recompute_granularity="selective",
            recompute_modules=["moe"],
        )


def test_mok_rejects_moe_cuda_graph_with_pipeline_parallelism():
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        _mok_transformer_config(
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe],
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
        )


def test_mok_accepts_moe_router_cuda_graph_with_pipeline_parallelism():
    # MOK runs eagerly after the router graph, so each call keeps its own saved state.
    config = _mok_transformer_config(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
    )

    assert config.pipeline_model_parallel_size == 2


def test_mok_moe_cuda_graph_requires_te_that_does_not_clone_param_grads(monkeypatch):
    original = transformer_config_module.is_te_min_version

    def is_te_min_version(version, *args, **kwargs):
        return version != "2.19.0" and original(version, *args, **kwargs)

    monkeypatch.setattr(transformer_config_module, "is_te_min_version", is_te_min_version)
    with pytest.raises(ValueError, match="requires Transformer Engine >= 2.19"):
        _mok_transformer_config(
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe],
        )


def test_native_moe_cuda_graph_still_requires_drop_padding():
    with pytest.raises(AssertionError, match="drop-padding MoE"):
        _mok_transformer_config(
            moe_megakernel_backend=None,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe],
        )


def test_mok_accepts_transformer_engine_moe_router_cuda_graph():
    cuda_graph_modules = [CudaGraphModule.attn, CudaGraphModule.moe_router]
    config = _mok_transformer_config(
        cuda_graph_impl="transformer_engine", cuda_graph_modules=cuda_graph_modules
    )

    assert config.cuda_graph_modules == cuda_graph_modules


def test_mok_rejects_local_moe_router_cuda_graph():
    with pytest.raises(ValueError, match="only with cuda_graph_impl='transformer_engine'"):
        _mok_transformer_config(
            cuda_graph_impl="local", cuda_graph_modules=[CudaGraphModule.moe_router]
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_accepts_per_layer_cuda_graph_outside_moe(cuda_graph_impl):
    cuda_graph_modules = [CudaGraphModule.attn]
    config = _mok_transformer_config(
        cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
    )

    assert config.cuda_graph_modules == cuda_graph_modules
