# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situlu
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel import (
    megakernel_fuses_shared_expert,
    prepare_megakernel_shared_expert_config,
)
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel
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
        "expert_tensor_parallel_size": 1,
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
        {"moe_shared_expert_gate": True, "activation_func": situlu},
        {"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True, "moe_shared_expert_gate": True},
        {
            "fp8": "hybrid",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_modules": [],
            "moe_layer_recompute": True,
        },
        # MXFP8 MoK under chunk-wise block capture: the scale re-swizzle is recorded
        # into the captured block, so no host-side refresh is needed between replays.
        {
            "fp8": "e4m3",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "chunk_wise",
            "cuda_graph_modules": [],
        },
    ],
)
def test_mok_accepts_key_supported_configurations(overrides):
    assert _mok_transformer_config(**overrides).moe_megakernel_backend == "mok"


@pytest.mark.parametrize("routing_map_mode", ["indices", "bool"])
def test_mok_keeps_the_configured_hybridep_routing_map_mode(routing_map_mode):
    """The router hands a megakernel backend the boolean token-to-expert map whatever the
    HybridEP routing-map option says, so a MOK config neither overrides it nor warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = _mok_transformer_config(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_hybridep_routing_map_mode=routing_map_mode,
        )
    assert config.moe_hybridep_routing_map_mode == routing_map_mode
    assert not [w for w in caught if "moe_hybridep_routing_map_mode" in str(w.message)]


def _detached_mok_backend(*, use_mxfp8_weights: bool, cuda_graph_impl: str = "chunk_wise"):
    """Build a MoKMegakernel with only the state quantized_routed_weights() reads.

    Bypasses __init__ (which needs the mok package, TE grouped parameters, and an EP
    group) so the view-refresh decision can be tested on any device.
    """
    backend = MoKMegakernel.__new__(MoKMegakernel)
    torch.nn.Module.__init__(backend)
    backend.use_mxfp8_weights = use_mxfp8_weights
    backend.native_single_grouped_weights = True
    backend.num_local_experts = 2
    backend.hidden_size = 128
    backend.intermediate_size = 256
    backend.register_parameter("routed_fc1_weight", None)
    backend.register_parameter("routed_fc2_weight", None)
    backend.is_first_microbatch = False
    backend._refresh_scales_in_every_capture = cuda_graph_impl == "chunk_wise"
    backend._routed_weight_view_cache = (object(), object())
    backend._split_main_grad_descriptor_cache = None
    return backend


@pytest.mark.parametrize(
    ("use_mxfp8_weights", "is_first_microbatch", "capturing", "cuda_graph_impl", "expect_rebuild"),
    [
        # Eager MXFP8: refresh exactly when the host flag says a new iteration began.
        (True, True, False, "chunk_wise", True),
        (True, False, False, "chunk_wise", False),
        # Captured chunk-wise MXFP8 block: the refresh is recorded into the graph regardless
        # of the flag, so replays re-swizzle TE's live scale_inv instead of keeping a stale
        # copy.
        (True, False, True, "chunk_wise", True),
        (True, True, True, "chunk_wise", True),
        # A full-iteration graph records the whole iteration and keeps the host schedule:
        # one refresh, at the first microbatch.
        (True, True, True, "full_iteration", True),
        (True, False, True, "full_iteration", False),
        # BF16 payloads alias TE storage directly; nothing to refresh in either mode.
        (False, True, False, "chunk_wise", False),
        (False, False, True, "chunk_wise", False),
    ],
)
def test_quantized_routed_weights_refreshes_scales_under_capture(
    monkeypatch, use_mxfp8_weights, is_first_microbatch, capturing, cuda_graph_impl, expect_rebuild
):
    backend = _detached_mok_backend(
        use_mxfp8_weights=use_mxfp8_weights, cuda_graph_impl=cuda_graph_impl
    )
    backend.is_first_microbatch = is_first_microbatch
    cached_fc1, cached_fc2 = backend._routed_weight_view_cache
    calls = []

    def fake_view(parameters, *, rows, columns, cached_view):
        # The refresh must write into the cached view (stable descriptor addresses).
        calls.append(cached_view)
        return cached_view

    monkeypatch.setattr(backend, "_routed_weight_view", fake_view)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    views = backend.quantized_routed_weights()

    assert views == (cached_fc1, cached_fc2)
    assert calls == ([cached_fc1, cached_fc2] if expect_rebuild else [])
    # Eager semantics are unchanged: the host flag is consumed by the call.
    assert backend.is_first_microbatch is False


@pytest.mark.parametrize("use_mxfp8_weights", [True, False])
@pytest.mark.parametrize("capturing", [False, True])
def test_quantized_routed_weights_first_call_builds_the_views(
    monkeypatch, use_mxfp8_weights, capturing
):
    """The first call builds the views wherever it runs: a full-iteration graph captured
    without an eager warm-up step records the build, like the native MoE's first forward."""
    backend = _detached_mok_backend(use_mxfp8_weights=use_mxfp8_weights)
    backend._routed_weight_view_cache = None
    built = []

    def fake_view(parameters, *, rows, columns, cached_view):
        assert cached_view is None
        built.append(object())
        return built[-1]

    monkeypatch.setattr(backend, "_routed_weight_view", fake_view)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    views = backend.quantized_routed_weights()

    assert len(built) == 2
    assert views == tuple(built)
    assert backend._routed_weight_view_cache == tuple(built)


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


def test_shared_execution_is_independent_of_activation_and_preserves_native_precision():
    for activation in (F.silu, situlu):
        for options in (
            None,
            {},
            {"fwd_num_comm_sms": 32},
            {"shared_expert_mode": "auto"},
            {"shared_expert_mode": "fused"},
        ):
            config = _mok_transformer_config(
                activation_func=activation, moe_megakernel_backend_config=options
            )
            assert megakernel_fuses_shared_expert(config)
        for overrides in (
            {"moe_megakernel_backend_config": {"shared_expert_mode": "native"}},
            {"moe_latent_size": 64},
            {"moe_shared_expert_intermediate_size": 512},
            {"moe_shared_expert_intermediate_size": None},
            {"tensor_model_parallel_size": 4, "sequence_parallel": True},
        ):
            config = _mok_transformer_config(
                activation_func=activation,
                fp8="hybrid",
                fp8_recipe="mxfp8",
                fp8_param=True,
                **overrides,
            )
            assert not megakernel_fuses_shared_expert(config)
            assert prepare_megakernel_shared_expert_config(config) is config
            assert config.fp8_param and config.fp8 == "hybrid"
    for overrides in (
        {"moe_latent_size": 64},
        {"moe_shared_expert_intermediate_size": 512},
        {"moe_shared_expert_intermediate_size": None},
        {"tensor_model_parallel_size": 4, "sequence_parallel": True},
    ):
        with pytest.raises(ValueError, match="Fused MOK shared execution requires"):
            _mok_transformer_config(
                moe_megakernel_backend_config={"shared_expert_mode": "fused"}, **overrides
            )
    for mode in ("invalid", None):
        with pytest.raises(ValueError, match="must be auto, fused, or native"):
            _mok_transformer_config(moe_megakernel_backend_config={"shared_expert_mode": mode})
    assert not megakernel_fuses_shared_expert(_mok_transformer_config(moe_megakernel_backend=None))
    for options in ({"shared_expert_mode": "auto"}, {"fwd_num_comm_sms": 32}):
        with pytest.raises(ValueError, match="requires moe_megakernel_backend to be set"):
            _mok_transformer_config(
                moe_megakernel_backend=None, moe_megakernel_backend_config=options
            )
    with pytest.raises(ValueError, match="requires sequence_parallel=True"):
        _mok_transformer_config(tensor_model_parallel_size=4)
    with pytest.raises(ValueError, match="requires expert TP=1"):
        _mok_transformer_config(expert_tensor_parallel_size=2)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"moe_shared_expert_intermediate_size": None, "add_bias_linear": True},
            "MOK requires add_bias_linear=False",
        ),
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
        [CudaGraphModule.moe],
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_preprocess],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ],
)
def test_mok_rejects_per_layer_cuda_graph_covering_moe(cuda_graph_impl, cuda_graph_modules):
    with pytest.raises(ValueError, match="moe/moe_router/moe_preprocess"):
        _mok_transformer_config(
            cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_accepts_per_layer_cuda_graph_outside_moe(cuda_graph_impl):
    cuda_graph_modules = [CudaGraphModule.attn]
    config = _mok_transformer_config(
        cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
    )

    assert config.cuda_graph_modules == cuda_graph_modules
