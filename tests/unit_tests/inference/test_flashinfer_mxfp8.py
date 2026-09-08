# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.inference.moe.flashinfer_mxfp8 import select_flashinfer_active_rows
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def reset_inference_mode():
    InferenceMode.unset_active()
    yield
    InferenceMode.unset_active()


def test_inference_mode_tracks_flashinfer_token_capacity():
    InferenceMode.set_active()
    assert InferenceMode.flashinfer_token_capacity() is None

    InferenceMode.set_flashinfer_token_capacity(512)
    assert InferenceMode.flashinfer_token_capacity() == 512

    InferenceMode.unset_active()
    assert InferenceMode.flashinfer_token_capacity() is None

    InferenceMode.set_active()
    assert InferenceMode.flashinfer_token_capacity() is None


def test_inference_mode_rejects_nonpositive_flashinfer_token_capacity():
    with pytest.raises(ValueError, match="must be positive"):
        InferenceMode.set_flashinfer_token_capacity(0)


def test_nvls_flashinfer_metadata_initializes_padding_routes_to_minus_one(monkeypatch):
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe import token_dispatcher_inference as dispatcher_module

    routing = torch.zeros(8, 2, dtype=torch.int64)
    monkeypatch.setattr(
        dispatcher_module.NVLSAllGatherVDispatcher, "_symm_agv_routing", {"tensor": routing}
    )
    monkeypatch.setattr(
        dispatcher_module.NVLSAllGatherVDispatcher,
        "_symm_metadata",
        {"tensor": torch.empty(2, dtype=torch.int32), "handle": object()},
    )
    monkeypatch.setattr(
        dispatcher_module.NVLSAllGatherVDispatcher,
        "_step_metadata",
        torch.zeros(3, dtype=torch.int32),
    )
    monkeypatch.setattr(dispatcher_module, "fused_metadata_update", lambda **kwargs: None)
    dispatcher = SimpleNamespace(
        config=SimpleNamespace(
            inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER
        ),
        ep_size=2,
    )

    dispatcher_module.NVLSAllGatherVDispatcher.update_metadata(dispatcher, local_tokens=4)

    assert torch.equal(routing, torch.full_like(routing, -1))


@pytest.mark.parametrize("bounded_rows_allowed", [False, True])
def test_controller_infers_flashinfer_capacity_only_when_policy_allows(bounded_rows_allowed):
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )

    context = SimpleNamespace(
        max_requests=128,
        expert_model_parallel_size=4,
        can_use_bounded_flashinfer_rows=lambda: bounded_rows_allowed,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        num_last_token_logits=1,
        has_vlm_data=False,
    )
    logits = torch.empty(1, 1, 4)
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context, run_one_forward_step=lambda inference_input: logits
    )
    controller.model_config = SimpleNamespace()
    controller.num_speculative_tokens = 0
    controller.model_is_pipeline_parallel = False
    controller._enable_cuda_graph = False

    controller._dynamic_step_forward_logits(
        torch.zeros(1, 1, dtype=torch.int64), torch.zeros(1, 1, dtype=torch.int64)
    )

    assert InferenceMode.flashinfer_token_capacity() == (512 if bounded_rows_allowed else None)


@pytest.mark.parametrize(
    ("role", "prefill_req_count", "regular_ep_result", "expected", "expected_sync_calls"),
    [
        ("decode", 0, False, True, 0),
        ("prefill", 0, True, False, 0),
        (None, 0, True, True, 1),
        (None, 1, False, False, 1),
    ],
    ids=[
        "disaggregated-decode",
        "disaggregated-prefill",
        "regular-all-decode",
        "regular-any-prefill",
    ],
)
def test_flashinfer_row_policy_covers_disaggregated_and_regular_modes(
    role, prefill_req_count, regular_ep_result, expected, expected_sync_calls
):
    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    sync_calls = []

    def sync_all_ep_ranks_decode_only(batch_dimensions):
        sync_calls.append(batch_dimensions)
        return regular_ep_result

    context = SimpleNamespace(
        inference_flashinfer_bounded_rows=True,
        is_creating_cuda_graphs=False,
        _disaggregated_inference_role=role,
        _sync_all_ep_ranks_decode_only=sync_all_ep_ranks_decode_only,
    )
    batch_dimensions = InferenceBatchDimensions(
        token_count=128,
        prefill_req_count=prefill_req_count,
        decode_req_count=128 if prefill_req_count == 0 else 0,
    )

    result = DynamicInferenceContext._resolve_bounded_flashinfer_rows(context, batch_dimensions)

    assert result is expected
    assert len(sync_calls) == expected_sync_calls


def test_flashinfer_row_policy_rejects_prefill_on_disaggregated_decode():
    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    context = SimpleNamespace(
        inference_flashinfer_bounded_rows=True,
        is_creating_cuda_graphs=False,
        _disaggregated_inference_role="decode",
    )
    batch_dimensions = InferenceBatchDimensions(
        token_count=128, prefill_req_count=1, decode_req_count=0
    )

    with pytest.raises(RuntimeError, match="decode engine cannot use a prefill-bearing batch"):
        DynamicInferenceContext._resolve_bounded_flashinfer_rows(context, batch_dimensions)


@pytest.mark.parametrize(
    ("token_capacity", "expected"),
    [
        (None, (65536, "full")),
        (1024, (1024, "bounded-decode")),
        (131072, (65536, "bounded-decode")),
    ],
)
def test_flashinfer_active_row_policy(token_capacity, expected):
    assert select_flashinfer_active_rows(65536, token_capacity=token_capacity) == expected


def _make_bounded_mxfp8_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_bounded_rows=True,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_mxfp8_config_accepts_nvls_ep():
    config = _make_bounded_mxfp8_config()

    assert config.inference_flashinfer_bounded_rows
    assert config.inference_moe_token_dispatcher_type == "nvls"
    assert config.expert_model_parallel_size == 2


def test_bf16_config_ignores_inactive_mxfp8_recipe_gates():
    config = _make_bounded_mxfp8_config(
        fp8=None, fp8_param=False, activation_func=F.gelu, inference_flashinfer_bounded_rows=False
    )

    assert config.fp8 is None


def _make_bounded_bf16_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_bounded_rows=True,
        fp8=None,
        fp8_param=False,
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_bf16_config_accepts_nvls_ep():
    config = _make_bounded_bf16_config()

    assert config.inference_flashinfer_bounded_rows


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inference_grouped_gemm_backend": "vllm"}, "requires.*backend='flashinfer'"),
        (
            {"fp8": "hybrid", "fp8_recipe": "delayed", "fp8_param": True},
            "requires.*BF16 parameters.*mxfp8",
        ),
        ({"params_dtype": torch.float32}, "requires.*BF16 parameters"),
        ({"inference_moe_token_dispatcher_type": "nccl"}, "requires.*nvls"),
        ({"expert_model_parallel_size": 1}, "requires.*expert_model_parallel_size > 1"),
    ],
)
def test_bounded_flashinfer_bf16_config_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_bounded_bf16_config(**overrides)


@pytest.mark.parametrize("activation_func", [F.gelu, F.silu, F.relu])
def test_flashinfer_mxfp8_config_rejects_unsupported_activation(activation_func):
    with pytest.raises(ValueError, match="supports only non-gated squared-ReLU experts"):
        _make_bounded_mxfp8_config(activation_func=activation_func)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"inference_grouped_gemm_backend": "vllm", "inference_flashinfer_bounded_rows": False},
            "vLLM Triton fused MoE only supports BF16",
        ),
        ({"fp8_param": False}, "fp8_param must be enabled"),
        ({"fp8": None, "fp8_param": False}, "requires.*BF16 parameters"),
        ({"inference_moe_token_dispatcher_type": "nccl"}, "requires.*nvls"),
        ({"expert_model_parallel_size": 1}, "requires.*expert_model_parallel_size > 1"),
    ],
)
def test_bounded_flashinfer_mxfp8_config_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_bounded_mxfp8_config(**overrides)


def test_missing_routed_mxfp8_capability_has_precise_error(monkeypatch):
    from megatron.core.inference.moe import flashinfer_mxfp8

    monkeypatch.setattr(flashinfer_mxfp8, "HAVE_FLASHINFER_ROUTED_MXFP8", False)
    monkeypatch.setattr(
        flashinfer_mxfp8,
        "_FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR",
        ImportError("missing routed MXFP8 API"),
    )

    with pytest.raises(RuntimeError, match="requires FlashInfer >= 0.6.4"):
        flashinfer_mxfp8.require_flashinfer_routed_mxfp8()


def test_flashinfer_mxfp8_refresh_reports_noop_before_weight_build():
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    grouped_mlp = SimpleNamespace(
        _concatenated_weights_built=False,
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER,
    )

    assert InferenceGroupedMLP.refresh_flashinfer_mxfp8_weights(grouped_mlp) is False


def test_bf16_concatenated_weights_remain_refittable():
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    class ExpertLinear(torch.nn.Module):
        def __init__(self, num_experts, rows, cols):
            super().__init__()
            for expert_idx in range(num_experts):
                self.register_parameter(
                    f"weight{expert_idx}",
                    torch.nn.Parameter(torch.randn(rows, cols, dtype=torch.bfloat16)),
                )

    class GroupedMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_local_experts = 2
            self.linear_fc1 = ExpertLinear(self.num_local_experts, 8, 4)
            self.linear_fc2 = ExpertLinear(self.num_local_experts, 4, 8)

    grouped_mlp = GroupedMLP()
    InferenceGroupedMLP._build_concatenated_weights(grouped_mlp)
    fc1_ptr = grouped_mlp._fc1_weight.data_ptr()
    fc2_ptr = grouped_mlp._fc2_weight.data_ptr()

    with torch.no_grad():
        for expert_idx in range(grouped_mlp.num_local_experts):
            for linear_name, buffer_name in (
                ("linear_fc1", "_fc1_weight"),
                ("linear_fc2", "_fc2_weight"),
            ):
                parameter = getattr(getattr(grouped_mlp, linear_name), f"weight{expert_idx}")
                refit_value = torch.full_like(parameter, expert_idx + 1)
                parameter.copy_(refit_value)
                assert torch.equal(getattr(grouped_mlp, buffer_name)[expert_idx], refit_value)

    assert grouped_mlp._fc1_weight.data_ptr() == fc1_ptr
    assert grouped_mlp._fc2_weight.data_ptr() == fc2_ptr


def test_bf16_flashinfer_nvls_uses_dispatcher_copy_fallback(monkeypatch):
    from megatron.core.transformer.moe import experts

    full_rows = 16
    expected = torch.empty(full_rows, 8, dtype=torch.bfloat16)
    captured = {}

    def cutlass_fused_moe(hidden_states, routing_map, probs, *args, **kwargs):
        captured["output"] = kwargs["output"]
        captured["hidden_rows"] = hidden_states.shape[0]
        captured["last_route"] = routing_map[-1].item()
        return (expected,)

    monkeypatch.setattr(experts, "HAVE_FLASHINFER", True)
    monkeypatch.setattr(
        experts, "fused_moe", SimpleNamespace(cutlass_fused_moe=cutlass_fused_moe), raising=False
    )

    grouped_mlp = SimpleNamespace(
        _fc1_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
        _fc2_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
        _flashinfer_activation_type=object(),
        _activation_clamp_scale=None,
        _nvls_dispatcher=True,
        ep_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    routing_map = torch.full((full_rows, 1), -1, dtype=torch.int64)
    routing_map[-1] = 0
    output, bias = experts.InferenceGroupedMLP._flashinfer_forward(
        grouped_mlp,
        torch.empty(full_rows, 8, dtype=torch.bfloat16),
        routing_map,
        torch.zeros(full_rows, 1, dtype=torch.float32),
    )

    assert output is expected
    assert bias is None
    assert captured["output"] is None
    assert captured["hidden_rows"] == full_rows
    assert captured["last_route"] == 0


def test_bounded_bf16_flashinfer_uses_active_prefix_and_full_rsv_output(monkeypatch):
    from megatron.core.transformer.moe import experts

    full_rows, active_rows, hidden_size = 16, 4, 8
    expected = torch.arange(active_rows * hidden_size, dtype=torch.float32).reshape(
        active_rows, hidden_size
    )
    expected = expected.to(torch.bfloat16)
    rsv_output = torch.full((full_rows, hidden_size), -1.0, dtype=torch.float32)
    captured = {}

    def cutlass_fused_moe(hidden_states, routing_map, probs, *args, **kwargs):
        captured["hidden_shape"] = tuple(hidden_states.shape)
        captured["routing_shape"] = tuple(routing_map.shape)
        captured["prob_shape"] = tuple(probs.shape)
        return (expected,)

    monkeypatch.setattr(experts, "HAVE_FLASHINFER", True)
    monkeypatch.setattr(
        experts, "fused_moe", SimpleNamespace(cutlass_fused_moe=cutlass_fused_moe), raising=False
    )
    monkeypatch.setattr(
        experts.NVLSAllGatherVDispatcher, "_get_rsv_tensor", staticmethod(lambda: rsv_output)
    )

    grouped_mlp = SimpleNamespace(
        _fc1_weight=torch.empty(2, hidden_size, hidden_size, dtype=torch.bfloat16),
        _fc2_weight=torch.empty(2, hidden_size, hidden_size, dtype=torch.bfloat16),
        _flashinfer_activation_type=object(),
        _activation_clamp_scale=None,
        _nvls_dispatcher=True,
        ep_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    InferenceMode.set_active()
    InferenceMode.set_flashinfer_token_capacity(active_rows)

    output, bias = experts.InferenceGroupedMLP._flashinfer_forward(
        grouped_mlp,
        torch.empty(full_rows, hidden_size, dtype=torch.bfloat16),
        torch.zeros(full_rows, 1, dtype=torch.int64),
        torch.zeros(full_rows, 1, dtype=torch.float32),
    )

    assert output is rsv_output
    assert bias is None
    assert captured == {
        "hidden_shape": (active_rows, hidden_size),
        "routing_shape": (active_rows, 1),
        "prob_shape": (active_rows, 1),
    }
    assert torch.equal(rsv_output[:active_rows], expected.float())
    assert torch.equal(
        rsv_output[active_rows:],
        torch.full((full_rows - active_rows, hidden_size), -1.0, dtype=torch.float32),
    )
