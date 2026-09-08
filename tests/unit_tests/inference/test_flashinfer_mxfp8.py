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


def test_inference_mode_tracks_bounded_flashinfer_rows():
    InferenceMode.set_active()
    assert not InferenceMode.use_bounded_flashinfer_rows()

    InferenceMode.set_bounded_flashinfer_rows(True)
    assert InferenceMode.use_bounded_flashinfer_rows()

    InferenceMode.unset_active()
    assert not InferenceMode.use_bounded_flashinfer_rows()

    InferenceMode.set_active()
    assert not InferenceMode.use_bounded_flashinfer_rows()


@pytest.mark.parametrize(
    ("token_capacity", "use_bounded_rows", "expected"),
    [
        (None, False, (65536, "full")),
        (1024, False, (65536, "full")),
        (1024, True, (1024, "bounded-decode")),
        (131072, True, (65536, "bounded-decode")),
    ],
)
def test_flashinfer_active_row_policy(token_capacity, use_bounded_rows, expected):
    assert (
        select_flashinfer_active_rows(
            65536, token_capacity=token_capacity, use_bounded_rows=use_bounded_rows
        )
        == expected
    )


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
        inference_flashinfer_token_capacity=1024,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_mxfp8_config_accepts_nvls_ep():
    config = _make_bounded_mxfp8_config()

    assert config.inference_flashinfer_token_capacity == 1024
    assert config.inference_moe_token_dispatcher_type == "nvls"
    assert config.expert_model_parallel_size == 2


def test_deprecated_mxfp8_capacity_alias_populates_shared_capacity():
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        config = _make_bounded_mxfp8_config(
            inference_flashinfer_token_capacity=None, inference_flashinfer_mxfp8_token_capacity=512
        )

    assert config.inference_flashinfer_token_capacity == 512


def test_deprecated_mxfp8_capacity_alias_rejects_conflicting_value():
    with pytest.raises(ValueError, match="must match"):
        _make_bounded_mxfp8_config(inference_flashinfer_mxfp8_token_capacity=512)


def test_bf16_config_ignores_inactive_mxfp8_recipe_gates():
    config = _make_bounded_mxfp8_config(
        fp8=None, fp8_param=False, activation_func=F.gelu, inference_flashinfer_token_capacity=None
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
        inference_flashinfer_token_capacity=1024,
        fp8=None,
        fp8_param=False,
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_bf16_config_accepts_nvls_ep():
    config = _make_bounded_bf16_config()

    assert config.inference_flashinfer_token_capacity == 1024


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inference_flashinfer_token_capacity": 0}, "must be > 0"),
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
            {"inference_grouped_gemm_backend": "vllm", "inference_flashinfer_token_capacity": None},
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

    expected = torch.empty(4, 8, dtype=torch.bfloat16)
    captured = {}

    def cutlass_fused_moe(*args, **kwargs):
        captured["output"] = kwargs["output"]
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
        _flashinfer_token_capacity=None,
        _nvls_dispatcher=True,
        ep_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    output, bias = experts.InferenceGroupedMLP._flashinfer_forward(
        grouped_mlp,
        torch.empty(4, 8, dtype=torch.bfloat16),
        torch.zeros(4, 1, dtype=torch.int64),
        torch.zeros(4, 1, dtype=torch.float32),
    )

    assert output is expected
    assert bias is None
    assert captured["output"] is None


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
        _flashinfer_token_capacity=active_rows,
        _nvls_dispatcher=True,
        ep_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    InferenceMode.set_active()
    InferenceMode.set_bounded_flashinfer_rows(True)

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
