# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.inference.moe.flashinfer_mxfp8 import (
    HAVE_FLASHINFER_ROUTED_MXFP8,
    select_routed_mxfp8_active_rows,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def reset_inference_mode():
    InferenceMode.unset_active()
    yield
    InferenceMode.unset_active()


def test_inference_mode_tracks_bounded_mxfp8_rows():
    InferenceMode.set_active()
    assert not InferenceMode.use_bounded_mxfp8_rows()

    InferenceMode.set_bounded_mxfp8_rows(True)
    assert InferenceMode.use_bounded_mxfp8_rows()

    InferenceMode.unset_active()
    assert not InferenceMode.use_bounded_mxfp8_rows()


@pytest.mark.parametrize(
    ("token_capacity", "use_bounded_rows", "expected"),
    [
        (None, False, (65536, "full")),
        (1024, False, (65536, "full")),
        (1024, True, (1024, "bounded-decode")),
        (131072, True, (65536, "bounded-decode")),
    ],
)
def test_flashinfer_mxfp8_active_row_policy(token_capacity, use_bounded_rows, expected):
    assert (
        select_routed_mxfp8_active_rows(
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
        inference_flashinfer_mxfp8_token_capacity=1024,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_mxfp8_config_accepts_nvls_ep():
    config = _make_bounded_mxfp8_config()

    assert config.inference_moe_token_dispatcher_type == "nvls"
    assert config.expert_model_parallel_size == 2


def test_flashinfer_mxfp8_config_accepts_batch_invariant_mode():
    config = _make_bounded_mxfp8_config(
        batch_invariant_mode=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
    )

    assert config.batch_invariant_mode


def test_torch_mxfp8_config_accepts_batch_invariant_mode():
    config = _make_bounded_mxfp8_config(
        inference_grouped_gemm_backend="torch",
        inference_flashinfer_mxfp8_token_capacity=None,
        batch_invariant_mode=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
    )

    assert config.batch_invariant_mode


def test_vllm_backend_accepts_mxfp8_config_for_per_layer_dispatch():
    config = _make_bounded_mxfp8_config(
        inference_grouped_gemm_backend="vllm", inference_flashinfer_mxfp8_token_capacity=None
    )

    assert config.inference_grouped_gemm_backend.value == "vllm"


def test_bf16_config_ignores_inactive_mxfp8_recipe_gates():
    config = _make_bounded_mxfp8_config(
        fp8=None,
        fp8_param=False,
        activation_func=F.gelu,
        inference_flashinfer_mxfp8_token_capacity=None,
    )

    assert config.fp8 is None


@pytest.mark.parametrize("activation_func", [F.gelu, F.silu, F.relu])
def test_flashinfer_mxfp8_config_rejects_unsupported_activation(activation_func):
    with pytest.raises(ValueError, match="supports only non-gated squared-ReLU experts"):
        _make_bounded_mxfp8_config(activation_func=activation_func)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fp8_param": False}, "fp8_param must be enabled"),
        ({"fp8": None, "fp8_param": False}, "requires.*FP8 enabled"),
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


def _blackwell_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.skipif(
    not HAVE_FLASHINFER_ROUTED_MXFP8 or not _blackwell_or_newer(),
    reason="FlashInfer routed MXFP8 requires Blackwell and FlashInfer >= 0.6.4",
)
@pytest.mark.parametrize("local_expert_offset", [0, 2, 4, 6])
def test_flashinfer_routed_mxfp8_is_batch_invariant(local_expert_offset):
    """A token's local-EP output must not depend on its batch or row position."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        flashinfer_routed_mxfp8_moe,
        prepare_routed_mxfp8_weights,
    )
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    torch.manual_seed(1234)
    num_experts, hidden_size, intermediate_size, topk = 8, 256, 256, 2

    def _prepare_weight(weight: torch.Tensor):
        quantized = [MXFP8Tensor.from_bf16(expert, backend="triton") for expert in weight]
        stacked = MXFP8Tensor(
            data=torch.stack([expert.data for expert in quantized]).contiguous(),
            scale=torch.stack([expert.scale for expert in quantized]).contiguous(),
            backend="triton",
            dtype=torch.bfloat16,
        )
        return prepare_routed_mxfp8_weights(stacked)

    fc1_weight = _prepare_weight(
        torch.randn(
            num_experts, intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )[local_expert_offset : local_expert_offset + 2]
    )
    fc2_weight = _prepare_weight(
        torch.randn(
            num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16
        )[local_expert_offset : local_expert_offset + 2]
    )
    target = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    target_experts = torch.tensor([2, 5], device="cuda", dtype=torch.int64)
    target_probs = torch.tensor([0.625, 0.375], device="cuda", dtype=torch.float32)

    def _run(active_tokens: int, target_row: int) -> torch.Tensor:
        num_tokens = 512
        hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        routing_map = torch.full((num_tokens, topk), -1, device="cuda", dtype=torch.int64)
        routing_map[:active_tokens, 0] = 2
        routing_map[:active_tokens, 1] = 3
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        probs[:active_tokens].fill_(0.5)
        hidden_states[target_row].copy_(target)
        routing_map[target_row].copy_(target_experts)
        probs[target_row].copy_(target_probs)
        return flashinfer_routed_mxfp8_moe(
            hidden_states,
            routing_map,
            probs,
            fc1_weight,
            fc2_weight,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            activation_type=6,
        )[target_row]

    with torch.no_grad():
        output_alone = _run(1, 0)
        output_batched = _run(128, 73)

    assert torch.equal(output_alone, output_batched), (
        "FlashInfer routed MXFP8 output changed with the batch; max abs diff: "
        f"{(output_alone.float() - output_batched.float()).abs().max().item()}"
    )
    if not any(
        local_expert_offset <= expert < local_expert_offset + 2
        for expert in target_experts.tolist()
    ):
        assert torch.count_nonzero(output_alone).item() == 0


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


def test_flashinfer_nvls_clears_routing_before_metadata_fence(monkeypatch):
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe import token_dispatcher_inference

    dispatcher_cls = token_dispatcher_inference.NVLSAllGatherVDispatcher
    base_cls = token_dispatcher_inference.InferenceAllGatherDispatcherBase
    calls = []

    class RoutingTensor:
        def fill_(self, value):
            calls.append(("clear", value))

    monkeypatch.setattr(dispatcher_cls, "_symm_agv_routing", {"tensor": RoutingTensor()})
    monkeypatch.setattr(
        dispatcher_cls, "_symm_metadata", {"tensor": object(), "handle": object()}, raising=False
    )
    monkeypatch.setattr(dispatcher_cls, "_step_metadata", object())
    monkeypatch.setattr(base_cls, "_host_valid_tokens_estimate", 0)
    monkeypatch.setattr(
        token_dispatcher_inference,
        "fused_metadata_update",
        lambda **kwargs: calls.append(("metadata", kwargs["local_tokens"])),
    )
    dispatcher = SimpleNamespace(
        config=SimpleNamespace(
            inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER
        ),
        ep_size=4,
    )

    dispatcher_cls.update_metadata(dispatcher, local_tokens=7)

    assert calls == [("clear", -1), ("metadata", 7)]
    assert base_cls._host_valid_tokens_estimate == 28
