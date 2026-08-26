# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.moe.flashinfer_mxfp8 import select_routed_mxfp8_active_rows
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def reset_inference_mode():
    InferenceMode.unset_active()
    yield
    InferenceMode.unset_active()


def test_inference_mode_tracks_decode_state():
    InferenceMode.set_active()
    assert not InferenceMode.is_decode_only()
    assert InferenceMode.decode_token_upper_bound() is None

    InferenceMode.set_decode_state(True, 512)
    assert InferenceMode.is_decode_only()
    assert InferenceMode.decode_token_upper_bound() == 512

    InferenceMode.unset_active()
    assert not InferenceMode.is_decode_only()
    assert InferenceMode.decode_token_upper_bound() is None


def test_inference_mode_rejects_invalid_decode_bound():
    with pytest.raises(ValueError, match="must be positive"):
        InferenceMode.set_decode_state(True, 0)


@pytest.mark.parametrize(
    ("token_capacity", "decode_only", "decode_upper_bound", "expected"),
    [
        (None, False, None, (65536, "full")),
        (1024, False, 512, (65536, "full-mixed")),
        (1024, True, 512, (1024, "bounded-decode")),
        (1024, True, 2048, (65536, "full-decode-over-capacity")),
        (131072, True, 512, (65536, "bounded-decode")),
    ],
)
def test_flashinfer_mxfp8_active_row_policy(
    token_capacity, decode_only, decode_upper_bound, expected
):
    assert (
        select_routed_mxfp8_active_rows(
            65536,
            token_capacity=token_capacity,
            decode_only=decode_only,
            decode_token_upper_bound=decode_upper_bound,
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
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_mxfp8_token_capacity=1024,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_mxfp8_config_accepts_nvls_ep():
    config = _make_bounded_mxfp8_config()

    assert config.inference_moe_token_dispatcher_type == "nvls"
    assert config.expert_model_parallel_size == 2


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inference_moe_token_dispatcher_type": "nccl"}, "requires.*nvls"),
        ({"expert_model_parallel_size": 1}, "requires.*expert_model_parallel_size > 1"),
    ],
)
def test_bounded_flashinfer_mxfp8_config_rejects_unsafe_dispatch(overrides, match):
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
