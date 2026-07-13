# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Qwen3.5-VL layer spec helpers."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.qwen35_vl import specs
from megatron.core.models.common.embeddings import rope_utils


def test_apply_rope_fp32_forwards_bshd_options(monkeypatch):
    """The BSHD wrapper preserves every supported non-default RoPE option."""
    captured = {}
    tensor = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)
    freqs = torch.randn(2, 1, 1, 8)
    expected = torch.randn_like(tensor, dtype=torch.float32)

    def fake_apply_bshd(input_tensor, input_freqs, **kwargs):
        captured["input_tensor"] = input_tensor
        captured["input_freqs"] = input_freqs
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_bshd", fake_apply_bshd)

    output = specs._apply_rope_fp32(
        tensor,
        freqs,
        SimpleNamespace(rotary_interleaved=True),
        mscale=0.75,
        mla_rotary_interleaved=True,
        inverse=True,
        mla_output_remove_interleaving=True,
    )

    assert captured["input_tensor"].dtype == torch.float32
    assert captured["input_freqs"] is freqs
    assert captured["kwargs"] == {
        "rotary_interleaved": True,
        "mla_rotary_interleaved": True,
        "mscale": 0.75,
        "inverse": True,
        "mla_output_remove_interleaving": True,
    }
    assert output.dtype == tensor.dtype
    torch.testing.assert_close(output, expected.to(tensor.dtype))


@pytest.mark.parametrize("config_value", [False, True])
def test_apply_rope_fp32_defaults_mla_interleaving_from_config(monkeypatch, config_value):
    """An omitted MLA option preserves the wrapper's config-derived legacy behavior."""
    captured = {}
    tensor = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)
    freqs = torch.randn(2, 1, 1, 8)

    def fake_apply_bshd(input_tensor, input_freqs, **kwargs):
        captured["kwargs"] = kwargs
        return input_tensor

    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_bshd", fake_apply_bshd)

    specs._apply_rope_fp32(
        tensor,
        freqs,
        SimpleNamespace(rotary_interleaved=False, multi_latent_attention=config_value),
    )

    assert captured["kwargs"]["mla_rotary_interleaved"] is config_value


def test_apply_rope_fp32_forwards_thd_options(monkeypatch):
    """The THD wrapper also preserves inverse/interleaving and max sequence length."""
    captured = {}
    tensor = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    freqs = torch.randn(2, 1, 1, 8)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    cp_group = object()
    expected = torch.randn_like(tensor, dtype=torch.float32)

    def fake_apply_thd(input_tensor, input_cu_seqlens, input_freqs, **kwargs):
        captured["input_tensor"] = input_tensor
        captured["input_cu_seqlens"] = input_cu_seqlens
        captured["input_freqs"] = input_freqs
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_thd", fake_apply_thd)

    output = specs._apply_rope_fp32(
        tensor,
        freqs,
        SimpleNamespace(rotary_interleaved=True),
        cu_seqlens=cu_seqlens,
        mscale=0.5,
        cp_group=cp_group,
        mla_rotary_interleaved=True,
        inverse=True,
        mla_output_remove_interleaving=True,
        max_seqlen=2,
    )

    assert captured["input_tensor"].dtype == torch.float32
    assert captured["input_cu_seqlens"] is cu_seqlens
    assert captured["input_freqs"] is freqs
    assert captured["kwargs"] == {
        "rotary_interleaved": True,
        "mla_rotary_interleaved": True,
        "mscale": 0.5,
        "inverse": True,
        "mla_output_remove_interleaving": True,
        "cp_group": cp_group,
        "max_seqlen": 2,
    }
    assert output.dtype == tensor.dtype
    torch.testing.assert_close(output, expected.to(tensor.dtype))


def test_apply_rope_fp32_no_cp_forwards_options_and_forces_trivial_group(monkeypatch):
    """The vision wrapper forwards RoPE options while overriding the caller's CP group."""
    captured = {}
    sentinel = object()

    def fake_apply_rope(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(specs, "_apply_rope_fp32", fake_apply_rope)

    output = specs._apply_rope_fp32_no_cp(
        object(),
        object(),
        object(),
        cu_seqlens=object(),
        mscale=0.25,
        cp_group=object(),
        mla_rotary_interleaved=True,
        inverse=True,
        mla_output_remove_interleaving=True,
        max_seqlen=17,
    )

    assert output is sentinel
    assert captured["kwargs"] == {
        "cp_group": specs._NO_CP_GROUP,
        "mla_rotary_interleaved": True,
        "inverse": True,
        "mla_output_remove_interleaving": True,
        "max_seqlen": 17,
    }
    assert captured["args"][4] == 0.25


@pytest.mark.parametrize("wrapper", [specs._apply_rope_fp32, specs._apply_rope_fp32_no_cp])
def test_apply_rope_fp32_wrappers_reject_unknown_options(wrapper):
    """Unsupported RoPE options fail loudly instead of being silently ignored."""
    signature = inspect.signature(wrapper)

    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        wrapper(object(), object(), object(), unsupported_option=True)


def test_qwen_vision_encoder_requires_observed_rotary_graph_inputs():
    """The vision helper must not capture Qwen attention without its runtime 2-D RoPE."""
    from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
    assert Qwen35VLVisionEncoder._cuda_graph_requires_observed_rotary_inputs is True
