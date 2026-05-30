# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for Qwen3.5-VL vision RoPE fusion dispatch."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import megatron.core.models.common.embeddings.rope_utils as rope_utils
from examples.multimodal_dev.models.qwen35_vl.configuration import get_qwen35_vl_vision_config
from examples.multimodal_dev.models.qwen35_vl.specs import _apply_rope_fp32_no_cp
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.fusions.fused_mrope import is_fused_mrope_available, mrope_freqs_to_rotary_emb


class _FakeVisionRotaryEmbedding:
    def __init__(self, axis_dim):
        self.axis_dim = axis_dim

    def __call__(self, seqlen, device=None):
        device = torch.device("cpu") if device is None else device
        positions = torch.arange(seqlen, device=device, dtype=torch.float32)[:, None]
        dims = torch.arange(self.axis_dim, device=device, dtype=torch.float32)[None, :]
        return positions * 0.125 + dims * 0.01


def test_vision_config_sets_2d_rope_as_sectioned_raw_mrope():
    config = get_qwen35_vl_vision_config(variant="0.8b")

    assert config.kv_channels == 64
    assert config.mrope_section == [0, 16, 16]
    assert config.mrope_interleaved is False
    assert config.rotary_interleaved is False


def test_vision_raw_mrope_freqs_match_legacy_materialized_rope():
    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 2]], dtype=torch.long)
    legacy_encoder = SimpleNamespace(
        spatial_merge_size=2,
        rot_pos_emb=_FakeVisionRotaryEmbedding(axis_dim=16),
        config=SimpleNamespace(mrope_section=None),
    )
    fused_encoder = SimpleNamespace(
        spatial_merge_size=2,
        rot_pos_emb=_FakeVisionRotaryEmbedding(axis_dim=16),
        config=SimpleNamespace(mrope_section=[0, 16, 16]),
    )

    legacy_freqs = Qwen35VLVisionEncoder._compute_rotary_pos_emb(legacy_encoder, grid_thw)
    raw_freqs = Qwen35VLVisionEncoder._compute_rotary_pos_emb(fused_encoder, grid_thw)

    expected = torch.cat((legacy_freqs, legacy_freqs), dim=-1).unsqueeze(1).unsqueeze(1)
    converted = mrope_freqs_to_rotary_emb(
        raw_freqs,
        [0, 16, 16],
        interleaved_mrope=False,
        rotary_interleaved=False,
    )

    assert raw_freqs.shape == (3, 1, legacy_freqs.shape[0], legacy_freqs.shape[1])
    torch.testing.assert_close(converted, expected)


def test_vision_fp32_wrapper_dispatches_raw_freqs_to_fused_mrope_thd(monkeypatch):
    calls = {}

    def fake_fused_apply_mrope_thd(
        t,
        cu_seqlens,
        freqs,
        mrope_section,
        interleaved_mrope=False,
        rotary_interleaved=False,
        cp_size=1,
        cp_rank=0,
        fp32_compute=False,
    ):
        calls["t_shape"] = tuple(t.shape)
        calls["t_dtype"] = t.dtype
        calls["cu_seqlens"] = cu_seqlens.tolist()
        calls["freqs_shape"] = tuple(freqs.shape)
        calls["mrope_section"] = list(mrope_section)
        calls["interleaved_mrope"] = interleaved_mrope
        calls["rotary_interleaved"] = rotary_interleaved
        calls["cp_size"] = cp_size
        calls["cp_rank"] = cp_rank
        calls["fp32_compute"] = fp32_compute
        return t + 1.0

    monkeypatch.setattr(rope_utils, "fused_apply_mrope_thd", fake_fused_apply_mrope_thd)
    monkeypatch.setattr(rope_utils, "get_fused_mrope_thd_unavailable_reason", lambda *args, **kwargs: None)

    config = SimpleNamespace(
        apply_rope_fusion=True,
        mrope_section=[0, 2, 2],
        mrope_interleaved=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )
    t = torch.zeros(6, 2, 8, dtype=torch.bfloat16)
    freqs = torch.zeros(3, 1, 6, 4, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

    out = _apply_rope_fp32_no_cp(t, freqs, config, cu_seqlens=cu_seqlens)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, torch.ones_like(out))
    assert calls == {
        "t_shape": (6, 2, 8),
        "t_dtype": torch.bfloat16,
        "cu_seqlens": [0, 3, 6],
        "freqs_shape": (3, 1, 6, 4),
        "mrope_section": [0, 2, 2],
        "interleaved_mrope": False,
        "rotary_interleaved": False,
        "cp_size": 1,
        "cp_rank": 0,
        "fp32_compute": True,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_vision_fused_rope_matches_unfused_forward_backward_cuda():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    total_tokens = 64
    num_heads = 2
    head_dim = 72
    half_rotary_dim = head_dim // 2
    section = [0, half_rotary_dim // 2, half_rotary_dim // 2]
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

    t_ref = torch.randn(
        total_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    t_fused = t_ref.detach().clone().requires_grad_(True)
    freqs = torch.randn(
        3,
        1,
        total_tokens,
        half_rotary_dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    ref_config = SimpleNamespace(
        apply_rope_fusion=False,
        mrope_section=section,
        mrope_interleaved=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )
    fused_config = SimpleNamespace(
        apply_rope_fusion=True,
        mrope_section=section,
        mrope_interleaved=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )

    ref = _apply_rope_fp32_no_cp(t_ref, freqs, ref_config, cu_seqlens=cu_seqlens)
    out = _apply_rope_fp32_no_cp(t_fused, freqs, fused_config, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref.float(), out.float(), rtol=2.0e-2, atol=5.0e-2)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), rtol=2.0e-2, atol=5.0e-2)
