# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the DeepSeek-V4 CSA THD context-parallel path.

Coverage strategy (see the module docstring of
``megatron.core.transformer.experimental_attention_variant.csa_cp_layout_kernels``):
the full ``_forward_thd_cp`` is gated behind CuTeDSL/CUDA kernels
(``prepare_cp_compressor_input`` and ``build_attention_indices``), so its numeric
``CP=1 == unsharded`` invariant is only reachable on a GPU with those kernels and
is marked ``gpu``. The CPU-reachable pieces -- the THD tensor layout built by the
dispatch, the zero left-boundary at ``cp_size == 1``, the boundary-KV projection,
the differentiability of that pre-CP path, and the pre-grouped compressor overlap
transform -- are exercised directly here by stubbing ``_forward_thd_cp``.

These tests require a real Transformer Engine install (Megatron Core's CSA module
imports ``transformer_engine.pytorch.float8_tensor``); they skip cleanly when it
is absent rather than failing collection.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch



def _csa():
    """Import the lite CSA module or skip (needs real TE + Megatron Core)."""
    return pytest.importorskip("megatron.lite.primitive.modules.attention.csa")


def _tiny_config():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        compress_ratios=[4],
        sliding_window=4,
        index_head_dim=8,
        index_n_heads=4,
        index_topk=4,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )


def _ps(cp_size: int = 1, cp_rank: int = 0):
    group = SimpleNamespace(size=lambda: cp_size, rank=lambda: cp_rank)
    return SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank, cp_group=group)


def _packed_seq_params(seq_len: int, device: torch.device):
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv=cu,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
    )


def _d_window(config) -> int:
    d_comp = 8 if config.compress_ratios[0] == 4 else config.compress_ratios[0]
    return max(int(config.sliding_window), d_comp)


# ---------------------------------------------------------------------------
# Pure-tensor logic (no CUDA / no CuTeDSL).
# ---------------------------------------------------------------------------


def test_overlap_transform_thd_layout():
    """Pre-grouped THD overlap transform matches the DSv4 windowing contract."""
    csa = _csa()
    d = 3
    fake_self = SimpleNamespace(head_dim=d)
    ratio, n = 4, 2
    # (total_comp, ratio, 1, coff * d) with coff == 2.
    tensor = torch.arange(n * ratio * 1 * (2 * d), dtype=torch.float32).reshape(n, ratio, 1, 2 * d)
    is_first = torch.tensor([True, False])
    out = csa.CompressedSequenceCompressor._overlap_transform_thd(
        fake_self, tensor, is_first, fill_value=0.0
    )
    assert out.shape == (n, 2 * ratio, 1, d)
    # Second half is the [d:] channel of the same group.
    assert torch.equal(out[:, ratio:], tensor[:, :, :, d:])
    # First group starts a segment -> its first half is fill (0).
    assert torch.equal(out[0, :ratio], torch.zeros(ratio, 1, d))
    # Second group pulls the [:d] channel of the previous group.
    assert torch.equal(out[1, :ratio], tensor[0, :, :, :d])


# ---------------------------------------------------------------------------
# THD dispatch + boundary-KV projection (CPU-reachable with real TE; the
# CuTeDSL-gated core is stubbed out).
# ---------------------------------------------------------------------------


def test_forward_thd_packed_builds_layout_and_is_differentiable(monkeypatch):
    csa = _csa()
    torch.manual_seed(0)
    config = _tiny_config()
    try:
        module = csa.CompressedSparseAttention(config, layer_idx=0, ps=_ps())
    except RuntimeError as exc:  # te.RMSNorm/te.Linear unavailable (stubbed TE)
        pytest.skip(f"real Transformer Engine required to build CSA: {exc}")

    seq_len = 8  # divisible by ratio; >= D_window so the boundary window fits.
    device = torch.device("cpu")
    x = torch.randn(1, seq_len, config.hidden_size, device=device, requires_grad=True)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    psp = _packed_seq_params(seq_len, device)

    captured = {}

    def _fake_forward_thd_cp(query, key, x_thd, qr, boundary_hidden, boundary_kv, packed):
        captured.update(
            query=query,
            key=key,
            x_thd=x_thd,
            qr=qr,
            boundary_hidden=boundary_hidden,
            boundary_kv=boundary_kv,
        )
        # (total_q, 1, np * hn) attention-context contract; keep it in the graph.
        return query.reshape(seq_len, 1, config.num_attention_heads * config.head_dim)

    monkeypatch.setattr(module, "_forward_thd_cp", _fake_forward_thd_cp)

    out = module(x, position_ids=position_ids, packed_seq_params=psp)

    nh, hd, hidden = config.num_attention_heads, config.head_dim, config.hidden_size
    assert captured["query"].shape == (seq_len, nh, hd)
    assert captured["key"].shape == (seq_len, 1, 1, hd)
    assert captured["x_thd"].shape == (seq_len, 1, hidden)
    assert captured["qr"].shape == (seq_len, 1, config.q_lora_rank)
    dwin = _d_window(config)
    assert captured["boundary_hidden"].shape == (dwin, 1, hidden)
    # cp_size == 1 -> zero left boundary.
    assert torch.count_nonzero(captured["boundary_hidden"]) == 0
    assert captured["boundary_kv"].shape == (dwin, 1, 1, hd)
    # Output is projected back to [B, S, hidden] for the SBHD shim.
    assert out.shape == (1, seq_len, hidden)

    # The pre-CP path (projection + boundary + differentiable layout) must carry
    # gradients back to the hidden input (no non-differentiable collective).
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_project_boundary_kv_shape(monkeypatch):
    csa = _csa()
    torch.manual_seed(0)
    config = _tiny_config()
    try:
        module = csa.CompressedSparseAttention(config, layer_idx=0, ps=_ps())
    except RuntimeError as exc:
        pytest.skip(f"real Transformer Engine required to build CSA: {exc}")

    dwin = _d_window(config)
    boundary_hidden = torch.randn(dwin, 1, config.hidden_size)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    bkv = module._project_boundary_kv(
        boundary_hidden, cu, global_start=0, rope_theta=config.compress_rope_theta
    )
    assert bkv.shape == (dwin, 1, 1, config.head_dim)
    assert torch.isfinite(bkv).all()


# ---------------------------------------------------------------------------
# CP=1 == unsharded numeric invariant (GPU + CuTeDSL only).
# ---------------------------------------------------------------------------


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + CuTeDSL kernels")
def test_cp1_thd_equals_bshd_fused():
    """CP=1 THD packed attention equals the (unsharded) CP=1 BSHD fused path.

    Both routes share the DSv4 sparse-attention formula (sliding window +
    compressed KV + learned indexer top-k + attention sink); for a single
    ratio-aligned sequence with no CP sharding they must produce the same
    output. This is the concrete ``CP=1 == unsharded`` CP invariant, runnable
    only where the CuTeDSL layout kernels and DSA kernels are available.
    """
    csa = _csa()
    torch.manual_seed(0)
    config = _tiny_config()
    device = torch.device("cuda")
    module = csa.CompressedSparseAttention(config, layer_idx=0, ps=_ps()).to(device)
    # Select a fused sparse backend for both the BSHD and THD routes.
    module.attention_backend = "flash"
    module.apply_dsa_kernel_fusion = True
    module.eval()

    seq_len = 8
    x = torch.randn(1, seq_len, config.hidden_size, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        out_bshd = module(x, position_ids=position_ids)
        out_thd = module(
            x, position_ids=position_ids, packed_seq_params=_packed_seq_params(seq_len, device)
        )

    assert out_thd.shape == out_bshd.shape == (1, seq_len, config.hidden_size)
    torch.testing.assert_close(out_thd, out_bshd, rtol=2e-2, atol=2e-2)
