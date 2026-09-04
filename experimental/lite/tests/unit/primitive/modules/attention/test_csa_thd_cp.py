# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the DeepSeek-V4 CSA THD context-parallel path.

Coverage strategy (see the module docstring of
``megatron.core.transformer.experimental_attention_variant.csa_cp_layout_kernels``):
the full ``_forward_thd_cp`` is gated behind CuTeDSL/CUDA kernels
(``prepare_cp_compressor_input`` and ``build_attention_indices``), so its numeric
``CP=1 == unsharded`` invariant is only reachable on a GPU with those kernels and
is marked ``gpu``. The THD tensor layout built by the dispatch, the zero
left-boundary at ``cp_size == 1``, the boundary-KV projection, and the
differentiability of that pre-CP path require a GPU for real Transformer Engine
layers. The pre-grouped compressor overlap transform remains CPU-only.

These tests require a real Transformer Engine install because Megatron Core CSA
imports ``transformer_engine.pytorch.float8_tensor``. A missing dependency
therefore fails the strict standard harness.
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


def _fused_kernel_config():
    config = _tiny_config()
    config.num_attention_heads = 64
    config.head_dim = 512
    config.qk_rope_head_dim = 64
    config.index_head_dim = 128
    config.index_n_heads = 64
    return config


@pytest.fixture
def _single_rank_nccl():
    import os

    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    try:
        yield dist.group.WORLD
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


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
    tensor = torch.arange(n * ratio * 1 * (2 * d), dtype=torch.float32).reshape(
        n, ratio, 1, 2 * d
    )
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


def test_thd_cp_uses_current_core_layout_interfaces(monkeypatch):
    """Lite consumes the current Core compressor-prep and layout contracts."""
    csa = _csa()
    ratio, local_rows, d_window = 4, 8, 8
    hidden_size, head_dim, group_capacity = 4, 4, 8
    cu_seqlens = torch.tensor([0, local_rows], dtype=torch.int32)
    cu_seqlens_compressed = torch.tensor([0, local_rows // ratio], dtype=torch.int32)
    hidden_compact = torch.zeros(group_capacity * ratio, 1, hidden_size)
    compressed_group_ids = torch.tensor([0, 1, -1, -1, -1, -1, -1, -1], dtype=torch.int32)
    compressed_position_ids = torch.tensor([0, 4, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    local_cu_seqlens = torch.tensor([0, local_rows], dtype=torch.int32)
    local_cu_seqlens_compressed = cu_seqlens_compressed.clone()
    seq_to_rank_row = torch.tensor([0, 1], dtype=torch.int32)
    captured = {}

    def fake_prepare_cp_compressor_input(
        hidden_local, boundary_hidden, cu_seqlens_arg, global_start, cp_size, ratio_arg
    ):
        captured['prepare_args'] = (
            hidden_local,
            boundary_hidden,
            cu_seqlens_arg,
            global_start,
            cp_size,
            ratio_arg,
        )
        return (
            hidden_compact,
            compressed_group_ids,
            compressed_position_ids,
            local_cu_seqlens,
            local_cu_seqlens_compressed,
            cu_seqlens_compressed,
            seq_to_rank_row,
        )

    class FakeCompressor:
        def _forward_thd(
            self,
            compact,
            cu_seqlens_arg,
            *,
            max_seqlen_q,
            compressed_group_ids,
            compressed_position_ids,
        ):
            captured['compressor_args'] = (
                compact,
                cu_seqlens_arg,
                max_seqlen_q,
                compressed_group_ids,
                compressed_position_ids,
            )
            return torch.zeros(group_capacity, 1, head_dim), None

    def fake_gather(tensor, *, group):
        captured['gather_group'] = group
        return tensor

    def fake_build_attention_indices(
        cu_seqlens_arg,
        global_start,
        local_rows_arg,
        d_window_arg,
        window_size,
        ratio_arg,
        compressed_width,
        compressed_topk,
        **kwargs,
    ):
        captured['layout_args'] = (
            cu_seqlens_arg,
            global_start,
            local_rows_arg,
            d_window_arg,
            window_size,
            ratio_arg,
            compressed_width,
            compressed_topk,
            kwargs,
        )
        width = window_size + compressed_width
        return (
            torch.zeros(local_rows_arg, width, dtype=torch.int32),
            torch.full((local_rows_arg,), width, dtype=torch.int32),
            None,
            None,
        )

    monkeypatch.setattr(
        csa.cp_utils, 'prepare_cp_compressor_input', fake_prepare_cp_compressor_input
    )
    monkeypatch.setattr(csa, 'gather_from_sequence_parallel_region', fake_gather)
    monkeypatch.setattr(
        csa.csa_cp_layout_kernels, 'build_attention_indices', fake_build_attention_indices
    )
    monkeypatch.setattr(
        csa, 'unfused_compressed_sparse_attn', lambda query, _kv, _sinks, _indices, _scale: query
    )

    ps = _ps()
    module = SimpleNamespace(
        ps=ps,
        compressor=FakeCompressor(),
        compress_ratio=ratio,
        indexer=None,
        dsa_indexer_loss_coeff=0.0,
        training=False,
        dsa_indexer_use_sparse_loss=False,
        calculate_per_token_loss=False,
        config=SimpleNamespace(sliding_window=4),
        apply_dsa_kernel_fusion=False,
        sinks=torch.zeros(1),
        softmax_scale=1.0,
        _thd_cu_seqlens=lambda packed: packed.cu_seqlens_q,
    )
    query = torch.randn(local_rows, 1, head_dim)
    key = torch.randn(local_rows, 1, 1, head_dim)
    hidden = torch.randn(local_rows, 1, hidden_size)
    boundary_hidden = torch.randn(d_window, 1, hidden_size)
    boundary_kv = torch.randn(d_window, 1, 1, head_dim)
    packed = SimpleNamespace(cu_seqlens_q=cu_seqlens, max_seqlen_q=local_rows)

    output = csa.CompressedSparseAttention._forward_thd_cp(
        module,
        query,
        key,
        hidden,
        torch.empty(local_rows, 1, 0),
        boundary_hidden,
        boundary_kv,
        packed,
    )

    assert output.shape == (local_rows, 1, 1, head_dim)
    assert captured['prepare_args'][2] is cu_seqlens
    assert captured['prepare_args'][3:] == (0, 1, ratio)
    assert captured['compressor_args'][3] is compressed_group_ids
    assert captured['compressor_args'][4] is compressed_position_ids
    assert captured['gather_group'] is ps.cp_group
    layout_kwargs = captured['layout_args'][-1]
    assert layout_kwargs['cu_seqlens_compressed'] is cu_seqlens_compressed
    assert layout_kwargs['seq_to_rank_row'] is seq_to_rank_row
    assert layout_kwargs['compressed_rows'] == group_capacity


# ---------------------------------------------------------------------------
# THD dispatch + boundary-KV projection (one GPU with real TE; the
# CuTeDSL-gated core is stubbed out).
# ---------------------------------------------------------------------------


@pytest.mark.gpus(1)
def test_forward_thd_packed_builds_layout_and_is_differentiable(monkeypatch):
    csa = _csa()
    torch.manual_seed(0)
    config = _tiny_config()
    device = torch.device("cuda")
    try:
        module = csa.CompressedSparseAttention(config, layer_idx=0, ps=_ps()).to(device)
    except RuntimeError as exc:  # te.RMSNorm/te.Linear unavailable (stubbed TE)
        pytest.skip(f"real Transformer Engine required to build CSA: {exc}")

    seq_len = 8  # divisible by ratio; >= D_window so the boundary window fits.
    x = torch.randn(1, seq_len, config.hidden_size, device=device, requires_grad=True)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    psp = _packed_seq_params(seq_len, device)

    captured = {}

    def _fake_forward_thd_cp(
        query, key, x_thd, qr, boundary_hidden, boundary_kv, packed
    ):
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


@pytest.mark.gpus(1)
def test_project_boundary_kv_shape(monkeypatch):
    csa = _csa()
    torch.manual_seed(0)
    config = _tiny_config()
    device = torch.device("cuda")
    try:
        module = csa.CompressedSparseAttention(config, layer_idx=0, ps=_ps()).to(device)
    except RuntimeError as exc:
        pytest.skip(f"real Transformer Engine required to build CSA: {exc}")

    dwin = _d_window(config)
    boundary_hidden = torch.randn(dwin, 1, config.hidden_size, device=device)
    cu = torch.tensor([0, 8], dtype=torch.int32, device=device)
    bkv = module._project_boundary_kv(
        boundary_hidden, cu, global_start=0, rope_theta=config.compress_rope_theta
    )
    assert bkv.shape == (dwin, 1, 1, config.head_dim)
    assert torch.isfinite(bkv).all()


# ---------------------------------------------------------------------------
# CP=1 == unsharded numeric invariant (GPU + CuTeDSL only).
# ---------------------------------------------------------------------------


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + CuTeDSL kernels"
)
def test_cp1_thd_equals_bshd_fused(_single_rank_nccl):
    """CP=1 THD packed attention equals the (unsharded) CP=1 BSHD fused path.

    Both routes share the DSv4 sparse-attention formula (sliding window +
    compressed KV + learned indexer top-k + attention sink); for a single
    ratio-aligned sequence with no CP sharding they must produce the same
    output. This is the concrete ``CP=1 == unsharded`` CP invariant, runnable
    only where the CuTeDSL layout kernels and DSA kernels are available.
    """
    csa = _csa()
    torch.manual_seed(0)
    config = _fused_kernel_config()
    device = torch.device("cuda")
    ps = _ps()
    ps.cp_group = _single_rank_nccl
    module = csa.CompressedSparseAttention(config, layer_idx=0, ps=ps).to(
        device=device, dtype=torch.bfloat16
    )
    # Select a fused sparse backend for both the BSHD and THD routes.
    module.attention_backend = "flash"
    module.apply_dsa_kernel_fusion = True
    module.eval()

    seq_len = 8
    x = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        out_bshd = module(x, position_ids=position_ids)
        out_thd = module(
            x,
            position_ids=position_ids,
            packed_seq_params=_packed_seq_params(seq_len, device),
        )

    assert out_thd.shape == out_bshd.shape == (1, seq_len, config.hidden_size)
    torch.testing.assert_close(out_thd, out_bshd, rtol=2e-2, atol=2e-2)
