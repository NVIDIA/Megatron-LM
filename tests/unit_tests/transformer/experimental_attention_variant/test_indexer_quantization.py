# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for DSA indexer quantization utilities."""

import pytest
import torch

from megatron.core.quantization.indexer_quantization import (
    HAVE_TE_MXFP8,
    HAVE_TRITON,
    create_indexer_mxfp8_quantization_buffers,
    indexer_mxfp8_scale_shape,
    indexer_mxfp8_thd_scale_capacity,
    indexer_mxfp8_thd_scale_shape,
    make_indexer_mxfp8_scale_cu_seqlens,
    quantize_indexer_mxfp8,
    refresh_indexer_mxfp8_scale_cu_seqlens,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 indexer quantization requires SM100+",
    ),
    pytest.mark.skipif(not HAVE_TE_MXFP8, reason="Transformer Engine MXFP8 not available"),
    pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available"),
]


def _unpack_scales(scale: torch.Tensor, logical_rows: int) -> torch.Tensor:
    """Undo the Blackwell 128x4 scale swizzle into logical ``(L, M, G)``."""
    l_size, _, padded_groups = scale.shape
    scale_groups = 4
    rows = torch.arange(logical_rows, device=scale.device).view(-1, 1)
    groups = torch.arange(scale_groups, device=scale.device).view(1, -1)
    tile_idx = (rows // 128) * (padded_groups // 4) + groups // 4
    offsets = tile_idx * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4 + groups % 4
    bytes_per_l = scale.shape[1] * padded_groups
    l_offsets = torch.arange(l_size, device=scale.device).view(-1, 1, 1)
    offsets = offsets.unsqueeze(0) + l_offsets * bytes_per_l
    scale_bytes = scale.view(torch.uint8).flatten()[offsets]
    return scale_bytes.contiguous().view(torch.float8_e8m0fnu).float()


def _reference_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for E4M3 values and round-up E8M0 scales."""
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = grouped.abs().amax(dim=-1)
    raw_scale = amax / 448.0
    scale = torch.where(
        raw_scale == 0,
        torch.zeros_like(raw_scale),
        torch.pow(2.0, torch.ceil(torch.log2(raw_scale))),
    )
    quant_scale = torch.where(scale == 0, torch.zeros_like(scale), scale.reciprocal())
    data = (grouped * quant_scale.unsqueeze(-1)).reshape_as(x).to(torch.float8_e4m3fn)
    return data, scale


def test_bshd_quantization_matches_reference():
    torch.manual_seed(123)
    batch_size, seqlen, num_heads, head_dim = 2, 3, 64, 128
    x = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    data, packed_scale = quantize_indexer_mxfp8(x)
    ref_data, ref_scale = _reference_quantize(x)
    unpacked_scale = _unpack_scales(packed_scale, seqlen * num_heads).reshape_as(ref_scale)

    assert data.dtype == torch.float8_e4m3fn
    assert packed_scale.dtype == torch.float8_e8m0fnu
    assert packed_scale.shape == indexer_mxfp8_scale_shape(batch_size, seqlen, num_heads, head_dim)
    assert torch.equal(data.float(), ref_data.float())
    torch.testing.assert_close(unpacked_scale, ref_scale, rtol=0, atol=0)


def test_thd_quantization_uses_concatenated_padded_scale_spans():
    torch.manual_seed(456)
    q_lens = [1, 2, 3, 4, 5]
    num_heads, head_dim = 64, 128
    cu_seqlens = torch.tensor([0, 1, 3, 6, 10, 15], dtype=torch.int32, device="cuda")
    cu_seqlens_scale_padded = make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens, num_heads)
    x = torch.randn(sum(q_lens), num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    data, packed_scale = quantize_indexer_mxfp8(
        x, cu_seqlens=cu_seqlens, cu_seqlens_scale_padded=cu_seqlens_scale_padded
    )
    ref_data, ref_scale = _reference_quantize(x)
    unpacked_scale = _unpack_scales(packed_scale, packed_scale.shape[1])[0]

    assert torch.equal(data.float(), ref_data.float())
    assert torch.equal(
        cu_seqlens_scale_padded,
        torch.tensor([0, 2, 4, 8, 12, 18], dtype=torch.int32, device="cuda"),
    )
    assert packed_scale.shape == indexer_mxfp8_thd_scale_shape(18, num_heads, head_dim)
    start = 0
    for batch, length in enumerate(q_lens):
        scale_start = int(cu_seqlens_scale_padded[batch].item()) * num_heads
        actual = unpacked_scale[scale_start : scale_start + length * num_heads]
        expected = ref_scale[start : start + length].reshape(length * num_heads, -1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        padded_end = int(cu_seqlens_scale_padded[batch + 1].item()) * num_heads
        padding = unpacked_scale[scale_start + length * num_heads : padded_end]
        # E8M0 has no numerical zero: a zero storage byte decodes to its minimum value.
        assert torch.all(padding == torch.finfo(torch.float8_e8m0fnu).tiny)
        start += length


@pytest.mark.parametrize("num_heads", [1, 64])
def test_thd_preallocated_quantization_cuda_graph_replay(num_heads):
    torch.manual_seed(789)
    capture_lens = [128, 128]
    replay_lens = [1, 255]
    head_dim = 128
    total_tokens = sum(capture_lens)
    cu_seqlens = torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda")
    cu_seqlens_scale_padded = make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens, num_heads)
    shape = (total_tokens, num_heads, head_dim) if num_heads > 1 else (total_tokens, head_dim)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    buffers = create_indexer_mxfp8_quantization_buffers(x)
    scale_capacity = indexer_mxfp8_thd_scale_capacity(total_tokens, len(capture_lens), num_heads)
    assert int(cu_seqlens_scale_padded[-1].item()) < scale_capacity
    out_scale = torch.zeros(
        indexer_mxfp8_thd_scale_shape(scale_capacity, num_heads, head_dim),
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )

    def run():
        refresh_indexer_mxfp8_scale_cu_seqlens(cu_seqlens_scale_padded, cu_seqlens, num_heads)
        return quantize_indexer_mxfp8(
            x,
            cu_seqlens=cu_seqlens,
            cu_seqlens_scale_padded=cu_seqlens_scale_padded,
            buffers=buffers,
            out_scale=out_scale,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_data, captured_scale = run()

    assert captured_data.data_ptr() == buffers.data.data_ptr()
    assert captured_scale.data_ptr() == out_scale.data_ptr()
    first_data = captured_data.float().clone()

    x.copy_(torch.randn_like(x))
    cu_seqlens.copy_(
        torch.tensor([0, replay_lens[0], sum(replay_lens)], dtype=torch.int32, device="cuda")
    )
    graph.replay()
    torch.cuda.synchronize()
    expected_scale_prefix = make_indexer_mxfp8_scale_cu_seqlens(cu_seqlens, num_heads)
    expected_data, expected_scale = quantize_indexer_mxfp8(
        x, cu_seqlens=cu_seqlens, cu_seqlens_scale_padded=expected_scale_prefix
    )
    assert int(expected_scale_prefix[-1].item()) == scale_capacity
    assert torch.equal(cu_seqlens_scale_padded, expected_scale_prefix)
    assert not torch.equal(captured_data.float(), first_data)
    assert torch.equal(captured_data.float(), expected_data.float())
    assert torch.equal(captured_scale.view(torch.uint8), expected_scale.view(torch.uint8))
