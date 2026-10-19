# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row chunking of the cuDNN indexer top-k wrapper (scratch budget, int32 limit, alignment)."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

_GIB = 1024**3
_FP32_BYTES = 4


def _scratch_bytes(rows: int, num_cols: int) -> int:
    # cuDNN radix top-k: int32 scratch of rows x 2 x num_cols for FP32 scores.
    return rows * 2 * num_cols * 4


@pytest.fixture(autouse=True)
def _eight_gib_scratch_budget(monkeypatch):
    """The cases below are sized for an 8 GiB scratch budget (1024 rows per 1M-key launch)."""
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 8 * _GIB)


# (case, rows, score columns, expected rows per launch). The 1M rows are the CP64 rank 0/21/42/63
# score-buffer widths of a 1M-token sequence; 128K is CP32 rank 0.
_TOPK_SHAPES = [
    ("1m_cp64_rank0", 16384, 1048576, 1024),
    ("1m_cp64_rank21", 16384, 876544, 1224),
    ("1m_cp64_rank42", 16384, 704512, 1524),
    ("1m_cp64_rank63", 16384, 532480, 2016),
    ("128k_cp32_rank0", 4096, 131072, 4096),
    ("4k", 4096, 4096, 4096),
    ("4k_many_rows", 65536, 4096, 65536),
]


@pytest.mark.parametrize(
    "rows,num_cols,expected", [case[1:] for case in _TOPK_SHAPES], ids=[c[0] for c in _TOPK_SHAPES]
)
def test_topk_chunk_rows_fill_the_8gib_budget_without_a_row_cap(rows, num_cols, expected):
    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(rows, num_cols, _FP32_BYTES)

    assert chunk_rows == expected
    assert _scratch_bytes(chunk_rows, num_cols) <= 8 * _GIB
    assert chunk_rows * 2 * num_cols <= 2**31
    if chunk_rows < rows:
        # The budget, not a row cap, is what stops the chunk from growing.
        assert _scratch_bytes(chunk_rows + 1, num_cols) > 8 * _GIB


def test_topk_chunk_rows_keep_the_128k_call_in_one_launch_above_1024_rows():
    assert dsa_cudnn_kernels._topk_wrapper_chunk_rows(4096, 131072, _FP32_BYTES) == 4096
    assert dsa_cudnn_kernels._topk_wrapper_chunk_rows(8192, 131072, _FP32_BYTES) == 8192
    assert dsa_cudnn_kernels._topk_wrapper_chunk_rows(8193, 131072, _FP32_BYTES) == 8192


@pytest.mark.parametrize(
    "num_cols,expected_rows,expected_alignment",
    [
        # 4 * 1048577 bytes per row is 4 (mod 32): starts must be multiples of 8 rows.
        (1048577, 1016, 8),
        # A 32-byte multiple row needs no alignment. A 512-row alignment floored this to 512.
        (1050000, 1022, 1),
        (1015808, 1057, 1),
    ],
)
def test_topk_chunk_rows_align_only_to_the_32_byte_input_alignment(
    num_cols, expected_rows, expected_alignment
):
    assert (
        dsa_cudnn_kernels._topk_wrapper_row_alignment(num_cols, _FP32_BYTES) == expected_alignment
    )
    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(16384, num_cols, _FP32_BYTES)
    assert chunk_rows == expected_rows
    assert chunk_rows % expected_alignment == 0
    assert (chunk_rows * num_cols * _FP32_BYTES) % 32 == 0


@pytest.mark.parametrize("element_size,num_cols,expected", [(2, 4097, 16), (2, 4096, 1), (4, 6, 4)])
def test_topk_wrapper_row_alignment_depends_on_row_bytes(element_size, num_cols, expected):
    assert dsa_cudnn_kernels._topk_wrapper_row_alignment(num_cols, element_size) == expected


def test_topk_chunk_rows_respect_the_frontend_int32_limit_above_the_byte_budget(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 64 * _GIB)

    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(16384, 1048576, _FP32_BYTES)

    assert chunk_rows == 1024
    assert chunk_rows * 2 * 1048576 <= 2**31


def test_topk_chunk_rows_follow_a_smaller_byte_budget(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", int(4.75 * _GIB))

    # 608 rows (one wave of 4 CTAs/SM on 152 SMs) is no longer floored to a 512-row multiple.
    assert dsa_cudnn_kernels._topk_wrapper_chunk_rows(16384, 1048576, _FP32_BYTES) == 608


class _RecordingTopK:
    """Fake ``indexer_top_k_wrapper`` that records the launched chunks on the meta device."""

    def __init__(self):
        self.calls = []

    def indexer_top_k_wrapper(self, scores_flat, seq_lens, top_k, next_n, return_val):
        assert scores_flat.is_contiguous()
        assert seq_lens.numel() == scores_flat.size(0)
        self.calls.append(
            (scores_flat.size(0), scores_flat.storage_offset() * scores_flat.element_size())
        )
        rows = scores_flat.size(0)
        indices = torch.empty((rows, top_k), dtype=torch.int32, device=scores_flat.device)
        values = (
            torch.empty((rows, top_k), dtype=scores_flat.dtype, device=scores_flat.device)
            if return_val
            else None
        )
        return {"indices": indices, "values": values}


@pytest.mark.parametrize(
    "rows,num_cols,expected_launches",
    [
        (16384, 1048576, [1024] * 16),
        (16384, 532480, [2016] * 8 + [256]),
        (4096, 131072, [4096]),
        (64, 1048577, [64]),
    ],
)
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_topk_wrapper_launches_budget_sized_chunks(
    monkeypatch, rows, num_cols, expected_launches, return_topk_scores
):
    fake = _RecordingTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    scores = torch.empty((rows, num_cols), dtype=torch.float32, device="meta")
    seq_lens = torch.empty((rows,), dtype=torch.int32, device="meta")

    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=2048, return_topk_scores=return_topk_scores
    )

    assert [launch_rows for launch_rows, _ in fake.calls] == expected_launches
    assert all(offset_bytes % 32 == 0 for _, offset_bytes in fake.calls)
    assert result["indices"].shape == (rows, 2048)
    assert (result["values"] is not None) == return_topk_scores


def test_topk_wrapper_keeps_unaligned_chunk_starts_32_byte_aligned(monkeypatch):
    fake = _RecordingTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    # 4 * 1048577 bytes per row: 1023 rows would fit the budget but misalign later chunks.
    scores = torch.empty((3000, 1048577), dtype=torch.float32, device="meta")
    seq_lens = torch.empty((3000,), dtype=torch.int32, device="meta")

    dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=2048, return_topk_scores=False
    )

    assert [launch_rows for launch_rows, _ in fake.calls] == [1016, 1016, 968]
    assert all(offset_bytes % 32 == 0 for _, offset_bytes in fake.calls)
