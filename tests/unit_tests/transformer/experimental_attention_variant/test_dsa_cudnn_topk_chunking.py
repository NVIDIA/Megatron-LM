# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row chunking of the cuDNN indexer top-k wrapper (scratch budget, int32 limit, alignment),
and ``dsa_cudnn_topk_scratch_gib`` from the config to every top-k launch."""

import math
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes

_GIB = 1024**3
_FP32_BYTES = 4


def _scratch_bytes(rows: int, num_cols: int) -> int:
    # cuDNN radix top-k: int32 scratch of rows x 2 x num_cols for FP32 scores.
    return rows * 2 * num_cols * 4


def test_topk_scratch_budget_defaults_to_2gib():
    assert dsa_cudnn_kernels._TOPK_WRAPPER_MAX_SCRATCH_BYTES == 2 * _GIB
    # 256 rows per launch at 1M keys, the same with the budget left unset or passed explicitly.
    assert dsa_cudnn_kernels._topk_wrapper_chunk_rows(16384, 1048576, _FP32_BYTES) == 256
    assert (
        dsa_cudnn_kernels._topk_wrapper_chunk_rows(
            16384, 1048576, _FP32_BYTES, max_scratch_bytes=2 * _GIB
        )
        == 256
    )


# (case, rows, score columns, expected rows per launch at the 2 GiB default and at 8 GiB). The 1M
# rows are the CP64 rank 0/21/42/63 score-buffer widths of a 1M-token sequence; 128K is CP32
# rank 0. At 1M keys 8 GiB is exactly the frontend's 2**31-element limit.
_TOPK_SHAPES = [
    ("1m_cp64_rank0", 16384, 1048576, 256, 1024),
    ("1m_cp64_rank21", 16384, 876544, 306, 1224),
    ("1m_cp64_rank42", 16384, 704512, 381, 1524),
    ("1m_cp64_rank63", 16384, 532480, 504, 2016),
    ("128k_cp32_rank0", 4096, 131072, 2048, 4096),
    ("4k", 4096, 4096, 4096, 4096),
    ("4k_many_rows", 65536, 4096, 65536, 65536),
]


@pytest.mark.parametrize("budget_gib", [None, 8], ids=["default_2gib", "8gib"])
@pytest.mark.parametrize(
    "rows,num_cols,expected_default,expected_8gib",
    [case[1:] for case in _TOPK_SHAPES],
    ids=[c[0] for c in _TOPK_SHAPES],
)
def test_topk_chunk_rows_fill_the_budget_without_a_row_cap(
    rows, num_cols, expected_default, expected_8gib, budget_gib
):
    budget = None if budget_gib is None else budget_gib * _GIB
    expected = expected_default if budget_gib is None else expected_8gib
    budget_bytes = 2 * _GIB if budget is None else budget

    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(
        rows, num_cols, _FP32_BYTES, max_scratch_bytes=budget
    )

    assert chunk_rows == expected
    assert _scratch_bytes(chunk_rows, num_cols) <= budget_bytes
    assert chunk_rows * 2 * num_cols <= 2**31
    if chunk_rows < rows:
        # The budget, not a row cap, is what stops the chunk from growing.
        assert _scratch_bytes(chunk_rows + 1, num_cols) > budget_bytes


def test_topk_chunk_rows_keep_the_128k_call_in_one_launch_above_1024_rows_at_8gib():
    def chunk_rows(rows):
        return dsa_cudnn_kernels._topk_wrapper_chunk_rows(
            rows, 131072, _FP32_BYTES, max_scratch_bytes=8 * _GIB
        )

    assert chunk_rows(4096) == 4096
    assert chunk_rows(8192) == 8192
    assert chunk_rows(8193) == 8192


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
    # An 8 GiB budget, so that the budget allows more than 512 rows.
    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(
        16384, num_cols, _FP32_BYTES, max_scratch_bytes=8 * _GIB
    )
    assert chunk_rows == expected_rows
    assert chunk_rows % expected_alignment == 0
    assert (chunk_rows * num_cols * _FP32_BYTES) % 32 == 0


@pytest.mark.parametrize("element_size,num_cols,expected", [(2, 4097, 16), (2, 4096, 1), (4, 6, 4)])
def test_topk_wrapper_row_alignment_depends_on_row_bytes(element_size, num_cols, expected):
    assert dsa_cudnn_kernels._topk_wrapper_row_alignment(num_cols, element_size) == expected


def test_topk_chunk_rows_respect_the_frontend_int32_limit_above_the_byte_budget():
    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(
        16384, 1048576, _FP32_BYTES, max_scratch_bytes=64 * _GIB
    )

    assert chunk_rows == 1024
    assert chunk_rows * 2 * 1048576 <= 2**31


def test_topk_chunk_rows_follow_a_fractional_byte_budget():
    # 608 rows (one wave of 4 CTAs/SM on 152 SMs) is no longer floored to a 512-row multiple.
    assert (
        dsa_cudnn_kernels._topk_wrapper_chunk_rows(
            16384, 1048576, _FP32_BYTES, max_scratch_bytes=int(4.75 * _GIB)
        )
        == 608
    )


def test_topk_chunk_rows_take_at_least_one_row_below_one_rows_scratch():
    assert (
        dsa_cudnn_kernels._topk_wrapper_chunk_rows(
            16384, 1048576, _FP32_BYTES, max_scratch_bytes=1024
        )
        == 1
    )


@pytest.mark.parametrize("element_size,group_rows", [(_FP32_BYTES, 8), (2, 16)])
def test_topk_chunk_rows_take_one_aligned_row_group_below_its_scratch(element_size, group_rows):
    # 0.05 GiB fits 6 rows of 1048577 columns, fewer than one 32-byte-aligned row group: 6-row
    # chunks would start the second chunk 24 (FP32) or 12 (BF16) bytes past a 32-byte boundary.
    budget = int(0.05 * _GIB)
    chunk_rows = dsa_cudnn_kernels._topk_wrapper_chunk_rows(
        3000, 1048577, element_size, max_scratch_bytes=budget
    )

    assert chunk_rows == group_rows
    assert (chunk_rows * 1048577 * element_size) % 32 == 0
    # A call with fewer rows than one group still runs as one launch.
    assert (
        dsa_cudnn_kernels._topk_wrapper_chunk_rows(
            5, 1048577, element_size, max_scratch_bytes=budget
        )
        == 5
    )


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


def _run_recorded_topk(monkeypatch, rows, num_cols, budget_gib, return_topk_scores=False):
    fake = _RecordingTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    scores = torch.empty((rows, num_cols), dtype=torch.float32, device="meta")
    seq_lens = torch.empty((rows,), dtype=torch.int32, device="meta")

    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores,
        seq_lens,
        topk_k=2048,
        return_topk_scores=return_topk_scores,
        topk_scratch_bytes=None if budget_gib is None else budget_gib * _GIB,
    )
    return fake.calls, result


@pytest.mark.parametrize(
    "rows,num_cols,budget_gib,expected_launches",
    [
        (16384, 1048576, None, [256] * 64),
        (16384, 532480, None, [504] * 32 + [256]),
        (4096, 131072, None, [2048] * 2),
        (64, 1048577, None, [64]),
        (16384, 1048576, 8, [1024] * 16),
        (16384, 532480, 8, [2016] * 8 + [256]),
        (4096, 131072, 8, [4096]),
    ],
)
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_topk_wrapper_launches_budget_sized_chunks(
    monkeypatch, rows, num_cols, budget_gib, expected_launches, return_topk_scores
):
    calls, result = _run_recorded_topk(
        monkeypatch, rows, num_cols, budget_gib, return_topk_scores=return_topk_scores
    )

    assert [launch_rows for launch_rows, _ in calls] == expected_launches
    assert all(offset_bytes % 32 == 0 for _, offset_bytes in calls)
    assert result["indices"].shape == (rows, 2048)
    assert (result["values"] is not None) == return_topk_scores


@pytest.mark.parametrize(
    "budget_gib,expected_launches",
    [
        # 255 rows would fit the 2 GiB default but misalign later chunks.
        (None, [248] * 12 + [24]),
        # 1023 rows would fit 8 GiB but misalign later chunks.
        (8, [1016, 1016, 968]),
    ],
)
def test_topk_wrapper_keeps_unaligned_chunk_starts_32_byte_aligned(
    monkeypatch, budget_gib, expected_launches
):
    # 4 * 1048577 bytes per row.
    calls, _ = _run_recorded_topk(monkeypatch, 3000, 1048577, budget_gib)

    assert [launch_rows for launch_rows, _ in calls] == expected_launches
    assert all(offset_bytes % 32 == 0 for _, offset_bytes in calls)


def test_topk_wrapper_budget_argument_overrides_the_module_default(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 1 * _GIB)

    default_calls, _ = _run_recorded_topk(monkeypatch, 16384, 1048576, None)
    explicit_calls, _ = _run_recorded_topk(monkeypatch, 16384, 1048576, 8)

    assert [launch_rows for launch_rows, _ in default_calls] == [128] * 128
    assert [launch_rows for launch_rows, _ in explicit_calls] == [1024] * 16


# ------------------------------------------------------------------- dsa_cudnn_topk_scratch_gib


def _transformer_config(**overrides):
    return TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4, **overrides)


def test_topk_scratch_budget_config_defaults_to_2gib():
    config = _transformer_config()

    assert config.dsa_cudnn_topk_scratch_gib == 2.0
    options = dsa_cudnn_kernels._scorer_options_from_config(config)
    assert options.topk_scratch_bytes == 2 * _GIB
    assert options.topk_scratch_bytes == dsa_cudnn_kernels._TOPK_WRAPPER_MAX_SCRATCH_BYTES


def test_config_accepts_a_larger_topk_scratch_budget():
    config = _transformer_config(dsa_cudnn_topk_scratch_gib=8.0)

    assert dsa_cudnn_kernels._scorer_options_from_config(config).topk_scratch_bytes == 8 * _GIB


def test_topk_scratch_budget_has_a_cli_flag():
    from megatron.training.arguments import add_megatron_arguments

    parser = ArgumentParser()
    add_megatron_arguments(parser)

    assert parser.parse_args([]).dsa_cudnn_topk_scratch_gib == 2.0
    args = parser.parse_args(["--dsa-cudnn-topk-scratch-gib", "8"])
    assert args.dsa_cudnn_topk_scratch_gib == 8.0


@pytest.mark.parametrize("scratch_gib", [0, -1.0, math.nan, math.inf])
def test_config_rejects_a_topk_scratch_budget_that_is_not_positive(scratch_gib):
    with pytest.raises(ValueError, match="dsa_cudnn_topk_scratch_gib"):
        _transformer_config(dsa_cudnn_topk_scratch_gib=scratch_gib)


@pytest.mark.parametrize(
    "config,expected",
    [
        (SimpleNamespace(dsa_cudnn_topk_scratch_gib=8), 8 * _GIB),
        (SimpleNamespace(dsa_cudnn_topk_scratch_gib=4.75), int(4.75 * _GIB)),
        (SimpleNamespace(), None),
        (None, None),
    ],
    ids=["8gib", "fractional", "unset", "no_config"],
)
def test_scorer_options_read_the_topk_scratch_budget(config, expected):
    assert dsa_cudnn_kernels._scorer_options_from_config(config).topk_scratch_bytes == expected


_BUDGET = 3 * _GIB


@pytest.fixture
def recorded_budgets(monkeypatch):
    """A fake cuDNN frontend 1.31 and the budget that every top-k wrapper call receives."""
    fakes.Fe131DSA.reset()
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fakes.Fe131DSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    budgets = []
    wrapper = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked

    def record(*args, **kwargs):
        budgets.append(kwargs.get("topk_scratch_bytes"))
        return wrapper(*args, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_top_k_wrapper_chunked", record)
    return budgets


def _indexer_inputs(sq, sk, heads=4, dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((1, sq, heads, dim), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sk, dim), generator=generator).to(torch.bfloat16)
    w = torch.rand((1, sq, heads), generator=generator).to(torch.bfloat16)
    return q, k, w


def _packed_kwargs(seq_lens):
    cu, starts, ends = [0], [], []
    for length in seq_lens:
        starts += [cu[-1]] * length
        ends += [cu[-1] + position + 1 for position in range(length)]
        cu.append(cu[-1] + length)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    return dict(
        varlen_starts=torch.tensor(starts),
        varlen_ends=torch.tensor(ends),
        use_local_indexer_varlen=True,
        packed_cu_seqlens_q=cu_seqlens,
        packed_cu_seqlens_k=cu_seqlens,
        packed_max_seqlen_q=max(seq_lens),
        packed_max_seqlen_k=max(seq_lens),
        packed_cp_size=1,
        local_packed_cp_rank=0,
    )


@pytest.mark.parametrize("return_scores", [False, True], ids=["topk_only", "dense_scores"])
@pytest.mark.parametrize("layout", ["plain_causal", "bounds", "packed_thd"])
def test_every_bshd_executor_launches_with_the_configured_budget(
    recorded_budgets, layout, return_scores
):
    if layout == "packed_thd" and return_scores:
        pytest.skip("the packed THD executor returns no dense scores")
    sq = 12
    q, k, w = _indexer_inputs(sq, sq, seed=7)
    kwargs = {}
    if layout == "bounds":
        kwargs = dict(
            varlen_starts=torch.zeros(sq, dtype=torch.int64),
            varlen_ends=torch.arange(1, sq + 1, dtype=torch.int64),
        )
    elif layout == "packed_thd":
        kwargs = _packed_kwargs((7, 5))
    options = dsa_cudnn_kernels.IndexerScorerOptions(topk_scratch_bytes=_BUDGET)

    dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        4,
        return_scores=return_scores,
        return_topk_scores=not return_scores,
        scorer_options=options,
        **kwargs,
    )

    assert recorded_budgets and all(budget == _BUDGET for budget in recorded_budgets)


def test_single_packed_cp_segments_launch_with_the_given_budget(recorded_budgets):
    cp_size, local_sq = 2, 8
    q, k, w = _indexer_inputs(local_sq, local_sq * cp_size, seed=3)

    dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k.unsqueeze(2), w, 4, True, cp_size, 1, topk_scratch_bytes=_BUDGET
    )

    assert len(recorded_budgets) >= 2 and all(budget == _BUDGET for budget in recorded_budgets)


def test_deepgemm_logits_top_k_launches_with_the_given_budget(recorded_budgets):
    logits = torch.randn((16, 64)).to(torch.bfloat16)
    lengths = torch.full((16,), 64, dtype=torch.int32)

    indices, values = dsa_cudnn_kernels._deepgemm_logits_topk(logits, lengths, 8, "cudnn", _BUDGET)

    assert recorded_budgets == [_BUDGET]
    assert indices.shape == (16, 8) and values.dtype == torch.bfloat16
