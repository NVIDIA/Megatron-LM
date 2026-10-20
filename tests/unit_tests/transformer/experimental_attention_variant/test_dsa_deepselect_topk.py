# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_topk_backend='deepselect'`` on CPU with fake DeepSelect builds.

``FakeDsTune`` (the GLM-5.2 ``ds_tune`` build: ``topk(x, k, end, cfg_id, out_val=, out_idx=)``
and ``config_for``) and ``FakeDeepSelect`` (upstream ``deep_select``: ``topk(x, k, end=,
output_idx=, return_value=)``) share one CPU model of the DeepSelect contract that MCore relies
on: input row stride a multiple of the input alignment and unit column stride (else it raises
like the real kernel), ``end`` per-row lengths, rows no longer than K gathered at the front and
padded with the OOB fill values, rows longer than K holding NaN flagged by ``0x3F3F3F3F`` in
slot 0 (rest unspecified) when ``abort_when_nan_found=False``, a random choice among equal
scores at the cutoff and an unspecified slot order (``sorted_index=False``).
"""

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_deepselect_topk,
)
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes
from tests.unit_tests.transformer.experimental_attention_variant.test_dsa_cudnn_topk_deterministic import (  # noqa: E501
    RadixTopK,
)

# Bound before tests patch the Tensor methods that synchronize with the host.
_TOLIST = torch.Tensor.tolist
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SENTINEL = dsa_deepselect_topk.NAN_ROW_SENTINEL
_DEFAULT_WIDTHS = dict(dsa_deepselect_topk.MIN_SCORE_WIDTH)
_DEVICE_SUPPORTED = dsa_deepselect_topk._device_supported
_DEFAULT_MANY_ROWS = dict(dsa_deepselect_topk.MANY_ROWS_MIN_SCORE_WIDTH)


class _FakeSelect:
    """Shared CPU model of a DeepSelect launch; records every call."""

    __file__ = "fake_deepselect.py"

    def __init__(self, input_align=1024, output_align=32, seed=0):
        self.input_align = input_align
        self.output_align = output_align
        self.generator = torch.Generator().manual_seed(seed)
        self.calls = []

    def _select(self, x, topk, end, out_idx, out_val, idx_fill, value_fill, abort, **record):
        n_rows, width = x.shape
        if x.stride(1) != 1 or (x.stride(0) * x.element_size()) % self.input_align:
            raise RuntimeError("input.stride(0) must be a multiple of the input alignment")
        assert end is not None and end.dtype == torch.int32 and end.is_contiguous()
        assert end.shape == (n_rows,)
        for out in (out_idx, out_val):
            if out is not None:
                assert out.shape == (n_rows, topk) and out.stride(1) == 1
                assert (out.stride(0) * out.element_size()) % self.output_align == 0
        self.calls.append(
            dict(
                rows=n_rows,
                width=width,
                row_stride=x.stride(0),
                data_ptr=x.data_ptr(),
                topk=topk,
                idx_fill=idx_fill,
                value_fill=value_fill,
                abort=abort,
                output_idx=out_idx,
                **record,
            )
        )
        for row, length in enumerate(_TOLIST(end)):
            assert 0 <= length <= width, "end must be <= vocab_size"
            scores = x[row, :length]
            if length <= topk:
                chosen = torch.arange(length)
            elif _TOLIST(torch.isnan(scores).any()):
                if abort:
                    raise RuntimeError("trap: NaN found")
                out_idx[row] = torch.randperm(length, generator=self.generator)[:topk].int()
                out_idx[row, 0] = _SENTINEL
                if out_val is not None:
                    out_val[row] = 12345.0
                continue
            else:
                # Random choice among equal scores, then an arbitrary slot order.
                order = torch.randperm(length, generator=self.generator)
                ranked = torch.sort(scores[order] + 0.0, descending=True, stable=True).indices
                chosen = order[ranked[:topk]]
                chosen = chosen[torch.randperm(topk, generator=self.generator)]
            count = chosen.numel()
            out_idx[row, :count] = chosen.to(out_idx.dtype)
            out_idx[row, count:] = idx_fill
            if out_val is not None:
                out_val[row, :count] = x[row, chosen]
                out_val[row, count:] = value_fill


class FakeDsTune(_FakeSelect):
    """CPU model of the ``ds_tune`` package (configs: FP32 max_topk 2048 and 4096)."""

    __version__ = "0.1.0+fake"

    def __init__(self, input_align=1024, output_align=32, seed=0, max_topks=(2048, 4096)):
        super().__init__(input_align, output_align, seed)
        self.INPUT_STRIDE_ALIGNMENT_BYTES = input_align
        self.OUTPUT_STRIDE_ALIGNMENT_BYTES = output_align
        self.CONFIGS = [
            {"id": cfg_id, "dtype": "fp32", "max_topk": max_topk}
            for cfg_id, max_topk in enumerate(max_topks)
        ]

    def config_for(self, dtype, k):
        if dtype != torch.float32:
            raise ValueError("fp32 only in this fake")
        fits = [c for c in self.CONFIGS if 0 < k <= c["max_topk"]]
        if not fits:
            raise ValueError(f"no config for k={k}")
        return min(fits, key=lambda c: c["max_topk"])

    def topk(
        self,
        x,
        k,
        end,
        cfg_id,
        sorted_index=False,
        out_val=None,
        out_idx=None,
        idx_oob_fill_value=-1,
        value_oob_fill_value=float("-inf"),
        abort_when_nan_found=False,
    ):
        assert k <= self.CONFIGS[cfg_id]["max_topk"]
        assert out_val is not None and out_idx is not None and out_idx.dtype == torch.int32
        self._select(
            x,
            k,
            end,
            out_idx,
            out_val,
            idx_oob_fill_value,
            value_oob_fill_value,
            abort_when_nan_found,
            sorted_index=sorted_index,
            return_value=True,
            cfg_id=cfg_id,
        )
        return out_val, out_idx


class FakeDeepSelect(_FakeSelect):
    """CPU model of upstream ``deep_select`` (``topk`` and ``get_stride_requirement``)."""

    __version__ = "1.0.0+fake"

    def get_stride_requirement(self):
        return (self.input_align, self.output_align)

    def topk(
        self,
        input,
        topk,
        sorted=False,
        begin=None,
        end=None,
        indices_type=torch.int64,
        sorted_index=False,
        hint=None,
        output_idx=None,
        output_idx_offset=None,
        idx_oob_fill_value=2147483647,
        value_oob_fill_value=float("-inf"),
        return_value=True,
        abort_when_nan_found=True,
    ):
        assert not sorted and begin is None and output_idx_offset is None
        n_rows = input.size(0)
        if output_idx is None:
            output_idx = torch.empty((n_rows, topk), dtype=indices_type)
        assert output_idx.dtype == indices_type
        values = torch.empty((n_rows, -(-topk // 8) * 8), dtype=input.dtype)[:, :topk]
        self._select(
            input,
            topk,
            end,
            output_idx,
            values,
            idx_oob_fill_value,
            value_oob_fill_value,
            abort_when_nan_found,
            sorted_index=sorted_index,
            return_value=return_value,
        )
        return (values if return_value else None), output_idx


_FAKES = {"ds_tune": FakeDsTune, "deep_select": FakeDeepSelect}


def _widths(width):
    return {kind: width for kind in _FAKES}


@pytest.fixture(params=sorted(_FAKES))
def provider_kind(request):
    return request.param


@pytest.fixture
def deepselect(monkeypatch):
    """Install a fake DeepSelect build and treat the CPU as a supported device."""

    def install(kind="ds_tune", supported=True, **kwargs):
        fake = _FAKES[kind](**kwargs)
        provider = dsa_deepselect_topk._Provider(kind, fake)
        monkeypatch.setattr(
            dsa_deepselect_topk, "_IMPORT_STATE", {"provider": provider, "error": None}
        )
        monkeypatch.setattr(dsa_deepselect_topk, "_device_supported", lambda device: supported)
        monkeypatch.setattr(dsa_deepselect_topk, "_LOGGED_REASONS", set())
        monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _widths(64))
        monkeypatch.setattr(dsa_deepselect_topk, "MANY_ROWS_MIN_SCORE_WIDTH", {})
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
        return fake

    return install


def _no_build(monkeypatch, error="no DeepSelect build: cannot import ds_tune: x"):
    monkeypatch.setattr(dsa_deepselect_topk, "_IMPORT_STATE", {"provider": None, "error": error})


@pytest.fixture
def cudnn_topk(monkeypatch):
    fake = RadixTopK(seed=3)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    return fake


def _rows(seed, n_rows=24, width=256, kind="gauss"):
    generator = torch.Generator().manual_seed(seed)
    if kind == "gauss":
        scores = torch.randn((n_rows, width), generator=generator)
    elif kind == "ties":
        scores = torch.round(torch.randn((n_rows, width), generator=generator) * 3) / 8
    elif kind == "zeros_inf":
        choices = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), 0.5])
        weights = torch.tensor([5.0, 5.0, 2.0, 2.0, 0.3, 1.0, 2.0])
        picks = torch.multinomial(weights, n_rows * width, True, generator=generator)
        scores = choices[picks].view(n_rows, width)
    else:
        raise AssertionError(kind)
    seq_lens = torch.randint(0, width + 1, (n_rows,), generator=generator, dtype=torch.int32)
    seq_lens[:6] = torch.tensor([width, 0, 1, 15, 16, 17], dtype=torch.int32)
    columns = torch.arange(width).view(1, width)
    # Past a row's length the buffer is unspecified with clean_logits=False: poison it.
    scores = torch.where(columns < seq_lens.view(-1, 1), scores, torch.full_like(scores, 7.0e30))
    return scores.contiguous(), seq_lens


def _call(scores, seq_lens, topk_k, return_topk_scores, backend="deepselect", **kwargs):
    return dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k, return_topk_scores, topk_backend=backend, **kwargs
    )


def _bits(values):
    return values.contiguous().view(torch.int32)


# --------------------------------------------------------------------------------- eligibility
def _static(**overrides):
    args = dict(
        n_rows=16,
        width=65536,
        topk_k=2048,
        dtype=torch.float32,
        device=torch.device("cpu"),
        deterministic=False,
    )
    args.update(overrides)
    return dsa_deepselect_topk.static_fallback_reason(**args)


def test_static_eligibility_covers_every_requirement(deepselect, monkeypatch, provider_kind):
    deepselect(provider_kind)
    monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _widths(65536))
    assert _static() is None
    assert _static(width=2**23 - 1) is None
    assert _static(topk_k=4096) is None
    assert "deterministic" in _static(deterministic=True)
    assert "top-k must be in" in _static(topk_k=4097)
    assert "top-k must be in" in _static(topk_k=0)
    assert "below 65536" in _static(width=65535)
    assert "below 2**23" in _static(width=2**23)
    assert "no rows" in _static(n_rows=0)
    assert "FP32" in _static(dtype=torch.bfloat16)
    assert "multiple of 8" in _static(topk_k=2044)
    monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _widths(64))
    assert "whole row" in _static(width=1024, topk_k=1024)


@pytest.mark.parametrize(
    "kind,width,eligible",
    [
        ("ds_tune", 1048576, True),
        ("ds_tune", 532480, True),
        ("ds_tune", 67584, True),
        ("ds_tune", 66560, True),
        ("ds_tune", 65535, False),
        ("deep_select", 1048576, True),
        ("deep_select", 532480, True),
        ("deep_select", 131072, False),
    ],
)
def test_each_build_has_its_measured_minimum_width(deepselect, monkeypatch, kind, width, eligible):
    """GLM-5.2 rank widths: 1M CP64 532480..1048576, 128K realized CP32 67584..131072 (66560
    at CP64)."""
    deepselect(kind)
    monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _DEFAULT_WIDTHS)
    reason = _static(width=width)
    assert (reason is None) is eligible, reason
    if not eligible:
        assert reason.startswith("score width below")


@pytest.mark.parametrize(
    "kind,n_rows,width,eligible",
    [
        ("ds_tune", 2048, 65536, True),
        ("ds_tune", 2048, 65535, False),
        ("ds_tune", 4096, 65536, True),
        ("ds_tune", 4096, 67584, True),
        ("ds_tune", 4096, 79872, True),
        ("ds_tune", 4097, 81919, False),
        ("ds_tune", 4097, 81920, True),
        ("ds_tune", 16384, 65536, False),
        ("ds_tune", 16384, 81919, False),
        ("ds_tune", 16384, 81920, True),
        ("ds_tune", 16384, 532480, True),
        ("deep_select", 16384, 532480, True),
        ("deep_select", 16384, 131072, False),
        ("deep_select", 4096, 131072, False),
    ],
)
def test_minimum_width_grows_with_the_row_count(
    deepselect, monkeypatch, kind, n_rows, width, eligible
):
    """GB300 sweep (job 4646028): ds_tune is 0.94-1.01x cuDNN at 65536 columns for 2048 and
    4096 rows (4096 rows of 67584..131072: the GLM-5.2 128K call at its realized CP32), but
    1.08-1.09x at 16384 rows, which it beats only from 81920 columns."""
    deepselect(kind)
    monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _DEFAULT_WIDTHS)
    monkeypatch.setattr(dsa_deepselect_topk, "MANY_ROWS_MIN_SCORE_WIDTH", _DEFAULT_MANY_ROWS)
    reason = _static(n_rows=n_rows, width=width)
    assert (reason is None) is eligible, reason
    assert (dsa_deepselect_topk.min_score_width(kind, n_rows) <= width) is eligible
    if not eligible:
        assert reason.startswith("score width below")
    if kind == "ds_tune" and n_rows > 4096 and width >= 65536 and not eligible:
        assert reason == (
            "score width below 81920 for ds_tune with more than 4096 rows (cuDNN is faster there)"
        )


@pytest.mark.parametrize("cp_rank", range(32))
def test_every_glm_128k_cp32_rank_uses_ds_tune(deepselect, monkeypatch, cp_rank):
    """The 128K recipe realizes CP32 with 4096 query rows per call; rank widths follow
    ``dsa_cudnn_kernels._indexer_topk_packed_thd``: max((r + 1) * half, 131072 - r * half)."""
    deepselect("ds_tune")
    monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _DEFAULT_WIDTHS)
    monkeypatch.setattr(dsa_deepselect_topk, "MANY_ROWS_MIN_SCORE_WIDTH", _DEFAULT_MANY_ROWS)
    half = 131072 // 64
    width = max((cp_rank + 1) * half, 131072 - cp_rank * half)
    assert 67584 <= width <= 131072
    assert _static(n_rows=4096, width=width) is None


def test_ds_tune_needs_a_compiled_config_for_k(deepselect):
    deepselect("ds_tune", max_topks=(2048,))
    assert _static(topk_k=2048) is None
    assert "no compiled config for top-k 4096" in _static(topk_k=4096)


@pytest.mark.parametrize(
    "sm,expected",
    [((10, 3), None), ((10, 0), "DeepSelect is not validated on SM 10.0"), ((9, 0), "needs")],
)
def test_only_validated_devices_run_deepselect(deepselect, monkeypatch, caplog, sm, expected):
    """SM 10.0 kernels are compiled but have never run, so GB200 keeps cuDNN with a warning."""
    deepselect()
    monkeypatch.setattr(dsa_deepselect_topk, "_device_supported", _DEVICE_SUPPORTED)
    monkeypatch.setattr(dsa_deepselect_topk, "_device_capability", lambda index: sm)
    assert (10, 0) in dsa_deepselect_topk.COMPILED_SM
    assert dsa_deepselect_topk.SUPPORTED_SM == frozenset({(10, 3)})
    reason = _static(device=torch.device("cuda", 0))
    if expected is None:
        assert reason is None
        return
    assert expected in reason
    with caplog.at_level(logging.INFO, logger=dsa_deepselect_topk.__name__):
        dsa_deepselect_topk.log_topk_backend_once("deepselect", reason)
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert _static(device=torch.device("cpu")).startswith("DeepSelect needs")


def test_unsupported_device_and_missing_build_fall_back(deepselect, monkeypatch):
    deepselect(supported=False)
    assert "SM" in _static(width=1024, topk_k=256)
    monkeypatch.setattr(dsa_deepselect_topk, "_device_supported", lambda device: True)
    _no_build(monkeypatch, "no DeepSelect build: nope")
    assert _static(width=1024, topk_k=256) == "no DeepSelect build: nope"


def test_import_probe_prefers_ds_tune_and_requires_the_named_arguments(monkeypatch):
    class OldDeepSelect:
        @staticmethod
        def get_stride_requirement():
            return (1024, 32)

        @staticmethod
        def topk(input, topk, sorted=False, end=None, sorted_index=False):
            raise AssertionError("not called")

    def importer(modules):
        def import_module(name):
            if modules.get(name) is None:
                raise ImportError(f"No module named '{name}'")
            return modules[name]

        return import_module

    def load(modules):
        monkeypatch.setattr(dsa_deepselect_topk, "_IMPORT_STATE", {})
        monkeypatch.setattr(dsa_deepselect_topk.importlib, "import_module", importer(modules))
        return dsa_deepselect_topk._load_provider()

    provider = load({"ds_tune": FakeDsTune(), "deep_select": FakeDeepSelect()})
    assert provider.kind == "ds_tune" and provider.input_align == 1024
    provider = load({"deep_select": FakeDeepSelect(input_align=512)})
    assert provider.kind == "deep_select" and provider.input_align == 512
    assert load({"deep_select": OldDeepSelect()}) is None
    error = dsa_deepselect_topk._IMPORT_STATE["error"]
    assert "No module named 'ds_tune'" in error
    assert "abort_when_nan_found" in error and "output_idx" in error
    assert load({}) is None
    assert dsa_deepselect_topk._IMPORT_STATE["error"].startswith("no DeepSelect build")


def test_call_eligibility_checks_strides_and_lengths(deepselect):
    deepselect()
    buffer = torch.zeros((4, 512))
    lens = torch.full((4,), 100, dtype=torch.int32)
    assert dsa_deepselect_topk.fallback_reason(buffer[:, :300], lens, 16, False) is None
    assert "stride" in dsa_deepselect_topk.fallback_reason(buffer[:, ::2], lens, 16, False)
    assert "stride" in dsa_deepselect_topk.fallback_reason(torch.zeros((4, 300)), lens, 16, False)
    assert "one per row" in dsa_deepselect_topk.fallback_reason(buffer, lens[:3], 16, False)
    assert "2-D" in dsa_deepselect_topk.fallback_reason(buffer.view(2, 2, 512), lens, 16, False)


def test_call_eligibility_checks_the_tma_start_address(deepselect, cudnn_topk):
    """A score view starting off a 16-byte boundary would fail the TMA descriptor: keep cuDNN."""
    fake = deepselect()
    flat = torch.zeros(4 * 512 + 4)
    lens = torch.full((4,), 100, dtype=torch.int32)
    for offset, eligible in ((0, True), (4, True), (1, False), (2, False)):
        view = flat[offset : offset + 4 * 512].view(4, 512)
        reason = dsa_deepselect_topk.fallback_reason(view, lens, 16, False)
        assert (reason is None) is eligible, (offset, reason)
        if not eligible:
            assert reason == "score buffer must start at a 16-byte boundary"
    calls = len(fake.calls)
    misaligned = flat[1 : 1 + 4 * 512].view(4, 512)
    result = _call(misaligned, lens, 16, True)
    assert len(fake.calls) == calls and cudnn_topk.calls, "the misaligned call must run cuDNN"
    assert result["indices"].shape == (4, 16)


@pytest.mark.parametrize("width", [1048576, 532480, 131072, 79872, 67584, 66560, 65536])
def test_glm_rank_widths_need_no_padded_score_buffer(deepselect, width):
    """1M CP64 rank widths are multiples of 8192 elements, 128K ones of 2048 at the realized
    CP32 (and of 1024 at CP64)."""
    deepselect()
    assert dsa_deepselect_topk.score_row_stride(width) == width


def test_score_row_stride_rounds_up_to_the_input_alignment(deepselect):
    deepselect()
    assert dsa_deepselect_topk.score_row_stride(100004) == 100096
    assert dsa_deepselect_topk.score_row_stride(256) == 256
    assert dsa_deepselect_topk.score_row_stride(257) == 512


# ------------------------------------------------------------------------- routing and contract
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_eligible_calls_run_one_deepselect_launch(
    deepselect, cudnn_topk, provider_kind, return_topk_scores
):
    fake = deepselect(provider_kind)
    scores, seq_lens = _rows(1, n_rows=40, width=256)

    result = _call(scores, seq_lens, 16, return_topk_scores)

    assert cudnn_topk.calls == []
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["rows"] == 40 and call["width"] == 256 and call["topk"] == 16
    assert call["data_ptr"] == scores.data_ptr()
    assert call["sorted_index"] is False and call["abort"] is False
    assert call["idx_fill"] == -1 and call["value_fill"] == float("-inf")
    # ds_tune always writes values (its configs are compiled with return_value=True).
    assert call["return_value"] is (return_topk_scores or provider_kind == "ds_tune")
    assert call["output_idx"] is result["indices"]
    if provider_kind == "ds_tune":
        assert call["cfg_id"] == 0
    assert result["indices"].dtype == torch.int32 and result["indices"].is_contiguous()
    assert (result["values"] is not None) is return_topk_scores


def test_scratch_budget_never_chunks_deepselect(deepselect, cudnn_topk, monkeypatch):
    fake = deepselect()
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 7 * 256 * 8)
    scores, seq_lens = _rows(2, n_rows=40, width=256)

    _call(scores, seq_lens, 16, True)

    assert [call["rows"] for call in fake.calls] == [40] and cudnn_topk.calls == []


@pytest.mark.parametrize(
    "case",
    ["deterministic_arg", "torch_deterministic", "narrow", "unaligned", "cudnn_backend", "no_pkg"],
)
def test_ineligible_calls_keep_the_cudnn_top_k(deepselect, cudnn_topk, monkeypatch, case):
    fake = deepselect()
    scores, seq_lens = _rows(3, n_rows=8, width=256)
    kwargs, backend = {}, "deepselect"
    if case == "deterministic_arg":
        kwargs["deterministic"] = True
    elif case == "torch_deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    elif case == "narrow":
        monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", _widths(257))
    elif case == "unaligned":
        scores, seq_lens = _rows(3, n_rows=8, width=200)
    elif case == "cudnn_backend":
        backend = "cudnn"
    else:
        _no_build(monkeypatch)

    result = _call(scores, seq_lens, 16, True, backend=backend, **kwargs)

    assert fake.calls == [] and len(cudnn_topk.calls) == 1
    assert result["indices"].shape == (8, 16)


def test_unknown_backend_is_rejected(cudnn_topk):
    scores, seq_lens = _rows(4, n_rows=8, width=64)
    with pytest.raises(ValueError, match="top-k backend"):
        _call(scores, seq_lens, 8, True, backend="flashinfer")


@pytest.mark.parametrize("kind", ["gauss", "ties", "zeros_inf"])
@pytest.mark.parametrize("seed", range(3))
def test_contract_matches_the_cudnn_path(deepselect, cudnn_topk, provider_kind, kind, seed):
    """Same slots, padding and payload; only the members among equal cutoff scores differ."""
    deepselect(provider_kind, seed=seed + 50)
    topk_k = 16
    scores, seq_lens = _rows(seed, n_rows=48, width=256, kind=kind)

    got = _call(scores, seq_lens, topk_k, True)
    want = _call(scores, seq_lens, topk_k, True, backend="cudnn")

    lengths = seq_lens.long()
    short = lengths <= topk_k
    # Rows no longer than K: identical layout, indices, value bits and padding bits.
    assert torch.equal(got["indices"][short], want["indices"][short])
    assert torch.equal(_bits(got["values"])[short], _bits(want["values"])[short])
    # Longer rows: K distinct in-row members, payload = scores at the indices, and the same
    # multiset of selected values (ties at the cutoff compare equal, -0.0 == +0.0).
    long_idx = got["indices"][~short].long()
    assert bool((long_idx >= 0).all()) and bool((long_idx < lengths[~short].view(-1, 1)).all())
    assert all(len(set(_TOLIST(row))) == topk_k for row in long_idx)
    gathered = torch.gather(scores[~short], 1, long_idx)
    assert torch.equal(_bits(gathered), _bits(got["values"][~short]))
    assert torch.equal(
        torch.sort(got["values"][~short], dim=1).values,
        torch.sort(want["values"][~short], dim=1).values,
    )
    if kind == "gauss":
        # No ties: the member sets are identical.
        assert torch.equal(
            torch.sort(got["indices"], dim=1).values, torch.sort(want["indices"], dim=1).values
        )


def test_indices_only_call_matches_the_value_call(deepselect, cudnn_topk, provider_kind):
    scores, seq_lens = _rows(7, n_rows=32, width=256, kind="ties")
    deepselect(provider_kind, seed=9)
    with_values = _call(scores, seq_lens, 16, True)
    deepselect(provider_kind, seed=9)

    indices_only = _call(scores, seq_lens, 16, False)

    assert indices_only["values"] is None
    assert torch.equal(indices_only["indices"], with_values["indices"])


# ------------------------------------------------------------------------------------- NaN rows
def test_nan_rows_become_the_first_in_row_columns_with_nan_scores(
    deepselect, cudnn_topk, provider_kind
):
    deepselect(provider_kind)
    width, topk_k = 256, 16
    scores, seq_lens = _rows(11, n_rows=8, width=width)
    seq_lens[:] = torch.tensor([width, 200, 12, 100, width, 64, 17, 0], dtype=torch.int32)
    scores[0, 3] = float("nan")  # long row: flagged and repaired
    scores[1, 150] = float("nan")
    scores[2, 5] = float("nan")  # short row: never NaN-checked, gathered as is
    scores[3, 100] = float("nan")  # past the end: not part of the row
    scores[6, 16] = float("nan")  # K + 1 long

    result = _call(scores, seq_lens, topk_k, True)

    for row in (0, 1, 6):
        assert _TOLIST(result["indices"][row]) == list(range(topk_k))
        assert bool(torch.isnan(result["values"][row]).all())
    assert _TOLIST(result["indices"][2]) == list(range(12)) + [-1] * 4
    assert bool(torch.isnan(result["values"][2, 5])) and bool(
        torch.isinf(result["values"][2, 12:]).all()
    )
    assert not bool(torch.isnan(result["values"][3]).any())
    assert sorted(_TOLIST(result["indices"][3]))[0] >= 0
    assert _TOLIST(result["indices"][7]) == [-1] * topk_k


def test_nan_repair_torch_fallback_rewrites_only_flagged_rows():
    indices = torch.arange(24, dtype=torch.int32).view(3, 8).clone()
    indices[0, 0] = _SENTINEL
    indices[2, 0] = _SENTINEL
    values = torch.ones((3, 8))
    lengths = torch.tensor([100, 100, 5], dtype=torch.int32)

    dsa_deepselect_topk.repair_nan_rows(indices, values, lengths)

    assert _TOLIST(indices) == [list(range(8)), list(range(8, 16)), [0, 1, 2, 3, 4, -1, -1, -1]]
    assert bool(torch.isnan(values[0]).all()) and bool(torch.isnan(values[2]).all())
    assert bool((values[1] == 1).all())
    dsa_deepselect_topk.repair_nan_rows(indices, None, lengths)


def _nan_repair_case(seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 1000, (64, 24), generator=generator, dtype=torch.int32)
    flagged = torch.rand((64,), generator=generator) < 0.3
    indices[flagged, 0] = _SENTINEL
    values = torch.randn((64, 32), generator=generator)[:, :24]
    lengths = torch.randint(0, 50, (64,), generator=generator, dtype=torch.int32)
    return indices, values, lengths


@pytest.mark.skipif(not dsa_deepselect_topk.HAVE_TRITON, reason="Triton is not installed")
def test_triton_nan_repair_matches_torch_in_the_interpreter(tmp_path):
    """Run the Triton repair kernel through ``TRITON_INTERPRET=1`` on CPU tensors."""
    script = tmp_path / "nan_repair_interpreter.py"
    script.write_text(textwrap.dedent("""
            try:
                import triton.runtime.interpreter  # noqa: F401
            except ImportError:
                print("TRITON_INTERPRETER_UNAVAILABLE")
                raise SystemExit(0)
            import torch
            from megatron.core.transformer.experimental_attention_variant import (
                dsa_deepselect_topk as ds,
            )
            from tests.unit_tests.transformer.experimental_attention_variant import (
                test_dsa_deepselect_topk as cases,
            )

            for seed in range(3):
                for with_values in (True, False):
                    indices, values, lengths = cases._nan_repair_case(seed)
                    want_i, want_v = indices.clone(), values.clone()
                    ds._repair_nan_rows_torch(want_i, want_v if with_values else None, lengths)
                    ds._repair_nan_rows_triton(indices, values if with_values else None, lengths)
                    assert torch.equal(indices, want_i), seed
                    if with_values:
                        assert torch.equal(values.isnan(), want_v.isnan()), seed
                        assert torch.equal(values.nan_to_num(), want_v.nan_to_num()), seed
            print("TRITON_INTERPRETER_OK")
            """))
    python_path = os.pathsep.join(filter(None, (str(_REPO_ROOT), os.environ.get("PYTHONPATH"))))
    environment = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=python_path)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "TRITON_INTERPRETER_UNAVAILABLE" in completed.stdout:
        pytest.skip("this Triton build has no interpreter")
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "TRITON_INTERPRETER_OK" in completed.stdout


def test_deepselect_path_does_not_synchronize_with_the_host(deepselect, monkeypatch):
    deepselect()
    scores, seq_lens = _rows(13, n_rows=16, width=256)
    scores[0, 0] = float("nan")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("host synchronization in the DeepSelect top-k path")

    for name in ("item", "tolist", "__bool__", "__int__", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch, "nonzero", forbidden)

    result = _call(scores, seq_lens, 16, True)

    assert result["indices"].shape == (16, 16)


# -------------------------------------------------------------------------------------- logging
def test_backend_choice_is_logged_once_per_reason(deepselect, cudnn_topk, caplog, monkeypatch):
    deepselect()
    scores, seq_lens = _rows(14, n_rows=8, width=256)
    narrow, narrow_lens = _rows(14, n_rows=8, width=32)
    with caplog.at_level(logging.INFO, logger=dsa_deepselect_topk.__name__):
        for _ in range(3):
            _call(scores, seq_lens, 16, True)
            _call(narrow, narrow_lens, 8, True)
            _call(scores, seq_lens, 16, True, deterministic=True)
    messages = [r.getMessage() for r in caplog.records]
    assert sum("uses DeepSelect" in m for m in messages) == 1
    assert sum("below 64" in m for m in messages) == 1
    assert sum("deterministic mode" in m for m in messages) == 1

    caplog.clear()
    _no_build(monkeypatch, "no DeepSelect build: cannot import y")
    with caplog.at_level(logging.INFO, logger=dsa_deepselect_topk.__name__):
        _call(scores, seq_lens, 16, True)
        _call(scores, seq_lens, 16, True)
    assert [(r.levelno, "cannot import y" in r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, True)
    ]


# ---------------------------------------------------------------------- packed THD score buffer
class OutBindingFe131(fakes.Fe131CleanLogitsDSA):
    """FE 1.31 fake whose dense scorer writes into a caller ``out`` like the real one."""

    calls: list = []
    outs: list = []

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.outs = []

    @classmethod
    def indexer_forward_wrapper(
        cls,
        q,
        k,
        w,
        ratio=4,
        qhead_per_kv_head=None,
        m_block_size=128,
        n_block_size=128,
        q_stage=2,
        kv_stage=4,
        sm_scale=1.0,
        out=None,
        stream=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        precision="bf16",
        q_scale=None,
        k_scale=None,
        cu_seqlens_q_scale_padded=None,
        cu_seqlens_k_scale_padded=None,
        sf_vec_size=32,
        return_lse=False,
        lse_out=None,
        clean_logits: bool = True,
    ):
        scores = super().indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=ratio,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            clean_logits=clean_logits,
        )["scores"]
        cls.outs.append(out)
        if out is None:
            return {"scores": scores}
        assert out.shape == scores.shape and out.stride(1) == 1
        assert out.shape[1] % 4 == 0, "FE 1.31 would stage this out through a copy"
        out.copy_(scores)
        return {"scores": out}

    indexer_top_k_wrapper = staticmethod(fakes.indexer_top_k_wrapper)


def _packed_cp1(seq_lens, seed):
    cu, starts, ends = [0], [], []
    for length in seq_lens:
        for position in range(length):
            starts.append(cu[-1])
            ends.append(cu[-1] + position + 1)
        cu.append(cu[-1] + length)
    sq = cu[-1]
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((1, sq, 4, 8), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sq, 8), generator=generator).to(torch.bfloat16)
    w = torch.rand((1, sq, 4), generator=generator).to(torch.bfloat16)
    cu_t = torch.tensor(cu, dtype=torch.int32)
    return q, k, w, torch.tensor(starts), torch.tensor(ends), cu_t, max(seq_lens)


def _packed_topk(inputs, topk, options, return_topk_scores=True):
    q, k, w, starts, ends, cu, max_seqlen = inputs
    return dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        topk,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        return_topk_scores=return_topk_scores,
        use_local_indexer_varlen=True,
        packed_cu_seqlens_q=cu,
        packed_cu_seqlens_k=cu,
        packed_max_seqlen_q=max_seqlen,
        packed_max_seqlen_k=max_seqlen,
        packed_cp_size=1,
        local_packed_cp_rank=0,
        scorer_options=options,
    )


@pytest.fixture
def out_frontend(monkeypatch):
    def install(version=(1, 31)):
        OutBindingFe131.reset()
        monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", OutBindingFe131)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
        return OutBindingFe131

    return install


@pytest.mark.parametrize("skip_fill", [True, False])
def test_packed_thd_scores_into_a_padded_stride_buffer_for_deepselect(
    deepselect, out_frontend, provider_kind, skip_fill
):
    """Width 68 (a multiple of 4, not of the fake's 64-element alignment) gets a 128-stride
    ``out``; DeepSelect reads that buffer in place and selects the same scores as cuDNN."""
    fake = deepselect(provider_kind, input_align=256, seed=5)
    out_frontend()
    inputs = _packed_cp1((68, 20, 28), seed=21)
    deepselect_options = dsa_cudnn_kernels.IndexerScorerOptions(
        skip_masked_score_fill=skip_fill, topk_backend="deepselect"
    )
    cudnn_options = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=skip_fill)

    got = _packed_topk(inputs, 8, deepselect_options)
    out = OutBindingFe131.outs[-1]
    want = _packed_topk(inputs, 8, cudnn_options)

    assert out is not None and out.shape == (116, 68) and out.stride() == (128, 1)
    assert [(c["row_stride"], c["data_ptr"]) for c in fake.calls] == [(128, out.data_ptr())]
    assert OutBindingFe131.outs[-1] is None
    got_indices, got_length, got_scores = got
    want_indices, want_length, want_scores = want
    assert torch.equal(got_length, want_length)
    # Equal scores at the cutoff (relu zeros) may pick different columns; the scores may not.
    assert torch.equal(
        torch.sort(got_scores, dim=-1).values, torch.sort(want_scores, dim=-1).values
    )
    _, _, _, starts, ends, _, _ = inputs
    valid = got_indices >= 0
    assert bool((got_indices >= starts.view(1, -1, 1))[valid].all())
    assert bool((got_indices < ends.view(1, -1, 1))[valid].all())


@pytest.mark.parametrize(
    "case,version,backend,seq_lens,topk",
    [
        ("aligned_width", (1, 31), "deepselect", (64, 20), 8),
        ("width_not_multiple_of_4", (1, 31), "deepselect", (66, 20), 8),
        ("old_frontend", (1, 29), "deepselect", (68, 20), 8),
        ("cudnn_backend", (1, 31), "cudnn", (68, 20), 8),
        ("k_covers_row", (1, 31), "deepselect", (68, 20), 80),
    ],
)
def test_packed_thd_keeps_the_default_score_buffer_otherwise(
    deepselect, out_frontend, case, version, backend, seq_lens, topk
):
    deepselect(input_align=256)
    out_frontend(version)
    inputs = _packed_cp1(seq_lens, seed=22)

    _packed_topk(inputs, topk, dsa_cudnn_kernels.IndexerScorerOptions(topk_backend=backend))

    assert OutBindingFe131.outs == [None]


def test_padded_buffer_is_not_requested_in_deterministic_mode(deepselect, out_frontend):
    deepselect(input_align=256)
    out_frontend()
    kwargs = dsa_cudnn_kernels._deepselect_score_out_kwargs(
        n_rows=8,
        width=68,
        topk_k=8,
        device=torch.device("cpu"),
        deterministic=True,
        topk_backend="deepselect",
    )
    assert kwargs == {}
    kwargs = dsa_cudnn_kernels._deepselect_score_out_kwargs(
        n_rows=8,
        width=68,
        topk_k=8,
        device=torch.device("cpu"),
        deterministic=False,
        topk_backend="deepselect",
    )
    assert kwargs["out"].shape == (8, 68) and kwargs["out"].stride() == (128, 1)


# --------------------------------------------------------------------------------------- config
def test_scorer_options_read_the_top_k_backend():
    config = SimpleNamespace(dsa_indexer_topk_backend="deepselect")
    assert dsa_cudnn_kernels._scorer_options_from_config(config).topk_backend == "deepselect"
    assert dsa_cudnn_kernels._scorer_options_from_config(None).topk_backend == "cudnn"


def _dsa_config(**overrides):
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    args = dict(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=16,
        add_bias_linear=False,
    )
    args.update(overrides)
    return MLATransformerConfig(**args)


def test_config_validates_the_top_k_backend(monkeypatch):
    from megatron.core.transformer import transformer_config

    assert _dsa_config().dsa_indexer_topk_backend == "cudnn"
    with pytest.raises(ValueError, match="'cudnn' or 'deepselect'"):
        _dsa_config(dsa_indexer_topk_backend="radix")
    with pytest.raises(ValueError, match="requires experimental_attention_variant='dsa' with"):
        _dsa_config(dsa_indexer_topk_backend="deepselect")
    # The cuDNN backend's package checks need a GPU; the backend pairing is what matters here.
    monkeypatch.setattr(
        transformer_config, "_validate_dsa_kernel_backend_dependencies", lambda backend: None
    )
    config = _dsa_config(dsa_indexer_topk_backend="deepselect", dsa_kernel_backend="cudnn")
    assert config.dsa_indexer_topk_backend == "deepselect"
