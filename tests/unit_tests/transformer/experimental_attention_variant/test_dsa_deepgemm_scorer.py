# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_scorer_backend='deepgemm'`` on CPU with fake DeepGEMM, DeepSelect and cuDNN.

``dsa_deepgemm_fakes`` models DeepGEMM 2.8.2 (BF16 compressed logits over per-row ``[ks, ke)``,
padded rows, poisoned unwritten columns) and its packed UE8M0 MXFP8 quantization;
``dsa_cudnn_fe_fakes.Fe131DSA`` models the cuDNN MXFP8 scorer and the sparse indexer score
recompute; the DeepSelect and cuDNN radix top-k fakes come from ``test_dsa_deepselect_topk``.
Covered: the import probe, the eligibility and fallback matrix with its log-once, the CUDA
graph capture guard and JIT warm-up (a CUDA out-of-memory error is not recorded as a JIT
failure), argument plumbing (original K, int32 ranges, the host width bound, metadata from
the same ranges), DeepGEMM's per-call row limit and the row chunks that respect it, row
lengths against the cuDNN path's limits on randomized layouts, exact index restoration on
ragged layouts with empty and uncovered rows, top-k routing (DeepSelect in place, cuDNN on the
compact full-row view, decided once per call; frontends without the top-k lane guard keep the
cuDNN MXFP8 scorer), the loss-on value path (FP32 recompute payload, NaN and non-finite
selections), the loss-score rule over precision x scorer backend (cuDNN, DeepGEMM, DeepGEMM
falling back) x loss on/frozen x ranked/every-key (it follows the MXFP8 precision, never the
backend), the frozen path, no host synchronization, config validation and the paths that
cannot take MXFP8. The recompute helper itself is tested with the MXFP8 scorer
(``test_dsa_cudnn_mxfp8_scorer.py``). The real kernels are covered by
``test_dsa_deepgemm_scorer_gpu.py``.
"""

import importlib.abc
import importlib.util
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.quantization import indexer_quantization
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_deepgemm_scorer,
    dsa_deepselect_topk,
    dsa_layout,
    dsa_masking,
)
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes
from tests.unit_tests.transformer.experimental_attention_variant import (
    dsa_deepgemm_fakes as dgfakes,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsa_deepselect_topk import (  # noqa: E501
    AnyDtypeRadixTopK,
    FakeDsTune,
)

_TOLIST = torch.Tensor.tolist
_REPO_ROOT = Path(__file__).resolve().parents[4]
HEADS, HEAD_DIM = 32, 128
DEEPGEMM = dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8", scorer_backend="deepgemm")
DEEPGEMM_DS = dsa_cudnn_kernels.IndexerScorerOptions(
    precision="mxfp8", scorer_backend="deepgemm", topk_backend="deepselect"
)
MXFP8 = dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8")
BF16 = dsa_cudnn_kernels.IndexerScorerOptions()


class DeepGemmFe(fakes.Fe131CleanLogitsDSA):
    """FE 1.31 fake (MXFP8 scorer, clean_logits, recompute) with a swappable radix top-k."""

    calls: list = []
    recompute_calls: list = []
    topk = None

    @classmethod
    def indexer_top_k_wrapper(
        cls,
        input_values,
        seq_lens,
        top_k,
        next_n=1,
        return_val=True,
        num_copy_bits=256,
        stream=None,
        tie_break=0,
    ):
        return cls.topk.indexer_top_k_wrapper(
            input_values, seq_lens, top_k, next_n, return_val, num_copy_bits, stream, tie_break
        )


@pytest.fixture
def dg(monkeypatch):
    """DeepGEMM, cuDNN FE 1.31 and quantizer fakes, a supported device, nothing warm."""
    DeepGemmFe.reset()
    DeepGemmFe.topk = AnyDtypeRadixTopK(seed=5)
    fake = dgfakes.FakeDeepGEMM()
    state = SimpleNamespace(dg=fake, fe=DeepGemmFe, dg_quant=[], fe_quant=[])
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", DeepGemmFe)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    monkeypatch.setattr(dsa_cudnn_kernels, "_mxfp8_scorer_device_problem", lambda device: None)
    monkeypatch.setattr(indexer_quantization, "HAVE_TE_MXFP8", True)
    monkeypatch.setattr(indexer_quantization, "HAVE_TRITON", True)
    monkeypatch.setattr(
        indexer_quantization,
        "quantize_indexer_mxfp8",
        lambda x, **kwargs: fakes.fake_quantize_indexer_mxfp8(x, calls=state.fe_quant, **kwargs),
    )
    monkeypatch.setattr(
        indexer_quantization,
        "quantize_indexer_mxfp8_packed_ue8m0",
        lambda x: dgfakes.fake_quantize_packed_ue8m0(x, calls=state.dg_quant),
    )
    monkeypatch.setattr(
        dsa_deepgemm_scorer,
        "_IMPORT_STATE",
        {"provider": dsa_deepgemm_scorer._Provider(fake, (2, 8)), "error": None},
    )
    monkeypatch.setattr(dsa_deepgemm_scorer, "_device_problem", lambda device: None)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: False)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_LOGGED_REASONS", set())
    monkeypatch.setattr(dsa_deepgemm_scorer, "_WARM", set())
    monkeypatch.setattr(dsa_deepgemm_scorer, "_JIT_FAILURES", {})
    monkeypatch.setattr(dsa_deepgemm_scorer, "_WARM_SPECS", set())
    monkeypatch.setattr(dsa_deepgemm_scorer, "_FAILED_SPECS", set())
    monkeypatch.setattr(dsa_deepgemm_scorer, "_PROCESS_DEVICE", {})
    monkeypatch.setattr(dsa_deepselect_topk, "_IMPORT_STATE", {"provider": None, "error": "none"})
    monkeypatch.setattr(dsa_deepselect_topk, "_LOGGED_REASONS", set())
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    return state


def _install_deepselect(monkeypatch, min_width=64, **kwargs):
    fake = FakeDsTune(**kwargs)
    provider = dsa_deepselect_topk._Provider("ds_tune", fake)
    monkeypatch.setattr(dsa_deepselect_topk, "_IMPORT_STATE", {"provider": provider, "error": None})
    monkeypatch.setattr(dsa_deepselect_topk, "_device_supported", lambda device: True)
    widths = {"ds_tune": min_width, "deep_select": min_width}
    monkeypatch.setattr(dsa_deepselect_topk, "MIN_SCORE_WIDTH", widths)
    monkeypatch.setattr(dsa_deepselect_topk, "MANY_ROWS_MIN_SCORE_WIDTH", {})
    # DeepGEMM's logits are BF16, which has its own width table.
    monkeypatch.setitem(
        dsa_deepselect_topk.SCORE_WIDTH_TABLES_BY_DTYPE, torch.bfloat16, (widths, {})
    )
    return fake


def _inputs(sq, sk, kind="gauss", seed=0, heads=HEADS, dim=HEAD_DIM, w_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    if kind == "ternary":
        # {-1, 0, 1} survive E4M3 and BF16 exactly; one-hot weights keep the head sum exact.
        q = torch.randint(-1, 2, (1, sq, heads, dim), generator=generator).to(torch.bfloat16)
        k = torch.randint(-1, 2, (1, sk, dim), generator=generator).to(torch.bfloat16)
        w = torch.zeros((1, sq, heads), dtype=w_dtype)
        w[0, torch.arange(sq), torch.arange(sq) % heads] = 1
        return q, k, w
    q = torch.randn((1, sq, heads, dim), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sk, dim), generator=generator).to(torch.bfloat16)
    w = (torch.randn((1, sq, heads), generator=generator) * heads**-0.5).to(w_dtype)
    return q, k, w


def _layout(seq_lens, cp_size, cp_rank, max_num_seqs=None, uncovered=0):
    """``packed_zigzag_layout`` plus ``uncovered`` padding rows past ``cu[-1]``."""
    starts, ends, cu, max_seqlen = fakes.packed_zigzag_layout(
        seq_lens, cp_size, cp_rank, max_num_seqs
    )
    if uncovered:
        total = int(_TOLIST(cu)[-1])
        positions = total + torch.arange(uncovered)
        starts = torch.cat((starts, torch.full((uncovered,), total, dtype=torch.int64)))
        ends = torch.cat((ends, positions + 1))
    return starts, ends, cu, max_seqlen


def _score(q, k, w, layout, cp_size, cp_rank, topk, options, return_topk_scores=True):
    starts, ends, cu, max_seqlen = layout
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
        packed_cp_size=cp_size,
        local_packed_cp_rank=cp_rank,
        scorer_options=options,
    )


def _case(seq_lens, cp_size, cp_rank, kind="gauss", seed=0, uncovered=0, max_num_seqs=None):
    layout = _layout(seq_lens, cp_size, cp_rank, max_num_seqs, uncovered)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * cp_size, kind=kind, seed=seed)
    return q, k, w, layout


def _width(cp_size, cp_rank, max_seqlen):
    return dsa_cudnn_kernels._packed_thd_score_width(
        cp_size=cp_size, cp_rank=cp_rank, packed_max_seqlen_k=max_seqlen
    )


def _cudnn_row_lengths(layout, cp_size, cp_rank, sk):
    """The skip-fill cuDNN path's per-row lengths, from the segment layout it scores."""
    starts, ends, cu, max_seqlen = layout
    sq = starts.numel()
    if cp_size == 1:
        score_cu_q = score_cu_k = cu
        offsets = None
    else:
        cp_layout = dsa_layout.build_packed_cp_indexer_layout(
            cu, cu, cp_size=cp_size, cp_rank=cp_rank, key_size=sk
        )
        score_cu_q, score_cu_k = cp_layout.segment_cu_q, cp_layout.segment_cu_k
        offsets = cp_layout.segment_k_lengths - cp_layout.segment_q_lengths
    limits = dsa_cudnn_kernels._packed_thd_score_column_limits(
        score_cu_q, score_cu_k, offsets, num_rows=sq, ratio=1
    )
    width = _width(cp_size, cp_rank, max_seqlen)
    return torch.minimum(ends - starts, limits).clamp(min=0).clamp(max=width)


def _strict_and_cutoff(values, lengths, topk):
    """Per row: the sorted values above the K-th largest in-row value and their count."""
    result = []
    for row, length in enumerate(_TOLIST(lengths)):
        row_values = values[row, :length].float()
        if length <= topk:
            result.append((sorted(_TOLIST(row_values)), 0))
            continue
        cutoff = torch.topk(row_values, topk).values[-1]
        above = row_values[row_values > cutoff]
        result.append((sorted(_TOLIST(above)), topk - above.numel()))
    return result


def _selected_sets(indices, lengths):
    rows = []
    for row, length in zip(_TOLIST(indices[0]), _TOLIST(lengths[0])):
        rows.append(sorted(row[:length]))
    return rows


# --------------------------------------------------------------------------------- import probe
class _Importer:
    def __init__(self, module=None, error=None):
        self.module, self.error, self.count = module, error, 0

    def __call__(self, name):
        assert name == "deep_gemm"
        self.count += 1
        if self.error is not None:
            raise self.error
        if self.module is None:
            raise ImportError("No module named 'deep_gemm'")
        return self.module


def _probe(monkeypatch, importer):
    monkeypatch.setattr(dsa_deepgemm_scorer, "_IMPORT_STATE", {})
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    monkeypatch.setattr(dsa_deepgemm_scorer.importlib, "import_module", importer)
    return dsa_deepgemm_scorer._load_provider()


class _PybindFunction:
    """A callable whose signature inspect cannot read, documented like a pybind11 function."""

    __signature__ = "not a signature"

    def __init__(self, doc):
        self.__doc__ = doc

    def __call__(self, *args, **kwargs):
        raise AssertionError("not called")


_PYBIND_LOGITS_DOC = (
    "fp8_fp4_mqa_logits(q: tuple[torch.Tensor, Optional[torch.Tensor]], kv: tuple[torch.Tensor,"
    " torch.Tensor], weights: torch.Tensor, cu_seq_len_k_start: torch.Tensor, cu_seq_len_k_end: "
    "torch.Tensor, max_seqlen_k: int, schedule_meta: Optional[torch.Tensor] = None) -> "
    "torch.Tensor\n"
)
# DeepGEMM 2.8.0's binding (66081d4, ea2b780): cleaned FP32 logits unless told otherwise.
_PYBIND_LOGITS_DOC_2_8_0 = (
    "fp8_fp4_mqa_logits(q: tuple[torch.Tensor, Optional[torch.Tensor]], kv: tuple[torch.Tensor,"
    " torch.Tensor], weights: torch.Tensor, cu_seq_len_k_start: torch.Tensor, cu_seq_len_k_end: "
    "torch.Tensor, clean_logits: bool = True, max_seqlen_k: int = 0, logits_dtype: torch.dtype "
    "= torch.float32, schedule_meta: Optional[torch.Tensor] = None) -> torch.Tensor\n"
)
_PYBIND_META_DOC = (
    "get_mqa_logits_metadata(cu_seq_len_k_start: torch.Tensor, cu_seq_len_k_end: torch.Tensor, "
    "num_kv_tokens: int, num_heads: int) -> torch.Tensor\n"
)


def _pybind_module(version="2.8.2", logits_doc=_PYBIND_LOGITS_DOC):
    return SimpleNamespace(
        __version__=version,
        __file__="site/deep_gemm/__init__.py",
        fp8_fp4_mqa_logits=_PybindFunction(logits_doc),
        get_mqa_logits_metadata=_PybindFunction(_PYBIND_META_DOC),
        get_num_sms=lambda: 152,
    )


def test_probe_accepts_a_python_build_and_a_pybind_build_and_imports_once(monkeypatch):
    importer = _Importer(dgfakes.FakeDeepGEMM())
    provider = _probe(monkeypatch, importer)
    assert provider is not None and provider.version == (2, 8, 2)
    assert dsa_deepgemm_scorer._load_provider() is provider and importer.count == 1
    provider = _probe(monkeypatch, _Importer(_pybind_module()))
    assert provider is not None and provider.num_sms() == 152
    assert "deep_gemm 2.8.2 from site/deep_gemm/__init__.py" in provider.description


@pytest.mark.parametrize(
    "version, parsed", [("2.8.1", (2, 8, 1)), ("2.8.2+5081f85", (2, 8, 2)), ("3.0", (3, 0, 0))]
)
def test_probe_accepts_2_8_1_and_newer(monkeypatch, version, parsed):
    provider = _probe(monkeypatch, _Importer(_pybind_module(version=version)))
    assert provider is not None and provider.version == parsed


@pytest.mark.parametrize(
    "importer, message",
    [
        (_Importer(), "cannot import deep_gemm: ImportError: No module named 'deep_gemm'"),
        (_Importer(error=OSError("libcudart.so.13")), "cannot import deep_gemm: OSError"),
        (_Importer(error=RuntimeError("torch ABI")), "cannot import deep_gemm: RuntimeError"),
        (_Importer(error=AttributeError("_C")), "cannot import deep_gemm: AttributeError"),
        (_Importer(_pybind_module(version="2.7.4")), "deep_gemm 2.7.4 is older than 2.8.1"),
        (
            _Importer(_pybind_module(version="2.8.0+ea2b780", logits_doc=_PYBIND_LOGITS_DOC_2_8_0)),
            "deep_gemm 2.8.0+ea2b780 is older than 2.8.1",
        ),
        (_Importer(_pybind_module(version="2.8")), "deep_gemm 2.8 is older than 2.8.1"),
        (
            _Importer(_pybind_module(version="2.8.1", logits_doc=_PYBIND_LOGITS_DOC_2_8_0)),
            "has the pre-2.8.1 fp8_fp4_mqa_logits(clean_logits=, logits_dtype=) API",
        ),
        (_Importer(_pybind_module(version="dev")), "no parsable __version__"),
        (
            _Importer(_pybind_module(logits_doc=_PYBIND_LOGITS_DOC.replace("schedule_meta", "m"))),
            "fp8_fp4_mqa_logits(schedule_meta=)",
        ),
        (
            _Importer(_pybind_module(logits_doc="fp8_fp4_mqa_logits(*args, **kwargs)\n")),
            "fp8_fp4_mqa_logits argument names",
        ),
        (
            _Importer(SimpleNamespace(__version__="2.8.2", get_num_sms=lambda: 1)),
            "lacks fp8_fp4_mqa_logits, get_mqa_logits_metadata",
        ),
    ],
    ids=[
        "missing",
        "oserror",
        "runtimeerror",
        "attributeerror",
        "old-version",
        "2.8.0-pin",
        "no-patch-version",
        "2.8.0-api",
        "no-version",
        "renamed-argument",
        "unnamed-arguments",
        "no-functions",
    ],
)
def test_probe_records_a_stable_reason(monkeypatch, importer, message):
    assert _probe(monkeypatch, importer) is None
    assert message in dsa_deepgemm_scorer._IMPORT_STATE["error"]
    assert dsa_deepgemm_scorer._load_provider() is None and importer.count == 1


def test_probe_reuses_an_already_imported_deep_gemm(monkeypatch):
    module = dgfakes.FakeDeepGEMM()
    monkeypatch.setattr(dsa_deepgemm_scorer, "_IMPORT_STATE", {})
    monkeypatch.setitem(sys.modules, "deep_gemm", module)
    importer = _Importer(error=AssertionError("must not import again"))
    monkeypatch.setattr(dsa_deepgemm_scorer.importlib, "import_module", importer)
    assert dsa_deepgemm_scorer._load_provider().module is module and importer.count == 0


class _RaisingFinder(importlib.abc.MetaPathFinder):
    def __init__(self, error):
        self.error = error

    def find_spec(self, name, path=None, target=None):
        if name == "deep_gemm":
            raise self.error
        return None


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError, AttributeError])
def test_a_failing_deep_gemm_import_keeps_megatron_importable(monkeypatch, error):
    """batch_invariant_kernels imports deep_gemm at Megatron import time; any loader failure
    of a build for another torch must leave both modules importable and in agreement."""
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_RaisingFinder(error("bad build"))] + sys.meta_path)
    path = (
        _REPO_ROOT / "megatron/core/transformer/custom_layers/batch_invariant_kernels.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("_batch_invariant_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.deep_gemm is None and module.HAVE_DEEPGEMM_BF16 is False
    monkeypatch.setattr(dsa_deepgemm_scorer, "_IMPORT_STATE", {})
    assert dsa_deepgemm_scorer._load_provider() is None
    assert f"cannot import deep_gemm: {error.__name__}: bad build" == (
        dsa_deepgemm_scorer._IMPORT_STATE["error"]
    )


# -------------------------------------------------------------------- eligibility and fallback
@pytest.mark.parametrize(
    "sm, message",
    [
        ((10, 3), None),
        ((10, 0), "DeepGEMM is not validated on SM 10.0"),
        ((9, 0), "need SM 10.x, got SM 9.0"),
    ],
)
def test_device_rule(monkeypatch, sm, message):
    monkeypatch.setattr(dsa_deepgemm_scorer, "_device_capability", lambda index: sm)
    problem = dsa_deepgemm_scorer._device_problem(torch.device("cuda", 0))
    assert problem is None if message is None else message in problem
    assert "needs a CUDA device" in dsa_deepgemm_scorer._device_problem(torch.device("cpu"))


def _fallback_case(dg, monkeypatch, case):
    heads, dim, w_dtype = HEADS, HEAD_DIM, torch.bfloat16
    deterministic = False
    if case == "sm100":
        monkeypatch.setattr(
            dsa_deepgemm_scorer,
            "_device_problem",
            lambda device: "DeepGEMM is not validated on SM 10.0 (validated: SM 10.3)",
        )
    elif case == "no-build":
        monkeypatch.setattr(
            dsa_deepgemm_scorer,
            "_IMPORT_STATE",
            {"provider": None, "error": "cannot import deep_gemm: ImportError: x"},
        )
    elif case == "torch-deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    elif case == "deterministic-arg":
        deterministic = True
    elif case == "jit-failure":

        def broken(*_args, **_kwargs):
            raise RuntimeError("nvcc not found")

        monkeypatch.setattr(dg.dg, "fp8_fp4_mqa_logits", broken)
    else:
        raise AssertionError(case)
    return heads, dim, w_dtype, deterministic


@pytest.mark.parametrize(
    "case, level, reason",
    [
        ("sm100", logging.WARNING, "not validated on SM 10.0"),
        ("no-build", logging.WARNING, "cannot import deep_gemm"),
        ("torch-deterministic", logging.INFO, "deterministic mode"),
        ("deterministic-arg", logging.INFO, "deterministic mode"),
        ("jit-failure", logging.WARNING, "JIT warm-up failed: RuntimeError: nvcc not found"),
    ],
)
def test_environment_fallbacks_run_the_cudnn_mxfp8_scorer_and_log_once(
    dg, monkeypatch, caplog, case, level, reason
):
    q, k, w, layout = _case((256, 128), 1, 0, seed=3)
    _, _, _, deterministic = _fallback_case(dg, monkeypatch, case)
    starts, ends, cu, max_seqlen = layout
    with caplog.at_level(logging.INFO, logger=dsa_deepgemm_scorer.__name__):
        for _ in range(3):
            dsa_cudnn_kernels._indexer_topk_bshd(
                q,
                k,
                w,
                16,
                varlen_starts=starts,
                varlen_ends=ends,
                return_scores=False,
                return_topk_scores=False,
                use_local_indexer_varlen=True,
                packed_cu_seqlens_q=cu,
                packed_cu_seqlens_k=cu,
                packed_max_seqlen_q=max_seqlen,
                packed_max_seqlen_k=max_seqlen,
                deterministic=deterministic,
                scorer_options=DEEPGEMM,
            )
    assert len(DeepGemmFe.calls) == 3
    assert all(call["precision"] == "mxfp8" for call in DeepGemmFe.calls)
    assert dg.dg.calls == []
    records = [r for r in caplog.records if r.name == dsa_deepgemm_scorer.__name__]
    assert [(r.levelno, reason in r.getMessage()) for r in records] == [(level, True)]
    if case == "jit-failure":
        assert len(dsa_deepgemm_scorer._JIT_FAILURES) == 1


@pytest.mark.parametrize(
    "case, message",
    [
        ("16-heads", "indexer heads must be 32 or 64"),
        ("head-dim-64", "head_dim must be 128"),
        ("fp32-weights", "indexer weights must be BF16"),
        ("no-te", "Transformer Engine MXFP8 quantization is unavailable"),
    ],
)
def test_static_mismatches_reach_the_cudnn_mxfp8_errors(dg, monkeypatch, caplog, case, message):
    heads = 16 if case == "16-heads" else HEADS
    dim = 64 if case == "head-dim-64" else HEAD_DIM
    w_dtype = torch.float32 if case == "fp32-weights" else torch.bfloat16
    if case == "no-te":
        monkeypatch.setattr(indexer_quantization, "HAVE_TE_MXFP8", False)
    layout = _layout((128, 128), 1, 0)
    q, k, w = _inputs(256, 256, heads=heads, dim=dim, w_dtype=w_dtype)
    with caplog.at_level(logging.INFO, logger=dsa_deepgemm_scorer.__name__):
        with pytest.raises(RuntimeError, match=message):
            _score(q, k, w, layout, 1, 0, 8, DEEPGEMM)
    assert dg.dg.calls == []
    records = [r for r in caplog.records if r.name == dsa_deepgemm_scorer.__name__]
    assert [r.levelno for r in records] == [logging.WARNING]


def test_eligible_call_logs_the_build_once_and_warns_about_the_jit_cache(dg, monkeypatch, caplog):
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    q, k, w, layout = _case((256, 128), 1, 0)
    with caplog.at_level(logging.INFO, logger=dsa_deepgemm_scorer.__name__):
        for _ in range(3):
            _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    messages = [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.name == dsa_deepgemm_scorer.__name__
    ]
    assert len(messages) == 2
    assert messages[0][0] == logging.INFO and "uses DeepGEMM (deep_gemm 2.8.2" in messages[0][1]
    assert messages[1][0] == logging.WARNING and "DG_JIT_CACHE_DIR is unset" in messages[1][1]
    assert DeepGemmFe.calls == []


# ------------------------------------------------------------------ capture guard and warm-up
def test_capture_with_a_cold_kernel_raises_and_eager_warms_it(dg, monkeypatch):
    q, k, w, layout = _case((256, 128), 1, 0)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert dg.dg.calls == []
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: False)
    eager = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    # The first eager call warmed the specialization (warm-up shape) and then scored.
    assert [c["max_seqlen_k"] for c in dg.dg.calls] == [256, 256]
    assert dg.dg.calls[0]["ks"].numel() == 128
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    captured = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert len(dg.dg.calls) == 3
    assert torch.equal(eager[1], captured[1])


def test_the_warm_key_covers_heads_and_the_deepgemm_sm_count(dg, monkeypatch):
    q, k, w, layout = _case((256, 128), 1, 0)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    q64, k64, w64 = _inputs(384, 384, heads=64)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        _score(q64, k64, w64, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    dg.dg.set_num_sms(120)  # another DeepGEMM user changed the SM count in-process
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)


def test_mqa_logits_and_warmup_guard_capture_directly(dg, monkeypatch):
    q8, qs = dgfakes.fake_quantize_packed_ue8m0(torch.ones((4, HEADS, HEAD_DIM)).bfloat16())
    k8, kscale = dgfakes.fake_quantize_packed_ue8m0(torch.ones((8, HEAD_DIM)).bfloat16())
    w = torch.ones((4, HEADS), dtype=torch.bfloat16)
    ks = torch.zeros(4, dtype=torch.int32)
    ke = torch.arange(1, 5, dtype=torch.int32)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=8)
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        dsa_deepgemm_scorer.warmup(torch.device("cpu"))
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: False)
    dsa_deepgemm_scorer.warmup(torch.device("cpu"), num_heads=HEADS)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    logits = dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=8)
    assert logits.dtype == torch.bfloat16 and logits.shape == (4, 8)


def _deep_gemm_calls_fail(dg, monkeypatch):
    """Make every deep_gemm function raise and record its name (restore with the return)."""
    called = []
    originals = {
        name: getattr(dg.dg, name)
        for name in ("get_num_sms", "get_mqa_logits_metadata", "fp8_fp4_mqa_logits")
    }

    def failing(name):
        def call(*_args, **_kwargs):
            called.append(name)
            raise AssertionError(f"deep_gemm.{name} was called")

        return call

    for name in originals:
        monkeypatch.setattr(dg.dg, name, failing(name))

    def restore():
        for name, function in originals.items():
            monkeypatch.setattr(dg.dg, name, function)

    return called, restore


def test_the_capture_guard_calls_no_deep_gemm_function_first(dg, monkeypatch):
    """A process's first deep_gemm call, even get_num_sms(), builds DeepGEMM's runtime (cuBLASLt
    handle, device properties), which a capture cannot record: in a fresh process the guard
    must fire before any of them (GB300: the old order raised another error from inside
    get_num_sms)."""
    q, k, w, layout = _case((256, 128), 1, 0)
    q8, qs = dgfakes.fake_quantize_packed_ue8m0(torch.ones((4, HEADS, HEAD_DIM)).bfloat16())
    k8, kscale = dgfakes.fake_quantize_packed_ue8m0(torch.ones((8, HEAD_DIM)).bfloat16())
    w4 = torch.ones((4, HEADS), dtype=torch.bfloat16)
    ks = torch.zeros(4, dtype=torch.int32)
    called, restore = _deep_gemm_calls_fail(dg, monkeypatch)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w4, ks, ks + 1, max_seqlen_k=8)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        dsa_deepgemm_scorer.jit_problem(torch.device("cpu"), HEADS, HEAD_DIM)
    assert called == [] and dsa_deepgemm_scorer._PROCESS_DEVICE == {}
    restore()
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: False)
    eager = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert dsa_deepgemm_scorer._WARM_SPECS == {(-1, HEADS, HEAD_DIM)}
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    captured = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert torch.equal(eager[1], captured[1])


def test_a_failed_warm_up_also_falls_back_inside_capture(dg, monkeypatch):
    """A recorded JIT failure is a per-device fact too: a later capture takes the cuDNN MXFP8
    scorer like the eager steps did instead of raising the guard."""
    q, k, w, layout = _case((256, 128), 1, 0, seed=3)
    real = dg.dg.fp8_fp4_mqa_logits

    def broken(*_args, **_kwargs):
        raise RuntimeError("nvcc not found")

    monkeypatch.setattr(dg.dg, "fp8_fp4_mqa_logits", broken)
    eager = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert dsa_deepgemm_scorer._FAILED_SPECS == {(-1, HEADS, HEAD_DIM)}
    monkeypatch.setattr(dg.dg, "fp8_fp4_mqa_logits", real)
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    captured = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert len(DeepGemmFe.calls) == 2 and dg.dg.calls == []
    assert torch.equal(eager[1], captured[1])


def test_one_deepgemm_device_per_process(dg, monkeypatch, caplog):
    """deep_jit keeps one kernel handle per kernel and process (GB300: ranks warmed up on
    cuda:0 failed on their own device with CUDA_ERROR_INVALID_HANDLE): the first device that
    calls DeepGEMM owns it; another device falls back with the reason, direct calls raise."""
    device = {"index": 0}
    monkeypatch.setattr(dsa_deepgemm_scorer, "_device_index", lambda _device: device["index"])
    q, k, w, layout = _case((256, 128), 1, 0, seed=5)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert dsa_deepgemm_scorer._PROCESS_DEVICE == {"index": 0} and len(dg.dg.calls) == 2
    assert DeepGemmFe.calls == []

    device["index"] = 1
    called, _restore = _deep_gemm_calls_fail(dg, monkeypatch)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=dsa_deepgemm_scorer.__name__):
        for _ in range(2):
            _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert len(DeepGemmFe.calls) == 2 and called == []
    records = [r for r in caplog.records if r.name == dsa_deepgemm_scorer.__name__]
    assert [r.levelno for r in records] == [logging.WARNING]
    message = records[0].getMessage()
    assert "already runs on cuda:0" in message and "cuda:1 cannot use it" in message
    assert "one device per process" in message
    reason = dsa_deepgemm_scorer.jit_problem(torch.device("cpu"), HEADS, HEAD_DIM)
    assert reason is not None and "one device per process" in reason
    # A capture on that device falls back like its eager steps (no guard error).
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: True)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert len(DeepGemmFe.calls) == 3
    monkeypatch.setattr(dsa_deepgemm_scorer, "_capturing", lambda: False)
    q8, qs = dgfakes.fake_quantize_packed_ue8m0(torch.ones((4, HEADS, HEAD_DIM)).bfloat16())
    k8, kscale = dgfakes.fake_quantize_packed_ue8m0(torch.ones((8, HEAD_DIM)).bfloat16())
    ks = torch.zeros(4, dtype=torch.int32)
    w4 = torch.ones((4, HEADS), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="one device per process"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w4, ks, ks + 1, max_seqlen_k=8)
    with pytest.raises(RuntimeError, match="one device per process"):
        dsa_deepgemm_scorer.warmup(torch.device("cpu"), num_heads=HEADS)
    assert called == [] and dsa_deepgemm_scorer._PROCESS_DEVICE == {"index": 0}


@pytest.mark.parametrize("failing", ["logits-kernel", "quantize"])
def test_the_first_deepgemm_call_claims_the_device_even_if_its_warm_up_fails(
    dg, monkeypatch, failing
):
    """The device is claimed before the first deep_gemm call (the warm key's get_num_sms
    already builds DeepGEMM's runtime there), so a warm-up that fails later, in the logits
    kernel or even before it, still owns the device."""
    device = {"index": 2}
    monkeypatch.setattr(dsa_deepgemm_scorer, "_device_index", lambda _device: device["index"])

    def broken(*_args, **_kwargs):
        raise RuntimeError("nvcc not found")

    if failing == "quantize":
        monkeypatch.setattr(indexer_quantization, "quantize_indexer_mxfp8_packed_ue8m0", broken)
    else:
        monkeypatch.setattr(dg.dg, "fp8_fp4_mqa_logits", broken)
    reason = dsa_deepgemm_scorer.jit_problem(torch.device("cpu"), HEADS, HEAD_DIM)
    assert reason.startswith("DeepGEMM JIT warm-up failed")
    assert dg.dg.calls == []
    assert dsa_deepgemm_scorer._PROCESS_DEVICE == {"index": 2}
    device["index"] = 0
    assert "one device per process" in dsa_deepgemm_scorer.jit_problem(
        torch.device("cpu"), HEADS, HEAD_DIM
    )


def test_out_of_memory_in_the_warm_up_propagates_and_is_not_recorded(dg, monkeypatch):
    """Only a compile/JIT failure is a lasting fallback reason; an OOM must surface."""
    q, k, w, layout = _case((256, 128), 1, 0)
    real_warmup = dsa_deepgemm_scorer.warmup

    def oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(dsa_deepgemm_scorer, "warmup", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert dsa_deepgemm_scorer._JIT_FAILURES == {} and DeepGemmFe.calls == []
    monkeypatch.setattr(dsa_deepgemm_scorer, "warmup", real_warmup)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, return_topk_scores=False)
    assert len(dg.dg.calls) == 2 and DeepGemmFe.calls == []  # warm-up + the call


def test_output_contract_violations_raise(dg, monkeypatch):
    q8, qs = dgfakes.fake_quantize_packed_ue8m0(torch.ones((4, HEADS, HEAD_DIM)).bfloat16())
    k8, kscale = dgfakes.fake_quantize_packed_ue8m0(torch.ones((8, HEAD_DIM)).bfloat16())
    w = torch.ones((4, HEADS), dtype=torch.bfloat16)
    ks = torch.zeros(4, dtype=torch.int32)
    ke = torch.arange(1, 5, dtype=torch.int32)
    monkeypatch.setattr(
        dg.dg, "fp8_fp4_mqa_logits", lambda **kwargs: torch.zeros((4, 8), dtype=torch.float32)
    )
    with pytest.raises(RuntimeError, match="expected BF16"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=8)
    monkeypatch.setattr(
        dg.dg, "fp8_fp4_mqa_logits", lambda **kwargs: torch.zeros((4, 8), dtype=torch.bfloat16)
    )
    with pytest.raises(RuntimeError, match="not a multiple of 1024 bytes"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=8)
    monkeypatch.setattr(
        dg.dg,
        "fp8_fp4_mqa_logits",
        lambda **kwargs: torch.zeros((1, 512), dtype=torch.bfloat16)[:, :8].expand(4, 8),
    )
    with pytest.raises(RuntimeError, match="strides"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=8)


# ------------------------------------------------------------------------- argument plumbing
@pytest.mark.parametrize(
    "seq_lens, cp_size, cp_rank, max_num_seqs",
    [((256, 128, 128), 1, 0, None), ((256, 128), 1, 0, 4), ((256, 512), 2, 1, 3)],
)
def test_deepgemm_receives_the_original_k_and_int32_ranges(
    dg, monkeypatch, seq_lens, cp_size, cp_rank, max_num_seqs
):
    def no_pool(*_args, **_kwargs):
        raise AssertionError("the DeepGEMM path must not build the duplicated-prefix K pool")

    q, k, w, layout = _case(seq_lens, cp_size, cp_rank, max_num_seqs=max_num_seqs)
    _score(q, k, w, layout, cp_size, cp_rank, 16, DEEPGEMM, return_topk_scores=False)  # warm-up
    dg.dg.calls.clear()
    dg.dg.metadata_calls.clear()
    dg.dg_quant.clear()
    monkeypatch.setattr(dsa_layout, "build_packed_cp_indexer_layout", no_pool)
    _score(q, k, w, layout, cp_size, cp_rank, 16, DEEPGEMM, return_topk_scores=False)

    starts, ends, cu, max_seqlen = layout
    sq, sk = q.size(1), k.size(1)
    width = _width(cp_size, cp_rank, max_seqlen)
    (call,) = dg.dg.calls
    (meta_call,) = dg.dg.metadata_calls
    assert DeepGemmFe.calls == []
    assert call["max_seqlen_k"] == width and isinstance(call["max_seqlen_k"], int)
    assert call["ks"].dtype == torch.int32 and call["ke"].dtype == torch.int32
    assert call["ks"].is_contiguous() and call["ke"].is_contiguous()
    assert torch.equal(call["ks"].long(), starts)
    assert bool(((call["ke"] - call["ks"]) <= width).all())
    assert meta_call["ks"] is call["ks"] and meta_call["ke"] is call["ke"]
    assert meta_call["num_kv_tokens"] == sk and meta_call["num_heads"] == HEADS
    assert call["schedule_meta"] is meta_call["meta"]
    assert torch.equal(call["weights"], w[0]) and call["weights"].dtype == torch.bfloat16
    k_quant, q_quant = dg.dg_quant  # K once for every row chunk, then each chunk's Q
    assert q_quant["shape"] == (sq, HEADS, HEAD_DIM) and torch.equal(q_quant["x"], q[0])
    assert k_quant["shape"] == (sk, HEAD_DIM) and torch.equal(k_quant["x"], k[0])
    assert call["q"].shape == (sq, HEADS, HEAD_DIM) and call["q_scale"].shape == (sq, HEADS)
    assert call["kv"].shape == (sk, HEAD_DIM) and call["kv_scale"].shape == (sk,)


@pytest.mark.parametrize("heads, limit", [(32, 116208), (64, 58104)])
def test_row_limit_matches_the_deepgemm_scheduler_shared_memory(heads, limit):
    assert dsa_deepgemm_scorer.max_rows_per_call(heads) == limit
    assert dgfakes.metadata_smem_bytes(limit, heads) <= dgfakes.SM100_SMEM_CAPACITY
    assert dgfakes.metadata_smem_bytes(limit + 1, heads) > dgfakes.SM100_SMEM_CAPACITY
    fake = dgfakes.FakeDeepGEMM()
    for rows, fits in ((limit, True), (limit + 1, False)):
        ks = torch.zeros(rows, dtype=torch.int32)
        call = lambda: fake.get_mqa_logits_metadata(  # noqa: E731
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ks + 1, num_kv_tokens=1, num_heads=heads
        )
        if fits:
            call()
        else:
            with pytest.raises(RuntimeError, match="smem_capacity"):
                call()


def test_mqa_logits_refuses_more_rows_than_one_call(dg, monkeypatch):
    monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda heads: 3)
    q8, qs = dgfakes.fake_quantize_packed_ue8m0(torch.ones((4, HEADS, HEAD_DIM)).bfloat16())
    k8, kscale = dgfakes.fake_quantize_packed_ue8m0(torch.ones((8, HEAD_DIM)).bfloat16())
    w = torch.ones((4, HEADS), dtype=torch.bfloat16)
    ks = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="at most 3 query rows with 32 heads, got 4"):
        dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ks + 1, max_seqlen_k=8)
    assert dg.dg.calls == [] and dg.dg.metadata_calls == []


@pytest.mark.parametrize(
    "n_rows, limit, want",
    [
        (384, 384, [(0, 384)]),
        (384, 200, [(0, 128), (128, 256), (256, 384)]),
        (1000, 300, [(0, 256), (256, 512), (512, 768), (768, 1000)]),
    ],
)
def test_row_chunks_respect_the_limit_at_aligned_starts(monkeypatch, n_rows, limit, want):
    monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda heads: limit)
    assert dsa_cudnn_kernels._deepgemm_row_chunks(n_rows, HEADS) == want


@pytest.mark.parametrize("heads", [32, 64])
def test_row_chunks_respect_the_te_mxfp8_cast_limit(monkeypatch, heads):
    """One TE MXFP8 cast takes at most 65535 * 64 rows of 128 (GB300: 65536 rows x 64 heads
    raised "invalid argument"); DeepGEMM's own limit is the smaller one today, and a chunk
    stays inside both if DeepGEMM's grows."""
    te_rows = dsa_cudnn_kernels._TE_MXFP8_CAST_MAX_ROWS // heads
    assert dsa_cudnn_kernels._TE_MXFP8_CAST_MAX_ROWS == 65535 * 64
    assert te_rows == {32: 131070, 64: 65535}[heads]
    for n_rows in (65536, 131072, 200000):
        chunks = dsa_cudnn_kernels._deepgemm_row_chunks(n_rows, heads)
        assert max(end - start for start, end in chunks) <= min(
            dsa_deepgemm_scorer.max_rows_per_call(heads), te_rows
        )
    monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda _heads: 10**7)
    chunks = dsa_cudnn_kernels._deepgemm_row_chunks(300000, heads)
    assert len(chunks) > 1 and max(end - start for start, end in chunks) <= te_rows
    assert all(start % 128 == 0 for start, _end in chunks)


@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 1)])
def test_q_is_quantized_per_row_chunk_and_k_once(dg, monkeypatch, cp_size, cp_rank):
    q, k, w, layout = _case((300, 84, 132), cp_size, cp_rank, seed=83)
    sq, sk = q.size(1), k.size(1)
    _score(q, k, w, layout, cp_size, cp_rank, 16, DEEPGEMM, False)  # warm-up
    monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda heads: 200)
    chunks = dsa_cudnn_kernels._deepgemm_row_chunks(sq, HEADS)
    assert len(chunks) > 1
    dg.dg_quant.clear()
    dg.dg.calls.clear()
    _score(q, k, w, layout, cp_size, cp_rank, 16, DEEPGEMM, True)
    k_quant, *q_quants = dg.dg_quant
    assert k_quant["shape"] == (sk, HEAD_DIM) and torch.equal(k_quant["x"], k[0])
    assert len(q_quants) == len(chunks) == len(dg.dg.calls)
    for (start, end), q_quant, call in zip(chunks, q_quants, dg.dg.calls):
        assert q_quant["shape"] == (end - start, HEADS, HEAD_DIM)
        assert torch.equal(q_quant["x"], q[0, start:end])
        assert call["q"].shape == (end - start, HEADS, HEAD_DIM)
        assert call["kv"].shape == (sk, HEAD_DIM)


def _selected_values(indices, lengths, logits):
    """Per row: the sorted reference logits of the selected keys (tie-aware comparison)."""
    return [
        sorted(_TOLIST(logits[row, chosen[:n]].float()))
        for row, (chosen, n) in enumerate(zip(_TOLIST(indices[0]), _TOLIST(lengths[0])))
    ]


@pytest.mark.parametrize("backend", ["cudnn", "deepselect"])
@pytest.mark.parametrize("topk", [16, 600], ids=["ranked", "every-key"])
@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 1)])
def test_row_chunks_select_and_score_what_one_call_does(
    dg, monkeypatch, backend, topk, cp_size, cp_rank
):
    """Rows past one DeepGEMM call's limit are scored and selected in chunks: same lengths,
    the same selection up to ties, values and the loss payload for each own selection."""
    if backend == "deepselect":
        _install_deepselect(monkeypatch, seed=7)
    options = DEEPGEMM_DS if backend == "deepselect" else DEEPGEMM
    q, k, w, layout = _case((300, 84, 132), cp_size, cp_rank, seed=79, uncovered=4 * cp_size)
    sq = q.size(1)
    _score(q, k, w, layout, cp_size, cp_rank, 16, options, False)  # warm-up
    results = {}
    for chunked in (False, True):
        if chunked:
            monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda heads: 200)
        dg.dg.calls.clear()
        DeepGemmFe.recompute_calls.clear()
        only = _score(q, k, w, layout, cp_size, cp_rank, topk, options, False)
        kept = _score(q, k, w, layout, cp_size, cp_rank, topk, options, True)
        rows = [call["ks"].numel() for call in dg.dg.calls]
        results[chunked] = (only, kept, rows)
    (only, kept, rows), (c_only, c_kept, c_rows) = results[False], results[True]
    width = _width(cp_size, cp_rank, layout[3])
    every_key = topk >= width
    calls_per_score = 0 if every_key else 1
    assert rows == [sq] * (calls_per_score + 1)
    chunk_rows = [end - start for start, end in dsa_cudnn_kernels._deepgemm_row_chunks(sq, HEADS)]
    assert c_rows == chunk_rows * (calls_per_score + 1) and max(c_rows) <= 200
    assert sum(chunk_rows) == sq and len(chunk_rows) > 1
    logits = dgfakes.reference_logits(q, k, w)
    for a, b in ((only, c_only), (kept, c_kept)):
        assert torch.equal(a[1], b[1])
        assert _selected_values(a[0], a[1], logits) == _selected_values(b[0], b[1], logits)
    for indices, lengths, scores in (kept, c_kept):
        want, valid = _bf16_reference_log_probs(q, k, w, indices, lengths)
        got = dsa_masking.masked_log_softmax(scores, valid)
        torch.testing.assert_close(got[valid], want[valid], rtol=0, atol=1e-5)


def test_packed_ue8m0_quantizer_views_te_logical_scales_as_int32(monkeypatch):
    rows, dim = 6, 128
    logical = torch.zeros((128, 4), dtype=torch.uint8)

    class Quantizer:
        def update_quantized(self, source, quantized):
            quantized["src"] = source
            logical[:rows] = torch.arange(rows * 4, dtype=torch.uint8).view(rows, 4) + 100

    buffers = SimpleNamespace(
        quantizer=Quantizer(),
        quantized={},
        data=torch.zeros((2, 3, dim), dtype=torch.float8_e4m3fn),
        logical_scale=logical,
        padded_input=None,
        num_rows=rows,
    )
    monkeypatch.setattr(
        indexer_quantization, "create_indexer_mxfp8_quantization_buffers", lambda x: buffers
    )
    x = torch.zeros((2, 3, dim), dtype=torch.bfloat16)
    data, scales = indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0(x)
    assert data is buffers.data and scales.dtype == torch.int32 and scales.shape == (2, 3)
    expected = [
        sum((100 + 4 * row + byte) << (8 * byte) for byte in range(4)) for row in range(rows)
    ]
    expected = [value - 2**32 if value >= 2**31 else value for value in expected]
    assert _TOLIST(scales.flatten()) == expected
    assert scales.data_ptr() == logical.data_ptr()
    with pytest.raises(ValueError, match="head_dim of 128"):
        indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0(torch.zeros((2, 64)).bfloat16())


def test_fake_quantizer_round_trips_ternary_and_flags_nan():
    x = torch.randint(-1, 2, (5, 2, 128)).to(torch.bfloat16)
    x[3, 1, 40] = float("nan")
    data, scales = dgfakes.fake_quantize_packed_ue8m0(x)
    back = dgfakes.dequantize_packed_ue8m0(data, scales)
    finite = torch.ones_like(back, dtype=torch.bool)
    finite[3, 1, 32:64] = False
    assert torch.equal(back[finite], x.float()[finite])
    assert bool(torch.isnan(back[3, 1, 32:64]).all())


# --------------------------------------------------------------------------------- row lengths
def _random_layout(rng, cp_size):
    divisor = 2 * cp_size if cp_size > 1 else 1
    count = rng.randint(1, 4)
    seq_lens = [divisor * rng.randint(1, 24 // divisor + 2) for _ in range(count)]
    max_num_seqs = rng.choice([None, count + rng.randint(0, 3)])
    # Rows past cu[-1] exist only at CP1: the CP>1 segment layout (and every CP>1 scheduler)
    # needs the packed sequences to fill the keys, so CP>1 pads with a dummy sequence instead.
    uncovered = rng.choice([0, 0, 2, 4]) if cp_size == 1 else 0
    return seq_lens, max_num_seqs, uncovered


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_row_lengths_equal_the_cudnn_column_limits(cp_size):
    rng = random.Random(cp_size)
    for _ in range(30):
        seq_lens, max_num_seqs, uncovered = _random_layout(rng, cp_size)
        for cp_rank in range(cp_size):
            layout = _layout(seq_lens, cp_size, cp_rank, max_num_seqs, uncovered)
            starts, ends, cu, max_seqlen = layout
            sq = starts.numel()
            width = _width(cp_size, cp_rank, max_seqlen)
            got = dsa_cudnn_kernels._packed_thd_row_lengths(
                starts, ends, cu, cp_size=cp_size, num_rows=sq, width=width
            )
            want = _cudnn_row_lengths(layout, cp_size, cp_rank, sq * cp_size)
            assert got.dtype == torch.int32
            assert torch.equal(got.long(), want.long()), (seq_lens, cp_size, cp_rank, uncovered)


def test_row_lengths_cover_a_single_sequence_pack_and_dummy_tail():
    layout = _layout((64,), 4, 3)
    starts, ends, cu, max_seqlen = layout
    got = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=4, num_rows=starts.numel(), width=_width(4, 3, max_seqlen)
    )
    assert torch.equal(got.long(), _cudnn_row_lengths(layout, 4, 3, 64).long())
    layout = _layout((48, 16), 1, 0, max_num_seqs=5, uncovered=6)
    starts, ends, cu, max_seqlen = layout
    got = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=1, num_rows=starts.numel(), width=max_seqlen
    )
    assert _TOLIST(got[-6:]) == [0] * 6 and _TOLIST(got[:3]) == [1, 2, 3]


def test_row_lengths_never_exceed_the_host_width_bound():
    """Like the cuDNN path's ``local_seq_lens.clamp(max=max_score_k)``: DeepGEMM requires
    ``ke - ks <= max_seqlen_k`` even if the metadata understated the longest sequence."""
    starts, ends, cu, _max = _layout((48, 16), 1, 0)
    got = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=1, num_rows=64, width=20
    )
    assert int(got.max()) == 20 and _TOLIST(got[:3]) == [1, 2, 3]


# ----------------------------------------------------------------------------------- selection
@pytest.mark.parametrize(
    "seq_lens, cp_size, cp_rank, uncovered",
    [
        ((300, 84, 40), 1, 0, 0),
        ((256, 64), 1, 0, 8),
        ((256, 128), 2, 0, 0),
        ((256, 128), 2, 1, 4),
        ((512, 256), 4, 3, 0),
    ],
)
@pytest.mark.parametrize("backend", ["cudnn", "deepselect"])
def test_selection_is_a_top_k_of_the_logits_restored_to_global_keys(
    dg, monkeypatch, seq_lens, cp_size, cp_rank, uncovered, backend
):
    if backend == "deepselect":
        _install_deepselect(monkeypatch, seed=7)
    options = DEEPGEMM_DS if backend == "deepselect" else DEEPGEMM
    q, k, w, layout = _case(seq_lens, cp_size, cp_rank, seed=11, uncovered=uncovered)
    starts, ends, cu, max_seqlen = layout
    topk = 16
    indices, lengths, _ = _score(q, k, w, layout, cp_size, cp_rank, topk, options, False)

    sq, sk = q.size(1), k.size(1)
    width = _width(cp_size, cp_rank, max_seqlen)
    row_lengths = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=cp_size, num_rows=sq, width=width
    )
    assert torch.equal(lengths[0].long(), row_lengths.long().clamp(max=topk))
    logits = dgfakes.reference_logits(q, k, w)
    columns = torch.arange(width).view(1, -1)
    keys = (starts.view(-1, 1) + columns).clamp(max=sk - 1)
    local = torch.gather(logits, 1, keys)
    want = _strict_and_cutoff(local, row_lengths, topk)
    for row, (selected, length) in enumerate(zip(_TOLIST(indices[0]), _TOLIST(lengths[0]))):
        chosen = selected[:length]
        start, count = int(_TOLIST(starts)[row]), int(_TOLIST(row_lengths)[row])
        assert all(start <= key < start + count for key in chosen), row
        assert len(set(chosen)) == length
        values = sorted(_TOLIST(logits[row, chosen].float()))
        above, cutoff_count = want[row]
        assert len(values) == len(above) + cutoff_count, row
        assert values[cutoff_count:] == above, row
        assert len(set(values[:cutoff_count])) <= 1, row
    if uncovered:
        assert _TOLIST(lengths[0, -uncovered:]) == [0] * uncovered


@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_ternary_selection_equals_the_cudnn_mxfp8_scorer_tie_aware(dg, cp_size, cp_rank):
    """Exact in E4M3 and BF16: both scorers rank identical values; ties may differ."""
    q, k, w, layout = _case((256, 128), cp_size, cp_rank, kind="ternary", seed=13)
    starts = layout[0]
    sk = k.size(1)
    got, got_len, _ = _score(q, k, w, layout, cp_size, cp_rank, 24, DEEPGEMM, False)
    want, want_len, _ = _score(q, k, w, layout, cp_size, cp_rank, 24, MXFP8, False)
    assert torch.equal(got_len, want_len)
    logits = dgfakes.reference_logits(q, k, w).float()
    for row, (a, b, n) in enumerate(zip(_TOLIST(got[0]), _TOLIST(want[0]), _TOLIST(got_len[0]))):
        values_a = sorted(_TOLIST(logits[row, a[:n]]))
        values_b = sorted(_TOLIST(logits[row, b[:n]]))
        assert values_a == values_b, row
        assert all(int(_TOLIST(starts)[row]) <= key < sk for key in a[:n])


def test_width_at_most_topk_selects_every_in_row_key_without_scoring(dg):
    q, k, w, layout = _case((24, 16), 1, 0, uncovered=2)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, False)  # warm-up
    dg.dg.calls.clear()
    indices, lengths, scores = _score(q, k, w, layout, 1, 0, 32, DEEPGEMM, False)
    assert dg.dg.calls == [] and scores is None
    starts, ends = layout[0], layout[1]
    want = torch.minimum(ends - starts, torch.tensor(24))
    want[-2:] = 0
    assert torch.equal(lengths[0].long(), want)
    for row, n in enumerate(_TOLIST(lengths[0])):
        start = int(_TOLIST(starts)[row])
        assert _TOLIST(indices[0, row, :n]) == list(range(start, start + n))


def test_poisoned_columns_are_never_selected_by_either_top_k(dg, monkeypatch):
    q, k, w, layout = _case((200, 56), 1, 0, seed=17)
    starts, ends = layout[0], layout[1]
    for options in (DEEPGEMM, DEEPGEMM_DS):
        if options is DEEPGEMM_DS:
            _install_deepselect(monkeypatch)
        indices, lengths, _ = _score(q, k, w, layout, 1, 0, 16, options, False)
        valid = torch.arange(indices.size(-1)).view(1, 1, -1) < lengths.unsqueeze(-1)
        in_row = (indices >= starts.view(1, -1, 1)) & (indices < ends.view(1, -1, 1))
        assert bool((in_row | ~valid).all())
        assert bool((lengths[0] == torch.minimum(ends - starts, torch.tensor(16))).all())


@pytest.mark.parametrize("options", [DEEPGEMM, DEEPGEMM_DS], ids=["cudnn-topk", "deepselect"])
def test_frontend_without_the_lane_guard_keeps_the_cudnn_mxfp8_scorer(
    dg, monkeypatch, caplog, options
):
    """DeepGEMM leaves each row's tail unwritten; only a frontend whose top-k never reads past
    a row's length (>= 1.29) may read it, so older ones keep the cuDNN MXFP8 scorer for every
    call (no per-batch switching under dynamic CP), including those DeepSelect would take."""
    if options is DEEPGEMM_DS:
        _install_deepselect(monkeypatch)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 21))
    q, k, w, layout = _case((200, 56), 1, 0, seed=19)
    with caplog.at_level(logging.INFO, logger=dsa_deepgemm_scorer.__name__):
        for topk in (16, 300):  # ranked and every-key
            for return_topk_scores in (False, True):
                _score(q, k, w, layout, 1, 0, topk, options, return_topk_scores)
    assert dg.dg.calls == [] and dg.dg.metadata_calls == []
    assert len(DeepGemmFe.calls) == 4
    assert all(call["precision"] == "mxfp8" for call in DeepGemmFe.calls)
    records = [r for r in caplog.records if r.name == dsa_deepgemm_scorer.__name__]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "(1, 21) is older than 1.29" in records[0].getMessage()


def test_frontend_without_the_lane_guard_or_the_mxfp8_scorer_raises(dg, monkeypatch):
    """FE 1.21.1 (the GLM-5.2 recipes' pin) has neither: deepgemm cannot run there."""
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 21))
    monkeypatch.setattr(
        dsa_cudnn_kernels,
        "_MXFP8_SCORER_WRAPPER_PARAMETERS",
        dsa_cudnn_kernels._MXFP8_SCORER_WRAPPER_PARAMETERS + ("not_in_fe_1_21",),
    )
    q, k, w, layout = _case((200, 56), 1, 0, seed=19)
    with pytest.raises(RuntimeError, match="cuDNN frontend >= 1.29 provides the MXFP8"):
        _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, False)
    assert dg.dg.calls == []


# ------------------------------------------------------------------------------ top-k routing
def test_deepselect_reads_the_deepgemm_logits_in_place(dg, monkeypatch):
    fake = _install_deepselect(monkeypatch)
    q, k, w, layout = _case((300, 84), 1, 0, seed=23)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM_DS, False)
    logits = dg.dg.calls[-1]["logits"]
    call = fake.calls[-1]
    assert call["data_ptr"] == logits.data_ptr() and call["width"] == 300
    assert call["row_stride"] == logits.stride(0) and call["return_value"] is True
    assert fake.CONFIGS[call["cfg_id"]]["dtype"] == "bf16"
    assert DeepGemmFe.topk.calls == []


def test_narrow_calls_run_the_cudnn_top_k_on_the_compact_full_row_view(dg, monkeypatch):
    fake = _install_deepselect(monkeypatch, min_width=4096)
    q, k, w, layout = _case((300, 84), 1, 0, seed=29)
    radix = DeepGemmFe.topk
    seen = []
    original = radix.indexer_top_k_wrapper

    def spy(input_values, *args, **kwargs):
        seen.append((input_values.shape, input_values.stride(), input_values.data_ptr()))
        return original(input_values, *args, **kwargs)

    radix.indexer_top_k_wrapper = spy
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM_DS, False)
    logits = dg.dg.calls[-1]["logits"]
    stride = logits.stride(0)
    assert fake.calls == []
    assert seen[-1] == ((384, stride), (stride, 1), logits.data_ptr())
    assert all(call["val"] for call in radix.calls)


def test_a_call_deepselect_declines_never_reaches_it_on_the_full_row_view(dg, monkeypatch):
    """The top-k backend is decided once, on the [rows, width] logits DeepSelect would read:
    a decline there sends the call to cuDNN even when the wider full-row view (1024 columns
    for width 400) would pass DeepSelect's width threshold (512 here)."""
    fake = _install_deepselect(monkeypatch, min_width=512, seed=1)
    q, k, w, layout = _case((400,), 1, 0, seed=3)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM_DS, False)
    logits = dg.dg.calls[-1]["logits"]
    assert logits.size(1) == 400 and logits.stride(0) == 1024
    assert fake.calls == [] and len(DeepGemmFe.topk.calls) == 1


def test_indices_only_and_value_calls_run_the_same_top_k(dg, monkeypatch):
    """A checkpoint's no-grad first call and its loss replay must select the same keys."""
    q, k, w, layout = _case((300, 84), 1, 0, seed=31, kind="ternary")
    for install in (False, True):
        if install:
            _install_deepselect(monkeypatch, seed=3)
        options = DEEPGEMM_DS if install else DEEPGEMM
        DeepGemmFe.topk = AnyDtypeRadixTopK(seed=9)
        first, first_len, _ = _score(q, k, w, layout, 1, 0, 16, options, False)
        if install:
            _install_deepselect(monkeypatch, seed=3)
        DeepGemmFe.topk = AnyDtypeRadixTopK(seed=9)
        replay, replay_len, scores = _score(q, k, w, layout, 1, 0, 16, options, True)
        assert torch.equal(first_len, replay_len)
        assert _selected_sets(first, first_len) == _selected_sets(replay, replay_len)
        assert scores.dtype == torch.float32


# --------------------------------------------------------------------------- loss-on payload
def _bf16_reference_log_probs(q, k, w, indices, lengths):
    """FP32 log-softmax over each row's selected keys of the BF16 dense logits."""
    q32, k32, w32 = q[0].float(), k[0].float(), w[0].float()
    dense = (torch.einsum("thd,kd->thk", q32, k32).relu() * w32.unsqueeze(-1)).sum(dim=1)
    valid = torch.arange(indices.size(-1)).view(1, 1, -1) < lengths.unsqueeze(-1)
    gathered = torch.gather(dense, 1, indices[0].clamp(min=0).long()).unsqueeze(0)
    return dsa_masking.masked_log_softmax(gathered, valid), valid


@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 1)])
@pytest.mark.parametrize("backend", ["cudnn", "deepselect"])
def test_loss_scores_are_the_bf16_recompute_over_the_selection(
    dg, monkeypatch, cp_size, cp_rank, backend
):
    if backend == "deepselect":
        _install_deepselect(monkeypatch)
    options = DEEPGEMM_DS if backend == "deepselect" else DEEPGEMM
    q, k, w, layout = _case((256, 128), cp_size, cp_rank, seed=37, uncovered=2)
    topk = 32
    indices, lengths, scores = _score(q, k, w, layout, cp_size, cp_rank, topk, options)

    if backend == "deepselect":
        assert DeepGemmFe.topk.calls == []
    (call,) = DeepGemmFe.recompute_calls
    assert call["q"].dtype == torch.bfloat16 and torch.equal(call["q"], q)
    assert torch.equal(call["k"], k) and torch.equal(call["w"], w)
    assert call["topk_indices"].shape[-1] == 128 and call["topk_length"] is None
    assert call["qhead_per_kv_head"] == HEADS and call["topk_indices_global"] is False
    assert scores.dtype == torch.float32
    want, valid = _bf16_reference_log_probs(q, k, w, indices, lengths)
    got = dsa_masking.masked_log_softmax(scores, valid)
    torch.testing.assert_close(got[valid], want[valid], rtol=0, atol=1e-5)
    assert bool(torch.isfinite(scores[valid]).all())
    assert _TOLIST(lengths[0, -2:]) == [0, 0]


def test_missing_recompute_entry_point_raises_instead_of_changing_the_loss(dg, monkeypatch):
    monkeypatch.delattr(fakes.Fe131DSA, "sparse_indexer_score_recompute_wrapper")
    q, k, w, layout = _case((256, 128), 1, 0)
    with pytest.raises(RuntimeError, match="lacks sparse_indexer_score_recompute_wrapper"):
        _score(q, k, w, layout, 1, 0, 16, DEEPGEMM)
    indices, lengths, scores = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, False)
    assert scores is None


def test_loss_path_gets_the_same_index_score_as_the_bf16_path(dg, monkeypatch):
    """Given the same selection, the KL predict and its gradient input equal the BF16 path's."""
    q, k, w, layout = _case((256, 128), 1, 0, seed=43)
    indices, lengths, payload = _score(q, k, w, layout, 1, 0, 24, DEEPGEMM)
    bf16_logits, valid = _bf16_reference_log_probs(q, k, w, indices, lengths)
    raw = torch.einsum("thd,kd->thk", q[0].float(), k[0].float()).relu()
    raw = (raw * w[0].float().unsqueeze(-1)).sum(dim=1)
    bf16_payload = torch.gather(raw, 1, indices[0].clamp(min=0).long()).unsqueeze(0)
    bf16_payload = bf16_payload.masked_fill(~valid, torch.finfo(torch.float32).min)
    seen = []

    def fake_target(q_attn, k_attn, lse, topk_indices, topk_length, softmax_scale, **kwargs):
        target = torch.ones(topk_indices.shape)
        return target

    def fake_backward(q, w, k, attn_score, index_score, topk_indices, **kwargs):
        seen.append(index_score.clone())
        return torch.zeros_like(q), torch.zeros_like(w), torch.zeros_like(k)

    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_target)
    monkeypatch.setattr(dsa_cudnn_kernels, "_run_sparse_indexer_backward", fake_backward)
    sq = q.size(1)
    common = dict(
        q_idx_bshd=q,
        k_idx_bsd=k,
        w_bsh=w,
        topk_indices_cmp=indices,
        topk_length_cmp=lengths,
        query=torch.randn((sq, 1, 64, 576)),
        kv_full=torch.randn((sq, 1, 576)),
        lse=torch.zeros((sq, 64)),
        softmax_scale=1.0,
        loss_coeff=0.1,
        query_valid_rows=None,
        calculate_per_token_loss=False,
        tp_group=None,
    )
    loss_dg = dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads(
        indexer_score_payload=payload, **common
    )[0]
    loss_bf16 = dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads(
        indexer_score_payload=bf16_payload, **common
    )[0]
    torch.testing.assert_close(loss_dg, loss_bf16, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(seen[0], seen[1], rtol=1e-5, atol=1e-7)
    assert bool(torch.isfinite(bf16_logits[valid]).all())


def test_o1_scores_from_the_first_call_are_accepted_by_the_replay(dg, monkeypatch):
    q, k, w, layout = _case((256, 128), 1, 0, seed=47)
    starts, ends, cu, max_seqlen = layout
    sq = q.size(1)
    config = SimpleNamespace(
        dsa_indexer_precision="mxfp8",
        dsa_indexer_scorer_backend="deepgemm",
        dsa_indexer_skip_masked_score_fill=False,
        kv_lora_rank=512,
    )
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    query = torch.randn((sq, 1, 64, 576))
    key = torch.randn((sq, 1, 1, 576))
    result = dsa_cudnn_kernels.run_fused_qk_topk_with_scores(
        q.permute(1, 0, 2, 3),
        k.permute(1, 0, 2),
        w.permute(1, 0, 2),
        24,
        starts,
        ends,
        64,
        use_local_indexer_varlen=True,
        packed_seq_params=packed,
        query=query,
        key=key,
        config=config,
    )
    assert result is not None
    indices, lengths, scores = result
    assert scores.dtype == torch.float32 and len(DeepGemmFe.recompute_calls) == 1
    applied = {}
    monkeypatch.setattr(
        dsa_cudnn_kernels,
        "SparseIndexerLossFromTopKFunc",
        SimpleNamespace(apply=lambda *args: applied.setdefault("args", args)),
    )
    dsa_cudnn_kernels.run_sparse_indexer_loss_from_topk(
        q.permute(1, 0, 2, 3),
        k.permute(1, 0, 2),
        w.permute(1, 0, 2),
        indices,
        lengths,
        scores,
        query,
        key,
        softmax_scale=1.0,
        loss_coeff=0.1,
        pg_collection=SimpleNamespace(tp=None),
        config=config,
    )
    assert applied["args"][7] is scores


def test_frozen_selection_never_recomputes(dg):
    q, k, w, layout = _case((256, 128), 1, 0)
    _score(q, k, w, layout, 1, 0, 16, DEEPGEMM, False)
    _score(q, k, w, layout, 1, 0, 300, DEEPGEMM, False)
    assert DeepGemmFe.recompute_calls == []


def _fall_back(monkeypatch):
    monkeypatch.setattr(
        dsa_deepgemm_scorer,
        "_device_problem",
        lambda device: "DeepGEMM is not validated on SM 10.0",
    )


@pytest.mark.parametrize("width", ["ranked", "every-key"])
@pytest.mark.parametrize("loss", ["on", "frozen"])
@pytest.mark.parametrize("backend", ["cudnn", "deepgemm", "deepgemm-fallback"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_loss_scores_follow_the_precision_not_the_scorer_backend(
    dg, monkeypatch, precision, backend, loss, width
):
    """Which path supplies the indexer-loss scores, per precision, configured scorer backend
    (``deepgemm-fallback``: DeepGEMM configured, the cuDNN MXFP8 scorer runs), loss on or
    frozen (indices only) and width (``every-key``: K covers every in-row key, the early exit
    of both scorers). The scores are recomputed from BF16 exactly when the precision is MXFP8
    and the loss is on; BF16 keeps the scorer's own values; frozen calls keep none; the
    backend only picks the kernel that selects (and BF16 cannot take DeepGEMM at all)."""
    if backend == "deepgemm-fallback":
        _fall_back(monkeypatch)
    options = dsa_cudnn_kernels.IndexerScorerOptions(
        precision=precision, scorer_backend="cudnn" if backend == "cudnn" else "deepgemm"
    )
    q, k, w, layout = _case((256, 128), 2, 1, seed=53)
    topk = 16 if width == "ranked" else _width(2, 1, layout[3])
    if precision == "bf16" and backend != "cudnn":
        with pytest.raises(ValueError, match="is the MXFP8 selection scorer"):
            _score(q, k, w, layout, 2, 1, topk, options, loss == "on")
        assert DeepGemmFe.calls == [] and dg.dg.calls == []
        assert DeepGemmFe.recompute_calls == []
        return
    if backend == "deepgemm":
        _score(q, k, w, layout, 2, 1, 16, options, False)  # JIT warm-up and a first call
        dg.dg.calls.clear()
    indices, lengths, scores = _score(q, k, w, layout, 2, 1, topk, options, loss == "on")

    deepgemm_ran = backend == "deepgemm"
    assert bool(dg.dg.calls) == (deepgemm_ran and (width == "ranked" or loss == "on"))
    assert [call["precision"] for call in DeepGemmFe.calls] == ([] if deepgemm_ran else [precision])
    assert torch.equal(lengths[0].long(), torch.minimum(layout[1] - layout[0], torch.tensor(topk)))
    if loss == "frozen":
        assert scores is None and DeepGemmFe.recompute_calls == []
        return
    want, valid = _bf16_reference_log_probs(q, k, w, indices, lengths)
    if precision == "bf16":
        assert DeepGemmFe.recompute_calls == []
        dense = (
            torch.einsum("thd,kd->thk", q[0].float(), k[0].float()).relu()
            * w[0].float().unsqueeze(-1)
        ).sum(dim=1)
        own = torch.gather(dense, 1, indices[0].clamp(min=0).long()).unsqueeze(0)
        torch.testing.assert_close(scores[valid], own[valid], rtol=1e-6, atol=1e-6)
        return
    (call,) = DeepGemmFe.recompute_calls
    assert torch.equal(call["q"], q) and torch.equal(call["k"], k) and torch.equal(call["w"], w)
    got = dsa_masking.masked_log_softmax(scores, valid)
    torch.testing.assert_close(got[valid], want[valid], rtol=0, atol=1e-5)


def test_every_key_loss_payload_is_identical_whichever_mxfp8_scorer_ran(dg, monkeypatch):
    """Given one selection (every in-row key), the cuDNN MXFP8 scorer, DeepGEMM and DeepGEMM's
    fallback hand the loss the very same payload."""
    q, k, w, layout = _case((256, 128), 2, 1, seed=57)
    topk = _width(2, 1, layout[3])
    runs = [_score(q, k, w, layout, 2, 1, topk, MXFP8)]
    assert dg.dg.calls == []
    runs.append(_score(q, k, w, layout, 2, 1, topk, DEEPGEMM))
    deepgemm_calls = len(dg.dg.calls)
    assert deepgemm_calls > 0
    _fall_back(monkeypatch)
    runs.append(_score(q, k, w, layout, 2, 1, topk, DEEPGEMM))
    assert len(dg.dg.calls) == deepgemm_calls and len(DeepGemmFe.recompute_calls) == 3
    assert [call["precision"] for call in DeepGemmFe.calls] == ["mxfp8", "mxfp8"]
    for other in runs[1:]:
        for expected, actual in zip(runs[0], other):
            assert torch.equal(actual, expected)


def test_fill_on_fallback_drops_uncovered_rows_with_the_loss_on(dg, monkeypatch):
    """The prefilled -inf columns of uncovered rows stay dropped through the recompute."""
    monkeypatch.setattr(
        dsa_deepgemm_scorer,
        "_device_problem",
        lambda device: "DeepGEMM is not validated on SM 10.0",
    )
    q, k, w, layout = _case((128, 64), 1, 0, seed=59, uncovered=4)
    indices, lengths, scores = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM)
    assert _TOLIST(lengths[0, -4:]) == [0, 0, 0, 0]
    _, mxfp8_lengths, _ = _score(q, k, w, layout, 1, 0, 16, MXFP8)
    assert torch.equal(lengths, mxfp8_lengths)


def test_nan_key_past_the_repaired_window_drops_the_flagged_rows(dg, monkeypatch):
    """DeepSelect flags every row longer than K that sees a NaN key; the repair points it at
    its first K keys with NaN values. With the loss on such a row is dropped (topk_length 0)
    even though the recompute over those K keys would be finite; indices-only keeps them."""
    _install_deepselect(monkeypatch)
    q, k, w, layout = _case((256, 128), 1, 0, seed=61)
    topk = 16
    k[0, 100] = float("nan")
    indices, lengths, _ = _score(q, k, w, layout, 1, 0, topk, DEEPGEMM_DS)
    rows = torch.arange(384)
    flagged = (rows >= 100) & (rows < 256)
    assert bool((lengths[0][flagged] == 0).all())
    assert bool((lengths[0][rows < 16] == rows[rows < 16] + 1).all())
    _install_deepselect(monkeypatch)
    only, only_lengths, _ = _score(q, k, w, layout, 1, 0, topk, DEEPGEMM_DS, False)
    assert bool((only_lengths[0][flagged] == topk).all())
    assert _TOLIST(only[0, 200, :topk]) == list(range(topk))


@pytest.mark.parametrize("nan_bits, kept", [(0x7FC0, 15), (-64, 16)], ids=["+nan", "-nan"])
def test_nan_key_picked_by_the_cudnn_top_k_drops_one_slot(dg, nan_bits, kept):
    """The radix top-k ranks a positive NaN above every number (it takes a slot, which the
    loss path then drops) and a negative NaN below every number (never selected)."""
    dg.dg.nan_bits = nan_bits
    q, k, w, layout = _case((256, 128), 1, 0, seed=67)
    k[0, 150] = float("nan")
    indices, lengths, scores = _score(q, k, w, layout, 1, 0, 16, DEEPGEMM)
    rows = torch.arange(384)
    sees = (rows >= 150) & (rows < 256)
    assert bool((lengths[0][sees] == kept).all())
    unaffected = torch.minimum(layout[1] - layout[0], torch.tensor(16))
    assert torch.equal(lengths[0][~sees].long(), unaffected[~sees])
    assert not bool((indices[0][sees] == 150).any())
    assert bool(torch.isfinite(scores[0][sees][:, :kept]).all())


# -------------------------------------------------------------------------------- host syncs
@pytest.mark.parametrize("chunked", [False, True])
def test_deepgemm_branch_does_not_synchronize_with_the_host(dg, monkeypatch, chunked):
    _install_deepselect(monkeypatch)
    if chunked:
        monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda heads: 200)
    q, k, w, layout = _case((256, 512), 2, 1, seed=71)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("host synchronization in the DeepGEMM scorer path")

    for name in ("item", "tolist", "__bool__", "__int__", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch, "nonzero", forbidden)
    for options in (DEEPGEMM, DEEPGEMM_DS):
        for return_topk_scores in (False, True):
            _score(q, k, w, layout, 2, 1, 16, options, return_topk_scores)


# --------------------------------------------------------------------- unsupported layouts
def test_plain_causal_and_dense_loss_raise_the_mxfp8_error(dg):
    q, k, w = _inputs(64, 64)
    with pytest.raises(RuntimeError, match="plain_causal"):
        dsa_cudnn_kernels._indexer_topk_bshd(
            q, k, w, 8, return_scores=False, return_topk_scores=True, scorer_options=DEEPGEMM
        )
    starts, ends, cu, max_seqlen = _layout((128, 128), 1, 0)
    q, k, w = _inputs(256, 256)
    with pytest.raises(RuntimeError, match="dense indexer loss"):
        dsa_cudnn_kernels._indexer_topk_bshd(
            q,
            k,
            w,
            8,
            varlen_starts=starts,
            varlen_ends=ends,
            return_scores=True,
            use_local_indexer_varlen=True,
            packed_cu_seqlens_q=cu,
            packed_cu_seqlens_k=cu,
            packed_max_seqlen_q=max_seqlen,
            packed_max_seqlen_k=max_seqlen,
            scorer_options=DEEPGEMM,
        )
    assert dg.dg.calls == []


def test_single_sequence_segments_without_metadata_raise(dg):
    starts, ends, _cu, _max = _layout((256,), 2, 0)
    q, k, w = _inputs(128, 256)
    with pytest.raises(RuntimeError, match="packed_single_segments"):
        dsa_cudnn_kernels._indexer_topk_bshd(
            q,
            k,
            w,
            8,
            varlen_starts=starts,
            varlen_ends=ends,
            return_scores=False,
            return_topk_scores=True,
            use_local_indexer_varlen=True,
            single_packed_thd_sequence=True,
            local_packed_cp_query_len=128,
            packed_cp_size=2,
            local_packed_cp_rank=0,
            scorer_options=DEEPGEMM,
        )


@pytest.mark.parametrize(
    "options, message",
    [
        (
            dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8", scorer_backend="flash"),
            "must be one of",
        ),
        (dsa_cudnn_kernels.IndexerScorerOptions(scorer_backend="deepgemm"), "set dsa_indexer"),
    ],
)
def test_scorer_options_are_checked_defensively(dg, options, message):
    q, k, w, layout = _case((128, 128), 1, 0)
    with pytest.raises(ValueError, match=message):
        _score(q, k, w, layout, 1, 0, 8, options)


# -------------------------------------------------------------------------------------- config
def _make_config(monkeypatch, **overrides):
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsa_cudnn_mxfp8_scorer import (  # noqa: E501
        _make_mxfp8_config,
    )

    fields = dict(dsa_indexer_scorer_backend="deepgemm")
    fields.update(overrides)
    return _make_mxfp8_config(monkeypatch, **fields)


@pytest.mark.parametrize("topk_backend", ["cudnn", "deepselect"])
def test_config_accepts_deepgemm_with_mxfp8(monkeypatch, topk_backend):
    config = _make_config(monkeypatch, dsa_indexer_topk_backend=topk_backend)
    options = dsa_cudnn_kernels._scorer_options_from_config(config)
    assert options.scorer_backend == "deepgemm" and options.precision == "mxfp8"
    assert options.topk_backend == topk_backend


def test_config_and_options_default_to_cudnn(monkeypatch):
    from megatron.core.transformer.transformer_config import TransformerConfig

    assert TransformerConfig.__dataclass_fields__["dsa_indexer_scorer_backend"].default == "cudnn"
    assert dsa_cudnn_kernels.IndexerScorerOptions().scorer_backend == "cudnn"
    assert dsa_cudnn_kernels._scorer_options_from_config(None).scorer_backend == "cudnn"
    config = _make_config(monkeypatch, dsa_indexer_scorer_backend="cudnn")
    assert dsa_cudnn_kernels._scorer_options_from_config(config).scorer_backend == "cudnn"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dsa_indexer_precision": "bf16"}, "set dsa_indexer_precision='mxfp8'"),
        ({"dsa_kernel_backend": "none"}, "requires experimental_attention_variant='dsa' with"),
        ({"deterministic_mode": True}, "does not support deterministic_mode"),
        ({"dsa_indexer_kpool": 4, "dsa_indexer_topk": 4}, "KPool selection .* bypasses it"),
        ({"dsa_indexer_scorer_backend": "flashinfer"}, "must be 'cudnn' or 'deepgemm'"),
        ({"attention_backend": AttnBackend.unfused}, "requires the fused DSA kernels"),
    ],
    ids=["bf16", "kernel-backend-none", "deterministic", "kpool", "unknown", "unfused"],
)
def test_config_rejects_what_deepgemm_cannot_honour(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_config(monkeypatch, **overrides)


def test_config_rejects_deepgemm_outside_dsa(monkeypatch):
    from megatron.core import utils as core_utils
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    with pytest.raises(ValueError, match="requires experimental_attention_variant='dsa'"):
        MLATransformerConfig(
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            add_bias_linear=False,
            dsa_indexer_scorer_backend="deepgemm",
        )


_FULL = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
_CORE_ATTN = dict(recompute_granularity="selective", recompute_modules=["core_attn"])


_REPLAYS = [
    (_FULL, "recompute_granularity='full'"),
    (_CORE_ATTN, "selective recompute of 'core_attn'"),
    (dict(recompute_granularity="selective"), "selective recompute of 'core_attn'"),
    (
        dict(recompute_granularity="selective", recompute_modules=["layernorm", "core_attn"]),
        "selective recompute of 'core_attn'",
    ),
]
_REPLAY_IDS = ["full", "core-attn", "default-modules", "layernorm-and-core-attn"]


@pytest.mark.parametrize("topk_backend", ["cudnn", "deepselect"])
@pytest.mark.parametrize("recompute, replay", _REPLAYS, ids=_REPLAY_IDS)
def test_config_rejects_a_loss_replay_that_selects_again(
    monkeypatch, topk_backend, recompute, replay
):
    """cuDNN's BF16 radix top-k (all calls with 'cudnn', the narrow ones with 'deepselect') does
    not repeat its choice among tied BF16 logits (GB300), so a recompute that replays the
    indexer must reuse the first call's top-k."""
    with pytest.raises(ValueError, match="requires dsa_indexer_reuse_checkpoint_topk=True") as e:
        _make_config(monkeypatch, dsa_indexer_topk_backend=topk_backend, **recompute)
    assert replay in str(e.value) and "cuDNN's BF16 radix top-k" in str(e.value)


@pytest.mark.parametrize("topk_backend", ["cudnn", "deepselect"])
@pytest.mark.parametrize("recompute, replay", _REPLAYS, ids=_REPLAY_IDS)
def test_config_rejects_a_frozen_replay_that_selects_again(
    monkeypatch, topk_backend, recompute, replay
):
    """A frozen indexer (loss coefficient 0) replays the dense indexer and top-k without the
    reuse too, and its selection feeds the recomputed attention, so the same rule applies."""
    with pytest.raises(ValueError, match="requires dsa_indexer_reuse_checkpoint_topk=True") as e:
        _make_config(
            monkeypatch,
            dsa_indexer_topk_backend=topk_backend,
            dsa_indexer_loss_coeff=0.0,
            **recompute,
        )
    assert replay in str(e.value) and "loss on or frozen" in str(e.value)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(_FULL, dsa_indexer_reuse_checkpoint_topk=True),
        dict(_CORE_ATTN, dsa_indexer_reuse_checkpoint_topk=True),
        dict(recompute_granularity="selective", recompute_modules=["layernorm"]),
        dict(recompute_granularity="selective", recompute_modules=["layernorm", "mla_up_proj"]),
        dict(_FULL, dsa_indexer_loss_coeff=0.0, dsa_indexer_reuse_checkpoint_topk=True),
        dict(_CORE_ATTN, dsa_indexer_loss_coeff=0.0, dsa_indexer_reuse_checkpoint_topk=True),
        dict(
            recompute_granularity="selective",
            recompute_modules=["layernorm"],
            dsa_indexer_loss_coeff=0.0,
        ),
        dict(_FULL, dsa_indexer_scorer_backend="cudnn"),
        dict(_FULL, dsa_indexer_scorer_backend="cudnn", dsa_indexer_loss_coeff=0.0),
        dict(),
        dict(dsa_indexer_loss_coeff=0.0),
    ],
    ids=[
        "full-reuse",
        "core-attn-reuse",
        "selective-layernorm",
        "selective-layernorm-up-proj",
        "full-frozen-reuse",
        "core-attn-frozen-reuse",
        "selective-layernorm-frozen",
        "full-cudnn-scorer",
        "full-frozen-cudnn-scorer",
        "no-recompute",
        "no-recompute-frozen",
    ],
)
def test_config_accepts_replays_that_reuse_or_never_rerun_the_indexer(monkeypatch, overrides):
    """The formal GLM-5.2 cells: 1M full recompute with the reuse (indexer loss on or frozen),
    4K/128K selective layernorm recompute (the indexer is not replayed), and no recompute."""
    config = _make_config(monkeypatch, dsa_indexer_topk_backend="deepselect", **overrides)
    assert config.dsa_indexer_precision == "mxfp8"
