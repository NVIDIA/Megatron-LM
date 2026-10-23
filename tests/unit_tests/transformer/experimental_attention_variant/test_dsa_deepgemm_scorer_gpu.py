# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-kernel checks of ``dsa_indexer_scorer_backend='deepgemm'`` (single GPU).

Needs a ``deep_gemm >= 2.8`` build, SM 10.3 (SM 10.0 when ``MCORE_DEEPGEMM_VALIDATE_SM100=1``,
which is how a GB200 run validates it), Transformer Engine MXFP8 and the cuDNN frontend DSA
namespace at >= 1.29 (older frontends keep the cuDNN MXFP8 scorer); DeepSelect is used where
installed. CP>1 layouts run on one GPU: the scorer needs only this rank's local query rows and
the globally ordered packed keys. Every DeepGEMM-scored call checks that DeepGEMM really ran
(no logged fallback reason, a logits launch), so a silent fallback cannot pass. Covered:

* G1 the output contract: BF16, 1024-byte rows, a full-row view inside the allocation, values
  against a dequantized reference;
* G2 exact mapping on ternary inputs (bitwise against integer math) for CP1 packs, zigzag CP2
  and CP4 ranks, unequal segments, uncovered tail rows (CP1) and a query block straddling the
  covered boundary; nothing is written outside ``[ks, ke)``, checked on a poisoned block the
  caching allocator must hand back (the test fails if it does not); tie-aware selection
  equality with the FE BF16 scorer;
* G3 Gaussian and outlier selection against FE MXFP8 (overlap) and FE BF16 (recall);
* G4 CUDA graph capture/replay equal to eager (tie-aware on the DeepGEMM logits for cuDNN's
  top-k), no host sync, and the capture guard in a fresh process, raised before any
  deep_gemm call, both by a direct ``mqa_logits`` and by the production entry point
  (``_deepgemm_scorer_fallback_reason``, which reaches ``jit_problem``) with no device claimed
  (the subprocess prints any exception's type and text);
* G5 repeatability, batch independence and indices-only against value calls (a layer
  recomputed without ``dsa_indexer_reuse_checkpoint_topk`` selects again in its replay):
  bitwise for DeepSelect BF16; for cuDNN's BF16 radix top-k a documented negative, tie-aware
  only (it keeps other tied keys on a repeat, which is why the config requires the reuse for
  a recompute that replays the indexer);
* G6 NaN in a K row and a Q row (records whether the logits are NaN and their sign);
* G7, the selected-score recompute against log-softmax of FE BF16 scores, lives in
  ``test_dsa_cudnn_mxfp8_scorer_gpu.py`` with the recompute itself: it is the loss rule of
  every MXFP8 scorer, not of DeepGEMM;
* G8 the sparse indexer loss and gradients equal the BF16 path's given the same selection,
  plus the cost of using DeepGEMM's own BF16 values instead (report only);
* G9 cuDNN's BF16 radix top-k on DeepGEMM-shaped logits at 4K..1M widths: exact, never past a
  row's length (lengths not a multiple of 16), repeatable up to ties (bitwise reported);
* G10 calls of more rows than one DeepGEMM call takes: row chunks select what one call
  selects (bitwise for DeepSelect, tie-aware for cuDNN) and feed the loss the same payload
  (bitwise where the slots match; where cuDNN returned the same keys in another slot order,
  which reorders the recompute's FP32 row sum, within that order's absolute bound on
  ``log p``, ``_log_prob_order_bound``), and 65536 rows with 64 heads (past the 58104-row
  limit and TE's 65535-row MXFP8 cast of Q) run in two calls.

Every tie-aware comparison (equal multisets of selected values) also requires each row's
valid slots to name distinct keys, which a multiset of values cannot show.

Lines starting with ``[deepgemm-gpu]`` report what was measured.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.quantization import indexer_quantization
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_deepgemm_scorer,
    dsa_deepselect_topk,
    dsa_masking,
)
from tests.unit_tests.transformer.experimental_attention_variant.dsa_cudnn_fe_fakes import (
    packed_zigzag_layout,
)

HEADS, HEAD_DIM = 32, 128
_GIB = 1024**3
_K = 2048
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VALIDATE_SM100_ENV = "MCORE_DEEPGEMM_VALIDATE_SM100"
DEEPGEMM = dsa_cudnn_kernels.IndexerScorerOptions(
    skip_masked_score_fill=True, precision="mxfp8", scorer_backend="deepgemm"
)
DEEPGEMM_DS = dsa_cudnn_kernels.IndexerScorerOptions(
    skip_masked_score_fill=True,
    precision="mxfp8",
    scorer_backend="deepgemm",
    topk_backend="deepselect",
)
FE_MXFP8 = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=True, precision="mxfp8")
FE_BF16 = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=True)


def _report(message: str) -> None:
    print(f"[deepgemm-gpu] {message}", flush=True)


@pytest.fixture(autouse=True)
def _validate_sm100_when_asked(monkeypatch):
    """SM 10.0 falls back in production; a GB200 validation run opts in here."""
    if torch.cuda.is_available() and os.environ.get(_VALIDATE_SM100_ENV) == "1":
        sm = tuple(torch.cuda.get_device_capability())
        if sm in dsa_deepgemm_scorer.KNOWN_SM and sm not in dsa_deepgemm_scorer.SUPPORTED_SM:
            monkeypatch.setattr(
                dsa_deepgemm_scorer, "SUPPORTED_SM", dsa_deepgemm_scorer.SUPPORTED_SM | {sm}
            )
            _report(f"validating SM {sm[0]}.{sm[1]}, which production keeps on cuDNN")


class _DeepGemmSpy:
    """Counts DeepGEMM logits launches and the scorer backend outcomes logged per call."""

    def __init__(self, monkeypatch):
        self.launches = 0
        self.reasons = []
        real_logits = dsa_deepgemm_scorer.mqa_logits
        real_log = dsa_deepgemm_scorer.log_scorer_backend_once

        def mqa_logits(*args, **kwargs):
            self.launches += 1
            return real_logits(*args, **kwargs)

        def log_once(reason):
            self.reasons.append(reason)
            return real_log(reason)

        monkeypatch.setattr(dsa_deepgemm_scorer, "mqa_logits", mqa_logits)
        monkeypatch.setattr(dsa_deepgemm_scorer, "log_scorer_backend_once", log_once)


_SPY = None


@pytest.fixture(autouse=True)
def _deepgemm_spy(monkeypatch):
    global _SPY
    _SPY = _DeepGemmSpy(monkeypatch)
    yield _SPY
    _SPY = None


def _require_deepgemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    sm = tuple(torch.cuda.get_device_capability())
    if sm not in dsa_deepgemm_scorer.KNOWN_SM:
        pytest.skip("DeepGEMM's MQA logits run on SM 10.x only")
    if sm not in dsa_deepgemm_scorer.SUPPORTED_SM:
        pytest.skip(f"SM {sm} is not validated; set {_VALIDATE_SM100_ENV}=1 to validate it")
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("cuDNN frontend DSA is not installed")
    dsa_cudnn_kernels._ensure_dsa_namespace()
    if not dsa_cudnn_kernels._cudnn_frontend_at_least(
        dsa_cudnn_kernels._TOPK_OOB_LANE_GUARD_FE_VERSION
    ):
        pytest.skip("cuDNN frontend < 1.29: the DeepGEMM scorer falls back there")
    if not indexer_quantization.HAVE_TE_MXFP8:
        pytest.skip("Transformer Engine MXFP8 quantization is unavailable")
    provider = dsa_deepgemm_scorer._load_provider()
    if provider is None:
        pytest.skip(dsa_deepgemm_scorer._IMPORT_STATE["error"])
    _report(f"provider {provider.description}")
    dsa_deepgemm_scorer.warmup(torch.device("cuda"), HEADS)


def _require_fe_mxfp8():
    if not all(
        dsa_cudnn_kernels._accepts_named_parameter(
            dsa_cudnn_kernels._cudnn_dsa.indexer_forward_wrapper, name
        )
        for name in dsa_cudnn_kernels._MXFP8_SCORER_WRAPPER_PARAMETERS
    ):
        pytest.skip("cuDNN frontend lacks the MXFP8 dense indexer scorer (needs >= 1.29)")
    if not indexer_quantization.HAVE_TRITON:
        pytest.skip("Triton is unavailable for the FE MXFP8 scale packing")


def _with_uncovered(layout, uncovered):
    starts, ends, cu, max_seqlen = layout
    if not uncovered:
        return layout
    total = int(cu[-1])
    positions = total + torch.arange(uncovered, device="cuda")
    starts = torch.cat((starts, torch.full((uncovered,), total, device="cuda")))
    return starts, torch.cat((ends, positions + 1)), cu, max_seqlen


def _layout(seq_lens, cp_size, cp_rank, max_num_seqs=None, uncovered=0):
    layout = packed_zigzag_layout(seq_lens, cp_size, cp_rank, max_num_seqs, device="cuda")
    return _with_uncovered(layout, uncovered)


def _inputs(sq, sk, seed, kind="gauss", heads=HEADS):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "ternary":
        q = torch.randint(-1, 2, (1, sq, heads, HEAD_DIM), generator=generator, device="cuda")
        k = torch.randint(-1, 2, (1, sk, HEAD_DIM), generator=generator, device="cuda")
        w = torch.zeros((1, sq, heads), device="cuda")
        rows = torch.arange(sq, device="cuda")
        w[0, rows, rows % heads] = 1
        return q.bfloat16(), k.bfloat16(), w.bfloat16()
    q = torch.randn((1, sq, heads, HEAD_DIM), generator=generator, device="cuda")
    k = torch.randn((1, sk, HEAD_DIM), generator=generator, device="cuda")
    if kind == "outlier":
        channels = torch.randperm(HEAD_DIM, generator=generator, device="cuda")[:4]
        q[..., channels] *= 16.0
        k[..., channels] *= 16.0
    w = torch.randn((1, sq, heads), generator=generator, device="cuda") * heads**-0.5
    return q.bfloat16(), k.bfloat16(), w.bfloat16()


def _score(q, k, w, layout, cp_size, cp_rank, topk, options, return_topk_scores=True):
    """``_indexer_topk_bshd``; with the DeepGEMM backend, also checks that DeepGEMM ran."""
    starts, ends, cu, max_seqlen = layout
    launches, logged = _SPY.launches, len(_SPY.reasons)
    result = _indexer_topk_bshd(
        q, k, w, topk, starts, ends, cu, max_seqlen, cp_size, cp_rank, options, return_topk_scores
    )
    if options.scorer_backend == "deepgemm":
        reasons = _SPY.reasons[logged:]
        assert reasons and set(reasons) == {None}, f"DeepGEMM fell back: {reasons}"
        width = dsa_cudnn_kernels._packed_thd_score_width(
            cp_size=cp_size, cp_rank=cp_rank, packed_max_seqlen_k=max_seqlen
        )
        if topk < width or return_topk_scores:
            assert _SPY.launches > launches, "no DeepGEMM logits launch"
    return result


def _indexer_topk_bshd(
    q, k, w, topk, starts, ends, cu, max_seqlen, cp_size, cp_rank, options, return_topk_scores
):
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


def _dense_reference(q, k, w):
    """FP32 ``[sq, sk]`` logits of the BF16 inputs (exact for ternary inputs)."""
    dense = torch.einsum("thd,kd->thk", q[0].float(), k[0].float()).relu()
    return (dense * w[0].float().unsqueeze(-1)).sum(dim=1)


def _assert_distinct_keys(indices, valid_slots, what="a tie-aware comparison"):
    """Each row's valid slots (``slot < valid_slots[row]``, index >= 0) name distinct keys.

    A tie-aware comparison compares the selected values as a multiset, which a top-k that took
    one tied key twice instead of two tied keys would pass; this check does not.
    """
    indices = indices.reshape(-1, indices.size(-1))
    slots = torch.arange(indices.size(-1), device=indices.device).view(1, -1)
    valid = (slots < valid_slots.reshape(-1, 1).long()) & (indices >= 0)
    keys = torch.where(valid, indices.long(), -1).sort(dim=-1).values
    repeated = ((keys[:, 1:] == keys[:, :-1]) & (keys[:, 1:] >= 0)).any(dim=-1)
    assert not bool(repeated.any()), f"{what}: {int(repeated.sum())} rows repeat a key"


def _tie_aware_values(indices, lengths, dense):
    """Per row: the sorted reference values of the selected keys."""
    _assert_distinct_keys(indices, lengths[0])
    rows = []
    for row, n in enumerate(lengths[0].tolist()):
        keys = indices[0, row, :n].long()
        rows.append(torch.sort(dense[row, keys]).values)
    return rows


def _tie_aware_local_values(indices, lengths, logits, ks):
    """Per row: the sorted DeepGEMM logits (``[rows, width]``, column = key - ks) selected."""
    _assert_distinct_keys(indices, lengths[0])
    rows = []
    for row, n in enumerate(lengths[0].tolist()):
        columns = indices[0, row, :n].long() - ks[row].long()
        rows.append(torch.sort(logits[row, columns]).values)
    return rows


def _selected_logits(indices, lengths, logits, ks):
    """``[rows, K]`` sorted FP32 logits of each row's selected keys (``-inf`` past its length).

    Two exact top-K selections of the same logits have equal results iff they differ only in
    which keys tied at the row's K-th value were kept (a tie-aware comparison), each row's keys
    being distinct.
    """
    _assert_distinct_keys(indices, lengths[0])
    columns = indices[0].long() - ks.long().view(-1, 1)
    valid = torch.arange(indices.size(-1), device="cuda").view(1, -1) < lengths[0].view(-1, 1)
    values = torch.gather(logits.float(), 1, columns.clamp(min=0, max=logits.size(1) - 1))
    return torch.where(valid, values, float("-inf")).sort(dim=-1).values


def _fp32_ulp_distance(actual, expected):
    """Elementwise count of FP32 steps between two equally shaped float tensors (int64).

    Maps each FP32 bit pattern to a signed integer that is monotonic in the value (``+0`` and
    ``-0`` both to 0), so the difference counts the representable values between the two.
    """

    def ordered(values):
        bits = values.float().contiguous().view(torch.int32).long()
        return torch.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    return (ordered(actual) - ordered(expected)).abs()


def _fp32_ulp(values):
    """Spacing of FP32 numbers at ``|values|``: ``2**(floor(log2|v|) - 23)``, and ``2**-149``
    below ``2**-126``. Built from the exponent bits, so it is exact on any device."""
    magnitude = values.float().abs()
    power = (magnitude.view(torch.int32) & 0x7F800000).view(torch.float32)
    return torch.where(
        magnitude < 2.0**-126, torch.full_like(magnitude, 2.0**-149), power * 2.0**-23
    )


def _log_prob_order_bound(actual, expected):
    """Elementwise bound on ``|actual - expected|`` for two selected-score payloads of the same
    keys whose slot orders differ (``log(max(p, FLT_MIN))`` from the cuDNN FE 1.31 recompute).

    Error model (sparse_score_recompute_sm100, indexer epilogue): each slot's logit ``s`` and the
    row max come from its own key and ``fmax``, so ``e = exp2((s - max) log2 e)`` is bitwise the
    same in any slot order; only the FP32 row sum of the ``e`` (all positive) follows it: 16
    slots per thread in slot order, a 32-lane butterfly, then 4 warp sums in turn, so each term
    passes at most 15 + 5 + 3 = 23 roundings and two orders differ by at most 2 gamma_23 = 46 u
    of the sum (u = 2**-24). Then ``p = e * (1 / sum)``: at most 2 u per run for the reciprocal
    and u for the product. So the exact logs of the two runs' ``p`` differ by at most 52 u
    (first order; 64 u = 2**-18 is used) whatever ``log p`` is, the FLT_MIN clamp only shrinks
    that, and ``torch.log`` (CUDA ``logf``, at most 1 ulp) adds at most one FP32 ulp of each
    result. A chunking defect (another row's keys, query or sum) moves ``log p`` by about
    1e-3 or more.
    """
    larger = torch.maximum(actual.abs(), expected.abs())
    return 2.0**-18 + 2 * _fp32_ulp(larger)


def _sorted_keys(indices):
    """``[1, rows, K]`` selected keys sorted per row (``-1`` padding last), for bitwise checks."""
    keys = torch.where(indices >= 0, indices, torch.iinfo(torch.int32).max).sort(dim=-1).values
    return torch.where(keys == torch.iinfo(torch.int32).max, -1, keys)


def _packed_logits(q, k, w, layout, cp_size, cp_rank):
    """One call's DeepGEMM logits, row lengths and ``ks`` for a packed layout."""
    starts, ends, cu, max_seqlen = layout
    width = dsa_cudnn_kernels._packed_thd_score_width(
        cp_size=cp_size, cp_rank=cp_rank, packed_max_seqlen_k=max_seqlen
    )
    lengths = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=cp_size, num_rows=starts.numel(), width=width
    )
    ks = starts.to(torch.int32).contiguous()
    logits = dsa_cudnn_kernels._deepgemm_packed_thd_logits(
        q, k.unsqueeze(2), w, ks, (ks + lengths).contiguous(), width
    )
    return logits, lengths, ks


# ---------------------------------------------------------------------------------------- G1
def test_logits_contract_and_full_row_view():
    _require_deepgemm()
    rows, keys = 1003, 5000
    generator = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn((rows, HEADS, HEAD_DIM), generator=generator, device="cuda").bfloat16()
    k = torch.randn((keys, HEAD_DIM), generator=generator, device="cuda").bfloat16()
    w = (torch.randn((rows, HEADS), generator=generator, device="cuda") / 6).bfloat16()
    ks = torch.randint(0, 2000, (rows,), generator=generator, device="cuda").to(torch.int32)
    lengths = torch.randint(0, 3001, (rows,), generator=generator, device="cuda").to(torch.int32)
    ke = (ks + lengths).contiguous()
    q8, qs = indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0(q)
    k8, kscale = indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0(k)
    logits = dsa_deepgemm_scorer.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=3000)
    assert logits.dtype == torch.bfloat16 and logits.shape == (rows, 3000)
    assert (logits.stride(0) * 2) % 1024 == 0 and logits.stride(1) == 1
    full = dsa_deepgemm_scorer.full_row_view(logits)
    assert full.shape == (rows, logits.stride(0)) and full.is_contiguous()
    assert full.data_ptr() == logits.data_ptr()

    def dequant(data, scales):
        shifts = torch.tensor([0, 8, 16, 24], device="cuda", dtype=torch.int32)
        exponent = ((scales.unsqueeze(-1) >> shifts) & 0xFF).float() - 127
        groups = data.float().view(*data.shape[:-1], 4, 32) * torch.exp2(exponent).unsqueeze(-1)
        return groups.view(data.shape)

    dense = torch.einsum("thd,kd->thk", dequant(q8, qs), dequant(k8, kscale)).relu()
    dense = (dense * w.float().unsqueeze(-1)).sum(dim=1)
    columns = torch.arange(3000, device="cuda").view(1, -1)
    valid = columns < lengths.view(-1, 1).long()
    reference = torch.gather(dense, 1, (ks.long().view(-1, 1) + columns).clamp(max=keys - 1))
    error = (logits.float() - reference).abs()[valid].max() / reference[valid].abs().max()
    _report(f"G1 max |logit - dequantized reference| / max |reference| = {error.item():.3e}")
    assert error.item() < 2e-2


# ---------------------------------------------------------------------------------------- G2
MAPPING_LAYOUTS = [
    pytest.param((2048, 1024, 1024), 1, 0, 5, 0, id="cp1-3seq-padded"),
    pytest.param((2000, 1000), 1, 0, None, 22, id="cp1-uncovered-straddling"),
    pytest.param((4096, 4096), 2, 0, 3, 0, id="cp2-rank0"),
    pytest.param((4096, 4096), 2, 1, 3, 0, id="cp2-rank1"),
    pytest.param((6144, 2048), 2, 1, None, 0, id="cp2-unequal"),
    pytest.param((8192, 8192), 4, 0, 3, 0, id="cp4-rank0"),
    pytest.param((8192, 8192), 4, 3, 3, 0, id="cp4-rank3"),
    pytest.param((8192,), 4, 2, 2, 0, id="cp4-one-seq-dummy-entry"),
    # Rows past cu[-1] at CP>1: only the DeepGEMM path takes them (the cuDNN CP>1 segment
    # layout needs the sequences to fill the keys), so there is no FE selection to compare.
    pytest.param((4096, 4096), 2, 1, 3, 6, id="cp2-uncovered-deepgemm-only"),
]


@pytest.mark.parametrize("seq_lens, cp_size, cp_rank, max_num_seqs, uncovered", MAPPING_LAYOUTS)
def test_exact_mapping_on_ternary_inputs(seq_lens, cp_size, cp_rank, max_num_seqs, uncovered):
    _require_deepgemm()
    layout = _layout(seq_lens, cp_size, cp_rank, max_num_seqs, uncovered)
    starts, ends, cu, max_seqlen = layout
    sq = starts.numel()
    sk = sq * cp_size
    q, k, w = _inputs(sq, sk, seed=sq + cp_rank, kind="ternary")
    width = dsa_cudnn_kernels._packed_thd_score_width(
        cp_size=cp_size, cp_rank=cp_rank, packed_max_seqlen_k=max_seqlen
    )
    lengths = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=cp_size, num_rows=sq, width=width
    )
    ks = starts.clamp(min=0, max=sk).to(torch.int32).contiguous()
    ke = (ks + lengths).contiguous()
    # Quantize first, so that the logits are the next large allocation after the poison.
    q8, q_scale = dsa_cudnn_kernels._deepgemm_quantized_q(q[0])
    k8, k_scale = dsa_cudnn_kernels._deepgemm_quantized_k(k.unsqueeze(2))
    w_rows = w[0].contiguous()

    def call():
        return dsa_deepgemm_scorer.mqa_logits(
            q8, q_scale, k8, k_scale, w_rows, ks, ke, max_seqlen_k=width
        )

    # Poison, failing closed: learn the size of DeepGEMM's logits allocation, hand the caching
    # allocator a block of exactly that size filled with a sentinel, and require DeepGEMM to
    # get that block back (nothing else allocates a large block in between).
    nbytes = call().untyped_storage().nbytes()
    poison_bits = 0x7F7F
    poison = torch.full((nbytes // 2,), poison_bits, dtype=torch.int16, device="cuda")
    poison_ptr = poison.data_ptr()
    del poison
    logits = call()
    assert logits.untyped_storage().data_ptr() == poison_ptr, (
        "the caching allocator did not hand DeepGEMM the poisoned block: the check that "
        "nothing is written past [ks, ke) did not run"
    )
    dense = _dense_reference(q, k, w)
    columns = torch.arange(width, device="cuda").view(1, -1)
    valid = columns < lengths.view(-1, 1).long()
    keys = (ks.long().view(-1, 1) + columns).clamp(max=sk - 1)
    reference = torch.gather(dense, 1, keys)
    assert torch.equal(logits.float()[valid], reference[valid]), "not bit-exact on ternary inputs"
    shifted = torch.gather(dense, 1, (keys + 1).clamp(max=sk - 1))
    assert not torch.equal(logits.float()[valid], shifted[valid]), "negative control"
    full = dsa_deepgemm_scorer.full_row_view(logits).view(torch.int16)
    tail = torch.arange(full.size(1), device="cuda").view(1, -1) >= lengths.view(-1, 1).long()
    assert bool(tail.any()) and bool(
        (full[tail] == poison_bits).all()
    ), "a column past [ks, ke) was written"
    if uncovered:
        assert int(lengths[sq - uncovered :].sum()) == 0

    # With the scores kept, both paths drop non-finite selections, so the prefilled FE path
    # (a frontend without clean_logits) also gives uncovered rows no key.
    topk = 512
    dg_indices, dg_lengths, _ = _score(q, k, w, layout, cp_size, cp_rank, topk, DEEPGEMM)
    assert torch.equal(dg_lengths[0].long(), lengths.long().clamp(max=topk))
    if cp_size > 1 and uncovered:
        _report(
            f"G2 {seq_lens} cp{cp_size}/{cp_rank} uncovered={uncovered}: bit-exact, "
            f"{int(tail.sum())} poisoned columns past [ks, ke) untouched"
        )
        return
    fe_indices, fe_lengths, _ = _score(q, k, w, layout, cp_size, cp_rank, topk, FE_BF16)
    assert torch.equal(dg_lengths, fe_lengths)
    for got, want in zip(
        _tie_aware_values(dg_indices, dg_lengths, dense),
        _tie_aware_values(fe_indices, fe_lengths, dense),
    ):
        assert torch.equal(got, want)
    _report(
        f"G2 {seq_lens} cp{cp_size}/{cp_rank} uncovered={uncovered}: bit-exact, "
        f"{int(tail.sum())} poisoned columns past [ks, ke) untouched"
    )


# ---------------------------------------------------------------------------------------- G3
def _overlap(a_indices, a_lengths, b_indices, b_lengths, num_keys):
    rows = a_indices.shape[1]
    member = torch.zeros((rows, num_keys + 1), dtype=torch.bool, device="cuda")
    a = a_indices[0].long()
    a_valid = torch.arange(a.size(-1), device="cuda").view(1, -1) < a_lengths[0].view(-1, 1)
    member.scatter_(1, torch.where(a_valid, a, num_keys), True)
    member[:, num_keys] = False
    b = b_indices[0].long()
    b_valid = torch.arange(b.size(-1), device="cuda").view(1, -1) < b_lengths[0].view(-1, 1)
    hits = (torch.gather(member, 1, torch.where(b_valid, b, num_keys)) & b_valid).sum(dim=1)
    sizes = a_lengths[0]
    keep = sizes > 0
    return hits[keep].float() / sizes[keep].float()


@pytest.mark.parametrize("kind", ["gauss", "outlier"])
def test_selection_overlaps_fe_mxfp8_and_keeps_its_recall(kind):
    _require_deepgemm()
    _require_fe_mxfp8()
    layout = _layout((8192, 4096, 4096), 1, 0, 5)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq, seed=3, kind=kind)
    dg = _score(q, k, w, layout, 1, 0, _K, DEEPGEMM, False)
    fe8 = _score(q, k, w, layout, 1, 0, _K, FE_MXFP8, False)
    bf16 = _score(q, k, w, layout, 1, 0, _K, FE_BF16, False)
    overlap = _overlap(fe8[0], fe8[1], dg[0], dg[1], sq)
    recall_dg = _overlap(bf16[0], bf16[1], dg[0], dg[1], sq).mean().item()
    recall_fe8 = _overlap(bf16[0], bf16[1], fe8[0], fe8[1], sq).mean().item()
    _report(
        f"G3 {kind}: overlap with FE MXFP8 mean {overlap.mean().item():.4f} min "
        f"{overlap.min().item():.4f}; recall vs FE BF16 deepgemm {recall_dg:.4f} fe-mxfp8 "
        f"{recall_fe8:.4f}"
    )
    assert torch.equal(dg[1], fe8[1])
    assert overlap.mean().item() >= 0.98
    assert recall_dg >= recall_fe8 - 0.002


# ---------------------------------------------------------------------------------------- G4
@pytest.mark.parametrize("options", [DEEPGEMM, DEEPGEMM_DS], ids=["cudnn-topk", "deepselect"])
def test_graph_capture_replay_equals_eager(options):
    _require_deepgemm()
    if options.topk_backend == "deepselect" and dsa_deepselect_topk._load_provider() is None:
        pytest.skip("DeepSelect is not installed")
    # 128K CP32 rank 0: 4096 rows, 131072-wide rows (DeepSelect's BF16 range).
    layout = _layout((131072,), 32, 0, 2)
    sq = layout[0].numel()
    sk = sq * 32
    q, k, w = _inputs(sq, sk, seed=5)
    static = [t.clone() for t in (q, k, w)]

    def call():
        return _score(*static, layout, 32, 0, _K, options, return_topk_scores=False)

    starts, ends, cu, max_seqlen = layout
    width = dsa_cudnn_kernels._packed_thd_score_width(
        cp_size=32, cp_rank=0, packed_max_seqlen_k=max_seqlen
    )
    lengths = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=32, num_rows=sq, width=width
    )
    ks = starts.to(torch.int32).contiguous()
    ke = (ks + lengths).contiguous()

    def same_selection(actual, expected):
        assert torch.equal(actual[1], expected[1])
        if options.topk_backend == "deepselect":
            assert torch.equal(actual[0], expected[0])
        else:  # cuDNN's tie_break=0 may keep other equal BF16 logits at the cutoff
            # Compare the logits the keys were selected on: DeepGEMM's, column = key - ks.
            logits = dsa_cudnn_kernels._deepgemm_packed_thd_logits(
                static[0], static[1].unsqueeze(2), static[2], ks, ke, width
            ).float()
            for got, want in zip(
                _tie_aware_local_values(actual[0], actual[1], logits, ks),
                _tie_aware_local_values(expected[0], expected[1], logits, ks),
            ):
                assert torch.equal(got, want)

    call()  # first-use imports and kernel compiles stay out of the sync check
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = call()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    torch.cuda.synchronize()
    same_selection(captured, eager)
    new = _inputs(sq, sk, seed=6)
    for target, source in zip(static, new):
        target.copy_(source)
    graph.replay()
    torch.cuda.synchronize()
    same_selection(captured, call())
    _report(f"G4 {options.topk_backend}: replay after a data swap equals eager")


_FRESH_PROCESS_CAPTURES = {
    # A direct mqa_logits call on quantized inputs.
    "mqa_logits": """
        q8, qs = iq.quantize_indexer_mxfp8_packed_ue8m0(q)
        k8, kscale = iq.quantize_indexer_mxfp8_packed_ue8m0(k)
        ks = torch.zeros(128, dtype=torch.int32, device="cuda")
        ke = torch.arange(1, 129, dtype=torch.int32, device="cuda")

        def captured():
            dg.mqa_logits(q8, qs, k8, kscale, w, ks, ke, max_seqlen_k=256)
        """,
    # The production entry point: the packed THD path asks _deepgemm_scorer_fallback_reason,
    # which runs the static checks and then jit_problem, before it scores.
    "fallback_reason": """
        from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

        dsa_cudnn_kernels._ensure_dsa_namespace()

        def captured():
            dsa_cudnn_kernels._deepgemm_scorer_fallback_reason(
                q.view(1, 128, 32, 128),
                k.view(1, 256, 1, 128),
                w.view(1, 128, 32),
                deterministic=False,
            )
        """,
}


@pytest.mark.parametrize("entry", sorted(_FRESH_PROCESS_CAPTURES))
def test_capture_without_warmup_raises_the_guard_in_a_fresh_process(entry):
    """A cold capture raises the guard before any deep_gemm call and claims no device."""
    _require_deepgemm()
    script = (
        textwrap.dedent("""
        import torch
        from megatron.core.quantization import indexer_quantization as iq
        from megatron.core.transformer.experimental_attention_variant import (
            dsa_deepgemm_scorer as dg,
        )

        q = torch.randn(128, 32, 128, device="cuda").bfloat16()
        k = torch.randn(256, 128, device="cuda").bfloat16()
        w = torch.randn(128, 32, device="cuda").bfloat16()
        """)
        + textwrap.dedent(_FRESH_PROCESS_CAPTURES[entry])
        + textwrap.dedent("""
        assert dg._load_provider() is not None
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream), torch.cuda.graph(graph):
                captured()
        except Exception as exc:  # any exception: print its type and text
            print("GUARD:", dg.CAPTURE_GUARD_MESSAGE in str(exc))
            print("EXCEPTION:", type(exc).__name__, repr(str(exc)[:4000]))
            if exc.__context__ is not None:
                context = exc.__context__
                print("CONTEXT:", type(context).__name__, repr(str(context)[:4000]))
        else:
            print("GUARD: missing")
        # The guard fires before any deep_gemm call, so no device was claimed for DeepGEMM.
        print("DEVICE CLAIMED:", dg._PROCESS_DEVICE)
        """)
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                filter(None, (str(_REPO_ROOT), os.environ.get("PYTHONPATH")))
            ),
        ),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = f"rc={completed.returncode}\n{completed.stdout}\n{completed.stderr[-4000:]}"
    _report(
        f"G4 fresh-process guard ({entry}): " + " | ".join(completed.stdout.strip().splitlines())
    )
    assert "GUARD: True" in completed.stdout, output
    assert "DEVICE CLAIMED: {}" in completed.stdout, output


# ---------------------------------------------------------------------------------------- G5
@pytest.mark.parametrize("topk_backend", ["cudnn", "deepselect"])
def test_selection_is_repeatable_batch_independent_and_call_independent(topk_backend):
    """A layer recomputed without ``dsa_indexer_reuse_checkpoint_topk`` selects again in its
    replay, which must repeat the first call's selection.

    DeepSelect BF16 must repeat bitwise: three repeats, the value call against the
    indices-only call, and two 1024-row slices (rows no longer than K, and rows of ~129K keys)
    against the full batch.

    cuDNN's BF16 radix top-k is a documented negative (GB300, cuDNN frontend 1.31, Slurm job
    317672): on these tie-heavy logits a repeat keeps other tied keys. Only the tie-aware
    contract is asserted for it (equal lengths and equal selected-logit multisets, so keys
    differ only among those equal to the row's K-th value) and whether the repeats were
    bitwise is reported. Because it is not repeatable, and it takes every call DeepSelect
    declines, the config rejects ``deepgemm`` with a recompute that replays the indexer (indexer
    loss on or frozen) unless ``dsa_indexer_reuse_checkpoint_topk`` is set.
    """
    _require_deepgemm()
    if topk_backend == "deepselect" and dsa_deepselect_topk._load_provider() is None:
        pytest.skip("DeepSelect is not installed")
    options = DEEPGEMM_DS if topk_backend == "deepselect" else DEEPGEMM
    layout = _layout((131072,), 32, 0, 2)
    sq = layout[0].numel()
    # Tie-heavy: small-integer indexer inputs give many equal BF16 logits at the cutoff.
    q, k, w = _inputs(sq, sq * 32, seed=7, kind="ternary")
    logits, lengths, ks = _packed_logits(q, k, w, layout, 32, 0)
    exact = topk_backend == "deepselect"

    def same_selection(actual, expected, what):
        """Bitwise for DeepSelect; tie-aware for cuDNN (returns whether it was bitwise)."""
        assert torch.equal(actual[1], expected[1]), f"{what}: lengths differ"
        bitwise = torch.equal(_sorted_keys(actual[0]), _sorted_keys(expected[0]))
        if exact:
            assert bitwise, f"{what}: DeepSelect selected other keys"
        else:
            assert torch.equal(
                _selected_logits(actual[0], actual[1], logits, ks),
                _selected_logits(expected[0], expected[1], logits, ks),
            ), f"{what}: cuDNN selected keys that are not ties at the cutoff"
        return bitwise

    first = _score(q, k, w, layout, 32, 0, _K, options, return_topk_scores=False)
    repeats = [
        same_selection(
            _score(q, k, w, layout, 32, 0, _K, options, return_topk_scores=False), first, "repeat"
        )
        for _ in range(3)
    ]
    with_scores = _score(q, k, w, layout, 32, 0, _K, options, return_topk_scores=True)
    value_call = same_selection(with_scores, first, "value call")
    # Batch independence: the logits of a row subset select what the full batch selects, on
    # rows of the front segment (no longer than K) and of the back segment (~129K keys).
    full = dsa_deepgemm_scorer.full_row_view(logits)
    whole = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        full, lengths, _K, True, topk_backend=topk_backend
    )

    def local_values(chosen, rows):
        _assert_distinct_keys(chosen, lengths[rows].clamp(max=_K))
        chosen = chosen.long()
        values = torch.gather(full[rows].float(), 1, chosen.clamp(min=0))
        return torch.where(chosen >= 0, values, float("-inf")).sort(dim=-1).values

    batch = []
    for rows in (slice(1024, 2048), slice(2048, 3072)):
        part = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
            full[rows], lengths[rows].contiguous(), _K, True, topk_backend=topk_backend
        )
        batch.append(
            torch.equal(
                part["indices"].sort(dim=-1).values, whole["indices"][rows].sort(dim=-1).values
            )
        )
        if exact:
            assert batch[-1], "a row subset selected other keys than the full batch"
        else:
            assert torch.equal(
                local_values(part["indices"], rows), local_values(whole["indices"][rows], rows)
            ), "a row subset selected keys that are not ties at the cutoff"
    _report(
        f"G5 {topk_backend}: repeats bitwise {repeats}, value call bitwise {value_call}, "
        f"row subsets bitwise {batch}" + ("" if exact else " (tie-aware equality asserted)")
    )


# ---------------------------------------------------------------------------------------- G6
def test_nan_rows_follow_the_documented_outcomes(monkeypatch):
    _require_deepgemm()
    # Let DeepSelect take these 8192-wide BF16 rows too (its BF16 minimum is 16384).
    monkeypatch.setitem(
        dsa_deepselect_topk.SCORE_WIDTH_TABLES_BY_DTYPE,
        torch.bfloat16,
        ({"ds_tune": 4096, "deep_select": 4096}, {}),
    )
    layout = _layout((8192, 8192), 1, 0, 3)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq, seed=9)
    k[0, 5000] = float("nan")  # >= K columns past the start of sequence 0
    q[0, 9000] = float("nan")  # one query row of sequence 1
    starts, ends, cu, max_seqlen = layout
    lengths = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=1, num_rows=sq, width=max_seqlen
    )
    ks = starts.to(torch.int32).contiguous()
    logits = dsa_cudnn_kernels._deepgemm_packed_thd_logits(
        q, k.unsqueeze(2), w, ks, (ks + lengths).contiguous(), max_seqlen
    )
    key_logit = logits[6000, 5000]
    query_logits = logits[9000, : int(lengths[9000])]
    key_nan, query_nan = bool(torch.isnan(key_logit)), bool(torch.isnan(query_logits).all())
    sign = int(key_logit.view(torch.int16)) & 0x8000 if key_nan else None
    _report(f"G6 NaN K row -> NaN logit: {key_nan} (sign bit {sign}); NaN Q row: {query_nan}")
    for options in (DEEPGEMM, DEEPGEMM_DS):
        if options.topk_backend == "deepselect" and dsa_deepselect_topk._load_provider() is None:
            continue
        loss_on = _score(q, k, w, layout, 1, 0, _K, options, return_topk_scores=True)
        indices_only = _score(q, k, w, layout, 1, 0, _K, options, return_topk_scores=False)
        assert bool(torch.isfinite(loss_on[2][loss_on[0] >= 0]).all())
        if query_nan:
            assert int(loss_on[1][0, 9000]) == 0
        if key_nan:
            sees = torch.zeros(sq, dtype=torch.bool, device="cuda")
            sees[5000:8192] = True
            if options.topk_backend == "deepselect":
                # Flagged rows: dropped with the loss on, the first K keys without it.
                assert bool((loss_on[1][0][sees] == 0).all())
                first = torch.arange(_K, device="cuda", dtype=torch.int32)
                assert torch.equal(indices_only[0][0, 6000], first)
            else:
                assert bool((loss_on[1][0][sees] >= _K - 1).all())
                assert not bool((loss_on[0][0][sees] == 5000).any())
        _report(f"G6 {options.topk_backend}: outcomes as documented")


# ---------------------------------------------------------------------------------------- G8
def _loss_and_grads(q, k, w, query, key, layout, scorer_options, topk_override=None):
    starts, ends, cu, max_seqlen = layout
    config = SimpleNamespace(
        kv_lora_rank=512,
        deterministic_mode=False,
        dsa_indexer_precision=scorer_options.precision,
        dsa_indexer_scorer_backend=scorer_options.scorer_backend,
        dsa_indexer_topk_backend=scorer_options.topk_backend,
        dsa_indexer_skip_masked_score_fill=True,
    )
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    q_sb = q.permute(1, 0, 2, 3).contiguous().requires_grad_(True)
    k_sb = k.permute(1, 0, 2).contiguous().requires_grad_(True)
    w_sb = w.permute(1, 0, 2).contiguous().requires_grad_(True)
    real = dsa_cudnn_kernels._indexer_topk_bshd
    seen = {}

    def topk(*args, **kwargs):
        result = real(*args, **kwargs) if topk_override is None else topk_override(*args, **kwargs)
        seen["result"] = result
        return result

    dsa_cudnn_kernels._indexer_topk_bshd = topk
    try:
        result = dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
            q_sb,
            k_sb,
            w_sb,
            _K,
            starts.to(torch.int32),
            ends.to(torch.int32),
            block_size=8192,
            query=query,
            key=key,
            softmax_scale=576**-0.5,
            loss_coeff=1.0,
            pg_collection=SimpleNamespace(tp=None),
            config=config,
            use_local_indexer_varlen=True,
            packed_seq_params=packed,
        )
    finally:
        dsa_cudnn_kernels._indexer_topk_bshd = real
    if result is None:
        pytest.skip("the fused sparse indexer loss declined this layout (FlashMLA support)")
    loss = result[2]
    loss.backward()
    return loss.detach(), (q_sb.grad, k_sb.grad, w_sb.grad), seen["result"]


def test_loss_and_gradients_equal_the_bf16_path_given_the_selection():
    _require_deepgemm()
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        pytest.skip("FlashMLA is required for the sparse indexer loss")
    layout = _layout((4096, 2048, 2048), 1, 0, 5)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq, seed=13)
    generator = torch.Generator(device="cuda").manual_seed(5)
    query = torch.randn((sq, 1, 64, 576), generator=generator, device="cuda").bfloat16()
    key = torch.randn((sq, 1, 1, 576), generator=generator, device="cuda").bfloat16()
    launches, logged = _SPY.launches, len(_SPY.reasons)
    dg_loss, dg_grads, (indices, lengths, payload) = _loss_and_grads(
        q, k, w, query, key, layout, DEEPGEMM
    )
    assert set(_SPY.reasons[logged:]) == {None} and _SPY.launches > launches, "DeepGEMM fell back"
    dense = _dense_reference(q, k, w)
    bf16_payload = torch.gather(dense, 1, indices[0].clamp(min=0).long()).unsqueeze(0)
    valid = torch.arange(_K, device="cuda").view(1, 1, -1) < lengths.unsqueeze(-1)
    bf16_payload = bf16_payload.masked_fill(~valid, torch.finfo(torch.float32).min)

    def same_selection(*_args, **_kwargs):
        return indices.clone(), lengths.clone(), bf16_payload.clone()

    bf16_loss, bf16_grads, _ = _loss_and_grads(
        q, k, w, query, key, layout, FE_BF16, topk_override=same_selection
    )
    rel = ((dg_loss - bf16_loss).abs() / bf16_loss.abs()).item()
    _report(f"G8 loss deepgemm {dg_loss.item():.6e} bf16 {bf16_loss.item():.6e} rel {rel:.2e}")
    assert rel <= 1e-5
    for name, a, b in zip("qkw", dg_grads, bf16_grads):
        cosine = torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
        _report(f"G8 grad {name} cosine {cosine:.8f}")
        assert cosine > 1 - 1e-5
    # Report only: the KL predict from DeepGEMM's own BF16 values instead of the recompute.
    logits = dsa_cudnn_kernels._deepgemm_packed_thd_logits(
        q,
        k.unsqueeze(2),
        w,
        layout[0].to(torch.int32).contiguous(),
        layout[1].to(torch.int32).contiguous(),
        layout[3],
    )
    columns = (indices[0].long() - layout[0].view(-1, 1)).clamp(min=0, max=layout[3] - 1)
    option_a = torch.gather(logits.float(), 1, columns).unsqueeze(0)
    p_a = dsa_masking.masked_log_softmax(option_a, valid).exp()
    p_b = dsa_masking.masked_log_softmax(payload, valid).exp()
    tv = 0.5 * (p_a - p_b).abs().sum(dim=-1)
    _report(
        f"G8 DeepGEMM values vs recompute: predict TV mean {tv.mean().item():.4f} "
        f"max {tv.max().item():.4f}"
    )


# ---------------------------------------------------------------------------------------- G9
@pytest.mark.parametrize(
    "n_rows, width",
    [(2048, 4096), (4096, 16384), (4096, 32768), (8192, 65536), (4096, 131072), (16384, 1048576)],
    ids=["4k", "16k", "32k", "64k", "128k", "1m"],
)
def test_cudnn_bf16_radix_on_deepgemm_shaped_logits(n_rows, width):
    """Reachable at every width: DeepSelect declining, topk_backend='cudnn', dynamic CP."""
    _require_deepgemm()
    row_stride = -(-(-(-width // 384) * 384) // 512) * 512
    if torch.cuda.mem_get_info()[0] < n_rows * row_stride * 2 * 2 + 12 * _GIB:
        pytest.skip("not enough device memory")
    generator = torch.Generator(device="cuda").manual_seed(width % 997)
    buffer = torch.empty((n_rows, row_stride), dtype=torch.bfloat16, device="cuda")
    lengths = torch.randint(_K + 1, width + 1, (n_rows,), generator=generator, device="cuda")
    lengths[:6] = torch.tensor([0, 1, _K, _K + 1, _K + 17, width - 3], device="cuda")
    lengths = lengths.to(torch.int32)
    poison = torch.tensor([float("nan"), float("inf"), 3.0e38], device="cuda").bfloat16()
    for start in range(0, n_rows, 256):
        rows = buffer[start : start + 256]
        rows.copy_(
            (torch.randn(rows.shape, generator=generator, device="cuda") * 2).round().bfloat16()
        )
        columns = torch.arange(row_stride, device="cuda").view(1, -1)
        tail = columns >= lengths[start : start + 256].view(-1, 1).long()
        rows.copy_(torch.where(tail, poison[columns % 3], rows))
    first = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        buffer, lengths, _K, True, topk_backend="cudnn"
    )
    indices = first["indices"].long()
    in_row = (indices >= 0) & (indices < lengths.view(-1, 1).long())
    assert bool((in_row | (indices < 0)).all()), "a lane past the row's length was selected"
    valid_slots = lengths.clamp(max=_K)
    short = lengths <= _K
    long_rows = (~short).nonzero().flatten()[:256]
    for row in long_rows.tolist():
        n = int(lengths[row])
        want = torch.topk(buffer[row, :n].float(), _K).values.sort().values
        got = buffer[row, indices[row]].float().sort().values
        assert torch.equal(got, want), f"row {row} is not a top-{_K}"

    def selected_values(result):
        _assert_distinct_keys(result["indices"], valid_slots, "a cuDNN call")
        chosen = result["indices"].long()
        values = torch.gather(buffer, 1, chosen.clamp(min=0)).float()
        return torch.where(chosen >= 0, values, float("-inf")).sort(dim=-1).values

    # Repeats: tie-aware (cuDNN's BF16 radix top-k keeps other tied keys on a repeat, a
    # documented negative, see G5); whether they were bitwise is reported.
    first_values = selected_values(first)
    bitwise = []
    for _ in range(3):
        again = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
            buffer, lengths, _K, True, topk_backend="cudnn"
        )
        assert torch.equal(
            selected_values(again), first_values
        ), "a repeat selected keys that are not ties at the cutoff"
        bitwise.append(
            torch.equal(again["indices"].sort(dim=-1).values, first["indices"].sort(dim=-1).values)
        )
    _report(
        f"G9 {n_rows}x{width}: in-row, exact top-{_K}, repeats equal tie-aware "
        f"(bitwise {bitwise})"
    )


# --------------------------------------------------------------------------------------- G10
@pytest.mark.parametrize("topk_backend", ["cudnn", "deepselect"])
def test_row_chunks_select_what_one_call_selects(monkeypatch, topk_backend):
    _require_deepgemm()
    if topk_backend == "deepselect" and dsa_deepselect_topk._load_provider() is None:
        pytest.skip("DeepSelect is not installed")
    options = DEEPGEMM_DS if topk_backend == "deepselect" else DEEPGEMM
    # 128K CP32 rank 0 (4096 rows), split into 1024-row DeepGEMM calls.
    layout = _layout((131072,), 32, 0, 2)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * 32, seed=19)
    one = _score(q, k, w, layout, 32, 0, _K, options)
    logits, _lengths, ks = _packed_logits(q, k, w, layout, 32, 0)  # one call, before the split
    launches = _SPY.launches
    monkeypatch.setattr(dsa_deepgemm_scorer, "max_rows_per_call", lambda heads: 1100)
    chunked = _score(q, k, w, layout, 32, 0, _K, options)
    assert _SPY.launches - launches == 4
    assert torch.equal(chunked[1], one[1])

    def by_key(result):
        keys = torch.where(result[0] >= 0, result[0], torch.iinfo(torch.int32).max)
        order = keys.argsort(dim=-1)
        return keys.gather(-1, order), result[2].gather(-1, order)

    # Rows are independent in DeepGEMM and in both top-k kernels (G5: batch independence),
    # bitwise for DeepSelect; cuDNN's BF16 radix top-k may keep other tied keys on any call
    # (G5's documented negative), so its chunks are compared tie-aware on the DeepGEMM logits.
    keys, payload = by_key(one)
    chunked_keys, chunked_payload = by_key(chunked)
    same_rows = (chunked_keys == keys).all(dim=-1)
    if topk_backend == "deepselect":
        assert bool(same_rows.all())
    else:
        assert torch.equal(
            _selected_logits(chunked[0], chunked[1], logits, ks),
            _selected_logits(one[0], one[1], logits, ks),
        ), "a chunk selected keys that are not ties at the cutoff"
    # The payload is the log-softmax over the row's own selection: compare rows that kept
    # the same keys (all rows for DeepSelect). It is bitwise where the slots match too; where
    # cuDNN's BF16 radix top-k returned the same keys in another slot order only the
    # recompute's FP32 row sum moves (_log_prob_order_bound: an absolute bound on log p, since
    # it shifts every log p of the row by about the sum's relative change, so its count of
    # FP32 ulps grows without limit as |log p| -> 0). GB300 U1 4695952 measured at most 1 ulp,
    # 3.8e-6, at |log p| in [32, 64) (52790 of 4178944 elements differed).
    valid = torch.arange(_K, device="cuda").view(1, 1, -1) < one[1].unsqueeze(-1)
    same_slots = valid & (chunked[0] == one[0]).all(dim=-1, keepdim=True)
    valid = valid & same_rows.unsqueeze(-1)
    assert torch.equal(
        chunked_payload[same_slots], payload[same_slots]
    ), "rows with the same keys in the same slots fed the loss another payload"
    got, want = chunked_payload[valid], payload[valid]
    diff = (got - want).abs()
    bound = _log_prob_order_bound(got, want)
    over = ~(diff <= bound)  # NaN counts as over
    ulps = _fp32_ulp_distance(got, want)
    max_ulps = int(ulps.max()) if ulps.numel() else 0
    max_diff = float(diff.max()) if diff.numel() else 0.0
    max_ratio = float((diff / bound).max()) if diff.numel() else 0.0
    assert not bool(over.any()), (
        f"{int(over.sum())} of {diff.numel()} payload elements of rows with the same keys "
        f"differ past the slot-order bound (max |diff| {max_diff:.3e}, max |diff|/bound "
        f"{max_ratio:.3f}, max {max_ulps} FP32 ulp)"
    )
    _report(
        f"G10 {topk_backend}: 4 row chunks select what one call selects "
        f"({int(same_rows.sum())}/{same_rows.numel()} rows with the same keys, "
        f"{int(same_slots.any(dim=-1).sum())} in the same slots; payload: "
        f"{int((diff > 0).sum())}/{diff.numel()} elements differ, max |diff| {max_diff:.3e}, "
        f"max |diff|/bound {max_ratio:.3f}, max {max_ulps} FP32 ulp)"
    )


def test_rows_past_one_deepgemm_call_run_in_chunks():
    """65536 rows with 64 heads: one call would fail DeepGEMM's 58104-row assert."""
    _require_deepgemm()
    heads = 64
    layout = _layout((4096,) * 16, 1, 0)
    starts, ends, cu, max_seqlen = layout
    sq = starts.numel()
    assert sq > dsa_deepgemm_scorer.max_rows_per_call(heads)
    q, k, w = _inputs(sq, sq, seed=23, kind="ternary", heads=heads)
    launches = _SPY.launches
    indices, lengths, _ = _score(q, k, w, layout, 1, 0, 512, DEEPGEMM, False)
    assert _SPY.launches - launches >= 2  # two chunks (plus the 64-head warm-up)
    want_lengths = dsa_cudnn_kernels._packed_thd_row_lengths(
        starts, ends, cu, cp_size=1, num_rows=sq, width=max_seqlen
    )
    assert torch.equal(lengths[0].long(), want_lengths.long().clamp(max=512))
    # Exact top-512 (tie-aware) on rows around the chunk boundary and spread over both chunks.
    boundary = dsa_cudnn_kernels._deepgemm_row_chunks(sq, heads)[1][0]
    sample = sorted(
        set(range(boundary - 64, boundary + 64)) | set(range(0, sq, sq // 128)) | {sq - 1}
    )
    for row in sample:
        n_keys = int(want_lengths[row])
        start = int(starts[row])
        keys = torch.arange(start, start + n_keys, device="cuda")
        dense = (
            torch.einsum("hd,kd->hk", q[0, row].float(), k[0, keys].float()).relu()
            * w[0, row].float().unsqueeze(-1)
        ).sum(dim=0)
        n = int(lengths[0, row])
        got = dense[indices[0, row, :n].long() - start].sort().values
        want = torch.topk(dense, n).values.sort().values
        assert torch.equal(got, want), f"row {row} is not an exact top-{n}"
    _report(f"G10 {sq} rows x {heads} heads: chunked at row {boundary}, exact on sampled rows")
