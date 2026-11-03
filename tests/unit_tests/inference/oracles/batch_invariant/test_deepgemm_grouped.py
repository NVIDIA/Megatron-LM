# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for DeepGEMM `m_grouped_bf16_gemm_nt_contiguous`, the batch-invariant grouped GEMM.

Kernel entry points (`megatron/core/transformer/custom_layers/batch_invariant_kernels.py`):
- `grouped_gemm_batch_invariant(a, w, offs=, m_total=)` (with s4f-grouped-gemm-alignment also
  `offs_alignment=`) -> `_bf16_grouped_gemm_aligned_contiguous`:
  inference, over expert blocks already aligned by the permute (`inference/moe/fused_moe.py`
  through `inference/moe/batch_invariant.grouped_mm`, FC1 and FC2 of every BF16 MoE layer in
  batch-invariant mode); CUDA-graph safe.
- `_bf16_grouped_gemm_contiguous(a, w, m_indices, counts)`: training and TE-implementation
  inference (`_te_general_grouped_gemm_patched` forward), unaligned counts padded internally.

Contract (written from the callers' semantics):
- `grouped_gemm_batch_invariant(a [M_total, K] bf16, w [E, N, K] bf16, offs [E] int, m_total)`:
  `offs[e]` is the end row of expert e's block (block e is rows [offs[e-1], offs[e]), offs[-1] =
  0), and every block starts at a multiple of A = `grouped_gemm_batch_invariant_alignment()`.
  The result y [M_total, N] bf16 has y[r] = bf16(a[r] @ w[e]^T), accumulated in fp32 and rounded
  once, for every row r of block e. A block's rows past the expert's live tokens are padding
  whose content is undefined (NaN here); they must not reach a live row. Rows r >= offs[E-1]
  belong to no expert: their output rows are unspecified (DeepGEMM computes them with expert 0's
  weights, and its own tests mask them; the wasted work is observation O8a-N6), but no write leaves the [M_total, N] output and their
  input never reaches a live row. The callers allow this: the output is a fresh `torch.empty`
  of M_total rows, and the activation and unpermute stop at offs[-1]. Inputs are read only.
- `_bf16_grouped_gemm_contiguous(a, w, m_indices, counts)` with rows grouped by expert and
  sum(counts) == M_total (no alignment): the same y[r] for every row.
- Batch invariance: a row's bits depend only on a[r] and w[e]: not on the other experts' counts,
  the block's position, M_total, which entry point ran, or eager vs CUDA-graph replay with new
  offsets.
- Precondition: aligned block starts. On main it is unchecked and undeclared. With
  s4f-grouped-gemm-alignment the caller declares the alignment the offsets were built with
  (`offs_alignment`): a declaration that is not a positive multiple of A, or an `offs` that is not
  one entry per expert ([E]), must be rejected before the GEMM; offsets declared with a multiple
  of A but holding misaligned values remain the documented, unchecked precondition (their values
  are on the device). A misaligned `offs` declared with its true alignment must raise or still
  give every live row its own expert's product; it must never mix experts silently.

Reference: a per-expert fp64 loop. Trusted implementation: per-expert `torch.mm` in bf16 on the
same device (cuBLAS). Tolerance class: floating point, bounded per row (default factor);
bit-exact for the batch-invariance properties; guard bands of one alignment block of rows around
DeepGEMM's output (exact).

Structure (read after the contract): `_offs_to_m_indices` maps rows to expert ids
(searchsorted, -1 past offs[-1]); DeepGEMM (pin ea2b7805) reads a block's expert from its first
row and uses block_m = A on SM90 and SM100 (SM100 always swap_ab with block_n = 128,
`heuristics/sm100.hpp`), block_k = 64. Covered: per-expert counts 0, 1, A-1, A, A+1, 2A-1, 2A+1,
all experts empty, one expert holding every row, E in {1, 2, 8, 32, 128}, N in {128, 256, 4096},
K in {64, 128, 2880, 7168}, 0 / 1 / A tail rows past offs[-1], and M_total = 0.

Main vs open PRs: no open PR changes these entry points' per-row results. #8866 (t3-r04f2) and
#8869 (t3-p4n3) add `_bf16_grouped_gemm_device_counts` (the TE grouped-tensor path) on the same
DeepGEMM kernel; it is not on main and is not tested here. s4f-grouped-gemm-alignment
(fix/bi-grouped-gemm-offsets-contract) adds the required `offs_alignment` keyword, which
`grouped_gemm` passes only where the entry point takes it, and the host-side checks: the
misaligned-offsets case, the foreign-alignment rejections and the offs-shape rejections fail on
main (finding O8a-N1: nothing is declared or checked there) and pass with that PR. Every other
case passes on both.

Hardware: H100 and GB200. Needs DeepGEMM with bf16 bindings; a missing DeepGEMM fails the test
on the target hardware.
"""

import inspect
import math

import pytest
import torch

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

_DEVICE = "cuda"

EXPERTS = (1, 2, 8, 32, 128)
N_SIZES = (128, 256, 4096)
K_SIZES = (64, 128, 2880, 7168)
PATTERNS = ("cycle0", "cycle3", "all_empty", "one_expert")
TAILS = ("none", "one", "block")


def _alignment() -> int:
    assert bik.HAVE_DEEPGEMM_BF16, (
        "DeepGEMM with bf16 bindings is required on the target hardware (pip extra"
        " 'batch_invariant')"
    )
    return bik.grouped_gemm_batch_invariant_alignment()


# s4f-grouped-gemm-alignment (branch fix/bi-grouped-gemm-offsets-contract) makes every caller
# declare the alignment its offsets were built with: `offs_alignment`, keyword-only and required
# there, absent on main. These tests are written against main and must also pass with that PR,
# so the keyword is passed exactly when the entry point takes it.
_DECLARES_ALIGNMENT = (
    "offs_alignment" in inspect.signature(bik.grouped_gemm_batch_invariant).parameters
)


def grouped_gemm(a, w, offs, m_total, offs_alignment):
    """`grouped_gemm_batch_invariant`, declaring ``offs_alignment`` where the API takes it."""
    kwargs = dict(offs=offs, m_total=m_total)
    if _DECLARES_ALIGNMENT:
        kwargs["offs_alignment"] = offs_alignment
    return bik.grouped_gemm_batch_invariant(a, w, **kwargs)


def expert_counts(num_experts: int, pattern: str, align: int):
    """Live tokens per expert: the boundary counts around the alignment, or an edge pattern."""
    values = (0, 1, align - 1, align, align + 1, 2 * align - 1, 2 * align + 1)
    if pattern.startswith("cycle"):
        shift = int(pattern[len("cycle") :])
        return [values[(e + shift) % len(values)] for e in range(num_experts)]
    if pattern == "all_empty":
        return [0] * num_experts
    if pattern == "one_expert":
        return [2 * align + 1 if e == num_experts // 2 else 0 for e in range(num_experts)]
    raise ValueError(pattern)


class AlignedBatch:
    """An aligned expert layout: live rows random, padding and tail rows NaN, as the permute
    leaves them (it does not zero BF16 padding)."""

    def __init__(self, counts, k, align, tail_rows, seed, live_rows=None):
        self.counts = list(counts)
        blocks = [math.ceil(c / align) * align for c in self.counts]
        ends = [sum(blocks[: e + 1]) for e in range(len(blocks))]
        self.starts = [end - block for end, block in zip(ends, blocks)]
        self.n_used = ends[-1] if ends else 0
        self.m_total = self.n_used + tail_rows
        self.offs = torch.tensor(ends, device=_DEVICE, dtype=torch.int32)
        self.a = ou.poisoned_empty((self.m_total, k), torch.bfloat16, device=_DEVICE)
        g = ou.make_generator(seed, device=_DEVICE)
        for e, (start, count) in enumerate(zip(self.starts, self.counts)):
            if live_rows is not None and e in live_rows:
                self.a[start : start + count] = live_rows[e]
            else:
                rows = torch.randn(count, k, device=_DEVICE, generator=g)
                self.a[start : start + count] = rows.to(torch.bfloat16)

    def live(self, e):
        """Row indices of expert e's live tokens."""
        return torch.arange(
            self.starts[e], self.starts[e] + self.counts[e], device=_DEVICE, dtype=torch.long
        )


def _weights(num_experts, n, k, seed=2):
    g = ou.make_generator(seed, device=_DEVICE)
    return (torch.randn(num_experts, n, k, device=_DEVICE, generator=g) / math.sqrt(k)).to(
        torch.bfloat16
    )


def _expert_reference(x, w_e):
    return x @ w_e.t()


def _check_experts(out, a, rows_per_expert, w, what):
    for e, rows in enumerate(rows_per_expert):
        if rows.numel() == 0:
            continue
        x = a[rows]
        ou.assert_close_to_reference(
            out[rows],
            ou.fp64_reference(_expert_reference, x, w[e]),
            torch.mm(x, w[e].t()),
            dim=-1,
            what=f"{what}: expert {e} ({rows.numel()} rows)",
        )


class _DeepGemmSpy:
    """Counts DeepGEMM grouped calls and runs each into a guarded, NaN-poisoned output.

    The guard bands hold one alignment block of rows on each side (the largest tile DeepGEMM
    stores); the result is copied into the caller's output afterwards.
    """

    def __init__(self, monkeypatch):
        self.calls = 0
        self.violations = []
        self._real = bik.deep_gemm.m_grouped_bf16_gemm_nt_contiguous
        monkeypatch.setattr(bik.deep_gemm, "m_grouped_bf16_gemm_nt_contiguous", self)

    def __call__(self, a, b, d, *args, **kwargs):
        self.calls += 1
        rows = bik.grouped_gemm_batch_invariant_alignment()
        guarded = ou.guarded_empty(
            d.shape, d.dtype, device=d.device, guard_bytes=rows * d.shape[1] * d.element_size()
        )
        result = self._real(a, b, guarded.tensor, *args, **kwargs)
        self.violations += guarded.violations()
        d.copy_(guarded.tensor)
        return result

    def check(self):
        assert not self.violations, "DeepGEMM wrote outside its output: " + "; ".join(
            self.violations
        )


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_CASES = ou.pairwise_cases(
    {"e": EXPERTS, "n": N_SIZES, "k": K_SIZES, "pattern": PATTERNS, "tail": TAILS}
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_aligned_grouped_gemm_matches_fp64(case, monkeypatch):
    """Inference entry point: live rows against fp64; padding and tail rows never leak into them;
    no write leaves the output."""
    align = _alignment()
    counts = expert_counts(case["e"], case["pattern"], align)
    tail = {"none": 0, "one": 1, "block": align}[case["tail"]]
    batch = AlignedBatch(counts, case["k"], align, tail, seed=1)
    w = _weights(case["e"], case["n"], case["k"])
    a_before, w_before = batch.a.clone(), w.clone()
    spy = _DeepGemmSpy(monkeypatch)
    out = grouped_gemm(batch.a, w, batch.offs, batch.m_total, align)
    assert out.shape == (batch.m_total, case["n"]) and out.dtype == torch.bfloat16
    assert spy.calls == (1 if batch.m_total else 0), f"DeepGEMM ran {spy.calls} times"
    ou.assert_exact(batch.a, a_before, what="a is read-only")
    ou.assert_exact(w, w_before, what="w is read-only")
    spy.check()
    _check_experts(
        out, batch.a, [batch.live(e) for e in range(case["e"])], w, f"aligned {_id(case)}"
    )


@pytest.mark.parametrize("case", _CASES[::2], ids=[_id(c) for c in _CASES[::2]])
def test_padded_entry_point_matches_aligned(case):
    """Training entry point (unaligned counts) against fp64 and bit-equal to the aligned one."""
    align = _alignment()
    counts = expert_counts(case["e"], case["pattern"], align)
    batch = AlignedBatch(counts, case["k"], align, 0, seed=1)
    w = _weights(case["e"], case["n"], case["k"])
    live = [batch.live(e) for e in range(case["e"])]
    all_live = torch.cat(live) if live else torch.empty(0, dtype=torch.long, device=_DEVICE)
    compact = batch.a[all_live].contiguous()
    m_indices = torch.repeat_interleave(
        torch.arange(case["e"], device=_DEVICE, dtype=torch.int32),
        torch.tensor(counts, device=_DEVICE),
    ).contiguous()
    out = bik._bf16_grouped_gemm_contiguous(compact, w, m_indices, counts)
    assert out.shape == (sum(counts), case["n"]) and out.dtype == torch.bfloat16
    compact_rows, start = [], 0
    for count in counts:
        compact_rows.append(torch.arange(start, start + count, device=_DEVICE))
        start += count
    _check_experts(out, compact, compact_rows, w, f"padded {_id(case)}")
    aligned = grouped_gemm(batch.a, w, batch.offs, batch.m_total, align)
    ou.assert_exact(out, aligned[all_live], what="padded vs aligned entry point")


def test_rows_do_not_depend_on_other_experts():
    """One expert's rows keep their bits as the other experts' counts move its block and M_total."""
    align = _alignment()
    num_experts, n, k, probe = 8, 256, 2880, 5
    w = _weights(num_experts, n, k)
    g = ou.make_generator(9, device=_DEVICE)
    probe_rows = torch.randn(align + 3, k, device=_DEVICE, generator=g).to(torch.bfloat16)
    expected = None
    layouts = [
        ([0] * 8, 0),
        ([1, 0, 2 * align + 1, 0, align, 0, 7, 0], 0),
        ([align - 1] * 8, 1),
        ([2 * align + 1, align + 1, 1, 0, 0, 0, 3, align], align),
    ]
    for counts, tail in layouts:
        counts = list(counts)
        counts[probe] = probe_rows.shape[0]
        batch = AlignedBatch(
            counts, k, align, tail, seed=sum(counts), live_rows={probe: probe_rows}
        )
        out = grouped_gemm(batch.a, w, batch.offs, batch.m_total, align)
        rows = out[batch.live(probe)]
        if expected is None:
            expected = rows
        ou.assert_exact(rows, expected, what=f"probe expert with counts {counts}, tail {tail}")


def test_cuda_graph_replay_with_new_offsets():
    """Captured once, replayed with new rows and offsets: the eager bits for every live row."""
    align = _alignment()
    num_experts, n, k = 8, 256, 128
    w = _weights(num_experts, n, k)
    capacity = num_experts * 3 * align
    a_buf = ou.poisoned_empty((capacity, k), torch.bfloat16, device=_DEVICE)
    offs_buf = torch.zeros(num_experts, device=_DEVICE, dtype=torch.int32)
    first = AlignedBatch(expert_counts(num_experts, "cycle0", align), k, align, 0, seed=1)
    a_buf[: first.m_total] = first.a
    offs_buf.copy_(first.offs)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm up (DeepGEMM JIT) outside the capture
        grouped_gemm(a_buf, w, offs_buf, capacity, align)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_buf = grouped_gemm(a_buf, w, offs_buf, capacity, align)
    for pattern in ("cycle3", "one_expert", "all_empty", "cycle0"):
        batch = AlignedBatch(expert_counts(num_experts, pattern, align), k, align, 0, seed=7)
        ou.poison_(a_buf)
        a_buf[: batch.m_total] = batch.a
        offs_buf.copy_(batch.offs)
        graph.replay()
        eager = grouped_gemm(batch.a, w, batch.offs, batch.m_total, align)
        for e in range(num_experts):
            rows = batch.live(e)
            ou.assert_exact(out_buf[rows], eager[rows], what=f"replay {pattern}, expert {e}")


def test_misaligned_offsets_raise_or_stay_correct():
    """A block start that is not a multiple of the alignment, declared with its true alignment
    (1): raise before the GEMM, or keep every row right.

    Fails on main: nothing checks the alignment (main's entry point takes no declaration), and
    DeepGEMM takes a block's expert from its first row, so expert 1's first rows are multiplied
    by expert 0's weights (finding O8a-N1). Passes with s4f-grouped-gemm-alignment, which rejects
    a declared alignment DeepGEMM cannot use. Offsets declared with DeepGEMM's alignment but
    holding misaligned values stay a documented, unchecked precondition there (the values live on
    the device), so they are not a case here. The inference permute aligns every block.
    """
    align = _alignment()
    k, n = 128, 256
    w = _weights(2, n, k)
    counts = [align + 5, align]
    ends = [counts[0], counts[0] + counts[1]]  # expert 1 starts at align + 5
    g = ou.make_generator(3, device=_DEVICE)
    a = torch.randn(ends[-1], k, device=_DEVICE, generator=g).to(torch.bfloat16)
    offs = torch.tensor(ends, device=_DEVICE, dtype=torch.int32)
    try:
        out = grouped_gemm(a, w, offs, ends[-1], offs_alignment=1)
        torch.cuda.synchronize()
    except (AssertionError, RuntimeError, ValueError):
        return
    rows = [
        torch.arange(0, ends[0], device=_DEVICE),
        torch.arange(ends[0], ends[1], device=_DEVICE),
    ]
    _check_experts(out, a, rows, w, "misaligned offsets")


class _GemmCalls:
    """Counts the aligned grouped GEMM's launches, to check that a rejection comes first."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = bik._bf16_grouped_gemm_aligned_contiguous

        def spy(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(bik, "_bf16_grouped_gemm_aligned_contiguous", spy)


def _expect_rejection(monkeypatch, call, what):
    """`call` must raise before the GEMM runs."""
    gemm = _GemmCalls(monkeypatch)
    with pytest.raises((AssertionError, ValueError)):
        call()
    torch.cuda.synchronize()
    assert gemm.calls == 0, f"{what}: the GEMM ran before the rejection"


# (case, declared alignment as a function of DeepGEMM's A): offsets built from raw cumulative
# counts, with torch grouped_mm's 16, with half of A, and a non-positive declaration.
_FOREIGN_ALIGNMENTS = [
    ("raw_counts", lambda a: 1),
    ("torch_grouped_mm", lambda a: 16),
    ("half", lambda a: a // 2),
    ("zero", lambda a: 0),
]


@pytest.mark.parametrize(
    "declared", [f for _, f in _FOREIGN_ALIGNMENTS], ids=[c for c, _ in _FOREIGN_ALIGNMENTS]
)
def test_declared_alignment_deepgemm_cannot_use_is_rejected(declared, monkeypatch):
    """Offsets built for counts [A + 5, A] with an alignment that is not a positive multiple of
    DeepGEMM's A, and declared so: rejected before the GEMM.

    Fails on main (no declaration: the GEMM runs and mixes experts, O8a-N1); passes with
    s4f-grouped-gemm-alignment.
    """
    align = _alignment()
    alignment = declared(align)
    step = max(alignment, 1)
    counts = [align + 5, align]
    blocks = [math.ceil(c / step) * step for c in counts]
    ends = [blocks[0], blocks[0] + blocks[1]]
    assert ends[0] % align, "the layout must be one DeepGEMM cannot use"
    w = _weights(2, 128, 64)
    a = torch.zeros(ends[-1], 64, device=_DEVICE, dtype=torch.bfloat16)
    offs = torch.tensor(ends, device=_DEVICE, dtype=torch.int32)
    _expect_rejection(
        monkeypatch,
        lambda: grouped_gemm(a, w, offs, ends[-1], alignment),
        f"declared alignment {alignment}",
    )


@pytest.mark.parametrize("num_offs", [1, 3], ids=["fewer_than_experts", "more_than_experts"])
def test_offsets_not_one_per_expert_are_rejected(num_offs, monkeypatch):
    """`offs` must hold one end row per expert of `w` ([E]); one or three offsets for two experts
    are rejected before the GEMM. The layouts are aligned and every extra block is empty, so a
    GEMM that ran anyway would stay inside `w`.

    Fails on main (offs' shape is unchecked); passes with s4f-grouped-gemm-alignment.
    """
    align = _alignment()
    ends = [align, 2 * align, 2 * align][:num_offs]
    w = _weights(2, 128, 64)
    m_total = 2 * align
    a = torch.zeros(m_total, 64, device=_DEVICE, dtype=torch.bfloat16)
    offs = torch.tensor(ends, device=_DEVICE, dtype=torch.int32)
    _expect_rejection(
        monkeypatch, lambda: grouped_gemm(a, w, offs, m_total, align), f"{num_offs} offsets"
    )


def test_empty_problem():
    """M_total = 0: an empty [0, N] result, on both entry points."""
    _alignment()
    w = _weights(4, 128, 64)
    a = torch.empty(0, 64, device=_DEVICE, dtype=torch.bfloat16)
    offs = torch.zeros(4, device=_DEVICE, dtype=torch.int32)
    out = grouped_gemm(a, w, offs, 0, _alignment())
    assert out.shape == (0, 128) and out.dtype == torch.bfloat16
    m_indices = torch.empty(0, device=_DEVICE, dtype=torch.int32)
    out = bik._bf16_grouped_gemm_contiguous(a, w, m_indices, [0, 0, 0, 0])
    assert out.shape == (0, 128) and out.dtype == torch.bfloat16


@pytest.mark.sanitizer
def test_sanitizer_case(monkeypatch):
    """Small case for compute-sanitizer: two experts, partial blocks, NaN padding, tail rows."""
    align = _alignment()
    batch = AlignedBatch([3, align + 1], 64, align, 2, seed=1)
    w = _weights(2, 128, 64)
    spy = _DeepGemmSpy(monkeypatch)
    out = grouped_gemm(batch.a, w, batch.offs, batch.m_total, align)
    spy.check()
    _check_experts(out, batch.a, [batch.live(e) for e in range(2)], w, "sanitizer")
