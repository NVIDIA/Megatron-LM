# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for DeepGEMM `m_grouped_bf16_gemm_nt_contiguous`, the batch-invariant grouped GEMM.

Kernel entry points (`megatron/core/transformer/custom_layers/batch_invariant_kernels.py`):
- `grouped_gemm_batch_invariant(a, w, offs=, m_total=)` -> `_bf16_grouped_gemm_aligned_contiguous`:
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
  belong to no expert: they are neither computed into live rows nor written. Inputs are read only.
- `_bf16_grouped_gemm_contiguous(a, w, m_indices, counts)` with rows grouped by expert and
  sum(counts) == M_total (no alignment): the same y[r] for every row.
- Batch invariance: a row's bits depend only on a[r] and w[e]: not on the other experts' counts,
  the block's position, M_total, which entry point ran, or eager vs CUDA-graph replay with new
  offsets.
- Precondition, unchecked on main: aligned block starts. A misaligned `offs` must raise or still
  give every live row its own expert's product; it must never mix experts silently.

Reference: a per-expert fp64 loop. Trusted implementation: per-expert `torch.mm` in bf16 on the
same device (cuBLAS). Tolerance class: floating point, bounded per row (default factor);
bit-exact for the batch-invariance properties; exact for the untouched tail rows (NaN poison).

Structure (read after the contract): `_offs_to_m_indices` maps rows to expert ids
(searchsorted, -1 past offs[-1]); DeepGEMM (pin ea2b7805) reads a block's expert from its first
row and uses block_m = A on SM90 and SM100 (SM100 always swap_ab with block_n = 128,
`heuristics/sm100.hpp`), block_k = 64. Covered: per-expert counts 0, 1, A-1, A, A+1, 2A-1, 2A+1,
all experts empty, one expert holding every row, E in {1, 2, 8, 32, 128}, N in {128, 256, 4096},
K in {64, 128, 2880, 7168}, 0 / 1 / A tail rows past offs[-1], and M_total = 0.

Main vs open PRs: no open PR changes these entry points' per-row results. #8866 (t3-r04f2) and
#8869 (t3-p4n3) add `_bf16_grouped_gemm_device_counts` (the TE grouped-tensor path) on the same
DeepGEMM kernel; it is not on main and is not tested here. The misaligned-offsets case is
expected to fail on main (finding O8a-N1); no open PR adds the guard.

Hardware: H100 and GB200. Needs DeepGEMM with bf16 bindings; a missing DeepGEMM fails the test
on the target hardware.
"""

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
    """Counts DeepGEMM grouped calls and NaN-poisons their output first, so rows the kernel does
    not write stay NaN."""

    def __init__(self, monkeypatch):
        self.calls = 0
        self._real = bik.deep_gemm.m_grouped_bf16_gemm_nt_contiguous
        monkeypatch.setattr(bik.deep_gemm, "m_grouped_bf16_gemm_nt_contiguous", self)

    def __call__(self, a, b, d, *args, **kwargs):
        self.calls += 1
        ou.poison_(d)
        return self._real(a, b, d, *args, **kwargs)


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_CASES = ou.pairwise_cases(
    {"e": EXPERTS, "n": N_SIZES, "k": K_SIZES, "pattern": PATTERNS, "tail": TAILS}
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_aligned_grouped_gemm_matches_fp64(case, monkeypatch):
    """Inference entry point: live rows against fp64; padding never leaks; the tail is untouched."""
    align = _alignment()
    counts = expert_counts(case["e"], case["pattern"], align)
    tail = {"none": 0, "one": 1, "block": align}[case["tail"]]
    batch = AlignedBatch(counts, case["k"], align, tail, seed=1)
    w = _weights(case["e"], case["n"], case["k"])
    a_before, w_before = batch.a.clone(), w.clone()
    spy = _DeepGemmSpy(monkeypatch)
    out = bik.grouped_gemm_batch_invariant(batch.a, w, offs=batch.offs, m_total=batch.m_total)
    assert out.shape == (batch.m_total, case["n"]) and out.dtype == torch.bfloat16
    assert spy.calls == (1 if batch.m_total else 0), f"DeepGEMM ran {spy.calls} times"
    ou.assert_exact(batch.a, a_before, what="a is read-only")
    ou.assert_exact(w, w_before, what="w is read-only")
    _check_experts(
        out, batch.a, [batch.live(e) for e in range(case["e"])], w, f"aligned {_id(case)}"
    )
    if batch.m_total > batch.n_used:
        ou.assert_poisoned(out[batch.n_used :], what="rows past offs[-1]")


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
    aligned = bik.grouped_gemm_batch_invariant(batch.a, w, offs=batch.offs, m_total=batch.m_total)
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
        out = bik.grouped_gemm_batch_invariant(batch.a, w, offs=batch.offs, m_total=batch.m_total)
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
        bik.grouped_gemm_batch_invariant(a_buf, w, offs=offs_buf, m_total=capacity)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_buf = bik.grouped_gemm_batch_invariant(a_buf, w, offs=offs_buf, m_total=capacity)
    for pattern in ("cycle3", "one_expert", "all_empty", "cycle0"):
        batch = AlignedBatch(expert_counts(num_experts, pattern, align), k, align, 0, seed=7)
        ou.poison_(a_buf)
        a_buf[: batch.m_total] = batch.a
        offs_buf.copy_(batch.offs)
        graph.replay()
        eager = bik.grouped_gemm_batch_invariant(batch.a, w, offs=batch.offs, m_total=batch.m_total)
        for e in range(num_experts):
            rows = batch.live(e)
            ou.assert_exact(out_buf[rows], eager[rows], what=f"replay {pattern}, expert {e}")


def test_misaligned_offsets_raise_or_stay_correct():
    """A block start that is not a multiple of the alignment: raise, or keep every row right.

    Expected to fail on main: nothing checks the alignment, and DeepGEMM takes a block's expert
    from its first row, so expert 1's first rows are multiplied by expert 0's weights (finding
    O8a-N1). The inference permute aligns every block, so production never builds such offsets.
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
        out = bik.grouped_gemm_batch_invariant(a, w, offs=offs, m_total=ends[-1])
        torch.cuda.synchronize()
    except (AssertionError, RuntimeError, ValueError):
        return
    rows = [
        torch.arange(0, ends[0], device=_DEVICE),
        torch.arange(ends[0], ends[1], device=_DEVICE),
    ]
    _check_experts(out, a, rows, w, "misaligned offsets")


def test_empty_problem():
    """M_total = 0: an empty [0, N] result, on both entry points."""
    _alignment()
    w = _weights(4, 128, 64)
    a = torch.empty(0, 64, device=_DEVICE, dtype=torch.bfloat16)
    offs = torch.zeros(4, device=_DEVICE, dtype=torch.int32)
    out = bik.grouped_gemm_batch_invariant(a, w, offs=offs, m_total=0)
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
    _DeepGemmSpy(monkeypatch)
    out = bik.grouped_gemm_batch_invariant(batch.a, w, offs=batch.offs, m_total=batch.m_total)
    _check_experts(out, batch.a, [batch.live(e) for e in range(2)], w, "sanitizer")
    ou.assert_poisoned(out[batch.n_used :], what="rows past offs[-1]")
