# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real cuDNN frontend checks of the DSA top-k chunking and deterministic tie-break path.

Run on one SM90+ GPU with the installed frontend:

* FE >= 1.31 (``tie_break``): the deterministic wrapper (``tie_break=1`` + index sort +
  zero-tie repair) must equal the full-row canonical rule bitwise, indices and value bits,
  on Gaussian, heavy-tie, signed-zero (mixed ``-0.0``/``+0.0`` at the cutoff, all-zero rows),
  NaN/infinity, short and garbage-tailed rows, at row widths that take the kernel's 16-bit,
  32-bit shared-memory and global-spill paths; it must repeat bitwise and give the same result
  from a CUDA graph replay as eagerly.
* FE < 1.31 (no ``tie_break``): the probe must report no support and the wrapper must fall
  back to the full-row rule.
* Both: 8 GiB chunks at 1M columns (1024 rows per launch, exactly 2**31 scratch elements)
  select the right set, and the scratch peak stays within the budget; eagerly and in a CUDA
  graph's private pool the chunked call reserves one scratch block, not one per launch.
* FE >= 1.31: ``tie_break=1`` and ``return_val=False`` reach the real frontend, and 1M-column
  rows whose cutoff is a signed zero (every row) match the full-row rule.

Lines starting with ``[topk-gpu]`` report timings and memory for the validation log.
"""

import gc
import time

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_topk_zero_ties,
)

_GIB = 1024**3


def _require_frontend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cuDNN radix top-k needs SM90+")
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("cuDNN frontend DSA is not installed")
    dsa_cudnn_kernels._ensure_dsa_namespace()


def _require_tie_break():
    _require_frontend()
    if not dsa_cudnn_kernels._supports_lowest_index_topk_tie_break():
        pytest.skip(
            f"cuDNN frontend {dsa_cudnn_kernels._cudnn_frontend_version()} has no tie_break"
        )


def _report(message: str) -> None:
    print(f"[topk-gpu] {message}", flush=True)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def _assert_bitwise_equal(actual: dict, expected: dict, label: str) -> None:
    mismatched_rows = (actual["indices"] != expected["indices"]).any(dim=1)
    assert not bool(mismatched_rows.any()), (
        f"{label}: {int(mismatched_rows.sum())} rows differ in indices, first rows "
        f"{torch.nonzero(mismatched_rows).view(-1)[:8].tolist()}"
    )
    if expected["values"] is None:
        assert actual["values"] is None
        return
    value_rows = (_bits(actual["values"]) != _bits(expected["values"])).any(dim=1)
    assert not bool(value_rows.any()), f"{label}: {int(value_rows.sum())} rows differ in values"


def _full_row_rule(scores, seq_lens, topk_k, return_topk_scores, monkeypatch):
    """Today's deterministic path: raw tie_break=0 top-k plus the full-row canonicalization."""
    with monkeypatch.context() as patch:
        patch.setattr(dsa_cudnn_kernels, "_supports_lowest_index_topk_tie_break", lambda: False)
        return dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
            scores, seq_lens, topk_k, return_topk_scores, deterministic=True
        )


def _tie_break_path(scores, seq_lens, topk_k, return_topk_scores):
    return dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k, return_topk_scores, deterministic=True
    )


def _lengths(n_rows, sk, topk_k, generator):
    lengths = torch.randint(topk_k + 1, sk + 1, (n_rows,), generator=generator, device="cuda")
    # Short rows (len <= K, including 0 and exactly K) take the kernel's trivial path.
    short = torch.tensor([0, 1, topk_k - 1, topk_k, topk_k + 1, sk], device="cuda")
    lengths[: short.numel()] = short.clamp(max=sk)
    return lengths.to(torch.int32)


def _make_rows(kind: str, n_rows: int, sk: int, topk_k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens = _lengths(n_rows, sk, topk_k, generator)
    if kind == "gauss":
        scores = torch.randn((n_rows, sk), generator=generator, device="cuda")
    elif kind == "heavy_ties":
        # 1/64-rounded scores: hundreds to thousands of copies of the cutoff value.
        scores = torch.round(torch.randn((n_rows, sk), generator=generator, device="cuda") * 4)
        scores = scores / 64
    elif kind == "signed_zeros":
        # Fewer than K positives, then zeros of random sign: the cutoff is +-0 in most rows.
        uniform = torch.rand((n_rows, sk), generator=generator, device="cuda")
        positives = torch.randint(0, topk_k, (n_rows, 1), generator=generator, device="cuda")
        rank = torch.argsort(uniform, dim=1).argsort(dim=1)
        sign = torch.rand((n_rows, sk), generator=generator, device="cuda") < 0.5
        zeros = torch.where(sign, torch.zeros_like(uniform), torch.full_like(uniform, -0.0))
        negatives = -uniform - 0.01
        scores = torch.where(rank < positives, uniform + 0.01, zeros)
        third = sk // 3
        scores = torch.where(rank >= sk - third, negatives, scores)
    elif kind == "denormal_zeros":
        # Fewer than K positives, then zeros of both signs mixed with +-denormals: denormals
        # are not zeros (no flush to zero), so they must never fill zero-tie slots.
        uniform = torch.rand((n_rows, sk), generator=generator, device="cuda")
        tiny = torch.finfo(torch.float32).tiny
        positives = uniform < (topk_k / 2) / sk
        scores = torch.where(
            uniform < 0.5, torch.full_like(uniform, 0.0), torch.full_like(uniform, -0.0)
        )
        pick = torch.rand((n_rows, sk), generator=generator, device="cuda")
        scores = torch.where(pick < 0.02, torch.full_like(scores, tiny / 4), scores)
        scores = torch.where(
            (pick >= 0.02) & (pick < 0.04), torch.full_like(scores, -tiny / 4), scores
        )
        scores = torch.where(
            (pick >= 0.04) & (pick < 0.05), torch.full_like(scores, tiny / 1024), scores
        )
        scores = torch.where(pick >= 0.7, -uniform - 0.01, scores)
        scores = torch.where(positives, uniform + 0.01, scores)
        assert bool(((scores != 0) & (scores.abs() < tiny)).any()), "denormals were flushed"
    elif kind == "all_zeros":
        sign = torch.rand((n_rows, sk), generator=generator, device="cuda") < 0.5
        scores = torch.where(
            sign,
            torch.zeros((n_rows, sk), device="cuda"),
            torch.full((n_rows, sk), -0.0, device="cuda"),
        )
        scores[0::3] = 0.0
        scores[1::3] = -0.0
    elif kind == "specials":
        scores = torch.round(torch.randn((n_rows, sk), generator=generator, device="cuda") * 2) / 4
        pick = torch.rand((n_rows, sk), generator=generator, device="cuda")
        scores = torch.where(pick < 0.002, torch.full_like(scores, float("nan")), scores)
        scores = torch.where(
            (pick >= 0.002) & (pick < 0.004), torch.full_like(scores, -float("nan")), scores
        )
        scores = torch.where(
            (pick >= 0.004) & (pick < 0.005), torch.full_like(scores, float("inf")), scores
        )
        scores = torch.where(
            (pick >= 0.005) & (pick < 0.05), torch.full_like(scores, -float("inf")), scores
        )
        scores = torch.where((pick >= 0.05) & (pick < 0.2), torch.full_like(scores, -0.0), scores)
    else:
        raise AssertionError(kind)
    # Columns past a row's length are undefined with clean_logits=False: poison them.
    columns = torch.arange(sk, device="cuda").view(1, sk)
    poison = torch.where(
        torch.rand((n_rows, sk), generator=generator, device="cuda") < 0.5,
        torch.full((n_rows, sk), float("nan"), device="cuda"),
        torch.full((n_rows, sk), 1.0e30, device="cuda"),
    )
    scores = torch.where(columns < seq_lens.view(n_rows, 1).long(), scores, poison)
    return scores.float().contiguous(), seq_lens


# (n_rows, sk): 16-bit column indices, 32-bit indices in shared memory, global spill.
_WIDTHS = [(96, 8192), (64, 70000), (24, 300000)]
_KINDS = ["gauss", "heavy_ties", "signed_zeros", "denormal_zeros", "all_zeros", "specials"]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("n_rows,sk", _WIDTHS, ids=[f"{r}x{c}" for r, c in _WIDTHS])
@pytest.mark.parametrize("topk_k", [2048, 512])
def test_tie_break_path_matches_the_full_row_rule(monkeypatch, kind, n_rows, sk, topk_k):
    _require_tie_break()
    seed = 1000 * _KINDS.index(kind) + sk % 997 + topk_k
    scores, seq_lens = _make_rows(kind, n_rows, sk, topk_k, seed=seed)

    expected = _full_row_rule(scores, seq_lens, topk_k, True, monkeypatch)
    actual = _tie_break_path(scores, seq_lens, topk_k, True)
    indices_only = _tie_break_path(scores, seq_lens, topk_k, False)

    _assert_bitwise_equal(actual, expected, f"{kind} {n_rows}x{sk} k={topk_k}")
    assert torch.equal(indices_only["indices"], expected["indices"])
    assert indices_only["values"] is None
    zero_cutoff_rows = int(
        ((actual["values"] == 0) & (actual["indices"] >= 0)).any(dim=1).sum().item()
    )
    _report(
        f"equal kind={kind} shape={n_rows}x{sk} k={topk_k} rows_with_zero_members={zero_cutoff_rows}"
    )


def test_tie_break_path_keeps_lowest_index_ties_against_a_sort_reference():
    _require_tie_break()
    n_rows, sk, topk_k = 128, 131072, 2048
    scores, seq_lens = _make_rows("heavy_ties", n_rows, sk, topk_k, seed=17)
    zeros = _make_rows("signed_zeros", n_rows, sk, topk_k, seed=18)
    scores = torch.cat((scores, zeros[0]))
    seq_lens = torch.cat((seq_lens, zeros[1]))
    columns = torch.arange(sk, device="cuda").view(1, sk)
    masked = torch.where(columns < seq_lens.view(-1, 1).long(), scores, float("-inf"))
    # Adding +0.0 turns -0.0 into +0.0, so the stable sort sees value ties (whatever its
    # float key order) and keeps them in index order.
    masked = masked + 0.0
    order = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :topk_k]
    keep = torch.arange(topk_k, device="cuda").view(1, -1) < seq_lens.view(-1, 1).clamp(max=topk_k)
    reference = torch.sort(torch.where(keep, order, sk), dim=1).values
    reference = torch.where(reference == sk, -1, reference).to(torch.int32)

    actual = _tie_break_path(scores, seq_lens, topk_k, True)

    assert torch.equal(actual["indices"], reference)
    gathered = torch.gather(scores, 1, actual["indices"].clamp(min=0).long())
    valid = actual["indices"] >= 0
    assert torch.equal(_bits(actual["values"])[valid], _bits(gathered)[valid])


def _inf_nan_rows(n_rows, sk, seed):
    """The NaN/inf row plan: per 8-row block NaN at 1000 / past the end / clean /
    at 10 / at L-1 / at L//2, five +inf with no NaN, NaN at 0."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn((n_rows, sk), generator=generator, device="cuda")
    seq_lens = torch.randint(sk - sk // 8, sk + 1, (n_rows,), generator=generator, device="cuda")
    seq_lens = seq_lens.to(torch.int32)
    seq_lens[1::8] = sk - 1
    lengths = seq_lens.tolist()
    for row in range(n_rows):
        length = lengths[row]
        plan = row % 8
        if plan == 0:
            scores[row, min(1000, length - 1)] = float("nan")
        elif plan == 1:
            scores[row, length] = float("nan")
        elif plan == 3:
            scores[row, min(10, length - 1)] = float("nan")
        elif plan == 4:
            scores[row, length - 1] = float("nan")
        elif plan == 5:
            scores[row, length // 2] = float("nan")
        elif plan == 6:
            scores[row, sorted({0, length // 7, length // 3, length // 2, length - 1})] = float(
                "inf"
            )
        elif plan == 7:
            scores[row, 0] = float("nan")
    return scores.contiguous(), seq_lens


@pytest.mark.parametrize("n_rows,sk", [(64, 70000), (16, 1048576)], ids=["70k", "1m"])
def test_infinity_rows_are_complete_and_nan_rows_lose_only_the_nan_slot(monkeypatch, n_rows, sk):
    """Real frontend: a rule that drops +inf members returns the +inf rows five members short.

    +inf members are selected; a row with one in-row (positive, so ranked above +inf) NaN
    has exactly K - 1 members (the NaN's slot is not refilled), equal to the full-row rule
    and to a sort reference.
    """
    _require_tie_break()
    topk_k = 2048
    scores, seq_lens = _inf_nan_rows(n_rows, sk, seed=sk % 1009)

    actual = _tie_break_path(scores, seq_lens, topk_k, True)
    expected = _full_row_rule(scores, seq_lens, topk_k, True, monkeypatch)

    _assert_bitwise_equal(actual, expected, f"inf/nan {n_rows}x{sk}")
    columns = torch.arange(sk, device="cuda").view(1, sk)
    in_row = columns < seq_lens.view(-1, 1).long()
    nan_rows = (torch.isnan(scores) & in_row).any(dim=1)
    counts = (actual["indices"] >= 0).sum(dim=1)
    assert torch.equal(counts, torch.where(nan_rows, topk_k - 1, topk_k).to(counts.dtype))
    keyed = torch.where(in_row & ~torch.isnan(scores), scores, float("-inf"))
    reference = torch.sort(keyed, dim=1, descending=True, stable=True).indices
    budget = torch.where(nan_rows, topk_k - 1, topk_k).view(-1, 1)
    reference = torch.where(
        torch.arange(topk_k, device="cuda").view(1, -1) < budget, reference[:, :topk_k], sk
    )
    reference = torch.sort(reference, dim=1).values
    reference = torch.where(reference == sk, -1, reference).to(torch.int32)
    assert torch.equal(actual["indices"], reference)
    inf_rows = int((torch.isinf(actual["values"]) & (actual["values"] > 0)).any(dim=1).sum())
    _report(f"inf/nan {n_rows}x{sk}: nan_rows={int(nan_rows.sum())} rows_with_inf={inf_rows}")


class _FrontendSpy:
    """Records the real frontend's top-k calls; its signature names ``tie_break``."""

    def __init__(self, namespace):
        self.namespace = namespace
        self.calls = []

    def indexer_top_k_wrapper(
        self, input_values, seq_lens, top_k, next_n=1, return_val=True, tie_break=0
    ):
        self.calls.append((int(input_values.size(0)), bool(return_val), int(tie_break)))
        return self.namespace.indexer_top_k_wrapper(
            input_values,
            seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=return_val,
            tie_break=tie_break,
        )


def test_tie_break_path_reaches_the_frontend_with_tie_break_1(monkeypatch):
    """Without this, a probe regression would make both sides of the equality run the fallback."""
    _require_tie_break()
    n_rows, sk, topk_k = 64, 70000, 2048
    scores, seq_lens = _make_rows("signed_zeros", n_rows, sk, topk_k, seed=71)
    spy = _FrontendSpy(dsa_cudnn_kernels._cudnn_dsa)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", spy)
    assert dsa_cudnn_kernels._supports_lowest_index_topk_tie_break()

    actual = _tie_break_path(scores, seq_lens, topk_k, True)
    assert spy.calls == [(n_rows, False, 1)]
    spy.calls.clear()
    expected = _full_row_rule(scores, seq_lens, topk_k, True, monkeypatch)
    assert spy.calls and all(tie_break == 0 for _, _, tie_break in spy.calls)
    spy.calls.clear()
    dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(scores, seq_lens, topk_k, True)
    assert spy.calls == [(n_rows, True, 0)]

    _assert_bitwise_equal(actual, expected, "spy")


def test_tie_break_path_repeats_bitwise():
    _require_tie_break()
    scores, seq_lens = _make_rows("heavy_ties", 64, 300000, 2048, seed=23)
    first = _tie_break_path(scores, seq_lens, 2048, True)
    for _ in range(3):
        again = _tie_break_path(scores, seq_lens, 2048, True)
        _assert_bitwise_equal(again, first, "repeat")


def test_tie_break_path_graph_replay_matches_eager(monkeypatch):
    _require_tie_break()
    n_rows, sk, topk_k = 64, 70000, 2048
    scores, seq_lens = _make_rows("signed_zeros", n_rows, sk, topk_k, seed=31)
    static_scores = scores.clone()
    static_lens = seq_lens.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _tie_break_path(static_scores, static_lens, topk_k, True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _tie_break_path(static_scores, static_lens, topk_k, True)

    for seed, kind in ((32, "signed_zeros"), (33, "heavy_ties"), (34, "specials")):
        new_scores, new_lens = _make_rows(kind, n_rows, sk, topk_k, seed=seed)
        static_scores.copy_(new_scores)
        static_lens.copy_(new_lens)
        graph.replay()
        torch.cuda.synchronize()
        replayed = {"indices": captured["indices"].clone(), "values": captured["values"].clone()}
        _assert_bitwise_equal(
            replayed, _full_row_rule(new_scores, new_lens, topk_k, True, monkeypatch), kind
        )
        _assert_bitwise_equal(replayed, _tie_break_path(new_scores, new_lens, topk_k, True), kind)


def test_zero_tie_kernel_matches_torch_on_long_rows():
    _require_frontend()
    if not dsa_topk_zero_ties.HAVE_TRITON:
        pytest.skip("Triton is not installed")
    generator = torch.Generator(device="cuda").manual_seed(41)
    n_rows, sk, topk_k = 32, 1048576, 2048
    pick = torch.rand((n_rows, sk), generator=generator, device="cuda")
    scores = torch.where(pick < 0.001, torch.zeros_like(pick), pick)
    scores = torch.where(pick < 0.0005, torch.full_like(pick, -0.0), scores)
    scores[:, -5:] = -0.0
    row_limits = torch.randint(0, sk + 1, (n_rows,), generator=generator, device="cuda")
    row_limits[0] = sk
    row_limits = row_limits.to(torch.int32)
    zero_counts = ((scores == 0) & (torch.arange(sk, device="cuda") < row_limits.view(-1, 1))).sum(
        dim=1, dtype=torch.int32
    )
    quotas = torch.minimum(zero_counts, torch.full_like(zero_counts, topk_k))
    quotas[1::4] = 0

    triton_result = dsa_topk_zero_ties._select_lowest_index_zero_ties_triton(
        scores, row_limits, quotas, topk_k
    )
    torch_result = dsa_topk_zero_ties._select_lowest_index_zero_ties_torch(
        scores, row_limits, quotas, topk_k
    )

    assert torch.equal(triton_result, torch_result)


def test_frontend_without_tie_break_falls_back_to_the_full_row_rule(monkeypatch):
    _require_frontend()
    if dsa_cudnn_kernels._supports_lowest_index_topk_tie_break():
        pytest.skip("this frontend supports tie_break; the fallback is covered on CPU")
    scores, seq_lens = _make_rows("signed_zeros", 32, 8192, 512, seed=51)

    actual = _tie_break_path(scores, seq_lens, 512, True)

    _assert_bitwise_equal(actual, _full_row_rule(scores, seq_lens, 512, True, monkeypatch), "fb")
    _report(f"fallback frontend={dsa_cudnn_kernels._cudnn_frontend_version()} equal")


def _timed(fn, repeats=3):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return sorted(samples)[len(samples) // 2]


def test_1m_columns_launch_1024_rows_and_select_the_right_set(monkeypatch):
    """2048 back rows of the 1M CP64 rank-0 call: two launches of exactly 2**31 elements."""
    _require_frontend()
    n_rows, sk, topk_k = 2048, 1048576, 2048
    assert dsa_cudnn_kernels._topk_wrapper_chunk_rows(n_rows, sk, 4) == 1024
    generator = torch.Generator(device="cuda").manual_seed(61)
    scores = torch.randn((n_rows, sk), generator=generator, device="cuda")
    seq_lens = torch.randint(
        1007873, 1015809, (n_rows,), generator=generator, device="cuda", dtype=torch.int32
    )
    namespace = dsa_cudnn_kernels._cudnn_dsa

    class CountingNamespace:
        calls = []

        @staticmethod
        def indexer_top_k_wrapper(chunk_scores, *args, **kwargs):
            CountingNamespace.calls.append(chunk_scores.size(0))
            return namespace.indexer_top_k_wrapper(chunk_scores, *args, **kwargs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with monkeypatch.context() as patch:
        patch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", CountingNamespace)
        result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
            scores, seq_lens, topk_k, True, deterministic=False
        )
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline
    _report(f"1m non-det launches={CountingNamespace.calls} peak_extra_gib={peak_extra / _GIB:.3f}")
    assert CountingNamespace.calls == [1024, 1024]
    # 8 GiB scratch, outputs and allocator rounding; two live scratch buffers would be 16 GiB.
    assert peak_extra < 8.6 * _GIB

    columns = torch.arange(sk, device="cuda").view(1, sk)
    masked = torch.where(columns < seq_lens.view(-1, 1).long(), scores, float("-inf"))
    expected_values = torch.topk(masked, topk_k, dim=1).values
    got_values = torch.sort(result["values"], dim=1, descending=True).values
    assert torch.equal(got_values, expected_values)
    gathered = torch.gather(scores, 1, result["indices"].long())
    assert torch.equal(_bits(gathered), _bits(result["values"]))
    assert bool((result["indices"].long() < seq_lens.view(-1, 1).long()).all())

    non_det_ms = _timed(
        lambda: dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(scores, seq_lens, topk_k, True)
    )
    _report(f"1m 2048 rows non-det {non_det_ms:.2f} ms")
    if dsa_cudnn_kernels._supports_lowest_index_topk_tie_break():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        _tie_break_path(scores, seq_lens, topk_k, True)
        torch.cuda.synchronize()
        det_peak_extra = torch.cuda.max_memory_allocated() - baseline
        _report(f"1m det tie_break peak_extra_gib={det_peak_extra / _GIB:.3f}")
        assert det_peak_extra < 8.8 * _GIB
        det_ms = _timed(lambda: _tie_break_path(scores, seq_lens, topk_k, True))
        full_row_ms = _timed(
            lambda: _full_row_rule(scores, seq_lens, topk_k, True, monkeypatch), repeats=1
        )
        _report(f"1m 2048 rows det tie_break {det_ms:.2f} ms, det full-row {full_row_ms:.2f} ms")
        _assert_bitwise_equal(
            _tie_break_path(scores, seq_lens, topk_k, True),
            _full_row_rule(scores, seq_lens, topk_k, True, monkeypatch),
            "1m",
        )


def _zero_cutoff_1m_rows(n_rows, sk, topk_k, seed):
    """1M index-score rows with fewer than K positives: every long row's cutoff is a signed zero."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens = _lengths(n_rows, sk, topk_k, generator)
    uniform = torch.rand((n_rows, sk), generator=generator, device="cuda")
    sign = torch.rand((n_rows, sk), generator=generator, device="cuda") < 0.5
    # About 1000 positives per row on 64 quantized levels (heavy ties above the cutoff),
    # 41% zeros of random sign, the rest quantized negatives.
    scores = torch.where(sign, torch.zeros_like(uniform), torch.full_like(uniform, -0.0))
    scores = torch.where(uniform >= 0.411, -torch.round(uniform * 16) / 64 - 1 / 64, scores)
    scores = torch.where(uniform < 0.001, torch.round(uniform * 64000) / 64 + 1 / 64, scores)
    del uniform, sign
    columns = torch.arange(sk, device="cuda").view(1, sk)
    scores = torch.where(columns < seq_lens.view(n_rows, 1).long(), scores, float("nan"))
    return scores.contiguous(), seq_lens


def test_1m_signed_zero_cutoff_rows_match_the_full_row_rule(monkeypatch):
    """2048 rows of 1M columns, all with a +-0 cutoff: the zero-tie repair runs on every row."""
    _require_tie_break()
    n_rows, sk, topk_k = 2048, 1048576, 2048
    scores, seq_lens = _zero_cutoff_1m_rows(n_rows, sk, topk_k, seed=81)

    actual = _tie_break_path(scores, seq_lens, topk_k, True)
    expected = _full_row_rule(scores, seq_lens, topk_k, True, monkeypatch)

    _assert_bitwise_equal(actual, expected, "1m signed-zero cutoffs")
    long_rows = seq_lens > topk_k
    zero_members = ((actual["values"] == 0) & (actual["indices"] >= 0)).any(dim=1)
    assert bool(zero_members[long_rows].all())
    valid = (actual["indices"] >= 0) & torch.isfinite(actual["values"])
    cutoff = torch.where(valid, actual["values"], float("inf")).amin(dim=1)
    zero_cutoff_rows = int(((cutoff == 0) & long_rows).sum())
    assert zero_cutoff_rows > n_rows // 2
    det_ms = _timed(lambda: _tie_break_path(scores, seq_lens, topk_k, True))
    _report(
        f"1m signed-zero cutoffs long_rows={int(long_rows.sum())} zero_cutoff_rows="
        f"{zero_cutoff_rows} equal; det tie_break {det_ms:.2f} ms"
    )


def _reserved_by(fn):
    """Bytes newly reserved by ``fn`` after releasing the cache (one value per segment)."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    return result, torch.cuda.max_memory_reserved() - reserved


@pytest.mark.parametrize("deterministic", [False, True], ids=["non_det", "det"])
def test_1m_chunked_top_k_reserves_one_scratch_block(deterministic):
    """Two 1024-row launches must reuse one 8 GiB scratch block, eagerly and in a graph pool.

    Chunk results kept until a final concatenation were carved out of the freed scratch block,
    so the second launch reserved another 8 GiB (16.02 GiB graph pool at 1M).
    """
    _require_frontend()
    if deterministic and not dsa_cudnn_kernels._supports_lowest_index_topk_tie_break():
        pytest.skip("the full-row fallback launches 31-row chunks at 1M")
    n_rows, sk, topk_k = 2048, 1048576, 2048
    generator = torch.Generator(device="cuda").manual_seed(91)
    scores = torch.randn((n_rows, sk), generator=generator, device="cuda")
    seq_lens = torch.randint(
        1007873, 1015809, (n_rows,), generator=generator, device="cuda", dtype=torch.int32
    )

    def call():
        return dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
            scores, seq_lens, topk_k, True, deterministic=deterministic
        )

    call()
    eager, eager_reserved = _reserved_by(call)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()

    def capture():
        with torch.cuda.graph(graph):
            return call()

    captured, graph_reserved = _reserved_by(capture)
    graph.replay()
    torch.cuda.synchronize()
    _report(
        f"1m 2048 rows {'det' if deterministic else 'non-det'} eager_reserved_gib="
        f"{eager_reserved / _GIB:.3f} graph_pool_gib={graph_reserved / _GIB:.3f}"
    )
    if deterministic:
        _assert_bitwise_equal(captured, eager, "1m graph replay")
    else:
        assert torch.equal(
            torch.sort(captured["values"], dim=1).values, torch.sort(eager["values"], dim=1).values
        )
    # One 8 GiB scratch block plus outputs and allocator rounding; one per launch is 16 GiB.
    assert eager_reserved < 8.6 * _GIB
    assert graph_reserved < 8.6 * _GIB
    del graph, captured
