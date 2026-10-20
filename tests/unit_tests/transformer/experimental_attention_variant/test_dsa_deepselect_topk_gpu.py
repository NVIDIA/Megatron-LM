# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real DeepSelect checks of ``dsa_indexer_topk_backend='deepselect'`` on SM 10.0 / 10.3.

The file runs on every SM the builds ship kernels for (``COMPILED_SM``), including SM 10.0,
which production keeps on cuDNN (``SUPPORTED_SM``) until this file has passed on a GB200; the
tests enable the device for themselves and log that they do.

Needs a DeepSelect build for the device (the GLM-5.2 ``ds_tune`` package or upstream
``deep_select``; the log names the one used) and the cuDNN frontend DSA namespace. Covered:

* contract: against the cuDNN radix top-k through ``_indexer_top_k_wrapper_chunked`` on
  Gaussian, heavy-tie, signed-zero/infinity, short, empty and garbage-tailed rows at 128K and
  1M widths: identical short rows (indices, value and padding bits), K distinct in-row members,
  payload equal to the scores at the indices, the same selected-value multiset, and the same
  member set on Gaussian rows;
* long rows with fewer than K selectable scores (masked ``-inf`` heads, as varlen ``starts``
  produce): every finite score selected, no finite payload at an out-of-row index;
* NaN rows: no trap, flagged rows become the first K in-row columns with NaN scores, every
  other row is bitwise what it is without the NaN (batch independence);
* determinism: bitwise repeats, and a row subset selects what the full batch selects;
* CUDA graph: capture with static buffers, replay after an in-place data swap equals eager,
  and the graph pool holds no top-k scratch (cuDNN reserves its 8 GiB budget);
* strided scores: a row-padded score view is read in place, and the cuDNN scorer binds a
  row-padded ``out`` directly (no staging copy) with and without ``clean_logits``;
* the GLM-5.2 1M CP64 rank-0 packed THD call end to end against the cuDNN backend.

Lines starting with ``[deepselect-gpu]`` report timings and memory for the validation log.
"""

import gc
import time

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_deepselect_topk,
)

_GIB = 1024**3
_K = 2048


@pytest.fixture(autouse=True)
def _enable_compiled_devices(monkeypatch):
    """Let an SM with compiled but not yet validated kernels run here: this is the validation."""
    if torch.cuda.is_available():
        sm = tuple(torch.cuda.get_device_capability())
        supported = dsa_deepselect_topk.SUPPORTED_SM
        if sm in dsa_deepselect_topk.COMPILED_SM and sm not in supported:
            monkeypatch.setattr(dsa_deepselect_topk, "SUPPORTED_SM", supported | {sm})
            _report(f"validating SM {sm[0]}.{sm[1]}, which production keeps on cuDNN")


def _require_deepselect():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if tuple(torch.cuda.get_device_capability()) not in dsa_deepselect_topk.COMPILED_SM:
        pytest.skip("DeepSelect builds ship kernels for SM 10.0 and 10.3 only")
    provider = dsa_deepselect_topk._load_provider()
    if provider is None:
        pytest.skip(dsa_deepselect_topk._IMPORT_STATE["error"])
    _report(f"provider {provider.description}")
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("cuDNN frontend DSA is not installed")
    dsa_cudnn_kernels._ensure_dsa_namespace()


def _report(message: str) -> None:
    print(f"[deepselect-gpu] {message}", flush=True)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def _topk(scores, seq_lens, backend, return_topk_scores=True, topk_k=_K):
    return dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k, return_topk_scores, topk_backend=backend
    )


def _ran_deepselect(scores, seq_lens, topk_k=_K) -> bool:
    return dsa_deepselect_topk.fallback_reason(scores, seq_lens, topk_k, False) is None


def _require_eligible(scores, seq_lens, topk_k=_K) -> None:
    """Skip widths below the loaded build's minimum (upstream deep_select starts at 512K)."""
    reason = dsa_deepselect_topk.fallback_reason(scores, seq_lens, topk_k, False)
    if reason is not None and reason.startswith("score width below"):
        pytest.skip(reason)
    assert reason is None, reason


def _fill_rows(kind, scores, seq_lens, generator):
    if kind == "ties":
        scores.mul_(4).round_().div_(64)
    elif kind == "zeros_inf":
        pick = torch.rand(scores.shape, generator=generator, device="cuda")
        scores.abs_()
        scores.masked_fill_(pick < 0.4, 0.0)
        scores.masked_fill_(pick < 0.2, -0.0)
        scores.masked_fill_(pick > 0.9995, float("inf"))
        scores.masked_fill_((pick > 0.5) & (pick < 0.6), float("-inf"))
    else:
        assert kind == "gauss"
    # Past a row's length the scorer leaves garbage with clean_logits=False: poison it.
    columns = torch.arange(scores.size(1), device="cuda").view(1, -1)
    scores.masked_fill_(columns >= seq_lens.view(-1, 1).long(), float("nan"))


def _make_rows(kind, n_rows, width, seed, buffer=None):
    """Scores (written into ``buffer`` if given, 512 rows at a time) and per-row lengths."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens = torch.randint(_K + 1, width + 1, (n_rows,), generator=generator, device="cuda")
    seq_lens[:6] = torch.tensor([0, 1, _K - 1, _K, _K + 1, width], device="cuda")
    seq_lens = seq_lens.to(torch.int32)
    if buffer is None:
        buffer = torch.empty((n_rows, width), device="cuda")
    for start in range(0, n_rows, 512):
        rows = buffer[start : start + 512]
        rows.copy_(torch.randn(rows.shape, generator=generator, device="cuda"))
        _fill_rows(kind, rows, seq_lens[start : start + 512], generator)
    return buffer, seq_lens


def _check_contract(got, want, scores, seq_lens, label, same_members):
    lengths = seq_lens.long()
    short = lengths <= _K
    assert torch.equal(got["indices"][short], want["indices"][short]), label
    assert torch.equal(_bits(got["values"])[short], _bits(want["values"])[short]), label
    indices = got["indices"][~short].long()
    assert bool((indices >= 0).all()) and bool((indices < lengths[~short].view(-1, 1)).all())
    ordered = torch.sort(indices, dim=1).values
    assert bool((ordered[:, 1:] != ordered[:, :-1]).all()), f"{label}: duplicate members"
    gathered = torch.gather(scores[~short], 1, indices)
    assert torch.equal(_bits(gathered), _bits(got["values"][~short])), f"{label}: payload"
    got_sorted = torch.sort(got["values"][~short], dim=1).values
    want_sorted = torch.sort(want["values"][~short], dim=1).values
    assert torch.equal(got_sorted, want_sorted), f"{label}: selected values differ"
    differing = (
        (torch.sort(got["indices"], dim=1).values != torch.sort(want["indices"], dim=1).values)
        .any(dim=1)
        .sum()
    )
    if same_members:
        assert int(differing) == 0, f"{label}: {int(differing)} rows select other members"
    return int(differing)


@pytest.mark.parametrize("kind", ["gauss", "ties", "zeros_inf"])
@pytest.mark.parametrize("n_rows,width", [(512, 131072), (64, 1048576)], ids=["128k", "1m"])
def test_contract_matches_the_cudnn_top_k(kind, n_rows, width):
    _require_deepselect()
    scores, seq_lens = _make_rows(kind, n_rows, width, seed=width % 997 + len(kind))
    _require_eligible(scores, seq_lens)

    for return_topk_scores in (True, False):
        got = _topk(scores, seq_lens, "deepselect", return_topk_scores)
        want = _topk(scores, seq_lens, "cudnn", True)
        if not return_topk_scores:
            assert got["values"] is None
            with_values = _topk(scores, seq_lens, "deepselect", True)
            assert torch.equal(got["indices"], with_values["indices"])
            continue
        differing = _check_contract(
            got, want, scores, seq_lens, f"{kind} {n_rows}x{width}", kind == "gauss"
        )
    _report(f"contract {kind} {n_rows}x{width}: rows with other tied members={differing}")


@pytest.mark.parametrize("n_rows,width", [(256, 131072), (32, 1048576)], ids=["128k", "1m"])
def test_long_rows_with_few_selectable_scores_match_the_cudnn_top_k(n_rows, width):
    """Rows longer than K whose head is masked to ``-inf`` (varlen ``starts``), leaving fewer
    than K finite scores, beside rows with more. The kernel pads partial tail segments with an
    ``-inf`` placeholder that ties with in-row ``-inf``: whatever fills the ``-inf`` slots, every
    finite score must be selected, a finite payload must sit at its own in-row index, and the
    in-bounds finite members (what ``_indexer_topk_bshd`` keeps) must equal cuDNN's."""
    _require_deepselect()
    generator = torch.Generator(device="cuda").manual_seed(width % 991)
    seq_lens = torch.randint(width // 2, width + 1, (n_rows,), generator=generator, device="cuda")
    seq_lens[:4] = torch.tensor([width, width - 1, _K + 1, width // 2 + 7], device="cuda")
    seq_lens = seq_lens.to(torch.int32)
    lengths = seq_lens.long()
    # Finite tails of 1 .. 3K scores: rows below K have fewer selectable scores than slots.
    finite = torch.randint(1, 3 * _K, (n_rows,), generator=generator, device="cuda")
    finite[:4] = torch.tensor([1, _K - 1, _K + 1, _K // 2], device="cuda")
    finite = torch.minimum(finite, lengths)
    starts = lengths - finite
    scores = torch.randn((n_rows, width), generator=generator, device="cuda")
    columns = torch.arange(width, device="cuda").view(1, -1)
    scores.masked_fill_(columns < starts.view(-1, 1), float("-inf"))
    scores.masked_fill_(columns >= lengths.view(-1, 1), float("nan"))
    _require_eligible(scores, seq_lens)

    got = _topk(scores, seq_lens, "deepselect")
    want = _topk(scores, seq_lens, "cudnn")

    def kept(result):
        indices, values = result["indices"].long(), result["values"]
        in_bounds = (indices >= starts.view(-1, 1)) & (indices < lengths.view(-1, 1))
        return torch.where(in_bounds & torch.isfinite(values), indices, -1)

    got_kept, want_kept = kept(got), kept(want)
    assert torch.equal((got_kept >= 0).sum(dim=1), torch.clamp(finite, max=_K))
    assert torch.equal(torch.sort(got_kept, dim=1).values, torch.sort(want_kept, dim=1).values)
    finite_slots = torch.isfinite(got["values"])
    indices = got["indices"].long()
    in_row = (indices >= 0) & (indices < lengths.view(-1, 1))
    assert bool((in_row | ~finite_slots).all()), "finite payload at an out-of-row index"
    gathered = torch.gather(scores, 1, indices.clamp(min=0, max=width - 1))
    assert torch.equal(_bits(gathered[finite_slots]), _bits(got["values"][finite_slots]))
    assert bool((got["values"][~finite_slots] == float("-inf")).all()), "non-finite slot not -inf"
    off_row = int((~in_row & ~finite_slots).sum())
    _report(f"few-selectable {n_rows}x{width}: -inf slots outside the row={off_row}")


def _nan_plan(scores, seq_lens):
    """The head-to-head plan on rows 8..15: NaN at 1000 / past the end / none / at 10 / at
    L-1 / at L//2, five +inf and no NaN, NaN at 0."""
    lengths = seq_lens.tolist()
    nan_rows = []
    for offset in range(8):
        row = 8 + offset
        length = lengths[row]
        if offset == 0:
            scores[row, min(1000, length - 1)] = float("nan")
        elif offset == 1:
            scores[row, length] = float("nan")
        elif offset == 3:
            scores[row, min(10, length - 1)] = float("nan")
        elif offset == 4:
            scores[row, length - 1] = float("nan")
        elif offset == 5:
            scores[row, length // 2] = float("nan")
        elif offset == 6:
            columns = sorted({0, length // 7, length // 3, length // 2, length - 1})
            scores[row, columns] = float("inf")
        elif offset == 7:
            scores[row, 0] = float("nan")
        if offset in (0, 3, 4, 5, 7):
            nan_rows.append(row)
    return nan_rows


def test_nan_rows_are_repaired_and_leave_other_rows_untouched():
    _require_deepselect()
    n_rows, width = 64, 1048576
    scores, seq_lens = _make_rows("gauss", n_rows, width, seed=7)
    # Rows 8..15 get long rows of fresh scores (no poison inside them) for the NaN plan.
    seq_lens[8:16] = torch.randint(width - 50000, width - 1, (8,), device="cuda").to(torch.int32)
    scores[8:16].normal_()
    columns = torch.arange(width, device="cuda").view(1, -1)
    scores[8:16].masked_fill_(columns >= seq_lens[8:16].view(-1, 1).long(), float("nan"))
    clean = _topk(scores, seq_lens, "deepselect")
    clean = {name: value.clone() for name, value in clean.items()}
    nan_rows = _nan_plan(scores, seq_lens)

    result = _topk(scores, seq_lens, "deepselect")
    torch.ones(4, device="cuda").add_(1).sum().item()  # the context is still usable

    arange = torch.arange(_K, dtype=torch.int32, device="cuda")
    for row in nan_rows:
        assert torch.equal(result["indices"][row], arange), row
        assert bool(torch.isnan(result["values"][row]).all()), row
    untouched = [row for row in range(n_rows) if row not in nan_rows and row != 14]
    assert torch.equal(result["indices"][untouched], clean["indices"][untouched])
    assert torch.equal(_bits(result["values"][untouched]), _bits(clean["values"][untouched]))
    inf_members = int(torch.isinf(result["values"][14]).sum())
    assert inf_members == 5 and int((result["indices"][14] >= 0).sum()) == _K
    _report(f"nan rows repaired={len(nan_rows)} +inf row members={inf_members}")


def test_selection_is_bitwise_repeatable_and_batch_independent():
    _require_deepselect()
    scores, seq_lens = _make_rows("ties", 256, 1048576, seed=11)
    first = _topk(scores, seq_lens, "deepselect")
    first = {name: value.clone() for name, value in first.items()}
    for _ in range(3):
        again = _topk(scores, seq_lens, "deepselect")
        assert torch.equal(again["indices"], first["indices"])
        assert torch.equal(_bits(again["values"]), _bits(first["values"]))
    subset = slice(64, 128)
    part = _topk(scores[subset], seq_lens[subset], "deepselect")
    assert torch.equal(part["indices"], first["indices"][subset])


def _graph_case(n_rows, width, backend, seeds=(21, 22, 23)):
    scores, seq_lens = _make_rows("gauss", n_rows, width, seed=seeds[0])
    static_scores, static_lens = scores, seq_lens.clone()

    def call():
        return _topk(static_scores, static_lens, backend)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    torch.cuda.synchronize()
    pool = torch.cuda.memory_reserved() - reserved
    for seed, kind in zip(seeds[1:], ("ties", "zeros_inf")):
        _, new_lens = _make_rows(kind, n_rows, width, seed=seed, buffer=static_scores)
        static_lens.copy_(new_lens.flip(0))  # other lengths than at capture
        columns = torch.arange(width, device="cuda").view(1, -1)
        for start in range(0, n_rows, 512):
            rows = static_scores[start : start + 512]
            rows.masked_fill_(
                columns >= static_lens[start : start + 512].view(-1, 1).long(), float("nan")
            )
            rows.nan_to_num_(nan=0.25, posinf=float("inf"), neginf=float("-inf"))
            rows.masked_fill_(
                columns >= static_lens[start : start + 512].view(-1, 1).long(), float("nan")
            )
        graph.replay()
        torch.cuda.synchronize()
        eager = _topk(static_scores, static_lens, backend)
        if backend == "deepselect":
            assert torch.equal(captured["indices"], eager["indices"]), kind
            assert torch.equal(_bits(captured["values"]), _bits(eager["values"])), kind
        else:
            assert torch.equal(
                torch.sort(captured["values"], dim=1).values,
                torch.sort(eager["values"], dim=1).values,
            )
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    del graph, captured
    return pool, sorted(times)[len(times) // 2]


@pytest.mark.parametrize("n_rows,width", [(4096, 131072), (16384, 1048576)], ids=["128k", "1m"])
def test_graph_replay_after_a_data_swap_matches_eager(n_rows, width):
    _require_deepselect()
    free, _ = torch.cuda.mem_get_info()
    if free < (n_rows * width * 4) * 1.5 + 10 * _GIB:
        pytest.skip("not enough device memory")
    probe_scores = torch.empty((1, width), device="cuda")
    _require_eligible(probe_scores, torch.ones((1,), dtype=torch.int32, device="cuda"))
    del probe_scores
    ds_pool, ds_ms = _graph_case(n_rows, width, "deepselect")
    gc.collect()
    torch.cuda.empty_cache()
    fe_pool, fe_ms = _graph_case(n_rows, width, "cudnn")
    _report(
        f"graph {n_rows}x{width}: deepselect {ds_ms:.3f} ms pool {ds_pool / _GIB:.3f} GiB; "
        f"cudnn {fe_ms:.3f} ms pool {fe_pool / _GIB:.3f} GiB"
    )
    # Outputs only (16384 x 2048 int32 + fp32 = 0.25 GiB at 1M), no top-k scratch.
    assert ds_pool < 0.5 * _GIB


def test_row_padded_score_view_is_read_in_place():
    _require_deepselect()
    n_rows, width, stride = 128, 100004, 100096
    buffer = torch.empty((n_rows, stride), device="cuda")
    view, seq_lens = _make_rows("gauss", n_rows, width, seed=31, buffer=buffer[:, :width])
    _require_eligible(view, seq_lens)
    assert not _ran_deepselect(view.contiguous(), seq_lens)

    padded = _topk(view, seq_lens, "deepselect")
    reference = _topk(view.contiguous(), seq_lens, "cudnn")

    _check_contract(padded, reference, view, seq_lens, "padded view", True)


def _thd_inputs(seq_lens, heads=32, dim=128, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn((total, heads, dim), generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn((total, 1, dim), generator=generator, device="cuda").to(torch.bfloat16)
    w = torch.rand((total, heads), generator=generator, device="cuda").to(torch.bfloat16)
    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + length)
    cu = torch.tensor(cu, dtype=torch.int32, device="cuda")
    return q, k, w, cu


@pytest.mark.parametrize("clean_logits", [True, False])
def test_cudnn_scorer_binds_a_row_padded_out_without_a_copy(clean_logits):
    _require_deepselect()
    if not dsa_cudnn_kernels._cudnn_frontend_at_least(dsa_cudnn_kernels._SCORE_OUT_FE_VERSION):
        pytest.skip("frontend older than 1.31")
    if not clean_logits and not dsa_cudnn_kernels._supports_indexer_clean_logits():
        pytest.skip("frontend has no clean_logits")
    seq_lens = (4100, 900)
    width = max(seq_lens)
    q, k, w, cu = _thd_inputs(seq_lens, seed=41)
    kwargs = dict(
        ratio=1,
        sm_scale=1.0,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=width,
        max_seqlen_k=width,
    )
    if not clean_logits:
        kwargs["clean_logits"] = False
    reference = dsa_cudnn_kernels._cudnn_dsa.indexer_forward_wrapper(q, k, w, **kwargs)["scores"]
    buffer = torch.full((q.size(0), 4352), 3.0, device="cuda")
    out = buffer[:, :width]

    scores = dsa_cudnn_kernels._cudnn_dsa.indexer_forward_wrapper(q, k, w, out=out, **kwargs)[
        "scores"
    ]

    assert scores.data_ptr() == out.data_ptr() and scores.stride() == (4352, 1)
    assert bool((buffer[:, width:] == 3.0).all()), "the padding columns were written"
    rows = torch.arange(q.size(0), device="cuda")
    starts = torch.where(rows < seq_lens[0], 0, seq_lens[0])
    limits = (rows - starts + 1).view(-1, 1)
    defined = torch.arange(width, device="cuda").view(1, -1) < limits
    assert torch.equal(_bits(scores)[defined], _bits(reference)[defined])


def _packed_cp_layout(seq_lens, cp_size, cp_rank):
    divisor = 2 * cp_size
    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + length)
    starts, ends = [], []
    for seq, length in enumerate(seq_lens):
        half = length // divisor
        front = torch.arange(cp_rank * half, (cp_rank + 1) * half)
        back = torch.arange((divisor - 1 - cp_rank) * half, (divisor - cp_rank) * half)
        positions = torch.cat((front, back))
        starts.append(torch.full_like(positions, cu[seq]))
        ends.append(cu[seq] + positions + 1)
    return (
        torch.cat(starts).cuda(),
        torch.cat(ends).cuda(),
        torch.tensor(cu, dtype=torch.int32, device="cuda"),
    )


def _median_ms(fn, repeats=3):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return sorted(samples)[len(samples) // 2]


def test_glm_1m_cp64_rank0_packed_thd_call_matches_cudnn():
    """The 1M CP64 rank-0 layout of the GLM-5.2 recipe (shapes.json one_m_cp64_rank0)."""
    _require_deepselect()
    free, _ = torch.cuda.mem_get_info()
    if free < 100 * _GIB:
        pytest.skip("needs about 100 GiB of free device memory")
    cp_size, cp_rank, seq_lens = 64, 0, (1015808, 32768)
    starts, ends, cu = _packed_cp_layout(seq_lens, cp_size, cp_rank)
    sq, sk = starts.numel(), sum(seq_lens)
    generator = torch.Generator(device="cuda").manual_seed(51)
    q = torch.randn((1, sq, 32, 128), generator=generator, device="cuda").to(torch.bfloat16)
    k = torch.randn((1, sk, 128), generator=generator, device="cuda").to(torch.bfloat16)
    w = torch.rand((1, sq, 32), generator=generator, device="cuda").to(torch.bfloat16)

    def call(backend, return_topk_scores=True):
        return dsa_cudnn_kernels._indexer_topk_bshd(
            q,
            k,
            w,
            _K,
            varlen_starts=starts,
            varlen_ends=ends,
            return_scores=False,
            return_topk_scores=return_topk_scores,
            use_local_indexer_varlen=True,
            packed_cu_seqlens_q=cu,
            packed_cu_seqlens_k=cu,
            packed_max_seqlen_q=sk,
            packed_max_seqlen_k=sk,
            packed_cp_size=cp_size,
            local_packed_cp_rank=cp_rank,
            scorer_options=dsa_cudnn_kernels.IndexerScorerOptions(
                skip_masked_score_fill=True, topk_backend=backend
            ),
        )

    got_indices, got_length, got_scores = call("deepselect")
    want_indices, want_length, want_scores = call("cudnn")
    assert torch.equal(got_length, want_length)
    assert torch.equal(
        torch.sort(got_scores, dim=-1).values, torch.sort(want_scores, dim=-1).values
    )
    differing = int(
        (torch.sort(got_indices, dim=-1).values != torch.sort(want_indices, dim=-1).values)
        .any(dim=-1)
        .sum()
    )
    attention = call("deepselect", False)
    reference = call("cudnn", False)
    assert torch.equal(attention[1], reference[1])
    attention_differing = int((attention[0] != reference[0]).any(dim=-1).sum())
    ds_ms = _median_ms(lambda: call("deepselect", False))
    fe_ms = _median_ms(lambda: call("cudnn", False))
    _report(
        f"1m cp64 rank0 packed THD rows={sq} width={sk}: rows with other tied members "
        f"loss={differing} attention={attention_differing}; scorer+top-k eager "
        f"deepselect {ds_ms:.2f} ms, cudnn {fe_ms:.2f} ms"
    )
    # Only exact FP32 ties at the cutoff may pick other members.
    assert differing <= sq // 100 and attention_differing <= sq // 100
