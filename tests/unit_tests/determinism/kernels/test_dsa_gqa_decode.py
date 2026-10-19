# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SDSA decode must be batch invariant in eager mode and CUDA Graph replay."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_gqa
from megatron.core.transformer.experimental_attention_variant import (
    dsa_gqa_decode_kernel as kernels,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import _simplified_decode_topk
from tests.unit_tests.determinism.kernels.harness import bytes_equal, seeded

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAVE_CUTEDSL), reason="CUDA and CuTeDSL required"
)


@pytest.fixture(params=["deepselect", "torch"])
def decode_topk_backend(request, monkeypatch):
    """Exercise DeepSelect explicitly so a silent fallback cannot satisfy its regressions."""
    if request.param == "torch":
        monkeypatch.setattr(dsa_gqa, "deep_select", None)
        yield
        return
    if not dsa_gqa._deepselect_decode_supported(torch.device("cuda"), 512):
        pytest.skip("DeepSelect and SM100/103 required")
    calls = []
    original_topk = dsa_gqa.deep_select.topk

    def capture_topk(*args, **kwargs):
        calls.append(1)
        return original_topk(*args, **kwargs)

    monkeypatch.setattr(dsa_gqa.deep_select, "topk", capture_topk)
    yield
    assert calls, "Decode must actually invoke DeepSelect on supported inputs"


def _inputs(head_dim, *, pages=3, lengths=(3, 65, 137), topk=16):
    seeded(17)
    batch, page_size, heads, groups = len(lengths), 64, 4, 2

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    return {
        "index_query": randn(batch, head_dim),
        "index_key": randn(batch * pages, page_size, head_dim),
        # Preserve the pitched token stride of a QKV projection view.
        "query": randn(batch, heads + 2, head_dim)[:, :heads],
        "key": randn(batch * pages, page_size, groups, head_dim),
        "value": randn(batch * pages, page_size, groups, head_dim),
        "table": torch.randperm(batch * pages, device="cuda").view(batch, pages).to(torch.int32),
        "lengths": torch.tensor(lengths, device="cuda", dtype=torch.int32),
        "indices": torch.arange(topk, device="cuda", dtype=torch.int32)
        .expand(batch, -1)
        .contiguous(),
        "scores": torch.empty(batch, max(lengths) + 14, device="cuda", dtype=torch.float32),
        "output": torch.empty(batch, heads, head_dim, device="cuda", dtype=torch.bfloat16),
        "scale": head_dim**-0.5,
    }


def _indexer(data):
    kernels.dsa_gqa_decode_indexer_score(
        data["index_query"], data["index_key"], data["table"], data["lengths"], out=data["scores"]
    )


def _attention(data):
    kernels.dsa_gqa_decode_attention(
        data["query"],
        data["key"],
        data["value"],
        data["table"],
        data["indices"],
        data["lengths"],
        data["scale"],
        out=data["output"],
    )


@pytest.mark.parametrize("head_dim", [128, 192, 256])
@pytest.mark.parametrize("operation", ["indexer", "attention"])
def test_decode_kernels_cuda_graph_replay(monkeypatch, head_dim, operation):
    """Default-stream compilation must also work on different capture streams."""
    for name in ("_COMPILED", "_OPERATORS", "_ATTENTION_COMPILED", "_ATTENTION_OPERATORS"):
        monkeypatch.setattr(kernels, name, {})
    compile_calls = []
    original_compile = kernels.cute.compile

    def count_compile(*args, **kwargs):
        compile_calls.append(1)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(kernels.cute, "compile", count_compile)
    data = _inputs(head_dim)
    if operation == "indexer":
        run, mutable, output = _indexer, data["index_query"], data["scores"]
    else:
        run, mutable, output = _attention, data["value"], data["output"]

    # Warm up before capture, as production does. The stream must remain a
    # runtime argument even when the compiled callable is reused.
    run(data)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(data)
        # Finish any stray eager launch from a broken capture before poisoning
        # the output; otherwise that launch could hide an empty graph.
        torch.cuda.synchronize()
        for factor in (-1.0, 0.5, -1.0):
            mutable.mul_(factor)
            run(data)
            expected = output.clone()
            torch.cuda.synchronize()
            for _ in range(3):
                output.fill_(torch.nan)
                graph.replay()
                torch.cuda.synchronize()
                assert bytes_equal(
                    output, expected
                ), f"{operation} graph replay did not reproduce eager output after input update"
    assert len(compile_calls) == 1


def _subset(data, rows, *, remap_pages=False):
    selected = dict(data)
    for name in ("index_query", "query", "table", "lengths", "indices"):
        selected[name] = data[name][rows].contiguous()
    if remap_pages:
        pages = selected["table"].long().flatten()
        order = torch.randperm(pages.numel(), device="cuda")
        for name in ("index_key", "key", "value"):
            selected[name] = data[name][pages[order]].contiguous()
        selected["table"] = order.argsort().reshape_as(selected["table"]).to(torch.int32)
    width = int(selected["lengths"].max())
    selected["scores"] = data["scores"].new_empty(len(rows), width)
    selected["output"] = torch.empty_like(data["output"][rows])
    return selected


@pytest.mark.parametrize("head_dim", [128, 192, 256])
@pytest.mark.parametrize("pattern", ["random", "all_ties", "integer_ties"])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("execution_mode", ["eager", "cuda_graph"])
def test_decode_batch_invariance(
    head_dim, pattern, score_dtype, execution_mode, decode_topk_backend
):
    """Both execution modes preserve bytes across batches, padding and page placement."""
    data = _inputs(head_dim, pages=17, lengths=(31, 513, 1025), topk=512)
    if pattern == "all_ties":
        data["index_query"].zero_()
        data["query"].zero_()
        # The same selected tokens in a different order can change FP32 PV
        # accumulation. This reproduces the original short-context failure.
        positions = torch.arange(31, device="cuda")
        pages = data["table"][0, positions // 64].long()
        values = torch.ones(31, 2, head_dim, device="cuda", dtype=torch.bfloat16)
        values[0] = 2**25
        values[-1] = -(2**25)
        data["value"][pages, positions % 64] = values
    elif pattern == "integer_ties":
        for name in ("index_query", "index_key"):
            data[name].copy_(torch.randint(-2, 3, data[name].shape, device="cuda"))

    def run(current, *, attention_only=False):
        def compute():
            if not attention_only:
                _indexer(current)
                current["indices"].copy_(
                    _simplified_decode_topk(
                        current["scores"].to(score_dtype),
                        current["lengths"],
                        512,
                        batch_invariant_mode=True,
                    )
                )
            _attention(current)

        compute()
        if execution_mode == "cuda_graph":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute()
            # Compare replay results, not outputs left by warmup.
            current["output"].fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
        assert torch.isfinite(current["output"]).all()

    run(data)
    for rows in ([2, 0, 1], [0], [1], [2]):
        current = _subset(data, rows, remap_pages=True)
        # Isolate the attention kernel with exactly the same selected indices.
        run(current, attention_only=True)
        assert bytes_equal(current["output"], data["output"][rows])
        # Then also exercise indexer and routing with a different score width.
        run(current)
        for i, row in enumerate(rows):
            length = int(data["lengths"][row])
            assert bytes_equal(current["scores"][i, :length], data["scores"][row, :length])
        assert bytes_equal(current["indices"], data["indices"][rows])
        assert bytes_equal(current["output"], data["output"][rows])


@pytest.mark.parametrize("topk", [16, 128, 512])
def test_decode_topk_fallback_ties_have_fixed_membership_and_order(monkeypatch, topk):
    """The BI fallback orders boundary ties by logical token position."""
    monkeypatch.setattr(dsa_gqa, "deep_select", None)
    lengths = torch.tensor([0, 31, 513], device="cuda", dtype=torch.int32)
    positions = torch.arange(1025, device="cuda")
    scores = (positions % 3).float().expand(3, -1).clone()
    scores.masked_fill_(positions[None, :] >= lengths[:, None], -torch.inf)
    actual = _simplified_decode_topk(scores, lengths, topk, batch_invariant_mode=True)
    for row, length in enumerate(lengths.tolist()):
        # An independent integer ordering oracle: descending score, then position.
        chosen = sorted(range(length), key=lambda token: (-(token % 3), token))[:topk]
        chosen += [-1] * (topk - len(chosen))
        expected = torch.tensor(chosen, device="cuda", dtype=torch.int32)
        assert bytes_equal(actual[row], expected)
        solo = _simplified_decode_topk(
            scores[row : row + 1, :length], lengths[row : row + 1], topk, batch_invariant_mode=True
        )
        assert bytes_equal(solo[0], expected)


@pytest.mark.parametrize("width", [31, 1025])
def test_decode_topk_default_uses_fast_path(monkeypatch, width):
    """Neither the default nor explicit BI-off mode may invoke the full sort."""
    monkeypatch.setattr(dsa_gqa, "deep_select", None)
    scores = torch.zeros(2, width, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([31, width], device="cuda", dtype=torch.int32)
    scores[0, 31:] = -torch.inf
    topk = 512
    expected = scores.topk(min(topk, width), dim=-1).indices.to(torch.int32)
    if width < topk:
        expected = torch.nn.functional.pad(expected, (0, topk - width), value=-1)

    def reject_sort(*args, **kwargs):
        pytest.fail("Stable sort must only run when batch_invariant_mode is enabled")

    monkeypatch.setattr(torch.Tensor, "argsort", reject_sort)
    assert bytes_equal(_simplified_decode_topk(scores, lengths, topk), expected)
    assert bytes_equal(
        _simplified_decode_topk(scores, lengths, topk, batch_invariant_mode=False), expected
    )


@pytest.mark.parametrize("topk", [1, 7, 16, 128, 512, 1024, 4096])
@pytest.mark.parametrize("pattern", ["random", "all_ties", "integer_ties"])
@pytest.mark.parametrize("batch_invariant_mode", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("execution_mode", ["eager", "cuda_graph"])
def test_decode_deepselect_topk_correct_and_batch_invariant(
    monkeypatch, topk, pattern, batch_invariant_mode, dtype, execution_mode
):
    """DeepSelect is correct and BI in both modes, including BF16 dispatch boundaries."""
    if not dsa_gqa._deepselect_decode_supported(torch.device("cuda"), topk):
        pytest.skip("DeepSelect and SM100/103 required")
    seeded(53)
    # Limit large dispatch cases to BF16 ties and K tiers that switch kernels.
    check_dispatch = (
        dtype == torch.bfloat16
        and topk in (512, 1024)
        and pattern != "random"
        and batch_invariant_mode
    )
    max_length = 65537 if check_dispatch else 16385
    # Include empty requests, K boundary cases, and a context spanning TMA tiles.
    lengths = torch.tensor(
        [0, 31, topk - 1, topk, topk + 1, max_length], device="cuda", dtype=torch.int64
    )
    # Noncontiguous input with an unaligned row pitch exercises the adapter.
    scores = torch.randn(6, 2 * max_length, device="cuda", dtype=dtype)[:, ::2]
    if pattern == "all_ties":
        scores.zero_()
    elif pattern == "integer_ties":
        scores.copy_(torch.randint(-2, 3, scores.shape, device="cuda"))
    # These deliberately high padding scores must never be selected.
    scores.masked_fill_(torch.arange(max_length, device="cuda")[None, :] >= lengths[:, None], 1000)

    def reject_sort(*args, **kwargs):
        pytest.fail("DeepSelect must not invoke stable argsort, including in BI mode")

    monkeypatch.setattr(torch.Tensor, "argsort", reject_sort)
    calls = []
    original_topk = dsa_gqa.deep_select.topk

    def capture_topk(values, *args, **kwargs):
        assert values.dtype == dtype, "DeepSelect must receive scores in their original precision"
        calls.append(1)
        return original_topk(values, *args, **kwargs)

    monkeypatch.setattr(dsa_gqa.deep_select, "topk", capture_topk)

    def select(values, ends):
        def compute():
            return _simplified_decode_topk(
                values, ends, topk, batch_invariant_mode=batch_invariant_mode
            )

        result = compute()
        if execution_mode == "cuda_graph":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = compute()
            # Ensure assertions observe graph replay rather than warmup results.
            result.fill_(-99)
            graph.replay()
            torch.cuda.synchronize()
        assert result.dtype == torch.int32 and result.is_contiguous()
        assert result.shape == (values.size(0), topk)
        return result

    actual = select(scores, lengths)
    assert calls, "Supported FP32/BF16 inputs must invoke DeepSelect"
    for _ in range(3):
        assert bytes_equal(select(scores, lengths), actual)
    order = [5, 2, 0, 4, 1, 3]
    assert bytes_equal(select(scores[order], lengths[order]), actual[order])
    for row, length in enumerate(lengths.tolist()):
        k = min(length, topk)
        indices = actual[row, :k].long()
        assert bool(((indices >= 0) & (indices < length)).all())
        assert indices.unique().numel() == k
        assert bool((actual[row, k:] == -1).all())
        assert torch.equal(
            scores[row, indices].sort(descending=True).values, scores[row, :length].topk(k).values
        )
        solo = select(scores[row : row + 1, :length], lengths[row : row + 1])
        assert bytes_equal(solo[0], actual[row])

    if check_dispatch:
        sm_count = torch.cuda.get_device_properties(scores.device).multi_processor_count
        alignment = dsa_gqa.deep_select.get_stride_requirement()[0] // scores.element_size()
        # Width >= 512K and batch <= 6 select the cluster kernel. The normal
        # kernel changes occupancy configuration when batch exceeds the SM count.
        for width in (max_length, 512 * 1024):
            aligned_width = (width + alignment - 1) // alignment * alignment
            for batch in (1, 6, 7, sm_count + 1):
                batched_scores = torch.full(
                    (batch, aligned_width), 1000, device="cuda", dtype=dtype
                )
                batched_lengths = torch.full((batch,), width, device="cuda", dtype=torch.int32)
                positions = sorted({0, batch - 1})
                for position in positions:
                    batched_scores[position, :max_length].copy_(scores[-1])
                    batched_lengths[position] = max_length
                batched = select(batched_scores, batched_lengths)
                for position in positions:
                    assert bytes_equal(batched[position], actual[-1]), (width, batch, position)


@pytest.mark.parametrize("reason", ["device", "topk", "dtype"])
@pytest.mark.parametrize("batch_invariant_mode", [False, True])
def test_decode_deepselect_unsupported_falls_back(monkeypatch, reason, batch_invariant_mode):
    """Unsupported GPU architectures, K, or dtypes must use the PyTorch fallback."""
    if dsa_gqa.deep_select is None:
        pytest.skip("DeepSelect required")
    topk = 4097 if reason == "topk" else 16
    dtype = torch.float16 if reason == "dtype" else torch.float32
    scores = torch.arange(33, device="cuda", dtype=dtype)[None, :]
    lengths = torch.tensor([33], device="cuda", dtype=torch.int32)

    def reject_deepselect(*args, **kwargs):
        pytest.fail("Unsupported inputs must use the PyTorch fallback")

    monkeypatch.setattr(dsa_gqa.deep_select, "topk", reject_deepselect)
    if reason == "device":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    actual = _simplified_decode_topk(
        scores, lengths, topk, batch_invariant_mode=batch_invariant_mode
    )
    expected = torch.arange(32, max(32 - topk, -1), -1, device="cuda", dtype=torch.int32)[None, :]
    if topk > 33:
        expected = torch.nn.functional.pad(expected, (0, topk - 33), value=-1)
    assert bytes_equal(actual, expected)
