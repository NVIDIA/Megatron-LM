# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CuTe SDSA decode must replay on the caller's CUDA stream with current inputs."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsa_gqa_decode_kernel as kernels,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import _simplified_decode_topk
from tests.unit_tests.determinism.kernels.harness import bytes_equal, seeded

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAVE_CUTEDSL), reason="CUDA and CuTeDSL required"
)


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


@pytest.mark.parametrize("head_dim", [128, 192, 256])
@pytest.mark.parametrize("batch_invariant_mode", [False, True])
def test_decode_graph_preserves_topk_dependencies(head_dim, batch_invariant_mode):
    """Capture the CuTe -> PyTorch TopK -> CuTe dependency chain on one stream."""
    data = _inputs(head_dim)

    def run():
        _indexer(data)
        data["indices"].copy_(
            _simplified_decode_topk(
                data["scores"],
                data["lengths"],
                data["indices"].size(1),
                batch_invariant_mode=batch_invariant_mode,
            )
        )
        _attention(data)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    for factor in (-1.0, 0.5, -1.0):
        data["index_query"].mul_(factor)
        data["value"].mul_(factor)
        run()
        expected = {name: data[name].clone() for name in ("scores", "indices", "output")}
        torch.cuda.synchronize()
        for _ in range(3):
            data["scores"].fill_(torch.nan)
            data["indices"].fill_(-1)
            data["output"].fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            for name, reference in expected.items():
                assert bytes_equal(data[name], reference), f"{name} differs after graph replay"


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
def test_decode_batch_invariance(head_dim, pattern):
    """Changing co-batches, row order, padding and page placement preserves bytes."""
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

    def run(current):
        _indexer(current)
        current["indices"] = _simplified_decode_topk(
            current["scores"], current["lengths"], 512, batch_invariant_mode=True
        )
        _attention(current)

    run(data)
    for rows in ([2, 0, 1], [0], [1], [2]):
        current = _subset(data, rows, remap_pages=True)
        # Isolate the attention kernel with exactly the same selected indices.
        _attention(current)
        assert bytes_equal(current["output"], data["output"][rows])
        # Then also exercise indexer and routing with a different score width.
        run(current)
        for i, row in enumerate(rows):
            length = int(data["lengths"][row])
            assert bytes_equal(current["scores"][i, :length], data["scores"][row, :length])
        assert bytes_equal(current["indices"], data["indices"][rows])
        assert bytes_equal(current["output"], data["output"][rows])


@pytest.mark.parametrize("topk", [16, 128, 512])
def test_decode_topk_ties_have_fixed_membership_and_order(topk):
    """Tie breaking must remain fixed even when a tie straddles the K boundary."""
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
