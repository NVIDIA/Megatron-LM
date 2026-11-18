# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compact MXFP8 quantization, packed CP scoring and graph lifetime coverage."""

import pytest
import torch

from megatron.core.ops.attention.csa.backends import select_compact_indexer
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture
def compact_backend():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("compact MXFP8 indexer requires SM100+")
    # Missing dependencies on the selected architecture are a failure, not a skip.
    return select_compact_indexer("mxfp8", workspaces={}, cuda_graphs=True, deterministic=True)


def test_indexer_mxfp8_quantization_replays(compact_backend):
    from megatron.core.ops.attention.csa.indexer_quantization import (
        create_indexer_mxfp8_quantization_buffers,
        make_indexer_mxfp8_scale_cu_seqlens,
        quantize_indexer_mxfp8,
    )

    seeded()
    x = torch.randn(261, 64, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 129, 129, 261], device="cuda", dtype=torch.int32)
    prefixes = make_indexer_mxfp8_scale_cu_seqlens(cu, 64)
    buffers = create_indexer_mxfp8_quantization_buffers(x)

    def run(value):
        data, scales = quantize_indexer_mxfp8(
            value, cu_seqlens=cu, cu_seqlens_scale_padded=prefixes, buffers=buffers
        )
        return data.view(torch.uint8).clone(), scales.view(torch.uint8).clone()

    assert_replays_bit_exact(
        run, (x,), backward=False, contention=True, what="CSA MXFP8 data and E8M0 scales"
    )


def _case():
    q = torch.randn(256, 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(256, 64, device="cuda", dtype=torch.bfloat16) / (64**0.5 * 128**0.5)
    cu = torch.tensor([0, 320, 512], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 80, 128], device="cuda", dtype=torch.int32)
    return q, k, w, cu, cu_k


def _run(backend, q, k, w, cu, cu_k):
    layout = cp_utils.build_cp_indexer_layout(cu, cu_k, 256, 256, k.shape[0])
    return backend.packed(
        q,
        k,
        w,
        256,
        4,
        cu_seqlens_compressed=cu_k,
        indexer_layout=layout,
        max_seqlen_q=320,
        return_softmax=True,
    )


def _assert_reference(q, k, w, cu, cu_k, actual):
    from megatron.core.ops.attention.csa.indexer_quantization import (
        create_indexer_mxfp8_quantization_buffers,
    )

    # TE dequantization is an independent oracle for the physical E8M0 repacking.
    def dequantize(value):
        buffers = create_indexer_mxfp8_quantization_buffers(value)
        source = value
        if buffers.padded_input is not None:
            buffers.padded_input[: buffers.num_rows].copy_(value.reshape(buffers.num_rows, 128))
            source = buffers.padded_input
        buffers.quantizer.update_quantized(source, buffers.quantized)
        return (
            buffers.quantized.dequantize()
            .reshape(-1, 128)[: buffers.num_rows]
            .reshape(value.shape)
            .float()
        )

    q_ref, k_ref = dequantize(q), dequantize(k)
    ids, lengths, predict = actual
    assert torch.isfinite(predict).all()
    docs, compressed = cu.tolist(), cu_k.tolist()
    for segment, (begin, end) in enumerate(zip(docs, docs[1:])):
        first, last = max(begin, 256), min(end, 512)
        if first >= last:
            continue
        rows = slice(first - 256, last - 256)
        keys = k_ref[compressed[segment] : compressed[segment + 1]]
        scores = torch.einsum("qhd,kd->qhk", q_ref[rows], keys).relu()
        scores = (scores * w[rows].float().unsqueeze(-1)).sum(1)
        visible = torch.arange(first - begin + 1, last - begin + 1, device=q.device) // 4
        key_ids = torch.arange(keys.shape[0], device=q.device)
        scores.masked_fill_(key_ids.unsqueeze(0) >= visible.unsqueeze(1), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        expected_ids = torch.arange(256, device=q.device).expand(last - first, -1)
        expected_ids = expected_ids.masked_fill(expected_ids >= visible.unsqueeze(1), -1).int()
        assert torch.equal(ids[rows], expected_ids)
        assert torch.equal(lengths[rows], visible.int())
        reference = torch.zeros_like(predict[rows])
        reference[:, : keys.shape[0]] = probabilities
        torch.testing.assert_close(predict[rows], reference, rtol=0.025, atol=0.002)


def test_packed_compact_indexer_reference_and_replay(compact_backend):
    seeded()
    args = _case()
    actual = _run(compact_backend, *args)
    _assert_reference(*args, actual)
    assert_replays_bit_exact(
        lambda q, k, w: _run(compact_backend, q, k, w, *args[3:]),
        args[:3],
        backward=False,
        contention=True,
        what="CP compact indexer ids and softmax",
    )


def test_packed_compact_graph_refresh_and_shared_workspace(compact_backend):
    seeded()
    args = _case()
    q, k, w, cu, cu_k = args
    for value in (q, k, w):
        value.requires_grad_()
    inputs_before = tuple(value.detach().clone() for value in (q, k, w))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(compact_backend, *args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run(compact_backend, *args)
    graph.replay()
    _assert_reference(*args, actual)
    saved = tuple(value.clone() for value in actual)
    sibling = select_compact_indexer(
        "mxfp8", workspaces=compact_backend.workspaces, cuda_graphs=True, deterministic=True
    )
    _run(sibling, q * 0.5, k, w, cu, cu_k)
    for value, previous in zip(actual, saved):
        assert torch.equal(value, previous), "later layer overwrote graph outputs"
    assert len(compact_backend.workspaces) == 1
    for value, previous in zip((q, k, w), inputs_before):
        assert torch.equal(value, previous)
        assert value.grad is None
    assert actual[2].grad_fn is None and not actual[2].requires_grad
    for workspace in compact_backend.workspaces.values():
        for buffers in (workspace.mxfp8.q_buffers, workspace.mxfp8.k_buffers):
            if buffers.padded_input is not None:
                assert buffers.padded_input.grad_fn is None
                assert not buffers.padded_input.requires_grad
    # Same static geometry, different document boundary and CP causal offsets.
    cu.copy_(cu.new_tensor([0, 304, 512]))
    cu_k.copy_(cu_k.new_tensor([0, 76, 128]))
    graph.replay()
    _assert_reference(*args, actual)


@pytest.mark.parametrize("real_tokens", [65533, 4095])
def test_compact_layout_cp16_static_capacity(compact_backend, real_tokens):
    """Eight metadata slots and an underfilled 65k bucket retain legal compact bounds."""
    from megatron.core.ops.attention.csa.compact_indexer import build_thd_compact_k_layout

    cu = torch.tensor([0] + [real_tokens] * 8, device="cuda", dtype=torch.int32)
    cu_k = cu // 4
    layout = cp_utils.build_cp_indexer_layout(cu, cu_k, 8 * 4096, 4096, 65536 // 4)
    compact_cu, source_rows = build_thd_compact_k_layout(layout[0], cu_k, 65536 // 4, 4)
    assert compact_cu[-1].item() == 65536 // 4 + 8
    assert source_rows.numel() == 65536 // 4 + 8
    assert (source_rows >= 0).all() and (source_rows <= 65536 // 4).all()
    lengths = compact_cu[1:] - compact_cu[:-1]
    assert (lengths <= source_rows.numel()).all()
    if real_tokens == 4095:
        assert lengths.max().item() > real_tokens // 4 + 2
