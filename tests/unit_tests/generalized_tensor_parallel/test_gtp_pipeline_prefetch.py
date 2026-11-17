# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independently scheduled pipeline chunks need separate GTP prefetch domains."""

import pytest
import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _make_gtp_linear,
    _make_gtp_remat_grouped_linear,
    _torchrun_dist_init,
    reset_gtp_globals,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grouped", [False, True])
def test_backward_gather_preserves_other_chunk_forward(dtype, grouped):
    """A backward in another virtual chunk must not overwrite a pending forward."""
    group = torch.distributed.group.WORLD
    if group.size() < 2:
        pytest.skip("requires at least two GPUs")

    def make_layer():
        if grouped:
            return _make_gtp_remat_grouped_linear(2, 64, 64 * group.size(), group, dtype)
        return _make_gtp_linear(64, 64 * group.size(), group, dtype)

    layers = [make_layer() for _ in range(3)]
    gtp.classify_gtp_remat_chains(layers)
    anchors = [layer.weight0 if grouped else layer.weight for layer in layers]
    for index, anchor in enumerate(anchors):
        for weight in anchor._weights:
            weight.data.fill_(3 + 3 * index)

    forward, _ = anchors[0]._all_gather_weight(async_op=False, fwd=True)
    if grouped:
        # Nonadjacent grouped blocks reuse the same double-buffer parity.
        for weight in anchors[1]._weights:
            weight._double_buffer_parity()
    anchors[2]._all_gather_weight(async_op=False, fwd=False)

    tensors = forward if grouped else [forward]
    for tensor in tensors:
        torch.testing.assert_close(tensor, torch.full_like(tensor, 3), atol=0, rtol=0)


def test_interleaved_chunks_preserve_forward_values(monkeypatch):
    """A chunk's backward may run while another chunk has a pending forward prefetch."""
    group = torch.distributed.group.WORLD
    if group.size() < 2:
        pytest.skip("requires at least two GPUs")
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", False)
    monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", False)
    chunks = [
        torch.nn.Sequential(
            *[_make_gtp_linear(64, 64, group, fuse_wgrad_accumulation=True) for _ in range(2)]
        )
        for _ in range(2)
    ]
    for chunk in chunks:
        for weight in chunk.parameters():
            weight.main_grad = torch.zeros_like(weight)
    gtp.classify_gtp_remat_chains(chunks)
    x = torch.randn(16, 1, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = [chunk(x).clone() for chunk in chunks]
    gtp.GTP_CONFIG.weight_prefetch = True
    try:
        for _ in range(3):
            for chunk, reference in zip(chunks, expected):
                for weight in chunk.parameters():
                    weight.main_grad.zero_()
                output = chunk(x.detach().clone().requires_grad_())
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
                output.sum().backward()
    finally:
        gtp.wait_async_comms()
