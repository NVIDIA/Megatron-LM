# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Byte-exact packing of NCCL MoE dispatch inputs; distributed replay lives in inference tests."""

import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher


@pytest.mark.parametrize("hidden,topk", [(128, 6), (2688, 6), (1024, 22), (2048, 22)])
@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("strided", [False, True])
def test_nccl_dispatch_pack_replay(hidden, topk, rows, strided):
    """Packing preserves every bit, including signed zero, infinities and NaN payloads."""
    h = torch.arange(rows * hidden, device="cuda", dtype=torch.int32).to(torch.int16)
    p = torch.arange(rows * topk, device="cuda", dtype=torch.int32)
    if h.numel() >= 4:
        h[:4] = torch.tensor([-32768, 0x7F80, 0x7FC1, 1], device="cuda", dtype=torch.int16)
        p[:4] = torch.tensor(
            [-2147483648, 0x7F800000, 0x7FC00001, 1], device="cuda", dtype=torch.int32
        )
    tensors = (
        h.view(torch.bfloat16).reshape(rows, hidden),
        p.view(torch.float32).reshape(rows, topk),
        torch.full((rows, topk), -1, device="cuda", dtype=torch.int64),
    )
    if strided:
        views = []
        for tensor in tensors:
            storage = tensor.new_empty(rows, tensor.shape[1] * 2)
            view = storage[:, ::2]
            view.copy_(tensor)
            views.append(view)
        tensors = tuple(views)
    layout = NCCLAllGatherDispatcher._packed_row_layout(tensors)
    assert layout is not None
    order, offsets, row_bytes = layout
    assert order == [2, 1, 0]
    assert row_bytes == hidden * 2 + topk * 12
    assert offsets == [topk * 12, topk * 8, 0]
    previous = None
    for _ in range(4):
        packed = NCCLAllGatherDispatcher._pack_rows(tensors, layout)
        outputs = NCCLAllGatherDispatcher._unpack_rows(packed, tensors, layout)
        if previous is not None:
            assert torch.equal(packed, previous)
        previous = packed.clone()
        for actual, expected in zip(outputs, tensors):
            assert actual.dtype == expected.dtype and actual.is_contiguous()
            # clone normalizes even empty strides before reinterpreting the dtype.
            expected = expected.clone(memory_format=torch.contiguous_format)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_nccl_dispatch_rejects_unaligned_layout():
    tensors = (
        torch.empty(2, 5, dtype=torch.bfloat16),
        torch.empty(2, 3),
        torch.empty(2, 3, dtype=torch.int64),
    )
    assert NCCLAllGatherDispatcher._packed_row_layout(tensors) is None
    assert NCCLAllGatherDispatcher._packed_row_layout((tensors[0].flatten(), *tensors[1:])) is None
