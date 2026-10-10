# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stream bookkeeping of ScheduleNode for Transformer Engine GroupedTensors, which MXFP8 token
dispatch hands to the experts (and MXFP8 combine backward hands back as the output grad)."""

from unittest import mock

import pytest
import torch

from megatron.core.pipeline_parallel.utils import ScheduleNode

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
except ImportError:
    GroupedTensor = None


def dispatched_tokens(token_counts=(128, 256, 0, 128), hidden=256):
    """Expert-major MXFP8 tokens wrapped as a per-expert GroupedTensor, as Transformer Engine's
    NCCL-EP dispatch returns them. The experts receive different token counts."""
    counts = torch.tensor(token_counts, dtype=torch.int64, device="cuda")
    rows = sum(token_counts)
    offsets = torch.zeros(len(token_counts) + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = torch.cumsum(counts * hidden, dim=0)
    return GroupedTensor(
        shape=(rows, hidden),
        dtype=torch.bfloat16,
        num_tensors=len(token_counts),
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False),
        data=torch.randint(0, 255, (rows * hidden,), dtype=torch.uint8, device="cuda"),
        scale_inv=torch.randint(100, 140, (rows * hidden // 32,), dtype=torch.uint8, device="cuda"),
        first_dims=counts,
        tensor_offsets=offsets,
    )


def buffers_of(grouped):
    return [b for b in grouped.get_data_tensors() if isinstance(b, torch.Tensor)]


def record_stream_calls():
    """A patch of torch.Tensor.record_stream that records the tensors it is called on, and the
    list it records them in."""
    tensors = []
    original = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        tensors.append(tensor)
        return original(tensor, stream)

    return mock.patch.object(torch.Tensor, "record_stream", record_stream), tensors


def includes(tensors, tensor):
    return any(t is tensor for t in tensors)


@pytest.mark.skipif(GroupedTensor is None, reason="Requires Transformer Engine's GroupedTensor")
class TestScheduleNodeGroupedTensor:

    def test_free_input_releases_grouped_tensor_input(self):
        tokens = dispatched_tokens()
        buffers = buffers_of(tokens)
        saved = []

        def forward_func(x):
            # Stand-in for the expert op, which keeps the quantized buffers for backward.
            saved.append(x.rowwise_data)
            return x.rowwise_data.float().sum()

        node = ScheduleNode(
            forward_func, torch.cuda.Stream(), torch.cuda.Event(), free_input=True, name="mlp"
        )
        patch, recorded = record_stream_calls()
        with patch:
            node.forward((tokens,))
        torch.cuda.synchronize()

        # Streams are recorded on the buffers, not on the grouped tensor.
        assert all(includes(recorded, b) for b in buffers)
        assert not includes(recorded, tokens)
        # The grouped tensors drop their buffers...
        for grouped in (tokens, node.inputs[0]):
            assert grouped.rowwise_data is None and grouped.scale_inv is None
        # ...without freeing the one the op kept.
        assert saved[0].untyped_storage().nbytes() == saved[0].numel()

    def test_backward_records_grouped_tensor_grad(self):
        node = ScheduleNode(
            lambda x: x * 2,
            torch.cuda.Stream(),
            torch.cuda.Event(),
            backward_func=lambda outputs, grads: grads,
            name="mlp",
        )
        node.forward((torch.ones(4, device="cuda", requires_grad=True),))
        grad = dispatched_tokens()
        patch, recorded = record_stream_calls()
        with patch:
            node.backward((grad,))
        torch.cuda.synchronize()

        assert all(includes(recorded, b) for b in buffers_of(grad))
        assert not includes(recorded, grad)
