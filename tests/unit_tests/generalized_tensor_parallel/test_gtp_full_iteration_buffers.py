# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from tests.unit_tests.test_utilities import Utils


def test_capture_does_not_borrow_eager_wgrad_storage(monkeypatch):
    """A graph must not keep a scratch address that eager work can later overwrite."""
    Utils.initialize_distributed()
    monkeypatch.setattr(gtp, '_wgrad_buf_pool', {})
    eager = gtp._wgrad_pool_get((1024,), torch.float32, 'cuda')
    gtp._wgrad_pool_put(eager)
    result = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch = gtp._wgrad_pool_get(
            eager.shape, eager.dtype, eager.device, graph_capture_safe=True
        )
        scratch.fill_(3)
        result.copy_(scratch)
    assert scratch.data_ptr() != eager.data_ptr()
    assert gtp._wgrad_pool_get(eager.shape, eager.dtype, eager.device) is eager
    for _ in range(3):
        eager.fill_(9)
        graph.replay()
        torch.testing.assert_close(result, torch.full_like(result, 3))


def test_plain_wgrad_reuse_waits_for_async_reader(monkeypatch):
    """Returning a buffer to the manual pool must not race its side-stream reader."""
    Utils.initialize_distributed()
    monkeypatch.setattr(gtp, '_wgrad_buf_pool', {})
    reader = torch.cuda.Stream()
    output = torch.empty(1024, device='cuda')
    for value in range(3):
        buffer = gtp._wgrad_pool_get(output.shape, output.dtype, output.device)
        buffer.fill_(value)
        reader.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(reader):
            torch.cuda._sleep(1000000)
            output.copy_(buffer)
            ready = torch.cuda.Event()
            ready.record()
        gtp._wgrad_pool_put(buffer, ready_event=ready)
        reused = gtp._wgrad_pool_get(output.shape, output.dtype, output.device)
        assert reused.data_ptr() == buffer.data_ptr()
        reused.fill_(-1)
        torch.testing.assert_close(output, torch.full_like(output, value))
        gtp._wgrad_pool_put(reused)
