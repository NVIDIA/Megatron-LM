# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph allocation helpers shared by asynchronous Megatron components."""

from contextlib import contextmanager

import torch


@contextmanager
def default_stream_allocation():
    """Run allocations in the current device's default-stream allocator pool.

    If the caller uses another stream, order that stream after the allocation. Callers must also
    use ``Tensor.record_stream`` when a tensor can be released before its consumer finishes.
    """
    caller_stream = torch.cuda.current_stream()

    # Captured allocations must remain on a stream in the active capture session. Their lifetime
    # is owned by the graph pool rather than the ordinary caching allocator.
    if torch.cuda.is_current_stream_capturing():
        yield
        return

    allocation_stream = torch.cuda.default_stream()

    if caller_stream == allocation_stream:
        yield
        return

    with torch.cuda.stream(allocation_stream):
        yield
        allocation_ready_event = torch.cuda.Event()
        allocation_ready_event.record(allocation_stream)
    caller_stream.wait_event(allocation_ready_event)
