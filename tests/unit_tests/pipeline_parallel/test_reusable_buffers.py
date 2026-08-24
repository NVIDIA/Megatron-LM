# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.pipeline_parallel.reusable_buffers import (
    ReusableOutputBufferPool,
    release_reusable_output_buffer,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reusable_output_buffer_pool_rotates_and_reuses_storage():
    pool = ReusableOutputBufferPool("test")
    pool.configure(2)
    stream = torch.cuda.current_stream()

    first = pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
    second = pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
    assert first.data_ptr() != second.data_ptr()
    assert release_reusable_output_buffer(first, stream)
    assert release_reusable_output_buffer(second, stream)

    reused = pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
    assert reused.data_ptr() == first.data_ptr()
    assert release_reusable_output_buffer(reused, stream)
    pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reusable_output_buffer_pool_rejects_early_reuse():
    pool = ReusableOutputBufferPool("test")
    pool.configure(1)
    pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))

    with pytest.raises(RuntimeError, match="before its consumer released it"):
        pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
    pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reusable_output_buffer_pool_requires_static_signature():
    pool = ReusableOutputBufferPool("test")
    pool.configure(1)
    tensor = pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
    assert release_reusable_output_buffer(tensor, torch.cuda.current_stream())

    with pytest.raises(RuntimeError, match="requires a static output signature"):
        pool.acquire((8, 8), torch.bfloat16, torch.device("cuda"))
    pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reusable_output_buffer_pool_crosses_eager_warmup_into_cuda_graph():
    pool = ReusableOutputBufferPool("test")
    pool.configure(1)
    stream = torch.cuda.current_stream()

    warmup = pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
    warmup.fill_(1)
    assert release_reusable_output_buffer(warmup, stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pool.acquire((4, 8), torch.bfloat16, torch.device("cuda"))
        captured.fill_(2)
        assert release_reusable_output_buffer(captured, torch.cuda.current_stream())

    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(captured == 2)
    pool.reset()
