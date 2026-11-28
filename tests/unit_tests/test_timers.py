# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.timers import Timer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_timer_outside_capture_measures_time():
    timer = Timer("t")
    timer.start()
    torch.cuda._sleep(1_000_000)
    timer.stop()
    assert timer.elapsed(reset=False) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_timer_inside_cuda_graph_capture_is_skipped():
    # barrier=True would need torch.distributed and is illegal while capturing; neither may be reached.
    timer = Timer("t")
    x = torch.zeros(16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        timer.start(barrier=True)
        x.add_(1)
        timer.stop(barrier=True)
    torch.cuda.current_stream().wait_stream(stream)
    assert timer.elapsed(reset=False) == 0.0
    graph.replay()
    torch.cuda.synchronize()
    assert x[0].item() == 1.0

    # The timer works normally again after the capture.
    timer.start()
    torch.cuda._sleep(1_000_000)
    timer.stop()
    assert timer.elapsed() > 0.0
