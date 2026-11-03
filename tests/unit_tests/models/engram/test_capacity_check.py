# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A fixed-capacity Engram bucket overflow fails eagerly and inside a captured CUDA graph."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import _check_bucket_capacity

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    # Under torchrun every rank would run the subprocess on the same device.
    pytest.mark.skipif(int(os.environ.get("RANK", "0")) != 0, reason="runs on rank 0 only"),
]


def test_eager_overflow_raises_with_the_factor_that_would_fit():
    peak = torch.tensor(48, device="cuda")
    _check_bucket_capacity(torch.tensor(32, device="cuda"), 32, num_tokens=64, ep_size=4)
    with pytest.raises(ValueError, match=r"48 requests for 32 slots.*at least 3\.00"):
        _check_bucket_capacity(peak, 32, num_tokens=64, ep_size=4)


_CAPTURED = textwrap.dedent(
    """
    import torch
    from megatron.core.models.engram.distributed_embedding import _check_bucket_capacity

    peak = torch.tensor(8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _check_bucket_capacity(peak, 16, num_tokens=64, ep_size=4)
    graph.replay()
    torch.cuda.synchronize()
    print("replay within capacity ok", flush=True)
    peak.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    print("overflowing replay did not fail", flush=True)
    """
)


def test_captured_overflow_fails_the_replay():
    result = subprocess.run(
        [sys.executable, "-c", _CAPTURED],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "0"},
    )
    output = result.stdout + result.stderr
    print(output)  # shows how the failure reads in a job log
    assert "replay within capacity ok" in result.stdout
    assert "overflowing replay did not fail" not in result.stdout
    assert result.returncode != 0
    assert "device-side assert" in output
