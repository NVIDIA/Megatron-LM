# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.text_generation_controllers.async_output import (
    AsyncStepOutput,
    AsyncStepOutputPool,
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for async output copy tests")


def test_async_step_output_requires_result_before_reading_cpu_views():
    _require_cuda()
    pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=2, depth=1)
    sampled_tokens_cuda = torch.arange(4, dtype=torch.int64, device="cuda")
    sampled_mtp_tokens_cuda = torch.arange(8, dtype=torch.int64, device="cuda").reshape(2, 4)
    accepted_tokens_cuda = torch.arange(8, dtype=torch.int64, device="cuda").reshape(4, 2)
    accepted_token_counts_cuda = torch.arange(4, dtype=torch.int64, device="cuda")

    output = AsyncStepOutput.begin_copy(
        pool=pool,
        active_request_count=4,
        sampled_tokens_cuda=sampled_tokens_cuda,
        sampled_mtp_tokens_cuda=sampled_mtp_tokens_cuda,
        accepted_tokens_cuda=accepted_tokens_cuda,
        accepted_token_counts_cuda=accepted_token_counts_cuda,
    )

    with pytest.raises(RuntimeError, match="result"):
        _ = output.sampled_tokens_cpu

    result = output.result()

    assert result.sampled_tokens_cpu.tolist() == [0, 1, 2, 3]
    assert result.sampled_mtp_tokens_cpu.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert result.accepted_tokens_cpu.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert result.accepted_token_counts_cpu.tolist() == [0, 1, 2, 3]
    assert result.sampled_tokens_cpu.is_pinned()
    assert result.output_ready_event.query()


def test_async_step_output_waits_for_compute_stream_before_copying():
    _require_cuda()
    pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, depth=1)
    sampled_tokens_cuda = torch.empty(4, dtype=torch.int64, device="cuda")
    expected = torch.tensor([11, 12, 13, 14], dtype=torch.int64, device="cuda")

    compute_stream = torch.cuda.Stream()
    with torch.cuda.stream(compute_stream):
        sampled_tokens_cuda.copy_(expected)

    output = AsyncStepOutput.begin_copy(
        pool=pool,
        active_request_count=4,
        sampled_tokens_cuda=sampled_tokens_cuda,
        compute_stream=compute_stream,
    )
    result = output.result()

    assert result.sampled_tokens_cpu.tolist() == [11, 12, 13, 14]

    # The single-slot pool should be reusable after result() retires the output.
    second_output = AsyncStepOutput.begin_copy(
        pool=pool,
        active_request_count=4,
        sampled_tokens_cuda=sampled_tokens_cuda,
    )
    assert second_output.result().sampled_tokens_cpu.tolist() == [11, 12, 13, 14]
