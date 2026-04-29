# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.gpu_input_state import GpuSampledTokenState


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU sampled-token state tests")


def test_gpu_sampled_token_state_records_base_tokens_without_cpu_sync():
    _require_cuda()
    handoff_stream = torch.cuda.Stream()
    source_stream = torch.cuda.Stream()
    state = GpuSampledTokenState(
        max_requests=4, num_speculative_tokens=0, device="cuda", stream=handoff_stream
    )
    sampled_tokens = torch.empty(4, dtype=torch.int64, device="cuda")

    with torch.cuda.stream(source_stream):
        sampled_tokens.copy_(torch.tensor([10, 11, 12, 13], dtype=torch.int64, device="cuda"))

    ready_event = state.record(
        sampled_tokens=sampled_tokens,
        active_request_count=4,
        source_stream=source_stream,
    )

    torch.cuda.current_stream().wait_event(ready_event)
    assert state.sampled_tokens.tolist() == [10, 11, 12, 13]
    assert state.accepted_token_counts.tolist() == [1, 1, 1, 1]
    assert state.last_active_request_count == 4


def test_gpu_sampled_token_state_records_speculative_payloads_and_debug_compares():
    _require_cuda()
    handoff_stream = torch.cuda.Stream()
    state = GpuSampledTokenState(
        max_requests=4, num_speculative_tokens=2, device="cuda", stream=handoff_stream
    )
    sampled_tokens = torch.tensor([20, 21, 22, 23], dtype=torch.int64, device="cuda")
    sampled_mtp_tokens = torch.tensor(
        [[30, 31, 32, 33], [40, 41, 42, 43]], dtype=torch.int64, device="cuda"
    )
    accepted_counts = torch.tensor([0, 1, 2, 1], dtype=torch.int64, device="cuda")

    state.record(
        sampled_tokens=sampled_tokens,
        active_request_count=3,
        sampled_mtp_tokens=sampled_mtp_tokens,
        accepted_token_counts=accepted_counts,
    )

    state.debug_compare_cpu(
        sampled_tokens_cpu=torch.tensor([20, 21, 22], dtype=torch.int64),
        active_request_count=3,
        sampled_mtp_tokens_cpu=torch.tensor(
            [[30, 31, 32], [40, 41, 42]], dtype=torch.int64
        ),
        accepted_token_counts_cpu=torch.tensor([0, 1, 2], dtype=torch.int64),
    )


def test_gpu_sampled_token_state_rejects_oversized_active_count():
    _require_cuda()
    state = GpuSampledTokenState(
        max_requests=2,
        num_speculative_tokens=0,
        device="cuda",
        stream=torch.cuda.Stream(),
    )

    with pytest.raises(ValueError, match="exceeds max_requests"):
        state.record(
            sampled_tokens=torch.zeros(3, dtype=torch.int64, device="cuda"),
            active_request_count=3,
        )
