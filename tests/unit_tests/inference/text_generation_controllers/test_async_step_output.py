# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``AsyncStepOutputPool`` and ``AsyncStepOutput.cpu_view``.

Covers the v3 plan commit 4 validation: D2H runs on the dedicated
``d2h_output`` stream into pinned destinations; ``cpu_view`` synchronizes
``d2h_done_event`` before returning views; the event is load-bearing —
calling ``cpu_view`` after the source GPU tensor is overwritten still yields
the original values.
"""

import pytest
import torch

from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput
from megatron.core.inference.text_generation_controllers.async_step_output import (
    AsyncStepOutputPool,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for AsyncStepOutputPool")
    return torch.device("cuda:0")


class TestAsyncStepOutputPool:
    def test_dedicated_stream_distinct_from_default(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        default_stream = torch.cuda.current_stream(device=device)
        assert pool.d2h_stream != default_stream
        assert pool.d2h_stream.device == default_stream.device

    def test_pinned_destinations_are_pinned_int64(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=2, device=device)
        assert pool._pinned_sampled_tokens.is_pinned()
        assert pool._pinned_sampled_tokens.dtype == torch.int64
        assert pool._pinned_sampled_mtp_tokens is not None
        assert pool._pinned_sampled_mtp_tokens.is_pinned()
        assert pool._pinned_sampled_mtp_tokens.shape == (2, 4)

    def test_begin_copy_payload_metadata(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = torch.arange(3, dtype=torch.int64, device=device)
        out = pool.begin_copy(
            step_id=7, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        assert isinstance(out, AsyncStepOutput)
        assert out.step_id == 7
        assert out.has_payload(AsyncStepOutputPool.SAMPLED_TOKENS_KEY) is True
        assert out.has_payload(AsyncStepOutputPool.SAMPLED_MTP_TOKENS_KEY) is False
        assert out.d2h_done_event is not None

    def test_cpu_view_returns_correct_values(self, device):
        pool = AsyncStepOutputPool(max_requests=8, num_speculative_tokens=0, device=device)
        sampled = (torch.arange(5, dtype=torch.int64, device=device) + 100)
        out = pool.begin_copy(
            step_id=0, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        view = out.cpu_view
        assert view[AsyncStepOutputPool.SAMPLED_TOKENS_KEY].tolist() == [100, 101, 102, 103, 104]

    def test_event_is_load_bearing(self, device):
        """Overwriting the source GPU tensor after cpu_view returns must not
        change the resolved CPU values — the event guarantees the d2h copy
        has finished by the time cpu_view returns."""
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = (torch.arange(3, dtype=torch.int64, device=device) + 10)
        out = pool.begin_copy(
            step_id=1, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        view = out.cpu_view
        sampled.fill_(-1)
        torch.cuda.synchronize(device)
        assert view[AsyncStepOutputPool.SAMPLED_TOKENS_KEY].tolist() == [10, 11, 12]

    def test_mtp_payload_round_trip(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=2, device=device)
        sampled = torch.arange(3, dtype=torch.int64, device=device)
        mtp = torch.arange(6, dtype=torch.int64, device=device).reshape(2, 3) + 100
        out = pool.begin_copy(step_id=2, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=mtp)
        view = out.cpu_view
        assert view[AsyncStepOutputPool.SAMPLED_MTP_TOKENS_KEY].tolist() == [
            [100, 101, 102],
            [103, 104, 105],
        ]

    def test_ensure_mtp_buffer_late_init(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        assert pool._pinned_sampled_mtp_tokens is None
        pool.ensure_mtp_buffer(num_speculative_tokens=3, max_requests=4)
        assert pool._pinned_sampled_mtp_tokens is not None
        assert pool._pinned_sampled_mtp_tokens.shape == (3, 4)
        assert pool._pinned_sampled_mtp_tokens.is_pinned()

    def test_ensure_mtp_buffer_idempotent(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=2, device=device)
        original = pool._pinned_sampled_mtp_tokens
        pool.ensure_mtp_buffer(num_speculative_tokens=2, max_requests=4)
        assert pool._pinned_sampled_mtp_tokens is original

    def test_result_alias_matches_cpu_view(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = torch.tensor([7, 8, 9], dtype=torch.int64, device=device)
        out = pool.begin_copy(
            step_id=3, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        view = out.result()
        assert view[AsyncStepOutputPool.SAMPLED_TOKENS_KEY].tolist() == [7, 8, 9]
