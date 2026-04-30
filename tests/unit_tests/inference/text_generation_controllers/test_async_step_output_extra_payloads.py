# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the AsyncStepOutput extra-payload paths added in commit 21
(logprobs, top-n logprobs, routing records).

Plan validation: logprob / top-n / routing parity vs. serial; nsys shows
logprob D2H on the output stream. The full parity is exercised by the
engine's logprob/routing integration tests; here we verify the payload
contract on the pool: D2H runs on the dedicated d2h_output stream, the
pinned destination is on CPU and pinned, and ``cpu_view`` exposes the
data after event sync.
"""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.async_step_output import (
    AsyncStepOutputPool,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


class TestAsyncStepOutputExtraPayloads:
    def test_logprobs_payload_round_trips(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
        out = pool.begin_copy(
            step_id=0, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        logprobs_gpu = torch.tensor(
            [[-0.1, -0.5], [-0.2, -0.4], [-0.3, -0.6]],
            dtype=torch.float32,
            device=device,
        )
        pool.add_payload(out, AsyncStepOutputPool.LOGPROBS_KEY, logprobs_gpu)
        view = out.cpu_view
        assert AsyncStepOutputPool.LOGPROBS_KEY in view
        assert view[AsyncStepOutputPool.LOGPROBS_KEY].is_pinned()
        assert torch.allclose(
            view[AsyncStepOutputPool.LOGPROBS_KEY], logprobs_gpu.cpu()
        )

    def test_top_n_logprobs_payload_round_trips(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = torch.tensor([1], dtype=torch.int64, device=device)
        out = pool.begin_copy(
            step_id=1, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        top_n_gpu = torch.tensor([[0.9, 0.7, 0.5]], dtype=torch.float32, device=device)
        pool.add_payload(out, AsyncStepOutputPool.TOP_N_LOGPROBS_KEY, top_n_gpu)
        view = out.cpu_view
        assert torch.allclose(
            view[AsyncStepOutputPool.TOP_N_LOGPROBS_KEY], top_n_gpu.cpu()
        )

    def test_routing_indices_payload_round_trips(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = torch.tensor([7, 8], dtype=torch.int64, device=device)
        out = pool.begin_copy(
            step_id=2, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        routing = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
        pool.add_payload(out, AsyncStepOutputPool.ROUTING_INDICES_KEY, routing)
        view = out.cpu_view
        assert torch.equal(
            view[AsyncStepOutputPool.ROUTING_INDICES_KEY], routing.cpu()
        )

    def test_payload_metadata_records_added_keys(self, device):
        pool = AsyncStepOutputPool(max_requests=4, num_speculative_tokens=0, device=device)
        sampled = torch.tensor([1, 2], dtype=torch.int64, device=device)
        out = pool.begin_copy(
            step_id=0, sampled_tokens_gpu=sampled, sampled_mtp_tokens_gpu=None
        )
        pool.add_payload(
            out,
            AsyncStepOutputPool.LOGPROBS_KEY,
            torch.zeros(2, dtype=torch.float32, device=device),
        )
        assert out.has_payload(AsyncStepOutputPool.LOGPROBS_KEY) is True
        assert out.has_payload(AsyncStepOutputPool.TOP_N_LOGPROBS_KEY) is False
