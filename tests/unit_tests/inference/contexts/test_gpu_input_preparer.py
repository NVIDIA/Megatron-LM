# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``GpuInputPreparer`` (v3 plan commit 17).

Plan validation: under ``async_overlap_debug_checks=True`` (queue depth 1,
GPU scatter taking the place of the CPU write), token output matches the
serial path bit-for-bit for ≥4096 tokens at deterministic seed.

The full bit-for-bit decode parity is exercised by the engine's
end-to-end tests; here we verify the lower-level scatter contract on a
synthesized snapshot view.
"""

import pytest
import torch

from megatron.core.inference.contexts.gpu_input_preparer import GpuInputPreparer, PrevSampleState
from megatron.core.inference.engines.async_pipeline_types import StepInputPlan


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


class _FakeSnapshot:
    """Stand-in for ContextGPUView with just the fields the preparer needs."""

    def __init__(self, max_tokens: int, device):
        self.token_to_input_ids = torch.full(
            (max_tokens,), -1, dtype=torch.int64, device=device
        )
        self.metadata_ready_event = None


class TestGpuInputPreparer:
    def test_first_step_skips_scatter(self, device):
        snapshot = _FakeSnapshot(max_tokens=8, device=device)
        prev = PrevSampleState(
            prev_sampled_token_ids=torch.empty(8, dtype=torch.int64, device=device),
            sample_done_event=None,
            prev_sample_ready_event=None,
        )
        plan = StepInputPlan(
            decode_request_slots=(0, 1, 2),
            decode_token_destination_indices=(0, 1, 2),
            prefill_cpu_token_ranges=(),
            speculative_width=0,
            previous_sample_source_step=None,
        )
        prep = GpuInputPreparer(torch.cuda.Stream(device=device))
        ev = prep.prepare(snapshot, plan, prev, is_first_step=True)
        assert ev is not None
        torch.cuda.synchronize(device)
        # Snapshot remains -1 because the scatter is bypassed.
        assert torch.all(snapshot.token_to_input_ids == -1)

    def test_steady_state_scatter_uses_decode_destination_indices(self, device):
        snapshot = _FakeSnapshot(max_tokens=8, device=device)
        prev_sampled = torch.tensor([10, 20, 30], dtype=torch.int64, device=device)
        full_buf = torch.zeros(8, dtype=torch.int64, device=device)
        full_buf[:3] = prev_sampled
        # Construct prev_sample_ready_event by recording on the default
        # stream; the preparer waits on it before scattering.
        ev = torch.cuda.Event()
        ev.record()
        prev = PrevSampleState(
            prev_sampled_token_ids=full_buf,
            sample_done_event=None,
            prev_sample_ready_event=ev,
        )
        plan = StepInputPlan(
            decode_request_slots=(0, 1, 2),
            decode_token_destination_indices=(2, 4, 6),
            prefill_cpu_token_ranges=(),
            speculative_width=0,
            previous_sample_source_step=None,
        )
        prep = GpuInputPreparer(torch.cuda.Stream(device=device))
        input_ready = prep.prepare(snapshot, plan, prev, is_first_step=False)
        input_ready.synchronize()
        out = snapshot.token_to_input_ids.cpu().tolist()
        # Slots 2, 4, 6 carry the scattered values; 0/1/3/5/7 stay at -1.
        assert out[2] == 10
        assert out[4] == 20
        assert out[6] == 30
        assert out[0] == -1
        assert out[1] == -1
        assert out[3] == -1

    def test_no_decode_indices_records_event_only(self, device):
        """When the input plan has no decode indices the preparer records
        an event but performs no scatter."""
        snapshot = _FakeSnapshot(max_tokens=8, device=device)
        prev = PrevSampleState(
            prev_sampled_token_ids=torch.empty(8, dtype=torch.int64, device=device),
            sample_done_event=None,
            prev_sample_ready_event=None,
        )
        plan = StepInputPlan(
            decode_request_slots=(),
            decode_token_destination_indices=(),
            prefill_cpu_token_ranges=(),
            speculative_width=0,
            previous_sample_source_step=None,
        )
        prep = GpuInputPreparer(torch.cuda.Stream(device=device))
        ev = prep.prepare(snapshot, plan, prev, is_first_step=False)
        torch.cuda.synchronize(device)
        assert ev.query() is True
        assert torch.all(snapshot.token_to_input_ids == -1)
