# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A drained prefill batch must not release handoff-owned recurrent storage."""

from unittest import mock

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    assert_released,
    disagg_config,
    prompt,
    real_engine,
    run_to_completion,
    sampling,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401


@pytest.mark.parametrize("mode", [AsyncScheduleMode.LEGACY, AsyncScheduleMode.ASYNC])
@torch.inference_mode()
def test_source_ssm_slot_survives_empty_batches(transport_world, mode):
    config = disagg_config(model_provider="hybrid", async_sched_mode=mode)
    with real_engine(config, role="prefill") as engine:
        context = engine.context
        metadata = context.mamba_metadata
        boundary = "update_requests" if mode == AsyncScheduleMode.LEGACY else "resolve_requests"
        handoff_tokens = 1 if mode == AsyncScheduleMode.LEGACY else 2
        with mock.patch.object(context, boundary, wraps=getattr(context, boundary)) as resolution:
            run_to_completion(
                engine,
                engine.add_request(101, prompt(33), sampling(handoff_tokens, do_kv_handoff=True)),
            )
            assert resolution.call_count > 0
            slot = engine._pinned_handoff_ssm_slots[101]
            assert context.total_request_count == 0
            assert metadata.mamba_state_free_slot_count == metadata.max_requests - 1
            assert (
                slot not in metadata.mamba_state_free_slots[: metadata.mamba_state_free_slot_count]
            )
            conv = context.mamba_conv_states[:, slot].clone()
            recurrent = context.mamba_ssm_states[:, slot].clone()
            calls = resolution.call_count
            # A different real request drains the batch again before RELEASE_KV.
            with ForwardWitness(engine, request_id=102) as witness:
                result = run_to_completion(
                    engine,
                    engine.add_request(102, [token + 1 for token in prompt(35)], sampling(2)),
                )
            assert witness.steps and len(result.generated_tokens) == 2
            assert resolution.call_count > calls
            assert context.total_request_count == 0
            assert metadata.mamba_state_free_slot_count == metadata.max_requests - 1
            assert (
                slot not in metadata.mamba_state_free_slots[: metadata.mamba_state_free_slot_count]
            )
            assert torch.equal(context.mamba_conv_states[:, slot], conv)
            assert torch.equal(context.mamba_ssm_states[:, slot], recurrent)
            assert not engine._pending_kv_pushes  # No outstanding transfer permits release here.
            engine.release_handoff_blocks(101)
            assert metadata.mamba_state_free_slot_count == metadata.max_requests
            assert sorted(metadata.mamba_state_free_slots.tolist()) == list(
                range(metadata.max_requests)
            )
            engine.release_handoff_blocks(101)
            assert metadata.mamba_state_free_slot_count == metadata.max_requests
            assert_released(engine)
