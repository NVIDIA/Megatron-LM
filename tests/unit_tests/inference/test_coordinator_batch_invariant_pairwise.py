# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference import coordinator_pairwise_utils as pairwise
from tests.unit_tests.inference import test_coordinator_features_pairwise as features


@pytest.mark.internal
@pytest.mark.asyncio
async def test_routed_batch_invariance_at_padding_boundary(monkeypatch, request):
    monkeypatch.setattr(DynamicInferenceContext, "TOKEN_ROUNDER", 64)
    monkeypatch.setattr(DynamicInferenceContext, "REQUEST_ROUNDER", 4)
    bik.enable_batch_invariant_mode("te_native", collective="ordered")
    request.addfinalizer(bik.disable_batch_invariant_mode)
    from transformer_engine.pytorch import cpp_extensions
    from transformer_engine.pytorch.module import layernorm_linear, linear

    modules = {"linear": linear, "layernorm_linear": layernorm_linear}
    assert all(module.general_gemm is cpp_extensions.general_gemm for module in modules.values())
    run_routed = features._exercise_all_owners
    records, phase, active = {"direct": [], "routed": []}, "direct", []

    for name, module in modules.items():

        def traced(*args, _name=name, **kwargs):
            if active:
                active[0].append((_name, tuple(args[0].shape), tuple(args[1].shape)))
            return cpp_extensions.general_gemm(*args, **kwargs)

        monkeypatch.setattr(module, "general_gemm", traced)

    params = pairwise.greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    async with pairwise.routed_model(
        monkeypatch,
        engine_factory=pairwise.batch_invariant_engine,
        context_max_requests=128,
        context_block_size_tokens=256,
        context_buffer_size_gb=0.125,
        materialize_only_last_token_logits=False,
        async_sched_mode=AsyncScheduleMode.LEGACY,
    ) as harness:
        controller = harness.engine.controller
        forward = controller._dynamic_step_forward_logits

        def observed(tokens, positions):
            context = harness.engine.context
            ids = features._active_target_ids(harness)
            if not ids:
                return forward(tokens, positions)
            request_index = context.request_ids.tolist().index(min(ids))
            mapping = context.token_to_request_idx[: context.active_token_count]
            rows = (mapping == request_index).nonzero().flatten()
            active.append([])
            result = forward(tokens, positions)
            calls = active.pop()
            assert {call[2][0] for call in calls} == {64 if phase == "direct" else 128}
            assert {call[0] for call in calls} == set(modules)
            if phase == "routed" and context.num_decode_requests:
                assert context.num_decode_requests == len(ids) == 65
            input_tokens = tokens[0, rows].clone()
            input_positions = positions[0, rows].clone()
            logits = controller._all_logits_cuda[0, rows, : harness.config.vocab_size].clone()
            records[phase].append((input_tokens, input_positions, logits))
            return result

        controller._dynamic_step_forward_logits = observed
        direct, phase = await harness.direct([4], params), "routed"

        def clear_routed(_):
            records["routed"].clear()

        await run_routed(harness, [4], params, direct, install=clear_routed, requests_per_client=65)
        positions = [[row[1].item() for row in rows] for rows in records.values()]
        assert positions == [[0, 1, 2, 3]] * 2
        for expected, actual in zip(*records.values()):
            assert all(torch.equal(left, right) for left, right in zip(expected, actual))
            assert torch.isfinite(expected[2]).all()
