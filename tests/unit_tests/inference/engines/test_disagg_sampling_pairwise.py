# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Handoff sampling compares fixed histories, not unrelated engine RNG streams."""

from contextlib import contextmanager
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from megatron.core.inference.sampling.torch_sampling import TorchSampling
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    admit_import,
    assert_import_equal,
    assert_released,
    canonical_weights,
    complete_transfer,
    disagg_config,
    exchange,
    prompt,
    real_engine,
    run_to_completion,
    snapshot_source,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401
from tests.unit_tests.test_utilities import Utils


@contextmanager
def sampled_distributions(engine, records, params):
    assert not engine.controller._enable_cuda_graph
    sampler = engine.controller._sampling
    original = sampler.sample_kernel

    def observe(logits, n, context, **kwargs):
        assert n == 1, "This fixed-history oracle deliberately owns one sampled request"
        indices = kwargs.get("gather_indices")
        assert (indices is None) == context.config.materialize_only_last_token_logits
        assert logits.shape[0] == (n if indices is None else context.padded_active_token_count)
        selected = logits[:n] if indices is None else logits[indices[:n]]
        raw = selected.detach().float().cpu().clone()
        output = original(logits, n, context, **kwargs)
        selected_tokens = output.detach().cpu().clone()
        filtered = TorchSampling.filter_logits(raw, params.temperature, params.top_k, params.top_p)
        probabilities = filtered.softmax(dim=-1)
        assert torch.isfinite(probabilities).all()
        assert (probabilities[torch.arange(n), selected_tokens] > 0).all()
        records.append((raw, probabilities, selected_tokens))
        return output

    with mock.patch.object(sampler, "sample_kernel", side_effect=observe):
        yield


@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
@pytest.mark.parametrize(
    "filters",
    [{}, {"top_k": 1}, {"top_k": 4}, {"top_p": 0.8}],
    ids=["temperature", "greedy", "top-k", "top-p"],
)
@torch.inference_mode()
def test_handoff_stochastic_fixed_history(transport_world, backend, filters):
    config = disagg_config(
        sampling_backend=backend,
        offset_sampling_seed_by_dp_rank=False,
        materialize_only_last_token_logits=filters.get("top_k") != 4,
    )
    sampler_type = TorchSampling if backend == "torch" else FlashInferSampling
    tokens = prompt(33)

    def params(count, **changes):
        return SamplingParams(
            num_tokens_to_generate=count, termination_id=-1, temperature=0.7, **filters, **changes
        )

    reference_samples = []
    with real_engine(config) as reference:
        assert type(reference.controller._sampling) is sampler_type
        weights = canonical_weights(reference)
        with sampled_distributions(reference, reference_samples, params(2)):
            expected = run_to_completion(reference, reference.add_request(101, tokens, params(2)))
    assert len(reference_samples) == 2
    source = dist.get_rank() % 2 == 0
    with real_engine(config, role="prefill" if source else "decode", weights=weights) as engine:
        assert type(engine.controller._sampling) is sampler_type
        records = []
        with sampled_distributions(engine, records, params(2)), ForwardWitness(engine) as witness:
            metadata = state = None
            if source:
                request = run_to_completion(
                    engine, engine.add_request(101, tokens, params(1, do_kv_handoff=True))
                )
                assert request.generated_tokens == expected.generated_tokens[:1]
                metadata, state = snapshot_source(engine, request)
            peer = exchange((metadata, state) if source else None, transport_world)
            if not source:
                metadata, state = peer
                assert metadata["kv_meta"]["resume_tokens"] == expected.generated_tokens[:1]
            pending, future = complete_transfer(
                engine, metadata, tokens, params(2), transport_world
            )
            if not source:
                assert_import_equal(engine, pending, state, len(tokens))
                admit_import(engine)
                actual = run_to_completion(engine, future)
                assert len(records) == 1
                # Both logits describe prompt + the SAME realized first token.
                # The next sampled token need not match: the decode engine has
                # not consumed the prefill engine's first random draw.
                assert torch.equal(records[0][0], reference_samples[1][0])
                assert torch.equal(records[0][1], reference_samples[1][1])
                assert actual.generated_tokens[0] == expected.generated_tokens[0]
                assert actual.generated_tokens[1] == records[0][2].item()
                assert len(actual.generated_tokens) == 2
                if filters.get("top_k") == 1:
                    assert actual.generated_tokens == expected.generated_tokens
                assert witness.steps and all(not step[2] for step in witness.steps)
                assert_released(engine)
            dist.barrier(group=transport_world)
            if source:
                assert len(records) == 1
                assert torch.equal(records[0][0], reference_samples[0][0])
                assert torch.equal(records[0][1], reference_samples[0][1])
                engine._poll_pending_kv_pushes()
                engine.release_handoff_blocks(101)
                assert_released(engine)


@pytest.mark.parametrize("skip_prompt", [False, True])
@pytest.mark.parametrize("score_fields", [{"return_log_probs": True}, {"top_n_logprobs": 2}])
@torch.inference_mode()
def test_handoff_rejects_unsupported_or_missing_scores_before_transfer(skip_prompt, score_fields):
    """Reject unsupported scores or a missing carried first-token score."""
    Utils.initialize_model_parallel()
    try:
        with real_engine(disagg_config(), role="decode") as engine:
            params = SamplingParams(
                num_tokens_to_generate=2, skip_prompt_log_probs=skip_prompt, **score_fields
            )
            allocator = engine.context.kv_block_allocator
            before = allocator.pool_avail
            if score_fields.get("top_n_logprobs"):
                error, message = NotImplementedError, "top-token log probabilities"
            elif skip_prompt:
                error, message = ValueError, "first-token log probability count"
            else:
                error, message = NotImplementedError, "prompt log probabilities"
            with pytest.raises(error, match=message):
                engine.add_request_with_kv_handoff(
                    101, prompt(33), params, {"resume_tokens": [9]}, [1, 2, 3]
                )
            assert allocator.pool_avail == before
            assert_released(engine)
    finally:
        Utils.destroy_model_parallel()
