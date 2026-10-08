# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request-local draws must survive reordering, pauses, and batch-size changes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    Status,
)
from megatron.core.inference.sampling.torch_sampling import TorchSampling
from megatron.core.inference.sampling_params import SamplingParams

_DEVICES = [
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    )
]


def context(seeds, positions, temperatures=None):
    n = len(seeds)
    return SimpleNamespace(
        total_request_count=n,
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=0),
        active_request_metadata={
            "seed": torch.tensor(seeds, dtype=torch.int64),
            "temperature": torch.tensor(temperatures or [1.0] * n),
            "top_k": torch.zeros(n, dtype=torch.int32),
            "top_p": torch.zeros(n),
        },
        get_active_sequence_lengths=lambda: torch.tensor(positions),
    )


def draw(sampler, logits, ctx, **kwargs):
    return sampler.sample_kernel(
        logits,
        ctx.total_request_count - ctx.paused_request_count,
        ctx,
        no_top_k=True,
        no_top_p=True,
        **kwargs,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_seeded_sequence_survives_batch_changes(device):
    logits = torch.randn(
        3, 127, device=device, generator=torch.Generator(device=device).manual_seed(1)
    )
    sampler = TorchSampling(torch.Generator(device=device).manual_seed(22), 127)
    before = sampler._rng.get_state().clone()
    for position in range(8, 24):
        full = draw(sampler, logits, context([91, 92, 93], [position] * 3, [0.7, 1.0, 0.7]))
        shuffled = draw(
            sampler, logits[[2, 0, 1]], context([93, 91, 92], [position] * 3, [0.7, 0.7, 1.0])
        )
        alone = draw(sampler, logits[1:2], context([92], [position]))
        assert torch.equal(full, shuffled[[1, 2, 0]])
        assert full[1] == alone[0]
    assert torch.equal(before, sampler._rng.get_state())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("separate_buckets", [False, True])
def test_seeded_request_does_not_consume_unseeded_rng(device, separate_buckets):
    logits = torch.arange(127, dtype=torch.float32, device=device).sin().expand(2, -1)
    mixed = TorchSampling(torch.Generator(device=device).manual_seed(2), 127)
    alone = TorchSampling(torch.Generator(device=device).manual_seed(2), 127)
    for position in range(32):
        temperatures = [0.7 if separate_buckets else 1.0, 1.0]
        result = draw(mixed, logits, context([14, -1], [position, position], temperatures))
        expected = draw(alone, logits[1:], context([-1], [position]))
        assert result[1] == expected[0]
    assert torch.equal(mixed._rng.get_state(), alone._rng.get_state())


def test_gather_output_and_paused_requests():
    sampler = TorchSampling(torch.Generator(device="cuda").manual_seed(42), 127)
    logits = torch.randn(5, 127, device="cuda")
    ctx = context([3, 4], [10, 20])
    expected = draw(sampler, logits[[1, 4]], ctx)
    ctx.paused_request_count = 2
    ctx.total_request_count = 4
    output = torch.empty(2, dtype=torch.long, device="cuda")
    result = draw(sampler, logits, ctx, gather_indices=torch.tensor([1, 4]), output=output)
    assert result is output
    assert torch.equal(result, expected)


def test_different_seeds_and_positions_change_streams():
    sampler = TorchSampling(torch.Generator(device="cuda").manual_seed(1), 127)
    logits = torch.zeros(2, 127, device="cuda")
    sequence = [draw(sampler, logits, context([11, 12], [i, i])).tolist() for i in range(20)]
    assert any(a != b for a, b in sequence)
    assert len({a for a, _ in sequence}) > 1


def test_speculative_seeds_rejected():
    sampler = TorchSampling(torch.Generator(device="cuda"), 127)
    ctx = context([1], [10])
    ctx.config.num_speculative_tokens = 1
    with pytest.raises(ValueError, match="speculative"):
        draw(sampler, torch.zeros(1, 127, device="cuda"), ctx)


@pytest.mark.parametrize("seed", [None, 0, 42, 2**63 - 1])
def test_metadata_and_serialization_preserve_seed(seed):
    params = SamplingParams(seed=seed, termination_id=0)
    restored = SamplingParams.deserialize(params.serialize())
    req = DynamicInferenceRequest(
        request_id=1, prompt="", prompt_tokens=torch.tensor([1]), sampling_params=restored
    )
    metadata = dict(zip([name for name, _ in req.get_metadata_types()], req.tracked_metadata))
    assert metadata["seed"] == (-1 if seed is None else seed)


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "42"])
def test_invalid_seed_validation(seed):
    with pytest.raises(ValueError, match="seed"):
        SamplingParams(seed=seed)
    with pytest.raises(ValueError, match="seed"):
        SamplingParams.deserialize({"seed": seed})


@pytest.mark.parametrize("top_k,top_p", [(1, 0.0), (8, 0.0), (0, 0.7)])
@pytest.mark.parametrize("device", _DEVICES)
def test_filter_buckets_preserve_seeded_draws(top_k, top_p, device):
    sampler = TorchSampling(torch.Generator(device=device).manual_seed(2), 127)
    logits = torch.randn(2, 127, device=device)
    ctx = context([11, 12], [9, 9])
    ctx.active_request_metadata["top_k"].fill_(top_k)
    ctx.active_request_metadata["top_p"].fill_(top_p)
    full = draw(sampler, logits, ctx)
    for i in range(2):
        single = context([11 + i], [9])
        single.active_request_metadata["top_k"].fill_(top_k)
        single.active_request_metadata["top_p"].fill_(top_p)
        assert full[i] == draw(sampler, logits[i : i + 1], single)[0]


@pytest.mark.parametrize("device", _DEVICES)
def test_unseeded_sampling_keeps_original_shared_generator_path(device):
    logits = torch.randn(3, 127, device=device)
    sampler = TorchSampling(torch.Generator(device=device).manual_seed(9), 127)
    generator = torch.Generator(device=device).manual_seed(9)
    ctx = context([-1, -1, -1], [10, 10, 10])
    result = draw(sampler, logits, ctx)
    expected = TorchSampling.sample_from_logits(
        logits, 1.0, 0, 0.0, generator=generator, vocab_size=127
    )
    assert torch.equal(result, expected)
    assert torch.equal(generator.get_state(), sampler._rng.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_seeded_sampling_distribution():
    # Distribution smoke test independent of replay: two categories with p=.2/.8,
    # plus an impossible token which must never win the exponential race.
    batch = 16384
    logits = torch.tensor([0.2, 0.8, 0.0], device="cuda").log().expand(batch, -1)
    sampled = TorchSampling.sample_from_logits(
        logits,
        1.0,
        0,
        0.0,
        generator=torch.Generator(device="cuda"),
        row_seeds=list(range(batch)),
        row_positions=[19] * batch,
    )
    assert not (sampled == 2).any()
    assert abs((sampled == 1).float().mean().item() - 0.8) < 0.02


@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
@pytest.mark.parametrize("speculative,has_seed_metadata", [(0, False), (2, True), (0, True)])
@pytest.mark.asyncio
async def test_seed_request_admission(backend, speculative, has_seed_metadata):
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    engine = object.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(
        config=SimpleNamespace(sampling_backend=backend, num_speculative_tokens=speculative),
        request_metadata={"seed": torch.empty(0)} if has_seed_metadata else {},
        max_sequence_length=16,
        max_tokens=16,
        num_speculative_tokens=speculative,
        block_size_tokens=8,
        kv_block_allocator=SimpleNamespace(pool_size=32),
        remove_vlm_request_data=Mock(),
    )
    engine.requests = {}
    engine._loop = asyncio.get_running_loop()
    engine._generation_epoch = None
    engine.rank = 1
    engine.enable_chunked_prefill = False
    engine.waiting_request_ids = []
    engine.failed_request_ids = []
    engine.use_coordinator = True
    engine.is_mp_coordinator = True
    engine._send_requests_to_coordinator = Mock()
    req = DynamicInferenceRequest(
        request_id=1,
        prompt="",
        prompt_tokens=torch.tensor([1]),
        sampling_params=SamplingParams(seed=42, termination_id=0, num_tokens_to_generate=1),
    )
    future = engine._add_request(req)
    if has_seed_metadata and not speculative:
        assert not future.done()
        assert req.status == Status.ACTIVE_AND_GENERATING_TOKENS
        assert engine.waiting_request_ids == [1]
        assert engine.failed_request_ids == []
        engine._send_requests_to_coordinator.assert_not_called()
        return
    assert future.done()
    result = await future
    assert result.status == Status.FAILED
    errors = [
        e.payload for e in result.events if e.type == DynamicInferenceEventType.ERROR_NONTRANSIENT
    ]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "Request-local seeds" in str(errors[0])
    assert engine.waiting_request_ids == []
    assert engine.failed_request_ids == [1]
    engine._send_requests_to_coordinator.assert_called_once_with([result])


@pytest.mark.parametrize("top_k", [0, 1])
def test_seeded_cpu_sampling_rejected(top_k):
    with pytest.raises(AssertionError, match="requires CUDA"):
        TorchSampling.sample_from_logits(
            torch.zeros(1, 127),
            1.0,
            top_k,
            0.0,
            generator=torch.Generator(),
            row_seeds=[42],
            row_positions=[7],
        )
