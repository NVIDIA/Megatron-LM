# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request-local FlashInfer draws replay independently of scheduling and filters."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

pytest.importorskip("flashinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_context(seeds, positions, filters):
    md = {
        "seed": torch.tensor(seeds, dtype=torch.int64),
        "temperature": torch.tensor([0.7] * len(seeds)),
        "top_k": torch.tensor([k for k, _ in filters], dtype=torch.int32),
        "top_p": torch.tensor([p for _, p in filters]),
    }
    return SimpleNamespace(
        total_request_count=len(seeds),
        paused_request_count=0,
        active_request_metadata=md,
        gpu_view=SimpleNamespace(**{key: value.cuda() for key, value in md.items()}),
        config=SimpleNamespace(num_speculative_tokens=0),
        get_active_sequence_lengths=lambda: torch.tensor(positions),
    )


def draw(sampler, logits, ctx, **kwargs):
    md = ctx.active_request_metadata
    return sampler.sample_kernel(
        logits,
        len(md["seed"]),
        ctx,
        no_top_k=bool((md["top_k"] <= 0).all()),
        no_top_p=bool(((md["top_p"] <= 0) | (md["top_p"] >= 1)).all()),
        **kwargs,
    )


@pytest.mark.parametrize("filters", [(0, 0.0), (0, 0.8), (16, 0.0), (16, 0.8)])
def test_request_seed_replays_across_batch_layouts(filters):
    sampler = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(99))
    logits = torch.randn(3, 257, device="cuda")
    seeds = [0, 1 << 32, (1 << 63) - 1]
    positions = [19, 1 << 32, 20]
    ctx = make_context(seeds, positions, [filters] * 3)
    state = sampler._rng.get_state().clone()
    ref, _ = assert_replays_bit_exact(
        lambda: draw(sampler, logits, ctx), (), backward=False, contention=True
    )
    expected = ref["out"]
    reordered = make_context(seeds[::-1], positions[::-1], [filters] * 3)
    assert torch.equal(expected, draw(sampler, logits.flip(0), reordered).flip(0))
    for row in range(3):
        single = make_context([seeds[row]], [positions[row]], [filters])
        assert expected[row] == draw(sampler, logits[row : row + 1], single)[0]
        # Neighbors changing the batch dispatch must not select a different RNG stream.
        mixed = make_context([seeds[row], -1], [positions[row], 7], [filters, (32, 0.6)])
        assert expected[row] == draw(sampler, logits[[row, (row + 1) % 3]], mixed)[0]
    sampler._rng.set_state(state)
    draw(sampler, logits, ctx)
    assert torch.equal(state, sampler._rng.get_state())


@pytest.mark.parametrize("filters", [(0, 0.0), (0, 0.8), (16, 0.0), (16, 0.8)])
def test_seeded_request_does_not_advance_shared_generator(filters):
    mixed = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    single = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    logits = torch.randn(2, 257, device="cuda")
    for pos in range(8):
        ctx = make_context([11, -1], [pos, pos], [(32, 0.6), filters])
        alone = make_context([-1], [pos], [filters])
        assert draw(mixed, logits, ctx)[1] == draw(single, logits[1:], alone)[0]
    assert torch.equal(mixed._rng.get_state(), single._rng.get_state())


def test_saved_positions_gather_output_and_changing_streams():
    sampler = FlashInferSampling(257, torch.Generator(device="cuda"))
    logits = torch.zeros(5, 257, device="cuda")
    ctx = make_context([42, 43], [99, 99], [(0, 0.0)] * 2)
    output = torch.empty(2, dtype=torch.long, device="cuda")
    sequence = []
    for pos in range(16):
        result = draw(
            sampler,
            logits,
            ctx,
            gather_indices=torch.tensor([1, 4], device="cuda"),
            sequence_lengths=torch.tensor([pos, pos]),
            output=output,
        )
        assert result is output
        expected = draw(sampler, logits[:2], make_context([42, 43], [pos] * 2, [(0, 0.0)] * 2))
        assert torch.equal(result, expected)
        sequence.append(result.tolist())
    assert len({a for a, _ in sequence}) > 1
    assert any(a != b for a, b in sequence)


@pytest.mark.parametrize("token_mapping", [False, True])
def test_seeded_speculative_sampling_rejected(token_mapping):
    sampler = FlashInferSampling(257, torch.Generator(device="cuda"))
    ctx = make_context([42], [1], [(0, 0.0)])
    ctx.config.num_speculative_tokens = 0 if token_mapping else 2
    with pytest.raises(ValueError, match="speculative"):
        draw(
            sampler,
            torch.zeros(1, 257, device="cuda"),
            ctx,
            token_to_request_index=torch.tensor([0], device="cuda") if token_mapping else None,
        )


def test_unseeded_speculative_rows_ignore_inactive_seed_metadata():
    import flashinfer

    sampler = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    reference_rng = torch.Generator(device="cuda").manual_seed(42)
    ctx = make_context([-1], [7], [(0, 0.0)])
    ctx.total_request_count = 4
    ctx.paused_request_count = 3
    ctx.config.num_speculative_tokens = 3
    # The pinned metadata buffer has spare capacity, unlike its active prefix.
    ctx.active_request_metadata["seed"] = torch.tensor([-1, 0, 1, 2])
    logits = torch.randn(4, 257, device="cuda")
    result = sampler.sample_kernel(
        logits,
        4,
        ctx,
        no_top_k=True,
        no_top_p=True,
        token_to_request_index=torch.zeros(4, dtype=torch.long, device="cuda"),
    )
    expected = flashinfer.sampling.sampling_from_logits(
        logits / ctx.gpu_view.temperature[0], deterministic=True, generator=reference_rng
    ).long()
    assert torch.equal(result, expected)
    assert torch.equal(sampler._rng.get_state(), reference_rng.get_state())
