# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer request-local RNG integration and batch-independent replay."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

pytest.importorskip("flashinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def context(seeds, positions, top_k=0, top_p=0.0):
    """Build aligned CPU request metadata and GPU sampling parameters."""
    n = len(seeds)
    metadata = {
        "seed": torch.tensor(seeds, dtype=torch.int64),
        "temperature": torch.ones(n),
        "top_k": torch.full((n,), top_k, dtype=torch.int32),
        "top_p": torch.full((n,), top_p),
    }
    return SimpleNamespace(
        active_request_metadata=metadata,
        gpu_view=SimpleNamespace(**{key: value.cuda() for key, value in metadata.items()}),
        config=SimpleNamespace(num_speculative_tokens=0),
        get_active_sequence_lengths=lambda: torch.tensor(positions, dtype=torch.int64),
    )


def draw(sampler, logits, ctx, **kwargs):
    """Sample the active rows using the context's batch-level filter flags."""
    md = ctx.active_request_metadata
    return sampler.sample_kernel(
        logits,
        len(md["seed"]),
        ctx,
        no_top_k=bool((md["top_k"] == 0).all()),
        no_top_p=bool((md["top_p"] == 0).all()),
        **kwargs,
    )


@pytest.mark.parametrize("top_k,top_p", [(0, 0.0), (0, 0.8), (64, 0.0), (64, 0.8)])
def test_request_seed_offset_replay(top_k, top_p):
    """Every dispatch honors each row's seed and position without consuming shared RNG."""
    seeds = [0, *range(2**32, 2**32 + 30), 2**63 - 1]
    positions = list(range(17, 49))
    ctx = context(seeds, positions, top_k, top_p)
    logits = torch.linspace(-1, 1, 257, device="cuda").expand(32, -1).contiguous()
    sampler = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    before = sampler._rng.get_state().clone()
    reference, _ = assert_replays_bit_exact(
        lambda: draw(sampler, logits, ctx),
        (),
        backward=False,
        contention=True,
        what="flashinfer_request_seed_offset",
    )
    expected = reference["out"]
    assert torch.equal(before, sampler._rng.get_state())

    # The first row is unchanged: this catches FlashInfer versions that broadcast
    # only seed[0] / offset[0] and silently ignore the remaining requests.
    for changed in (
        context([seeds[0]] + [seed ^ 1234567 for seed in seeds[1:]], positions, top_k, top_p),
        context(seeds, [positions[0]] + [p + 1 for p in positions[1:]], top_k, top_p),
    ):
        actual = draw(sampler, logits, changed)
        assert actual[0] == expected[0]
        assert (actual[1:] != expected[1:]).any()

    # Use the saved positions of pending logits, not the advanced async context.
    advanced = context(seeds, [p + 10 for p in positions], top_k, top_p)
    output = torch.empty_like(expected)
    gathered = draw(
        sampler,
        torch.cat((torch.zeros_like(logits), logits)),
        advanced,
        gather_indices=torch.arange(32, 64, device="cuda"),
        sequence_lengths=torch.tensor(positions),
        output=output,
    )
    assert gathered is output
    assert torch.equal(output, expected)


@pytest.mark.parametrize("top_k,top_p", [(0, 0.0), (0, 0.8), (64, 0.0), (64, 0.8)])
def test_seeded_rows_do_not_consume_shared_rng(top_k, top_p):
    """Adding seeded requests leaves unseeded draws and generator advancement unchanged."""
    mixed = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    unseeded = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    logits = torch.linspace(-1, 1, 257, device="cuda").expand(3, -1).contiguous()
    for position in range(10, 14):
        actual = draw(mixed, logits, context([11, -1, 12], [position] * 3, top_k, top_p))
        expected = draw(unseeded, logits[1:2], context([-1], [position], top_k, top_p))
        assert actual[1] == expected[0]
        assert torch.equal(mixed._rng.get_state(), unseeded._rng.get_state())


def test_seeded_dispatch_is_independent_of_other_filter_groups():
    """Other requests' filters must not change the seeded row's sampling algorithm."""
    logits = torch.linspace(-1, 1, 257, device="cuda").expand(4, -1).contiguous()
    ctx = context([11, 12, 13, 14], [17] * 4)
    ctx.active_request_metadata["top_k"][:] = torch.tensor([0, 0, 64, 64])
    ctx.active_request_metadata["top_p"][:] = torch.tensor([0.0, 0.8, 0.0, 0.8])
    ctx.gpu_view.top_k.copy_(ctx.active_request_metadata["top_k"])
    ctx.gpu_view.top_p.copy_(ctx.active_request_metadata["top_p"])
    sampler = FlashInferSampling(257, torch.Generator(device="cuda").manual_seed(42))
    actual = draw(sampler, logits, ctx)
    for row, (top_k, top_p) in enumerate([(0, 0.0), (0, 0.8), (64, 0.0), (64, 0.8)]):
        expected = draw(sampler, logits[row : row + 1], context([11 + row], [17], top_k, top_p))
        assert actual[row] == expected[0]


@pytest.mark.parametrize("token_mapping", [None, torch.tensor([0])])
def test_speculative_seed_rejected(token_mapping):
    """Reject both speculative configuration and per-token request mappings."""
    ctx = context([42], [17])
    ctx.config.num_speculative_tokens = 1 if token_mapping is None else 0
    sampler = FlashInferSampling(257, torch.Generator(device="cuda"))
    with pytest.raises(ValueError, match="speculative"):
        draw(sampler, torch.zeros(1, 257, device="cuda"), ctx, token_to_request_index=token_mapping)


@pytest.mark.parametrize("top_k,top_p", [(0, 0.0), (0, 0.8), (64, 0.0), (64, 0.8)])
def test_request_seed_survives_batch_reordering(top_k, top_p):
    """Logical draws survive reordering, compaction, and singleton batches."""
    seeds = list(range(32))
    ctx = context(seeds, [17] * 32, top_k, top_p)
    reordered = context(seeds[::-1], [17] * 32, top_k, top_p)
    sampler = FlashInferSampling(257, torch.Generator(device="cuda"))
    logits = torch.linspace(-1, 1, 257, device="cuda").expand(32, -1).contiguous()
    expected = draw(sampler, logits, ctx)
    actual = draw(sampler, logits.flip(0), reordered).flip(0)
    assert torch.equal(actual, expected)
    for rows in ([23, 2, 11], [23]):
        compacted = context([seeds[row] for row in rows], [17] * len(rows), top_k, top_p)
        assert torch.equal(draw(sampler, logits[rows], compacted), expected[rows])
