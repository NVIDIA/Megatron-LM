# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay seeded CUDA draws under changing batch layouts."""

import pytest
import torch

from megatron.core.inference.sampling.request_seed_noise import (
    _request_seed_noise_kernel,
    fill_request_seed_noise,
)
from megatron.core.inference.sampling.torch_sampling import TorchSampling
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_request_noise_replays_and_preserves_logical_coordinates():
    seeds = [0, 1 << 32, (1 << 63) - 1, None, 123]
    positions = [(1 << 32) + 19, 19, 20, 0, 19]

    def noise():
        result = torch.full((5, 4099), -1.0, device="cuda")
        fill_request_seed_noise(result, seeds, positions)
        return result

    reference, _ = assert_replays_bit_exact(
        noise, (), backward=False, contention=True, what="request_seed_noise"
    )
    full = reference["out"]
    assert torch.equal(full[3], torch.full_like(full[3], -1.0))
    metadata = torch.tensor(
        [[-1 if seed is None else seed for seed in seeds], positions],
        dtype=torch.int64,
        device="cuda",
    )
    # Changing launch tiling must not reassign logical RNG coordinates.
    for block in (256, 2048):
        retiled = torch.full_like(full, -1.0)
        _request_seed_noise_kernel[(5, (4099 + block - 1) // block)](
            retiled, metadata[0], metadata[1], 4099, False, block
        )
        assert torch.equal(full, retiled)
    for row in (0, 1, 2, 4):
        # Exercise the scalar-seed specialization and the masked last tile.
        alone = torch.empty((1, 4099), device="cuda")
        fill_request_seed_noise(alone, [seeds[row]], [positions[row]])
        assert torch.equal(full[row], alone[0])
    assert torch.isfinite(full).all()
    assert (full[[0, 1, 2, 4]] > 0).all()
    # High seed/position bits must not silently alias low-bit coordinates.
    changed = torch.empty((5, 4099), device="cuda")
    fill_request_seed_noise(changed, [0, 0, 0, None, 123], [19] * 5)
    for row in (0, 1, 2):
        assert not torch.equal(full[row], changed[row])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_seeded_top_p_replays_across_batch_layouts_at_exact_cutoff():
    logits = torch.zeros(16, 16, device="cuda")
    seeds = list(range(16))
    generator = torch.Generator(device="cuda").manual_seed(42)
    initial_state = generator.get_state().clone()

    def draw(rows, row_seeds):
        return TorchSampling.sample_from_logits(
            rows,
            temperature=1.0,
            top_k=0,
            top_p=0.5,
            generator=generator,
            row_seeds=row_seeds,
            row_positions=[23] * len(row_seeds),
        )

    reference, _ = assert_replays_bit_exact(
        lambda: draw(logits, seeds),
        (),
        backward=False,
        contention=True,
        what="seeded top-p at an exact cutoff",
    )
    full = reference["out"]
    # Uniform logits put exactly half the mass in the removed lower tail.
    # Match the policy's ascending-sort tie order and exclude the cutoff token.
    indices = logits.sort(dim=-1, descending=False).indices
    support = indices[:, 8:]
    assert (full[:, None] == support).any(dim=-1).all()
    assert torch.equal(full, draw(logits.flip(0), seeds[::-1]).flip(0))
    for row, seed in enumerate(seeds):
        assert torch.equal(full[row : row + 1], draw(logits[row : row + 1], [seed]))
    assert torch.equal(initial_state, generator.get_state())
