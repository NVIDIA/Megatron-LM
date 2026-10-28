# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay seeded CUDA draws under changing batch layouts."""

import pytest
import torch

from megatron.core.inference.sampling.request_seed_noise import (
    _request_seed_noise_kernel,
    fill_request_seed_noise,
)
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
