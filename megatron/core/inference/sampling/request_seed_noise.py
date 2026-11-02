# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batched, request-local exponential noise for CUDA sampling."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Seeds", "Positions"])
def _request_seed_noise_kernel(
    Noise, Seeds, Positions, V: tl.constexpr, SINGLE_ROW: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    if SINGLE_ROW:
        seed = Seeds
        position = Positions
    else:
        seed = tl.load(Seeds + row)
        position = tl.load(Positions + row)
    # Unseeded rows have already been filled from the shared Torch generator.
    if seed >= 0:
        token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        position64 = position.to(tl.uint64)
        # Philox's key is the complete request seed; its counter is the
        # vocabulary index and 64-bit absolute position, never the batch row.
        bits, _, _, _ = tl.philox(
            seed,
            token.to(tl.uint32),
            position64.to(tl.uint32),
            (position64 >> 32).to(tl.uint32),
            tl.full((), 0, tl.uint32),
        )
        uniform = tl.uint_to_uniform_float(bits)
        # Uniform is in [0, 1). Clamp its lower endpoint before log; its
        # excluded upper endpoint also keeps exponential noise strictly positive.
        noise = -tl.log(tl.maximum(uniform, 1.1754943508222875e-38))
        tl.store(Noise + row.to(tl.int64) * V + token, noise, token < V)


def fill_request_seed_noise(
    noise: torch.Tensor, seeds: list[int | None], positions: list[int] | None = None
) -> None:
    """Fill all seeded rows in one launch; leave shared-generator rows untouched.

    Every variate is indexed by (request seed, absolute position, vocabulary
    index), not batch slot, launch geometry, or shared-generator state. CUDA
    draws are not bit-identical to Torch's generator-specific implementation.
    """
    if positions is None:
        positions = [0] * len(seeds)
    single_row = len(seeds) == 1
    if single_row:
        # Avoid a host-to-device allocation/copy for single-request decode.
        seed_data = -1 if seeds[0] is None else seeds[0]
        position_data = positions[0]
    else:
        metadata = torch.tensor(
            [[-1 if seed is None else seed for seed in seeds], positions],
            dtype=torch.int64,
            device=noise.device,
        )
        seed_data, position_data = metadata.unbind()
    _request_seed_noise_kernel[(noise.shape[0], triton.cdiv(noise.shape[1], 1024))](
        noise, seed_data, position_data, noise.shape[1], single_row, 1024
    )
