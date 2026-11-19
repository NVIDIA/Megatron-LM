# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP32 -> BF16 copies with a selectable rounding mode.

Round-to-nearest maps an FP32 value to the closest BF16 value, so any change smaller than
half a BF16 ULP is lost. Stochastic rounding picks one of the two neighbouring BF16 values
with probability proportional to proximity, so the BF16 result equals the FP32 input in
expectation:

    P(round up) = (w - w_lo) / (w_hi - w_lo)

The optimizer uses this when copying FP32 main params back to BF16 model params: with the
small learning rates typical of RL post-training, an Adam update is often much smaller than
half a BF16 ULP, and round-to-nearest would leave the BF16 weights unchanged for many steps
while the FP32 main params keep drifting. See "Closing the BF16-FP16 Convergence Speed Gap
in RL Post-Training with Stochastic Rounding" (arXiv:2611.13545).

The random bits come from a stateless integer hash of (FP32 bit pattern, element index,
seed) rather than from an RNG. Ranks holding replicas of the same tensor therefore round
identically without communication, nothing needs to be checkpointed, and a run is
reproducible. When the input changes (e.g. main params after each step), so do the hash
inputs.
"""

from typing import Literal

import torch

RoundingMode = Literal["nearest", "stochastic"]

# Elements processed per chunk; bounds the temporary FP32/int64 workspace to ~128 MiB.
_CHUNK = 1 << 22
_MASK32 = 0xFFFFFFFF


def _fmix32_(h: torch.Tensor) -> torch.Tensor:
    """MurmurHash3 32-bit finalizer, in place on an int64 tensor holding values in [0, 2**32)."""
    h ^= h >> 16
    h.mul_(0x85EBCA6B).bitwise_and_(_MASK32)
    h ^= h >> 13
    h.mul_(0xC2B2AE35).bitwise_and_(_MASK32)
    h ^= h >> 16
    return h


def _hash_seed(seed: int) -> int:
    h = seed & _MASK32
    h ^= h >> 16
    h = h * 0x85EBCA6B & _MASK32
    h ^= h >> 13
    h = h * 0xC2B2AE35 & _MASK32
    return h ^ h >> 16


def stochastic_round_to_bf16_(dst: torch.Tensor, src: torch.Tensor, offset: int, seed: int) -> None:
    """Stochastically round FP32 ``src`` into BF16 ``dst``. ``src`` is overwritten.

    Args:
        dst: BF16 output, same number of elements as ``src``.
        src: FP32 scratch tensor on the same device as ``dst``; modified in place.
        offset: index of ``src[0]`` within the full tensor, so that results do not depend on
            how the tensor is chunked.
        seed: user seed mixed into the hash.

    NaN is preserved; +-0, +-inf and values exactly representable in BF16 are unchanged.
    """
    nan = src.isnan()
    bits = src.view(torch.int32)
    h = torch.arange(offset, offset + src.numel(), dtype=torch.int64, device=src.device)
    h.mul_(0x9E3779B1).add_(_hash_seed(seed)).bitwise_xor_(bits.to(torch.int64))
    h.bitwise_and_(_MASK32)
    bits += (_fmix32_(h) >> 16).to(torch.int32)
    # With the low 16 bits cleared, the FP32 -> BF16 cast below is exact.
    bits &= -(1 << 16)
    src.masked_fill_(nan, float("nan"))
    dst.copy_(src)


def copy_with_rounding_(
    dst: torch.Tensor, src: torch.Tensor, rounding: RoundingMode = "nearest", seed: int = 0
) -> None:
    """Copy ``src`` into ``dst`` with the given rounding mode.

    ``src`` may live on another device (e.g. CPU); rounding is done on ``dst``'s device.
    ``rounding="nearest"`` is a plain non-blocking ``copy_``. ``"stochastic"`` applies only to
    FP32 -> BF16 copies; other dtype pairs fall back to ``copy_``.
    """
    if rounding == "nearest" or dst.dtype != torch.bfloat16 or src.dtype != torch.float32:
        dst.copy_(src, non_blocking=True)
        return
    assert rounding == "stochastic", f"Unknown rounding mode: {rounding}"
    flat_dst, flat_src = dst.view(-1), src.view(-1)
    for start in range(0, flat_dst.numel(), _CHUNK):
        chunk = slice(start, start + _CHUNK)
        scratch = flat_src[chunk].to(dst.device, non_blocking=True, copy=True)
        stochastic_round_to_bf16_(flat_dst[chunk], scratch, start, seed)
