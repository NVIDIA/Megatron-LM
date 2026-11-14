# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rounding used when copying FP32 main params back to BF16 model params.

With the small learning rates typical of RL post-training, an Adam update is often much
smaller than half a BF16 ULP of the weight it is applied to. Round-to-nearest then maps the
updated FP32 main param back to the same BF16 value, so the BF16 weights used by forward
passes (and by inference engines that receive them) can stay unchanged for many steps while
the FP32 main params keep drifting. Stochastic rounding picks one of the two neighbouring
BF16 values with probability proportional to proximity, so the BF16 weight equals the main
param in expectation:

    P(round up) = (w - w_lo) / (w_hi - w_lo)

See "Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic
Rounding" (arXiv:2611.13545).

The random bits come from a stateless integer hash of (FP32 bit pattern, element index,
seed) rather than from an RNG. Ranks holding replicas of the same parameter therefore round
identically without communication, nothing needs to be checkpointed, and a run is
reproducible. Because the main params change every step, so do the hash inputs.
"""

from typing import Literal

import torch

MainParamsRounding = Literal["nearest", "stochastic"]

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
        offset: index of ``src[0]`` within the full parameter, so that results do not depend
            on how the parameter is chunked.
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


def copy_main_to_model_param_(
    model_param: torch.Tensor,
    main_param: torch.Tensor,
    rounding: MainParamsRounding = "nearest",
    seed: int = 0,
) -> None:
    """Copy an FP32 main param (on CPU or GPU) into its model param on GPU.

    ``rounding="nearest"`` is a plain ``copy_``. ``"stochastic"`` applies only to FP32 -> BF16
    copies; other dtype pairs fall back to ``copy_``.
    """
    if (
        rounding == "nearest"
        or model_param.dtype != torch.bfloat16
        or main_param.dtype != torch.float32
    ):
        model_param.copy_(main_param, non_blocking=True)
        return
    assert rounding == "stochastic", f"Unknown main params rounding: {rounding}"
    flat_model, flat_main = model_param.view(-1), main_param.view(-1)
    for start in range(0, flat_model.numel(), _CHUNK):
        chunk = slice(start, start + _CHUNK)
        scratch = flat_main[chunk].to(model_param.device, non_blocking=True, copy=True)
        stochastic_round_to_bf16_(flat_model[chunk], scratch, start, seed)
