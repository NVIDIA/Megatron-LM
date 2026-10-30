# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger's Megatron RMSNorm adapted to the normalization interface."""

from liger_kernel.megatron import LigerMegatronRMSNorm  # type: ignore[import-untyped]


class LigerRMSNorm(LigerMegatronRMSNorm):
    """RMSNorm with checkpoint-compatible weights and no fused residual output.

    The inherited implementation supports zero-centered gamma and marks its weight
    for sequence-parallel gradient reduction. Parameter placement follows PyTorch's
    default device and dtype, as with the other local normalization modules.
    """

    returns_residual: bool = False
