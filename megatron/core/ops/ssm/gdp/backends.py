# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate and load only the selected GDP training or context-parallel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from megatron.core.ops.kernel_metadata import DeterminismPolicy, validate_kernel
from megatron.core.ops.ssm.gdp.kernel_metadata import (
    GDP_CUTEDSL,
    GDP_CUTEDSL_CP,
    GDP_FLA,
    GDP_FLA_CP,
)

if TYPE_CHECKING:
    from megatron.core.ops.ssm.context_parallel.chunkwise import LinearAttentionCPBackend


def select_gated_delta_product(use_cutedsl: bool = False, deterministic: bool = False) -> Callable:
    """Return the requested callable; availability never changes the selection."""
    policy = DeterminismPolicy.WARN if deterministic else DeterminismPolicy.IGNORE
    validate_kernel(GDP_CUTEDSL if use_cutedsl else GDP_FLA, determinism=policy)
    if use_cutedsl:
        from gdp_attn import chunk_gated_delta_product
    else:
        from fla.ops.gated_delta_product import chunk_gated_delta_product

    return chunk_gated_delta_product


def select_gdp_cp_backend(
    use_cutedsl: bool = False, *, recompute_chunk_num: int = 0, deterministic: bool = False
) -> LinearAttentionCPBackend:
    """Construct the selected chunkwise-CP adapter after validating its protocol."""
    policy = DeterminismPolicy.WARN if deterministic else DeterminismPolicy.IGNORE
    validate_kernel(GDP_CUTEDSL_CP if use_cutedsl else GDP_FLA_CP, determinism=policy)
    if use_cutedsl:
        from megatron.core.ops.ssm.context_parallel.gdp_cutedsl import (
            CuTeDSLGatedDeltaProductCPBackend,
        )

        return CuTeDSLGatedDeltaProductCPBackend(recompute_chunk_num=recompute_chunk_num)
    from megatron.core.ops.ssm.context_parallel.gdp import FLAGatedDeltaProductCPBackend

    return FLAGatedDeltaProductCPBackend()
