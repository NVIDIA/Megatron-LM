# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time selection of the existing GDN and GDN2 recurrences."""

from typing import Literal, cast

from megatron.core.ops.ssm.gated_delta import GatedDeltaRuleInterface


def select_gated_delta_rule(
    variant: Literal["gdn", "gdn2"], deterministic: bool = False
) -> GatedDeltaRuleInterface:
    """Return the original reference or FLA callable without wrapping its forward."""
    if variant == "gdn":
        if deterministic:
            from megatron.core.ops.ssm.gated_delta.reference import torch_chunk_gated_delta_rule

            # The shared protocol cannot express the variant's required gate keywords.
            return cast(GatedDeltaRuleInterface, torch_chunk_gated_delta_rule)
        from megatron.core.ops.ssm.gated_delta.fla import chunk_gated_delta_rule

        if chunk_gated_delta_rule is None:
            raise ImportError(
                "GDN requires flash-linear-attention with the gated_delta_rule kernel."
            )
        return chunk_gated_delta_rule
    if variant == "gdn2":
        if deterministic:
            from megatron.core.ops.ssm.gated_delta.reference_gdn2 import torch_chunk_gdn2

            return cast(GatedDeltaRuleInterface, torch_chunk_gdn2)
        from megatron.core.ops.ssm.gated_delta.fla import chunk_gdn2

        if chunk_gdn2 is None:
            raise ImportError("GDN2 requires flash-linear-attention >= 0.5.1 with the gdn2 kernel.")
        return chunk_gdn2
    raise ValueError(f"Unknown gated delta variant: {variant!r}")
