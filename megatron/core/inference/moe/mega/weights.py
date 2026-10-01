# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Convert Megatron inference GroupedMLP weights to FlashInfer MoEWeightPack."""

from __future__ import annotations

import torch

from megatron.core.inference.moe.mega._deps import require_flashinfer_moe_ep


def megatron_grouped_weights_to_moe_pack(fc1_weight: torch.Tensor, fc2_weight: torch.Tensor):
    """Build FlashInfer ``MoEWeightPack`` from stacked expert BF16 weights.

    The mega kernel's ``preprocess_mega_weights`` requires exactly these shapes
    and bfloat16 dtype; the gate+up stacking means only gated activations
    (SwiGLU) are representable.

    Args:
        fc1_weight: [num_local_experts, 2 * intermediate, hidden]
        fc2_weight: [num_local_experts, hidden, intermediate]
    """
    require_flashinfer_moe_ep()
    from flashinfer.moe_ep.weights import MoEWeightPack

    if fc1_weight.ndim != 3 or fc2_weight.ndim != 3:
        raise ValueError(
            "Expected 3D expert stacks for mega MoE; "
            f"got fc1={tuple(fc1_weight.shape)}, fc2={tuple(fc2_weight.shape)}"
        )
    if fc1_weight.shape[1] != 2 * fc2_weight.shape[2]:
        raise ValueError(
            "Mega MoE expects gate+up stacked fc1 (2 * intermediate rows); got "
            f"fc1={tuple(fc1_weight.shape)}, fc2={tuple(fc2_weight.shape)}. "
            "Set gated_linear_unit=True with a SwiGLU activation."
        )
    return MoEWeightPack(w13=fc1_weight.contiguous(), w2=fc2_weight.contiguous())
