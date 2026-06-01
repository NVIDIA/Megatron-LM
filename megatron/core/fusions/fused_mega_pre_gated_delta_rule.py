# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mega fused pre-gated-delta-rule API placeholder."""

from typing import Optional, Tuple

import torch
from torch import Tensor


def fused_mega_pre_gated_delta_rule(
    qkvzba: Tensor,
    conv1d_weight: Tensor,
    conv1d_bias: Optional[Tensor],
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    use_qk_l2norm: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    seq_idx: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Mega fused pre-gated-delta-rule entry point.

    Args:
        qkvzba: ``[seq_len, batch, in_proj_dim]`` projection output.
        conv1d_weight: ``[conv_dim, 1, k_w]`` depthwise conv weight.
        conv1d_bias: Must be ``None`` in the planned mega path.
        A_log: ``[num_value_heads]`` raw decay parameter.
        dt_bias: ``[num_value_heads]`` time-step bias.
        num_key_heads / num_value_heads / key_head_dim / value_head_dim: GDN
            architecture parameters.
        use_qk_l2norm: Must be ``True`` for parity with the streamed path.
        cu_seqlens: Optional packed THD cumulative sequence lengths.
        seq_idx: Optional precomputed token-to-sequence map for packed THD mode.

    Returns:
        ``(query, key, value, gate, beta, g)`` matching the unfused and streamed
        fused pre-GDR APIs.
    """

    assert qkvzba.is_cuda, (
        "fused_mega_pre_gated_delta_rule requires CUDA inputs; "
        f"got qkvzba.device={qkvzba.device}."
    )
    assert conv1d_bias is None, (
        "Conv bias is not supported by fused_mega_pre_gated_delta_rule "
        "(production GDN config has none)."
    )
    assert use_qk_l2norm, (
        "use_qk_l2norm=False is not supported by fused_mega_pre_gated_delta_rule "
        "(the planned backward closes over the l2norm path)."
    )
    assert num_value_heads % num_key_heads == 0, (
        f"{num_value_heads=} must be a multiple of {num_key_heads=}."
    )
    if cu_seqlens is not None:
        assert cu_seqlens.is_cuda, (
            "Packed fused_mega_pre_gated_delta_rule requires CUDA cu_seqlens; "
            f"got cu_seqlens.device={cu_seqlens.device}."
        )
        assert cu_seqlens.dtype == torch.int32, (
            "Packed fused_mega_pre_gated_delta_rule requires int32 cu_seqlens; "
            f"got {cu_seqlens.dtype=}."
        )
        assert cu_seqlens.dim() == 1, (
            "Packed fused_mega_pre_gated_delta_rule expects 1-D cu_seqlens; "
            f"got {cu_seqlens.shape=}."
        )
        assert qkvzba.shape[1] == 1, (
            "Packed THD fused_mega_pre_gated_delta_rule expects batch dimension 1; "
            f"got qkvzba.shape={qkvzba.shape}."
        )
        assert cu_seqlens.shape[0] >= 2, (
            "Packed fused_mega_pre_gated_delta_rule requires at least one packed sequence; "
            f"got {cu_seqlens.shape=}."
        )
    else:
        assert seq_idx is None, "seq_idx requires cu_seqlens for packed THD mode."

    # TODO: Implement the mega fused forward/backward kernels behind this API.
    raise NotImplementedError("fused_mega_pre_gated_delta_rule is an API placeholder.")
