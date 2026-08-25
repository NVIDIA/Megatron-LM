# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compact observations used to diagnose MoE router load balancing."""

from enum import IntEnum

import torch


class RouterDiagnosticChannel(IntEnum):
    """Channels in the compact per-sequence router diagnostic tensor."""

    MEAN_SCORE = 0
    AUX_LOAD = 1
    ACTUAL_LOAD = 2
    EXPERT_BIAS = 3
    AUX_ACTUAL_OVERLAP = 4
    VALID_TOKEN_COUNT = 5
    TOPK_BOUNDARY_RELATIVE_MARGIN = 6


ROUTER_DIAGNOSTIC_CHANNEL_COUNT = len(RouterDiagnosticChannel)


@torch.no_grad()
def build_router_diagnostics(
    scores_for_aux_loss: torch.Tensor,
    routing_map_for_aux_loss: torch.Tensor,
    actual_routing_map: torch.Tensor,
    expert_bias: torch.Tensor | None,
    seq_length: int,
    batch_size: int,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build compact score, load, and bias summaries for each local sequence.

    The returned tensor has shape ``[batch_size, channels, num_experts]``. Distribution-valued
    channels are normalized over experts. Scalar channels use element zero and leave the
    remaining expert elements as zero.

    Args:
        scores_for_aux_loss: Normalized all-expert scores with shape ``[tokens, num_experts]``.
        routing_map_for_aux_loss: Unbiased top-k assignments with the same shape.
        actual_routing_map: Assignments used for token dispatch with the same shape.
        expert_bias: Current expert-selection bias, or ``None`` when bias routing is disabled.
        seq_length: Local sequence length before the token dimension was flattened.
        batch_size: Local micro-batch size.
        padding_mask: Flattened mask where ``True`` marks padding.

    Returns:
        Float32 diagnostic tensor with one compact observation per sequence.
    """
    if seq_length <= 0 or batch_size <= 0:
        raise ValueError("Router diagnostics require positive sequence and batch dimensions.")
    if scores_for_aux_loss.ndim != 2:
        raise ValueError("Router diagnostic scores must have shape [tokens, num_experts].")
    expected_tokens = seq_length * batch_size
    num_tokens, num_experts = scores_for_aux_loss.shape
    if num_tokens != expected_tokens or num_experts < 2:
        raise ValueError(
            "Router diagnostic scores must match seq_length * batch_size and have at least "
            "two experts."
        )
    for name, value in (
        ("routing_map_for_aux_loss", routing_map_for_aux_loss),
        ("actual_routing_map", actual_routing_map),
    ):
        if value.shape != scores_for_aux_loss.shape:
            raise ValueError(f"{name} must have the same shape as scores_for_aux_loss.")
    if expert_bias is not None and expert_bias.shape != (num_experts,):
        raise ValueError("expert_bias must have shape [num_experts].")

    scores = scores_for_aux_loss.float().reshape(seq_length, batch_size, num_experts)
    aux_map = routing_map_for_aux_loss.bool().reshape(seq_length, batch_size, num_experts)
    actual_map = actual_routing_map.bool().reshape(seq_length, batch_size, num_experts)
    if padding_mask is None:
        valid_mask = torch.ones(
            (seq_length, batch_size), dtype=torch.bool, device=scores_for_aux_loss.device
        )
    else:
        if padding_mask.numel() != expected_tokens:
            raise ValueError("padding_mask must contain seq_length * batch_size elements.")
        valid_mask = ~padding_mask.bool().reshape(seq_length, batch_size)
        expanded_valid_mask = valid_mask.unsqueeze(-1)
        scores = scores * expanded_valid_mask
        aux_map = aux_map & expanded_valid_mask
        actual_map = actual_map & expanded_valid_mask

    valid_token_count = valid_mask.sum(dim=0).float()
    mean_score = scores.sum(dim=0) / valid_token_count.clamp_min(1).unsqueeze(-1)
    aux_counts = aux_map.sum(dim=0).float()
    actual_counts = actual_map.sum(dim=0).float()
    aux_load = aux_counts / aux_counts.sum(dim=-1, keepdim=True).clamp_min(1)
    actual_load = actual_counts / actual_counts.sum(dim=-1, keepdim=True).clamp_min(1)
    overlap = (aux_map & actual_map).sum(dim=(0, 2)).float()
    overlap = overlap / aux_counts.sum(dim=-1).clamp_min(1)

    has_boundary = aux_map.any(dim=-1) & (~aux_map).any(dim=-1) & valid_mask
    selected_floor = scores.masked_fill(~aux_map, float("inf")).amin(dim=-1)
    unselected_ceiling = scores.masked_fill(aux_map, float("-inf")).amax(dim=-1)
    # Rows with no selected/unselected boundary use equal finite sentinels so their
    # relative margin is zero without introducing infinities or NaNs.
    selected_floor = torch.where(has_boundary, selected_floor, torch.ones_like(selected_floor))
    unselected_ceiling = torch.where(has_boundary, unselected_ceiling, selected_floor)
    boundary_margin = (selected_floor - unselected_ceiling).clamp_min(0)
    boundary_relative_margin = boundary_margin / selected_floor.clamp_min(
        torch.finfo(scores.dtype).eps
    )
    boundary_relative_margin = boundary_relative_margin.sum(dim=0) / has_boundary.sum(
        dim=0
    ).clamp_min(1)

    diagnostics = torch.zeros(
        (batch_size, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, num_experts),
        dtype=torch.float32,
        device=scores_for_aux_loss.device,
    )
    diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE] = mean_score
    diagnostics[:, RouterDiagnosticChannel.AUX_LOAD] = aux_load
    diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD] = actual_load
    if expert_bias is not None:
        diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS] = expert_bias.float()
    diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0] = overlap
    diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0] = valid_token_count
    diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0] = (
        boundary_relative_margin
    )
    return diagnostics
