# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.transformer.moe.router_diagnostics import (
    ROUTER_DIAGNOSTIC_CHANNEL_COUNT,
    RouterDiagnosticChannel,
    build_router_diagnostics,
)


def test_build_router_diagnostics_compacts_per_sequence_scores_loads_and_bias():
    scores = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.4, 0.4, 0.2],
            [0.7, 0.2, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    aux_map = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.bool
    )
    actual_map = torch.tensor(
        [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=torch.bool
    )
    padding_mask = torch.tensor([False, False, False, True, True, False])

    diagnostics = build_router_diagnostics(
        scores,
        aux_map,
        actual_map,
        torch.tensor([0.1, -0.1, 0.0]),
        seq_length=3,
        batch_size=2,
        padding_mask=padding_mask,
    )

    assert diagnostics.shape == (2, ROUTER_DIAGNOSTIC_CHANNEL_COUNT, 3)
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.MEAN_SCORE],
        torch.tensor([[0.5, 0.35, 0.15], [0.15, 0.35, 0.5]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.AUX_LOAD],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.ACTUAL_LOAD],
        torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS],
        torch.tensor([[0.1, -0.1, 0.0], [0.1, -0.1, 0.0]]),
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.AUX_ACTUAL_OVERLAP, 0], torch.tensor([0.5, 1.0])
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.VALID_TOKEN_COUNT, 0], torch.tensor([2.0, 2.0])
    )
    torch.testing.assert_close(
        diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0],
        torch.tensor([0.25, 0.55714285]),
    )


def test_build_router_diagnostics_uses_zero_margin_without_a_topk_boundary():
    scores = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
    routing_map = torch.tensor([[True, True], [False, False]])

    diagnostics = build_router_diagnostics(
        scores,
        routing_map,
        routing_map,
        None,
        seq_length=1,
        batch_size=2,
        padding_mask=torch.tensor([False, True]),
    )

    margins = diagnostics[:, RouterDiagnosticChannel.TOPK_BOUNDARY_RELATIVE_MARGIN, 0]
    torch.testing.assert_close(margins, torch.zeros(2))
    assert torch.isfinite(margins).all()
