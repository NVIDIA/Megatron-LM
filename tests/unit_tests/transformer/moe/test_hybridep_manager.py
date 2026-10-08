# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager


@pytest.mark.parametrize(
    ("expert_rank_capacity_factor", "expected_num_permuted_tokens"), [(None, None), (1.5, 128)]
)
def test_combine_only_releases_dynamic_token_count(
    monkeypatch, expert_rank_capacity_factor, expected_num_permuted_tokens
):
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(moe_permute_fusion_into_hybridep=False)
    manager.handle = object()
    manager.num_permuted_tokens = 128
    manager.pad_multiple = None
    manager.drop_and_pad = False
    manager.moe_expert_rank_capacity_factor = expert_rank_capacity_factor
    manager._original_num_tokens = None
    manager._padded_num_tokens = None

    def fake_hybrid_ep_combine(**kwargs):
        assert kwargs["handle"] is manager.handle
        assert kwargs["num_permuted_tokens"] == 128
        return kwargs["x"]

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine", fake_hybrid_ep_combine
    )

    hidden_states = torch.empty(4, 8)
    assert manager.combine(hidden_states) is hidden_states
    assert manager.handle is None
    assert manager.num_permuted_tokens == expected_num_permuted_tokens
