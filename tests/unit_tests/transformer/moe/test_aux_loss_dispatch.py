# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Aux-loss dispatch across legacy and deterministic TE APIs."""

import pytest
import torch

from megatron.core.transformer.moe import moe_utils


@pytest.fixture(autouse=True)
def restore_torch_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def _loss(**kwargs):
    return moe_utils.switch_load_balancing_loss_func(
        probs=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        tokens_per_expert=torch.tensor([1, 1]),
        total_num_tokens=2,
        topk=1,
        num_experts=2,
        moe_aux_loss_coeff=0.01,
        **kwargs,
    )


@pytest.mark.internal
@pytest.mark.parametrize("supports_deterministic", [False, True])
@pytest.mark.parametrize(
    "requested,torch_enabled", [(False, False), (True, False), (False, True), (True, True)]
)
def test_aux_loss_dispatch(monkeypatch, supports_deterministic, requested, torch_enabled):
    calls = []
    result = torch.tensor(0.01)

    def legacy(probs, tokens_per_expert, total_num_tokens, topk, num_experts, coeff):
        calls.append(None)
        return result

    def supported(
        probs, tokens_per_expert, total_num_tokens, topk, num_experts, coeff, deterministic=False
    ):
        calls.append(deterministic)
        return result

    monkeypatch.setattr(moe_utils, "HAVE_TE", True)
    monkeypatch.setattr(
        moe_utils, "fused_moe_aux_loss", supported if supports_deterministic else legacy
    )
    monkeypatch.setattr(
        moe_utils, "te_supports_deterministic_moe_aux_loss", lambda: supports_deterministic
    )
    torch.use_deterministic_algorithms(torch_enabled)
    effective = requested or torch_enabled

    if effective and not supports_deterministic:
        with pytest.raises(ValueError, match="deterministic"):
            _loss(fused=True, deterministic=requested)
        assert not calls
    else:
        assert _loss(fused=True, deterministic=requested) is result
        assert calls == [effective if supports_deterministic else None]


@pytest.mark.internal
def test_unfused_aux_loss_does_not_require_te(monkeypatch):
    monkeypatch.setattr(moe_utils, "HAVE_TE", False)
    monkeypatch.setattr(moe_utils, "fused_moe_aux_loss", None)
    torch.use_deterministic_algorithms(True)
    torch.testing.assert_close(_loss(fused=False, deterministic=True), torch.tensor(0.01))
    with pytest.raises(ValueError, match="not available"):
        _loss(fused=True, deterministic=True)
