# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The ``--deterministic-mode`` arg guard in ``megatron/training/determinism.py``.

``TransformerConfig.__post_init__`` resolves ``moe_router_aux_loss_fusion`` when it is
unset, but ``apply_determinism_to_args`` runs on the argparse Namespace before any config
exists, so it re-derives the same fallback. These tests pin the two derivations together:
the guard must reject exactly the arg combinations whose config ends up with the fusion on.
"""

import argparse

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.determinism import (
    ARG_VALUES_REQUIRED_FOR_DETERMINISM,
    apply_determinism_to_args,
)


@pytest.fixture(autouse=True)
def restore_torch_determinism():
    """apply_determinism_to_args flips a torch global on the paths that don't raise."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def make_args(**kwargs):
    """A Namespace holding only what apply_determinism_to_args reads."""
    defaults = {
        **ARG_VALUES_REQUIRED_FOR_DETERMINISM,
        "moe_router_fusion": False,
        "moe_router_aux_loss_fusion": None,
    }
    return argparse.Namespace(**{**defaults, **kwargs})


@pytest.mark.internal
@pytest.mark.parametrize("router_fusion", [False, True])
@pytest.mark.parametrize("aux_loss_fusion", [None, False, True])
def test_guard_agrees_with_config_resolution(router_fusion, aux_loss_fusion):
    """The guard rejects exactly the args whose config resolves the fusion on.

    Unset is the case that matters: it inherits ``moe_router_fusion``, so dropping the
    fallback here would let ``--moe-router-fusion --deterministic-mode`` through.
    """
    fusion_flags = dict(
        moe_router_fusion=router_fusion, moe_router_aux_loss_fusion=aux_loss_fusion
    )
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, num_moe_experts=4, **fusion_flags
    )

    if config.moe_router_aux_loss_fusion:
        with pytest.raises(AssertionError, match="moe_router_aux_loss_fusion"):
            apply_determinism_to_args(make_args(**fusion_flags))
    else:
        # Opting out explicitly keeps fused TopK routing available under determinism.
        apply_determinism_to_args(make_args(**fusion_flags))


@pytest.mark.internal
def test_other_required_args_still_checked():
    """The aux-loss branch is additive -- it must not shadow the dict-driven checks."""
    with pytest.raises(AssertionError, match="cross_entropy_loss_fusion"):
        apply_determinism_to_args(make_args(cross_entropy_loss_fusion=True))
