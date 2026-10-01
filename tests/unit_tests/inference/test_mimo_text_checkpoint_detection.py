# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MIMO text extraction must retain the trained global-batch router biases."""

from argparse import Namespace

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    vlm_dynamic_inference,
)


@pytest.mark.parametrize("provider", ["nemotron-moe-vlm", "nemotron-moe-mistral-vit"])
@pytest.mark.parametrize("scope", ["global_batch", "micro_batch", None])
@pytest.mark.parametrize("routing_type", ["quantile_balancing", "none"])
@pytest.mark.parametrize("enable_expert_bias", [False, True])
def test_mimo_text_restores_global_batch_qb_bias(
    monkeypatch, provider, scope, routing_type, enable_expert_bias
):
    """Translate only global-batch QB, preserving ordinary expert-bias settings."""
    args = Namespace(
        model_provider="gpt",
        moe_router_enable_expert_bias=enable_expert_bias,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
    )
    saved = Namespace(
        model_provider=provider,
        moe_router_load_balancing_type=routing_type,
        moe_router_quantile_balancing_estimation_scope=scope,
    )
    monkeypatch.setattr(vlm_dynamic_inference, "load_args_from_checkpoint", lambda _: (args, saved))
    assert not vlm_dynamic_inference._detect_vlm_from_checkpoint(args)
    assert args.model_provider == "hybrid"
    assert args.checkpoint_model_prefix == "language_model.module.module."
    global_batch_qb = routing_type == "quantile_balancing" and scope == "global_batch"
    assert args.moe_router_enable_expert_bias == (enable_expert_bias or global_batch_qb)
    assert args.moe_router_load_balancing_type == "none"
