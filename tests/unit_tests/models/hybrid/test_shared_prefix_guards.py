# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared-prefix HybridStack guard tests that need no GPU or process-group setup."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.models.hybrid import shared_prefix
from megatron.core.models.hybrid.shared_prefix import (
    _validate_hybrid_stack,
    forward_hybrid_stack_shared_prefix,
)
from megatron.core.models.hybrid.shared_prefix_layout import SharedPrefixLayout
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


def _group(size):
    return SimpleNamespace(size=lambda: size, rank=lambda: 0)


def _config(*, cp_size=1, moe=False, **overrides):
    """Return an MCore config with its defaults, then apply raw attribute overrides."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4 if moe else None,
        moe_ffn_hidden_size=8 if moe else None,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    config.context_parallel_size = cp_size
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _stack(config, layers=(), cp_size=1):
    tp, cp = _group(1), _group(cp_size)
    return SimpleNamespace(
        config=config,
        tp_group=tp,
        pp_group=_group(1),
        pg_collection=SimpleNamespace(tp=tp, cp=cp),
        layers=list(layers),
        training=True,
        post_process=False,
        post_layer_norm=False,
    )


def _mamba_layer(cp_size, *, contiguous):
    layer = Mock(spec=MambaLayer)
    layer.mixer = Mock(spec=MambaMixer)
    layer.mixer.pg_collection = SimpleNamespace(tp=_group(1), cp=_group(cp_size))
    layer.mixer.cp = SimpleNamespace(cp_size=cp_size, sequence_is_contiguous=contiguous)
    return layer


# Global star [3 | 2 | 5 | 1] is 11 tokens; CP2 pads it to a multiple of 2 * CP.
_LAYOUT = SharedPrefixLayout(3, (2, 5, 1))
_PHYSICAL_LEN = {1: 11, 2: 12}


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("contiguous", [False, True])
def test_mamba_cp_requires_zigzag_linear_layout(cp_size, contiguous):
    stack = _stack(
        _config(cp_size=cp_size), [_mamba_layer(cp_size, contiguous=contiguous)], cp_size
    )

    def validate():
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[cp_size])

    # Isolate the stack guard from the Mamba kernel/TP checks of the per-layer fork helper.
    with patch.object(shared_prefix, "_validate_mamba_fork"):
        if cp_size > 1 and contiguous:
            with pytest.raises(NotImplementedError, match="linear_cp_layout='zigzag'"):
                validate()
        else:
            validate()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"moe_expert_rank_capacity_factor": 1.0}, "expert rank capacity"),
        ({"moe_enable_routing_replay": True}, "routing replay"),
        ({"moe_expert_capacity_factor": 1.0}, "expert capacity or token dropping"),
    ],
)
def test_moe_token_dropping_and_routing_replay_rejected(overrides, message):
    stack = _stack(_config(moe=True, **overrides))
    with pytest.raises(NotImplementedError, match=message):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])


@pytest.mark.parametrize(
    "load_balancing, coeff",
    [
        (None, None),  # MCore defaults: aux_loss with a zero coefficient.
        ("none", 0.0),
        ("seq_aux_loss", 0.0),
        ("global_aux_loss", 0.0),
        (["aux_loss", "seq_aux_loss"], [0.0, 0.0]),
    ],
)
def test_inactive_aux_loss_load_balancing_accepted(load_balancing, coeff):
    if load_balancing is None:
        config = _config(moe=True)
        assert config.moe_router_load_balancing_type == "aux_loss"
        assert config.moe_aux_loss_coeff == 0.0
    else:
        config = _config(
            moe=True, moe_router_load_balancing_type=load_balancing, moe_aux_loss_coeff=coeff
        )
    _validate_hybrid_stack(_stack(config), _LAYOUT, physical_len=_PHYSICAL_LEN[1])


@pytest.mark.parametrize(
    "load_balancing, coeff, message",
    [
        ("sinkhorn", 0.0, "load balancing type 'sinkhorn'"),
        ("quantile_balancing", 0.0, "load balancing type 'quantile_balancing'"),
        (["aux_loss", "sinkhorn"], [0.0, 0.0], "load balancing type"),
        ("aux_loss", 1e-3, "auxiliary router loss"),
        (["aux_loss", "seq_aux_loss"], [0.0, 1e-3], "auxiliary router loss"),
    ],
)
def test_routing_load_balancing_and_active_aux_loss_rejected(load_balancing, coeff, message):
    config = _config(
        moe=True, moe_router_load_balancing_type=load_balancing, moe_aux_loss_coeff=coeff
    )
    with pytest.raises(NotImplementedError, match=message):
        _validate_hybrid_stack(_stack(config), _LAYOUT, physical_len=_PHYSICAL_LEN[1])


def test_stack_validation_needs_only_the_physical_length():
    stack = _stack(_config())
    _validate_hybrid_stack(stack, _LAYOUT, physical_len=_LAYOUT.total_len)
    with pytest.raises(ValueError, match="does not match layout"):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_LAYOUT.total_len + 1)


def _moe_identity_layer(calls):
    layer = Mock(spec=TransformerLayer)
    layer.self_attention = IdentityOp()
    layer.is_moe_layer = True
    layer.mlp = Mock(spec=MoELayer)
    layer.mlp.router = Mock(spec=TopKRouter)
    # MoELayer declares the scoped slot with a None class default.
    layer.mlp._shared_prefix_token_multiplicities = None

    def forward(hidden_states, attention_mask):
        calls.append(layer.mlp._shared_prefix_token_multiplicities.clone())
        return hidden_states

    layer.side_effect = forward
    return layer


# Branch 1 carries two rows of per-sequence padding; the 9-token star is padded to 12.
_PADDED_LAYOUT = SharedPrefixLayout(3, (5, 1), logical_completion_lens=(3, 1), padding_multiple=4)


@pytest.mark.parametrize("exclude", [None, False, True])
@pytest.mark.parametrize("hybridep", [False, True])
def test_expert_bias_padding_convention_is_caller_specified(exclude, hybridep):
    config = _config(moe=True, moe_router_enable_expert_bias=True)
    if hybridep:
        config.moe_token_dispatcher_type = "flex"
        config.moe_flex_dispatcher_backend = "hybridep"
    calls = []
    stack = _stack(config, [_moe_identity_layer(calls)])
    kwargs = {} if exclude is None else {"exclude_sequence_padding_from_expert_bias": exclude}
    hidden_states = torch.zeros(12, 1, 8, dtype=torch.bfloat16)
    forward_hybrid_stack_shared_prefix(
        stack, hidden_states, _PADDED_LAYOUT, position_embedding_type="none", **kwargs
    )
    # The dispatcher type no longer selects the convention; only the caller's flag does.
    padding = 0.0 if exclude else 1.0
    expected = torch.tensor([2, 2, 2, 1, 1, 1, padding, padding, 1, 0, 0, 0])
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], expected, rtol=0, atol=0)


def test_forward_rejects_unsupported_stack_before_running_layers():
    calls = []
    config = _config(
        moe=True, moe_router_enable_expert_bias=True, moe_expert_rank_capacity_factor=1.0
    )
    stack = _stack(config, [_moe_identity_layer(calls)])
    hidden_states = torch.zeros(12, 1, 8, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="expert rank capacity"):
        forward_hybrid_stack_shared_prefix(
            stack, hidden_states, _PADDED_LAYOUT, position_embedding_type="none"
        )
    assert calls == []
    with pytest.raises(TypeError, match="fp16 or bf16"):
        forward_hybrid_stack_shared_prefix(
            _stack(_config()), hidden_states.float(), _PADDED_LAYOUT, position_embedding_type="none"
        )
