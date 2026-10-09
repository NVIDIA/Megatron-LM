# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared-prefix HybridStack guard tests that need no GPU or process-group setup."""

import subprocess
import sys
from pathlib import Path
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
from megatron.core.transformer.attention import SelfAttention
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
        # A coefficient paired with "none" adds no loss (TopKRouter.get_aux_loss_coeff).
        ("none", 1e-4),
        (["none", "aux_loss"], [1e-3, 0.0]),
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
        (["none", "global_aux_loss"], [0.0, 1e-3], "auxiliary router loss"),
    ],
)
def test_routing_load_balancing_and_active_aux_loss_rejected(load_balancing, coeff, message):
    config = _config(
        moe=True, moe_router_load_balancing_type=load_balancing, moe_aux_loss_coeff=coeff
    )
    with pytest.raises(NotImplementedError, match=message):
        _validate_hybrid_stack(_stack(config), _LAYOUT, physical_len=_PHYSICAL_LEN[1])


@pytest.mark.parametrize("window_size", [None, (-1, -1), [-1, -1]])
def test_full_attention_window_accepted(window_size):
    """A YAML/CLI list [-1, -1] is the same full window as the tuple default."""
    stack = _stack(_config(window_size=window_size))
    _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])


def _attention_layer():
    layer = Mock(spec=TransformerLayer)
    layer.self_attention = Mock(spec=SelfAttention)
    layer.self_attention.checkpoint_core_attention = False
    layer.self_attention.pg_collection = SimpleNamespace(tp=_group(1), cp=_group(1))
    layer.is_moe_layer = False
    return layer


@pytest.mark.parametrize("installed", [False, True])
def test_old_or_missing_flash_attn_rejected_before_any_layer(monkeypatch, installed):
    def is_fa_min_version(version):
        if not installed:
            raise ImportError("No module named 'flash_attn'")
        return False

    monkeypatch.setattr(shared_prefix, "is_fa_min_version", is_fa_min_version)
    stack = _stack(_config(), [_attention_layer()])
    with pytest.raises(RuntimeError, match="flash-attn >= 2.7.0"):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])
    # Attention-free stacks never reach the fused attention kernel.
    _validate_hybrid_stack(_stack(_config()), _LAYOUT, physical_len=_PHYSICAL_LEN[1])


# Every configuration guard of _validate_hybrid_stack that the other tests do not cover.
_REJECTED_CONFIGS = [
    ({"moe_num_hash_layers": 1}, NotImplementedError, "hash MoE routing"),
    ({"quant_recipe": object()}, NotImplementedError, "quantization recipes"),
    ({"wide_residual": object()}, NotImplementedError, "wide residual streams"),
    ({"enable_mhc_connections": True}, NotImplementedError, "mHC connections"),
    ({"attn_logit_softcapping": 30.0}, NotImplementedError, "logit softcapping"),
    ({"sequence_parallel": True}, NotImplementedError, "sequence parallelism requires TP>1"),
    ({"tensor_model_parallel_size": 2}, RuntimeError, "tensor-parallel config"),
    ({"context_parallel_size": 2}, RuntimeError, "context-parallel config"),
    (
        {"recompute_granularity": "full", "recompute_method": "block", "recompute_num_layers": 1},
        NotImplementedError,
        "uniform method",
    ),
    (
        {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 0},
        ValueError,
        "recompute_num_layers >= 1",
    ),
    ({"fine_grained_activation_offloading": True}, NotImplementedError, "activation offloading"),
    ({"cuda_graph_impl": "local"}, NotImplementedError, "CUDA graphs"),
    ({"fp8": "hybrid"}, NotImplementedError, "fp16/bf16 only"),
    ({"fp4": "e2m1"}, NotImplementedError, "fp16/bf16 only"),
    ({"hidden_dropout": 0.1}, NotImplementedError, "zero dropout"),
    ({"attention_dropout": 0.1}, NotImplementedError, "zero dropout"),
    ({"window_size": (128, 0)}, NotImplementedError, "sliding-window attention"),
    ({"softmax_type": "off-by-one"}, NotImplementedError, "vanilla softmax"),
]
_REJECTED_MOE_CONFIGS = [
    ({"moe_router_force_load_balancing": True}, "forced MoE routing"),
    ({"moe_router_force_biased": 1.0}, "forced MoE router bias"),
    ({"moe_z_loss_coeff": 1e-3}, "router z-loss"),
    ({"moe_input_jitter_eps": 0.1}, "input jitter"),
    ({"mlp_chunks_for_training": 2}, "MLP chunking"),
    # Expert-bias accounting needs logical completion lengths; _LAYOUT has none.
    ({"moe_router_enable_expert_bias": True}, "explicit physical branch padding"),
]


@pytest.mark.parametrize(
    "overrides, error, message",
    _REJECTED_CONFIGS,
    ids=lambda case: "-".join(case) if isinstance(case, dict) else None,
)
def test_unsupported_stack_config_rejected(overrides, error, message):
    stack = _stack(_config(**overrides))
    with pytest.raises(error, match=message):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])


@pytest.mark.parametrize(
    "overrides, message",
    _REJECTED_MOE_CONFIGS,
    ids=lambda case: "-".join(case) if isinstance(case, dict) else None,
)
def test_unsupported_moe_config_rejected(overrides, message):
    stack = _stack(_config(moe=True, **overrides))
    with pytest.raises(NotImplementedError, match=message):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])


def test_unsupported_stack_topology_and_layers_rejected():
    stack = _stack(_config())
    stack.pp_group = _group(2)
    with pytest.raises(NotImplementedError, match="PP1 only"):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])
    stack = _stack(_config(), [Mock(spec=torch.nn.Linear)])
    with pytest.raises(NotImplementedError, match="not implemented for"):
        _validate_hybrid_stack(stack, _LAYOUT, physical_len=_PHYSICAL_LEN[1])
    checkpointed = _attention_layer()
    checkpointed.self_attention.checkpoint_core_attention = True
    with pytest.raises(NotImplementedError, match="selective core-attention recomputation"):
        _validate_hybrid_stack(
            _stack(_config(), [checkpointed]), _LAYOUT, physical_len=_PHYSICAL_LEN[1]
        )
    with pytest.raises(NotImplementedError, match="QK-clipping"):
        _validate_hybrid_stack(
            _stack(_config(qk_clip=True), [_attention_layer()]),
            _LAYOUT,
            physical_len=_PHYSICAL_LEN[1],
        )


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


# Run in a fresh interpreter as if einops were not installed (Transformer Engine needs einops, so
# it is hidden too), import HybridModel, and check that the adapter was not loaded.
_IMPORT_HYBRID_MODEL_WITHOUT_EINOPS = """
import sys

sys.path.insert(0, sys.argv[1])


class HideModules:
    def find_spec(self, name, path=None, target=None):
        top = name.partition(".")[0]
        if top in ("einops", "transformer_engine"):
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None


sys.meta_path.insert(0, HideModules())
import megatron.core.models.hybrid.hybrid_model

assert "megatron.core.models.hybrid.shared_prefix" not in sys.modules
"""


def test_hybrid_model_import_does_not_load_the_adapter():
    """Without a layout, HybridModel keeps its import graph and does not need einops."""
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [sys.executable, "-c", _IMPORT_HYBRID_MODEL_WITHOUT_EINOPS, str(repo)], check=True
    )
