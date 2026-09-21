# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The GDN output-gate activation is selectable independently of the model-wide activation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net.common import resolve_output_gate_activation
from megatron.core.transformer.transformer_config import TransformerConfig


def test_default_reuses_the_model_activation():
    assert resolve_output_gate_activation(SimpleNamespace(), F.silu) is F.silu
    assert (
        resolve_output_gate_activation(SimpleNamespace(gdn_output_gate_activation=None), F.gelu)
        is F.gelu
    )


def test_sigmoid_switches_only_the_gate():
    gate = resolve_output_gate_activation(
        SimpleNamespace(gdn_output_gate_activation="sigmoid"), F.silu
    )
    assert gate is torch.sigmoid
    x = torch.linspace(-4, 4, 9)
    torch.testing.assert_close(gate(x), torch.sigmoid(x))
    assert not torch.allclose(gate(x), F.silu(x))


def test_silu_is_explicit_default():
    assert resolve_output_gate_activation(
        SimpleNamespace(gdn_output_gate_activation="silu"), F.gelu
    ) is F.silu


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="gdn_output_gate_activation"):
        resolve_output_gate_activation(SimpleNamespace(gdn_output_gate_activation="tanh"), F.silu)


def test_transformer_config_accepts_the_knob():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        gdn_output_gate_activation="sigmoid",
    )
    assert config.gdn_output_gate_activation == "sigmoid"
    assert resolve_output_gate_activation(config, F.silu) is torch.sigmoid
