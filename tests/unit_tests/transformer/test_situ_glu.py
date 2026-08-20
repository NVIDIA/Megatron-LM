# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.extensions.transformer_engine as te_extension
from megatron.core.activations import situlu
from megatron.core.transformer.transformer_config import TransformerConfig


class _FakeActivation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_te_activation_ops(monkeypatch, *, include_situ=True):
    ops = SimpleNamespace()
    if include_situ:
        ops.SiTUGLU = type("SiTUGLU", (_FakeActivation,), {})

    monkeypatch.setattr(
        te_extension, "te", SimpleNamespace(pytorch=SimpleNamespace(ops=ops)), raising=False
    )
    monkeypatch.setattr(te_extension, "HAVE_TE", True)
    monkeypatch.setattr(te_extension, "is_te_min_version", lambda *_args, **_kwargs: True)
    return ops


def _config(activation_func, gated_linear_unit, **kwargs):
    defaults = dict(
        activation_func=activation_func,
        gated_linear_unit=gated_linear_unit,
        activation_func_fp8_input_store=False,
        activation_func_clamp_value=None,
        use_fused_weighted_squared_relu=False,
        situ_glu_beta1=4.0,
        situ_glu_beta2=25.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_ordinary_te_situ_glu_activation_selection(monkeypatch):
    """The ordinary TE path selects SiTUGLU with the configured betas."""
    _patch_te_activation_ops(monkeypatch)
    op = te_extension.TEActivationOp(_config(situlu, True))

    assert type(op).__name__ == "SiTUGLU"
    assert op.kwargs == {"beta1": 4.0, "beta2": 25.0}


def test_situ_glu_requires_te_operations(monkeypatch):
    """An older TE cannot cause the SiTU marker to fall through to SwiGLU."""
    _patch_te_activation_ops(monkeypatch, include_situ=False)

    with pytest.raises(RuntimeError, match="pytorch.ops.SiTUGLU"):
        te_extension.TEActivationOp(_config(situlu, True))


def test_situlu_reference_matches_two_branch_formula_and_backpropagates():
    """The fallback soft-caps both branches; it is not unary SiLU times an untouched up branch."""
    x = torch.randn(3, 10, dtype=torch.float64, requires_grad=True)
    gate, up = x.chunk(2, dim=-1)
    expected = 3.0 * torch.tanh(gate / 3.0) * torch.sigmoid(gate)
    expected = expected * (7.0 * torch.tanh(up / 7.0))

    actual = situlu(x, 3.0, 7.0)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_transformer_config_accepts_situ_glu_defaults():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        activation_func=situlu,
        gated_linear_unit=True,
        use_te_activation_func=True,
    )

    assert config.situ_glu_beta1 == 4.0
    assert config.situ_glu_beta2 == 25.0


def test_transformer_config_accepts_pytorch_situ_glu_fallback():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        activation_func=situlu,
        gated_linear_unit=True,
        use_te_activation_func=False,
    )

    assert config.activation_func is situlu


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"gated_linear_unit": False}, "gated_linear_unit=True"),
        ({"activation_func_clamp_value": 1.0}, "does not use activation_func_clamp_value"),
        ({"glu_linear_offset": 1.0}, "requires glu_linear_offset=0.0"),
        ({"situ_glu_beta1": 0.0}, "situ_glu_beta1 must be finite and positive"),
        ({"situ_glu_beta2": float("inf")}, "situ_glu_beta2 must be finite and positive"),
    ],
)
def test_transformer_config_rejects_unsupported_situ_glu(overrides, match):
    kwargs = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        activation_func=situlu,
        gated_linear_unit=True,
        use_te_activation_func=True,
    )
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        TransformerConfig(**kwargs)
