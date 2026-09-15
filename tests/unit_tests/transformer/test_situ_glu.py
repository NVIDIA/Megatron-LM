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


def test_situlu_reference_matches_kimi_bf16_precision_and_backward():
    """The fallback evaluates both branches in FP32 and returns the original dtype."""
    x = torch.linspace(-20, 20, 30, device="cuda", dtype=torch.bfloat16).reshape(3, 10)
    x = x.detach().requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    gate, up = reference_x.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    expected = 3.0 * torch.tanh(gate / 3.0) * torch.sigmoid(gate)
    expected = (expected * (7.0 * torch.tanh(up / 7.0))).to(reference_x.dtype)

    actual = situlu(x, 3.0, 7.0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)


def test_situlu_reference_matches_fixed_values():
    """Check independent values, including positive and negative saturation."""
    x = torch.tensor(
        [[0.0, 1.0, 0.0, 2.0], [-1.0, 4.0, -2.0, 25.0], [100.0, -100.0, 100.0, -100.0]],
        device="cuda",
    )
    expected = torch.tensor(
        [[0.0, 1.4293511], [0.5258289, 56.959320], [99.932930, 3.717581e-42]], device="cuda"
    )

    torch.testing.assert_close(situlu(x), expected, rtol=1e-6, atol=1e-5)


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
