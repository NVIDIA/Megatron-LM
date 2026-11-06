# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ownership, projection gradients and MTP metrics for both private-loss paths."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models import backends
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def tp1():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _select_path(implementation, monkeypatch):
    if implementation == "te":
        if not backends._te_has_memory_efficient_cross_entropy():
            pytest.skip("TE >= 2.19 with overwrite_input API required")
    else:
        monkeypatch.setattr(backends, "_te_has_memory_efficient_cross_entropy", lambda: False)


def _config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="auto",
        cross_entropy_inplace_backward=True,
        use_cpu_initialization=True,
    )


@pytest.mark.parametrize("implementation", ["fallback", "te"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_private_language_loss_preserves_forward_and_reuses_backward(
    implementation, dtype, monkeypatch
):
    _select_path(implementation, monkeypatch)
    model = LanguageModule(_config())
    torch.manual_seed(2493)
    source = torch.randn(17, 3, 4097, device="cuda", dtype=dtype)
    ordinary, owned = source.clone().requires_grad_(), source.clone().requires_grad_()
    labels = torch.randint(4097, (3, 17), device="cuda")
    labels[:, ::5] = -100
    scale = torch.randn(3, 17, device="cuda")
    expected = model.compute_language_model_loss(labels, ordinary)
    actual = model.compute_language_model_loss(labels, owned, inplace_backward=True)
    assert torch.equal(owned, source), "Forward must preserve logits for MTP metrics"
    assert torch.equal(actual, expected)
    reference_grad = torch.autograd.grad((expected * scale).sum(), ordinary)[0]
    actual_grad = torch.autograd.grad((actual * scale).sum(), owned, retain_graph=True)[0]
    assert torch.equal(actual_grad, reference_grad)
    assert actual_grad.data_ptr() == owned.data_ptr()
    assert torch.equal(ordinary, source), "Public auto loss must preserve its input"
    with pytest.raises(RuntimeError, match="consumed|repeated backward"):
        torch.autograd.grad(actual.sum(), owned)


@pytest.mark.parametrize("implementation", ["fallback", "te"])
def test_private_loss_projection_gradients(implementation, monkeypatch):
    _select_path(implementation, monkeypatch)
    model = LanguageModule(_config())
    torch.manual_seed(2494)
    source = torch.randn(17, 3, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4097, 8, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(4097, (3, 17), device="cuda")
    labels[:, ::5] = -100
    gradients = []
    for private in (False, True):
        x, w = source.clone().requires_grad_(), weight.clone().requires_grad_()
        logits = torch.nn.functional.linear(x, w)
        loss = model.compute_language_model_loss(labels, logits, inplace_backward=private)
        gradients.append(torch.autograd.grad(loss.mean(), (x, w)))
    for expected, actual in zip(*gradients, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("observing", [False, True])
@pytest.mark.parametrize("supports_private", [False, True])
def test_mtp_private_ownership_keeps_argmax_metrics_and_callbacks(
    observing, supports_private, monkeypatch
):
    calls, metrics = [], {}
    monkeypatch.setattr(mtp_module, "is_observing_tensor", lambda name: observing)
    monkeypatch.setattr(mtp_module, "observe_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mtp_module.MTPLossLoggingHelper,
        "save_metrics_to_tracker",
        lambda loss, correct, total, *args, **kwargs: metrics.update(correct=correct, total=total),
    )
    config = _config()
    config.mtp_num_layers = 1
    config.mtp_loss_scaling_factor = 1.0
    predictions = torch.tensor([[1], [2], [0], [1]])

    def private_loss(labels, logits, return_argmax=False, *, inplace_backward=False):
        calls.append(inplace_backward)
        result = torch.ones_like(labels, dtype=logits.dtype)
        return (result, predictions) if return_argmax else result

    def legacy_loss(labels, logits, return_argmax=False):
        return private_loss(labels, logits, return_argmax)

    mtp_module.process_mtp_loss(
        hidden_states=torch.zeros(8, 1, 3),
        labels=torch.tensor([[0, 1, 2, 0]]),
        loss_mask=torch.ones(1, 4),
        output_layer=lambda hidden, **kwargs: (hidden, None),
        output_weight=None,
        is_training=True,
        compute_language_model_loss=private_loss if supports_private else legacy_loss,
        config=config,
        metric_avg_group=object(),
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    assert calls == [supports_private and not observing]
    assert metrics["correct"].item() == 3
    assert metrics["total"].item() == 3
