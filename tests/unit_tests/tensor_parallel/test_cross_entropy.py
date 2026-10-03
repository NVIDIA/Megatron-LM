# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.models.common.language_module import language_module as language_module_module
from megatron.core.tensor_parallel import cross_entropy as cross_entropy_module
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils


class _FakeTPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


def test_vocab_parallel_cross_entropy_uses_explicit_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    all_reduce_groups = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_groups.append(group)
        return tensor

    def fail_parallel_state_call(*args, **kwargs):
        raise AssertionError("explicit tp_group should avoid parallel_state")

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        cross_entropy_module, "get_tensor_model_parallel_group", fail_parallel_state_call
    )

    vocab_parallel_logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    target = torch.tensor([2, 0])
    expected_output = torch.nn.functional.cross_entropy(
        vocab_parallel_logits.clone(), target, reduction="none"
    )

    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=tp_group)

    torch.testing.assert_close(output, expected_output)
    assert all_reduce_groups == [tp_group, tp_group, tp_group]


def test_language_module_loss_calls_the_selected_cross_entropy():
    """The loss uses the target the backend picked at construction, with no branch left here."""
    tp_group = _FakeTPGroup()
    captured = {}

    def fake_vocab_parallel_cross_entropy(logits, labels, tp_group=None):
        captured["logits"] = logits
        captured["labels"] = labels
        captured["tp_group"] = tp_group
        return torch.zeros_like(labels, dtype=logits.dtype)

    module = SimpleNamespace(
        config=SimpleNamespace(cross_entropy_loss_fusion=False),
        tp_group=tp_group,
        vocab_parallel_cross_entropy=fake_vocab_parallel_cross_entropy,
    )
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.randn(3, 2, 4)

    loss = language_module_module.LanguageModule.compute_language_model_loss(
        module, labels=labels, logits=logits
    )

    assert captured["logits"] is logits
    assert captured["tp_group"] is tp_group
    torch.testing.assert_close(captured["labels"], labels.transpose(0, 1).contiguous())
    assert loss.shape == labels.shape


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 2)
    vocab_parallel_logits = torch.range(0, 7).repeat(16, 4).cuda()
    target = torch.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    assert torch.equal(torch.round(expected_output), torch.round(output))
    Utils.destroy_model_parallel()


def test_vocab_parallel_label_smoothing_matches_dense_loss_and_gradient():
    Utils.initialize_model_parallel(2, 1)
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        for smoothing in (0.0, 0.1):
            for scale in (1.0, 1000.0):
                torch.manual_seed(7)
                logits = torch.randn(6, 8, device="cuda") * scale
                target = torch.tensor([0, 7, 2, 4, 1, 6], device="cuda")
                local = logits.chunk(2, dim=-1)[tp_rank].clone().requires_grad_()
                loss = vocab_parallel_cross_entropy(local, target, smoothing, tp_group)
                loss.mean().backward()

                dense = logits.double().requires_grad_()
                log_probs = dense.log_softmax(dim=-1)
                target_log_probs = log_probs.gather(-1, target[:, None]).squeeze(-1)
                expected = (
                    -(1 - smoothing) * target_log_probs
                    - smoothing * (log_probs.sum(dim=-1) - target_log_probs) / 7
                )
                expected.mean().backward()

                assert torch.isfinite(loss).all()
                torch.testing.assert_close(loss.double(), expected, atol=3e-4, rtol=2e-6)
                torch.testing.assert_close(
                    local.grad.double(), dense.grad.chunk(2, dim=-1)[tp_rank], atol=1e-7, rtol=2e-6
                )
    finally:
        Utils.destroy_model_parallel()
