# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.models.common.language_module import language_module as language_module_module
from megatron.core.tensor_parallel import cross_entropy as cross_entropy_module
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
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


def test_language_module_loss_returns_argmax_of_the_cross_entropy():
    """With return_argmax, the loss passes on the argmax a cross entropy returns."""
    tp_group = _FakeTPGroup()
    logits = torch.randn(3, 2, 4)
    argmax = torch.argmax(logits, dim=-1)

    def fake_cross_entropy_with_argmax(logits, labels, tp_group=None, return_argmax=False):
        loss = torch.zeros_like(labels, dtype=logits.dtype)
        return (loss, argmax) if return_argmax else loss

    def fake_cross_entropy(logits, labels, tp_group=None):
        return torch.zeros_like(labels, dtype=logits.dtype)

    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    for cross_entropy, expected_argmax in (
        (fake_cross_entropy_with_argmax, argmax),
        (fake_cross_entropy, None),
    ):
        module = SimpleNamespace(tp_group=tp_group, vocab_parallel_cross_entropy=cross_entropy)
        loss, returned_argmax = language_module_module.LanguageModule.compute_language_model_loss(
            module, labels=labels, logits=logits, return_argmax=True
        )
        assert loss.shape == labels.shape
        assert returned_argmax is expected_argmax
        loss = language_module_module.LanguageModule.compute_language_model_loss(
            module, labels=labels, logits=logits
        )
        assert loss.shape == labels.shape


def test_language_module_loss_on_gathered_logits_matches_full_vocab_cross_entropy():
    """Gathered logits give the loss and gradients of the full vocabulary (#7452, finding 2).

    With `parallel_output=False` or `runtime_gather_output=True`, every TP rank holds all the
    logits. Passed to the vocab-parallel cross entropy as they are, the loss is log(tp_size) too
    high and the gradients lose the target term; this rank's shard of them gives the exact loss.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=4)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size, tp_rank = tp_group.size(), tp_group.rank()
    language_module = language_module_module.LanguageModule
    module = SimpleNamespace(
        tp_group=tp_group, vocab_parallel_cross_entropy=cross_entropy_module.unfused_cross_entropy
    )
    seq_length, batch_size, vocab_size = 5, 3, 16
    generator = torch.Generator().manual_seed(1234)
    logits = torch.randn(seq_length, batch_size, vocab_size, generator=generator).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), generator=generator).cuda()

    full_logits = logits.clone().requires_grad_()
    expected_loss = torch.nn.functional.cross_entropy(
        full_logits.reshape(-1, vocab_size), labels.transpose(0, 1).reshape(-1), reduction="none"
    )
    expected_loss = expected_loss.view(seq_length, batch_size).transpose(0, 1)
    expected_loss.sum().backward()

    shard = logits.chunk(tp_size, dim=-1)[tp_rank].clone().requires_grad_()
    assert language_module._vocab_parallel_logits(module, shard, False) is shard
    gathered = gather_from_tensor_model_parallel_region(shard, group=tp_group)
    loss = language_module.compute_language_model_loss(
        module, labels, language_module._vocab_parallel_logits(module, gathered, True)
    )
    loss.sum().backward()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(shard.grad, full_logits.grad.chunk(tp_size, dim=-1)[tp_rank])
    Utils.destroy_model_parallel()


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
