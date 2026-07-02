"""Tests for the Liger-Kernel vocab-parallel cross-entropy wrapper."""

import pytest
import torch

pytest.importorskip("liger_kernel.megatron")

from megatron.core.fusions.liger_cross_entropy import liger_vocab_parallel_cross_entropy
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_liger_ce_matches_torch_cross_entropy_tp1(dtype):
    """At TP=1, Liger CE must match torch.nn.functional.cross_entropy(reduction='none')."""
    Utils.initialize_model_parallel(1, 1)
    tp_group = get_tensor_model_parallel_group()

    seq_len, batch, vocab = 8, 4, 32
    logits = torch.randn(seq_len, batch, vocab, device="cuda", dtype=dtype).requires_grad_()
    target = torch.randint(0, vocab, (seq_len, batch), device="cuda", dtype=torch.long)

    loss = liger_vocab_parallel_cross_entropy(logits, target, tp_group)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab).float(), target.reshape(-1), reduction="none"
    ).reshape(seq_len, batch)

    tols = dict(rtol=1e-5, atol=1e-5) if dtype is torch.float32 else dict(rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(loss.float(), expected, **tols)

    # Backward parity: gradient w.r.t. logits must match the unfused reference.
    loss.sum().backward()
    grad_liger = logits.grad.detach().clone()

    logits2 = logits.detach().clone().requires_grad_()
    ref = torch.nn.functional.cross_entropy(
        logits2.reshape(-1, vocab).float(), target.reshape(-1), reduction="none"
    ).reshape(seq_len, batch)
    ref.sum().backward()

    torch.testing.assert_close(grad_liger.float(), logits2.grad.float(), **tols)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_liger_ce_matches_unfused_vocab_parallel_tp2(dtype):
    """At TP=2, Liger CE must match Megatron's unfused vocab_parallel_cross_entropy."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for TP=2 test")

    Utils.initialize_model_parallel(2, 1)
    tp_group = get_tensor_model_parallel_group()
    tp_size = tp_group.size()
    tp_rank = tp_group.rank()

    seq_len, batch, vocab_global = 8, 4, 32
    assert vocab_global % tp_size == 0
    vocab_local = vocab_global // tp_size

    # Build the same full-vocab logits on every rank, then slice to this rank's shard.
    torch.manual_seed(0)
    full_logits = torch.randn(seq_len, batch, vocab_global, device="cuda", dtype=dtype)
    local_logits_liger = (
        full_logits[..., tp_rank * vocab_local : (tp_rank + 1) * vocab_local]
        .detach()
        .clone()
        .requires_grad_()
    )
    local_logits_ref = (
        full_logits[..., tp_rank * vocab_local : (tp_rank + 1) * vocab_local]
        .detach()
        .clone()
        .requires_grad_()
    )
    target = torch.randint(0, vocab_global, (seq_len, batch), device="cuda", dtype=torch.long)

    liger_loss = liger_vocab_parallel_cross_entropy(local_logits_liger, target, tp_group)
    ref_loss = vocab_parallel_cross_entropy(local_logits_ref, target, tp_group=tp_group)

    tols = dict(rtol=1e-4, atol=1e-4) if dtype is torch.float32 else dict(rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(liger_loss.float(), ref_loss.float(), **tols)

    liger_loss.sum().backward()
    ref_loss.sum().backward()
    torch.testing.assert_close(
        local_logits_liger.grad.float(), local_logits_ref.grad.float(), **tols
    )
    Utils.destroy_model_parallel()


def test_liger_ce_missing_package_raises(monkeypatch):
    """When liger_kernel isn't importable, the wrapper must raise ImportError."""
    from megatron.core.fusions import liger_cross_entropy as module

    monkeypatch.setattr(module, "HAVE_LIGER", False)
    monkeypatch.setattr(module, "LigerMegatronCrossEntropy", None)

    logits = torch.empty(1, 1, 4)
    target = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ImportError, match="Liger-Kernel"):
        module.liger_vocab_parallel_cross_entropy(logits, target, tp_group=None)
