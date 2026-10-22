# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions import fused_padding_aware_cross_entropy as pace
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and pace.HAVE_TRITON),
    reason="padding-aware cross entropy needs a GPU and Triton",
)


def _setup(tp):
    Utils.initialize_model_parallel(tp, 1)
    return parallel_state.get_tensor_model_parallel_group()


def _reference(logits, labels, tp_group):
    ref_logits = logits.detach().clone().requires_grad_(True)
    loss = fused_vocab_parallel_cross_entropy(ref_logits, labels, tp_group)
    return ref_logits, loss


@pytest.mark.parametrize("vocab_per_rank", [1000, 4096, 30016])  # odd, pow2, > one block
@pytest.mark.parametrize("m_valid", [128, 640, 1024])
def test_matches_native_on_live_rows_and_zeros_the_tail(vocab_per_rank, m_valid):
    tp = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    tp_group = _setup(tp)
    rows = 1024
    full_vocab = vocab_per_rank * tp
    torch.manual_seed(1234)  # labels (and the loss gradient below) are identical on all TP ranks
    labels = torch.randint(0, full_vocab, (rows, 1), device="cuda", dtype=torch.int64)
    grad_loss = torch.rand(rows, 1, device="cuda")
    torch.manual_seed(100 + parallel_state.get_tensor_model_parallel_rank())  # own vocab shard
    logits = (torch.randn(rows, 1, vocab_per_rank, device="cuda") * 4).to(torch.bfloat16)
    # Padding rows carry label 0 and would be masked by the loss mask downstream.
    labels[m_valid:] = 0
    bound = torch.tensor([m_valid], device="cuda", dtype=torch.int64)

    pa_logits = logits.clone().requires_grad_(True)
    loss = pace.padding_aware_cross_entropy(pa_logits, labels, bound, tp_group)
    ref_logits, ref_loss = _reference(logits, labels, tp_group)

    assert loss.shape == ref_loss.shape == (rows, 1)
    torch.testing.assert_close(loss[:m_valid], ref_loss[:m_valid], rtol=1e-5, atol=1e-4)
    assert torch.all(loss[m_valid:] == 0)

    grad_loss[m_valid:] = 0  # what the loss mask does to padded rows
    loss.backward(grad_loss)
    ref_loss.backward(grad_loss)
    assert pa_logits.grad.dtype == torch.bfloat16
    torch.testing.assert_close(
        pa_logits.grad[:m_valid].float(), ref_logits.grad[:m_valid].float(), rtol=2e-2, atol=2e-4
    )
    assert torch.all(pa_logits.grad[m_valid:] == 0)
    Utils.destroy_model_parallel()


def test_cuda_graph_replay_follows_the_device_bound():
    tp_group = _setup(1)
    torch.manual_seed(0)
    rows, vocab = 512, 2048
    logits = (torch.randn(rows, 1, vocab, device="cuda") * 3).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (rows, 1), device="cuda", dtype=torch.int64)
    bound = torch.tensor([256], device="cuda", dtype=torch.int64)
    static_logits = logits.clone().requires_grad_(True)
    grad_loss = torch.ones(rows, 1, device="cuda")

    def step():
        loss = pace.padding_aware_cross_entropy(static_logits, labels, bound, tp_group)
        (grad,) = torch.autograd.grad(loss, static_logits, grad_loss)
        return loss, grad

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        loss_out, grad_out = step()
    for m in (256, 384, 128, 512):
        bound.fill_(m)
        g.replay()
        torch.cuda.synchronize()
        ref_logits, ref_loss = _reference(logits, labels, tp_group)
        torch.testing.assert_close(loss_out[:m], ref_loss[:m], rtol=1e-5, atol=1e-4)
        assert torch.all(loss_out[m:] == 0)
        assert torch.all(grad_out[m:] == 0)
        ref_loss.backward(grad_loss)
        torch.testing.assert_close(
            grad_out[:m].float(), ref_logits.grad[:m].float(), rtol=2e-2, atol=2e-4
        )
    Utils.destroy_model_parallel()


def test_triton_cross_entropy_without_a_bound_matches_native_on_every_row():
    tp_group = _setup(1)
    torch.manual_seed(0)
    rows, vocab = 1024, 4096
    logits = (torch.randn(rows, 1, vocab, device="cuda") * 4).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (rows, 1), device="cuda", dtype=torch.int64)
    a = logits.clone().requires_grad_()
    loss = pace.triton_cross_entropy(a, labels, tp_group)  # no live-row bound: all rows
    ref_logits, ref_loss = _reference(logits, labels, tp_group)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-4)
    g = torch.rand(rows, 1, device="cuda")
    loss.backward(g)
    ref_loss.backward(g)
    torch.testing.assert_close(a.grad.float(), ref_logits.grad.float(), rtol=2e-2, atol=2e-4)
    Utils.destroy_model_parallel()
