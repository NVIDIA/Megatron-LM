# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Frozen head replay: full row coverage, native BI kernels, bounded saved state."""
import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.head import token_log_probs

pytestmark = pytest.mark.mlite


def operands(device='cpu', rows=23, hidden=32, vocab=96):
    generator = torch.Generator().manual_seed(7024)
    x = (torch.randint(-8, 8, (rows, hidden), generator=generator).float() / 16).bfloat16()
    w = (torch.randint(-8, 8, (vocab, hidden), generator=generator).float() / 32).bfloat16()
    labels = torch.arange(rows).remainder(vocab)
    return x.to(device).requires_grad_(), w.to(device), labels.to(device)


@pytest.mark.parametrize('rows', [0, 1, 23])
@pytest.mark.parametrize('entropy', [False, True])
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_cpu_row_coverage_bits_and_vjp(rows, entropy, dtype):
    # Dyadic inputs have exact FP32 dot products, avoiding CPU BLAS row-count
    # effects unrelated to the deployment BI contract (covered by the GPU gate).
    x, w, labels = operands(rows=rows)
    other = x.detach().clone().requires_grad_()
    ref, ref_ent = token_log_probs(
        x, w, labels, chunk_size=0, dtype=dtype, temperature=0.75, calculate_entropy=entropy
    )
    got, ent = token_log_probs(
        other, w, labels, chunk_size=7, dtype=dtype, temperature=0.75, calculate_entropy=entropy
    )
    assert torch.equal(got, ref)
    if entropy:
        assert torch.equal(ent, ref_ent)
    if rows:
        g = torch.linspace(-0.25, 1.25, rows)
        ref_loss, loss = (ref * g).sum(), (got * g).sum()
        if entropy:
            ref_loss, loss = ref_loss + ref_ent.sum() * 0.03, loss + ent.sum() * 0.03
        ref_loss.backward()
        loss.backward()
        assert torch.isfinite(other.grad).all()
        torch.testing.assert_close(other.grad, x.grad, atol=0.002, rtol=0.02)


def test_no_full_distribution_saved_and_entropy_only_vjp():
    x, w, labels = operands()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(tuple(t.shape)) or t, lambda t: t
    ):
        lp, entropy = token_log_probs(x, w, labels, chunk_size=7, calculate_entropy=True)
    assert saved == [tuple(x.shape), tuple(w.shape), tuple(labels.shape)]
    assert (x.shape[0], w.shape[0]) not in saved
    entropy.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.count_nonzero()


def test_trainable_head_keeps_weight_gradient_and_bad_contracts_fail():
    x, w, labels = operands()
    w.requires_grad_()
    lp, _ = token_log_probs(x, w, labels, chunk_size=7)
    lp.sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.count_nonzero()
    with pytest.raises(ValueError, match='one label'):
        token_log_probs(x, w, labels[:-1])
    for chunk in [-1, 1.5]:
        with pytest.raises(ValueError, match='nonnegative integer'):
            token_log_probs(x, w, labels, chunk_size=chunk)


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native BI requires CUDA')
def test_native_bi_chunked_forward_bits_and_finite_backward():
    # All row counts include an incomplete final block; compare raw FP32 bits.
    x, w, labels = operands('cuda', rows=35, hidden=128, vocab=256)
    other = x.detach().clone().requires_grad_()
    ref, ref_ent = token_log_probs(x, w, labels, chunk_size=0, calculate_entropy=True)
    got, ent = token_log_probs(other, w, labels, chunk_size=16, calculate_entropy=True)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    assert torch.equal(ent.view(torch.int32), ref_ent.view(torch.int32))
    (got.sum() + ent.sum() * 0.01).backward()
    assert torch.isfinite(other.grad).all() and other.grad.count_nonzero()


def test_native_head_hook_preserves_unlabelled_and_ce_outputs(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.model.deepseek_v41.lite import model as native

    logits = torch.tensor([[[1.0, 2.0, -1.0]], [[-0.5, 1.0, 2.0]]])
    owner = SimpleNamespace(
        head=lambda x: (logits.clone(), None), ps=SimpleNamespace(cp_rank=0, tp_group=None)
    )
    hidden = torch.zeros(2, 1, 3)
    full = native.DeepseekV41Model._head_output(owner, hidden, 2, None, None, 1.0, False)
    assert torch.equal(full['logits'], logits.squeeze(1))
    assert full['rows'].tolist() == [0, 1]
    labels = torch.tensor([1, 2])
    monkeypatch.setattr(
        native,
        'vocab_parallel_cross_entropy',
        lambda x, y, **kw: -x.log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1),
    )
    output = native.DeepseekV41Model._head_output(owner, hidden, 2, labels, None, 0.75, True)
    distribution = (logits.float() / 0.75).log_softmax(-1)
    assert torch.equal(
        output['log_probs'], distribution.gather(-1, labels[:, None, None]).flatten()
    )
    assert torch.equal(output['entropy'], -(distribution.exp() * distribution).sum(-1).flatten())
    assert torch.equal(output['loss'], -output['log_probs'].mean())
