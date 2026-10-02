# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical and replay coverage for deterministic CSA indexer gradients."""

import os
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as dk,
)


def _indexer_case(batch, dtype, device):
    generator = torch.Generator(device=device).manual_seed(193)
    q = torch.randn(batch, 4, 2, 3, generator=generator, device=device, dtype=dtype)
    w = torch.randn(batch, 4, 2, generator=generator, device=device, dtype=dtype)
    k = torch.randn(batch, 3, 3, generator=generator, device=device, dtype=dtype)
    indices = torch.tensor(
        [[0, 2, -1], [1, 2, -1], [-1, -1, -1], [0, 1, 2]], device=device, dtype=torch.int32
    ).expand(batch, -1, -1)
    # Row 1's target sums to 0.75: its score gradient is predict * 0.75 - target, not
    # predict - target.
    target = torch.tensor(
        [[0.25, 0.75, 0.0], [0.5, 0.25, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], device=device
    ).expand(batch, -1, -1)
    return q, w, k, indices, target


def _reference_indexer_grads(q, w, k, indices, target, loss_coeff, grad_loss, sm_scale):
    """Differentiate per-sample KL directly; no flattened key addressing or manual dK."""
    ref_k = k.detach().float().requires_grad_()
    ref_w = w.detach().float().requires_grad_()
    predict = torch.zeros_like(target)
    loss = torch.zeros((), device=q.device)
    for batch in range(q.shape[0]):
        for row in range(q.shape[1]):
            valid = indices[batch, row] >= 0
            if not valid.any():
                continue
            selected_keys = ref_k[batch, indices[batch, row, valid].long()]
            dot = q[batch, row].float() @ selected_keys.T
            score = (dot.relu() * ref_w[batch, row, :, None]).sum(0) * sm_scale
            log_predict = torch.log_softmax(score, dim=0)
            predict[batch, row, valid] = log_predict.detach().exp()
            loss = loss - (target[batch, row, valid] * log_predict).sum()
    loss = loss * (loss_coeff * grad_loss / (q.shape[0] * q.shape[1]))
    grad_w, grad_k = torch.autograd.grad(loss, (ref_w, ref_k))
    return predict, grad_w.to(w.dtype), grad_k.to(k.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_grad_w", [False, True])
@pytest.mark.parametrize("chunk_rows", [None, 3])
@pytest.mark.parametrize("torch_deterministic", [True, False])
def test_deterministic_indexer_grads_match_reference_and_replay(
    batch, dtype, compute_grad_w, chunk_rows, torch_deterministic, monkeypatch
):
    # B=1 is also the synthetic-batch contract used by packed THD and CP callers.
    # Three query rows per chunk forces the B=2 case to cross a sample boundary.
    # The torch flag selects index_add_ (on) or the stable sort + prefix sums (off).
    # Tiny test heads otherwise allocate a large unused padding-id array under the 1 GiB budget.
    monkeypatch.setattr(dk, "_DETERMINISTIC_INDEXER_DK_CHUNK_MAX_BYTES", 4096)
    if chunk_rows is not None:
        monkeypatch.setattr(dk, "_deterministic_indexer_dk_chunk_rows", lambda *_: chunk_rows)
    device = torch.device("cuda", torch.cuda.current_device())
    q, w, k, indices, target = _indexer_case(batch, dtype, device)
    loss_coeff, grad_loss, sm_scale = 0.7, 1.3, 0.5
    predict, expected_w, expected_k = _reference_indexer_grads(
        q, w, k, indices, target, loss_coeff, grad_loss, sm_scale
    )
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(torch_deterministic)
        results = [
            dk._deterministic_sparse_indexer_grads_wk(
                q.clone(),
                w.clone(),
                k.clone(),
                target.clone(),
                predict.clone(),
                indices.clone(),
                loss_coeff=loss_coeff,
                grad_loss=torch.tensor(grad_loss, device=device),
                sm_scale=sm_scale,
                compute_grad_w=compute_grad_w,
            )
            for _ in range(2)
        ]
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
    tolerance = (
        dict(rtol=2e-2, atol=2e-3) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=2e-6)
    )
    for grad_w, grad_k in results:
        torch.testing.assert_close(grad_k, expected_k, **tolerance)
        if compute_grad_w:
            torch.testing.assert_close(grad_w, expected_w, **tolerance)
        else:
            assert grad_w is None
    for first, second in zip(results[0], results[1]):
        if first is not None:
            assert torch.equal(
                first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_indexer_grads_match_cudnn_kernel():
    """The replacement must follow cuDNN's clipped-log KL, not only well-normalized targets."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("compares against the SM100 cuDNN sparse-indexer backward")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    dk._ensure_dsa_namespace()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(11)
    rows, keys, heads, dim, topk = 256, 256, 64, 128, 128
    q = torch.randn(1, rows, heads, dim, generator=generator, device=device, dtype=torch.bfloat16)
    w = torch.randn(1, rows, heads, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, keys, dim, generator=generator, device=device, dtype=torch.bfloat16)
    order = torch.rand(rows, keys, generator=generator, device=device).argsort(dim=-1)
    indices = order[:, :topk].int()
    indices[::7, topk // 2 :] = -1
    valid = indices >= 0
    logits = torch.randn(rows, topk, generator=generator, device=device).masked_fill(
        ~valid, float("-inf")
    )
    predict = torch.softmax(logits, dim=-1)
    # Probabilities below exp(-100), including exact zeros, drop out of cuDNN's KL gradient.
    predict[::5, 0] = 0.0
    predict[1::5, 1] = 1e-45
    target = torch.rand(rows, topk, generator=generator, device=device).masked_fill(~valid, 0.0)
    target = target / target.sum(dim=-1, keepdim=True)
    target[::3] *= 0.5
    grad_loss = torch.ones((), device=device)
    det_w, det_k = dk._deterministic_sparse_indexer_grads_wk(
        q,
        w,
        k,
        target.view(1, rows, topk),
        predict.view(1, rows, topk),
        indices.view(1, rows, topk),
        loss_coeff=0.3,
        grad_loss=grad_loss,
        sm_scale=0.1,
        compute_grad_w=True,
    )
    expected = dk._DSA.indexer_backward_wrapper(
        q,
        w,
        k,
        target.view(1, rows, topk).clone(),
        predict.view(1, rows, topk).clone(),
        indices.view(1, rows, topk),
        sm_scale=0.1,
        loss_coeff=0.3,
        grad_loss=grad_loss,
        block_I=128,
    )
    for actual, reference in ((det_k, expected["d_index_k"]), (det_w, expected["d_weights"])):
        error = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert error < 1e-2, f"relative L2 error {error.item():.3e}"


@pytest.mark.parametrize("global_mode", [False, True])
def test_stable_topk_ties_causal_padding_and_short_keys(monkeypatch, global_mode):
    # Force one row per slab and tied zero logits: selection must prefer smaller IDs.
    monkeypatch.setattr(dk, "_STABLE_TOPK_SORT_BYTES", 1)

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio):
            return {"scores": torch.zeros(2, 4, 3)}

        @staticmethod
        def indexer_top_k_wrapper(*args, **kwargs):
            raise AssertionError("Deterministic TopK must not use radix selection")

    monkeypatch.setattr(dk, "_DSA", FakeDSA)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(global_mode)
        actual, lengths, _ = dk._indexer_topk_bshd(
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 3, 2),
            torch.ones(2, 4, 1),
            topk=5,
            ratio=2,
            deterministic=True,
        )
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
    expected = torch.tensor(
        [[-1, -1, -1, -1, -1], [0, -1, -1, -1, -1], [0, -1, -1, -1, -1], [0, 1, -1, -1, -1]],
        dtype=torch.int32,
    ).expand(2, -1, -1)
    assert torch.equal(actual, expected)
    assert torch.equal(lengths, torch.tensor([[0, 1, 1, 2]], dtype=torch.int32).expand(2, -1))


def test_deterministic_backward_rejects_missing_cudnn_api(monkeypatch):
    monkeypatch.setattr(dk, "_DSA", SimpleNamespace(sparse_attention_backward_wrapper=lambda: None))
    q = torch.zeros(1, 16, 8)
    with pytest.raises(RuntimeError, match="no deterministic DSA.*>= 1.29"):
        dk._deterministic_sparse_bwd_kwargs(q, requested=True)


def test_deterministic_backward_requires_sm100_and_supported_heads(monkeypatch):
    def backward(q, kv, out, dout, lse, sink, indices, *, deterministic=False):
        raise AssertionError("the support checks must not launch the kernel")

    monkeypatch.setattr(dk, "_DSA", SimpleNamespace(sparse_attention_backward_wrapper=backward))
    with pytest.raises(RuntimeError, match="needs an SM100 GPU"):
        dk._deterministic_sparse_bwd_kwargs(torch.zeros(1, 16, 8), requested=True)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
        q = torch.zeros(1, 16, 8, device="cuda")
        assert dk._deterministic_sparse_bwd_kwargs(q, requested=True) == {"deterministic": True}
        with pytest.raises(RuntimeError, match="query-head count"):
            dk._deterministic_sparse_bwd_kwargs(q[:, :8], requested=True)


def test_central_sparse_backward_forwards_deterministic_flag(monkeypatch):
    seen = {}

    def deterministic_kwargs(q, requested):
        assert requested is True
        return {"deterministic": True}

    def backward(q, kv, out, dout, lse, sink, indices, **kwargs):
        seen.update(kwargs)
        return {"dq": dout.clone(), "dkv": torch.zeros_like(kv), "d_sink": torch.zeros_like(sink)}

    monkeypatch.setattr(dk, "_deterministic_sparse_bwd_kwargs", deterministic_kwargs)
    monkeypatch.setattr(dk, "_DSA", SimpleNamespace(sparse_attention_backward_wrapper=backward))
    q, kv, sink = torch.zeros(2, 16, 8), torch.zeros(3, 8), torch.zeros(16)
    dq, dkv, dsink = dk._csa_sparse_attention_backward(
        q,
        kv,
        q,
        torch.ones_like(q),
        torch.zeros(2, 16),
        sink,
        torch.zeros(2, 1, dtype=torch.int32),
        1.0,
        None,
        deterministic=True,
    )
    # cuDNN's wrapper allocates its scratch workspace per call when none is passed.
    assert seen["deterministic"] is True and "workspace" not in seen
    assert torch.equal(dq, torch.ones_like(q))
    assert dkv.shape == kv.shape and dsink.shape == sink.shape


def test_deterministic_dense_indexer_loss_fails_before_launch(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(dk, "_DSA", SimpleNamespace())
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="requires dsa_indexer_use_sparse_loss=True"):
        dk.fused_csa_indexer_sparse_attn(
            *([empty] * 7),
            indexer_topk=2,
            ratio=4,
            softmax_scale=1.0,
            loss_coeff=0.01,
            sparse_loss=False,
            deterministic=True,
        )


def test_torch_flag_alone_preserves_external_kernel_dispatch(monkeypatch):
    """The global torch flag must not opt H100 callers into the SM100-only path."""
    seen = []

    def radix_topk(scores, lengths, **kwargs):
        seen.append(kwargs)
        return {"indices": torch.zeros(scores.shape[0], 1, dtype=torch.int32)}

    monkeypatch.setattr(
        dk,
        "_DSA",
        SimpleNamespace(
            indexer_forward_wrapper=lambda *args, **kwargs: {"scores": torch.zeros(1, 4, 2)},
            indexer_top_k_wrapper=radix_topk,
        ),
    )
    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        q = torch.zeros(1, 4, 1, 2)
        dk._indexer_topk_bshd(q, torch.zeros(1, 2, 2), torch.ones(1, 4, 1), 1, 2)
        assert dk._deterministic_sparse_bwd_kwargs(q, requested=False) == {}
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)
    assert len(seen) == 1
