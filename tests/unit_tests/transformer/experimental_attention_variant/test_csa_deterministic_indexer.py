# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical and replay coverage for deterministic CSA indexer gradients."""

import os

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
    # The torch flag selects index_add_ accumulation; without it the helper sorts key ids
    # and gathers prefix sums. Both orders use the same batch-offset key ids.
    # Tiny test heads otherwise allocate a large unused padding-id array under the 1 GiB budget.
    monkeypatch.setattr(dk, "_DETERMINISTIC_INDEXER_DK_CHUNK_MAX_BYTES", 4096)
    # The fp32 tolerance below assumes true fp32 GEMMs; an earlier test in the same process may
    # have enabled TF32, which cuBLAS honours for the small bmm/matmul shapes used here.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    prior_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
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
        torch.set_float32_matmul_precision(prior_precision)
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
    # Attention-sink and local-window mass leave the selected targets summing below one.
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
