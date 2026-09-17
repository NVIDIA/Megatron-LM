# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Deterministic-mode canonicalisation of the compact indexer Top-K output."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    _canonicalize_compact_topk,
)


def _kernel_like_output(logits_full, topk, valid_counts, generator):
    """Emulate the cuDNN compact kernel: the selected ids in a random slot order, ``-1`` padding,
    and the softmax over the selected logits accumulated in that slot order."""
    rows = logits_full.shape[0]
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    softmax = torch.zeros((rows, topk), dtype=torch.float32)
    logits = torch.zeros((rows, topk), dtype=torch.float32)
    for r in range(rows):
        n = int(valid_counts[r])
        if n == 0:
            continue
        chosen = torch.topk(logits_full[r], n).indices
        perm = torch.randperm(n, generator=generator)
        chosen = chosen[perm]
        indices[r, :n] = chosen.int()
        logits[r, :n] = logits_full[r, chosen]
        softmax[r, :n] = torch.softmax(logits_full[r, chosen], dim=-1)
    return indices, logits, softmax


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_canonical_order_is_slot_order_invariant(dtype):
    torch.manual_seed(0)
    rows, n_keys, topk = 64, 512, 32
    logits_full = torch.randn(rows, n_keys)
    valid_counts = torch.randint(0, topk + 1, (rows,))
    valid_counts[0] = 0  # a fully padded row
    valid_counts[1] = topk

    outs = []
    for seed in (1, 2):
        g = torch.Generator().manual_seed(seed)
        idx, logits, softmax = _kernel_like_output(logits_full, topk, valid_counts, g)
        outs.append(_canonicalize_compact_topk(idx, logits, softmax.to(dtype)))

    (idx_a, sm_a), (idx_b, sm_b) = outs
    # Same selected set, different slot order in -> identical outputs out.
    assert torch.equal(idx_a, idx_b)
    assert torch.equal(sm_a, sm_b)
    # Ascending ids first, padding last, per row.
    for r in range(rows):
        n = int(valid_counts[r])
        assert torch.equal(idx_a[r, :n], torch.sort(idx_a[r, :n]).values)
        assert torch.all(idx_a[r, n:] == -1)
    # Softmax is the softmax over the selected logits, zero on padding and on padded rows.
    for r in range(rows):
        n = int(valid_counts[r])
        if n == 0:
            assert torch.all(sm_a[r] == 0)
            continue
        ref = torch.softmax(logits_full[r, idx_a[r, :n].long()], dim=-1).to(dtype)
        # The canonical softmax reduces over the padded width (exp(-inf) = 0), the reference over
        # the n valid entries only, so allow fp32 reduction-order rounding.
        torch.testing.assert_close(
            sm_a[r, :n], ref, rtol=0, atol=(1e-6 if dtype is torch.float32 else 1e-2)
        )
        assert torch.all(sm_a[r, n:] == 0)


def test_canonicalize_without_softmax():
    idx = torch.tensor([[5, -1, 2], [7, 3, 1]], dtype=torch.int32)
    logits = torch.zeros(2, 3)
    out_idx, out_sm = _canonicalize_compact_topk(idx, logits, None)
    assert torch.equal(out_idx, torch.tensor([[2, 5, -1], [1, 3, 7]], dtype=torch.int32))
    assert out_sm is None


def _reference_indexer_grads(q, w, k, attn, index, topk_idx, loss_coeff, sm_scale):
    """Loop reference for the sparse-indexer dK / dW (fp64)."""
    b, s, h, d = q.shape
    rows = b * s
    w_shape = w.shape
    q = q.reshape(rows, h, d).double()
    w = w.reshape(rows, h).double()
    kf = k.reshape(-1, d).double()
    attn = attn.reshape(rows, -1).double()
    index = index.reshape(rows, -1).double()
    idx = topk_idx.reshape(rows, -1)
    scale = (loss_coeff / rows) * sm_scale
    dk = torch.zeros_like(kf)
    dw = torch.zeros_like(w)
    for r in range(rows):
        for t in range(idx.shape[1]):
            key = int(idx[r, t])
            if key < 0:
                continue
            g = (index[r, t] - attn[r, t]) * scale
            dot = q[r] @ kf[key]  # (h,)
            relu = dot.clamp_min(0)
            dw[r] += relu * g
            coeff = w[r] * (dot > 0).double() * g  # (h,)
            dk[key] += coeff @ q[r]
    return dw.reshape(w_shape), dk.reshape(k.shape)


@pytest.mark.parametrize("use_torch_deterministic", [True, False])
def test_deterministic_indexer_dk_matches_reference(use_torch_deterministic):
    from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
        _deterministic_sparse_indexer_grads_wk,
    )

    torch.manual_seed(0)
    b, s, h, d, s_k, topk = 1, 48, 4, 16, 40, 8
    q = torch.randn(b, s, h, d)
    w = torch.rand(b, s, h)
    k = torch.randn(b, s_k, d)
    attn = torch.rand(b, s, topk)
    index = torch.rand(b, s, topk)
    topk_idx = torch.randint(0, s_k, (b, s, topk), dtype=torch.int32)
    topk_idx[0, 0, :] = -1  # a fully padded row
    topk_idx[0, 1, 3:] = -1  # partially padded row
    ref_dw, ref_dk = _reference_indexer_grads(q, w, k, attn, index, topk_idx, 0.7, 0.3)

    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(use_torch_deterministic)
    try:
        dw, dk = _deterministic_sparse_indexer_grads_wk(
            q, w, k, attn, index, topk_idx, loss_coeff=0.7, grad_loss=torch.ones(()),
            sm_scale=0.3, compute_grad_w=True,
        )
        # Run twice: bit-identical output regardless of the accumulation path.
        _, dk2 = _deterministic_sparse_indexer_grads_wk(
            q, w, k, attn, index, topk_idx, loss_coeff=0.7, grad_loss=torch.ones(()),
            sm_scale=0.3, compute_grad_w=False,
        )
    finally:
        torch.use_deterministic_algorithms(prev)
    torch.testing.assert_close(dk.double(), ref_dk, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dw.double(), ref_dw, rtol=1e-5, atol=1e-5)
    assert torch.equal(dk, dk2)


def test_needs_input_grad_of_activation_is_false_in_no_grad_pass():
    """The fused CSA function skips the deterministic indexer dK when its query activation does
    not need grad, which is how the activation-checkpoint no_grad pass presents itself inside
    ``Function.forward`` (parameters passed alongside still report requires_grad=True)."""
    seen = {}

    class _Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, param):
            seen["needs"] = tuple(ctx.needs_input_grad)
            return x * param

        @staticmethod
        def backward(ctx, g):
            return g, g

    param = torch.nn.Parameter(torch.ones(2))
    x = torch.ones(2, requires_grad=True) * 3  # an activation with grad
    _Probe.apply(x, param)
    assert seen["needs"] == (True, True)
    with torch.no_grad():
        y = x + 1  # an activation produced under no_grad, like a checkpointed forward
        _Probe.apply(y, param)
    assert seen["needs"][0] is False
    assert seen["needs"][1] is True  # a parameter input alone would defeat an any() check
