# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SparseMLA shape adaptation, independent of any model-specific indexer."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops import tilelang_dsa


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [64, 65, 127])
@pytest.mark.parametrize("heads", [8, 16, 32])
def test_absorbed_shape_adapter_preserves_inputs_and_gradients(monkeypatch, dim, topk, heads):
    """Check padding, sentinels, scale and unpadding without requiring the optional kernel."""
    query = torch.randn(3, 2, heads, dim, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(5, 2, 1, dim, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.full((2, 3, topk), -1, dtype=torch.int32)
    indices[..., 0] = 0
    originals = [tensor.detach().clone() for tensor in (query, key, indices)]
    scale = 0.037

    def fake_sparse_mla(q, k, slots, softmax_scale):
        assert softmax_scale == scale
        assert q.shape == (2, 3, max(heads, 16), 576)
        assert k.shape == (2, 5, 1, 576)
        assert slots.shape == (2, 3, 1, ((topk + 63) // 64) * 64)
        torch.testing.assert_close(slots[:, :, 0, :topk], indices)
        assert (slots[..., topk:] == -1).all()
        if dim == 512:
            assert torch.count_nonzero(q[..., 512:]) == 0
            assert torch.count_nonzero(k[..., 512:]) == 0
        if heads < 16:
            assert torch.count_nonzero(q[:, :, heads:]) == 0
        return q[..., :512] + k[:, :1, :, :512], None

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", SimpleNamespace(apply=fake_sparse_mla))
    output = tilelang_dsa.fused_sparse_mla_absorbed(query, key, indices, scale, 512)
    assert output is not None
    assert output.shape == (3, 2, heads, 512)
    output.sum().backward()
    expected_q = torch.zeros_like(query)
    expected_q[..., :512] = 1
    expected_k = torch.zeros_like(key)
    expected_k[0, ..., :512] = 3 * heads
    torch.testing.assert_close(query.grad, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(key.grad, expected_k, atol=0, rtol=0)
    for tensor, original in zip((query, key, indices), originals):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)


def _reference(query, key, indices, scale):
    """Compute sparse absorbed attention directly in FP32 using the unpadded inputs."""
    # Each row selects its own keys; the first 512 channels are also the values.
    query = query.permute(1, 0, 2, 3).float()
    key = key[:, :, 0].permute(1, 0, 2).float()
    batch = torch.arange(key.size(0), device=key.device)[:, None, None]
    selected = key[batch, indices.clamp_min(0).long()]
    valid = indices >= 0
    logits = torch.einsum("bshd,bskd->bshk", query, selected) * scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # All-invalid rows must produce zero output and zero gradients.
    logits = torch.where(valid.any(-1)[:, :, None, None], logits, torch.zeros_like(logits))
    probs = logits.softmax(-1).masked_fill(~valid.unsqueeze(2), 0)
    return torch.einsum("bshk,bskd->bshd", probs, selected[..., :512]).permute(1, 0, 2, 3)


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [64, 65, 127])
@pytest.mark.parametrize("heads", [8, 16, 32])
def test_absorbed_shapes_match_reference_on_cuda(dim, topk, heads):
    """Compare real TileLang output and both input gradients to unpadded sparse attention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("tilelang")
    if tilelang_dsa.SparseMLA is None:
        pytest.skip("TileLang SparseMLA is unavailable")
    torch.manual_seed(17)
    length = 129
    query = torch.randn(length, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(length, 1, 1, dim, device="cuda", dtype=torch.bfloat16)
    query.requires_grad_()
    key.requires_grad_()
    rows = torch.arange(length, device="cuda")[:, None]
    indices = (rows - torch.arange(topk, device="cuda")[None, :]).clamp_min(-1)
    indices[0] = -1
    indices = indices.unsqueeze(0).to(torch.int32)
    scale = 0.037
    output = tilelang_dsa.fused_sparse_mla_absorbed(query, key, indices, scale, 512)
    assert output is not None, "supported shapes must reach the fused kernel"
    grad = torch.randn_like(output) * 0.01
    output.backward(grad)
    q_ref = query.detach().clone().requires_grad_()
    k_ref = key.detach().clone().requires_grad_()
    reference = _reference(q_ref, k_ref, indices, scale)
    reference.backward(grad.float())
    for actual, expected in ((output, reference), (query.grad, q_ref.grad), (key.grad, k_ref.grad)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.01)
        relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative_error < 0.02
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(query.grad[0]) == 0
