# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_masking import masked_softmax


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("reuse_logits", [False, True])
def test_masked_softmax_matches_selected_logits(device, dtype, dim, grad_enabled, reuse_logits):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(12)
    logits = torch.randn(2, 3, 17, device=device, dtype=dtype)
    valid = torch.rand_like(logits) > 0.4
    valid[0, 0] = False
    valid[0, 1] = False
    valid[0, 1, 2] = True
    if dim == 1:
        logits, valid = logits.transpose(1, 2), valid.transpose(1, 2)
    logits.requires_grad_()
    original = logits.detach().clone()
    with torch.set_grad_enabled(grad_enabled):
        input_logits = logits.clone() if reuse_logits else logits
        result = masked_softmax(input_logits, valid, dim=dim, reuse_logits=reuse_logits)

    # Independent reference: normalize only selected entries in each row.
    ref_logits = logits.detach().clone().requires_grad_()
    rows = ref_logits.movedim(dim, -1).reshape(-1, 17)
    masks = valid.movedim(dim, -1).reshape(-1, 17)
    ref_rows = []
    for row, mask in zip(rows, masks):
        ref_row = row * 0
        if mask.any():
            ref_row = ref_row.masked_scatter(mask, row[mask].softmax(-1))
        ref_rows.append(ref_row)
    reference = torch.stack(ref_rows).reshape(2, 3, 17).movedim(-1, dim)
    tol = {torch.float64: 1e-12, torch.float32: 2e-7, torch.float16: 2e-3, torch.bfloat16: 2e-2}[
        dtype
    ]
    torch.testing.assert_close(result, reference, rtol=tol, atol=tol)
    assert torch.equal(logits.detach(), original)
    assert torch.count_nonzero(result[~valid]) == 0
    if not grad_enabled:
        assert not result.requires_grad
        if reuse_logits:
            assert result.data_ptr() == input_logits.data_ptr()
        return
    upstream = torch.randn_like(result)
    grad = torch.autograd.grad(result, logits, upstream)[0]
    ref_grad = torch.autograd.grad(reference, ref_logits, upstream)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=tol, atol=tol)
    assert torch.count_nonzero(grad[~valid]) == 0


@pytest.mark.parametrize("reuse_logits", [False, True])
def test_masked_softmax_saves_only_one_probability_buffer(reuse_logits):
    logits = torch.randn(2, 3, 17, requires_grad=True)
    valid = (torch.rand(2, 1, 17) > 0.4).expand_as(logits)
    storages = set()

    def pack(tensor):
        if tensor.shape == logits.shape and tensor.dtype == logits.dtype:
            storages.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        input_logits = logits.clone() if reuse_logits else logits
        result = masked_softmax(input_logits, valid, reuse_logits=reuse_logits)
    # Exp and normalization can share the same saved probabilities. An out-of-place
    # mask between them would retain another full attention-score-sized allocation.
    assert len(storages) == 1
    if reuse_logits:
        assert input_logits.untyped_storage().data_ptr() in storages
    result.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("mask_shape", [(2, 1, 3, 17), (1, 1, 3, 17), (2, 4, 3, 1)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_masked_softmax_preserves_broadcast_mask_storage(mask_shape, transpose, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 3, 17, device=device)
    mask = torch.rand(mask_shape, device=device) > 0.3
    valid = mask.expand_as(logits)
    if transpose:
        logits, valid = logits.transpose(2, 3), valid.transpose(2, 3)
    dim = -2 if transpose else -1
    original_mask = mask.clone()
    logits.requires_grad_()
    ref_logits = logits.detach().clone().requires_grad_()
    saved_mask_bytes = []

    def pack(tensor):
        if tensor.dtype == torch.bool:
            saved_mask_bytes.append(tensor.untyped_storage().nbytes())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = masked_softmax(logits, valid, dim=dim)
    # Neither the shared validity mask nor the small row-validity mask should
    # become a full [batch, heads, query, key] allocation saved for backward.
    assert saved_mask_bytes
    assert max(saved_mask_bytes) <= max(mask.numel(), logits.numel() // logits.size(dim))
    reference = masked_softmax(ref_logits, valid.contiguous(), dim=dim)
    upstream = torch.randn_like(result)
    grad = torch.autograd.grad(result, logits, upstream)[0]
    ref_grad = torch.autograd.grad(reference, ref_logits, upstream)[0]
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
    assert torch.equal(mask, original_mask)


@pytest.mark.parametrize("eps", [1e-10, 2.0])
@pytest.mark.parametrize("reuse_logits", [False, True])
def test_masked_softmax_first_and_second_derivatives(eps, reuse_logits):
    torch.manual_seed(3)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor(
        [[True, False, True, True, False], [False] * 5, [False, True, False, False, False]]
    )
    fn = lambda x: masked_softmax(x.clone(), valid, eps=eps, reuse_logits=reuse_logits)
    assert torch.autograd.gradcheck(fn, (logits,))
    assert torch.autograd.gradgradcheck(fn, (logits,))


def test_masked_softmax_keeps_invalid_entries_zero_for_nonfinite_rows():
    logits = torch.tensor([[float("inf"), 1.0, 0.0], [float("nan"), 2.0, 3.0]])
    valid = torch.tensor([[True, True, False], [False, False, False]])
    result = masked_softmax(logits, valid)
    assert torch.count_nonzero(result[~valid]) == 0
