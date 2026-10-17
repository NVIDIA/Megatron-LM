# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_masking import masked_softmax


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_masked_softmax_matches_selected_logits(device, dtype, dim, grad_enabled):
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
        result = masked_softmax(logits, valid, dim=dim)

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
        return
    upstream = torch.randn_like(result)
    grad = torch.autograd.grad(result, logits, upstream)[0]
    ref_grad = torch.autograd.grad(reference, ref_logits, upstream)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=tol, atol=tol)
    assert torch.count_nonzero(grad[~valid]) == 0


def test_masked_softmax_saves_only_one_probability_buffer():
    logits = torch.randn(2, 3, 17, requires_grad=True)
    valid = (torch.rand(2, 1, 17) > 0.4).expand_as(logits)
    storages = set()

    def pack(tensor):
        if tensor.shape == logits.shape and tensor.dtype == logits.dtype:
            storages.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = masked_softmax(logits, valid)
    # Exp and normalization can share the same saved probabilities. An out-of-place
    # mask between them would retain another full attention-score-sized allocation.
    assert len(storages) == 1
    result.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("eps", [1e-10, 2.0])
def test_masked_softmax_first_and_second_derivatives(eps):
    torch.manual_seed(3)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor(
        [[True, False, True, True, False], [False] * 5, [False, True, False, False, False]]
    )
    fn = lambda x: masked_softmax(x, valid, eps=eps)
    assert torch.autograd.gradcheck(fn, (logits,))
    assert torch.autograd.gradgradcheck(fn, (logits,))


def test_masked_softmax_keeps_invalid_entries_zero_for_nonfinite_rows():
    logits = torch.tensor([[float("inf"), 1.0, 0.0], [float("nan"), 2.0, 3.0]])
    valid = torch.tensor([[True, True, False], [False, False, False]])
    result = masked_softmax(logits, valid)
    assert torch.count_nonzero(result[~valid]) == 0
