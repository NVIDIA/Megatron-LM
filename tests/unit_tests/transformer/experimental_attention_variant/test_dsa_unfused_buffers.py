# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.experimental_attention_variant.dsa import (
    _DSASoftmaxValue,
    _unfused_absorbed_dsa_fn,
)


@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
def test_unfused_absorbed_dsa_softmax_value_derivatives_and_reuse(needs_grad):
    torch.manual_seed(17)
    scores = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=needs_grad[0])
    value = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=needs_grad[1])
    invalid = torch.tensor([[[[True, True, True]], [[False, True, False]]]])

    def calculate(scores, value):
        private = scores.masked_fill(invalid, float("-inf"))
        private.masked_fill_(invalid.all(-1, keepdim=True), 0.0)
        output, probs = _DSASoftmaxValue.apply(private, value, invalid)
        assert probs.data_ptr() == private.data_ptr()
        return output, probs

    assert torch.autograd.gradcheck(calculate, (scores, value))
    assert torch.autograd.gradgradcheck(calculate, (scores, value))
    output, _ = calculate(scores, value)
    upstream = torch.randn_like(output[:1, :1]).expand_as(output)
    original_upstream = upstream.clone()
    inputs = tuple(x for x in (scores, value) if x.requires_grad)
    first = torch.autograd.grad(output, inputs, upstream, retain_graph=True)
    second = torch.autograd.grad(output, inputs, upstream)
    assert torch.equal(upstream, original_upstream)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unfused_absorbed_dsa_keeps_shared_kv_storage(dtype):
    query = torch.randn(7, 2, 3, 8, dtype=dtype, requires_grad=True)
    key = torch.randn(19, 2, 1, 8, dtype=dtype, requires_grad=True)
    topk = torch.arange(19).expand(2, 7, 19)
    saved_kv_bytes = []
    probability_storages = set()

    def pack(tensor):
        if tensor.ndim == 3 and sorted(tensor.shape[-2:]) == [8, 19]:
            saved_kv_bytes.append(tensor.untyped_storage().nbytes())
        if tensor.numel() == 2 * 7 * 3 * 19:
            probability_storages.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = _unfused_absorbed_dsa_fn(query, key, topk, 8**-0.5, 8)
    # FP32 score matmul may upcast K, but neither GEMM should save per-head K/V.
    assert saved_kv_bytes
    assert max(saved_kv_bytes) <= key.numel() * 4
    # The value GEMM must not keep a separate BF16 copy of the FP32 probabilities.
    assert len(probability_storages) == 1
    output.square().sum().backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()


def _selected_key_reference(
    query, key, topk, scale, v_channels, layout, mask, starts, ends, positions
):
    outputs = []
    for qi in range(query.size(0)):
        batches = []
        for bi in range(query.size(1)):
            indices = topk[bi, qi]
            indices = indices[indices >= 0].unique()
            if layout == "causal":
                # The default mask uses a top-left aligned causal triangle.
                indices = indices[indices <= qi]
            elif layout == "varlen":
                keep = (positions[indices] >= starts[qi]) & (positions[indices] < ends[qi])
                indices = indices[keep]
            else:
                indices = indices[torch.isfinite(mask[bi, qi, indices])]
            if indices.numel() == 0:
                output = query[qi, bi, :, :v_channels] * 0 + key[0, bi, :, :v_channels] * 0
            else:
                scores = query[qi, bi].float() @ key[indices, bi, 0].float().T * scale
                if mask is not None:
                    scores = scores + mask[bi, qi, indices]
                probs = scores.softmax(-1).to(key.dtype)
                output = probs @ key[indices, bi, 0, :v_channels]
            batches.append(output)
        outputs.append(torch.stack(batches))
    return torch.stack(outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["no_grad", "grad", "checkpoint", "autocast"])
@pytest.mark.parametrize("layout", ["causal", "varlen", "additive"])
def test_unfused_absorbed_dsa_reuses_only_private_buffers(dtype, mode, layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(31)
    query = torch.randn(7, 2, 3, 8, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn(19, 2, 1, 8, device="cuda", dtype=dtype, requires_grad=True)
    topk = torch.randint(0, 19, (2, 7, 6), device="cuda")
    topk[:, 0] = -1
    topk[:, 1, 1] = topk[:, 1, 0]
    starts = torch.tensor([0, 0, 0, 4, 4, 4, 4], device="cuda")
    ends = torch.tensor([0, 2, 5, 7, 10, 13, 19], device="cuda")
    positions = torch.randperm(19, device="cuda")
    mask = None
    kwargs = {}
    if layout == "varlen":
        kwargs = dict(varlen_starts=starts, varlen_ends=ends, key_positions=positions)
    elif layout == "additive":
        mask = torch.randn(2, 7, 19, device="cuda")
        mask[:, 0] = float("-inf")
        mask[:, 2, :3] = float("-inf")
        mask.requires_grad_()
        kwargs = {"mask": mask}
    inputs = [query, key] + ([mask] if mask is not None else [])
    snapshots = [tensor.detach().clone() for tensor in inputs]
    ref_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    def calculate(q, k, *bias):
        call_kwargs = {"mask": bias[0]} if bias else kwargs
        return _unfused_absorbed_dsa_fn(q, k, topk, 8**-0.5, 5, **call_kwargs)

    with (
        torch.set_grad_enabled(mode != "no_grad"),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode == "autocast"),
    ):
        output = (
            checkpoint(calculate, *inputs, use_reentrant=True)
            if mode == "checkpoint"
            else calculate(*inputs)
        )
    reference = _selected_key_reference(
        ref_inputs[0],
        ref_inputs[1],
        topk,
        8**-0.5,
        5,
        layout,
        ref_inputs[2] if mask is not None else None,
        starts,
        ends,
        positions,
    )
    tol = 2e-6 if dtype == torch.float32 and mode != "autocast" else 3e-2
    if mode == "autocast":
        assert output.dtype == torch.bfloat16
        reference = reference.to(output.dtype)
    torch.testing.assert_close(output, reference, rtol=tol, atol=tol)
    assert torch.count_nonzero(output[0]) == 0
    for tensor, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    if mode != "no_grad":
        upstream = torch.randn_like(output)
        output.backward(upstream)
        reference.backward(upstream)
        for tensor, ref_tensor in zip(inputs, ref_inputs):
            torch.testing.assert_close(tensor.grad, ref_tensor.grad, rtol=tol, atol=tol)
            assert torch.isfinite(tensor.grad).all()
