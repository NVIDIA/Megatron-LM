# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.experimental_attention_variant.dsa import _unfused_absorbed_dsa_fn


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
@pytest.mark.parametrize("mode", ["no_grad", "grad", "checkpoint"])
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

    with torch.set_grad_enabled(mode != "no_grad"):
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
    tol = 2e-6 if dtype == torch.float32 else 3e-2
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
