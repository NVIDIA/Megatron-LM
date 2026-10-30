# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic (torch) GDN on packed THD input.

``torch_chunk_gated_delta_rule`` with ``cu_seqlens`` and ``segmented_causal_conv1d`` must give
every document its own recurrence / convolution: the packed result equals running each document
separately, in outputs and gradients. Documents deliberately end inside chunks and span several
of them, and the pack has a padding tail.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net.gdn import (
    segmented_causal_conv1d,
    torch_chunk_gated_delta_rule,
)

LENGTHS = [50, 130, 7, 61]  # 248 tokens: boundaries inside chunks of 64, a 3-chunk document
PAD = 8  # padding tail, its own "document" in cu_seqlens
HEADS, DK, DV = 2, 16, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def full_fp32_matmuls():
    """The packed run chunks a document at different offsets than the per-document run, and
    the chunked recurrence is not associativity-exact: under TF32 that alone moves outputs by
    ~2e-4 (the same as re-chunking one document with chunk_size 16 instead of 64). Compare
    in full fp32 so the tolerance below tests the masking, not the rounding."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _cu_seqlens():
    cu = torch.tensor([0] + LENGTHS + [PAD]).cumsum(0).to(torch.int32)
    return cu.to(DEVICE)


def _inputs(total, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(DEVICE).requires_grad_()

    q = rand(1, total, HEADS, DK, scale=0.3)
    k = rand(1, total, HEADS, DK, scale=0.3)
    v = rand(1, total, HEADS, DV)
    g = (-torch.rand(1, total, HEADS, generator=gen) * 0.5).to(DEVICE).requires_grad_()
    beta = torch.rand(1, total, HEADS, generator=gen).to(DEVICE).requires_grad_()
    return q, k, v, g, beta


def _grads(tensors):
    return [t.grad.detach().clone() for t in tensors]


def test_packed_recurrence_equals_per_document_runs():
    cu = _cu_seqlens()
    total = int(cu[-1])
    inputs = _inputs(total)
    packed, _ = torch_chunk_gated_delta_rule(*inputs, cu_seqlens=cu)
    weight = torch.randn_like(packed)
    (packed * weight).sum().backward()
    packed_grads = _grads(inputs)

    reference_inputs = _inputs(total)
    outputs = []
    bounds = cu.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        doc_inputs = [t[:, start:end] for t in reference_inputs]
        out, _ = torch_chunk_gated_delta_rule(*doc_inputs)
        outputs.append(out)
    reference = torch.cat(outputs, dim=1)
    (reference * weight).sum().backward()

    torch.testing.assert_close(packed, reference, rtol=1e-4, atol=1e-5)
    for got, want in zip(packed_grads, _grads(reference_inputs)):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_packed_recurrence_is_bitwise_reproducible():
    cu = _cu_seqlens()
    total = int(cu[-1])
    runs = []
    for _ in range(2):
        inputs = _inputs(total)
        out, _ = torch_chunk_gated_delta_rule(*inputs, cu_seqlens=cu)
        out.sum().backward()
        runs.append([out.detach()] + _grads(inputs))
    for first, second in zip(*runs):
        assert torch.equal(first, second)


def test_segmented_conv_equals_per_document_conv():
    cu = _cu_seqlens()
    total = int(cu[-1])
    channels, width = 12, 4
    gen = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(1, channels, total, generator=gen).to(DEVICE).requires_grad_()
    weight = torch.randn(channels, 1, width, generator=gen).to(DEVICE).requires_grad_()
    bias = torch.randn(channels, generator=gen).to(DEVICE).requires_grad_()

    packed = segmented_causal_conv1d(x, weight, bias, cu)
    upstream = torch.randn_like(packed)
    (packed * upstream).sum().backward()
    packed_grads = _grads([x, weight, bias])

    x2, w2, b2 = (t.detach().clone().requires_grad_() for t in (x, weight, bias))
    bounds = cu.tolist()
    pieces = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        out = F.conv1d(x2[..., start:end], w2, b2, padding=width - 1, groups=channels)
        pieces.append(out[..., : end - start])
    reference = torch.cat(pieces, dim=-1)
    (reference * upstream).sum().backward()

    torch.testing.assert_close(packed, reference, rtol=1e-5, atol=1e-5)
    for got, want in zip(packed_grads, _grads([x2, w2, b2])):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_per_document_states_are_rejected():
    cu = _cu_seqlens()
    inputs = _inputs(int(cu[-1]))
    with pytest.raises(AssertionError, match="Per-document"):
        torch_chunk_gated_delta_rule(*inputs, cu_seqlens=cu, output_final_state=True)
