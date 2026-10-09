# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay, partition, allocator, and gradient gates for canonical GDP."""

import os

import pytest
import torch

from megatron.core.ssm.ops.gdp.batch_invariant import GDPCache, append, prefill, train
from megatron.core.ssm.ops.gdp.batch_invariant_conv import causal_conv
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def device_and_seed():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    seeded()


def inputs(batch, tokens, heads=24):
    def normalized(shape):
        return torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()

    return (
        normalized((batch, tokens, heads, 128)),
        normalized((batch, tokens, 3, heads, 128)),
        torch.randn(batch, tokens, 3, heads, 64, device="cuda", dtype=torch.bfloat16),
        -torch.rand(batch, tokens, heads, device="cuda") * 0.1,
        torch.rand(batch, tokens, 3, heads, device="cuda"),
    )


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("block_size", [16, 32])
def test_recurrence_replays_and_batch_partitions(block_size):
    xs = inputs(8, 513)
    expected, expected_cache = prefill(*xs, block_size=block_size)
    actual, state = train(*xs, block_size=block_size)
    exact(actual, expected)
    exact(state, expected_cache.state)
    alone, alone_state = train(*(x[:1].contiguous() for x in xs), block_size=block_size)
    exact(alone, expected[:1])
    exact(alone_state, state[:1])
    differentiable = tuple(x.requires_grad_() for x in inputs(2, 129, heads=4))
    assert_replays_bit_exact(
        lambda *args: train(*args, block_size=block_size),
        differentiable,
        contention=True,
        what="canonical GDP forward and backward",
    )


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("batch,tokens,seed", [(8, 4096, 17), (1, 8193, 41)])
def test_long_uneven_prefill_chunks(block_size, batch, tokens, seed):
    # Rare rounding ties in factor sums and RHS arithmetic escaped shorter
    # tests. Exercise partial blocks followed by a long, multi-block append.
    torch.manual_seed(seed)
    xs = inputs(batch, tokens)
    initial = torch.randn(batch, 24, 128, 64, device="cuda") * 0.1
    expected, reference = prefill(*xs, block_size=block_size, initial_state=initial)
    cache = GDPCache.allocate(batch, 24, 128, 64, block_size=block_size)
    cache.state.copy_(initial)
    parts = []
    start = 0
    for width in (1, 5, 17, 7, 128, 29, 200, tokens - 387):
        parts.append(append(*(x[:, start : start + width].contiguous() for x in xs), cache))
        start += width
    exact(torch.cat(parts, 1), expected)
    for actual, wanted in zip(cache.tensors(), reference.tensors()):
        exact(actual, wanted)


@pytest.mark.parametrize("block_size", [16, 32])
def test_allocator_owned_cache(block_size):
    xs = inputs(2, 67, heads=4)
    size = GDPCache.storage_size(4, 128, 64, block_size)
    # Nonzero base offset represents a layer in the engine's state slab.
    slab = torch.zeros(3, 4, size, device="cuda", dtype=torch.float32)
    cache = GDPCache.from_storage(slab[1], 4, 128, 64, block_size)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    chunks = []
    start = 0
    for width in (1, 5, 7, 31, 23):
        chunks.append(
            append(*(x[:, start : start + width].contiguous() for x in xs), cache, slots=slots)
        )
        start += width
    expected, ref = prefill(*xs, block_size=block_size)
    exact(torch.cat(chunks, 1), expected)
    for a, b in zip(cache.tensors(), ref.tensors()):
        exact(a[slots], b)
    # The allocator can save/restore the raw slab without knowing GDP fields.
    saved = slab[1].clone()
    append(*(x[:, :1].contiguous() for x in xs), cache, slots=slots)
    slab[1].copy_(saved)
    for a, b in zip(cache.tensors(), ref.tensors()):
        exact(a[slots], b)
    assert not slab[0].count_nonzero() and not slab[2].count_nonzero()


def test_convolution_replay_and_gradients():
    x = torch.randn(2, 129, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert_replays_bit_exact(causal_conv, (x, w, bias), contention=True, what="canonical conv4")
    actual = causal_conv(x, w, bias)
    go = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, w, bias), go)
    ox, ow, ob = (z.detach().float().requires_grad_() for z in (x, w, bias))
    acc = ob[None, None, :].expand_as(ox)
    for tap in range(4):
        shifted = torch.nn.functional.pad(ox, (0, 0, 3 - tap, 0))[:, : x.shape[1]]
        acc = acc + shifted * ow[:, tap]
    reference = torch.nn.functional.silu(acc)
    expected = torch.autograd.grad(reference, (ox, ow, ob), go.float())
    for a, b in zip(grads, expected):
        torch.testing.assert_close(a.float(), b.to(a.dtype).float(), rtol=0.02, atol=0.02)
    history = torch.zeros(2, 128, 4, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        decoded = torch.cat(
            [
                causal_conv(
                    x[:, j : j + 1].contiguous(), w, bias, initial_state=history, update_state=True
                )
                for j in range(x.shape[1])
            ],
            1,
        )
    exact(actual, decoded)
