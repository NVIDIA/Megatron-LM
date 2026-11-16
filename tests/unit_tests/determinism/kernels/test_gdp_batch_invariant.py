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
    with pytest.raises(ValueError, match="slot mapping are inference-only"):
        causal_conv(x, w, bias, slots=torch.tensor([1, 0], device="cuda", dtype=torch.int32))
    actual = causal_conv(x, w, bias)
    with torch.no_grad():
        exact(actual, causal_conv(x, w, bias))
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


@pytest.mark.parametrize("block_size", [16, 32])
def test_prefill_writes_allocator_storage(block_size):
    """Parallel prefill must preserve the complete caller-owned cache and decode tail."""
    xs = inputs(2, 23, heads=4)
    expected, expected_cache = prefill(*xs, block_size=block_size)
    storage = torch.zeros(
        2, GDPCache.storage_size(4, 128, 64, block_size), device="cuda", dtype=torch.float32
    )
    cache = GDPCache.from_storage(storage, 4, 128, 64, block_size)
    actual, returned = prefill(*xs, cache=cache, block_size=block_size)
    assert returned is cache
    exact(actual, expected)
    for a, b in zip(cache.tensors(), expected_cache.tensors()):
        exact(a, b)
    one = tuple(x[:, :1].contiguous() for x in xs)
    exact(append(*one, cache), append(*one, expected_cache))


@pytest.mark.parametrize("with_bias", [False, True])
def test_packed_convolution_gradients(with_bias):
    """Tiled parameter gradients must respect packed boundaries and odd channel counts."""
    x = torch.randn(1, 517, 127, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(127, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = (
        torch.randn(127, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if with_bias
        else None
    )
    offsets = [0, 1, 9, 129, 256, 517]
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    variables = (x, weight, bias) if with_bias else (x, weight)
    assert_replays_bit_exact(
        lambda *args: causal_conv(*args, cu_seqlens=cu),
        variables,
        contention=True,
        what="packed canonical conv4 gradients",
    )
    actual = causal_conv(*variables, cu_seqlens=cu)
    grad_output = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, variables, grad_output)
    reference_variables = tuple(v.detach().double().requires_grad_() for v in variables)
    rx, rw = reference_variables[:2]
    outputs = []
    for start, end in zip(offsets, offsets[1:]):
        segment = rx[:, start:end]
        acc = (
            reference_variables[2][None, None, :].expand_as(segment)
            if with_bias
            else torch.zeros_like(segment)
        )
        for tap in range(4):
            shifted = torch.nn.functional.pad(segment, (0, 0, 3 - tap, 0))[:, : end - start]
            acc = acc + shifted * rw[:, tap]
        outputs.append(torch.nn.functional.silu(acc))
    reference = torch.cat(outputs, 1)
    expected = torch.autograd.grad(reference, reference_variables, grad_output.double())
    for value, target in zip(gradients, expected):
        torch.testing.assert_close(
            value.float(), target.to(value.dtype).float(), rtol=0.02, atol=0.02
        )


@pytest.mark.parametrize("updates,block_size", [(1, 32), (3, 16), (3, 32)])
def test_packed_prefix_snapshots_and_restore(updates, block_size):
    """Snapshots preserve partial blocks and remain valid when graph hit counts change."""
    from megatron.core.ssm.ops.gdp.batch_invariant import append_packed
    from tests.unit_tests.ssm.test_gdp_batch_invariant_recurrence import inputs as recurrence_inputs

    prefix = recurrence_inputs(2, 7, 2, 32, 32, updates)
    xs = recurrence_inputs(1, 84, 2, 32, 32, updates)
    future = recurrence_inputs(1, 1, 2, 32, 32, updates)
    cache = GDPCache.allocate(3, 2, 32, 32, block_size=block_size)
    slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    append(*(x[:1, :5].contiguous() for x in prefix), cache, slots=slots[:1])
    append(*(x[1:].contiguous() for x in prefix), cache, slots=slots[1:])
    original = cache.snapshot(torch.arange(3, device="cuda"))
    cu = torch.tensor([0, 35, 84], device="cuda", dtype=torch.int32)
    ends = [1, 13, 35, 36, 64, 84]
    positions = torch.tensor(ends, device="cuda", dtype=torch.int32)
    count = torch.zeros(1, device="cuda", dtype=torch.int32)
    storage = torch.full(
        (len(ends), GDPCache.storage_size(2, 32, 32, block_size)),
        9.0,
        device="cuda",
        dtype=torch.float32,
    )
    snapshots = GDPCache.from_storage(storage, 2, 32, 32, block_size)

    def forward():
        return append_packed(
            *xs,
            cache,
            cu,
            slots,
            snapshots=snapshots,
            snapshot_positions=positions,
            snapshot_count=count,
        )

    expected = forward().clone()
    cache.restore(torch.arange(3, device="cuda"), original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    cache.restore(torch.arange(3, device="cuda"), original)
    count.fill_(len(ends))
    graph.replay()
    exact(output, expected)
    for index, end in enumerate(ends):
        lane = 0 if end <= 35 else 1
        start, prefix_length = (0, 5) if lane == 0 else (35, 7)
        history = tuple(
            torch.cat([p[lane : lane + 1, :prefix_length], x[:, start:end]], 1).contiguous()
            for p, x in zip(prefix, xs)
        )
        _, reference = prefill(*history, block_size=block_size)
        for actual, target in zip(snapshots.tensors(), reference.tensors()):
            exact(actual[index : index + 1], target)
        restored = GDPCache.allocate(1, 2, 32, 32, block_size=block_size)
        restored.restore(
            torch.tensor([0], device="cuda"),
            snapshots.snapshot(torch.tensor([index], device="cuda")),
        )
        resumed = append(*future, restored)
        complete, _ = prefill(
            *(
                torch.cat([past, next_token], 1).contiguous()
                for past, next_token in zip(history, future)
            ),
            block_size=block_size,
        )
        exact(resumed, complete[:, -1:])
    cache.restore(torch.arange(3, device="cuda"), original)
    count.zero_()
    storage.fill_(9.0)
    graph.replay()
    exact(output, expected)
    exact(storage, torch.full_like(storage, 9.0))


def test_convolution_more_than_65535_rows():
    """The launch grid must support the token counts produced by batched long sequences."""
    x = torch.ones(2, 32769, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(17, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = causal_conv(x, weight)
    expected = torch.full_like(output, torch.nn.functional.silu(torch.tensor(4.0)).item())
    for token in range(3):
        expected[:, token].fill_(torch.nn.functional.silu(torch.tensor(float(token + 1))).item())
    exact(output, expected)
    dx, dw = torch.autograd.grad(output.sum(), (x, weight))
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
    exact(dx[:1], dx[1:])


def test_cache_offsets_beyond_int32():
    """Slot indexing must promote before multiplying a large cache row stride."""
    from tests.unit_tests.ssm.test_gdp_batch_invariant_recurrence import inputs as recurrence_inputs

    if torch.cuda.mem_get_info()[0] < 12 * 2**30:
        pytest.skip("Large-offset regression requires 12 GiB of free GPU memory")
    slot = 2**31 // (128 * 128)
    cache = GDPCache.allocate(slot + 1, 1, 128, 128)
    reference = GDPCache.allocate(1, 1, 128, 128)
    xs = recurrence_inputs(1, 7, 1, 128, 128, 3)
    indices = torch.tensor([slot], device="cuda", dtype=torch.int32)
    expected = append(*xs, reference)
    actual = append(*xs, cache, slots=indices)
    exact(actual, expected)
    for value, target in zip(cache.tensors(), reference.tensors()):
        exact(value[slot : slot + 1], target)
