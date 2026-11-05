# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import fused_a2a
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def hybrid_ep_buffers(monkeypatch):
    """Exercise the real autograd wrappers with a buffer that checks handle ownership."""
    buffers = []

    class FakeHybridEPBuffer:
        def __init__(self, group, **kwargs):
            self.group = group
            self.owner = object()
            self.overflow = torch.tensor(0)
            buffers.append(weakref.ref(self))

        def dispatch_with_permute(
            self, hidden, probs=None, handle=None, fuse_permute_dispatch=False, **kwargs
        ):
            if handle is None:
                handle = (self.owner, self.overflow)
            else:
                assert handle[0] is self.owner, "dispatch used another buffer's handle"
            return (
                hidden * self.group.scale,
                probs * self.group.scale if probs is not None else None,
                None,
                torch.tensor([hidden.shape[0]], dtype=torch.int32),
                handle,
            )

        def combine_with_unpermute(self, hidden, handle, probs=None, **kwargs):
            assert handle[0] is self.owner, "combine used another buffer's handle"
            return (
                hidden * self.group.scale,
                probs * self.group.scale if probs is not None else None,
            )

    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", FakeHybridEPBuffer, raising=False)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    monkeypatch.setattr(fused_a2a, "HAVE_HYBRIDEP_DENSE_ROUTING", False)
    yield buffers
    fused_a2a.reset_hybrid_ep_buffer()


def _dispatch(group, fused=False):
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    probs = torch.ones(2, 1, requires_grad=True)
    result = fused_a2a.HybridEPDispatch.apply(
        hidden,
        torch.ones(2, 1, dtype=torch.bool),
        probs,
        group,
        1,
        None,
        None,
        None,
        None,
        fused,
        None,
        None,
        108,
        None,
        None,
    )
    return hidden, probs, result


def test_hybridep_cache_requires_the_same_group_object(hybrid_ep_buffers):
    group = SimpleNamespace(scale=2)
    replacement = SimpleNamespace(scale=2)
    assert group == replacement and group is not replacement

    first = _dispatch(group)
    first_buffer = fused_a2a._hybrid_ep_buffer
    second = _dispatch(group)
    assert len(hybrid_ep_buffers) == 1
    assert fused_a2a._hybrid_ep_buffer is first_buffer

    third = _dispatch(replacement)
    assert len(hybrid_ep_buffers) == 2
    third_buffer = fused_a2a._hybrid_ep_buffer
    assert third_buffer.group is replacement
    assert third_buffer is not first_buffer

    fused_a2a.reset_hybrid_ep_buffer()
    fourth = _dispatch(replacement)
    assert len(hybrid_ep_buffers) == 3
    assert fused_a2a._hybrid_ep_buffer is not third_buffer


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("reset_before_combine", [False, True])
def test_hybridep_interleaved_groups_keep_forward_and_backward_ownership(
    hybrid_ep_buffers, fused, reset_before_combine
):
    first = _dispatch(SimpleNamespace(scale=2), fused)
    second = _dispatch(SimpleNamespace(scale=3), fused)
    assert len(hybrid_ep_buffers) == 2
    if reset_before_combine:
        fused_a2a.reset_hybrid_ep_buffer()

    outputs = []
    for buffer_ref, (hidden, probs, result) in zip(hybrid_ep_buffers, (first, second)):
        dispatched, dispatched_probs, _, _, handle = result
        # The token dispatcher's capacity check must still see the backend overflow flag.
        assert isinstance(handle, tuple)
        assert handle[-1] is buffer_ref().overflow
        combined = fused_a2a.HybridEPCombine.apply(dispatched, handle, None, None, fused)
        torch.testing.assert_close(combined, hidden * buffer_ref().group.scale**2)
        outputs.append(combined.sum() + dispatched_probs.sum())

    # Both autograd nodes must use their originating buffer even after the cache is cleared.
    fused_a2a.reset_hybrid_ep_buffer()
    sum(outputs).backward()
    for scale, (hidden, probs, _) in zip((2, 3), (first, second)):
        torch.testing.assert_close(hidden.grad, torch.full_like(hidden, scale**2))
        torch.testing.assert_close(probs.grad, torch.full_like(probs, scale))


def test_hybridep_reset_releases_buffer_after_last_handle_and_graph(hybrid_ep_buffers):
    hidden, probs, result = _dispatch(SimpleNamespace(scale=2))
    buffer_ref = hybrid_ep_buffers[0]
    fused_a2a.reset_hybrid_ep_buffer()
    gc.collect()
    assert buffer_ref() is not None

    del result, hidden, probs
    gc.collect()
    assert buffer_ref() is None


def test_hybridep_dispatch_keeps_selected_buffer_if_cache_is_reset(hybrid_ep_buffers, monkeypatch):
    dispatch = fused_a2a.HybridEPBuffer.dispatch_with_permute

    def dispatch_and_reset(buffer, *args, **kwargs):
        result = dispatch(buffer, *args, **kwargs)
        fused_a2a.reset_hybrid_ep_buffer()
        return result

    monkeypatch.setattr(fused_a2a.HybridEPBuffer, "dispatch_with_permute", dispatch_and_reset)
    hidden, _, (dispatched, _, _, _, handle) = _dispatch(SimpleNamespace(scale=2))
    combined = fused_a2a.HybridEPCombine.apply(dispatched, handle, None, None, False)
    combined.sum().backward()
    torch.testing.assert_close(combined, hidden * 4)
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 4))


def _check_group_sequence(groups):
    """Keep graph/handle locals within a scope that ends before group teardown."""
    pending = []
    rank = torch.distributed.get_rank()
    for group in groups:
        hidden = torch.full((64, 128), rank + 1, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_()
        routing_map = torch.zeros(64, group.size(), device="cuda", dtype=torch.bool)
        routing_map[:, 0] = True
        dispatched, _, _, _, handle = fused_a2a.hybrid_ep_dispatch(
            hidden, routing_map, routing_map.float(), group, 1
        )
        assert fused_a2a._hybrid_ep_buffer.group is group
        if group.rank() == 0:
            expected = (
                torch.tensor(torch.distributed.get_process_group_ranks(group), device="cuda")
                .add(1)
                .repeat_interleave(64)
            )
            torch.testing.assert_close(
                dispatched[:, 0].float().sort().values, expected.float().sort().values
            )
        pending.append((hidden, dispatched, handle))

    # Interleave two real communicators, then clear the cache before either combine/backward.
    fused_a2a.reset_hybrid_ep_buffer()
    outputs = []
    for hidden, dispatched, handle in pending:
        combined = fused_a2a.hybrid_ep_combine(dispatched, handle, dispatched.shape[0], None)
        torch.testing.assert_close(combined, hidden, rtol=0, atol=0)
        outputs.append(combined)
    # HybridEP requires contiguous gradient inputs; sum() would supply a stride-zero view.
    torch.autograd.backward(outputs, [torch.ones_like(output) for output in outputs])
    for hidden, _, _ in pending:
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="requires CUDA and HybridEP",
)
def test_hybridep_different_rank_layouts_forward_backward():
    """Changing EP membership must dispatch to the new ranks and preserve earlier graphs."""
    Utils.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size < 8 or world_size % 4:
        pytest.skip("requires at least eight ranks and a world size divisible by four")

    created_groups = []
    local_groups = []
    rank = torch.distributed.get_rank()
    rank_sets = [list(range(start, start + 4)) for start in range(0, world_size, 4)]
    rank_sets += [
        list(range(start, world_size, world_size // 4)) for start in range(world_size // 4)
    ]
    fused_a2a.reset_hybrid_ep_buffer()
    try:
        for ranks in rank_sets:
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                created_groups.append(group)
                local_groups.append(group)
        _check_group_sequence(local_groups)
    finally:
        torch.cuda.synchronize()
        fused_a2a.reset_hybrid_ep_buffer()
        gc.collect()
        for group in reversed(created_groups):
            torch.distributed.destroy_process_group(group)
