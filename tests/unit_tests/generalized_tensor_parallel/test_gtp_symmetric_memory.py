# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the GTP symmetric-memory (NVLS) path.

Test groups
-----------
- TestRegisteredLIFOPool    - LIFO recycling, keying, tagging, capture guard (single process)
- TestWgradSendBufferSplit  - get_wgrad_tensor / _prepare_wgrad_reduce_scatter_inputs routing:
                              symm scratch is a logical view of a registered padded LIFO
                              parent sent whole (zero-copy, padded or not) with ownership
                              transferred into the caller's RS-input list; the copy fallback
                              covers foreign wgrads
- TestSymmBackwardNumerics  - symm-flagged backward produces the same main_grad as plain GTP
- TestRealPoolRegistration  - register_gtp_symm_pool + gtp_symm_pool_ctx + a
                              collective on pool memory; skips where NCCL window
                              registration is unsupported

The buffer-routing tests monkeypatch ``is_gtp_symm_pool_registered`` in BOTH consuming
namespaces
(the gtp module's binding for routing decisions AND gtp_symmetric_memory's own for the LIFO's
branch), so LIFO allocations genuinely flow through ``gtp_symm_pool_ctx`` into the group's
ncclMemAlloc pool — just without the (collective, environment-dependent) window registration,
which TestRealPoolRegistration covers for real.

Multi-GPU tests skip when ``torch.distributed.get_world_size()`` != 4.
"""

import logging

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
import megatron.core.tensor_parallel.gtp_symmetric_memory as gtp_symm
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG, GTPShardedParam
from megatron.core.tensor_parallel.gtp_symmetric_memory import (
    RegisteredLIFOPool,
    deregister_and_clear_gtp_symm_pools,
    gtp_symm_pool_ctx,
    is_gtp_symm_pool_registered,
    register_gtp_symm_pool,
    symmetric_wgrad_pool,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


class _StubGroup:
    """Minimal stand-in for a dist process group (never used for real comms)."""

    def __init__(self, name="stub_gtp_symm_group", size=2, rank=0):
        self.group_name = name
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


_CONFIG_FIELDS = (
    "reduce_scatter_with_fp32_accumulation",
    "pad_for_alignment",
    "check_param_states",
    "calculate_per_token_loss",
)


@pytest.fixture(autouse=True)
def _restore_gtp_config():
    """Snapshot/restore the GTP_CONFIG fields these tests mutate."""
    saved = {f: getattr(GTP_CONFIG, f) for f in _CONFIG_FIELDS}
    yield
    for f, v in saved.items():
        setattr(GTP_CONFIG, f, v)


# ---------------------------------------------------------------------------
# RegisteredLIFOPool (single process)
# ---------------------------------------------------------------------------


class TestTeardownContract:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pool test")
    def test_teardown_clears_own_state_but_not_foreign_allocations(self):
        group = _StubGroup(name="teardown_contract_group")
        # A buffer owned by another subsystem (ring-slot style), allocated in the pool.
        with gtp_symm.gtp_symm_pool_ctx(group):
            foreign = torch.full((8,), 7.0, device="cuda")
        assert group.group_name in gtp_symm._pools
        # A LIFO-owned buffer sitting in the free list.
        buf = symmetric_wgrad_pool.alloc((4,), torch.bfloat16, "cuda", group)
        symmetric_wgrad_pool.free(buf)
        del buf

        deregister_and_clear_gtp_symm_pools()

        # Own state is gone: registries empty, free lists dropped.
        assert not gtp_symm._pools and not gtp_symm._registered
        assert not symmetric_wgrad_pool._free
        # Foreign allocations survive teardown untouched.
        torch.cuda.synchronize()
        assert torch.equal(foreign, torch.full((8,), 7.0, device="cuda"))


class TestRegisterVersionGuard:
    def test_register_rejects_old_torch(self, monkeypatch):
        monkeypatch.setattr(gtp_symm, "is_torch_min_version", lambda v: False)
        with pytest.raises(RuntimeError, match="PyTorch >= 2.9"):
            gtp_symm.register_gtp_symm_pool(_StubGroup(size=2))


class TestRegisteredLIFOPool:
    def test_alloc_returns_tagged_view(self):
        pool = RegisteredLIFOPool()
        group = _StubGroup()
        buf = pool.alloc((8, 4), torch.bfloat16, "cuda", group)
        assert tuple(buf.shape) == (8, 4)
        assert buf.dtype == torch.bfloat16
        assert getattr(buf, "_gtp_symm_group", None) is group

    def test_free_then_alloc_recycles_lifo(self):
        pool = RegisteredLIFOPool()
        group = _StubGroup()
        a = pool.alloc((8, 4), torch.bfloat16, "cuda", group)
        b = pool.alloc((8, 4), torch.bfloat16, "cuda", group)
        a_ptr, b_ptr = a.data_ptr(), b.data_ptr()
        assert a_ptr != b_ptr
        pool.free(a)
        pool.free(b)
        # LIFO: last freed comes back first; storage identity is preserved.
        assert pool.alloc((8, 4), torch.bfloat16, "cuda", group).data_ptr() == b_ptr
        assert pool.alloc((8, 4), torch.bfloat16, "cuda", group).data_ptr() == a_ptr
        # A 1-D key serves any shape with that numel.
        pool.free(a)
        c = pool.alloc((4, 8), torch.bfloat16, "cuda", group)
        assert c.data_ptr() == a_ptr and tuple(c.shape) == (4, 8)

    def test_keying_isolates_dtype_numel_group(self):
        pool = RegisteredLIFOPool()
        g1, g2 = _StubGroup("g1"), _StubGroup("g2")
        a = pool.alloc((8,), torch.bfloat16, "cuda", g1)
        pool.free(a)
        assert pool.alloc((8,), torch.float32, "cuda", g1).data_ptr() != a.data_ptr()
        assert pool.alloc((16,), torch.bfloat16, "cuda", g1).data_ptr() != a.data_ptr()
        assert pool.alloc((8,), torch.bfloat16, "cuda", g2).data_ptr() != a.data_ptr()
        assert pool.alloc((8,), torch.bfloat16, "cuda", g1).data_ptr() == a.data_ptr()

    def test_free_untagged_is_noop(self):
        pool = RegisteredLIFOPool()
        pool.free(torch.empty(8, device="cuda"))
        assert not pool._free  # nothing entered the free lists

    def test_reuse_waits_for_release_stream(self, monkeypatch):
        class FakeEvent:
            def __init__(self, *, external=False):
                self.external = external
                self.recorded_stream = None

            def record(self, stream=None):
                self.recorded_stream = stream

        class FakeStream:
            def __init__(self):
                self.waited_events = []

            def wait_event(self, event):
                self.waited_events.append(event)

        pool = RegisteredLIFOPool()
        group = _StubGroup()
        producer_stream = object()
        consumer_stream = FakeStream()
        event = FakeEvent(external=True)
        buffer = pool.alloc((8,), torch.bfloat16, "cuda", group)

        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: consumer_stream)

        event.record(producer_stream)
        pool.free(buffer, ready_event=event)
        reused = pool.alloc(buffer.shape, buffer.dtype, "cuda", group)

        assert reused.data_ptr() == buffer.data_ptr()
        assert event.external
        assert event.recorded_stream is producer_stream
        assert consumer_stream.waited_events == [event]

        pool.free(reused)
        assert pool.alloc(buffer.shape, buffer.dtype, "cuda", group).data_ptr() == buffer.data_ptr()
        assert consumer_stream.waited_events == [event]

    def test_capture_guard_raises_on_empty_bucket(self, monkeypatch):
        pool = RegisteredLIFOPool()
        group = _StubGroup()
        warm = pool.alloc((8,), torch.bfloat16, "cuda", group)
        pool.free(warm)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        # Pop from the warmed bucket is capture-safe...
        got = pool.alloc((8,), torch.bfloat16, "cuda", group)
        assert got.data_ptr() == warm.data_ptr()
        # ...but a fresh allocation during capture must fail loudly.
        with pytest.raises(RuntimeError, match="during CUDA-graph capture"):
            pool.alloc((8,), torch.bfloat16, "cuda", group)


# ---------------------------------------------------------------------------
# get_wgrad_tensor / _prepare_wgrad_reduce_scatter_inputs routing (world 4)
# ---------------------------------------------------------------------------


def _worker_sync_plain_recycle(rank, world_size, port):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))
    layer = _make_gtp_linear(64, 128, group, dtype)
    w = layer.weight
    w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")

    # Unregistered group: get_wgrad_tensor hands out plain pool scratch (the native
    # zero-copy path with the ub flags off).
    scratch = w.get_wgrad_tensor()
    assert getattr(scratch, "_from_gtp_wgrad_pool", False)
    ptr = scratch.data_ptr()
    scratch.normal_()

    # Plain unpadded branch: sent as-is, released as itself.
    send_bufs, release_bufs = w._prepare_wgrad_reduce_scatter_inputs([scratch])
    assert send_bufs[0] is scratch and release_bufs[0] is scratch

    # Plain padded branch: a padded copy is sent, the original is released.
    padded_layer = _make_gtp_linear(64, 100, group, dtype)
    wp = padded_layer.weight
    wp.main_grad = torch.zeros(wp.shape, dtype=dtype, device="cuda")
    p_scratch = wp.get_wgrad_tensor()
    send_bufs, release_bufs = wp._prepare_wgrad_reduce_scatter_inputs([p_scratch])
    assert send_bufs[0] is not p_scratch
    assert tuple(send_bufs[0].shape) == tuple(wp._unsharded_shape_padded)
    assert release_bufs[0] is p_scratch
    gtp_module._wgrad_pool_put(p_scratch)

    # Chain head -> synchronous reduce-scatter; the release path must return the
    # scratch to the plain pool (regression test for the sync-path dual release).
    w.wgrad_reduce_scatter(scratch)
    torch.cuda.synchronize()
    again = gtp_module._wgrad_pool_get(tuple(w._unsharded_shape), dtype, "cuda")
    assert again.data_ptr() == ptr, "sync RS did not recycle the plain wgrad scratch"
    gtp_module._wgrad_pool_put(again)


def _worker_wgrad_split(rank, world_size, port):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))
    # pad_for_alignment=16, world 4 -> alignment 64: out=128 -> pad 0; out=100 -> pad 28.
    aligned = _make_gtp_linear(64, 128, group, dtype)
    padded = _make_gtp_linear(64, 100, group, dtype)
    wa, wp = aligned.weight, padded.weight
    assert wa.pad_length == 0 and wp.pad_length > 0
    for w in (wa, wp):
        w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")

    saved_pred = gtp_module.is_gtp_symm_pool_registered
    saved_symm_pred = gtp_symm.is_gtp_symm_pool_registered
    # Patch BOTH consuming namespaces: gtp_module's binding drives the routing decisions;
    # gtp_symm's own drives RegisteredLIFOPool.alloc's pool-vs-plain branch, so LIFO
    # allocations genuinely land in the group's ncclMemAlloc pool.
    gtp_module.is_gtp_symm_pool_registered = lambda g: g is group
    gtp_symm.is_gtp_symm_pool_registered = lambda g: g is group
    try:
        # Unpadded symm weight: the GEMM scratch is a view of a registered LIFO parent.
        assert group.group_name not in gtp_symm._pools
        t = wa.get_wgrad_tensor()
        parent = wa._wgrad_symm_slot
        assert parent is not None
        assert getattr(parent, "_gtp_symm_group", None) is group
        assert t.data_ptr() == parent.data_ptr()
        assert tuple(t.shape) == tuple(wa._unsharded_shape)
        # Positive proof the allocation routed through gtp_symm_pool_ctx: the group's
        # pool was created by this alloc.
        assert group.group_name in gtp_symm._pools
        # _prepare sends the parent whole, consuming the slot — zero-copy (alias hit) —
        # and returns it in both lists (ownership transfer); the input list is untouched.
        bufs = [t]
        send_bufs, release_bufs = wa._prepare_wgrad_reduce_scatter_inputs(bufs)
        assert send_bufs[0] is parent
        assert release_bufs[0] is parent
        assert bufs[0] is t
        assert wa._wgrad_symm_slot is None

        # Release (the RS-input path) recycles it through the registered LIFO.
        ptr = parent.data_ptr()
        wa._wgrad_input_bufs = release_bufs
        wa._release_wgrad_scratch()
        assert wa._wgrad_input_bufs is None
        t2 = symmetric_wgrad_pool.alloc(
            tuple(wa._unsharded_shape_padded), dtype, parent.device, group
        )
        assert t2.data_ptr() == ptr
        symmetric_wgrad_pool.free(t2)

        # Padded symm weight: same mechanism — logical view of a padded parent whose
        # tail was zeroed at alloc; sent whole with no copy.
        g = wp.get_wgrad_tensor()
        pparent = wp._wgrad_symm_slot
        assert pparent is not None
        assert g.data_ptr() == pparent.data_ptr()
        assert tuple(g.shape) == tuple(wp._unsharded_shape)
        assert tuple(pparent.shape) == tuple(wp._unsharded_shape_padded)
        g.normal_()
        n = wp._unsharded_shape[0]
        assert torch.count_nonzero(pparent[n:]) == 0
        pbufs = [g]
        send_bufs, release_bufs = wp._prepare_wgrad_reduce_scatter_inputs(pbufs)
        assert send_bufs[0] is pparent
        assert release_bufs[0] is pparent
        assert pbufs[0] is g
        assert torch.equal(pparent[:n], g)
        assert torch.count_nonzero(pparent[n:]) == 0
        symmetric_wgrad_pool.free(pparent)

        # Foreign (untagged) wgrad with no recorded parent: the copy fallback covers it
        # (e.g. fuse_wgrad_accumulation off, where get_wgrad_tensor is never asked).
        assert wa._wgrad_symm_slot is None
        f = torch.randn(tuple(wa._unsharded_shape), dtype=dtype, device="cuda")
        fbufs = [f]
        send_bufs, release_bufs = wa._prepare_wgrad_reduce_scatter_inputs(fbufs)
        assert send_bufs[0] is not f
        assert release_bufs[0] is send_bufs[0]  # the allocated parent is what gets freed
        assert fbufs[0] is f
        assert getattr(send_bufs[0], "_gtp_symm_group", None) is group
        assert torch.equal(send_bufs[0][: wa._unsharded_shape[0]], f)
        symmetric_wgrad_pool.free(send_bufs[0])

        # Calling get_wgrad_tensor again before _prepare consumed the slot is an
        # invariant violation (recycled storage would alias two live views).
        t = wa.get_wgrad_tensor()
        with pytest.raises(RuntimeError, match="before the previous"):
            wa.get_wgrad_tensor()
        symmetric_wgrad_pool.free(wa._wgrad_symm_slot)
        wa._wgrad_symm_slot = None

        # A registered pool takes precedence over fp32-accum: the symm send still wins.
        GTP_CONFIG.reduce_scatter_with_fp32_accumulation = True
        t = wa.get_wgrad_tensor()
        assert wa._wgrad_symm_slot is not None
        symmetric_wgrad_pool.free(wa._wgrad_symm_slot)
        wa._wgrad_symm_slot = None
    finally:
        gtp_module.is_gtp_symm_pool_registered = saved_pred
        gtp_symm.is_gtp_symm_pool_registered = saved_symm_pred
        GTP_CONFIG.reduce_scatter_with_fp32_accumulation = False
        # Drop the LIFO buffers and the (unregistered) pools created via gtp_symm_pool_ctx.
        deregister_and_clear_gtp_symm_pools()


class TestSyncPlainScratchRecycle:
    def test_sync_rs_returns_plain_scratch_to_pool(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_sync_plain_recycle, 4)


class TestWgradSendBufferSplit:
    def test_send_buffer_routing(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_wgrad_split, 4)


# ---------------------------------------------------------------------------
# Backward numerics: symm buffers must not change the reduced gradient (world 4)
# ---------------------------------------------------------------------------


def _worker_symm_backward_numerics(rank, world_size, port):
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))
    inp = torch.randn(16, 64, dtype=dtype, device="cuda")
    gout = torch.randn(16, 128, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)
    dist.broadcast(gout, src=0)

    saved_pred = gtp_module.is_gtp_symm_pool_registered
    saved_symm_pred = gtp_symm.is_gtp_symm_pool_registered
    grads = {}
    try:
        for symm in (False, True):
            # Fresh chain state per phase so both layers are standalone chain heads
            # (sync RS) instead of linking into one chain with a dangling async RS.
            gtp_module.reset_gtp_state()
            torch.manual_seed(0)
            layer = _make_gtp_linear(64, 128, group, dtype)
            w = layer.weight
            w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")
            pred = (lambda g: g is group) if symm else saved_pred
            gtp_module.is_gtp_symm_pool_registered = pred
            gtp_symm.is_gtp_symm_pool_registered = pred if symm else saved_symm_pred
            x = inp.clone().requires_grad_(True)
            out = layer(x, is_first_microbatch=True)
            out.backward(gout)
            grads[symm] = w.main_grad.clone()
    finally:
        gtp_module.is_gtp_symm_pool_registered = saved_pred
        gtp_symm.is_gtp_symm_pool_registered = saved_symm_pred
        deregister_and_clear_gtp_symm_pools()

    # Same inputs, same weights, same collective — the send buffer's provenance must
    # not change the reduced gradient.
    torch.testing.assert_close(grads[True], grads[False], atol=0.0, rtol=0.0)


class TestSymmBackwardNumerics:
    def test_main_grad_matches_plain_gtp(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_symm_backward_numerics, 4)


# ---------------------------------------------------------------------------
# Real pool registration + collective on pool memory (world 4)
# ---------------------------------------------------------------------------


def _worker_real_pool_registration(rank, world_size, port):
    group = dist.new_group(list(range(world_size)))
    try:
        register_gtp_symm_pool(group)
    except Exception as e:  # NCCL window registration unsupported in this environment
        pytest.skip(f"NCCL symmetric registration unavailable: {e}")
    try:
        assert is_gtp_symm_pool_registered(group)
        # Idempotent re-register must not raise or re-issue the warmup.
        register_gtp_symm_pool(group)
        with gtp_symm_pool_ctx(group):
            src = torch.full((4,), float(rank), device="cuda")
            out = torch.empty(4 * world_size, device="cuda")
        dist.all_gather_into_tensor(out, src, group=group)
        expected = torch.repeat_interleave(
            torch.arange(world_size, device="cuda", dtype=torch.float32), 4
        )
        assert torch.equal(out, expected)
        # Pool tensors must be gone before the pools are deregistered and dropped
        # (mirrors production teardown, which clears the LIFO first): a pool torn
        # down with live allocations later frees blocks into a dead pool and a
        # subsequent NCCL op hits deregistered memory (intermittent IMA).
        del src, out
        torch.cuda.synchronize()
    finally:
        # Mandatory: leftover windows abort the ProcessGroupNCCL destructor at teardown.
        deregister_and_clear_gtp_symm_pools()
    assert not is_gtp_symm_pool_registered(group)


class TestRealPoolRegistration:
    def test_register_alloc_collective_deregister(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_real_pool_registration, 4)
