# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP eager-chain buffer policies (``GTP_CONFIG.eager_buffer_policy``: slab, transient, pooled).

TestTransientTicketLifecycle - the cache's consume-epoch rule on a stub parameter: a buffer is
                               allocated at acquire, survives its own consume epoch, and is freed
                               at the first acquire of a later epoch; reduce-scatter outputs are
                               freed at release; nothing is pooled for transient tickets.
TestGTPTransientBufferParity - a prefetch-chained stack (with and without recompute, BF16 and
                               native MXFP8) produces the same gradients with transient buffers
                               as with the pooled cache, pins nothing, and its first iteration
                               peaks no higher than its steady state.
TestGTPTransientScratchParity - transient wgrad / FP32-RS scratch matches the recycled pools,
                               with and without symmetric memory.
TestWeightSlabPlan           - the weight slab on stub tickets: two transient steps are recorded,
                               lifetimes that never overlap share bytes, each region knows its
                               predecessors, the slab serves the following steps along the recorded
                               acquire order, a ticket acquired twice per step gets two regions, a
                               cross-stream reader is remembered for the next writer, an acquire out
                               of order ends the fast path for the step (transient fallbacks, a
                               fence at the boundary) and two such steps re-plan, oversized tickets
                               and differing schedules stay transient, evaluation passes keep the
                               plan.
TestGTPWeightSlabParity      - the same stacks produce the same gradients with the slab (plan from
                               steps 2-3, slices from step 4) as with the pooled cache, with no
                               conflicts or misses.
"""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPWeightCache
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _requires_multi_gpu,
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


class _StubParam:
    """The attributes GTPWeightCache reads from a GTPShardedParam."""

    def __init__(self, rows, cols, chain_id="GTP_ungraphed"):
        self._unsharded_shape_padded = (rows, cols)
        self._sharded_padded_shape = (rows // 2, cols)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._debug_name = f"stub{rows}x{cols}"
        self.chain_id = chain_id

    def _get_cache_key(self, dtype, fwd, reduce_scatter, recompute=False):
        return (self._unsharded_shape_padded, dtype, fwd, reduce_scatter, recompute)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
class TestTransientTicketLifecycle:
    def setup_method(self):
        gtp_module.update_gtp_config(eager_buffer_policy="transient")
        self.cache = GTPWeightCache()
        self.producer = torch.cuda.Stream()

    def teardown_method(self):
        self.cache.clear()
        gtp_module.update_gtp_config(eager_buffer_policy="slab")
        torch.cuda.synchronize()

    def test_consume_epoch_rule_frees_at_the_next_epoch(self):
        cache, producer = self.cache, self.producer
        a = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        b = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        assert cache.is_transient(a)
        cache.begin_consume()
        buf_a = cache.acquire(a, producer)
        assert cache.get(a) is buf_a and cache.transient_bytes == 64 * 32 * 2
        cache.consume(a)  # the reader is enqueued after this call returns...
        cache.acquire(b, producer)  # ...so an acquire in the same epoch keeps ``a``
        assert cache._slots[a].buf is buf_a and cache.transient_bytes == 2 * 64 * 32 * 2
        cache.begin_consume()
        cache.consume(b)
        cache.acquire(a, producer)  # a later epoch: ``a``'s old buffer is freed, a new one bound
        assert cache._slots[a].buf is not buf_a
        assert cache._slots[b].buf is not None  # consumed in this epoch: still live
        assert cache.transient_bytes == 2 * 64 * 32 * 2
        # The earlier epoch's buffer is freed before the new one is allocated, so two buffers,
        # not three, were ever in flight.
        assert cache.peak_transient_bytes == 2 * 64 * 32 * 2
        # Nothing was pooled.
        assert not any(cache._pool.values()) and cache._total_bytes == 0

    def test_reduce_scatter_output_is_freed_at_release_and_pooled_tickets_still_pool(self):
        cache, producer = self.cache, self.producer
        rs = cache.reserve(_StubParam(64, 32), torch.float32, fwd=False, reduce_scatter=True)
        cache.acquire(rs, producer)
        assert cache.transient_bytes == 32 * 32 * 4
        cache.release(rs)  # main_grad.add_ enqueued on the allocating stream
        assert cache._slots[rs].buf is None and cache.transient_bytes == 0
        graphed = cache.reserve(
            _StubParam(64, 32, chain_id="GTP_graphed"), torch.bfloat16, fwd=True
        )
        assert not cache.is_transient(graphed)
        buf = cache.get(graphed)
        cache.release(graphed)
        assert cache.get(graphed) is buf  # pooled: the same buffer comes back
        assert cache._total_bytes == 64 * 32 * 2

    def test_gathered_serves_a_repeated_consume_within_the_step(self):
        cache, producer = self.cache, self.producer
        a = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        b = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        assert cache.gathered(a) is None  # nothing gathered yet
        cache.begin_consume()
        buf_a = cache.acquire(a, producer)
        assert cache.gathered(a) is None and cache.unconsumed_this_step(a)  # consumer pending
        cache.consume(a)
        assert not cache.unconsumed_this_step(a)
        assert cache.gathered(a) is buf_a  # a repeated consume: its own epoch, no new gather
        cache.acquire(b, producer)  # an acquire in that epoch keeps ``a`` for the new reader
        assert cache._slots[a].buf is buf_a
        cache.consume(b)
        cache.begin_consume()
        cache.acquire(cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True), producer)
        assert cache._slots[a].buf is None and cache._slots[b].buf is None
        # Across a step boundary the weights change: a buffer that survived is not served.
        cache.begin_consume()
        buf_b = cache.acquire(b, producer)
        cache.consume(b)
        cache.step_boundary()
        assert cache._slots[b].buf is buf_b and cache.gathered(b) is None

    def test_legacy_release_of_a_gathered_weight_is_the_consume_marker(self):
        cache, producer = self.cache, self.producer
        a = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        cache.begin_consume()
        cache.acquire(a, producer)
        cache.release(a)  # legacy coat-check release: marks consumed, does not free
        assert cache._slots[a].buf is not None and cache._slots[a].consumed
        cache.begin_consume()
        cache.acquire(cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True), producer)
        assert cache._slots[a].buf is None


def _in_slab(cache, tensor) -> bool:
    base = cache._slab.data_ptr()
    return base <= tensor.data_ptr() < base + cache.slab_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
class TestWeightSlabPlan:
    """Plan step 1: step 1 is recorded, the plan is built at its boundary and the slab serves
    step 2 on. Per step: ``a`` then ``b`` consumed in successive epochs (so b's acquire frees
    a), ``c`` prefetched while b is live and consumed last, a reduce-scatter output released
    after its add. c is still pending at the boundary; the recording step's boundary frees it
    before planning, later steps leave it to the next step's first acquire.
    """

    def setup_method(self):
        gtp_module.update_gtp_config(eager_buffer_policy="slab", weight_slab_plan_step=1)
        self.cache = GTPWeightCache()
        self.producer = torch.cuda.Stream()
        cache = self.cache
        self.a = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        self.b = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        self.c = cache.reserve(_StubParam(128, 32), torch.bfloat16, fwd=False)
        self.rs = cache.reserve(_StubParam(64, 32), torch.float32, fwd=False, reduce_scatter=True)

    def teardown_method(self):
        self.cache.clear()
        gtp_module.update_gtp_config(
            eager_buffer_policy="slab", weight_slab_plan_step=2, weight_slab_max_bytes=1 << 30
        )
        torch.cuda.synchronize()

    def _step(self, skip_c=False):
        cache, producer = self.cache, self.producer
        cache.begin_consume()
        cache.acquire(self.a, producer)
        cache.consume(self.a)
        cache.begin_consume()
        cache.acquire(self.b, producer)
        if not skip_c:
            cache.acquire(self.c, producer)
        cache.consume(self.b)
        cache.begin_consume()
        if not skip_c:
            cache.consume(self.c)
        cache.acquire(self.rs, producer)
        cache.release(self.rs)
        cache.step_boundary()

    def test_plan_packs_disjoint_lifetimes_into_shared_bytes(self):
        cache = self.cache
        self._step()  # the recorded step: the plan is built at its boundary
        plan = cache._plan
        assert plan is not None and cache._slab_disabled is None
        a, b, c, rs = (plan[(t, 0)] for t in (self.a, self.b, self.c, self.rs))
        assert (a.nbytes, b.nbytes, c.nbytes, rs.nbytes) == (4096, 4096, 8192, 4096)
        # Largest first: c (live until the step boundary) takes offset 0; a never overlaps c
        # in time and shares its bytes; b and the reduce-scatter output overlap c and go above
        # it, the output reusing b's offset since b is freed before it.
        assert c.offset == 0 and len(c.arc) == 1  # consumed last: freed at the step boundary
        assert a.offset == 0 and b.offset == 8192 and rs.offset == 8192
        assert cache.slab_bytes == 2 << 20  # rounded like a large allocation
        assert cache.slab_lower_bound == 8192 + 4096  # b and c live at once
        # Predecessors: who last used each byte before the acquire. a's bytes were c's; c's
        # bytes were a's (first half) and its own (second half); rs reuses b's offset.
        assert set(a.preds) == {c} and set(c.preds) == {a, c} and set(rs.preds) == {b}
        assert [t for t, _ in cache._order] == [self.a, self.b, self.c, self.rs]

    def test_slab_serves_the_steps_after_the_plan(self):
        cache, producer = self.cache, self.producer
        self._step()
        self._step()
        cache.begin_consume()
        buf_a = cache.acquire(self.a, producer)
        assert _in_slab(cache, buf_a) and cache.get(self.a) is buf_a
        assert cache._slots[self.a].region is cache._plan[(self.a, 0)]
        assert cache.transient_bytes == 0  # step 2's pending transient buffers were freed
        cache.consume(self.a)
        cache.begin_consume()
        buf_b = cache.acquire(self.b, producer)
        buf_c = cache.acquire(self.c, producer)
        assert _in_slab(cache, buf_b) and _in_slab(cache, buf_c)
        assert buf_a.data_ptr() == cache._slab.data_ptr()  # a shares c's bytes
        assert buf_c.data_ptr() == cache._slab.data_ptr()
        assert buf_b.data_ptr() == buf_a.data_ptr() + 8192
        cache.consume(self.b)
        cache.begin_consume()
        cache.consume(self.c)
        out = cache.acquire(self.rs, producer)
        assert _in_slab(cache, out) and out.data_ptr() == buf_b.data_ptr()
        cache.release(self.rs)
        cache.step_boundary()
        assert cache.slab_conflicts == 0 and cache.slab_misses == 0
        # b and c were live together; b was freed by the reduce-scatter's acquire (a later
        # epoch), c is still pending across the boundary (freed by the next step's first acquire).
        assert cache.peak_slab_live_bytes == 8192 + 4096 and cache.slab_live_bytes == 8192
        # The same views come back next step.
        cache.begin_consume()
        assert cache.acquire(self.a, producer) is buf_a

    def test_an_acquire_out_of_order_makes_the_rest_of_the_step_transient(self):
        cache, producer = self.cache, self.producer
        self._step()
        self._step()
        cache.begin_consume()
        cache.acquire(self.a, producer)  # first in the recorded order: its slice
        buf_c = cache.acquire(self.c, producer)  # b was expected: out of order
        assert not _in_slab(cache, buf_c) and cache.slab_conflicts == 1 and not cache._fast
        assert cache.transient_bytes == 8192
        buf_b = cache.acquire(self.b, producer)  # the rest of the step stays transient
        assert not _in_slab(cache, buf_b) and cache.slab_misses == 1
        d = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)
        assert not _in_slab(cache, cache.acquire(d, producer)) and cache.slab_misses == 2
        cache.consume(self.a)
        cache.consume(self.b)
        cache.consume(self.c)
        cache.consume(d)
        cache.begin_consume()
        out = cache.acquire(self.rs, producer)  # frees a (slab) and b, c, d (transient)
        assert not _in_slab(cache, out) and cache.transient_bytes == 4096
        assert cache.slab_live_bytes == 0
        cache.release(self.rs)
        cache.step_boundary()  # fences the slab's streams, keeps the plan
        assert cache._plan is not None and cache._fast and cache._deviating_steps == 1

    def test_oversized_tickets_stay_transient(self):
        cache, producer = self.cache, self.producer
        gtp_module.update_gtp_config(weight_slab_max_bytes=6000)  # c (8192 B) is too large
        self._step()
        assert cache._plan is not None and (self.c, 0) not in cache._plan
        assert dict(cache._order)[self.c] is None
        cache.begin_consume()
        assert _in_slab(cache, cache.acquire(self.a, producer))
        cache.consume(self.a)
        cache.begin_consume()
        cache.acquire(self.b, producer)
        assert not _in_slab(cache, cache.acquire(self.c, producer)) and cache.slab_misses == 0

    def test_ticket_acquired_twice_per_step_gets_two_regions(self):
        cache, producer = self.cache, self.producer
        e = cache.reserve(_StubParam(64, 32), torch.bfloat16, fwd=True)

        def step():
            for _ in range(2):  # the same ticket twice per step, in successive epochs
                cache.begin_consume()
                cache.acquire(e, producer)
                cache.consume(e)
            cache.step_boundary()

        step()
        step()
        first, second = cache._plan[(e, 0)], cache._plan[(e, 1)]
        assert first is not second and first.offset == second.offset == 0
        cache.begin_consume()
        assert cache.acquire(e, producer) is first.view
        cache.consume(e)
        cache.begin_consume()
        assert cache.acquire(e, producer) is second.view
        assert cache.slab_misses == 0 and cache.slab_conflicts == 0

    def test_cross_stream_reader_is_remembered_for_the_next_writer(self):
        cache, producer = self.cache, self.producer
        self._step()
        self._step()
        other = torch.cuda.Stream()
        cache.begin_consume()
        cache.acquire(self.a, producer)
        with torch.cuda.stream(other):
            cache.consume(self.a)  # the reader runs on another stream
        region_a, region_c = cache._plan[(self.a, 0)], cache._plan[(self.c, 0)]
        assert region_a.consumed and region_a.reader_stream == other
        assert region_a in region_c.preds and other in gtp_module._STREAM_BY_HANDLE.values()
        cache.begin_consume()
        cache.acquire(self.b, producer)  # a later epoch: frees a, whose slice b does not touch
        assert not region_a.live and region_a.reader_stream == other  # kept for c's acquire
        cache.acquire(self.c, producer)  # c's writer waits on `other` through region_a
        cache.consume(self.b)
        cache.begin_consume()
        cache.consume(self.c)
        cache.acquire(self.rs, producer)
        cache.release(self.rs)
        assert cache.slab_conflicts == 0 and cache._fast

    def test_reacquire_drops_an_unconsumed_use_but_never_a_consumed_one(self):
        cache, producer = self.cache, self.producer
        self._step()
        self._step()
        cache.begin_consume()
        first = cache.acquire(self.a, producer)
        # A second acquire in the same step (a skipped prefetch) drops the unconsumed first use;
        # it is also out of the recorded order, so the rest of the step runs transient.
        again = cache.acquire(self.a, producer)
        assert again is not first and not _in_slab(cache, again)
        assert cache.slab_conflicts == 1 and not cache._fast  # out of the recorded order
        assert cache.slab_live_bytes == 0 and cache.transient_bytes == 4096
        cache.consume(self.a)
        with pytest.raises(AssertionError, match="re-acquired inside the call"):
            cache.acquire(self.a, producer)

    def test_range_helpers(self):
        ranges = []
        GTPWeightCache._ranges_insert(ranges, 200, 300, "b")
        GTPWeightCache._ranges_insert(ranges, 0, 100, "a")
        assert [r[2] for r in ranges] == ["a", "b"]
        assert [r[2] for r in GTPWeightCache._ranges_in(ranges, 100, 200)] == []
        assert sorted(r[2] for r in GTPWeightCache._ranges_in(ranges, 50, 250)) == ["a", "b"]
        GTPWeightCache._ranges_clear(ranges, 50, 250)
        assert ranges == [(0, 50, "a"), (250, 300, "b")]
        GTPWeightCache._ranges_clear(ranges, 0, 1000)
        assert ranges == []

    def test_repeated_schedule_changes_record_again_then_give_up(self):
        cache, producer = self.cache, self.producer
        self._step()  # recorded; the plan is built at its boundary
        assert cache._plan is not None and cache._plan_attempts == 0
        for attempt in range(1, gtp_module._SLAB_PLAN_ATTEMPTS + 1):
            self._step(skip_c=True)
            self._step(skip_c=True)  # two steps off the plan: dropped, the next step is recorded
            assert cache._plan is None and cache._slab_disabled is None
            assert cache._record_from == cache._step and cache._plan_attempts == attempt
            self._step()
            assert cache._plan is not None
        # One schedule change too many: transient buffers for the rest of the run.
        self._step(skip_c=True)
        self._step(skip_c=True)
        assert cache._plan is None and "giving up" in cache._slab_disabled
        cache.begin_consume()
        buf = cache.acquire(self.a, producer)
        assert cache._slab is None and cache.transient_bytes == 4096 and buf is cache.get(self.a)
        assert "slab off" in cache.format_buffer_report()

    def test_schedule_change_fences_then_replans(self):
        cache, producer = self.cache, self.producer
        self._step()
        self._step()
        plan, slab = cache._plan, cache._slab
        assert plan is not None
        self._step()  # on plan
        assert cache.slab_conflicts == 0 and cache.slab_misses == 0 and cache._fast
        # One step off the recorded order (c skipped): the first acquire out of order ends the
        # fast path, the rest of the step is transient, the boundary fences the slab's streams
        # and keeps the plan.
        self._step(skip_c=True)
        assert cache.slab_conflicts == 1 and cache.slab_misses == 0  # rs where c was expected
        assert cache._plan is plan and cache._deviating_steps == 1 and cache._fast
        self._step()  # back on plan: the slab serves it, the deviation count resets
        assert cache._deviating_steps == 0 and cache._plan is plan
        cache.begin_consume()
        assert _in_slab(cache, cache.acquire(self.a, producer))
        cache.consume(self.a)
        cache.begin_consume()
        cache.acquire(self.b, producer)
        cache.acquire(self.c, producer)
        cache.consume(self.b)
        cache.begin_consume()
        cache.consume(self.c)
        cache.acquire(self.rs, producer)
        cache.release(self.rs)
        cache.step_boundary()
        # Two steps in a row off the plan drop it; the next two steps are recorded anew.
        self._step(skip_c=True)
        self._step(skip_c=True)
        assert cache._plan is None and cache._slab is None and cache.slab_bytes == 0
        assert cache._record_from == cache._step and cache._slab_disabled is None
        self._step()
        self._step()
        assert cache._plan is not None and cache._plan is not plan and cache._slab is not slab
        cache.begin_consume()
        assert _in_slab(cache, cache.acquire(self.a, producer))

    def test_forward_only_passes_use_transient_buffers_and_keep_the_plan(self):
        cache, producer = self.cache, self.producer
        self._step()
        self._step()
        misses = cache.slab_misses
        cache.set_forward_only(True)
        for _ in range(3):  # several evaluation iterations re-gather the same forward weights
            cache.begin_consume()
            buf = cache.acquire(self.a, producer)
            assert not _in_slab(cache, buf)
            cache.consume(self.a)
        cache.set_forward_only(False)
        assert cache.slab_misses == misses and cache.slab_conflicts == 0
        # The next training step maps onto the plan as if nothing happened.
        self._step()
        assert cache.slab_misses == misses and cache.slab_conflicts == 0
        assert cache._plan is not None and cache.slab_live_bytes == 8192


HIDDEN = 512
OTHER = 256
NUM_LAYERS = 4
BATCH = 32  # MXFP8 wgrad GEMM needs K = batch divisible by 32


def _build_stack(world_size, dtype, recipe):
    from tests.unit_tests.generalized_tensor_parallel.test_gtp_basics import (
        _make_native_fp8_gtp_linear,
    )

    gtp_remat_group = dist.new_group(list(range(world_size)))
    if recipe is None:

        def make(in_f, out_f):
            return _make_gtp_linear(
                in_f, out_f, gtp_remat_group, dtype, fuse_wgrad_accumulation=True
            )

    else:

        def make(in_f, out_f):
            return _make_native_fp8_gtp_linear(in_f, out_f, gtp_remat_group, dtype, recipe)

    recomputed = [make(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS)]
    spacers = [(make(HIDDEN, OTHER), make(OTHER, HIDDEN)) for _ in range(NUM_LAYERS)]
    layers = recomputed + [module for pair in spacers for module in pair]
    for layer in layers:
        layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    return recomputed, spacers, layers


def _step(recomputed, spacers, x, recompute, recipe):
    from transformer_engine.pytorch import fp8_autocast

    context = fp8_autocast(enabled=True, fp8_recipe=recipe) if recipe is not None else nullcontext()
    with context:
        h = x
        for layer, (down, up) in zip(recomputed, spacers):
            if recompute:
                checkpoint = CheckpointWithoutOutput()
                y = checkpoint.checkpoint(lambda inp, l=layer: l(inp), h)
                h = torch.nn.functional.gelu(y)
                checkpoint.discard_output_and_register_recompute(h)
            else:
                h = torch.nn.functional.gelu(layer(h))
            h = torch.nn.functional.gelu(down(h))
            h = torch.nn.functional.gelu(up(h))
    loss = h.float().sum()
    loss.backward()
    return x.grad.detach().clone()


def _run_variant(world_size, dtype, recipe, transient, recompute, iterations=4, slab=False):
    """Fresh chains and cache; returns per-iteration (dgrad, wgrads...), pooled bytes, the
    transient peak and the per-iteration allocator peaks. ``slab``: plan from steps 2-3."""
    gtp_module.reset_gtp_state()
    gtp_module._GTP_CACHE = None
    policy = "slab" if slab else ("transient" if transient else "pooled")
    gtp_module.update_gtp_config(eager_buffer_policy=policy, weight_slab_plan_step=2)
    try:
        torch.manual_seed(0)
        recomputed, spacers, layers = _build_stack(world_size, dtype, recipe)
        torch.manual_seed(1)
        x = torch.randn(BATCH, HIDDEN, dtype=dtype, device="cuda", requires_grad=True)
        dist.broadcast(x, src=0)
        results, peaks = [], []
        for _ in range(iterations):
            for layer in layers:
                layer.weight.main_grad.zero_()
            x.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            dgrad = _step(recomputed, spacers, x, recompute, recipe)
            gtp_module.get_global_GTP_cache().step_boundary()
            results.append([dgrad] + [layer.weight.main_grad.clone() for layer in layers])
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated())
        cache = gtp_module.get_global_GTP_cache()
        if slab:
            assert cache._slab is not None and cache._plan, cache._slab_disabled
            assert cache.slab_conflicts == 0 and cache.slab_misses == 0, (
                cache.slab_conflicts,
                cache.slab_misses,
            )
            assert 0 < cache.peak_slab_live_bytes <= cache.slab_bytes
            assert cache.slab_lower_bound <= cache.slab_bytes
        if transient:
            transient_slots = [slot for slot in cache._slots.values() if slot.transient]
            assert transient_slots, "no transient tickets"
            # Nothing is pooled for transient tickets, and only the buffers consumed in the last
            # epoch (plus a prefetch) may still be bound, never one per ticket.
            assert not any(cache._pool.get(slot.key) for slot in transient_slots), "pooled"
            bound = sum(1 for slot in transient_slots if slot.buf is not None)
            assert bound <= max(3, len(transient_slots) // 3), (bound, len(transient_slots))
        return results, cache._total_bytes, cache.peak_transient_bytes, peaks
    finally:
        gtp_module.update_gtp_config(eager_buffer_policy="slab")
        gtp_module.reset_gtp_state()
        gtp_module._GTP_CACHE = None


def _worker_slab_matches_pooled_cache(rank, world_size, port, recompute, use_mxfp8):
    dtype = torch.bfloat16
    recipe = None
    if use_mxfp8:
        from transformer_engine.common.recipe import MXFP8BlockScaling

        recipe = MXFP8BlockScaling()
    reference, _, _, _ = _run_variant(world_size, dtype, recipe, False, recompute, iterations=6)
    candidate, _, _, _ = _run_variant(
        world_size, dtype, recipe, True, recompute, iterations=6, slab=True
    )
    for iteration, (ref_values, values) in enumerate(zip(reference, candidate)):
        for index, (ref, value) in enumerate(zip(ref_values, values)):
            torch.testing.assert_close(
                value.float(),
                ref.float(),
                rtol=1e-3,
                atol=1e-3,
                msg=f"iteration {iteration}, tensor {index} differs with the weight slab",
            )


def _worker_transient_matches_pooled_cache(rank, world_size, port, recompute, use_mxfp8):
    dtype = torch.bfloat16
    recipe = None
    if use_mxfp8:
        from transformer_engine.common.recipe import MXFP8BlockScaling

        recipe = MXFP8BlockScaling()
    reference, pooled_bytes, _, _ = _run_variant(world_size, dtype, recipe, False, recompute)
    candidate, _, transient_peak, peaks = _run_variant(world_size, dtype, recipe, True, recompute)
    for iteration, (ref_values, values) in enumerate(zip(reference, candidate)):
        for index, (ref, value) in enumerate(zip(ref_values, values)):
            torch.testing.assert_close(
                value.float(),
                ref.float(),
                rtol=1e-3,
                atol=1e-3,
                msg=f"iteration {iteration}, tensor {index} differs with transient buffers",
            )
    assert pooled_bytes > 0 and transient_peak > 0
    # No warm-up, nothing pinned: the first iteration never exceeds the steady state beyond the
    # allocator's rounding slack.
    assert peaks[0] <= max(peaks[1:]) + (4 << 20), peaks


def _reset_scratch_state():
    gtp_module.reset_gtp_state()
    gtp_module._GTP_CACHE = None
    gtp_module._wgrad_buf_pool.clear()


def _run_scratch_variant(world_size, dtype, transient, fp32_rs, symm, iterations=5):
    """Fresh state; returns per-iteration main_grads and whether the recycled scratch pool
    stayed empty."""
    from megatron.core.tensor_parallel import gtp_symmetric_memory as gtp_symm

    _reset_scratch_state()
    gtp_module.update_gtp_config(
        eager_buffer_policy="transient" if transient else "pooled",
        reduce_scatter_with_fp32_accumulation=fp32_rs,
    )
    gtp_remat_group = dist.new_group(list(range(world_size)))
    if symm:
        gtp_symm.register_gtp_symm_pool(gtp_remat_group)
    try:
        torch.manual_seed(0)
        layers = [
            _make_gtp_linear(HIDDEN, HIDDEN, gtp_remat_group, dtype, fuse_wgrad_accumulation=True),
            _make_gtp_linear(HIDDEN, OTHER, gtp_remat_group, dtype, fuse_wgrad_accumulation=True),
            _make_gtp_linear(OTHER, HIDDEN, gtp_remat_group, dtype, fuse_wgrad_accumulation=True),
        ]
        for layer in layers:
            layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
        torch.manual_seed(1)
        x = torch.randn(BATCH, HIDDEN, dtype=dtype, device="cuda", requires_grad=True)
        dist.broadcast(x, src=0)
        results = []
        for _ in range(iterations):
            for layer in layers:
                layer.weight.main_grad.zero_()
            x.grad = None
            h = x
            for layer in layers:
                h = torch.nn.functional.gelu(layer(h))
            h.float().sum().backward()
            results.append([x.grad.clone()] + [layer.weight.main_grad.clone() for layer in layers])
        torch.cuda.synchronize()
        pool_empty = not any(gtp_module._wgrad_buf_pool.values())
        return results, pool_empty
    finally:
        del layers
        torch.cuda.synchronize()
        if symm:
            gtp_symm.deregister_and_clear_gtp_symm_pools()
        gtp_module.update_gtp_config(
            eager_buffer_policy="slab", reduce_scatter_with_fp32_accumulation=False
        )
        _reset_scratch_state()


def _worker_transient_scratch_matches_pools(rank, world_size, port, fp32_rs, symm):
    from megatron.core.tensor_parallel import gtp_symmetric_memory as gtp_symm

    dtype = torch.bfloat16
    if symm:
        group = dist.new_group(list(range(world_size)))
        try:
            gtp_symm.register_gtp_symm_pool(group)
        except Exception as exc:  # NCCL window registration unavailable here
            pytest.skip(f"NCCL symmetric registration unavailable: {exc}")
        gtp_symm.deregister_and_clear_gtp_symm_pools()
    reference, _ = _run_scratch_variant(world_size, dtype, False, fp32_rs, symm)
    candidate, pool_empty = _run_scratch_variant(world_size, dtype, True, fp32_rs, symm)
    for iteration, (ref_values, values) in enumerate(zip(reference, candidate)):
        for index, (ref, value) in enumerate(zip(ref_values, values)):
            torch.testing.assert_close(
                value.float(),
                ref.float(),
                rtol=1e-3,
                atol=1e-3,
                msg=f"iteration {iteration}, tensor {index} differs with transient scratch",
            )
    assert pool_empty, "transient scratch was recycled into the per-shape pool"


class TestGTPTransientScratchParity:
    @pytest.mark.parametrize("fp32_rs", [False, True], ids=["rs", "fp32rs"])
    def test_plain_scratch_matches_pools(self, fp32_rs):
        _requires_multi_gpu(4)
        _run_distributed(_worker_transient_scratch_matches_pools, 4, fp32_rs, False)

    @pytest.mark.parametrize("fp32_rs", [False, True], ids=["rs", "fp32rs"])
    def test_registered_send_path_matches_pools(self, fp32_rs):
        _requires_multi_gpu(4)
        _run_distributed(_worker_transient_scratch_matches_pools, 4, fp32_rs, True)


class TestGTPTransientBufferParity:
    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "recompute"])
    def test_bf16_matches_pooled_cache(self, recompute):
        _requires_multi_gpu(4)
        _run_distributed(_worker_transient_matches_pooled_cache, 4, recompute, False)

    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "recompute"])
    def test_mxfp8_matches_pooled_cache(self, recompute):
        _requires_mxfp8()
        _requires_multi_gpu(4)
        _run_distributed(_worker_transient_matches_pooled_cache, 4, recompute, True)


class TestGTPWeightSlabParity:
    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "recompute"])
    def test_bf16_slab_matches_pooled_cache(self, recompute):
        _requires_multi_gpu(4)
        _run_distributed(_worker_slab_matches_pooled_cache, 4, recompute, False)

    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "recompute"])
    def test_mxfp8_slab_matches_pooled_cache(self, recompute):
        _requires_mxfp8()
        _requires_multi_gpu(4)
        _run_distributed(_worker_slab_matches_pooled_cache, 4, recompute, True)
