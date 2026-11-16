# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import copy
from unittest import mock

import pytest
import torch

from megatron.core.inference import unified_memory as unified_memory_module
from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.contexts import dynamic_context as dynamic_context_module
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from megatron.core.transformer.cuda_graphs import _CudaGraphRunner
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model, until
from tests.unit_tests.inference.engines import test_dynamic_engine as engine_tests
from tests.unit_tests.inference.test_coordinator_features_pairwise import _exercise_all_owners

pytestmark = [pytest.mark.internal, pytest.mark.asyncio]


async def _pause_active(h, monkeypatch, prompt, params):
    arrived, release = asyncio.Event(), asyncio.Event()
    step = h.engine.async_step

    async def gated_step():
        if arrived.is_set():
            return None
        result = await step()
        if h.rank == 0 and not arrived.is_set() and 0 in h.engine.requests:
            if len(h.engine.get_request(0).generated_tokens) >= 2:
                arrived.set()
                await asyncio.wait_for(release.wait(), timeout=30)
        return result

    monkeypatch.setattr(h.engine, "async_step", gated_step)
    future = None
    if h.rank == 0:
        future = h.clients[0].add_request(prompt, copy.deepcopy(params))
        await asyncio.wait_for(arrived.wait(), timeout=30)
        h.clients[0].pause_engines()
        await until(lambda: any(e["header"] == Headers.PAUSE for e in h.service.events))
        release.set()
    await asyncio.wait_for(h.engine.wait_until(EngineState.PAUSED), timeout=60)
    monkeypatch.setattr(h.engine, "async_step", step)
    await h.barrier()
    if h.rank == 0:
        assert 0 < len(h.engine.get_request(0).generated_tokens) < params.num_tokens_to_generate
        assert h.witnesses and not future.done()
    return future


@pytest.mark.parametrize("mode", list(AsyncScheduleMode))
@pytest.mark.parametrize("residency", list(KVCacheManagementMode))
async def test_routed_suspend_resumes_active_request(monkeypatch, mode, residency):
    """Supported residency transitions preserve a partly generated routed result."""
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    offload = residency == KVCacheManagementMode.OFFLOAD
    if offload:
        saver = dynamic_context_module.torch_memory_saver
        for name in ("region", "pause", "resume"):
            monkeypatch.setattr(saver, name, mock.Mock(wraps=getattr(saver, name)))
    async with routed_model(
        monkeypatch,
        async_sched_mode=mode,
        kv_cache_management_mode=residency.value,
        static_kv_memory_pointers=offload,
        num_cuda_graphs=1 if offload else None,
        force_build_cuda_graphs=offload,
        use_cuda_graphs_for_non_decode_steps=not offload,
    ) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        context, replays = h.engine.context, []
        if offload:
            replay = _CudaGraphRunner.replay_graph_capture

            def observed_replay(runner, *args, **kwargs):
                result = replay(runner, *args, **kwargs)
                active = slice(context.paused_request_count, context.total_request_count)
                ids = context.request_ids[active]
                live = 0 in h.engine.requests and not context._bookkeeping_no_real_work
                replays.extend([True] if live and 0 in ids else [])
                return result

            monkeypatch.setattr(_CudaGraphRunner, "replay_graph_capture", observed_replay)
        future = await _pause_active(h, monkeypatch, prompt, params)
        pointer = context.memory_buffer.data_ptr()
        if h.rank == 0:
            route = dict(h.service.coordinator.request_id_to_rank)
            assert route[0] == f"mp-coord-{h.rank}".encode() and 0 in h.engine.requests
            row = context.request_ids.tolist().index(0)
            blocks = context.request_to_kv_block_ids[row]
            blocks = blocks[: context.request_kv_block_counts[row]].long()
            target_kv = context.memory_buffer[:, :, blocks].view(torch.int16).clone()
            before = list(h.engine.get_request(0).generated_tokens)
            h.clients[0].suspend_engines()
        await asyncio.wait_for(h.engine.wait_until(EngineState.SUSPENDED), timeout=60)
        assert not context.is_tensor_state_allocated
        if residency == KVCacheManagementMode.PERSIST:
            assert context.memory_buffer.data_ptr() == pointer
        elif offload:
            saver.region.assert_called_once_with(tag=context.TMS_TAG, enable_cpu_backup=True)
            saver.pause.assert_called_once_with(context.TMS_TAG)
        else:
            assert not hasattr(context, "memory_buffer")
            assert not h.engine.controller._async_sched_forward.is_valid
        await h.barrier()
        if h.rank == 0:
            assert h.service.coordinator.request_id_to_rank == route and not future.done()
            h.clients[0].resume_engines()
        await asyncio.wait_for(h.engine.wait_until(EngineState.RESUMED), timeout=60)
        assert context.is_tensor_state_allocated and h.engine.state == EngineState.PAUSED
        assert h.engine._weight_epoch == 1
        if offload:
            assert context.memory_buffer.data_ptr() == pointer
            saver.resume.assert_called_once_with(context.TMS_TAG)
            if h.rank == 0:
                assert context.memory_buffer[:, :, blocks].view(torch.int16).equal(target_kv)
            replays[:] = h.witnesses[:] = []
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(future, timeout=60)
            assert final["request_id"] == 0 and final["generated_tokens"][: len(before)] == before
            assert final["generated_tokens"] == direct["generated_tokens"]
            assert final["status"] == direct["status"]
            if residency == KVCacheManagementMode.RECOMPUTE:
                assert sum(bool(s["prefill"]) for s in h.witnesses) >= 2
        if offload:
            assert await h.sync.all_reduce_max(int(bool(replays))) == 1
        await h.barrier()
        h.assert_retired()


@pytest.mark.parametrize("active", [False, True], ids=["queued", "active"])
async def test_routed_abort_retires_only_cancelled_request(monkeypatch, active):
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(monkeypatch) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        if active:
            cancelled = await _pause_active(h, monkeypatch, prompt, params)
        else:
            await h.pause()
            if h.rank == 0:
                cancelled = h.clients[0].add_request(prompt, copy.deepcopy(params))
                await until(lambda: 0 in h.engine.requests)
                assert not h.witnesses
        if h.rank == 0:
            survivor_id, survivor = h.clients[1].add_request_with_id(prompt, copy.deepcopy(params))
            assert survivor_id == 0
            await until(lambda: len(h.service.coordinator.request_id_to_rank) == 2)
            h.clients[0].abort_request(0)
            assert cancelled.cancelled()
            await until(lambda: any(e["header"] == Headers.ABORT_REQUEST for e in h.service.events))
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(survivor, timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
            await until(lambda: not h.service.coordinator.request_id_to_rank)
            await until(lambda: not h.clients[0].aborted_request_ids)
        await h.barrier()
        h.assert_retired()


@pytest.mark.parametrize("synchronous", [False, True])
async def test_routed_epoch_change_reaches_all_engines(monkeypatch, synchronous):
    """A paused generation update reaches idle peers and the active request."""
    prompt, params = list(range(4, 20)), greedy_params(num_tokens_to_generate=16)
    async with routed_model(monkeypatch) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        h.engine.use_synchronous_zmq_collectives = synchronous
        calls, operation = [], h.engine.world_zmq_communicator.all_reduce_max

        async def observed_reduce(*args, **kwargs):
            result = await operation(*args, **kwargs)
            calls.append(kwargs["async_op"])
            return result

        monkeypatch.setattr(h.engine.world_zmq_communicator, "all_reduce_max", observed_reduce)
        future = await _pause_active(h, monkeypatch, prompt, params)
        if h.rank == 0:
            h.clients[0].set_generation_epoch(7)
        await until(lambda: h.engine._generation_epoch == 7)
        if h.rank == 0:
            request = h.engine.get_request(0)
            assert (request.policy_epoch, request.kv_cache_epoch) == ([(0, 7)], [(0, 7)])
        await h.barrier()
        await h.unpause()
        assert len(calls) >= 2 and set(calls) == {not synchronous}
        if h.rank == 0:
            final = await asyncio.wait_for(future, timeout=60)
            assert final["generated_tokens"] == direct["generated_tokens"]
            assert (final["policy_epoch"], final["kv_cache_epoch"]) == ([[0, 7]], [[0, 7]])
        await h.barrier()
        h.assert_retired()


async def test_routed_allocator_pressure_preserves_victim_identity(monkeypatch):
    prompt, params = list(range(4, 20)), greedy_params()
    # Sized for five active blocks and one paused block at the fixture's hidden size.
    async with routed_model(
        monkeypatch, context_buffer_size_gb=0.00008, context_paused_buffer_size_gb=0.000016
    ) as h:
        direct = await h.direct(prompt, params)
        context, transitions = h.engine.context, []
        allocator, update = context.kv_block_allocator, context.update_requests
        resume_spy = mock.Mock(wraps=context.resume_paused_requests)
        assert (allocator.pool_size, allocator.paused_limit) == (5, 1)

        def observed_update(*args, **kwargs):
            result = update(*args, **kwargs)
            evicted = result["evict_request_ids"]
            if evicted is not None:
                paused = result["newly_paused_request_ids"]
                transitions.append((paused.shape, paused.flatten().tolist(), evicted.tolist()))
            return result

        monkeypatch.setattr(context, "resume_paused_requests", resume_spy)
        monkeypatch.setattr(context, "update_requests", observed_update)
        await h.start()
        count = h.dp_size * (allocator.pool_size - 2)
        finals, owners, local_ids = await h.run_paused_batch([(0, prompt, params)] * count)
        reference = (direct["generated_tokens"], direct["status"])
        assert [(r["generated_tokens"], r["status"]) for r in finals] == [reference] * count
        assert local_ids == [(0, request_id) for request_id in range(count)]
        [(shape, paused, evicted)] = transitions
        assert resume_spy.call_args_list[0].args[1].shape == (allocator.pool_size - 2,)
        assert paused == evicted and shape == (1,)
        target = evicted[0]
        assert owners[target] == f"mp-coord-{h.rank}".encode()
        assert sum(bool(s["prefill"]) for s in h.witnesses if target in s["ids"]) >= 2
        h.assert_retired()


async def test_routed_uvm_drained_reset_preserves_live_controls(monkeypatch):
    config_constructor = engine_tests.InferenceConfig.__init__
    pool_spy = mock.Mock(wraps=unified_memory_module.MemPool)

    def uvm_config(config, *args, **kwargs):
        kwargs["unified_memory_level"] = 1
        config_constructor(config, *args, **kwargs)

    def engine_factory(config):
        with mock.patch.object(engine_tests.InferenceConfig, "__init__", uvm_config):
            return engine_tests.DynamicInferenceEngineTestBase._build_test_env(config).engine

    monkeypatch.setattr(unified_memory_module, "MemPool", pool_spy)
    prompt, params = list(range(4, 20)), greedy_params()
    async with routed_model(monkeypatch, engine_factory=engine_factory) as h:
        context = h.engine.context
        assert context.unified_memory_level == pool_spy.call_count == 1
        assert pool_spy.call_args.kwargs["allocator"] is unified_memory_module._alloc
        pointer, end = context.memory_buffer.data_ptr(), context.memory_buffer.nbytes
        assert any(
            s["address"] <= pointer and pointer + end <= s["address"] + s["total_size"]
            for s in context.unified_memory_mempool.snapshot()
        )
        unified_memory_module.advise_managed_tensor_preferred_location(
            context.memory_buffer, device=-1
        )
        unified_memory_module.prefetch_managed_tensor(context.memory_buffer, device=-1)
        unified_memory_module.prefetch_managed_tensor(context.memory_buffer, device=h.rank)
        torch.cuda.synchronize()
        direct = await h.direct(prompt, params)
        await _exercise_all_owners(h, prompt, params, direct)
        names = ("_cond", "_state_events", "_pending_signals", "world_zmq_communicator")
        runtime = {name: getattr(h.engine, name) for name in names}
        h.engine.reset()
        assert all(getattr(h.engine, name) is value for name, value in runtime.items())
        assert h.engine.use_coordinator and h.engine.state == EngineState.PAUSED
        assert h.engine._state_events[EngineState.PAUSED].is_set()
        h.witnesses.clear()
        reference = direct["generated_tokens"], direct["status"]
        await h.barrier()
        await h.unpause()
        if h.rank == 0:
            final = await asyncio.wait_for(h.clients[0].add_request(prompt, params), 60)
            assert (final["generated_tokens"], final["status"]) == reference
        await h.barrier()
        assert await h.sync.all_reduce_max(bool(h.witnesses)) == 1
        h.assert_retired()
