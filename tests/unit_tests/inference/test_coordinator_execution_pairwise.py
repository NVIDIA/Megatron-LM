# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import Counter
from contextlib import asynccontextmanager

import pytest
import torch

from megatron.core.inference.config import CudaGraphSizingDistribution
from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode
from tests.unit_tests.inference import test_coordinator_features_pairwise as features
from tests.unit_tests.inference import test_coordinator_schedule_pairwise as schedule
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model

pytestmark = [pytest.mark.internal, pytest.mark.asyncio]


@pytest.mark.parametrize("all_prefills", [False, True])
@pytest.mark.parametrize("distribution", list(CudaGraphSizingDistribution))
async def test_routed_prefill_graph_selection(monkeypatch, all_prefills, distribution):
    original = schedule.routed_model

    @asynccontextmanager
    async def instrumented(*args, **kwargs):
        async with original(
            *args,
            **kwargs,
            hidden_size=128,
            num_cuda_graphs=3,
            force_build_cuda_graphs=True,
            use_cuda_graphs_for_non_decode_steps=True,
            cuda_graph_all_prefills=all_prefills,
            cuda_graph_max_tokens=16,
            cuda_graph_mixed_prefill_count=2,
            cuda_graph_sizing_distribution=distribution,
        ) as h:
            replay, observations = _CudagraphReplayNode.forward, []

            def observed(ctx, runner, first, *inputs):
                c = h.engine.context
                ids, padded = features._active_target_ids(h), c.padded_batch_dimensions
                counts = (c.active_token_count, c.num_prefill_requests, c.num_decode_requests)
                result = replay(ctx, runner, first, *inputs)
                if h.engine.use_coordinator and any(
                    len(h.engine.get_request(rid).prompt_tokens) == 65 for rid in ids
                ):
                    observations.append((padded, counts, inputs[0].shape[1]))
                return result

            monkeypatch.setattr(_CudagraphReplayNode, "forward", staticmethod(observed))
            yield h
            invalid = any(
                dim not in h.engine.context.cuda_graph_batch_dimensions_list
                or dim.token_count != physical
                for dim, _, physical in observations
            )
            is_linear = distribution == CudaGraphSizingDistribution.LINEAR
            expected = 12 if is_linear else 8 * (1 + all_prefills)
            mixed = any(
                n == 7 and p and d and dim.token_count == expected
                for dim, (n, p, d), _ in observations
            )
            prefill = any(
                p and not d and dim.token_count > 16 for dim, (_, p, d), _ in observations
            )
            assert await h.sync.all_reduce_max(mixed, prefill, invalid) == (1, int(all_prefills), 0)

    monkeypatch.setattr(schedule, "routed_model", instrumented)
    await schedule.test_routed_chunk_partition_beside_decode(monkeypatch)


@pytest.mark.parametrize("interval", [20, 3, None])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_routed_idle_expert_progress(monkeypatch, interval, synchronous):
    params = greedy_params(num_tokens_to_generate=32, return_log_probs=True)
    params.skip_prompt_log_probs = True
    async with routed_model(
        monkeypatch,
        expert_model_parallel_size=2 if interval else 1,
        use_moe_layer_spec=True,
        transformer_impl="inference_optimized",
        inference_moe_token_dispatcher_type="nccl",
    ) as h:
        h.engine.disable_ep_consensus = interval is None
        h.engine.ep_consensus_interval = interval or 20
        h.engine.use_synchronous_zmq_collectives = synchronous
        assert h.dp_size == 2 and h.engine.pg_collection.ep.size() == (2 if interval else 1)
        runtime, spans, cadence = Counter(), {False: set(), True: set()}, []
        span_key = "nccl-token-dispatches" if interval else "model-forwards"
        model = h.engine.controller.inference_wrapped_model.model
        for module in model.modules():
            features._instrument_nccl_dispatch_runtime(module, runtime)
        forward, consensus = model.forward, h.engine._ep_establish_consensus

        def observed(*args, **kwargs):
            real, before = bool(features._active_target_ids(h)), runtime[span_key]
            result = forward(*args, **kwargs)
            if h.engine.use_coordinator:
                runtime["model-forwards"] += 1
                spans[real].update(range(before, runtime[span_key]))
            return result

        def observed_dummy(dummy=h.engine.controller.dummy_forward):
            dummy()
            runtime["dummy-forwards"] += int(h.engine.use_coordinator)

        async def observed_consensus(work, *args, **kwargs):
            runtime["ep-consensus-calls"] += int(h.engine.use_coordinator)
            if work and h.engine._last_ep_consensus[0]:
                cadence.append(h.engine._ep_consensus_loop_counter)
            return await consensus(work, *args, **kwargs)

        monkeypatch.setattr(model, "forward", observed)
        monkeypatch.setattr(h.engine.controller, "dummy_forward", observed_dummy)
        monkeypatch.setattr(h.engine, "_ep_establish_consensus", observed_consensus)
        direct = await h.direct([4, 5, 6], params)
        await h.start()
        await h.pause()
        runtime.clear()
        spans, cadence = {False: set(), True: set()}, []
        [result], owners, local_ids = await h.run_paused_batch([(0, [4, 5, 6], params)])
        await h.pause()
        assert local_ids == [(0, 0)] and len(owners) == 1
        assert result["status"] == direct["status"] == "COMPLETED"
        assert result["generated_tokens"] == direct["generated_tokens"]
        scores = torch.tensor([r["generated_log_probs"] for r in (direct, result)])
        assert scores.shape == (2, 32) and torch.isfinite(scores).all()
        records = [None] * h.dp_size
        torch.distributed.all_gather_object(records, (spans, runtime, cadence))
        owner = next(i for i in range(h.dp_size) if owners[0] == f"mp-coord-{i}".encode())
        real, idle = records[owner][0][True], records[1 - owner][0]
        assert real and idle[False] and not idle[True] and (not interval or real <= idle[False])
        for _, stats, _ in records:
            assert interval or not stats["ep-consensus-calls"]
            assert interval or stats["dummy-forwards"] > 0
            assert stats["nccl-token-dispatches"] == stats["nccl-token-combines"] >= bool(interval)
            assert stats["nccl-combine-before-dispatch"] == stats["nccl-dispatch-inflight"] == 0
        calls = records[owner][2]
        assert (bool(calls) and all(c % interval == 0 for c in calls)) if interval else not calls
        h.assert_retired()
