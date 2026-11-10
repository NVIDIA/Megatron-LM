# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collocated parity and exact transfer witnesses for real disaggregated engines."""

from collections import deque
from dataclasses import replace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from flashinfer.rope import apply_rope_with_cos_sin_cache

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.ssm.mamba_mixer import MambaMixer
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    admit_import,
    assert_import_equal,
    assert_released,
    collocated_reference,
    complete_transfer,
    decode_peer_meta,
    disagg_config,
    exchange,
    prompt,
    real_engine,
    requires_nixl,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def transport_world():
    Utils.initialize_model_parallel()
    assert dist.get_world_size() >= 2 and dist.get_world_size() % 2 == 0
    assert dist.get_backend() == "nccl"
    control = dist.new_group(backend="gloo")
    assert dist.get_backend(control) == "gloo"
    yield control
    dist.destroy_process_group(control)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "backend,length,changes,count",
    [
        pytest.param("nccl", 32, {}, 7, id="nccl-boundary"),
        pytest.param("nccl", 33, {}, 7, id="nccl-partial-tail"),
        pytest.param("nixl", 33, {}, 7, id="nixl-partial-tail", marks=requires_nixl),
        pytest.param(
            "nccl", 33, dict(context_block_size_tokens=256, flash_attention_version=2), 7, id="fa2"
        ),
        pytest.param(
            "nccl",
            33,
            {"hidden_size": 256, "flash_attention_version": 4, "position_embedding_type": "rope"},
            7,
            id="fa4-d64",
        ),
        pytest.param(
            "nccl", 33, {"async_sched_mode": AsyncScheduleMode.ASYNC}, 7, id="async-decode"
        ),
        pytest.param(
            "nccl",
            129,
            {
                "enable_chunked_prefill": True,
                "context_max_tokens": 64,
                "num_cuda_graphs": 2,
                "force_build_cuda_graphs": True,
            },
            7,
            id="chunked-prefill",
        ),
        pytest.param("nccl", 33, {"sampling_backend": "flashinfer"}, 7, id="flashinfer"),
        pytest.param("nccl", 33, {}, 1, id="terminal-first-token"),
        pytest.param("nccl", 33, {"model_provider": "hybrid"}, 7, id="mamba-exact-state"),
        pytest.param(
            "nccl",
            33,
            {
                "num_cuda_graphs": 2,
                "force_build_cuda_graphs": True,
                "cuda_graph_max_tokens": 16,
                "cuda_graph_all_prefills": True,
            },
            7,
            id="all-prefills-graph-budget",
        ),
        pytest.param(
            "nccl",
            33,
            {"num_cuda_graphs": 2, "force_build_cuda_graphs": True},
            7,
            id="decode-graph",
        ),
    ],
)
@torch.inference_mode()
@mock.patch("flashinfer.rope.apply_rope_with_cos_sin_cache", wraps=apply_rope_with_cos_sin_cache)
@mock.patch.object(MambaMixer, "forward", autospec=True, side_effect=MambaMixer.forward)
def test_disagg_real_engine_parity(mixer, rope, transport_world, backend, length, changes, count):
    config = disagg_config(**changes)
    tokens = prompt(length)
    reference_config = replace(config, async_sched_mode=AsyncScheduleMode.LEGACY)
    weights, expected = collocated_reference(reference_config, tokens, sampling(count))
    assert exchange(expected, transport_world) == expected
    source = dist.get_rank() % 2 == 0
    async_decode = config.async_sched_mode == AsyncScheduleMode.ASYNC
    chunked = changes.get("enable_chunked_prefill")
    neighbor_tokens = [91, 92, 93, 94]
    if async_decode or chunked:
        with real_engine(reference_config, weights=weights) as reference:
            future = reference.add_request(102, neighbor_tokens, sampling(16))
            neighbor_expected = run_to_completion(reference, future).generated_tokens
    role = "prefill" if source else "decode"
    with real_engine(config, role=role, backend=backend, weights=weights) as engine:
        with ForwardWitness(engine) as witness:
            mixer.reset_mock()
            rope.reset_mock()
            neighbor = None
            if async_decode:
                metadata = None
                if source:
                    seed = engine.add_request(102, neighbor_tokens, sampling(1, do_kv_handoff=True))
                    metadata = run_to_completion(engine, seed).disaggregated_params
                metadata = exchange(metadata, transport_world)
                _, seeded = complete_transfer(
                    engine, metadata, neighbor_tokens, sampling(16), transport_world, 102
                )
                if source:
                    engine._poll_pending_kv_pushes()
                    engine.release_handoff_blocks(102)
                else:
                    admit_import(engine)
                    neighbor = seeded
                    engine.step_modern()
                    assert engine.controller._async_sched_forward.is_valid
            elif source and chunked:
                neighbor = engine.add_request(102, neighbor_tokens, sampling(16))
                engine.step_modern()
                assert engine.context.request_ids[0] == 102 and not neighbor.done()
                assert not engine.context.request_in_prefill_status_tensor[0].item()
            metadata = state = None
            if source:
                request = run_to_completion(
                    engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
                )
                assert request.generated_tokens == expected[:1]
                metadata, state = snapshot_source(engine, request)
                block_size = engine.context.block_size_tokens
                assert len(metadata["block_ids"]) == (length + block_size - 1) // block_size
                assert witness.steps and all(step[2] for step in witness.steps)
                if chunked:
                    mixed_offsets = {step[0] for step in witness.steps if 102 in step[3]}
                    assert len(mixed_offsets) >= 3
                    assert all(102 in step[3] for step in witness.steps)
            transferred = exchange((metadata, state) if source else None, transport_world)
            if not source:
                metadata, state = transferred
            params = sampling(count)
            if count == 1:
                params.num_tokens_to_generate, params.num_tokens_total = None, length + count
            pending, future = complete_transfer(engine, metadata, tokens, params, transport_world)
            if not source:
                assert_import_equal(engine, pending, state, length)
                if length == 32:
                    block = pending.local_blocks[0]
                    transferred = engine.context.memory_buffer[0, 0, block, 0, 0, 0]
                    saved = transferred.clone()
                    transferred.copy_(saved + 1)
                    with pytest.raises(AssertionError):
                        assert_import_equal(engine, pending, state, length)
                    transferred.copy_(saved)
                    assert_import_equal(engine, pending, state, length)
                assert pending.resume_tokens == expected[:1]
                if neighbor is None:
                    assert not engine.context.total_request_count
                    admit_import(engine)
                else:
                    assert engine.controller._async_sched_forward.is_valid
                    assert engine.context.total_request_count == 1
                    assert engine._poll_pending_kv_imports() == 1
                    engine.step_modern()  # Resolve neighbor before normal import admission.
                if count > 1:
                    row = engine.context.request_ids.tolist().index(101)
                    if neighbor is None:
                        assert engine.context.request_kv_length_offsets[row].item() == length
                    assert not engine.context.request_in_prefill_status_tensor[row].item()
                    result = run_to_completion(engine, future)
                    assert witness.steps and all(not step[2] for step in witness.steps)
                    assert witness.steps[0][0] == length
                    if neighbor is not None:
                        assert any(102 in step[3] for step in witness.pending_forwards)
                else:
                    assert future.done()
                    result = future.result()
                    assert not witness.steps
                assert result.generated_tokens == expected
                assert result.sampling_params.num_tokens_to_generate == count
                assert result.sampling_params.num_tokens_total is None
                assert result.num_cached_tokens == length
                if neighbor is None:
                    assert_released(engine)
            if neighbor is not None:
                assert run_to_completion(engine, neighbor).generated_tokens == neighbor_expected
                if not source:
                    assert_released(engine)
            assert config.model_provider != "hybrid" or mixer.call_count == len(witness.steps)
            dist.barrier(group=transport_world)
            if source:
                engine._poll_pending_kv_pushes()
                assert not engine._pending_kv_pushes
                assert not engine.requests
                with pytest.raises(RuntimeError, match="handoff state remains pinned"):
                    engine.reset()
                engine.release_handoff_blocks(101)
                engine.release_handoff_blocks(101)
                assert_released(engine)
            if not source and config.flash_attention_version in (2, 4):
                assert witness.attention_calls > 0
            if config.position_embedding_type == "rope":
                assert rope.call_count > 0
            if changes.get("force_build_cuda_graphs"):
                assert witness.graph_replays > 0
            if source and changes.get("cuda_graph_all_prefills"):
                assert changes["cuda_graph_max_tokens"] < length
                source_shapes = [shape for shape in witness.graph_shapes if shape[1] > 0]
                assert source_shapes
                assert all(shape[0] >= length for shape in source_shapes)
                assert any(shape[0] > changes["cuda_graph_max_tokens"] for shape in source_shapes)


class _LoopbackP2P:
    """Match the NCCL backend's planned sends and receives in post order, in one process."""

    class _Work:
        done = False

        def is_completed(self):
            return self.done

        def wait(self):
            assert self.done, "loopback: a point-to-point operation has no peer"

    def __init__(self):
        self.sends, self.receives = deque(), deque()

    def batch_isend_irecv(self, ops):
        works = []
        for op in ops:
            work = self._Work()
            (self.sends if op.op is dist.isend else self.receives).append((op.tensor, work))
            works.append(work)
        while self.sends and self.receives:
            (source, sent), (destination, received) = self.sends.popleft(), self.receives.popleft()
            assert source.shape == destination.shape
            destination.copy_(source)
            sent.done = received.done = True
        return works


def _has_work(engine):
    """Whether the engine loop would step: resident active rows, waiting work or an import."""
    return (
        engine.context.get_active_request_count() > 0
        or engine.waiting_request_ids
        or engine.has_admittable_kv_import
    )


@pytest.mark.parametrize("mode", [AsyncScheduleMode.LEGACY, AsyncScheduleMode.ASYNC])
@torch.inference_mode()
def test_decode_paused_behind_an_import_resumes_after_it(mode):
    """A lone decode paused because an import holds the last blocks waits, then resumes.

    Both real engines run in one process; a loopback carries the NCCL backend's own plans.
    """
    Utils.initialize_model_parallel()
    try:
        config = disagg_config(async_sched_mode=mode)
        reference_config = replace(config, async_sched_mode=AsyncScheduleMode.LEGACY)
        block = config.context_block_size_tokens
        a_tokens, b_tokens = prompt(block - 1), prompt(6 * block)
        weights, a_expected = collocated_reference(reference_config, a_tokens, sampling(4))
        with real_engine(reference_config, weights=weights) as reference:
            b_reference = reference.add_request(102, b_tokens, sampling(4))
            b_expected = run_to_completion(reference, b_reference).generated_tokens

        def handoff(engine, request_id, tokens):
            request = engine.add_request(request_id, tokens, sampling(0, do_kv_handoff=True))
            return run_to_completion(engine, request).disaggregated_params

        loopback = _LoopbackP2P()
        with real_engine(config, role="prefill", weights=weights) as prefill:
            buffer = prefill.context.memory_buffer
            block_bytes = buffer[:, :, 0].numel() * buffer.element_size()
            # Eight usable decode blocks.
            decode_config = disagg_config(
                async_sched_mode=mode, context_buffer_size_gb=9.5 * block_bytes / 1024**3
            )
            with (
                real_engine(decode_config, role="decode", weights=weights) as decode,
                mock.patch.object(dist, "batch_isend_irecv", loopback.batch_isend_irecv),
            ):
                context, allocator = decode.context, decode.context.kv_block_allocator
                assert allocator.pool_size - 1 == 8 and allocator.paused_limit >= 1
                a_meta, b_meta = handoff(prefill, 101, a_tokens), handoff(prefill, 102, b_tokens)
                decode._notify_kv_transfer_ready = lambda request_id, cached: None

                # A's one-block import completes; its first decode token fills the block.
                a_future = decode.add_request_with_kv_handoff(
                    101, a_tokens, sampling(4), a_meta["kv_meta"], a_meta["block_ids"]
                )
                a_pending = decode._pending_kv_imports[0]
                prefill.push_handoff_kv(101, [decode_peer_meta(decode, a_pending)], 0)
                assert decode._poll_pending_kv_imports() == 1

                # B's import takes the last seven blocks before A is admitted.
                b_future = decode.add_request_with_kv_handoff(
                    102, b_tokens, sampling(4), b_meta["kv_meta"], b_meta["block_ids"]
                )
                b_pending = decode._pending_kv_imports[-1]
                assert b_pending.request_id == 102 and allocator.pool_avail == 0

                # A is admitted, needs a new block for its second token, and pauses with no
                # request active while B's transfer is still in flight.
                for _ in range(4):
                    if _has_work(decode):
                        decode.step_modern()
                    if context.paused_request_count == 1:
                        break
                assert context.paused_request_count == context.total_request_count == 1
                assert context.kv_blocks_held_for_imports() == 7

                # B arrives, decodes and finishes; its blocks let A resume and finish.
                prefill.push_handoff_kv(102, [decode_peer_meta(decode, b_pending)], 0)
                for _ in range(64):
                    if a_future.done() and b_future.done():
                        break
                    decode._poll_pending_kv_imports()
                    if _has_work(decode):
                        decode.step_modern()
                assert b_future.result().generated_tokens == b_expected
                assert a_future.result().generated_tokens == a_expected
                assert_released(decode)

                assert prefill._poll_pending_kv_pushes() == 2
                prefill.release_handoff_blocks(101)
                prefill.release_handoff_blocks(102)
                assert_released(prefill)
    finally:
        Utils.destroy_model_parallel()
