# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collocated parity and exact transfer witnesses for real disaggregated engines."""

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
    disagg_config,
    exchange,
    prompt,
    real_engine,
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
        pytest.param("nixl", 33, {}, 7, id="nixl-partial-tail"),
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
