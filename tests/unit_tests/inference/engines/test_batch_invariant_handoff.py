# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant logits across a real prefill-to-decode NIXL handoff."""

import gc
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.engine import StateHandoffDynamicInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.engines import batch_invariant_test_utils as fixture
from tests.unit_tests.inference.engines.disagg_test_utils import requires_nixl


def _params(count, **changes):
    return SamplingParams(num_tokens_to_generate=count, top_k=1, termination_id=-1, **changes)


def _exchange(value, group):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=group)
    return values[dist.get_rank() ^ 1]


def _build(case, backend, version):
    with mock.patch.object(fixture, "DynamicInferenceEngine", StateHandoffDynamicInferenceEngine):
        engine = fixture.build_engine(case, backend, version)
    assert isinstance(engine, StateHandoffDynamicInferenceEngine)
    return engine


def _canonical_state(engine):
    model = engine.controller.inference_wrapped_model.model
    for parameter in model.parameters():
        assert parameter.is_cuda
        dist.broadcast(parameter.data, src=0)
    return {
        name: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for name, value in model.state_dict().items()
    }


def _load_canonical(engine, expected):
    model = engine.controller.inference_wrapped_model.model
    model.load_state_dict(expected, strict=True)
    actual = model.state_dict()
    for name, value in actual.items():
        if torch.is_tensor(value):
            assert torch.equal(value.detach().cpu(), expected[name]), name


def _run(engine, future, limit=128):
    for _ in range(limit):
        if future.done():
            return future.result()
        engine.step_modern()
    raise AssertionError("request did not finish within the real-engine step bound")


def _rows(witness):
    rows = {}
    for step in witness.steps:
        assert step["positions"].is_cuda and step["tokens"].is_cuda
        assert step["logits"].is_cuda
        for position, token, logits in zip(step["positions"], step["tokens"], step["logits"]):
            key = int(position), int(token)
            assert key not in rows
            rows[key] = logits
    assert rows
    return rows


def _assert_rows(expected, witness):
    observed = _rows(witness)
    for key, logits in observed.items():
        assert key in expected, key
        assert torch.equal(logits, expected[key]), key
    return set(observed)


def _close(engine):
    if engine._kv_transfer_agent is not None:
        engine._kv_transfer_agent.close()
    for socket in getattr(engine, "zmq_sockets", []):
        socket.close(linger=0)
    gc.collect()
    torch.cuda.empty_cache()


def _assert_pages(engine, blocks, expected):
    actual = engine.context.memory_buffer[:, :, blocks].detach().cpu()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@requires_nixl
@pytest.mark.parametrize("length", [17, 256, 257], ids=["64-to-128", "full-page", "partial-tail"])
@torch.inference_mode()
def test_batch_invariant_real_nixl_handoff(length):
    case = fixture.Case(
        f"handoff-{length}", context={"enable_prefix_caching": True}, prompt_length=length
    )
    tokens = [(7 + index * 13) % (fixture.VOCAB - 2) + 1 for index in range(length)]
    with fixture.invariant_runtime(case) as (backend, version):
        assert dist.get_world_size() == 2 and dist.get_backend() == "nccl"
        source = dist.get_rank() == 0
        control = dist.new_group(backend="gloo")
        assert dist.get_backend(control) == "gloo"

        reference = _build(case, backend, version)
        weights = _canonical_state(reference)
        with pytest.MonkeyPatch.context() as patch:
            reference_witness = fixture.ForwardWitness(reference, patch)
            expected_request = _run(reference, reference.add_request(101, tokens, _params(4)))
        reference_witness.assert_active(version)
        expected_rows = _rows(reference_witness)
        expected_tokens = list(expected_request.generated_tokens)
        assert _exchange(expected_tokens, control) == expected_tokens
        assert reference.context.total_request_count == 0
        if length == 17:
            assert {step["physical"] for step in reference_witness.steps} == {64}
        _close(reference)
        del reference, reference_witness

        engine = _build(case, backend, version)
        _load_canonical(engine, weights)
        role = "prefill" if source else "decode"
        engine.context.config.kv_transport_backend = "nixl"
        engine.setup_kv_transfer(role)
        agent = engine._kv_transfer_agent
        assert agent._memory_buffer.data_ptr() == engine.context.memory_buffer.data_ptr()
        assert agent._memory_buffer.is_cuda

        metadata = state = None
        source_tail_finite = True
        continuation_zero = True
        observed_keys = set()
        with pytest.MonkeyPatch.context() as patch:
            witness = fixture.ForwardWitness(engine, patch)
            if source:
                for request_id in range(201, 265):
                    engine.add_request(request_id, [request_id % 97 + 1], _params(1))
                future = engine.add_request(101, tokens, _params(1, do_kv_handoff=True))
                source_result = _run(engine, future)
                witness.assert_active(version)
                observed_keys = _assert_rows(expected_rows, witness)
                target_steps = [step for step in witness.steps if not step["decode"]]
                assert any(step["requests"] == 65 for step in target_steps)
                if length == 17:
                    assert {step["physical"] for step in target_steps} == {128}
                assert source_result.generated_tokens == expected_tokens[:1]
                assert not engine.has_unfinished_requests()
                metadata = source_result.disaggregated_params
                blocks = metadata["block_ids"]
                assert len(blocks) == (length + 255) // 256
                refs = engine.context.kv_block_allocator.block_ref_counts[blocks]
                assert (refs > 0).all() and engine._pinned_handoff_blocks[101] == blocks
                state = engine.context.memory_buffer[:, :, blocks].detach().cpu().clone()
                tail = length % 256
                if tail:
                    values = engine.context.memory_buffer[1, :, blocks[-1], tail:]
                    source_tail_finite = bool(torch.isfinite(values).all())

            packet = _exchange((metadata, state) if source else None, control)
            if not source:
                metadata, state = packet
                if length == 17:
                    for scores in ({"return_log_probs": True}, {"top_n_logprobs": 2}):
                        with pytest.raises(NotImplementedError, match="log probabilities"):
                            engine.add_request_with_kv_handoff(
                                909,
                                tokens,
                                _params(4, **scores),
                                metadata["kv_meta"],
                                metadata["block_ids"],
                            )
                future = engine.add_request_with_kv_handoff(
                    101, tokens, _params(4), metadata["kv_meta"], metadata["block_ids"]
                )
                pending = engine._pending_kv_imports[0]
                assert len(pending.continuation_blocks) == int(length == 256)
                owned = pending.local_blocks + pending.continuation_blocks
                assert (engine.context.kv_block_allocator.block_ref_counts[owned] > 0).all()
                handle = pending.handle
                assert handle.xfers and handle.contexts
                assert (
                    agent._num_outer * agent._bytes_per_slice * len(pending.local_blocks)
                    == state.numel() * state.element_size()
                )
                handle.wait()
                assert handle.poll() and handle.done
                _assert_pages(engine, pending.local_blocks, state)
                probe = engine.context.memory_buffer[0, 0, pending.local_blocks[0], 0, 0, 0]
                probe.add_(1)
                with pytest.raises(AssertionError):
                    _assert_pages(engine, pending.local_blocks, state)
                restored = agent.begin_pull_blocks(
                    metadata["kv_meta"], metadata["block_ids"], pending.local_blocks
                )
                restored.wait()
                assert restored.poll() and restored.done
                _assert_pages(engine, pending.local_blocks, state)
                if pending.continuation_blocks:
                    continuation = engine.context.memory_buffer[1, :, pending.continuation_blocks]
                    continuation_zero = bool(torch.count_nonzero(continuation) == 0)

            _exchange(True, control)
            if source:
                engine.release_handoff_blocks(101)
                assert not engine._pinned_handoff_blocks
            else:
                assert engine._poll_pending_kv_imports() == 1
                assert engine._admit_pending_kv_imports() == 1
                result = _run(engine, future)
                witness.assert_active(version)
                observed_keys = _assert_rows(expected_rows, witness)
                assert result.generated_tokens == expected_tokens
                assert result.num_cached_tokens == length

        assert engine.context.total_request_count == 0
        assert (engine.context.kv_block_allocator.block_ref_counts == 0).all()
        peer_tail, peer_continuation, peer_keys = _exchange(
            (source_tail_finite, continuation_zero, observed_keys), control
        )
        all_keys = observed_keys | peer_keys
        dist.barrier(group=control)
        _close(engine)
        del engine, witness
        dist.destroy_process_group(control)
        assert all_keys == set(expected_rows)
        assert source_tail_finite and peer_tail
        assert continuation_zero and peer_continuation
        print("BI_HANDOFF_WITNESS", length, backend, version, role, len(all_keys))
