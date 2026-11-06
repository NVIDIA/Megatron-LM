# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Imported prefix reuse and speculative decode continuation on real engines."""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.disaggregation.mtp_draft_handoff import (
    MTP_DRAFT_Q_KEY,
    MTP_VERIFICATION_KEY,
    decode_mtp_draft_q,
)
from megatron.core.inference.sampling_params import SamplingParams
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
    requires_nixl,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401


@pytest.mark.parametrize(
    "mode",
    [pytest.param("cached-prefix", marks=requires_nixl), "mtp-depth-two", "mtp-kv-depth-two"],
)
@torch.inference_mode()
def test_cached_prefix_and_mtp_handoff(transport_world, mode):
    cached = mode == "cached-prefix"
    depth = 0 if cached else 2
    mtp_kv = mode == "mtp-kv-depth-two"
    config = disagg_config(
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        num_speculative_tokens=depth,
        mtp_use_repeated_layer=mtp_kv,
    )
    tokens = prompt(33 if cached else 31)
    weights, expected = collocated_reference(config, tokens, sampling(7))
    source = dist.get_rank() % 2 == 0
    with real_engine(
        config,
        role="prefill" if source else "decode",
        backend="nixl" if cached else "nccl",
        weights=weights,
    ) as engine:
        assert engine.context.enable_mtp_kv_cache == mtp_kv
        inner_calls = [0] * (1 if mtp_kv else depth)
        hooks = []
        if depth:
            model = engine.controller.inference_wrapped_model.model
            assert len(model.mtp.layers) == len(inner_calls)
            for index, layer in enumerate(model.mtp.layers):

                def record_inner(module, args, output, index=index):
                    ids = engine.context.request_ids[: engine.context.total_request_count]
                    if 101 in ids.tolist():
                        inner_calls[index] += 1

                hooks.append(layer.mtp_model_layer.register_forward_hook(record_inner))
        try:
            metadata = state = None
            if source:
                request = run_to_completion(
                    engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
                )
                assert request.generated_tokens == expected[:1]
                metadata, state = snapshot_source(engine, request)
                assert len(metadata["kv_meta"]["resume_tokens"]) == depth + 1
                if depth:
                    # Greedy requests keep exact-match verification: no draft q is sent.
                    assert metadata["kv_meta"][MTP_VERIFICATION_KEY] == "rejection"
                    assert MTP_DRAFT_Q_KEY not in metadata["kv_meta"]
                if mtp_kv:
                    slot = engine.context.mtp_kv_layer_slot
                    assert metadata["kv_meta"]["num_layers_global"] == slot + 1
                    plane = state["kv"][:, slot].flatten(1, 2)[:, : len(tokens)]
                    assert torch.isfinite(plane).all() and plane.count_nonzero() > 0
            peer = exchange((metadata, state) if source else None, transport_world)
            if not source:
                metadata, state = peer
            for iteration in range(2 if cached else 1):
                agent = engine._kv_transfer_agent
                method = "prepare_pull_blocks" if agent.is_push else "begin_pull_blocks"
                with mock.patch.object(agent, method, wraps=getattr(agent, method)) as pull:
                    pending, future = complete_transfer(
                        engine, metadata, tokens, sampling(7), transport_world
                    )
                if not source:
                    assert pull.call_count == 1
                    sent_source_blocks, imported_blocks = pull.call_args.args[1:]
                    expected_cached = 2 if cached and iteration else 0
                    assert pending.cached_prefix_block_count == expected_cached
                    assert len(imported_blocks) == len(metadata["block_ids"]) - expected_cached
                    assert sent_source_blocks == metadata["block_ids"][expected_cached:]
                    if expected_cached:
                        assert len(imported_blocks) == 1, "Only the uncached tail should transfer"
                        assert pending.local_blocks[:2] == previously_imported_prefix
                    assert_import_equal(engine, pending, state, len(tokens))
                    if mtp_kv:
                        assert state["kv"].shape[1] == engine.context.mtp_kv_layer_slot + 1
                    if iteration == 0:
                        value = engine.context.memory_buffer[0, 0, pending.local_blocks[0], 0, 0, 0]
                        saved = value.clone()
                        value.copy_(saved + 1)
                        with pytest.raises(AssertionError):
                            assert_import_equal(engine, pending, state, len(tokens))
                        value.copy_(saved)
                        assert_import_equal(engine, pending, state, len(tokens))
                    previously_imported_prefix = list(pending.local_blocks[:2])
                    transferred_inputs = list(pending.resume_tokens)
                    assert transferred_inputs == metadata["kv_meta"]["resume_tokens"]
                    if depth:
                        assert len(transferred_inputs) == 3
                        assert len(pending.continuation_blocks) == 1
                        expected_resume_blocks = [
                            pending.local_blocks[-1],
                            *pending.continuation_blocks * 2,
                        ]
                    admit_import(engine)
                    context = engine.context
                    assert context.request_query_lengths[0].item() == depth + 1
                    assert context.token_to_input_ids[: depth + 1].tolist() == transferred_inputs
                    assert context.token_to_pos_ids[: depth + 1].tolist() == list(
                        range(len(tokens), len(tokens) + depth + 1)
                    )
                    if depth:
                        assert context.token_to_block_idx[:3].tolist() == expected_resume_blocks
                    with ForwardWitness(engine) as witness:
                        actual = run_to_completion(engine, future)
                    assert actual.generated_tokens == expected
                    if depth:
                        # The imported first token counts as the prefill engine's step.
                        steps = actual.acceptance_step_lengths
                        assert steps[0] == 1 and sum(steps) == len(actual.generated_tokens)
                    assert witness.steps and all(not step[2] for step in witness.steps)
                    assert witness.steps[0][0] == len(tokens)
                    assert_released(engine)
                dist.barrier(group=transport_world)
            if depth:
                assert all(inner_calls), "Every inner MTP layer must execute for the handoff target"
                if mtp_kv:
                    assert inner_calls[0] >= depth, "Shared MTP head must execute every draft depth"
            if source:
                engine._poll_pending_kv_pushes()
                engine.release_handoff_blocks(101)
                assert_released(engine)
        finally:
            for hook in hooks:
                hook.remove()


@torch.inference_mode()
def test_sampled_mtp_handoff_verifies_first_drafts_with_transferred_q(transport_world):
    """A sampled import's first decode step verifies the prefill engine's drafts by rejection
    sampling, using the exact draft distributions q sent with the handoff."""
    depth = 2
    config = disagg_config(num_speculative_tokens=depth)
    tokens = prompt(31)
    params = SamplingParams(temperature=1.0, top_p=0.9, termination_id=-1, num_tokens_to_generate=7)
    weights, _ = collocated_reference(config, tokens, params)
    source = dist.get_rank() % 2 == 0
    with real_engine(
        config, role="prefill" if source else "decode", backend="nccl", weights=weights
    ) as engine:
        controller = engine.controller
        assert controller.mtp_draft_q_shape() == (depth, controller.vocab_size)
        metadata = state = q_ref = None
        if source:
            handoff_params = SamplingParams(
                temperature=1.0,
                top_p=0.9,
                termination_id=-1,
                num_tokens_to_generate=1,
                do_kv_handoff=True,
            )
            request = run_to_completion(engine, engine.add_request(101, tokens, handoff_params))
            metadata, state = snapshot_source(engine, request)
            kv_meta = metadata["kv_meta"]
            assert kv_meta[MTP_VERIFICATION_KEY] == "rejection"
            drafts = kv_meta["resume_tokens"][1:]
            row = controller._mtp_draft_row_of_request[101]
            q_ref = controller._mtp_draft_probs[:, row].cpu()
            assert drafts == controller._sampled_mtp_tokens_cuda[:depth, row].tolist()
            decoded = decode_mtp_draft_q(
                kv_meta[MTP_DRAFT_Q_KEY],
                num_depths=depth,
                vocab=controller.vocab_size,
                drafts=drafts,
            )
            assert torch.equal(decoded, q_ref)
        peer = exchange((metadata, state, q_ref) if source else None, transport_world)
        if not source:
            metadata, state, q_ref = peer
        pending, future = complete_transfer(engine, metadata, tokens, params, transport_world)
        if not source:
            assert torch.equal(pending.mtp_draft_q, q_ref)
            admit_import(engine)
            assert set(controller._imported_mtp_draft_q) == {101}
            chain_calls = []
            real_chain = controller._sampling.chain_speculative_sampling

            def spy(draft_probs, draft_tokens, target_probs):
                chain_calls.append((draft_probs.clone(), draft_tokens.clone()))
                return real_chain(draft_probs, draft_tokens, target_probs)

            controller._sampling.chain_speculative_sampling = spy
            actual = run_to_completion(engine, future)
            # The first verification used the imported q for the prefill engine's drafts.
            assert chain_calls, "the imported request's first drafts must use rejection sampling"
            first_q, first_drafts = chain_calls[0]
            assert torch.equal(first_q[0].cpu(), q_ref)
            assert first_drafts[0].tolist() == metadata["kv_meta"]["resume_tokens"][1:]
            assert not controller._imported_mtp_draft_q
            assert len(actual.generated_tokens) == 7
            assert actual.generated_tokens[0] == metadata["kv_meta"]["resume_tokens"][0]
            steps = actual.acceptance_step_lengths
            assert steps[0] == 1 and sum(steps) == len(actual.generated_tokens)
            assert_released(engine)
        dist.barrier(group=transport_world)
        if source:
            engine._poll_pending_kv_pushes()
            engine.release_handoff_blocks(101)
            assert_released(engine)
