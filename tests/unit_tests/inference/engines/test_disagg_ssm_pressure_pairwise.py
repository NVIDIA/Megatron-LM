# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Real SSM allocator pressure must defer complete KV/recurrent imports in FIFO order."""

from unittest import mock

import torch
import torch.distributed as dist

from megatron.core.ssm.mamba_mixer import MambaMixer
from tests.unit_tests.inference.engines import disagg_test_utils as du
from tests.unit_tests.inference.engines import test_dynamic_engine as dynamic_tests
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401


@torch.inference_mode()
@mock.patch.object(MambaMixer, "forward", autospec=True, side_effect=MambaMixer.forward)
def test_ssm_slot_pressure_defers_real_handoffs(mixer, transport_world, monkeypatch):
    constructor = dynamic_tests.TransformerConfig
    monkeypatch.setattr(
        dynamic_tests,
        "TransformerConfig",
        lambda **kw: constructor(**(kw | {"num_attention_heads": 4})),
    )
    config = du.disagg_config(model_provider="hybrid", flash_attention_version=4)
    tokens = du.prompt(33)
    weights, expected = du.collocated_reference(config, tokens, du.sampling())
    source = dist.get_rank() % 2 == 0
    with du.real_engine(config, role="prefill" if source else "decode", weights=weights) as engine:
        context, publication = engine.context, []
        slots = context.mamba_metadata
        states = {"conv": context.mamba_conv_states, "recurrent": context.mamba_ssm_states}
        if source:
            for rid in (101, 102):
                request = du.run_to_completion(
                    engine, engine.add_request(rid, tokens, du.sampling(1, do_kv_handoff=True))
                )
                metadata = request.disaggregated_params
                slot = engine._pinned_handoff_ssm_slots[rid]
                state = {name: value[:, slot].cpu().clone() for name, value in states.items()}
                state["kv"] = context.memory_buffer[:, :, metadata["block_ids"]].cpu().clone()
                publication.append((metadata, state))
            assert len(set(engine._pinned_handoff_ssm_slots.values())) == 2
        peer = du.exchange(publication, transport_world)
        with du.deferred_pulls(engine) as pulls:
            if not source:
                publication = peer
                held = slots.batch_allocate_slots(slots.max_requests)
                for value in states.values():
                    value[:, held] = 7.25
                assert slots.allocate_slot() is None and context.kv_block_allocator.pool_avail > 6
                futures = [
                    engine.add_request_with_kv_handoff(
                        rid, tokens, du.sampling(), metadata["kv_meta"], metadata["block_ids"]
                    )
                    for rid, (metadata, _) in zip((101, 102), publication)
                ]
                assert [item.request_id for item in engine._deferred_kv_handoffs] == [101, 102]
                assert not engine._pending_kv_imports and not pulls
                slots.free_slot(int(held[0]))
            for index, rid in enumerate((101, 102)):
                if not source:
                    assert engine._drain_deferred_kv_handoffs() == 1
                    pending = engine._pending_kv_imports[0]
                    assert pending.request_id == rid and pending.ssm.live_slot == int(held[0])
                    # One prepared transfer owns KV and both SSM components.
                    assert len(pulls) == index + 1
                peer = du.exchange(
                    None if source else du.decode_peer_meta(engine, pending), transport_world
                )
                if source:
                    du.push_handoff(engine, rid, [peer])
                    handles = engine._pending_kv_pushes[-1][1]
                else:
                    handles = engine._pending_transfer_handles(pending)
                for handle in handles:
                    handle.wait()
                    assert handle.poll()
                if not source:
                    du.assert_import_equal(engine, pending, publication[index][1], len(tokens))
                    du.admit_import(engine)
                    mixer.reset_mock()
                    with du.ForwardWitness(engine, rid) as witness:
                        result = du.run_to_completion(engine, futures[index])
                    assert result.generated_tokens == expected and mixer.call_count > 0
                    assert witness.steps and all(not step[2] for step in witness.steps)
                else:
                    assert engine._poll_pending_kv_pushes() == 1
                    engine.release_handoff_blocks(rid)
                dist.barrier(group=transport_world)
            if not source:
                for slot in held[1:]:
                    slots.free_slot(int(slot))
                assert all(torch.all(value[:, held[1:]] == 7.25) for value in states.values())
            assert slots.mamba_state_free_slot_count == slots.max_requests
            assert sorted(slots.mamba_state_free_slots.tolist()) == list(range(slots.max_requests))
            du.assert_released(engine)
