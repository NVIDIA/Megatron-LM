# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.disaggregation.coordinator_flow_control import DisaggStateFlowControl


def test_registration_uses_conservative_model_parallel_limits():
    flow = DisaggStateFlowControl()

    capacity = flow.register_engine(
        b"decode", "decode", [{"ssm_slot_capacity": 12}, {"ssm_slot_capacity": 10}]
    )

    assert capacity == 10
    assert flow.capacity(b"decode") == 10


def test_weighted_decode_reservations_are_fifo_and_held_until_release():
    flow = DisaggStateFlowControl()
    flow.register_engine(b"decode", "decode", [{"ssm_slot_capacity": 5}])

    assert flow.try_reserve(b"decode", 1, 4)
    assert not flow.try_reserve(b"decode", 2, 2)
    flow.enqueue(b"decode", 2, b"second", 2)
    flow.enqueue(b"decode", 3, b"third", 1)
    assert flow.pop_next_admissible(b"decode") is None

    assert flow.release_decode(1) == b"decode"
    second = flow.pop_next_admissible(b"decode")
    third = flow.pop_next_admissible(b"decode")

    assert [second.request_id, third.request_id] == [2, 3]
    assert flow.decode_usage(b"decode") == 3


def test_re_registration_replaces_stale_capacity():
    flow = DisaggStateFlowControl()
    flow.register_engine(b"decode", "decode", [{"ssm_slot_capacity": 5}])

    flow.register_engine(b"decode", "decode", [{"global_rank": 0}])

    assert flow.capacity(b"decode") is None


def test_queue_admission_reserves_only_one_handoff_at_a_time():
    flow = DisaggStateFlowControl()
    flow.register_engine(b"decode", "decode", [{"ssm_slot_capacity": 5}])
    flow.enqueue(b"decode", 1, b"first", 2)
    flow.enqueue(b"decode", 2, b"second", 2)

    admitted = flow.pop_next_admissible(b"decode")

    assert admitted.request_id == 1
    assert flow.decode_usage(b"decode") == 2
    assert flow.has_queued(b"decode")


def test_oversized_handoff_is_rejected_without_mutating_usage():
    flow = DisaggStateFlowControl()
    flow.register_engine(b"decode", "decode", [{"ssm_slot_capacity": 4}])

    assert not flow.can_ever_fit(b"decode", 5)
    assert flow.decode_usage(b"decode") == 0


def test_invalid_advertised_capacity_is_rejected():
    flow = DisaggStateFlowControl()

    with pytest.raises(ValueError, match="must be positive"):
        flow.register_engine(b"decode", "decode", [{"ssm_slot_capacity": 0}])


def test_partially_advertised_model_parallel_capacity_is_rejected():
    flow = DisaggStateFlowControl()

    with pytest.raises(ValueError, match="missing from part"):
        flow.register_engine(b"decode", "decode", [{"ssm_slot_capacity": 4}, {"global_rank": 1}])


def test_prefill_reservations_use_advertised_handoff_bound_and_fifo_queue():
    flow = DisaggStateFlowControl()
    flow.register_engine(
        b"prefill", "prefill", [{"ssm_slot_capacity": 1, "ssm_handoff_max_slots": 1}]
    )
    prompt = list(range(513))
    slot_cost = flow.prefill_slot_cost(b"prefill")

    assert slot_cost == 1
    assert flow.try_reserve_prefill(b"prefill", 1, slot_cost, max_requests=32)
    assert not flow.try_reserve_prefill(b"prefill", 2, slot_cost, max_requests=32)
    flow.enqueue_prefill(b"prefill", 2, prompt, {}, slot_cost)
    assert flow.pop_next_prefill(b"prefill", max_requests=32) is None

    assert flow.release_prefill(1) == b"prefill"
    admitted = flow.pop_next_prefill(b"prefill", max_requests=32)

    assert admitted.request_id == 2
    assert flow.prefill_usage(b"prefill") == 1
