# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from types import SimpleNamespace

import msgpack
import pytest

from megatron.core.inference.disaggregation.coordinator_runtime import DisaggCoordinatorRuntime
from megatron.core.inference.headers import Headers


def _runtime(*, request_capacity=32, backend="nixl", ssm_capacity=None):
    sent = []
    coordinator = SimpleNamespace(
        identities_of_data_parallel_ranks=deque(),
        request_id_to_client_id={5: b"client", 6: b"client"},
        request_id_to_client_request_id={5: 50, 6: 60},
        client_request_to_request_id={(b"client", 50): 5, (b"client", 60): 6},
        _register_rank_identity=lambda identity: None,
        router_socket=SimpleNamespace(
            send_multipart=lambda message: sent.append((b"client", message))
        ),
    )
    coordinator._send_to_engine = lambda identity, payload: (
        sent.append((identity, msgpack.unpackb(payload, raw=False))) or True
    )
    runtime = DisaggCoordinatorRuntime(coordinator, "round_robin")
    coordinator.disagg = runtime

    prefill_meta = {"global_rank": 0, "request_capacity": request_capacity}
    decode_meta = {"global_rank": 1, "request_capacity": request_capacity}
    if ssm_capacity is not None:
        for meta in (prefill_meta, decode_meta):
            meta["ssm_slot_capacity"] = ssm_capacity
            meta["ssm_handoff_max_slots"] = 1
    runtime.register_engine(b"prefill", "prefill", backend, [prefill_meta])
    runtime.register_engine(b"decode", "decode", backend, [decode_meta])
    return runtime, sent


def test_request_routes_prefill_then_decode():
    runtime, sent = _runtime()
    sampling_params = {"temperature": 0.0, "return_log_probs": True, "skip_prompt_log_probs": True}
    runtime.route_submit(5, [1, 2, 3], sampling_params)

    identity, message = sent.pop()
    assert identity == b"prefill"
    assert Headers(message[0]) == Headers.SUBMIT_REQUEST
    assert message[3]["do_kv_handoff"] is True
    assert message[3]["num_tokens_to_generate"] == 0
    assert message[3]["return_log_probs"] is True
    assert message[3]["skip_prompt_log_probs"] is True

    handoff = {"kv_meta": {"agent": "prefill"}, "block_ids": [4, 5], "request_id": 5}
    runtime.handle_prefill_done(5, {"request_id": 5, "disaggregated_params": handoff})

    identity, message = sent.pop()
    assert identity == b"decode"
    assert Headers(message[0]) == Headers.SUBMIT_REQUEST_WITH_KV
    assert message[1:4] == [5, [1, 2, 3], sampling_params]
    assert message[4:] == [handoff["kv_meta"], handoff["block_ids"]]


def test_nccl_send_waits_for_decode_destinations():
    runtime, sent = _runtime(backend="nccl")
    runtime.route_submit(5, [1], {})
    sent.clear()
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"agent": "prefill"}, "block_ids": [4]},
        },
    )

    assert [(identity, Headers(message[0])) for identity, message in sent] == [
        (b"decode", Headers.SUBMIT_REQUEST_WITH_KV)
    ]

    runtime.handle_kv_transfer_ready(b"decode", 5, 1)

    assert sent[-1][0] == b"prefill"
    assert Headers(sent[-1][1][0]) == Headers.SEND_KV
    assert sent[-1][1][2:] == [[{"global_rank": 1, "request_capacity": 32}], 1]


def test_registration_rejects_mixed_transports():
    runtime, _ = _runtime()

    with pytest.raises(ValueError, match="same state transport"):
        runtime.register_engine(
            b"decode-2", "decode", "nccl", [{"global_rank": 2, "request_capacity": 32}]
        )


def test_read_done_releases_prefill_and_admits_queued_request():
    runtime, sent = _runtime(request_capacity=1)
    runtime.route_submit(5, [1], {})
    runtime.route_submit(6, [2], {})
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"agent": "prefill"}, "block_ids": [4]},
        },
    )
    sent.clear()

    runtime.handle_kv_read_done(b"decode", 5)

    assert [(identity, Headers(message[0])) for identity, message in sent] == [
        (b"prefill", Headers.RELEASE_KV),
        (b"prefill", Headers.SUBMIT_REQUEST),
    ]
    assert runtime.prefill_by_request[6] == b"prefill"


def test_read_done_from_wrong_decode_does_not_release_prefill():
    runtime, sent = _runtime()
    runtime.route_submit(5, [1], {})
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"agent": "prefill"}, "block_ids": [4]},
        },
    )
    sent.clear()

    runtime.handle_kv_read_done(b"other-decode", 5)

    assert 5 in runtime.prefill_by_request
    assert sent == []


def test_decode_ssm_capacity_is_released_on_generation_completion():
    runtime, sent = _runtime(ssm_capacity=1)
    for request_id in (5, 6):
        runtime.route_submit(request_id, [request_id], {})
    sent.clear()
    handoff = {"kv_meta": {"ssm": {"positions": [0]}}, "block_ids": [4]}
    for request_id in (5, 6):
        runtime.handle_prefill_done(
            request_id, {"request_id": request_id, "disaggregated_params": handoff}
        )

    decode_submits = [
        message[1]
        for identity, message in sent
        if identity == b"decode" and Headers(message[0]) == Headers.SUBMIT_REQUEST_WITH_KV
    ]
    assert decode_submits == [5]

    runtime.handle_decode_done(5)

    decode_submits = [
        message[1]
        for identity, message in sent
        if identity == b"decode" and Headers(message[0]) == Headers.SUBMIT_REQUEST_WITH_KV
    ]
    assert decode_submits == [5, 6]


def test_active_decode_cancellation_waits_for_engine_safety():
    runtime, sent = _runtime()
    runtime.route_submit(5, [1], {})
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"agent": "prefill"}, "block_ids": [4]},
        },
    )
    sent.clear()

    runtime.abort_request(5)
    assert sent == [(b"decode", [Headers.ABORT_REQUEST.value, 5])]

    runtime.handle_engine_aborted(5, source_safe=True)
    engine_messages = [(identity, Headers(message[0])) for identity, message in sent[:-1]]
    assert engine_messages == [(b"decode", Headers.ABORT_REQUEST), (b"prefill", Headers.RELEASE_KV)]
    client_frames = sent[-1][1]
    assert client_frames[0] == b"client"
    assert Headers(msgpack.unpackb(client_frames[1], raw=False)[0]) == Headers.REQUEST_ABORTED
    assert 5 not in runtime.prefill_by_request
    assert 5 not in runtime.cancelled_request_ids


def test_unsafe_cancellation_keeps_prefill_capacity_until_read_completes():
    runtime, sent = _runtime(ssm_capacity=1)
    runtime.route_submit(5, [1], {})
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"ssm": {"positions": [0]}}, "block_ids": [4]},
        },
    )

    runtime.abort_request(5)
    runtime.handle_engine_aborted(5, source_safe=False)

    assert runtime.prefill_by_request[5] == b"prefill"
    assert runtime.flow.prefill_usage(b"prefill") == 1

    runtime.handle_kv_read_done(b"decode", 5)

    assert 5 not in runtime.prefill_by_request
    assert runtime.flow.prefill_usage(b"prefill") == 0
    assert runtime.router.decode_for_request(5) is None


def test_decode_removal_does_not_release_inflight_source():
    runtime, sent = _runtime(ssm_capacity=1)
    runtime.route_submit(5, [1], {})
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"ssm": {"positions": [0]}}, "block_ids": [4]},
        },
    )
    sent.clear()

    runtime.remove_engine(b"decode")

    assert runtime.prefill_by_request[5] == b"prefill"
    assert runtime.flow.prefill_usage(b"prefill") == 1
    assert not any(
        identity == b"prefill" and Headers(message[0]) == Headers.RELEASE_KV
        for identity, message in sent
    )
