# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from types import SimpleNamespace

import msgpack
import pytest

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.disaggregation.coordinator_runtime import DisaggCoordinatorRuntime
from megatron.core.inference.headers import Headers


def _runtime(*, request_capacity=32, backend="nixl", ssm_capacity=None):
    sent = []
    coordinator = SimpleNamespace(
        identities_of_data_parallel_ranks=deque(),
        request_id_to_client_id={5: b"client", 6: b"client"},
        request_id_to_client_request_id={5: 50, 6: 60},
        client_request_to_request_id={(b"client", 50): 5, (b"client", 60): 6},
        router_socket=SimpleNamespace(
            send_multipart=lambda message: sent.append((b"client", message))
        ),
        prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
        prefix_caching_routing_alpha=0.5,
        identity_to_rank_index={},
        hash_updates=[],
    )

    def register_identity(identity):
        coordinator.identity_to_rank_index.setdefault(
            identity, len(coordinator.identity_to_rank_index)
        )

    coordinator._register_rank_identity = register_identity
    coordinator.compute_request_hashes = lambda prompt: list(prompt)
    coordinator._update_rank_hashes = lambda identity, hashes: coordinator.hash_updates.append(
        (identity, hashes)
    )
    coordinator._match_vector = lambda _hashes: (
        [0.0] * len(coordinator.identity_to_rank_index),
        [0.0] * len(coordinator.identity_to_rank_index),
    )
    coordinator._send_to_engine = lambda identity, payload, **_kwargs: (
        sent.append((identity, msgpack.unpackb(payload, raw=False))) or True
    )
    runtime = DisaggCoordinatorRuntime(coordinator)
    coordinator.disagg = runtime
    coordinator._remove_engine = runtime.remove_engine

    common_meta = {
        "request_capacity": request_capacity,
        "tokens_per_block": 256,
        "element_size": 2,
        "head_dim": 128,
        "num_layers_global": 4,
        "num_kv_heads_global": 8,
    }
    prefill_meta = {"global_rank": 0, **common_meta}
    decode_meta = {"global_rank": 1, **common_meta}
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
    assert runtime.coordinator.hash_updates == [(b"prefill", [1, 2, 3]), (b"decode", [1, 2, 3])]


def test_prompt_log_probs_are_rejected_before_prefill():
    runtime, sent = _runtime()

    runtime.route_submit(5, [1, 2, 3], {"return_log_probs": True, "skip_prompt_log_probs": False})

    assert all(identity not in (b"prefill", b"decode") for identity, _ in sent)
    response = msgpack.unpackb(sent[-1][1][1], raw=False)
    assert Headers(response[0]) == Headers.REQUEST_ERROR
    assert "prompt log probabilities" in response[2]


@pytest.mark.parametrize(
    "handoff",
    [
        {"kv_meta": {"agent": "prefill"}},
        {"block_ids": [4]},
        {"kv_meta": {"agent": "prefill"}, "block_ids": "4"},
    ],
)
def test_malformed_prefill_handoff_fails_the_request(handoff):
    runtime, sent = _runtime()
    runtime.route_submit(5, [1], {})
    sent.clear()

    runtime.handle_prefill_done(5, {"disaggregated_params": handoff})

    client_frame = next(message for identity, message in sent if identity == b"client")
    response = msgpack.unpackb(client_frame[1], raw=False)
    assert Headers(response[0]) == Headers.REQUEST_ERROR
    assert "invalid handoff metadata" in response[2]
    assert 5 not in runtime.requests


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
    assert sent[-1][1][2:] == [runtime.engine_metadata[b"decode"], 1]


def test_registration_rejects_mixed_transports():
    runtime, _ = _runtime()

    with pytest.raises(ValueError, match="same state transport"):
        runtime.register_engine(b"decode-2", "decode", "nccl", runtime.engine_metadata[b"decode"])


def test_registration_rejects_incompatible_transfer_geometry():
    runtime, _ = _runtime()
    incompatible = dict(runtime.engine_metadata[b"decode"][0])
    incompatible["head_dim"] = 64

    with pytest.raises(ValueError, match="incompatible transfer geometry"):
        runtime.register_engine(b"decode-2", "decode", "nixl", [incompatible])


def test_reconnection_replaces_stale_engine_accounting():
    runtime, _ = _runtime(request_capacity=1)
    assert runtime.scheduler.try_reserve(b"decode", 99, 0)

    runtime.register_engine(b"decode", "decode", "nixl", runtime.engine_metadata[b"decode"])

    assert runtime.scheduler.decode_load(b"decode") == (0, 0, 0)
    assert runtime.scheduler.try_reserve(b"decode", 100, 0)


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
    assert runtime.scheduler.reserved_engine("prefill", 6) == b"prefill"


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

    assert runtime.scheduler.reserved_engine("prefill", 5) == b"prefill"
    assert sent == []


def test_decode_ssm_capacity_is_released_on_generation_completion():
    runtime, sent = _runtime(ssm_capacity=1)
    for request_id in (5, 6):
        runtime.route_submit(request_id, [request_id], {})
    sent.clear()
    handoff = {"kv_meta": {"ssm": {"positions": [0]}}, "block_ids": [4]}
    runtime.handle_prefill_done(5, {"request_id": 5, "disaggregated_params": handoff})
    runtime.handle_kv_read_done(b"decode", 5)
    runtime.handle_prefill_done(6, {"request_id": 6, "disaggregated_params": handoff})

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
    assert runtime.scheduler.reserved_engine("prefill", 5) is None
    assert 5 not in runtime.terminating_request_ids


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

    assert runtime.scheduler.reserved_engine("prefill", 5) == b"prefill"
    assert runtime.scheduler.prefill_load(b"prefill") == (0, 1, 1)
    assert 5 in runtime.terminating_request_ids

    runtime.handle_kv_read_done(b"decode", 5)
    assert runtime.scheduler.reserved_engine("prefill", 5) is None
    assert runtime.scheduler.prefill_load(b"prefill") == (0, 0, 0)

    runtime.handle_engine_aborted(5, source_safe=True)
    assert runtime.scheduler.assigned_engine("decode", 5) is None
    assert 5 not in runtime.terminating_request_ids


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

    assert runtime.scheduler.reserved_engine("prefill", 5) == b"prefill"
    assert runtime.scheduler.prefill_load(b"prefill") == (0, 1, 1)
    assert not any(
        identity == b"prefill" and Headers(message[0]) == Headers.RELEASE_KV
        for identity, message in sent
    )


def test_undelivered_decode_handoff_releases_prefill_source():
    runtime, sent = _runtime(ssm_capacity=1)
    runtime.route_submit(5, [1], {})

    def reject_decode(identity, payload, *, remove_unreachable=True):
        if identity == b"decode":
            assert not remove_unreachable
            return False
        sent.append((identity, msgpack.unpackb(payload, raw=False)))
        return True

    runtime.coordinator._send_to_engine = reject_decode
    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"ssm": {"positions": [0]}}, "block_ids": [4]},
        },
    )

    assert runtime.scheduler.reserved_engine("prefill", 5) is None
    assert runtime.scheduler.prefill_load(b"prefill") == (0, 0, 0)
    assert runtime.scheduler.assigned_engine("decode", 5) is None
    assert 5 not in runtime.terminating_request_ids
    assert 5 not in runtime.coordinator.request_id_to_client_id
    response = msgpack.unpackb(sent[-1][1][1], raw=False)
    assert Headers(response[0]) == Headers.REQUEST_ERROR
    assert response[3] is True


def test_engine_removal_after_kv_read_preserves_source_safety():
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

    def reject_prefill_release(identity, payload, *, remove_unreachable=True):
        message = msgpack.unpackb(payload, raw=False)
        if identity == b"prefill" and Headers(message[0]) == Headers.RELEASE_KV:
            assert remove_unreachable
            runtime.remove_engine(identity)
            return False
        sent.append((identity, message))
        return True

    runtime.coordinator._send_to_engine = reject_prefill_release
    runtime.handle_kv_read_done(b"decode", 5)

    assert 5 in runtime.requests
    assert runtime.scheduler.assigned_engine("decode", 5) == b"decode"
    assert 5 in runtime.coordinator.request_id_to_client_id
    assert 5 not in runtime.terminating_request_ids
    assert sent == []

    runtime.remove_engine(b"decode")

    assert 5 not in runtime.requests
    assert 5 not in runtime.coordinator.request_id_to_client_id
    assert 5 not in runtime.terminating_request_ids
    response = msgpack.unpackb(sent[-1][1][1], raw=False)
    assert Headers(response[0]) == Headers.REQUEST_ERROR
    assert response[3] is True


def test_late_source_safety_releases_prefill_after_request_failure():
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

    runtime.handle_engine_failure(5, "transfer failed", source_safe=False)

    assert runtime.scheduler.reserved_engine("prefill", 5) == b"prefill"
    assert runtime.scheduler.prefill_load(b"prefill") == (0, 1, 1)

    runtime.handle_kv_read_done(b"decode", 5)

    assert runtime.scheduler.reserved_engine("prefill", 5) is None
    assert runtime.scheduler.prefill_load(b"prefill") == (0, 0, 0)

    runtime.handle_engine_aborted(5, source_safe=True)

    assert 5 not in runtime.coordinator.request_id_to_client_id
    assert any(
        identity == b"prefill" and Headers(message[0]) == Headers.RELEASE_KV
        for identity, message in sent
    )
    assert any(
        identity == b"client"
        and Headers(msgpack.unpackb(message[1], raw=False)[0]) == Headers.REQUEST_ABORTED
        for identity, message in sent
    )


def test_serialized_handoff_releases_prompt_storage():
    runtime, _ = _runtime()
    runtime.route_submit(5, [1, 2, 3], {})

    runtime.handle_prefill_done(
        5,
        {
            "request_id": 5,
            "disaggregated_params": {"kv_meta": {"agent": "prefill"}, "block_ids": [4]},
        },
    )

    assert runtime.requests[5].prompt is None
    assert runtime.requests[5].sampling_params == {}
    assert runtime.requests[5].block_hashes == []


def test_load_balanced_routing_uses_free_capacity_independent_of_prefix_alpha():
    runtime, sent = _runtime()
    runtime.register_engine(b"prefill-2", "prefill", "nixl", runtime.engine_metadata[b"prefill"])
    runtime.coordinator.prefix_caching_coordinator_policy = (
        PrefixCachingCoordinatorPolicy.LOAD_BALANCED
    )
    runtime.coordinator.prefix_caching_routing_alpha = 1.0
    assert runtime.scheduler.try_reserve_prefill(b"prefill", 99, 0)

    runtime.route_submit(5, [1], {})

    assert sent[-1][0] == b"prefill-2"


def test_prefix_affinity_routes_and_is_computed_once():
    runtime, sent = _runtime()
    runtime.register_engine(b"prefill-2", "prefill", "nixl", runtime.engine_metadata[b"prefill"])
    calls = 0

    def counted_match_vector(_hashes):
        nonlocal calls
        calls += 1
        return [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]

    runtime.coordinator._match_vector = counted_match_vector
    runtime.route_submit(5, [1], {})

    assert calls == 1
    assert sent[-1][0] == b"prefill-2"
