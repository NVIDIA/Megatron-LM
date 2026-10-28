# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import threading

import pytest

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPayloadStageResult,
)
from megatron.rl.gym_capture import (
    NG_COMMIT_COORDS_FIELD,
    GymPromptPreparer,
    PrefixStoreClient,
    PrefixStoreServer,
    replace_prefix_tokens,
)
from megatron.rl.rollout_bank import LedgerPayloadStager

CHAIN_HASH = "1" * 64


def _admission(**fields):
    admission = {
        "capture_key": "r",
        "model_call_id": "c2",
        "mode": "token_in",
        "parent_call_id": "c1",
        "prev_len": 5,
        "staging_chain": ["u1"],
        "parent_chain_hash": CHAIN_HASH,
    }
    admission.update(fields)
    return admission


@pytest.fixture
def store():
    server = PrefixStoreServer(host="127.0.0.1")
    client = PrefixStoreClient(server.address, timeout_ms=2000)
    yield server, client
    client.close()
    server.close()


@pytest.mark.parametrize(
    "prefix, expected",
    [
        pytest.param([10, 11, 7, 2], [10, 11, 7, 2, 12], id="exact-turn-ends-with-eos"),
        pytest.param([10, 11, 7], [10, 11, 7, 2, 12], id="cut-turn-takes-the-rendered-eos"),
    ],
)
def test_replace_prefix_tokens_keeps_the_exact_turn_and_the_new_suffix(prefix, expected):
    assert replace_prefix_tokens(prefix, [10, 11, 2], [10, 11, 2, 12], [2]) == expected
    with pytest.raises(ValueError, match="EOS"):
        replace_prefix_tokens(prefix, [10, 11], [10, 11, 12], [2])


def test_store_survives_bad_frames_and_clients_time_out(store):
    server, client = store
    # The launch wiring tests the server for None, never for truthiness, and pings it once.
    assert server is not None and bool(server) and server.size() == 0
    client.ping()
    dead = PrefixStoreClient("tcp://127.0.0.1:1", timeout_ms=200)
    with pytest.raises(RuntimeError, match="failed"):
        dead.ping()
    dead.close()  # closing after a failed exchange, or twice, must not hang or raise
    dead.close()
    # One socket serves every stager thread and the engine loop; exchanges serialize cleanly.
    threads = [
        threading.Thread(target=lambda i=i: client.put(f"t{i}", [i], [i, i])) for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert server.size() == 8 and client.get("t3") == ([3], [3, 3])
    client.put("u1", [1, 2], [3])
    assert client.get("u1") == ([1, 2], [3]) and client.get("absent") is None
    with pytest.raises(RuntimeError, match="malformed"):
        client._request("bogus-op")
    assert client.get("u1") == ([1, 2], [3])  # the server kept serving after the bad frame
    client.forget(["u1", "never", *(f"t{i}" for i in range(8))])
    assert server.size() == 0


def test_preparer_resolves_the_parent_from_the_store_and_inlines_the_prefix(store):
    server, client = store
    preparer = GymPromptPreparer(client)
    client.put("u1", [1, 2], [3, 2])
    params = {
        "ng_capture": _admission(prev_len=4),
        PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [1, 2, 9, 2],
        PREFIX_EOS_TOKEN_ID_FIELD: [2],
    }
    result = preparer.prepare_prompt([1, 2, 9, 2, 5], offload_params=params)
    assert result.prompt == [1, 2, 3, 2, 5]
    assert result.offload_params["ng_capture"]["required_prefix_token_ids"] == [1, 2, 3, 2]
    assert params["ng_capture"].get("required_prefix_token_ids") is None  # input left alone
    root = _admission(
        mode="text", parent_call_id=None, prev_len=0, staging_chain=[], parent_chain_hash=None
    )
    for passthrough in ({}, {"ng_capture": root}):
        result = GymPromptPreparer(client).prepare_prompt([7, 8], offload_params=passthrough)
        assert (result.prompt, result.offload_params) == ([7, 8], passthrough)
    failures = [
        (
            {**params, "ng_capture": _admission(staging_chain=["absent"])},
            LookupError,
            "does not hold",
        ),
        ({**params, "ng_capture": _admission(prev_len=3)}, ValueError, "expects a prefix of 3"),
        ({"ng_capture": params["ng_capture"]}, ValueError, "splice metadata"),
        ({**params, PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [1, 9]}, ValueError, "EOS"),
    ]
    for offload_params, error, match in failures:
        with pytest.raises(error, match=match):
            GymPromptPreparer(client).prepare_prompt([1, 2, 9, 2, 5], offload_params=offload_params)
    client.forget(["u1"])
    # Cached from the first resolution: the forgotten parent still serves this preparer.
    assert preparer.prepare_prompt([1, 2, 9, 2, 5], offload_params=params).prompt == [1, 2, 3, 2, 5]


@pytest.mark.skipif(importlib.util.find_spec("nemo_gym") is None, reason="needs nemo_gym")
@pytest.mark.parametrize(
    "admission, expected",
    [
        pytest.param(
            {"capture_key": "r", "model_call_id": "c1", "mode": "text"},
            ("staged", 0, 5, "u9"),
            id="root",
        ),
        pytest.param(
            _admission(prev_len=3, required_prefix_token_ids=[1, 2, 3]),
            ("staged", 3, 5, "u9"),
            id="exact-prefix",
        ),
        pytest.param(_admission(prev_len=3), ("capture_failed", 3, 3, None), id="unresolved-chain"),
        pytest.param(
            {"capture_key": "r", "model_call_id": "c1", "mode": "no-such-mode"},
            ("capture_failed", 0, 0, None),
            id="unknown-schema",
        ),
    ],
)
def test_stager_answers_gym_admissions_with_commit_coordinates(store, admission, expected):
    server, client = store
    stager = LedgerPayloadStager(prefix_store=client)
    # Spans are (token position, epoch): admitted under epoch 3, continued under epoch 4.
    record = FinishedRequestRecord(
        policy_epoch=[(0, 3), (2, 4)], kv_cache_epoch=None, num_evictions=0
    )
    payload = OffloadedRequestPayload(
        prompt_token_ids=[1, 2, 3, 6],
        generated_token_ids=[7],
        generated_log_probs=[-0.5],
        prompt_log_probs=None,
        routing_indices=None,
    )
    result = stager.stage(
        "u9", payload, finished_metadata=record, offload_params={"ng_capture": admission}
    )
    coords = result.response_metadata[NG_COMMIT_COORDS_FIELD]
    assert (
        coords["disposition"],
        coords["prev_len"],
        coords["cum_len"],
        coords.get("staging_key"),
    ) == expected
    assert (coords["capture_key"], coords["weight_version"]) == ("r", 3)
    assert client.get("u9") == ([1, 2, 3, 6], [7])  # published for the next turn's prefix
    # Without an admission the stager only takes custody and publishes nothing.
    assert stager.stage("u10", payload, finished_metadata=record) == RequestPayloadStageResult()
    assert server.size() == 1
    # A store that stopped answering poisons the capture instead of raising out of stage():
    # the record stays indexed and the reply stays token-free.
    dead = LedgerPayloadStager(prefix_store=PrefixStoreClient("tcp://127.0.0.1:1", timeout_ms=200))
    result = dead.stage(
        "u11", payload, finished_metadata=record, offload_params={"ng_capture": admission}
    )
    assert result.response_metadata[NG_COMMIT_COORDS_FIELD]["disposition"] == "capture_failed"
    assert set(dead.drain()) == {"u11"}
