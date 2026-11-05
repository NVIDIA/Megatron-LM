# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import sys
import threading

import pytest

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPayloadStageResult,
)
from megatron.rl import gym_capture
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
    # Each calling thread gets its own socket; concurrent puts never queue behind one exchange.
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


def test_stager_acknowledges_capture_failed_without_nemo_gym(monkeypatch, caplog):
    """Without nemo_gym (the CI container) every admitted call is acknowledged capture_failed in
    Gym's coordinate shape, the condition is logged once, and the records are still indexed."""
    cached = [name for name in sys.modules if name.partition(".")[0] == "nemo_gym"]
    for name in cached:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "nemo_gym", None)  # every nemo_gym import now fails
    monkeypatch.setattr("megatron.rl.gym_capture._GYM_MISSING_LOGGED", False)
    stager = LedgerPayloadStager()
    record = FinishedRequestRecord(policy_epoch=[(0, 3)], kv_cache_epoch=None, num_evictions=0)
    payload = OffloadedRequestPayload(
        prompt_token_ids=[1, 2],
        generated_token_ids=[3],
        generated_log_probs=[-0.5],
        prompt_log_probs=None,
        routing_indices=None,
    )
    with caplog.at_level("ERROR", logger="megatron.rl.gym_capture"):
        results = [
            stager.stage(
                uid, payload, finished_metadata=record, offload_params={"ng_capture": _admission()}
            )
            for uid in ("u1", "u2")
        ]
    coords = [result.response_metadata[NG_COMMIT_COORDS_FIELD] for result in results]
    assert coords[0] == {
        "capture_key": "r",
        "model_call_id": "c2",
        "parent_call_id": "c1",
        "prev_len": 5,
        "delta_len": 0,
        "cum_len": 5,
        "weight_version": 3,
        "disposition": "capture_failed",
    }
    assert coords[1] == coords[0]
    assert sum("nemo_gym is not importable" in rec.message for rec in caplog.records) == 1
    assert set(stager.drain()) == {"u1", "u2"}


def test_stager_ignores_a_malformed_admission(caplog):
    """A non-dict value under the admission key is not an admission: the record is indexed, the
    reply carries no coordinates, and the engine log says why."""
    stager = LedgerPayloadStager()
    record = FinishedRequestRecord(policy_epoch=[(0, 3)], kv_cache_epoch=None, num_evictions=0)
    payload = OffloadedRequestPayload(
        prompt_token_ids=[1],
        generated_token_ids=[2],
        generated_log_probs=[-0.5],
        prompt_log_probs=None,
        routing_indices=None,
    )
    with caplog.at_level("WARNING", logger="megatron.rl.rollout_bank"):
        result = stager.stage(
            "u1", payload, finished_metadata=record, offload_params={"ng_capture": "bogus"}
        )
    assert result == RequestPayloadStageResult() and set(stager.drain()) == {"u1"}
    assert any("malformed" in rec.message for rec in caplog.records)


def test_preparer_honors_the_multimodal_prefix_contract(store):
    """When the endpoint stamps a media count, a spliced token_in prefix reports its expanded
    length so the engine expands only the suffix, and a Gym text root drops the count so the
    engine expands every placeholder itself; a request without an admission is left alone."""
    server, client = store
    client.put("u1", [1, 2], [3, 2])
    params = {
        "ng_capture": _admission(prev_len=4),
        PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [1, 2, 9, 2],
        PREFIX_EOS_TOKEN_ID_FIELD: [2],
        PREFIX_MEDIA_COUNT_FIELD: 1,
    }
    result = GymPromptPreparer(client).prepare_prompt([1, 2, 9, 2, 5], offload_params=params)
    assert result.prompt == [1, 2, 3, 2, 5]
    assert result.offload_params[PREFIX_EXPANDED_TOKEN_COUNT_FIELD] == 4
    assert result.offload_params[PREFIX_MEDIA_COUNT_FIELD] == 1
    root = _admission(
        mode="text", parent_call_id=None, prev_len=0, staging_chain=[], parent_chain_hash=None
    )
    result = GymPromptPreparer(client).prepare_prompt(
        [7, 8], offload_params={"ng_capture": root, PREFIX_MEDIA_COUNT_FIELD: 1}
    )
    assert (result.prompt, result.offload_params) == ([7, 8], {"ng_capture": root})
    plain = {PREFIX_MEDIA_COUNT_FIELD: 1}
    result = GymPromptPreparer(client).prepare_prompt([7, 8], offload_params=plain)
    assert (result.prompt, result.offload_params) == ([7, 8], plain)


@pytest.mark.skipif(importlib.util.find_spec("nemo_gym") is None, reason="needs nemo_gym")
def test_copied_gym_wire_names_match_the_installed_gym():
    """The constants and dict keys gym_capture copies from Gym still match the installed Gym.
    Runs only where nemo_gym is importable (not in Megatron CI)."""
    try:
        from nemo_gym.models.token_capture import NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD
        from nemo_gym.models.token_capture.staging.records import CaptureAdmission, CommitCoords
    except ImportError:  # nemo_gym before 0.8.0 shipped the package as token_id_capture
        from nemo_gym.token_id_capture import NG_CAPTURE_FIELD, NG_COMMIT_COORDS_FIELD
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission, CommitCoords

    assert (gym_capture.NG_CAPTURE_FIELD, gym_capture.NG_COMMIT_COORDS_FIELD) == (
        NG_CAPTURE_FIELD,
        NG_COMMIT_COORDS_FIELD,
    )
    # Every key the preparer reads off an admission is a Gym admission field.
    assert {"mode", "required_prefix_token_ids", "staging_chain", "prev_len"} <= set(
        CaptureAdmission.model_fields
    )
    # The hand-built capture_failed acknowledgement is a CommitCoords Gym accepts.
    failed = gym_capture._failed_coordinates(_admission(prev_len=3), 3)
    assert set(failed) <= set(CommitCoords.model_fields)
    parsed = CommitCoords.model_validate(failed)
    assert parsed.disposition == "capture_failed"
    assert parsed.model_dump(mode="json").items() >= failed.items()
