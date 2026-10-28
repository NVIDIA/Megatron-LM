# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util

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
        "rollout_id": "r",
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
    yield server, PrefixStoreClient(server.address)
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
    client.forget(["u1"])
    assert len(server) == 0 and client.get("u1") is None
    # Cached from the first resolution: the forgotten parent still serves this preparer.
    assert preparer.prepare_prompt([1, 2, 9, 2, 5], offload_params=params).prompt == [1, 2, 3, 2, 5]
    untouched = [
        {},
        {
            "ng_capture": _admission(
                mode="text",
                parent_call_id=None,
                prev_len=0,
                staging_chain=[],
                parent_chain_hash=None,
            )
        },
        {**params, "ng_capture": _admission(staging_chain=["absent"])},
        {**params, "ng_capture": _admission(prev_len=3)},
        {"ng_capture": params["ng_capture"]},
    ]
    for offload_params in untouched:
        result = GymPromptPreparer(client).prepare_prompt([7, 8], offload_params=offload_params)
        assert (result.prompt, result.offload_params) == ([7, 8], offload_params)


@pytest.mark.skipif(importlib.util.find_spec("nemo_gym") is None, reason="needs nemo_gym")
@pytest.mark.parametrize(
    "admission, expected",
    [
        pytest.param(
            {"rollout_id": "r", "model_call_id": "c1", "mode": "text"},
            ("staged", 0, 5, "u9"),
            id="root",
        ),
        pytest.param(
            _admission(prev_len=3, required_prefix_token_ids=[1, 2, 3]),
            ("staged", 3, 5, "u9"),
            id="exact-prefix",
        ),
        pytest.param(_admission(prev_len=3), ("capture_failed", 3, 3, None), id="unresolved-chain"),
    ],
)
def test_stager_answers_gym_admissions_with_commit_coordinates(store, admission, expected):
    server, client = store
    stager = LedgerPayloadStager(prefix_store=client)
    record = FinishedRequestRecord(
        policy_epoch=[(3, 2), (2, 1)], kv_cache_epoch=None, num_evictions=0
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
        coords["staging_key"],
    ) == expected
    assert coords["weight_version"] == 2  # the oldest policy epoch that generated the call
    assert client.get("u9") == ([1, 2, 3, 6], [7])  # published for the next turn's prefix
    # Without an admission the stager only takes custody and publishes nothing.
    assert stager.stage("u10", payload, finished_metadata=record) == RequestPayloadStageResult()
    assert len(server) == 1
