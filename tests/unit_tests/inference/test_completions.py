# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Payload-stager plumbing shared by /v1/completions and /v1/chat/completions.

Two contracts are pinned here for both endpoints: the stager's response metadata
(``payload_stage_metadata`` on each reply) is surfaced once at the top level of
the JSON body, and client-supplied ``offload_params`` cannot carry engine-owned
``_``-prefixed control keys.
"""

import asyncio

import pytest

quart = pytest.importorskip("quart")
from quart import Quart

from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    bp as chat_completions_blueprint,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    attach_stage_metadata,
    collect_stage_metadata,
    validate_offload_params,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions import (
    bp as completions_blueprint,
)

pytestmark = [pytest.mark.internal]

_CHAT_PATH = "/v1/chat/completions"
_COMPLETIONS_PATH = "/v1/completions"
_CHAT_BODY = {"messages": [{"role": "user", "content": "hello"}]}
_COMPLETIONS_BODY = {"prompt": "hello"}


class _Tokenizer:
    chat_template = "test-template"
    bos = None
    eos_id = 2
    eod = 2

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [10, 11]

    def tokenize(self, prompt):
        del prompt
        return [10, 11]

    def detokenize(self, token_ids, **kwargs):
        return "".join(f"<{tok}>" for tok in token_ids)


def _reply(uid, prompt_tokens, generated_tokens, **extra):
    """A completed, non-serialized reply dict of the shape the coordinator forwards."""
    reply = {
        "uid": uid,
        "status": "COMPLETED",
        "events": [],
        "prompt_tokens": list(prompt_tokens),
        "prompt_length": len(prompt_tokens),
        "generated_tokens": list(generated_tokens),
        "generated_log_probs": None,
        "routing_indices": None,
        "num_cached_tokens": 0,
        "sampling_params": {"num_tokens_to_generate": 16},
    }
    reply.update(extra)
    return reply


class _ReplyingClient:
    """Records each submission and answers it with the next canned reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.offload_params = []
        self.aborted = []

    def add_request_with_id(
        self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
    ):
        del prompt_tokens, sampling_params, multi_modal_data
        self.offload_params.append(offload_params)
        request_id = len(self.offload_params)
        future = asyncio.get_running_loop().create_future()
        future.set_result(self.replies.pop(0))
        return request_id, future

    def abort_request(self, request_id):
        self.aborted.append(request_id)


def _build_app(blueprint, client):
    app = Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=_Tokenizer(),
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(),
        eval_mode=True,
    )
    app.register_blueprint(blueprint)
    return app


_ENDPOINTS = [
    pytest.param(chat_completions_blueprint, _CHAT_PATH, _CHAT_BODY, id="chat"),
    pytest.param(completions_blueprint, _COMPLETIONS_PATH, _COMPLETIONS_BODY, id="completions"),
]


# --- stage metadata ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("blueprint", "path", "body"), _ENDPOINTS)
async def test_stage_metadata_is_surfaced_at_the_top_level(blueprint, path, body):
    """An offloaded reply strips the log probs but hands the client the stager's handle."""
    client = _ReplyingClient(
        [
            _reply(
                "req-0",
                [10, 11],
                [12, 13],
                payload_offloaded=True,
                payload_stage_metadata={"store_key": "abc"},
            )
        ]
    )
    app = _build_app(blueprint, client)

    response = await app.test_client().post(path, json={**body, "logprobs": 1})

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["store_key"] == "abc"
    assert payload["id"] == "req-0"
    assert payload["choices"][0]["logprobs"] is None
    assert "generation_log_probs" not in payload["choices"][0].get("message", payload["choices"][0])


@pytest.mark.asyncio
async def test_completions_batch_merges_identical_stage_metadata():
    """Every prompt in a batch goes through the same stager; identical values merge once."""
    client = _ReplyingClient(
        [
            _reply("req-0", [10, 11], [12], payload_stage_metadata={"store_key": "abc"}),
            _reply("req-1", [10, 11], [13], payload_stage_metadata={"store_key": "abc"}),
        ]
    )
    app = _build_app(completions_blueprint, client)

    response = await app.test_client().post(_COMPLETIONS_PATH, json={"prompt": ["a", "b"]})

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["store_key"] == "abc"
    assert len(payload["choices"]) == 2


@pytest.mark.asyncio
async def test_completions_batch_rejects_conflicting_stage_metadata():
    client = _ReplyingClient(
        [
            _reply("req-0", [10, 11], [12], payload_stage_metadata={"store_key": "abc"}),
            _reply("req-1", [10, 11], [13], payload_stage_metadata={"store_key": "xyz"}),
        ]
    )
    app = _build_app(completions_blueprint, client)

    # Quart's test client turns the handler's ValueError into a 500; the message
    # itself is pinned by test_collect_stage_metadata_rejects_conflicting_values.
    response = await app.test_client().post(_COMPLETIONS_PATH, json={"prompt": ["a", "b"]})

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_completions_without_stager_has_no_extra_top_level_keys():
    client = _ReplyingClient([_reply("req-0", [10, 11], [12, 13])])
    app = _build_app(completions_blueprint, client)

    response = await app.test_client().post(_COMPLETIONS_PATH, json=_COMPLETIONS_BODY)

    assert response.status_code == 200
    payload = await response.get_json()
    assert set(payload) == {"id", "object", "created", "model", "choices", "usage"}


def test_collect_stage_metadata_tolerates_missing_and_none():
    response_metadata = {}
    collect_stage_metadata(response_metadata, {})
    collect_stage_metadata(response_metadata, {"payload_stage_metadata": None})
    assert response_metadata == {}


def test_collect_stage_metadata_rejects_conflicting_values():
    response_metadata = {}
    collect_stage_metadata(response_metadata, {"payload_stage_metadata": {"store_key": "abc"}})
    with pytest.raises(ValueError, match="conflicting response metadata for 'store_key'"):
        collect_stage_metadata(response_metadata, {"payload_stage_metadata": {"store_key": "xyz"}})


def test_attach_stage_metadata_refuses_to_overwrite_reserved_fields():
    with pytest.raises(ValueError, match=r"reserved fields: \['choices', 'id'\]"):
        attach_stage_metadata({"id": "x", "choices": []}, {"id": "y", "choices": [], "k": 1})


# --- offload_params validation ---------------------------------------------


@pytest.mark.parametrize(
    ("offload_params", "expected"),
    [
        (None, None),
        ({}, None),
        ({"store": "x", "ng_capture": {"rollout_id": "r0"}}, None),
        ("not-a-dict", "'offload_params' must be an object"),
        ([("_k", 1)], "'offload_params' must be an object"),
        (
            {"_request_prompt_preparation_error": "boom"},
            "'offload_params' keys starting with '_' are reserved: "
            "['_request_prompt_preparation_error']",
        ),
        (
            {"store": "x", "_a": 1, "_b": 2},
            "'offload_params' keys starting with '_' are reserved: ['_a', '_b']",
        ),
    ],
)
def test_validate_offload_params(offload_params, expected):
    assert validate_offload_params(offload_params) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("blueprint", "path", "body"), _ENDPOINTS)
@pytest.mark.parametrize(
    "offload_params",
    [{"_anything": 1}, {"_request_prompt_preparation_error": "boom"}, {"store": "x", "_a": 1}],
)
async def test_underscore_offload_params_are_rejected_before_submission(
    blueprint, path, body, offload_params
):
    client = _ReplyingClient([])
    app = _build_app(blueprint, client)

    response = await app.test_client().post(path, json={**body, "offload_params": offload_params})

    assert response.status_code == 400
    text = await response.get_data(as_text=True)
    assert "'offload_params' keys starting with '_' are reserved" in text
    assert client.offload_params == []  # nothing reached the engine


@pytest.mark.asyncio
@pytest.mark.parametrize(("blueprint", "path", "body"), _ENDPOINTS)
async def test_non_object_offload_params_are_rejected(blueprint, path, body):
    client = _ReplyingClient([])
    app = _build_app(blueprint, client)

    response = await app.test_client().post(path, json={**body, "offload_params": "x"})

    assert response.status_code == 400
    assert "'offload_params' must be an object" in await response.get_data(as_text=True)
    assert client.offload_params == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("blueprint", "path", "body"), _ENDPOINTS)
async def test_plain_offload_params_are_forwarded_unchanged(blueprint, path, body):
    offload_params = {"store": "x", "nested": {"key": [1, 2]}}
    client = _ReplyingClient([_reply("req-0", [10, 11], [12, 13])])
    app = _build_app(blueprint, client)

    response = await app.test_client().post(path, json={**body, "offload_params": offload_params})

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.offload_params == [offload_params]
