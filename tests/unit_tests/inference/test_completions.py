# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HTTP-level tests for /v1/completions, plus the payload-stager plumbing both endpoints share.

The handler runs under Quart's test client against the fake inference client from
test_endpoints_common.py, which also pins the behavior the two endpoints share; chat-only
behavior is pinned in test_chat_completions.py.
"""

import pytest

pytest.importorskip("quart")

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    attach_stage_metadata,
    collect_stage_metadata,
    validate_offload_params,
)
from tests.unit_tests.inference.test_endpoints_common import (
    BODIES,
    CHAT_PATH,
    COMPLETIONS_BODY,
    COMPLETIONS_PATH,
    NOT_A_NUMBER_ERROR,
    PATHS,
    ReplyingClient,
    Tokenizer,
    build_app,
    completed_reply,
)

pytestmark = [pytest.mark.internal]


# --- stage metadata ---------------------------------------------------------


@pytest.mark.asyncio
@PATHS
async def test_stage_metadata_is_surfaced_at_the_top_level(path):
    """An offloaded reply strips the log probs but hands the client the stager's handle."""
    client = ReplyingClient(
        [
            completed_reply(
                "req-0",
                [10, 11],
                [12, 13],
                payload_offloaded=True,
                payload_stage_metadata={"store_key": "abc"},
            )
        ]
    )
    app = build_app(path, client)

    response = await app.test_client().post(
        path,
        json={
            **BODIES[path],
            "logprobs": 1,
            "return_tokenized_data": True,
            "return_raw_text": True,
        },
    )

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["store_key"] == "abc"
    assert payload["id"] == "req-0"
    choice = payload["choices"][0]
    assert choice["logprobs"] is None
    message = choice.get("message", choice)
    assert "generation_log_probs" not in message
    if path == CHAT_PATH:
        # The token echo fields travel with the offloaded payload, not the reply.
        assert set(message) == {"role", "content"}


@pytest.mark.asyncio
async def test_completions_batch_merges_identical_stage_metadata():
    """Every prompt in a batch goes through the same stager; identical values merge once."""
    client = ReplyingClient(
        [
            completed_reply("req-0", [10, 11], [12], payload_stage_metadata={"store_key": "abc"}),
            completed_reply("req-1", [10, 11], [13], payload_stage_metadata={"store_key": "abc"}),
        ]
    )
    app = build_app(COMPLETIONS_PATH, client)

    response = await app.test_client().post(COMPLETIONS_PATH, json={"prompt": ["a", "b"]})

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["store_key"] == "abc"
    assert len(payload["choices"]) == 2


@pytest.mark.asyncio
async def test_completions_batch_rejects_conflicting_stage_metadata():
    client = ReplyingClient(
        [
            completed_reply("req-0", [10, 11], [12], payload_stage_metadata={"store_key": "abc"}),
            completed_reply("req-1", [10, 11], [13], payload_stage_metadata={"store_key": "xyz"}),
        ]
    )
    app = build_app(COMPLETIONS_PATH, client)

    # Quart's test client turns the handler's ValueError into a 500; the message
    # itself is pinned by test_collect_stage_metadata_rejects_conflicting_values.
    response = await app.test_client().post(COMPLETIONS_PATH, json={"prompt": ["a", "b"]})

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_completions_without_stager_has_no_extra_top_level_keys():
    client = ReplyingClient([completed_reply("req-0", [10, 11], [12, 13])])
    app = build_app(COMPLETIONS_PATH, client)

    response = await app.test_client().post(COMPLETIONS_PATH, json=COMPLETIONS_BODY)

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
@PATHS
@pytest.mark.parametrize(
    "offload_params",
    [{"_anything": 1}, {"_request_prompt_preparation_error": "boom"}, {"store": "x", "_a": 1}],
)
async def test_underscore_offload_params_are_rejected_before_submission(path, offload_params):
    client = ReplyingClient([])
    app = build_app(path, client)

    response = await app.test_client().post(
        path, json={**BODIES[path], "offload_params": offload_params}
    )

    assert response.status_code == 400
    text = await response.get_data(as_text=True)
    assert "'offload_params' keys starting with '_' are reserved" in text
    assert client.offload_params == []  # nothing reached the engine


@pytest.mark.asyncio
@PATHS
async def test_non_object_offload_params_are_rejected(path):
    client = ReplyingClient([])
    app = build_app(path, client)

    response = await app.test_client().post(path, json={**BODIES[path], "offload_params": "x"})

    assert response.status_code == 400
    assert "'offload_params' must be an object" in await response.get_data(as_text=True)
    assert client.offload_params == []


@pytest.mark.asyncio
@PATHS
async def test_plain_offload_params_are_forwarded_unchanged(path):
    offload_params = {"store": "x", "nested": {"key": [1, 2]}}
    client = ReplyingClient([completed_reply("req-0", [10, 11], [12, 13])])
    app = build_app(path, client)

    response = await app.test_client().post(
        path, json={**BODIES[path], "offload_params": offload_params}
    )

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.offload_params == [offload_params]


# --- request validation -----------------------------------------------------


class _RaisingTokenizer(Tokenizer):
    def tokenize(self, prompt):
        raise ValueError(f"cannot tokenize {prompt!r}")


_INVALID_PROMPT_FORMAT = (
    "Invalid 'prompt' format. Must be str, list[str], list[int], or list[list[int]]"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "app_config", "status", "expected_error"),
    [
        pytest.param({}, {}, 400, "Missing 'prompt' field", id="missing-prompt"),
        pytest.param({"prompt": []}, {}, 400, "Missing 'prompt' field", id="empty-prompt-list"),
        pytest.param(
            {"prompt": 42},
            {},
            400,
            "Invalid 'prompt' type. Must be str or list",
            id="prompt-wrong-type",
        ),
        pytest.param({"prompt": [1.5]}, {}, 400, _INVALID_PROMPT_FORMAT, id="prompt-floats"),
        pytest.param({"prompt": ["a", 1]}, {}, 400, _INVALID_PROMPT_FORMAT, id="prompt-mixed-list"),
        # A tokenizer error is reported as an internal failure.
        pytest.param(
            {"prompt": "hello"},
            {"tokenizer": _RaisingTokenizer()},
            500,
            "Error tokenizing prompt: cannot tokenize 'hello'",
            id="tokenizer-error",
        ),
        # A sampling field of the wrong type is a client error when the conversion raises
        # ValueError; a TypeError (a list where a number is expected) escapes as a 500.
        pytest.param(
            {"prompt": "hello", "temperature": "hot"},
            {},
            400,
            NOT_A_NUMBER_ERROR,
            id="temperature-not-a-number",
        ),
        pytest.param({"prompt": "hello", "top_k": [1]}, {}, 500, None, id="top-k-list"),
        # Sampling fields are read with a plain .get(), so an explicit null reaches the
        # float()/int() conversion and escapes as a 500 (chat treats null as "use the default").
        pytest.param(
            {
                "prompt": "hello",
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_tokens": None,
                "streaming_interval": None,
            },
            {},
            500,
            None,
            id="null-sampling-fields",
        ),
    ],
)
async def test_malformed_requests_are_rejected_before_submission(
    body, app_config, status, expected_error
):
    client = ReplyingClient([])
    app = build_app(COMPLETIONS_PATH, client, **app_config)

    response = await app.test_client().post(COMPLETIONS_PATH, json=body)

    assert response.status_code == status
    if expected_error is not None:
        assert await response.get_data(as_text=True) == expected_error
    assert client.prompt_tokens == []  # nothing reached the engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_prompt_tokens", "expected_texts"),
    [
        pytest.param("hello", [[10, 11]], ["hello<12><13>"], id="string"),
        pytest.param(
            ["a", "b"], [[10, 11], [10, 11]], ["a<12><13>", "b<12><13>"], id="list-of-strings"
        ),
        pytest.param([5, 6], [[5, 6]], ["<5><6><12><13>"], id="token-ids"),
        pytest.param(
            [[5], [6, 7]], [[5], [6, 7]], ["<5><12><13>", "<6><7><12><13>"], id="list-of-token-ids"
        ),
    ],
)
async def test_completions_prompt_variants_are_tokenized_and_echoed(
    prompt, expected_prompt_tokens, expected_texts
):
    client = ReplyingClient()
    app = build_app(COMPLETIONS_PATH, client)

    response = await app.test_client().post(COMPLETIONS_PATH, json={"prompt": prompt, "echo": True})

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.prompt_tokens == expected_prompt_tokens
    payload = await response.get_json()
    assert [choice["text"] for choice in payload["choices"]] == expected_texts


# --- response formatting --------------------------------------------------------

_LOGPROBS_WITHOUT_ECHO = {
    "tokens": ["<12>", "<13>"],
    "token_logprobs": [None, -0.5, -9999.0],
    "top_logprobs": [None, {"<12>": -0.5}, {"<13>": -9999.0}],
    "text_offset": [0, 4],
}
_LOGPROBS_WITH_ECHO = {
    "tokens": ["<10>", "<11>", "<12>", "<13>"],
    "token_logprobs": [None, -1.0, -0.5, -9999.0],
    "top_logprobs": [None, {"<11>": -1.0}, {"<12>": -0.5}, {"<13>": -9999.0}],
    "text_offset": [0, 4, 8, 12],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("echo", "logprobs", "expected_logprobs"),
    [
        pytest.param(False, None, None, id="plain"),
        # Non-finite values are clamped for JSON; the leading None is the OpenAI first-token slot.
        pytest.param(False, 1, _LOGPROBS_WITHOUT_ECHO, id="logprobs"),
        pytest.param(True, 1, _LOGPROBS_WITH_ECHO, id="echo-with-logprobs"),
    ],
)
async def test_completions_response_format(echo, logprobs, expected_logprobs, recwarn):
    first = completed_reply(
        "req-0",
        [10, 11],
        [12, 13],
        num_cached_tokens=2,
        routing_indices=[7, 8, 9, 6],
        generated_log_probs=[-0.5, float("-inf")],
        generated_top_n_logprobs=[{"<12>": -0.5}, {"<13>": float("-inf")}],
        prompt_log_probs=[-1.0],
        prompt_top_n_logprobs=[{"<11>": -1.0}],
    )
    # Hit its token limit; without prompt_length the prompt count falls back to the token ids.
    second = completed_reply(
        "req-1", [10, 11, 12], [14, 15], sampling_params={"num_tokens_to_generate": 2}
    )
    del second["prompt_length"]
    client = ReplyingClient([first, second])
    app = build_app(COMPLETIONS_PATH, client)
    body = {"prompt": ["p0", "p1"], "echo": echo, **({"logprobs": logprobs} if logprobs else {})}

    response = await app.test_client().post(COMPLETIONS_PATH, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["id"] == "req-0"
    assert payload["object"] == "text_completion"
    assert payload["model"] == "EMPTY"
    assert isinstance(payload["created"], int)
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    choice_0, choice_1 = payload["choices"]
    assert (choice_0["index"], choice_1["index"]) == (0, 1)
    assert (choice_0["text"], choice_1["text"]) == (
        ("p0<12><13>", "p1<14><15>") if echo else ("<12><13>", "<14><15>")
    )
    assert (choice_0["finish_reason"], choice_1["finish_reason"]) == ("stop", "length")
    assert choice_0["logprobs"] == expected_logprobs
    assert (choice_1["logprobs"] is None) is (logprobs is None)
    assert choice_0["generation_log_probs"] == [-0.5, -9999.0]
    assert (choice_0["prompt_token_ids"], choice_0["generation_token_ids"]) == ([10, 11], [12, 13])
    assert (choice_0["moe_topk_indices"], choice_0["prompt_moe_topk_indices"]) == (
        [7, 8, 9, 6],
        [7, 8],
    )
    assert "moe_topk_indices" not in choice_1
    assert {key: choice_0[key] for key in ("acceptance_step_lengths", "ttft", "tpot")} == {
        "acceptance_step_lengths": None,
        "ttft": None,
        "tpot": None,
    }
    # The per-prompt SamplingParams copies must not re-feed the derived, deprecated mirror field.
    assert not [w for w in recwarn if "return_prompt_top_n_logprobs" in str(w.message)]
