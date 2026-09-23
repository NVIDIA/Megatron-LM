# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import base64
from unittest import mock

import pytest

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    compute_media_cache_key,
    serialize_multimodal_data,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _expanded_prefix_stitching_metadata,
    _extract_media_url_bytes,
    _has_previous_turn_tokens,
    _last_assistant_message,
    _replace_prefix_tokens,
    _replace_prefix_tokens_metadata,
    _sanitize_messages_for_template,
    _suffix_tokens_after_prefix,
    _tokenize_with_media_slots_sync,
)


def test_extract_media_data_url_accepts_payload_at_limit():
    payload = b"four"
    url = f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"

    assert _extract_media_url_bytes(url, max_bytes=len(payload)) == payload


def test_extract_media_data_url_rejects_decoded_payload_over_limit():
    # Four- and five-byte payloads both occupy eight base64 characters, so
    # this exercises the decoded-size check in addition to the encoded bound.
    payload = b"five!"
    url = f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"

    with pytest.raises(ValueError, match="data:video/mp4;base64 payload exceeds 4 byte limit"):
        _extract_media_url_bytes(url, max_bytes=4)


def test_replace_prefix_tokens_metadata_ships_the_rendered_prefix_and_eos():
    eos = 99
    template_prefix = (1, 99, 2, 99)
    offload_params = {"ng_capture": {"staging_chain": ["k1"]}}

    out = _replace_prefix_tokens_metadata(eos, template_prefix, offload_params)

    assert out[PREFIX_TEMPLATE_TOKEN_IDS_FIELD] == [1, 99, 2, 99]
    assert out[PREFIX_EOS_TOKEN_ID_FIELD] == [99]
    assert out["ng_capture"] == {"staging_chain": ["k1"]}
    assert offload_params == {"ng_capture": {"staging_chain": ["k1"]}}  # input not mutated


def test_expanded_prefix_stitching_metadata_marks_expanded_prefix():
    assert _expanded_prefix_stitching_metadata(1, 6) == {
        PREFIX_MEDIA_COUNT_FIELD: 1,
        PREFIX_EXPANDED_TOKEN_COUNT_FIELD: 6,
    }


def test_suffix_tokens_after_prefix_keeps_only_new_turn_suffix():
    assert _suffix_tokens_after_prefix(
        2, [1, 10, 42, 11, 500, 2], [1, 10, 42, 11, 500, 2, 12, 13]
    ) == [2, 12, 13]


def test_suffix_tokens_after_prefix_accepts_multiple_eos_token_ids():
    assert _suffix_tokens_after_prefix([2, 11], [1, 11, 10, 2], [1, 11, 10, 2, 12, 11, 13]) == [
        2,
        12,
        11,
        13,
    ]


@pytest.mark.parametrize(
    ("template_prefix", "current_tokens", "error"),
    [
        ([1, 10, 42], [1, 10, 42, 2, 12], "Could not locate an EOS-delimited"),
        ([1, 2, 10, 2], [1, 2, 10, 12], "Expected 2 EOS token"),
    ],
)
def test_suffix_tokens_after_prefix_rejects_missing_boundary(
    template_prefix, current_tokens, error
):
    with pytest.raises(ValueError, match=error):
        _suffix_tokens_after_prefix(2, template_prefix, current_tokens)


@pytest.mark.parametrize(
    ("previous_turn", "template_prefix", "current_turn", "expected"),
    [
        ([], [10, 2, 20, 2], [10, 2, 20, 2, 30, 31], [10, 2, 20, 2, 30, 31]),
        (
            [100, 101, 200, 201, 2],
            [10, 2, 20, 2],
            [10, 2, 20, 2, 30, 31],
            [100, 101, 200, 201, 2, 30, 31],
        ),
        (
            [100, 101, 200, 201],
            [10, 2, 20, 2],
            [10, 2, 20, 2, 30, 31],
            [100, 101, 200, 201, 2, 30, 31],
        ),
        (
            [100, 101, 200, 201, 2],
            [10, 2, 20, 2],
            [10, 999, 998, 2, 20, 2, 30, 31],
            [100, 101, 200, 201, 2, 30, 31],
        ),
    ],
    ids=(
        "empty-exact-prefix-keeps-template",
        "exact-prefix-ended-in-eos",
        "generation-hit-token-limit",
        "template-length-shifted",
    ),
)
def test_text_prefix_stitching_preserves_exact_previous_tokens(
    previous_turn, template_prefix, current_turn, expected
):
    assert _replace_prefix_tokens(2, previous_turn, template_prefix, current_turn) == expected


def _legacy_replace_prefix_tokens(
    eos_token_id,
    previous_turn_token_ids,
    retokenized_previous_turn_token_ids,
    current_turn_token_ids,
):
    """Pre-change positional implementation, retained only for equivalence tests."""
    if previous_turn_token_ids and previous_turn_token_ids[-1] == eos_token_id:
        previous_turn_token_ids = previous_turn_token_ids[:-1]
    boundary = len(retokenized_previous_turn_token_ids) - 1
    scan_len = min(len(retokenized_previous_turn_token_ids), len(current_turn_token_ids))
    for position in reversed(range(scan_len)):
        if current_turn_token_ids[position] == eos_token_id:
            boundary = position
            break
    return previous_turn_token_ids + current_turn_token_ids[boundary:]


@pytest.mark.parametrize(
    ("previous_turn", "template_prefix", "current_turn"),
    [
        ([100, 101, 200, 201, 2], [10, 2, 20, 2], [10, 2, 20, 2, 30, 31]),
        ([100, 101, 200, 201], [10, 2, 20, 2], [10, 2, 20, 2, 30, 31]),
    ],
)
def test_text_prefix_stitching_matches_legacy_when_template_prefix_is_unchanged(
    previous_turn, template_prefix, current_turn
):
    assert _replace_prefix_tokens(2, previous_turn, template_prefix, current_turn) == (
        _legacy_replace_prefix_tokens(2, previous_turn, template_prefix, current_turn)
    )


def test_text_prefix_stitching_tracks_turn_when_template_strips_prior_reasoning():
    previous_turn = [100, 101, 200, 201, 2]
    # Rendering the assistant as the final message retains reasoning tokens.
    template_prefix = [10, 2, 50, 51, 52, 20, 2]
    # Rendering it as history strips those tokens. The new turn also contains
    # an EOS, so "scan backwards within the old prefix length" selects the
    # wrong delimiter.
    current_turn = [10, 2, 20, 2, 30, 2, 31]

    assert _replace_prefix_tokens(2, previous_turn, template_prefix, current_turn) == [
        100,
        101,
        200,
        201,
        2,
        30,
        2,
        31,
    ]
    assert _legacy_replace_prefix_tokens(
        2, previous_turn, template_prefix, current_turn
    ) != _replace_prefix_tokens(2, previous_turn, template_prefix, current_turn)


def test_text_prefix_stitching_preserves_exact_trailing_eos_token():
    assert _replace_prefix_tokens(
        [2, 11], [100, 101, 200, 11], [10, 11, 20, 2], [10, 11, 20, 2, 30]
    ) == [100, 101, 200, 11, 30]


_USER = {"role": "user", "content": "hi"}
_ASSISTANT_TEXT = {"role": "assistant", "content": "hello"}
_ASSISTANT_WITH_TOKENS = {
    "role": "assistant",
    "content": "hello",
    "prompt_token_ids": [1, 2],
    "generation_token_ids": [3, 99],
}
_ENGINE_METADATA = {"ng_capture": {"staging_chain": ["k1"]}}


def test_has_previous_turn_tokens():
    assert _has_previous_turn_tokens(None) is False
    assert _has_previous_turn_tokens(_ASSISTANT_TEXT) is False  # dataset-provided history
    assert (
        _has_previous_turn_tokens(
            {"role": "assistant", "content": "", "prompt_token_ids": [], "generation_token_ids": []}
        )
        is False
    )
    assert _has_previous_turn_tokens(_ASSISTANT_WITH_TOKENS) is True


def test_last_assistant_message_returns_the_last_assistant_turn():
    assert _last_assistant_message([_USER]) == (None, None)
    assert _last_assistant_message([_USER, _ASSISTANT_TEXT, _USER]) == (1, _ASSISTANT_TEXT)
    messages = [_USER, _ASSISTANT_WITH_TOKENS, _USER, _ASSISTANT_TEXT, _USER]
    assert _last_assistant_message(messages) == (3, _ASSISTANT_TEXT)


def test_media_slot_uses_tokenizer_id_when_model_id_is_unspecified():
    class _Tokenizer:
        unk_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return "__MEDIA__"

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return []

    spec = MediaPromptSpec(model_token="<image>")
    prompt_config = MultimodalPromptConfig(image_spec=spec, video_spec=spec)

    tokens = _tokenize_with_media_slots_sync(
        _Tokenizer(),
        messages=[],
        media_slots=[("__MEDIA__", "image", 0)],
        prompt_config=prompt_config,
        tools=None,
        chat_template_kwargs={},
    )

    assert tokens == [99]


def test_temporal_video_slot_uses_the_configured_compact_wrapper():
    class _Tokenizer:
        unk_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return "__VIDEO__"

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [7] if text else []

    prompt_config = MultimodalPromptConfig(
        video_spec=MediaPromptSpec(
            model_token="<image>",
            prefix="<img>",
            suffix="</img>",
            expansion_mode="temporal_patch",
            include_frame_timestamps_for_nemotron_vl=True,
        )
    )

    tokens = _tokenize_with_media_slots_sync(
        _Tokenizer(),
        messages=[],
        media_slots=[("__VIDEO__", "video", 0)],
        prompt_config=prompt_config,
        tools=None,
        chat_template_kwargs={},
    )

    assert tokens == [7, 99, 7]


def test_media_content_uses_the_configured_part_separator():
    prompt_config = MultimodalPromptConfig(video_spec=MediaPromptSpec(content_part_separator="\n"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {"type": "text", "text": "__VIDEO__"},
            ],
        }
    ]

    sanitized = _sanitize_messages_for_template(
        messages, media_slots=[("__VIDEO__", "video", 0)], prompt_config=prompt_config
    )

    assert sanitized[0]["content"] == "question\n__VIDEO__"


def test_media_tokenization_is_synchronous_so_it_can_be_offloaded_whole():
    """Lowering media slots must not be a coroutine.

    Rendering is only the first of 3N+1 tokenizer calls for N slots. While this
    was async, awaiting the render put the remaining encodes back on the event
    loop, where they stall every other request the replica owns. Being a plain
    function is what lets the endpoint hand the whole thing to the tokenize
    executor in one hop, on the thread that owns the private tokenizer copy.
    """
    import inspect

    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    assert not inspect.iscoroutinefunction(chat_completions._tokenize_with_media_slots_sync)
    # And the endpoint must not have kept a direct call that skips the executor.
    src = inspect.getsource(chat_completions.chat_completions)
    assert "_tokenize_with_media_slots_sync" in src
    for line in src.splitlines():
        if "_tokenize_with_media_slots_sync" in line:
            assert "await" not in line, f"must be dispatched via the executor, got: {line.strip()}"


class _PrefixStitchingTokenizer:
    chat_template = "test-template"
    unk_token_id = 0
    eos_id = 2
    bos = None
    eod = None

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=True, **_kwargs
    ):
        assert tokenize is True
        if len(messages) == 2 and not add_generation_prompt:
            # The template's prior assistant text tokenizes differently from the
            # exact tokens returned by the model in the preceding request.
            return [10, 2, 20, 2]
        return [10, 2, 20, 2, 30, 31]

    def detokenize(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token) for token in tokens)

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<image>" else self.unk_token_id


class _PrefixStitchingClient:
    def __init__(self):
        self.submissions = []

    def add_request_with_id(
        self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
    ):
        self.submissions.append(
            {
                "prompt_tokens": list(prompt_tokens),
                "multi_modal_data": multi_modal_data,
                "offload_params": offload_params,
            }
        )
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "uid": "prefix-stitching",
                "status": "COMPLETED",
                "generated_tokens": [77],
                "prompt_length": len(prompt_tokens),
                "prompt_tokens": list(prompt_tokens),
                "num_cached_tokens": 0,
                "sampling_params": sampling_params.serialize(),
                "routing_indices": None,
            }
        )
        return 1, future

    def abort_request(self, _request_id):
        raise AssertionError("Successful request must not be aborted")


def _prefix_stitching_app(quart, chat_completions, *, eval_mode=False, tokenizer=None):
    tokenizer = tokenizer or _PrefixStitchingTokenizer()
    client = _PrefixStitchingClient()
    spec = MediaPromptSpec(model_token="<image>")
    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=tokenizer,
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(image_spec=spec, video_spec=spec),
        default_temperature=1.0,
        default_top_p=1.0,
        default_top_k=0,
        eval_mode=eval_mode,
    )
    app.register_blueprint(chat_completions.bp)
    return app, client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prevent_retokenization", "eval_mode", "expected_prompt"),
    [
        (False, False, [10, 2, 20, 2, 30, 31]),
        (True, True, [100, 101, 200, 201, 2, 30, 31]),
        (None, True, [10, 2, 20, 2, 30, 31]),
        (None, False, [100, 101, 200, 201, 2, 30, 31]),
    ],
    ids=(
        "explicitly-disabled",
        "explicitly-enabled",
        "eval-default-disabled",
        "rl-default-enabled",
    ),
)
async def test_text_only_prefix_stitching_respects_prevent_retokenization(
    prevent_retokenization, eval_mode, expected_prompt
):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(quart, chat_completions, eval_mode=eval_mode)
    request_json = {
        "messages": [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "first answer",
                "prompt_token_ids": [100, 101],
                "generation_token_ids": [200, 201, 2],
            },
            {"role": "user", "content": "second question"},
        ],
        "max_tokens": 1,
    }
    if prevent_retokenization is not None:
        request_json["prevent_retokenization"] = prevent_retokenization
    response = await app.test_client().post("/v1/chat/completions", json=request_json)

    assert response.status_code == 200
    assert client.submissions[0]["prompt_tokens"] == expected_prompt
    assert client.submissions[0]["offload_params"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("prevent_retokenization", [False, True])
async def test_exact_message_tokens_and_offload_prefix_source_are_mutually_exclusive(
    prevent_retokenization,
):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(quart, chat_completions)
    response = await app.test_client().post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "first question"},
                {
                    "role": "assistant",
                    "content": "first answer",
                    "prompt_token_ids": [100, 101],
                    "generation_token_ids": [200, 201, 2],
                },
                {"role": "user", "content": "second question"},
            ],
            "offload_params": {"stager": {"request_id": "r0"}},
            "prevent_retokenization": prevent_retokenization,
            "max_tokens": 1,
        },
    )

    assert response.status_code == 400
    assert "mutually exclusive prefix sources" in (await response.get_data()).decode()
    assert client.submissions == []


@pytest.mark.asyncio
async def test_multimodal_exact_prefix_requires_prior_media_in_message_history():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(quart, chat_completions)
    response = await app.test_client().post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "image omitted from replayed history"},
                {
                    "role": "assistant",
                    "content": "first answer",
                    "prompt_token_ids": [100, 99, 99, 101],
                    "generation_token_ids": [200, 2],
                },
                {"role": "user", "content": "second question"},
            ],
            "prevent_retokenization": True,
            "max_tokens": 1,
        },
    )

    assert response.status_code == 400
    assert "media tokens" in (await response.get_data()).decode()
    assert "missing from message history" in (await response.get_data()).decode()
    assert client.submissions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prevent_retokenization", [False, True])
@pytest.mark.parametrize("current_turn_has_media", [False, True])
async def test_multimodal_prefix_stitching_submits_exact_prefix_metadata_and_compact_suffix(
    current_turn_has_media, prevent_retokenization
):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(quart, chat_completions)
    image_url = f"data:image/png;base64,{base64.b64encode(b'image').decode()}"
    current_content = [{"type": "text", "text": "second question"}]
    if current_turn_has_media:
        current_content.append({"type": "image_url", "image_url": {"url": image_url}})

    def fake_multimodal_tokenize(
        _tokenizer,
        messages,
        _media_slots,
        _prompt_config,
        *,
        tools,
        chat_template_kwargs,
        add_generation_prompt=True,
    ):
        del tools, chat_template_kwargs
        if len(messages) == 2 and not add_generation_prompt:
            return [10, 42, 2, 20, 2]
        return [10, 42, 2, 20, 2, 30, *([42] if current_turn_has_media else [])]

    with mock.patch.object(
        chat_completions, "_tokenize_with_media_slots_sync", side_effect=fake_multimodal_tokenize
    ):
        response = await app.test_client().post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    },
                    {
                        "role": "assistant",
                        "content": "first answer",
                        "prompt_token_ids": [100, 99, 99, 101],
                        "generation_token_ids": [200, 2],
                    },
                    {"role": "user", "content": current_content},
                ],
                "prevent_retokenization": prevent_retokenization,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200
    submission = client.submissions[0]
    assert submission["multi_modal_data"] is not None
    if prevent_retokenization:
        assert submission["prompt_tokens"] == [
            100,
            99,
            99,
            101,
            200,
            2,
            30,
            *([42] if current_turn_has_media else []),
        ]
        assert submission["offload_params"] == {
            PREFIX_MEDIA_COUNT_FIELD: 1,
            PREFIX_EXPANDED_TOKEN_COUNT_FIELD: 6,
        }
    else:
        assert submission["prompt_tokens"] == [
            10,
            42,
            2,
            20,
            2,
            30,
            *([42] if current_turn_has_media else []),
        ]
        assert submission["offload_params"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("prevent_retokenization", [False, True])
@pytest.mark.parametrize("prefix_has_media", [False, True])
async def test_offloaded_prefix_stitching_metadata_covers_text_and_multimodal_history(
    prefix_has_media, prevent_retokenization
):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(quart, chat_completions)
    image_url = f"data:image/png;base64,{base64.b64encode(b'image').decode()}"
    first_content = (
        [{"type": "image_url", "image_url": {"url": image_url}}]
        if prefix_has_media
        else "first question"
    )

    def fake_multimodal_tokenize(
        _tokenizer,
        messages,
        _media_slots,
        _prompt_config,
        *,
        tools,
        chat_template_kwargs,
        add_generation_prompt=True,
    ):
        del tools, chat_template_kwargs
        if len(messages) == 2 and not add_generation_prompt:
            return [10, 42, 2, 20, 2]
        return [10, 42, 2, 20, 2, 30]

    patch = (
        mock.patch.object(
            chat_completions,
            "_tokenize_with_media_slots_sync",
            side_effect=fake_multimodal_tokenize,
        )
        if prefix_has_media
        else mock.patch.object(
            chat_completions,
            "_apply_chat_template_sync",
            side_effect=lambda _tokenizer, messages, _tools, _kwargs, add_generation_prompt=True: (
                [10, 2, 20, 2]
                if len(messages) == 2 and not add_generation_prompt
                else [10, 2, 20, 2, 30]
            ),
        )
    )
    with patch:
        response = await app.test_client().post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": first_content},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                ],
                "offload_params": {"stager": {"request_id": "r0"}},
                "prevent_retokenization": prevent_retokenization,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200
    submission = client.submissions[0]
    expected_prompt = [10, 42, 2, 20, 2, 30] if prefix_has_media else [10, 2, 20, 2, 30]
    assert submission["prompt_tokens"] == expected_prompt
    expected_template_prefix = [10, 42, 2, 20, 2] if prefix_has_media else [10, 2, 20, 2]
    assert submission["offload_params"] == {
        "stager": {"request_id": "r0"},
        PREFIX_TEMPLATE_TOKEN_IDS_FIELD: expected_template_prefix,
        PREFIX_EOS_TOKEN_ID_FIELD: [2],
        **({PREFIX_MEDIA_COUNT_FIELD: 1} if prefix_has_media else {}),
    }


class _MultiEosPrefixStitchingTokenizer(_PrefixStitchingTokenizer):
    eod = 2
    generation_config = {"eos_token_id": [2, 11]}


def _image_history_request(
    image_url, generation_token_ids, prompt_token_ids=(100, 99, 99, 101), **request_kwargs
):
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
            {
                "role": "assistant",
                "content": "first answer",
                "prompt_token_ids": list(prompt_token_ids),
                "generation_token_ids": generation_token_ids,
            },
            {"role": "user", "content": "second question"},
        ],
        "prevent_retokenization": True,
        "max_tokens": 1,
        **request_kwargs,
    }


def _fake_image_history_tokenize(
    _tokenizer,
    messages,
    _media_slots,
    _prompt_config,
    *,
    tools,
    chat_template_kwargs,
    add_generation_prompt=True,
):
    del tools, chat_template_kwargs
    if len(messages) == 2 and not add_generation_prompt:
        return [10, 42, 2, 20, 2]
    return [10, 42, 2, 20, 2, 30]


@pytest.mark.asyncio
async def test_text_prefix_stitching_recognizes_every_model_eos():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(
        quart, chat_completions, tokenizer=_MultiEosPrefixStitchingTokenizer()
    )
    response = await app.test_client().post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "first question"},
                {
                    "role": "assistant",
                    "content": "first answer",
                    "prompt_token_ids": [100, 101],
                    "generation_token_ids": [200, 201, 11],
                },
                {"role": "user", "content": "second question"},
            ],
            "prevent_retokenization": True,
            "max_tokens": 1,
        },
    )

    assert response.status_code == 200
    assert client.submissions[0]["prompt_tokens"] == [100, 101, 200, 201, 11, 30, 31]


@pytest.mark.asyncio
async def test_multimodal_prefix_stitching_recognizes_every_model_eos():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(
        quart, chat_completions, tokenizer=_MultiEosPrefixStitchingTokenizer()
    )
    image_url = f"data:image/png;base64,{base64.b64encode(b'image').decode()}"
    with mock.patch.object(
        chat_completions,
        "_tokenize_with_media_slots_sync",
        side_effect=_fake_image_history_tokenize,
    ):
        response = await app.test_client().post(
            "/v1/chat/completions", json=_image_history_request(image_url, [200, 11])
        )

    assert response.status_code == 200
    submission = client.submissions[0]
    assert submission["prompt_tokens"] == [100, 99, 99, 101, 200, 11, 30]
    assert submission["offload_params"][PREFIX_EXPANDED_TOKEN_COUNT_FIELD] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("add_BOS", "expected_prompt", "expected_prefix_count"),
    [(False, [100, 99, 99, 101, 200, 2, 30], 6), (True, [1, 100, 99, 99, 101, 200, 2, 30], 7)],
)
async def test_multimodal_expanded_prefix_count_follows_bos_handling(
    add_BOS, expected_prompt, expected_prefix_count
):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    tokenizer = _PrefixStitchingTokenizer()
    tokenizer.bos = 1
    app, client = _prefix_stitching_app(quart, chat_completions, tokenizer=tokenizer)
    image_url = f"data:image/png;base64,{base64.b64encode(b'image').decode()}"
    with mock.patch.object(
        chat_completions,
        "_tokenize_with_media_slots_sync",
        side_effect=_fake_image_history_tokenize,
    ):
        response = await app.test_client().post(
            "/v1/chat/completions",
            json=_image_history_request(
                image_url, [200, 2], prompt_token_ids=[1, 100, 99, 99, 101], add_BOS=add_BOS
            ),
        )

    assert response.status_code == 200
    submission = client.submissions[0]
    assert submission["prompt_tokens"] == expected_prompt
    assert submission["offload_params"][PREFIX_EXPANDED_TOKEN_COUNT_FIELD] == expected_prefix_count


@pytest.mark.asyncio
async def test_text_prefix_with_new_suffix_media_stitches_before_media_expansion():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    app, client = _prefix_stitching_app(quart, chat_completions)
    image_url = f"data:image/png;base64,{base64.b64encode(b'image').decode()}"

    with mock.patch.object(
        chat_completions, "_tokenize_with_media_slots_sync", return_value=[10, 2, 20, 2, 30, 42]
    ):
        response = await app.test_client().post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "first question"},
                    {
                        "role": "assistant",
                        "content": "first answer",
                        "prompt_token_ids": [100, 101],
                        "generation_token_ids": [200, 201, 2],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    },
                ],
                "prevent_retokenization": True,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200
    assert client.submissions[0]["prompt_tokens"] == [100, 101, 200, 201, 2, 30, 42]
    assert client.submissions[0]["offload_params"] is None


@pytest.mark.asyncio
async def test_n_choices_prepare_and_serialize_shared_media_once():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    class _Tokenizer:
        chat_template = "test-template"
        unk_token_id = 0
        eod = None

        def apply_chat_template(self, messages, **_kwargs):
            return "".join(message["content"] for message in messages)

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return []

        def detokenize(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    class _Client:
        def __init__(self):
            self.serialized_media = []

        def add_request_with_id(
            self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None
        ):
            wire = serialize_multimodal_data(multi_modal_data)
            self.serialized_media.append(wire)
            request_id = len(self.serialized_media)
            future = asyncio.get_running_loop().create_future()
            future.set_result(
                {
                    "uid": f"choice-{request_id}",
                    "status": "COMPLETED",
                    "generated_tokens": [request_id],
                    "prompt_length": len(prompt_tokens),
                    "prompt_tokens": prompt_tokens,
                    "num_cached_tokens": 0,
                    "sampling_params": sampling_params.serialize(),
                    "routing_indices": None,
                }
            )
            return request_id, future

        def abort_request(self, _request_id):
            raise AssertionError("Successful choices must not be aborted")

    tokenizer = _Tokenizer()
    client = _Client()
    spec = MediaPromptSpec(model_token="<image>")
    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=tokenizer,
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(image_spec=spec, video_spec=spec),
        default_temperature=1.0,
        default_top_p=1.0,
        default_top_k=0,
        eval_mode=False,
    )
    app.register_blueprint(chat_completions.bp)
    image = b"shared-image"
    image_url = f"data:image/png;base64,{base64.b64encode(image).decode()}"

    with mock.patch(
        "megatron.core.inference.inference_request.compute_media_cache_key",
        wraps=compute_media_cache_key,
    ) as compute_key:
        response = await app.test_client().post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    }
                ],
                "n": 3,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200
    assert len((await response.get_json())["choices"]) == 3
    assert len(client.serialized_media) == 3
    assert all(wire == client.serialized_media[0] for wire in client.serialized_media)
    assert all(wire is not client.serialized_media[0] for wire in client.serialized_media[1:])
    assert compute_key.call_count == 1
