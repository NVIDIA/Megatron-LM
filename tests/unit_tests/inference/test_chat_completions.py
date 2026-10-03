# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for /v1/chat/completions: its helpers, and chat-only behavior pinned over HTTP.

The HTTP tests reuse the fake inference client and app builder from test_endpoints_common.py,
which pins the behavior shared with /v1/completions.
"""

import base64
import contextlib
import copy
import inspect
import logging
import re
import socket
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    compute_media_cache_key,
    serialize_multimodal_data,
)
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
    chat_completions as chat_completions_module,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _coerce_to_token_id_list,
    _expanded_prefix_stitching_metadata,
    _extract_media_url_bytes,
    _extract_multimodal_from_messages,
    _has_previous_turn_tokens,
    _last_assistant_message,
    _NoRedirectHandler,
    _normalize_tool_calls,
    _redact_token_id_lists_for_logging,
    _replace_prefix_tokens,
    _replace_prefix_tokens_metadata,
    _sanitize_messages_for_template,
    _sanitize_tools_for_template,
    _serialize_eos_token_ids,
    _suffix_tokens_after_prefix,
    _TemplateRenderer,
    _tokenize_with_media_slots_sync,
)
from tests.unit_tests.inference.test_endpoints_common import (
    CHAT_BODY,
    CHAT_PATH,
    NOT_A_NUMBER_ERROR,
    NOT_AN_INT_ERROR,
    ReplyingClient,
    Tokenizer,
    build_app,
    build_streaming_app,
    completed_reply,
    sse_payloads,
)


def _media_block(modality, index):
    payload = base64.b64encode(f"{modality}-{index}".encode()).decode()
    if modality == "image":
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}
    return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{payload}"}}


def _image_data_url(payload=b"image"):
    return f"data:image/png;base64,{base64.b64encode(payload).decode()}"


# --- media fetching -----------------------------------------------------------


class _FakeMediaResponse:
    def __init__(self, data) -> None:
        self._data = data
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        self.read_sizes.append(size)
        return self._data if size is None or size < 0 else self._data[:size]


_PAYLOAD = b"x" * 64
_PUBLIC_ADDRESS = "93.184.216.34"
_REMOTE_URL = "https://example.com/cat.png"


@pytest.mark.parametrize(
    ("url", "max_fetch_bytes", "resolved", "data", "expected_read_sizes", "error"),
    [
        # Data URLs arrive in the request body, which Quart already bounds, so the limit is moot.
        pytest.param(_image_data_url(_PAYLOAD), None, None, _PAYLOAD, [], None, id="data-url"),
        pytest.param(
            _image_data_url(_PAYLOAD), 4, None, _PAYLOAD, [], None, id="data-url-ignores-limit"
        ),
        pytest.param(
            _REMOTE_URL, None, _PUBLIC_ADDRESS, b"image", [-1], None, id="remote-unbounded"
        ),
        pytest.param(_REMOTE_URL, 5, _PUBLIC_ADDRESS, b"image", [6], None, id="remote-at-limit"),
        # Reading one byte past the limit detects oversize without buffering the rest.
        pytest.param(
            _REMOTE_URL,
            5,
            _PUBLIC_ADDRESS,
            b"image!",
            [6],
            "example.com exceeds 5 byte limit",
            id="remote-over-limit",
        ),
        pytest.param("http:///a.png", None, None, None, [], "Invalid media URL", id="no-host"),
        pytest.param(
            "ftp://example.com/a.png",
            None,
            None,
            None,
            [],
            "Unsupported media URL scheme",
            id="ftp",
        ),
        pytest.param(
            "file:///etc/passwd", None, None, None, [], "Unsupported media URL scheme", id="file"
        ),
        pytest.param(
            "data:image/png;base64",
            None,
            None,
            None,
            [],
            "Malformed media data URL",
            id="bad-data-url",
        ),
        pytest.param(
            "https://nowhere.invalid/a.png",
            None,
            socket.gaierror("nowhere.invalid"),
            None,
            [],
            "Cannot resolve media URL host: nowhere.invalid",
            id="unresolvable-host",
        ),
        # One address per refused class: loopback, private, link-local, multicast, reserved,
        # unspecified.
        *(
            pytest.param(
                "http://media.example/a.png",
                None,
                address,
                None,
                [],
                "non-public address: media.example",
                id=address,
            )
            for address in (
                "127.0.0.1",
                "10.0.0.1",
                "169.254.169.254",
                "224.0.0.1",
                "240.0.0.1",
                "0.0.0.0",
            )
        ),
    ],
)
def test_extract_media_url_bytes(
    monkeypatch, url, max_fetch_bytes, resolved, data, expected_read_sizes, error
):
    """`data` is what the remote serves; None means the URL must be refused before any fetch."""
    response = _FakeMediaResponse(data)

    def resolve(host):
        del host
        if isinstance(resolved, Exception):
            raise resolved
        return resolved

    def open_(request, timeout):
        assert data is not None, "fetched a URL that should have been refused"
        assert request.get_header("User-agent") == chat_completions_module._MEDIA_FETCH_USER_AGENT
        assert timeout == chat_completions_module._MEDIA_FETCH_TIMEOUT_S
        return response

    monkeypatch.setattr(chat_completions_module.socket, "gethostbyname", resolve)
    monkeypatch.setattr(chat_completions_module._no_redirect_opener, "open", open_)
    expectation = pytest.raises(ValueError, match=error) if error else contextlib.nullcontext()

    with expectation:
        assert _extract_media_url_bytes(url, max_fetch_bytes=max_fetch_bytes) == data

    assert response.read_sizes == expected_read_sizes


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_media_fetch_refuses_redirects(code):
    request = urllib.request.Request("https://example.com/a.png")
    redirect = getattr(_NoRedirectHandler(), f"http_error_{code}")

    with pytest.raises(urllib.error.HTTPError) as error:
        redirect(request, None, code, "Moved", {"Location": "http://127.0.0.1/"})

    assert error.value.code == code


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}},
    ],
    ids=["image", "video"],
)
def test_extract_multimodal_forwards_fetch_limit_to_media_urls(block):
    url = block[block["type"]]["url"]
    messages = [{"role": "user", "content": [block]}]
    with mock.patch.object(
        chat_completions_module, "_extract_media_url_bytes", return_value=b"media"
    ) as extract:
        _, image_bytes_list, video_bytes_list, _ = _extract_multimodal_from_messages(
            messages, MultimodalPromptConfig(), max_fetch_bytes=123
        )

    assert image_bytes_list + video_bytes_list == [b"media"]
    extract.assert_called_once_with(url, max_fetch_bytes=123)


@pytest.mark.asyncio
async def test_chat_endpoint_bounds_remote_media_by_max_content_length():
    app = build_app(CHAT_PATH, ReplyingClient([]), MAX_CONTENT_LENGTH=2**30)

    with mock.patch.object(
        chat_completions_module, "_extract_multimodal_from_messages", side_effect=ValueError("stop")
    ) as extract:
        response = await app.test_client().post(CHAT_PATH, json=CHAT_BODY)

    assert response.status_code == 400
    assert extract.call_args.args[2] == 2**30


# --- multimodal message extraction --------------------------------------------

_SLOT_0 = "__MCORE_MEDIA_SLOT_0__"
_IMAGE_WITHOUT_URL = [{"type": "image_url", "image_url": {}}, {"type": "text", "text": "hi"}]


_SLOT_0 = "__MCORE_MEDIA_SLOT_0__"
_IMAGE_WITHOUT_URL = [{"type": "image_url", "image_url": {}}, {"type": "text", "text": "hi"}]
_INPUT_VIDEO = {
    "type": "input_video",
    "video": f"data:video/mp4;base64,{base64.b64encode(b'v').decode()}",
}


def _user(content):
    return {"role": "user", "content": content}


@pytest.mark.parametrize(
    ("messages", "prompt_config", "expected", "error"),
    [
        pytest.param(
            "hi", MultimodalPromptConfig(), ("hi", [], [], []), None, id="non-list-pass-through"
        ),
        pytest.param(
            [_user(_IMAGE_WITHOUT_URL)],
            MultimodalPromptConfig(),
            ([_user(_IMAGE_WITHOUT_URL)], [], [], []),
            None,
            id="image-block-without-url-is-skipped",
        ),
        pytest.param(
            [_user([_INPUT_VIDEO])],
            MultimodalPromptConfig(),
            ([_user([{"type": "text", "text": _SLOT_0}])], [], [b"v"], [(_SLOT_0, "video", 0)]),
            None,
            id="input-video-bare-data-url",
        ),
        pytest.param(
            [_user([{"type": "text", "text": "look <image> here"}, _media_block("image", 0)])],
            MultimodalPromptConfig(image_spec=MediaPromptSpec(input_marker="<image>")),
            (
                [
                    _user(
                        [{"type": "text", "text": "look  here"}, {"type": "text", "text": _SLOT_0}]
                    )
                ],
                [b"image-0"],
                [],
                [(_SLOT_0, "image", 0)],
            ),
            None,
            id="input-marker-is-stripped",
        ),
        pytest.param(
            [_user([_media_block("image", 0)]), _user([_media_block("video", 0)])],
            MultimodalPromptConfig(),
            None,
            "Mixing image and video",
            id="mixed-image-and-video",
        ),
        pytest.param(
            [_user([{"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}])],
            MultimodalPromptConfig(),
            None,
            "must be base64 data URLs",
            id="remote-video",
        ),
        pytest.param(
            [_user([{"type": "input_video", "video": None}])],
            MultimodalPromptConfig(),
            None,
            "must be base64 data URLs",
            id="no-video",
        ),
    ],
)
def test_extract_multimodal_from_messages(messages, prompt_config, expected, error):
    original = copy.deepcopy(messages)
    expectation = pytest.raises(ValueError, match=error) if error else contextlib.nullcontext()

    with expectation:
        assert _extract_multimodal_from_messages(messages, prompt_config) == expected

    assert messages == original


# --- prefix stitching helpers -------------------------------------------------


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


@pytest.mark.parametrize(
    ("eos_token_ids", "template_prefix", "current_tokens", "expected", "error"),
    [
        pytest.param(
            2,
            [1, 10, 42, 11, 500, 2],
            [1, 10, 42, 11, 500, 2, 12, 13],
            [2, 12, 13],
            None,
            id="one-eos",
        ),
        pytest.param(
            [2, 11],
            [1, 11, 10, 2],
            [1, 11, 10, 2, 12, 11, 13],
            [2, 12, 11, 13],
            None,
            id="several-eos",
        ),
        pytest.param(
            2,
            [1, 10, 42],
            [1, 10, 42, 2, 12],
            None,
            "Could not locate an EOS-delimited",
            id="no-eos-in-prefix",
        ),
        pytest.param(
            2, [1, 2, 10, 2], [1, 2, 10, 12], None, "Expected 2 EOS token", id="fewer-eos-in-turn"
        ),
    ],
)
def test_suffix_tokens_after_prefix_keeps_only_the_new_turn(
    eos_token_ids, template_prefix, current_tokens, expected, error
):
    expectation = pytest.raises(ValueError, match=error) if error else contextlib.nullcontext()

    with expectation:
        assert _suffix_tokens_after_prefix(eos_token_ids, template_prefix, current_tokens) == (
            expected
        )


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
    ("eos", "previous_turn", "template_prefix", "current_turn", "expected", "legacy_agrees"),
    [
        pytest.param(
            2,
            [],
            [10, 2, 20, 2],
            [10, 2, 20, 2, 30, 31],
            [10, 2, 20, 2, 30, 31],
            False,
            id="empty-exact-prefix-keeps-template",
        ),
        pytest.param(
            2,
            [100, 101, 200, 201, 2],
            [10, 2, 20, 2],
            [10, 2, 20, 2, 30, 31],
            [100, 101, 200, 201, 2, 30, 31],
            True,
            id="exact-prefix-ended-in-eos",
        ),
        pytest.param(
            2,
            [100, 101, 200, 201],
            [10, 2, 20, 2],
            [10, 2, 20, 2, 30, 31],
            [100, 101, 200, 201, 2, 30, 31],
            True,
            id="generation-hit-token-limit",
        ),
        pytest.param(
            2,
            [100, 101, 200, 201, 2],
            [10, 2, 20, 2],
            [10, 999, 998, 2, 20, 2, 30, 31],
            [100, 101, 200, 201, 2, 30, 31],
            False,
            id="template-length-shifted",
        ),
        # Rendering the assistant as the final message retains its reasoning tokens (50-52);
        # rendering it as history strips them. The new turn also contains an EOS, so "scan
        # backwards within the old prefix length" selects the wrong delimiter.
        pytest.param(
            2,
            [100, 101, 200, 201, 2],
            [10, 2, 50, 51, 52, 20, 2],
            [10, 2, 20, 2, 30, 2, 31],
            [100, 101, 200, 201, 2, 30, 2, 31],
            False,
            id="template-strips-prior-reasoning",
        ),
        pytest.param(
            [2, 11],
            [100, 101, 200, 11],
            [10, 11, 20, 2],
            [10, 11, 20, 2, 30],
            [100, 101, 200, 11, 30],
            False,
            id="exact-trailing-eos-among-several",
        ),
    ],
)
def test_text_prefix_stitching_preserves_exact_previous_tokens(
    eos, previous_turn, template_prefix, current_turn, expected, legacy_agrees
):
    assert _replace_prefix_tokens(eos, previous_turn, template_prefix, current_turn) == expected
    # The positional implementation only agrees while the template prefix is unchanged.
    legacy = _legacy_replace_prefix_tokens(eos, previous_turn, template_prefix, current_turn)
    assert (legacy == expected) is legacy_agrees


_USER = {"role": "user", "content": "hi"}
_ASSISTANT_TEXT = {"role": "assistant", "content": "hello"}
_ASSISTANT_WITH_TOKENS = {
    "role": "assistant",
    "content": "hello",
    "prompt_token_ids": [1, 2],
    "generation_token_ids": [3, 99],
}


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


# --- template sanitization and media slots -------------------------------------


_SLOT_TEXT = [{"type": "text", "text": "__S0__"}]


@pytest.mark.parametrize(
    ("content", "media_slots", "prompt_config", "expected", "error"),
    [
        pytest.param({"type": "text", "text": "hi"}, [], None, "hi", None, id="dict"),
        pytest.param(None, [], None, "", None, id="none"),
        pytest.param(5, [], None, "5", None, id="number"),
        # Bare strings and any dict with `text` are kept; other blocks and values are dropped.
        pytest.param(
            ["a", {"type": "text", "text": "b"}, {"text": "c"}, {"type": "image_url"}, 7],
            [],
            None,
            "abc",
            None,
            id="mixed-list",
        ),
        pytest.param(
            [{"type": "text", "text": "question"}, {"type": "text", "text": "__VIDEO__"}],
            [("__VIDEO__", "video", 0)],
            MultimodalPromptConfig(video_spec=MediaPromptSpec(content_part_separator="\n")),
            "question\n__VIDEO__",
            None,
            id="configured-part-separator",
        ),
        pytest.param(
            [
                {"type": "text", "text": "question"},
                {"type": "text", "text": "__IMAGE_0__"},
                {"type": "text", "text": "Image 1:"},
                {"type": "text", "text": "__IMAGE_1__"},
                {"type": "text", "text": "Image 2:"},
            ],
            [("__IMAGE_0__", "image", 0), ("__IMAGE_1__", "image", 0)],
            MultimodalPromptConfig(
                image_spec=MediaPromptSpec(content_part_separator="\n"),
                content_part_order="media_first",
            ),
            "__IMAGE_0__\n__IMAGE_1__\nquestion\nImage 1:\nImage 2:",
            None,
            id="media-first-matches-structured-hf-rendering",
        ),
        pytest.param(
            _SLOT_TEXT,
            [("__S0__", "image", 0)],
            None,
            None,
            "requires a prompt config",
            id="media-without-a-prompt-config",
        ),
        pytest.param(
            _SLOT_TEXT,
            [("__S0__", "image", 0), ("__S1__", "video", 0)],
            MultimodalPromptConfig(
                image_spec=MediaPromptSpec(content_part_separator="\n"),
                video_spec=MediaPromptSpec(content_part_separator=" "),
            ),
            None,
            "same content-part separator",
            id="conflicting-separators",
        ),
    ],
)
def test_sanitize_messages_flattens_content_to_a_string(
    content, media_slots, prompt_config, expected, error
):
    expectation = pytest.raises(ValueError, match=error) if error else contextlib.nullcontext()

    with expectation:
        (sanitized,) = _sanitize_messages_for_template(
            [_user(content)], media_slots=media_slots, prompt_config=prompt_config
        )
        assert sanitized["content"] == expected


def test_sanitize_messages_coerces_tool_call_arguments_to_mappings():
    calls = [
        {"function": {"name": "a", "arguments": '{"x": 1}'}},
        {"function": {"name": "b", "arguments": "[1, 2]"}},
        {"function": {"name": "c", "arguments": "not-json"}},
        {"function": {"name": "d"}},
        "opaque",
    ]
    message = {"role": "assistant", "content": None, "tool_calls": calls}
    original = copy.deepcopy(message)

    (sanitized,) = _sanitize_messages_for_template([message])

    assert [call["function"]["arguments"] for call in sanitized["tool_calls"][:4]] == [
        {"x": 1},
        {},
        {},
        {},
    ]
    assert sanitized["tool_calls"][4] == "opaque"
    assert message == original


def test_sanitize_tools_drops_non_dicts_and_fills_missing_parameters():
    valid = {"type": "function", "function": {"name": "g", "parameters": {"type": "object"}}}
    tools = [{"type": "function", "function": {"name": "f", "parameters": "bad"}}, "junk", valid]
    original = copy.deepcopy(tools)

    assert _sanitize_tools_for_template(tools) == [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
        },
        valid,
    ]
    assert tools == original
    assert _sanitize_tools_for_template({"not": "a list"}) is None


class _MediaSlotTokenizer:
    """Renders a fixed template; `<image>` maps to `media_token_id`, other text to `text_token`."""

    unk_token_id = 0

    def __init__(self, rendered, media_token_id=99, text_token=None):
        self.rendered = rendered
        self.media_token_id = media_token_id
        self.text_token = text_token

    def apply_chat_template(self, *_args, **_kwargs):
        return self.rendered

    def convert_tokens_to_ids(self, token):
        return self.media_token_id if token == "<image>" else self.unk_token_id

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.text_token] if text and self.text_token is not None else []


_TEMPORAL_VIDEO_CONFIG = MultimodalPromptConfig(
    video_spec=MediaPromptSpec(
        model_token="<image>",
        prefix="<img>",
        suffix="</img>",
        expansion_mode="temporal_patch",
        include_frame_timestamps_for_nemotron_vl=True,
    )
)


@pytest.mark.parametrize(
    ("prompt_config", "modality", "tokenizer", "expected_tokens", "error"),
    [
        # Without a configured model token id the slot takes the tokenizer's id for the token.
        pytest.param(
            MultimodalPromptConfig(image_spec=MediaPromptSpec(model_token="<image>")),
            "image",
            _MediaSlotTokenizer("__S0__"),
            [99],
            None,
            id="image-token-via-tokenizer-id",
        ),
        pytest.param(
            _TEMPORAL_VIDEO_CONFIG,
            "video",
            _MediaSlotTokenizer("__S0__", text_token=7),
            [7, 99, 7],
            None,
            id="temporal-video-compact-wrapper",
        ),
        pytest.param(
            MultimodalPromptConfig(),
            "image",
            _MediaSlotTokenizer([1, 2]),
            None,
            "must return a string",
            id="render-not-a-string",
        ),
        pytest.param(
            MultimodalPromptConfig(),
            "image",
            _MediaSlotTokenizer("no slot"),
            None,
            "did not preserve media slot __S0__",
            id="slot-dropped",
        ),
        pytest.param(
            MultimodalPromptConfig(),
            "image",
            _MediaSlotTokenizer("__S0__ __S0__"),
            None,
            "did not preserve media slot __S0__",
            id="slot-duplicated",
        ),
        pytest.param(
            MultimodalPromptConfig(),
            "image",
            _MediaSlotTokenizer("__S0__", media_token_id=0),
            None,
            "does not define media token",
            id="unknown-media-token",
        ),
    ],
)
def test_tokenize_with_media_slots_lowers_slots_to_model_tokens(
    prompt_config, modality, tokenizer, expected_tokens, error
):
    expectation = (
        pytest.raises((TypeError, ValueError), match=error) if error else contextlib.nullcontext()
    )

    with expectation:
        tokens = _tokenize_with_media_slots_sync(
            tokenizer,
            messages=[],
            media_slots=[("__S0__", modality, 0)],
            prompt_config=prompt_config,
            tools=None,
            chat_template_kwargs={},
        )
        assert tokens == expected_tokens


# --- endpoint media placement -------------------------------------------------

_MEDIA_TAG_PATTERN = re.compile(r"(<img>|<image>|</img>)")


class _SegmentTokenizer:
    """Emits each run of plain text as one token so expected prompts stay readable."""

    unk_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(f"<{message['role']}>{message['content']}" for message in messages)

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<image>" else self.unk_token_id

    def tokenize(self, text):
        return [
            99 if part == "<image>" else part for part in _MEDIA_TAG_PATTERN.split(text) if part
        ]

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.tokenize(text)


def _omni_prompt_config(content_part_order, frame_timestamps):
    defaults = NemotronOmniInferenceWrapper.multimodal_prompt_config
    return replace(
        defaults,
        content_part_order=content_part_order,
        video_spec=replace(
            defaults.video_spec, include_frame_timestamps_for_nemotron_vl=frame_timestamps
        ),
    )


def _endpoint_prompt_tokens(messages, prompt_config):
    messages, _images, _videos, media_slots = _extract_multimodal_from_messages(
        messages, prompt_config
    )
    template_messages = _sanitize_messages_for_template(messages, media_slots, prompt_config)
    return _tokenize_with_media_slots_sync(
        _SegmentTokenizer(),
        template_messages,
        media_slots,
        prompt_config,
        tools=None,
        chat_template_kwargs={},
    )


@pytest.mark.parametrize(
    "modality, frame_timestamps",
    [("image", False), ("video", False), ("video", True)],
    ids=["image", "video", "video_with_timestamps"],
)
@pytest.mark.parametrize(
    "content_part_order, expected_tokens",
    [
        (
            "preserve",
            [
                "<system>Be brief.<user>Compare these.\n",
                *("<img>", 99, "</img>"),
                "\nFirst.\n",
                *("<img>", 99, "</img>"),
                "\nSecond.",
            ],
        ),
        (
            "media_first",
            [
                "<system>Be brief.<user>",
                *("<img>", 99, "</img>"),
                "\n",
                *("<img>", 99, "</img>"),
                "\nCompare these.\nFirst.\nSecond.",
            ],
        ),
    ],
)
def test_endpoint_places_media_by_content_part_order(
    modality, frame_timestamps, content_part_order, expected_tokens
):
    """Frame timestamps are rendered later by the model wrapper, so the endpoint's
    compact prompt depends only on the content-part order."""
    messages = [
        # A list-content message without media must be left as-is under either order.
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these."},
                _media_block(modality, 0),
                {"type": "text", "text": "First."},
                _media_block(modality, 1),
                {"type": "text", "text": "Second."},
            ],
        },
    ]

    tokens = _endpoint_prompt_tokens(
        messages, _omni_prompt_config(content_part_order, frame_timestamps)
    )

    assert tokens == expected_tokens


@pytest.mark.parametrize(
    "frame_timestamps, expanded_video_tokens",
    [
        (
            True,
            [
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.00 seconds: ",
                *("<img>", -1, "</img>"),
                "\nFrame 3 sampled at 2.00 seconds and frame 4 sampled at 3.00 seconds: ",
                *("<img>", -1, "</img>"),
            ],
        ),
        (False, [*("<img>", -1, "</img>"), "\n", *("<img>", -1, "</img>")]),
    ],
    ids=["with_timestamps", "without_timestamps"],
)
def test_omni_expands_media_first_video_prompt_ahead_of_text(
    frame_timestamps, expanded_video_tokens
):
    assert NemotronOmniInferenceWrapper.multimodal_prompt_config.content_part_order == (
        "media_first"
    )
    prompt_config = _omni_prompt_config("media_first", frame_timestamps)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe it."}, _media_block("video", 0)],
        }
    ]

    tokens = _endpoint_prompt_tokens(messages, prompt_config)
    assert tokens == ["<user>", "<img>", 99, "</img>", "\nDescribe it."]

    wrapper = object.__new__(NemotronOmniInferenceWrapper)
    wrapper.multimodal_prompt_config = prompt_config
    wrapper.model = SimpleNamespace(
        image_token_index=-200,
        dynamic_resolution=True,
        patch_dim=16,
        vision_model=SimpleNamespace(temporal_patch_dim=2),
    )
    # 32x32 frames give one embedding per frame after pixel shuffle, so each
    # two-frame tubelet expands to a single -1 placeholder.
    expanded, _masks = wrapper.expand_image_tokens(
        [tokens],
        imgs_sizes=torch.tensor([[32, 32]] * 4),
        num_frames=torch.tensor([4]),
        image_token_id=99,
        tokenizer=_SegmentTokenizer(),
        video_frame_indices=[[0, 10, 20, 30]],
        video_fps=[10.0],
    )

    assert expanded == [["<user>", *expanded_video_tokens, "\nDescribe it."]]


def test_media_tokenization_is_synchronous_so_it_can_be_offloaded_whole():
    """Lowering media slots must not be a coroutine.

    Rendering is only the first of 3N+1 tokenizer calls for N slots. While this
    was async, awaiting the render put the remaining encodes back on the event
    loop, where they stall every other request the replica owns. Being a plain
    function is what lets the endpoint hand the whole thing to the tokenize
    executor in one hop, on the thread that owns the private tokenizer copy.
    """
    assert not inspect.iscoroutinefunction(_tokenize_with_media_slots_sync)
    # Executor dispatch is owned by the renderer; the endpoint must not call the sync function
    # directly, and the renderer must hand it to the executor rather than await it.
    assert "_tokenize_with_media_slots_sync" not in inspect.getsource(
        chat_completions_module.chat_completions
    )
    src = inspect.getsource(_TemplateRenderer.tokenize)
    assert "_tokenize_with_media_slots_sync" in src
    for line in src.splitlines():
        if "_tokenize_with_media_slots_sync" in line:
            assert "await" not in line, f"must be dispatched via the executor, got: {line.strip()}"


# --- tool calls, token ids, redaction, parsers ---------------------------------


class _Unserializable:
    def __str__(self):
        return "unserializable"


_SCHEMA_TOOLS = [
    {
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array"},
                    "options": {"anyOf": [{"type": "null"}, {"type": "object"}]},
                    "tags": {"type": ["array", "null"]},
                    "note": {"type": "string"},
                    "broken": {"type": "array"},
                },
            },
        }
    }
]


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        pytest.param([{"function": {"arguments": "{}"}}, {}], [], id="nameless-calls-are-skipped"),
        pytest.param(
            [{"function": {"name": "f", "arguments": "{}"}}],
            [
                {
                    "id": f"call_{'a' * 24}",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
            id="missing-id-is-generated",
        ),
        pytest.param(
            [SimpleNamespace(id="c1", function=SimpleNamespace(name="f", arguments={"x": 1}))],
            [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}],
            id="attribute-style-call",
        ),
    ],
)
def test_normalize_tool_calls_shapes_calls(monkeypatch, tool_calls, expected):
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))

    assert _normalize_tool_calls(tool_calls) == expected


@pytest.mark.parametrize(
    ("arguments", "tools", "expected"),
    [
        pytest.param("not json", None, "not json", id="invalid-json"),
        pytest.param("[1]", None, "[1]", id="json-array"),
        pytest.param([1, 2], None, "[1, 2]", id="list"),
        pytest.param(None, None, "null", id="none"),
        pytest.param(_Unserializable(), None, "unserializable", id="unserializable"),
        pytest.param({"city": "Zürich"}, None, '{"city": "Zürich"}', id="non-ascii"),
        # String values are parsed only where the tool schema declares a structured type.
        pytest.param(
            {
                "items": "[1, 2]",
                "options": '{"a": 1}',
                "tags": '["x"]',
                "note": "[3]",
                "broken": "[1,",
            },
            _SCHEMA_TOOLS,
            '{"items": [1, 2], "options": {"a": 1}, "tags": ["x"], "note": "[3]", "broken": "[1,"}',
            id="schema-declared-structures",
        ),
    ],
)
def test_normalize_tool_calls_serializes_arguments(arguments, tools, expected):
    (normalized,) = _normalize_tool_calls(
        [{"function": {"name": "f", "arguments": arguments}}], tools
    )

    assert normalized["function"]["arguments"] == expected


@pytest.mark.parametrize(
    "result",
    [
        [1, 2],
        (1, 2),
        {"input_ids": [1, 2]},
        {"input_ids": [[1, 2]]},
        {"input_ids": torch.tensor([[1, 2]])},
        SimpleNamespace(ids=[1, 2]),
        torch.tensor([1, 2]),
        torch.tensor([[1, 2]]),
    ],
    ids=[
        "list",
        "tuple",
        "batch-encoding",
        "batched-batch-encoding",
        "batch-encoding-tensor",
        "fast-encoding",
        "tensor",
        "batched-tensor",
    ],
)
def test_coerce_to_token_id_list(result):
    assert _coerce_to_token_id_list(result) == [1, 2]


@pytest.mark.parametrize(
    ("eos_token_ids", "expected"),
    [
        (2, [2]),
        ({3, 1}, [1, 3]),
        ((2, 2), [2]),
        ([], None),
        (None, None),
        (True, None),
        ("2", None),
        ([2, "3"], None),
        ([2.0], None),
    ],
)
def test_serialize_eos_token_ids(eos_token_ids, expected):
    """Non-empty integer ids serialize sorted and unique; anything else is rejected."""
    expectation = (
        contextlib.nullcontext()
        if expected is not None
        else pytest.raises(ValueError, match="EOS token IDs")
    )

    with expectation:
        assert _serialize_eos_token_ids(eos_token_ids) == expected


def test_redact_token_id_lists_for_logging():
    record = {
        "uid": "r",
        "prompt_tokens": [1, 2],
        "routing_indices": [[1, 2], [3, 4]],
        "extra_token_ids": [5],
        "precomputed_block_hashes": [9],
        "tpot": [0.1, 0.2],
        "nested": [{"generated_tokens": [3]}],
        "generated_tokens": "not-a-list",
        "events": [{"type": "ADD"}],
    }

    truncated = "...truncated..."
    assert _redact_token_id_lists_for_logging(record) == {
        "uid": "r",
        "prompt_tokens": truncated,
        "routing_indices": truncated,
        "extra_token_ids": truncated,
        "precomputed_block_hashes": truncated,
        "tpot": truncated,
        "nested": [{"generated_tokens": truncated}],
        "generated_tokens": "not-a-list",
        "events": [{"type": "ADD"}],
    }


def _parser(parse_result):
    parser = mock.MagicMock()
    parser.implicit_reasoning_end_markers = ()
    # A fresh copy per call: the endpoint normalizes, and may drop, the tool calls in place.
    parser.parse.side_effect = lambda *args, **kwargs: copy.deepcopy(parse_result)
    return parser


@pytest.mark.parametrize("tools_requested", [False, True])
def test_apply_parsers_chains_parsers_and_forwards_markers_only_with_tools(tools_requested):
    reasoning = _parser(("answer <tool_call>", {"reasoning": "why"}))
    reasoning.implicit_reasoning_end_markers = ("<tool_call>",)
    tool = _parser(("answer", {}))
    mapping = {"reasoning": reasoning, "tool": tool}

    with mock.patch.object(chat_completions_module, "PARSER_MAPPING", mapping):
        text, metadata = chat_completions_module.apply_parsers(
            "raw", None, ["reasoning", "tool"], tools_requested, finished=False
        )

    assert (text, metadata) == ("answer", {"reasoning": "why"})
    assert tool.parse.call_args.args == ("answer <tool_call>",)
    markers = ("<tool_call>",) if tools_requested else ()
    for parser in (reasoning, tool):
        assert parser.parse.call_args.kwargs["implicit_reasoning_end_markers"] == markers
        assert parser.parse.call_args.kwargs["finished"] is False


@pytest.mark.parametrize(
    ("mapping", "parsers", "exc_type", "error"),
    [
        pytest.param({}, ["missing"], ValueError, "Parser missing not found", id="unknown-parser"),
        pytest.param(
            {"a": _parser(("x", {"reasoning": "1"})), "b": _parser(("x", {"reasoning": "2"}))},
            ["a", "b"],
            AssertionError,
            "Multiple parsers",
            id="same-field-twice",
        ),
    ],
)
def test_apply_parsers_rejects_bad_parser_lists(mapping, parsers, exc_type, error):
    with mock.patch.object(chat_completions_module, "PARSER_MAPPING", mapping):
        with pytest.raises(exc_type, match=error):
            chat_completions_module.apply_parsers("raw", None, parsers, False)


# --- HTTP: prefix stitching ---------------------------------------------------


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


class _MultiEosPrefixStitchingTokenizer(_PrefixStitchingTokenizer):
    eod = 2
    generation_config = {"eos_token_id": [2, 11]}


class _BosPrefixStitchingTokenizer(_PrefixStitchingTokenizer):
    bos = 1


def _prefix_stitching_app(*, eval_mode=False, tokenizer=None, client=None):
    client = client or ReplyingClient()
    spec = MediaPromptSpec(model_token="<image>")
    app = build_app(
        CHAT_PATH,
        client,
        tokenizer=tokenizer or _PrefixStitchingTokenizer(),
        multimodal_prompt_config=MultimodalPromptConfig(image_spec=spec, video_spec=spec),
        default_temperature=1.0,
        default_top_p=1.0,
        default_top_k=0,
        eval_mode=eval_mode,
    )
    return app, client


def _text_history_request(
    generation_token_ids=(200, 201, 2),
    prompt_token_ids=(100, 101),
    current_content="second question",
    **request_kwargs,
):
    """A second turn whose text-only history carries the exact tokens of the first answer."""
    return {
        "messages": [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "first answer",
                "prompt_token_ids": list(prompt_token_ids),
                "generation_token_ids": list(generation_token_ids),
            },
            {"role": "user", "content": current_content},
        ],
        "max_tokens": 1,
        **request_kwargs,
    }


_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": _image_data_url()}}
_TEXT_PART = {"type": "text", "text": "second question"}


def _image_history_request(
    generation_token_ids,
    prompt_token_ids=(100, 99, 99, 101),
    current_content="second question",
    **request_kwargs,
):
    """A second turn whose image-first history replays the exact tokens of the first answer."""
    return {
        "messages": [
            _user([_IMAGE_BLOCK]),
            {
                "role": "assistant",
                "content": "first answer",
                "prompt_token_ids": list(prompt_token_ids),
                "generation_token_ids": list(generation_token_ids),
            },
            _user(current_content),
        ],
        "prevent_retokenization": True,
        "max_tokens": 1,
        **request_kwargs,
    }


_MEDIA_HISTORY = [10, 42, 2, 20, 2]
_TEXT_HISTORY = [10, 2, 20, 2]


def _fake_tokenize(history, suffix):
    """Stands in for either tokenize helper: the two-message history render, or history + turn."""

    def tokenize(_tokenizer, messages, *_args, add_generation_prompt=True, **_kwargs):
        if len(messages) == 2 and not add_generation_prompt:
            return list(history)
        return [*history, *suffix]

    return tokenize


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prevent_retokenization", "eval_mode", "tokenizer_cls", "generation_token_ids", "expected"),
    [
        pytest.param(
            False,
            False,
            _PrefixStitchingTokenizer,
            [200, 201, 2],
            [10, 2, 20, 2, 30, 31],
            id="explicitly-disabled",
        ),
        pytest.param(
            True,
            True,
            _PrefixStitchingTokenizer,
            [200, 201, 2],
            [100, 101, 200, 201, 2, 30, 31],
            id="explicitly-enabled",
        ),
        pytest.param(
            None,
            True,
            _PrefixStitchingTokenizer,
            [200, 201, 2],
            [10, 2, 20, 2, 30, 31],
            id="eval-default-disabled",
        ),
        pytest.param(
            None,
            False,
            _PrefixStitchingTokenizer,
            [200, 201, 2],
            [100, 101, 200, 201, 2, 30, 31],
            id="rl-default-enabled",
        ),
        # The exact prefix may end in any of the model's EOS ids.
        pytest.param(
            True,
            False,
            _MultiEosPrefixStitchingTokenizer,
            [200, 201, 11],
            [100, 101, 200, 201, 11, 30, 31],
            id="every-model-eos",
        ),
    ],
)
async def test_text_only_prefix_stitching_respects_prevent_retokenization(
    prevent_retokenization, eval_mode, tokenizer_cls, generation_token_ids, expected
):
    app, client = _prefix_stitching_app(eval_mode=eval_mode, tokenizer=tokenizer_cls())
    request_json = _text_history_request(generation_token_ids)
    if prevent_retokenization is not None:
        request_json["prevent_retokenization"] = prevent_retokenization

    response = await app.test_client().post(CHAT_PATH, json=request_json)

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.prompt_tokens == [expected]
    assert client.offload_params == [None]
    assert client.aborted == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_json", "expected_errors"),
    [
        *(
            pytest.param(
                _text_history_request(
                    offload_params={"stager": {"request_id": "r0"}},
                    prevent_retokenization=prevent_retokenization,
                ),
                ["mutually exclusive prefix sources"],
                id=f"exact-tokens-and-offload-prefix-{prevent_retokenization}",
            )
            for prevent_retokenization in (False, True)
        ),
        pytest.param(
            _text_history_request(
                prompt_token_ids=[100, 99, 99, 101],
                generation_token_ids=[200, 2],
                current_content="image omitted from replayed history",
                prevent_retokenization=True,
            ),
            ["media tokens", "missing from message history"],
            id="exact-media-tokens-without-media-history",
        ),
    ],
)
async def test_prefix_stitching_rejects_inconsistent_history(request_json, expected_errors):
    app, client = _prefix_stitching_app()

    response = await app.test_client().post(CHAT_PATH, json=request_json)

    assert response.status_code == 400
    text = await response.get_data(as_text=True)
    for expected_error in expected_errors:
        assert expected_error in text
    assert client.prompt_tokens == []


_EXPANDED_PREFIX_6 = {PREFIX_MEDIA_COUNT_FIELD: 1, PREFIX_EXPANDED_TOKEN_COUNT_FIELD: 6}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tokenizer_cls", "request_json", "tokenize", "expected_prompt", "expected_offload"),
    [
        # Retokenized history submits the compact template render as-is.
        pytest.param(
            _PrefixStitchingTokenizer,
            _image_history_request(
                [200, 2], current_content=[_TEXT_PART], prevent_retokenization=False
            ),
            _fake_tokenize(_MEDIA_HISTORY, [30]),
            [*_MEDIA_HISTORY, 30],
            None,
            id="retokenized",
        ),
        pytest.param(
            _PrefixStitchingTokenizer,
            _image_history_request(
                [200, 2], current_content=[_TEXT_PART, _IMAGE_BLOCK], prevent_retokenization=False
            ),
            _fake_tokenize(_MEDIA_HISTORY, [30, 42]),
            [*_MEDIA_HISTORY, 30, 42],
            None,
            id="retokenized-with-new-media",
        ),
        # An exact prefix replays the history tokens before the compact suffix; the metadata lets
        # the engine expand the prefix's media.
        pytest.param(
            _PrefixStitchingTokenizer,
            _image_history_request([200, 2], current_content=[_TEXT_PART]),
            _fake_tokenize(_MEDIA_HISTORY, [30]),
            [100, 99, 99, 101, 200, 2, 30],
            _EXPANDED_PREFIX_6,
            id="exact-prefix",
        ),
        pytest.param(
            _PrefixStitchingTokenizer,
            _image_history_request([200, 2], current_content=[_TEXT_PART, _IMAGE_BLOCK]),
            _fake_tokenize(_MEDIA_HISTORY, [30, 42]),
            [100, 99, 99, 101, 200, 2, 30, 42],
            _EXPANDED_PREFIX_6,
            id="exact-prefix-with-new-media",
        ),
        # The exact prefix may end in any of the model's EOS ids.
        pytest.param(
            _MultiEosPrefixStitchingTokenizer,
            _image_history_request([200, 11]),
            _fake_tokenize(_MEDIA_HISTORY, [30]),
            [100, 99, 99, 101, 200, 11, 30],
            _EXPANDED_PREFIX_6,
            id="every-model-eos",
        ),
        # A template BOS among the replayed tokens collapses; add_BOS decides whether one is kept
        # and counted.
        pytest.param(
            _BosPrefixStitchingTokenizer,
            _image_history_request([200, 2], [1, 100, 99, 99, 101], add_BOS=False),
            _fake_tokenize(_MEDIA_HISTORY, [30]),
            [100, 99, 99, 101, 200, 2, 30],
            _EXPANDED_PREFIX_6,
            id="bos-collapsed",
        ),
        pytest.param(
            _BosPrefixStitchingTokenizer,
            _image_history_request([200, 2], [1, 100, 99, 99, 101], add_BOS=True),
            _fake_tokenize(_MEDIA_HISTORY, [30]),
            [1, 100, 99, 99, 101, 200, 2, 30],
            {PREFIX_MEDIA_COUNT_FIELD: 1, PREFIX_EXPANDED_TOKEN_COUNT_FIELD: 7},
            id="bos-kept",
        ),
        # A text-only prefix with media only in the new turn is stitched before media expansion.
        pytest.param(
            _PrefixStitchingTokenizer,
            _text_history_request(current_content=[_IMAGE_BLOCK], prevent_retokenization=True),
            _fake_tokenize(_TEXT_HISTORY, [30, 42]),
            [100, 101, 200, 201, 2, 30, 42],
            None,
            id="text-prefix-with-new-media",
        ),
    ],
)
async def test_multimodal_prefix_stitching_submits_exact_prefix_and_compact_suffix(
    tokenizer_cls, request_json, tokenize, expected_prompt, expected_offload
):
    app, client = _prefix_stitching_app(tokenizer=tokenizer_cls())

    with mock.patch.object(
        chat_completions_module, "_tokenize_with_media_slots_sync", side_effect=tokenize
    ):
        response = await app.test_client().post(CHAT_PATH, json=request_json)

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.prompt_tokens == [expected_prompt]
    assert client.offload_params == [expected_offload]
    assert client.multi_modal_data != [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("prevent_retokenization", [False, True])
@pytest.mark.parametrize("prefix_has_media", [False, True])
async def test_offloaded_prefix_stitching_metadata_covers_text_and_multimodal_history(
    prefix_has_media, prevent_retokenization
):
    app, client = _prefix_stitching_app()
    history = _MEDIA_HISTORY if prefix_has_media else _TEXT_HISTORY
    # A text-only history renders through the plain template path.
    tokenize_helper = (
        "_tokenize_with_media_slots_sync" if prefix_has_media else "_apply_chat_template_sync"
    )

    with mock.patch.object(
        chat_completions_module, tokenize_helper, side_effect=_fake_tokenize(history, [30])
    ):
        response = await app.test_client().post(
            CHAT_PATH,
            json={
                "messages": [
                    _user([_IMAGE_BLOCK] if prefix_has_media else "first question"),
                    {"role": "assistant", "content": "first answer"},
                    _user("second question"),
                ],
                "offload_params": {"stager": {"request_id": "r0"}},
                "prevent_retokenization": prevent_retokenization,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.prompt_tokens == [[*history, 30]]
    assert client.offload_params == [
        {
            "stager": {"request_id": "r0"},
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD: history,
            PREFIX_EOS_TOKEN_ID_FIELD: [2],
            **({PREFIX_MEDIA_COUNT_FIELD: 1} if prefix_has_media else {}),
        }
    ]


@pytest.mark.asyncio
async def test_n_choices_prepare_and_serialize_shared_media_once():
    class _Tokenizer(_MediaSlotTokenizer):
        chat_template = "test-template"
        eod = None

        def apply_chat_template(self, messages, **_kwargs):
            return "".join(message["content"] for message in messages)

        def detokenize(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    class _Client(ReplyingClient):
        """Serializes each choice's media at submission, as the real client would."""

        def __init__(self):
            super().__init__()
            self.serialized_media = []

        def add_request_with_id(
            self, prompt_tokens, sampling_params, *, multi_modal_data=None, **kwargs
        ):
            self.serialized_media.append(serialize_multimodal_data(multi_modal_data))
            return super().add_request_with_id(
                prompt_tokens, sampling_params, multi_modal_data=multi_modal_data, **kwargs
            )

    app, client = _prefix_stitching_app(tokenizer=_Tokenizer(rendered=None), client=_Client())
    message = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": _image_data_url(b"shared-image")}}],
    }

    with mock.patch(
        "megatron.core.inference.inference_request.compute_media_cache_key",
        wraps=compute_media_cache_key,
    ) as compute_key:
        response = await app.test_client().post(
            CHAT_PATH, json={"messages": [message], "n": 3, "max_tokens": 1}
        )

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert len((await response.get_json())["choices"]) == 3
    assert len(client.serialized_media) == 3
    assert all(wire == client.serialized_media[0] for wire in client.serialized_media)
    assert all(wire is not client.serialized_media[0] for wire in client.serialized_media[1:])
    assert compute_key.call_count == 1


# --- HTTP: request validation and tokenization ----------------------------------


class _RecordingTokenizer(Tokenizer):
    def __init__(self, prompt_tokens=(10, 11), error=None):
        self.prompt_tokens = list(prompt_tokens)
        self.error = error
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return list(self.prompt_tokens)


class _BosRecordingTokenizer(_RecordingTokenizer):
    bos = 1


class _NoTemplateTokenizer(Tokenizer):
    chat_template = None

    def __init__(self):
        self.texts = []

    def tokenize(self, prompt):
        self.texts.append(prompt)
        return [10, 11]


def _wrapping(hf_tokenizer):
    """A Megatron tokenizer wrapping a Hugging Face one; the endpoint renders with the latter."""
    wrapper = _RecordingTokenizer()
    wrapper._tokenizer = SimpleNamespace(tokenizer=hf_tokenizer)
    return wrapper


_FTP_IMAGE = {"type": "image_url", "image_url": {"url": "ftp://example.com/a.png"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "app_config", "status", "expected_error"),
    [
        pytest.param({}, {}, 400, "Missing 'messages' field", id="missing-messages"),
        pytest.param(
            {"messages": "hi"}, {}, 400, "'messages' must be a list", id="messages-not-a-list"
        ),
        # A sampling field of the wrong type is a client error, whether the conversion raises
        # ValueError or TypeError.
        pytest.param(
            {**CHAT_BODY, "temperature": "hot"},
            {},
            400,
            NOT_A_NUMBER_ERROR,
            id="temperature-not-a-number",
        ),
        pytest.param({**CHAT_BODY, "top_k": [1]}, {}, 400, NOT_AN_INT_ERROR, id="top-k-list"),
        pytest.param(
            {"messages": [{"role": "user", "content": [_FTP_IMAGE]}]},
            {},
            400,
            "Failed to load image_url: Unsupported media URL scheme: 'ftp://example.com/a.png'",
            id="image-cannot-be-loaded",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": [_media_block("image", 0)]}]},
            {"tokenizer": _NoTemplateTokenizer()},
            400,
            "Invalid 'messages': Multimodal chat requests require a chat template.",
            id="media-without-a-chat-template",
        ),
        pytest.param(
            CHAT_BODY,
            {"tokenizer": _RecordingTokenizer(error=ValueError("bad role"))},
            400,
            "Invalid 'messages': bad role",
            id="template-value-error",
        ),
        pytest.param(
            CHAT_BODY,
            {"tokenizer": _RecordingTokenizer(error=RuntimeError("boom"))},
            500,
            "Error processing 'messages': boom",
            id="template-other-error",
        ),
    ],
)
async def test_malformed_chat_requests_are_rejected_before_submission(
    body, app_config, status, expected_error
):
    client = ReplyingClient([])
    app = build_app(CHAT_PATH, client, **app_config)

    response = await app.test_client().post(CHAT_PATH, json=body)

    assert response.status_code == status
    if expected_error is not None:
        assert await response.get_data(as_text=True) == expected_error
    assert client.prompt_tokens == []  # nothing reached the engine


@pytest.mark.asyncio
async def test_chat_without_a_template_joins_message_contents():
    tokenizer = _NoTemplateTokenizer()
    client = ReplyingClient()
    app = build_app(CHAT_PATH, client, tokenizer=tokenizer)
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]

    with pytest.warns(UserWarning, match="does not support 'apply_chat_template'"):
        response = await app.test_client().post(CHAT_PATH, json={"messages": messages})

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert tokenizer.texts == ["a\nb"]
    assert client.prompt_tokens == [[10, 11]]


@pytest.mark.asyncio
@pytest.mark.parametrize("server_template", [None, "server-template"])
async def test_chat_template_comes_from_the_server_never_the_request(server_template):
    tokenizer = _RecordingTokenizer()
    app = build_app(CHAT_PATH, ReplyingClient(), tokenizer=tokenizer, chat_template=server_template)
    body = {
        **CHAT_BODY,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "chat_template_kwargs": {"chat_template": "{{ request }}", "enable_thinking": False},
    }

    response = await app.test_client().post(CHAT_PATH, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    ((messages, kwargs),) = tokenizer.template_calls
    assert messages == CHAT_BODY["messages"]
    assert kwargs.get("chat_template") == server_template
    assert kwargs["enable_thinking"] is False
    assert (kwargs["tokenize"], kwargs["add_generation_prompt"]) == (True, True)
    # Tools reach the template with the parameters it expects.
    assert kwargs["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app_config", "body", "expected_prompt"),
    [
        pytest.param({}, {}, [10, 11], id="server-tokenizer"),
        pytest.param(
            {"tokenizer_copy": _RecordingTokenizer([20, 21])}, {}, [20, 21], id="private-copy"
        ),
        pytest.param(
            {"tokenizer": _wrapping(_RecordingTokenizer([30, 31]))},
            {},
            [30, 31],
            id="wrapped-hf-tokenizer",
        ),
        pytest.param(
            {"tokenizer_copy": _wrapping(_RecordingTokenizer([30, 31]))},
            {},
            [30, 31],
            id="private-copy-of-wrapped-hf-tokenizer",
        ),
        # A template may add its own BOS: duplicates collapse; add_BOS decides whether one stays.
        pytest.param(
            {"tokenizer": _BosRecordingTokenizer([1, 1, 10, 11])},
            {"add_BOS": False},
            [10, 11],
            id="template-bos-collapsed",
        ),
        pytest.param(
            {"tokenizer": _BosRecordingTokenizer([1, 1, 10, 11])},
            {"add_BOS": True},
            [1, 10, 11],
            id="template-bos-kept",
        ),
    ],
)
async def test_chat_prompt_comes_from_the_most_specific_tokenizer(
    app_config, body, expected_prompt
):
    client = ReplyingClient()
    app = build_app(CHAT_PATH, client, **{"tokenizer": _RecordingTokenizer(), **app_config})

    response = await app.test_client().post(CHAT_PATH, json={**CHAT_BODY, **body})

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.prompt_tokens == [expected_prompt]


# --- HTTP: response formatting ------------------------------------------------

_TOOLS = [
    {"type": "function", "function": {"name": "f1"}},
    {"type": "function", "function": {"name": "f2"}},
]
_TOOL_CALLS = [
    {"id": "c1", "function": {"name": "f1", "arguments": '{"x": 1}'}},
    {"id": "c2", "function": {"name": "f2", "arguments": {"y": 2}}},
]
_NORMALIZED_TOOL_CALLS = [
    {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": '{"x": 1}'}},
    {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": '{"y": 2}'}},
]
_TOOL_PARSE_RESULT = ("parsed", {"tool_calls": _TOOL_CALLS, "reasoning": "thinking"})
_TOOL_MESSAGE = {
    "content": "parsed",
    "tool_calls": _NORMALIZED_TOOL_CALLS,
    "reasoning_content": "thinking",
}
_NAMED_TOOL_CHOICE = {"type": "function", "function": {"name": "f1"}}
_CHAT_LOGPROBS = {
    "content": [
        {
            "token": "<30>",
            "logprob": -0.5,
            "bytes": [60, 51, 48, 62],
            "top_logprobs": [
                {"token": "30", "logprob": -0.5, "bytes": [51, 48]},
                {"token": "7", "logprob": -9999.0, "bytes": [55]},
            ],
        },
        {"token": "<31>", "logprob": -0.25, "bytes": [60, 51, 49, 62], "top_logprobs": []},
    ]
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "body",
        "app_config",
        "parse_result",
        "expected_message",
        "expected_finish_reason",
        "expected_logprobs",
    ),
    [
        pytest.param({}, {}, None, {"content": "<30><31>"}, "stop", None, id="compact-message"),
        pytest.param(
            {"return_tokenized_data": True, "return_raw_text": True},
            {},
            None,
            {
                "content": "<30><31>",
                "prompt_token_ids": [10, 2],
                "generation_token_ids": [30, 31],
                "raw_text": "<10><2><30><31>",
            },
            "stop",
            None,
            id="token-ids-and-raw-text",
        ),
        # Outside eval mode prevent_retokenization is on by default, which echoes the token ids.
        pytest.param(
            {},
            {"eval_mode": False},
            None,
            {"content": "<30><31>", "prompt_token_ids": [10, 2], "generation_token_ids": [30, 31]},
            "stop",
            None,
            id="rl-default-echoes-token-ids",
        ),
        # Only the OpenAI block is clamped for JSON; the message-level list is raw engine output.
        pytest.param(
            {"logprobs": True},
            {},
            None,
            {"content": "<30><31>"},
            "stop",
            _CHAT_LOGPROBS,
            id="logprobs",
        ),
        pytest.param(
            {},
            {},
            ("answer", {"reasoning": "why"}),
            {"content": "answer", "reasoning_content": "why"},
            "stop",
            None,
            id="reasoning-without-tool-calls",
        ),
        # finish_reason follows vLLM: "tool_calls" under auto/required, "stop" for a named tool.
        pytest.param(
            {"tools": _TOOLS},
            {},
            _TOOL_PARSE_RESULT,
            _TOOL_MESSAGE,
            "tool_calls",
            None,
            id="tool-calls-auto",
        ),
        pytest.param(
            {"tools": _TOOLS, "tool_choice": "required"},
            {},
            _TOOL_PARSE_RESULT,
            {**_TOOL_MESSAGE, "content": ""},
            "tool_calls",
            None,
            id="tool-choice-required-empties-content",
        ),
        pytest.param(
            {"tools": _TOOLS, "tool_choice": _NAMED_TOOL_CHOICE},
            {},
            _TOOL_PARSE_RESULT,
            {**_TOOL_MESSAGE, "content": ""},
            "stop",
            None,
            id="named-tool-choice-reports-stop",
        ),
        pytest.param(
            {"tools": _TOOLS, "parallel_tool_calls": False},
            {},
            _TOOL_PARSE_RESULT,
            {**_TOOL_MESSAGE, "tool_calls": _NORMALIZED_TOOL_CALLS[:1]},
            "tool_calls",
            None,
            id="parallel-tool-calls-disabled-keeps-first",
        ),
        # Incidental tool-call syntax is ignored when the client opted out of tools.
        pytest.param(
            {"tools": _TOOLS, "tool_choice": "none"},
            {},
            _TOOL_PARSE_RESULT,
            {"content": "<30><31>", "reasoning_content": "thinking"},
            "stop",
            None,
            id="tool-choice-none-drops-tool-calls",
        ),
    ],
)
async def test_chat_response_format(
    body, app_config, parse_result, expected_message, expected_finish_reason, expected_logprobs
):
    replies = [
        completed_reply(
            "chat-0",
            [10, 2],
            [30, 31],
            num_cached_tokens=2,
            routing_indices=[5, 6, 7, 8],
            generated_log_probs=[-0.5, -0.25],
            # Only the first position reports a top-N distribution.
            generated_top_n_logprobs=[{"30": -0.5, "7": float("-inf")}],
        ),
        # n=2 fan-out: the second choice hit its token limit.
        completed_reply("chat-1", [10, 2], [40], sampling_params={"num_tokens_to_generate": 1}),
    ]
    client = ReplyingClient(replies)
    parsers = ["p"] if parse_result is not None else []
    app = build_app(CHAT_PATH, client, parsers=parsers, **app_config)

    with mock.patch.object(chat_completions_module, "PARSER_MAPPING", {"p": _parser(parse_result)}):
        response = await app.test_client().post(CHAT_PATH, json={**CHAT_BODY, "n": 2, **body})

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["id"] == "chat-0"
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "EMPTY"
    assert payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    choice_0, choice_1 = payload["choices"]
    assert (choice_0["index"], choice_1["index"]) == (0, 1)
    assert choice_0["message"] == {
        "role": "assistant",
        "generation_log_probs": [-0.5, -0.25],
        **expected_message,
    }
    assert choice_0["finish_reason"] == expected_finish_reason
    assert choice_0["logprobs"] == expected_logprobs
    assert (choice_0["moe_topk_indices"], choice_0["prompt_moe_topk_indices"]) == (
        [5, 6, 7, 8],
        [5, 6],
    )
    # The second choice is parsed the same way, but hitting the token limit outranks tool calls.
    assert ("tool_calls" in choice_1["message"]) is ("tool_calls" in expected_message)
    assert choice_1["finish_reason"] == "length"
    assert "moe_topk_indices" not in choice_1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logprobs", "reply_fields", "warns"),
    [
        (True, {}, True),
        (False, {}, False),
        (True, {"generated_log_probs": [-0.5, -0.25]}, False),
        (True, {"payload_offloaded": True}, False),
    ],
    ids=["missing", "not-requested", "present", "offloaded"],
)
async def test_chat_warns_when_generation_log_probs_are_missing(
    logprobs, reply_fields, warns, caplog
):
    client = ReplyingClient([completed_reply("chat-0", [10, 2], [30, 31], **reply_fields)])
    app = build_app(CHAT_PATH, client)
    body = {**CHAT_BODY, "logprobs": logprobs}

    with caplog.at_level(logging.WARNING):
        response = await app.test_client().post(CHAT_PATH, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert ("Generation log probs is None" in caplog.text) is warns


# The registered parsers end to end: with the model's text fixed, the message is whatever the
# chain of reasoning and tool parsers makes of it. The table above pins the formatting of a parse
# result; this one pins what the real parsers produce.


class _DecodesTo(Tokenizer):
    """Detokenizes every generation to one fixed string."""

    def __init__(self, text):
        self.text = text

    def detokenize(self, token_ids, **kwargs):
        del token_ids, kwargs
        return self.text


def _qwen_call(name):
    return f"<tool_call><function={name}><parameter=count>7</parameter></function></tool_call>"


_EXECUTE_TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "execute",
                "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}},
            },
        }
    ],
    "tool_choice": "auto",
}
_EXECUTE_CALL = [{"name": "execute", "arguments": '{"count": 7}'}]
_FINISH_TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}
_FINISH_CALL = (
    "</think><tool_call>\n<function=finish>\n<parameter=message>done</parameter>\n</function>\n"
    "</tool_call>"
)
_REASONING_PARSERS = ["nemotron-v3-reasoning", "qwen3-coder-tool"]
_THINKING = {"chat_template_kwargs": {"enable_thinking": True}}
# Either opts the request out of reasoning-only replies: the reasoning comes back as content.
_EXPLICIT_FALLBACKS = {
    "thinking-disabled": {"chat_template_kwargs": {"enable_thinking": False}},
    "force-nonempty": {"chat_template_kwargs": {"force_nonempty_content": True}},
}
_EMPTY_REASONING = {"empty": "", "close-only": "</think>", "open-close": "<think></think>"}
# One generated token, and a finish reason whose token budget agrees with it.
_STOP = {
    "generated_tokens": [12],
    "finish_reason": "stop",
    "sampling_params": {"num_tokens_to_generate": 2},
}
_LENGTH = {
    "generated_tokens": [12],
    "finish_reason": "length",
    "sampling_params": {"num_tokens_to_generate": 1},
}
_FINISH_REASONS = {"stop": _STOP, "length": _LENGTH}

_REAL_PARSER_ROWS = [
    # qwen3-coder-tool normalizes the function name before the argument-schema lookup.
    *(
        pytest.param(
            _qwen_call(name),
            ["qwen3-coder-tool"],
            _EXECUTE_TOOLS,
            _STOP,
            {"content": "", "tool_calls": _EXECUTE_CALL},
            "tool_calls",
            id=f"tool-name-{label}",
        )
        for label, name in {
            "plain": "execute",
            "leading-space": " execute",
            "trailing-space": "execute ",
            "surrounding-whitespace": " \texecute\n",
        }.items()
    ),
    # A whitespace-only name, no tools requested, or no parsers: the tool syntax stays text.
    *(
        pytest.param(
            _qwen_call(name), parsers, body, _STOP, {"content": _qwen_call(name)}, "stop", id=label
        )
        for label, name, parsers, body in (
            ("whitespace-only-tool-name", " \t\n", ["qwen3-coder-tool"], _EXECUTE_TOOLS),
            ("tool-syntax-without-tools", " execute ", ["qwen3-coder-tool"], {}),
            ("tool-syntax-without-parsers", " execute ", [], _EXECUTE_TOOLS),
        )
    ),
    # Reasoning without a final answer is a null content, as vLLM reports it.
    *(
        pytest.param(
            text,
            _REASONING_PARSERS,
            _THINKING,
            reply,
            {"content": None, "reasoning_content": reasoning},
            finish_reason,
            id=f"reasoning-only-{label}-{finish_reason}",
        )
        for label, text, reasoning in (
            ("unclosed", "still thinking", "still thinking"),
            ("closed", "thinking</think>", "thinking"),
        )
        for finish_reason, reply in _FINISH_REASONS.items()
    ),
    *(
        pytest.param(
            "still thinking",
            _REASONING_PARSERS,
            body,
            _LENGTH,
            {"content": "still thinking"},
            "length",
            id=f"reasoning-only-{label}-is-content",
        )
        for label, body in _EXPLICIT_FALLBACKS.items()
    ),
    pytest.param(
        "thinking</think>answer",
        _REASONING_PARSERS,
        _THINKING,
        _LENGTH,
        {"content": "answer", "reasoning_content": "thinking"},
        "length",
        id="reasoning-then-answer",
    ),
    *(
        pytest.param(
            reasoning + _FINISH_CALL,
            _REASONING_PARSERS,
            {**_THINKING, **_FINISH_TOOLS},
            _STOP,
            {
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [{"name": "finish", "arguments": '{"message": "done"}'}],
            },
            "tool_calls",
            id=f"{label}-reasoning-then-tool-call",
        )
        for label, reasoning in (("some", "thinking"), ("empty", ""))
    ),
    # Empty reasoning keeps the parser's marker, so it is reasoning-only too.
    *(
        pytest.param(
            text,
            _REASONING_PARSERS,
            _THINKING,
            reply,
            {"content": None, "reasoning_content": ""},
            finish_reason,
            id=f"empty-reasoning-{label}-{finish_reason}",
        )
        for label, text in _EMPTY_REASONING.items()
        for finish_reason, reply in _FINISH_REASONS.items()
    ),
    # Only the EOS token was generated: nothing to decode.
    pytest.param(
        "",
        _REASONING_PARSERS,
        _THINKING,
        {**_STOP, "generated_tokens": [2]},
        {"content": None, "reasoning_content": ""},
        "stop",
        id="empty-reasoning-immediate-eos",
    ),
    *(
        pytest.param(
            text,
            _REASONING_PARSERS,
            body,
            reply,
            {"content": ""},
            finish_reason,
            id=f"empty-reasoning-{label}-{fallback}-{finish_reason}",
        )
        for label, text in _EMPTY_REASONING.items()
        for fallback, body in _EXPLICIT_FALLBACKS.items()
        for finish_reason, reply in _FINISH_REASONS.items()
    ),
    *(
        pytest.param(
            text,
            [],
            _THINKING,
            reply,
            {"content": text},
            finish_reason,
            id=f"empty-reasoning-{label}-without-parsers-{finish_reason}",
        )
        for label, text in _EMPTY_REASONING.items()
        for finish_reason, reply in _FINISH_REASONS.items()
    ),
    pytest.param(
        "<think></think>answer",
        _REASONING_PARSERS,
        _THINKING,
        _LENGTH,
        {"content": "answer", "reasoning_content": ""},
        "length",
        id="empty-reasoning-then-answer",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "parsers", "body", "reply", "expected_message", "expected_finish_reason"),
    _REAL_PARSER_ROWS,
)
async def test_chat_real_parsers_shape_the_message(
    text, parsers, body, reply, expected_message, expected_finish_reason
):
    client = ReplyingClient([completed_reply("forced", [10, 11], **reply)])
    app = build_app(CHAT_PATH, client, parsers=parsers, tokenizer=_DecodesTo(text))
    budget = reply["sampling_params"]["num_tokens_to_generate"]

    response = await app.test_client().post(
        CHAT_PATH, json={**CHAT_BODY, "max_completion_tokens": budget, **body}
    )

    assert response.status_code == 200, await response.get_data(as_text=True)
    choice = (await response.get_json())["choices"][0]
    message = choice["message"]
    if "tool_calls" in message:
        # Ids are minted per call; the function is what the parser decided.
        message["tool_calls"] = [call["function"] for call in message["tool_calls"]]
    assert message == {"role": "assistant", "generation_log_probs": None, **expected_message}
    assert choice["finish_reason"] == expected_finish_reason


# --- HTTP: streaming ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "marker_prefixes", "named_tool_choice"),
    [
        ({}, (), False),
        ({"tools": _TOOLS}, ("<tool_call>",), False),
        ({"tools": _TOOLS, "tool_choice": "none"}, (), False),
        ({"tools": _TOOLS, "tool_choice": _NAMED_TOOL_CHOICE}, ("<tool_call>",), True),
    ],
    ids=["no-tools", "tools", "tool-choice-none", "named-tool-choice"],
)
async def test_chat_streaming_builds_a_parser_per_choice(
    body, marker_prefixes, named_tool_choice, monkeypatch
):
    parser = _parser(("<12><13>", {}))
    parser.streaming_markers = ("<tool_call>",)
    monkeypatch.setattr(chat_completions_module, "PARSER_MAPPING", {"p": parser})
    streaming_parser = mock.Mock(wraps=chat_completions_module.StreamingChatParser)
    monkeypatch.setattr(chat_completions_module, "StreamingChatParser", streaming_parser)
    app = build_streaming_app(CHAT_PATH, ReplyingClient(), monkeypatch, parsers=["p"])

    response = await app.test_client().post(
        CHAT_PATH, json={**CHAT_BODY, "n": 2, "stream": True, **body}
    )

    assert response.status_code == 200, await response.get_data(as_text=True)
    _, done = await sse_payloads(response)
    assert done
    assert streaming_parser.call_count == 2
    assert streaming_parser.call_args.kwargs == {
        "marker_prefixes": marker_prefixes,
        "named_tool_choice": named_tool_choice,
    }
