# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for `ChatRenderer`, the chat rendering and reply building shared with the chat endpoint.

Each behavior is pinned against the endpoint itself: what `render` returns must be what
`/v1/chat/completions` submits, and what `build_response` returns must be what it answers.
"""

import base64
import subprocess
import sys
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _replace_prefix_tokens,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_renderer import (
    ChatGeneration,
    ChatRenderer,
    RenderedChat,
)
from tests.unit_tests.inference.test_endpoints_common import (
    CHAT_PATH,
    ReplyingClient,
    build_app,
    completed_reply,
)

_CHAT_TEMPLATE = (
    "{{ bos_token }}{% if tools %}<|tools|>{{ tools | length }}{% endif %}"
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{{ eos_token }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": "not-a-dict"}}]
_IMAGE_URL = "data:image/png;base64," + base64.b64encode(b"img").decode()


class _MegatronTokenizer:
    """The surface the chat path reads off a Megatron text tokenizer that wraps HF."""

    def __init__(self, hf_tokenizer):
        self._tokenizer = SimpleNamespace(tokenizer=hf_tokenizer)
        self.hf = hf_tokenizer
        self.chat_template = hf_tokenizer.chat_template
        self.bos = hf_tokenizer.bos_token_id
        self.eod = hf_tokenizer.eos_token_id
        self.eos_id = hf_tokenizer.eos_token_id

    def tokenize(self, text):
        return self.hf.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids, skip_special_tokens=False, **kwargs):
        del kwargs
        return self.hf.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _tokenizer():
    """A byte-level fast tokenizer with BOS/EOS, an image token and a chat template."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    hf_tokenizer.add_special_tokens(
        {"bos_token": "<s>", "eos_token": "</s>", "additional_special_tokens": ["<image>"]}
    )
    hf_tokenizer.chat_template = _CHAT_TEMPLATE
    return _MegatronTokenizer(hf_tokenizer)


def _template_ids(tokenizer, messages, add_generation_prompt, **kwargs):
    """The template rendered to text, then encoded: an oracle independent of the renderer."""
    text = tokenizer.hf.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )
    return tokenizer.hf(text, add_special_tokens=False)["input_ids"]


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content, **extra):
    return {"role": "assistant", "content": content, **extra}


_TEXT_BODIES = {
    "single-turn": {"messages": [_user("hello")]},
    "multi-turn": {"messages": [_user("hi"), _assistant("yo"), _user("again")]},
    "tools": {"messages": [_user("hello")], "tools": _TOOLS},
    "structured-content": {"messages": [_user([{"type": "text", "text": "hello"}])]},
    "add-bos": {"messages": [_user("hello")], "add_BOS": True},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("body", _TEXT_BODIES.values(), ids=_TEXT_BODIES.keys())
async def test_render_returns_the_prompt_the_endpoint_submits(body):
    tokenizer = _tokenizer()
    client = ReplyingClient()
    app = build_app(CHAT_PATH, client, tokenizer=tokenizer)

    response = await app.test_client().post(CHAT_PATH, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    rendered = ChatRenderer(tokenizer, eval_mode=True).render(
        body, add_bos=body.get("add_BOS", False)
    )
    assert client.prompt_tokens == [rendered.prompt_token_ids]
    assert rendered.multi_modal_data is None and rendered.media_slots == []


def test_render_applies_the_template_tools_and_bos_policy():
    tokenizer = _tokenizer()
    renderer = ChatRenderer(tokenizer)
    body = {"messages": [_user("hello")], "tools": _TOOLS}

    raw = renderer.render(body)
    collapsed = renderer.render(body, add_bos=False)
    kept = renderer.render(body, add_bos=True)

    expected = _template_ids(
        tokenizer,
        [_user("hello")],
        True,
        tools=[
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    )
    assert raw.prompt_token_ids == expected
    assert raw.prompt_token_ids[0] == tokenizer.bos
    assert collapsed.prompt_token_ids == expected[1:]
    assert kept.prompt_token_ids == expected
    assert raw.template_tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert raw.eos_token_ids == frozenset({tokenizer.eod})


@pytest.mark.parametrize(
    ("messages", "with_history", "history_messages"),
    [
        pytest.param([_user("hi")], True, None, id="no-assistant-turn"),
        pytest.param([_user("hi"), _assistant("yo"), _user("again")], False, None, id="not-asked"),
        pytest.param(
            [_user("hi"), _assistant("yo"), _user("again")],
            True,
            [_user("hi"), _assistant("yo")],
            id="up-to-last-assistant",
        ),
        pytest.param(
            [_user("a"), _assistant("b"), _user("c"), _assistant("d"), _user("e")],
            True,
            [_user("a"), _assistant("b"), _user("c"), _assistant("d")],
            id="last-of-several",
        ),
    ],
)
def test_history_is_the_conversation_through_the_last_assistant_turn(
    messages, with_history, history_messages
):
    tokenizer = _tokenizer()

    rendered = ChatRenderer(tokenizer).render({"messages": messages}, with_history=with_history)

    if history_messages is None:
        assert rendered.history_token_ids is None
    else:
        expected = _template_ids(tokenizer, history_messages, False)
        assert rendered.history_token_ids == expected
        assert rendered.prompt_token_ids[: len(expected)] == expected


@pytest.mark.asyncio
async def test_history_and_eos_ids_reproduce_the_endpoints_previous_turn_splice():
    """An external caller splicing with `render`'s outputs submits what the endpoint submits."""
    tokenizer = _tokenizer()
    first_prompt = _template_ids(tokenizer, [_user("hi")], True)
    # The exact previous reply differs from its re-rendering, which is what the splice preserves.
    first_generation = tokenizer.tokenize("y o") + [tokenizer.eod]
    body = {
        "messages": [
            _user("hi"),
            _assistant("yo", prompt_token_ids=first_prompt, generation_token_ids=first_generation),
            _user("again"),
        ]
    }
    client = ReplyingClient()
    app = build_app(CHAT_PATH, client, tokenizer=tokenizer, eval_mode=False)

    response = await app.test_client().post(CHAT_PATH, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    renderer = ChatRenderer(tokenizer)
    rendered = renderer.render(body, with_history=True)
    spliced = _replace_prefix_tokens(
        rendered.eos_token_ids,
        first_prompt + first_generation,
        rendered.history_token_ids,
        rendered.prompt_token_ids,
    )
    assert client.prompt_tokens == [renderer.apply_bos_policy(spliced, add_bos=False)]
    assert client.prompt_tokens[0] != renderer.apply_bos_policy(
        rendered.prompt_token_ids, add_bos=False
    )


@pytest.mark.asyncio
async def test_render_lowers_media_blocks_like_the_endpoint():
    tokenizer = _tokenizer()
    body = {
        "messages": [
            _user(
                [
                    {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
                    {"type": "text", "text": "what"},
                ]
            )
        ]
    }
    client = ReplyingClient()
    app = build_app(CHAT_PATH, client, tokenizer=tokenizer)

    response = await app.test_client().post(CHAT_PATH, json=body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    rendered = ChatRenderer(tokenizer).render(body, add_bos=False)
    assert client.prompt_tokens == [rendered.prompt_token_ids]
    image_token_id = tokenizer.hf.convert_tokens_to_ids("<image>")
    assert rendered.prompt_token_ids.count(image_token_id) == 1
    assert rendered.multi_modal_data == {"image": [b"img"]}
    assert [slot[1:] for slot in rendered.media_slots] == [("image", 0)]


@pytest.mark.parametrize(
    ("body", "error"),
    [
        pytest.param({}, "Missing 'messages' field", id="missing"),
        pytest.param({"messages": {"role": "user"}}, "'messages' must be a list", id="not-a-list"),
        pytest.param(
            {"messages": [_user("hi")], "stream_options": 1},
            "'stream_options' must be an object",
            id="bad-option",
        ),
        pytest.param(
            {"messages": [_user([{"type": "video_url", "video_url": {"url": "https://x"}}])]},
            "must be base64 data URLs",
            id="bad-media",
        ),
    ],
)
def test_render_rejects_malformed_requests(body, error):
    with pytest.raises(ValueError, match=error):
        ChatRenderer(_tokenizer()).render(body)


def test_request_cannot_override_the_server_chat_template():
    tokenizer = _tokenizer()
    server_template = "{{ bos_token }}<|server|>"
    renderer = ChatRenderer(tokenizer, chat_template=server_template)

    rendered = renderer.render(
        {"messages": [_user("hi")], "chat_template_kwargs": {"chat_template": "{{ 'evil' }}"}}
    )

    assert rendered.options.chat_template_kwargs == {"chat_template": server_template}
    assert (
        rendered.prompt_token_ids
        == tokenizer.hf("<s><|server|>", add_special_tokens=False)["input_ids"]
    )


def test_eval_mode_sets_the_prevent_retokenization_default():
    body = {"messages": [_user("hi")]}
    assert ChatRenderer(_tokenizer()).parse_options(body).prevent_retokenization is True
    assert (
        ChatRenderer(_tokenizer(), eval_mode=True).parse_options(body).prevent_retokenization
        is False
    )


@pytest.mark.asyncio
async def test_tokenize_async_matches_tokenize():
    tokenizer = _tokenizer()
    renderer = ChatRenderer(tokenizer)
    messages = [_user("hi")]

    assert await renderer.tokenize_async(messages) == renderer.tokenize(messages)


_RESPONSE_BODIES = {
    "plain": ({}, "plain answer"),
    "reasoning": ({}, "<think>why</think>answer"),
    "reasoning-only": ({}, "<think>still thinking"),
    "logprobs": ({"logprobs": True, "top_logprobs": 1}, "ok"),
    "tokenized-data": ({"return_tokenized_data": True, "return_raw_text": True}, "ok"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("body", "text"), _RESPONSE_BODIES.values(), ids=_RESPONSE_BODIES.keys())
async def test_build_response_from_generations_matches_the_endpoint_body(body, text):
    tokenizer = _tokenizer()
    parsers = ["deepseek-r1-reasoning"]
    generated = tokenizer.tokenize(text) + [tokenizer.eod]
    log_probs = [-0.5] * len(generated)
    top_n = [{"x": -0.5}] * len(generated)
    request_body = {"messages": [_user("hello")], "max_tokens": 64, **body}
    client = ReplyingClient(
        [
            completed_reply(
                "chat-0",
                [1, 2, 3],
                generated,
                finish_reason="stop",
                num_cached_tokens=1,
                generated_log_probs=log_probs,
                generated_top_n_logprobs=top_n,
            )
        ]
    )
    app = build_app(CHAT_PATH, client, tokenizer=tokenizer, parsers=parsers)

    response = await app.test_client().post(CHAT_PATH, json=request_body)

    assert response.status_code == 200, await response.get_data(as_text=True)
    renderer = ChatRenderer(tokenizer, parsers, eval_mode=True)
    built = renderer.build_response(
        [
            ChatGeneration(
                generated_tokens=generated,
                finish_reason="stop",
                prompt_tokens=[1, 2, 3],
                num_cached_tokens=1,
                generated_log_probs=log_probs,
                generated_top_n_logprobs=top_n,
                uid="chat-0",
            )
        ],
        client.sampling_params[0],
        renderer.parse_options(request_body),
    )
    payload = await response.get_json()
    payload.pop("created")
    built.pop("created")
    assert built == payload


def test_build_response_reports_length_and_usage_without_an_engine():
    tokenizer = _tokenizer()
    renderer = ChatRenderer(tokenizer)
    options = renderer.parse_options({"messages": [_user("hi")]})
    sampling_params = SimpleNamespace(detokenize_stop_sequence=False, return_log_probs=False)

    built = renderer.build_response(
        [
            ChatGeneration(tokenizer.tokenize("ab"), "length", prompt_length=5, uid="r0"),
            ChatGeneration(tokenizer.tokenize("c") + [tokenizer.eod], prompt_length=5),
        ],
        sampling_params,
        options,
    )

    assert built["id"] == "r0"
    assert [choice["finish_reason"] for choice in built["choices"]] == ["length", "stop"]
    assert [choice["message"]["content"] for choice in built["choices"]] == ["ab", "c"]
    assert built["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 4,
        "total_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_rendered_chat_is_immutable():
    rendered = ChatRenderer(_tokenizer()).render({"messages": [_user("hi")]})
    assert isinstance(rendered, RenderedChat)
    with pytest.raises(AttributeError):
        rendered.prompt_token_ids = []


_SERVER_PACKAGE = "megatron.core.inference.text_generation_server.dynamic_text_gen_server"


@pytest.mark.parametrize(
    "first_import", [f"{_SERVER_PACKAGE}.endpoints.chat_renderer", _SERVER_PACKAGE]
)
def test_any_import_order_keeps_the_chat_blueprint(first_import):
    """`endpoints/__init__` drops blueprints on ImportError, so a cycle would lose the route."""
    code = (
        f"import importlib; importlib.import_module({first_import!r}); "
        f"from {_SERVER_PACKAGE}.endpoints import ChatCompletions"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
