# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import builtins
import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _sanitize_chat_template_kwargs,
    _sanitize_messages_for_template,
    bp,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.incremental_detokenizer import (
    HuggingFaceFastIncrementalDetokenizer,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.openai_streaming import (
    JSON_SAFE_LOGPROB_FLOOR,
    StreamingChatParser,
    json_safe_logprob,
    json_safe_logprobs,
    json_safe_top_n_logprobs,
    openai_stream,
)
from megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser import Qwen3CoderToolParser

NEG_INF = float("-inf")


class _Tokenizer:
    def detokenize(self, tokens):
        return "".join(chr(ord("a") + token - 1) for token in tokens)


class _IncrementalDetokenizer:
    def __init__(self):
        self._text = ""

    def update(self, tokens):
        delta = "".join(chr(ord("a") + token - 1) for token in tokens)
        self._text += delta
        return delta

    @property
    def text(self):
        return self._text

    @property
    def text_length(self):
        return len(self._text)


def _make_byte_level_fast_tokenizer():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    huggingface_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    return SimpleNamespace(
        _tokenizer=SimpleNamespace(tokenizer=huggingface_tokenizer, include_special_tokens=True)
    )


@pytest.mark.asyncio
async def test_chat_completions_uses_generated_logprobs_only_when_requested():
    quart = pytest.importorskip("quart")

    class FakeTokenizer:
        bos, chat_template = None, None

        @staticmethod
        def tokenize(_text):
            return [11, 12]

        @staticmethod
        def detokenize(tokens):
            return "".join(chr(ord("a") + token - 1) for token in tokens)

    class FakeInferenceClient:
        return_log_probs = []

        async def add_request(self, _prompt_tokens, sampling_params, multi_modal_data=None):
            self.return_log_probs.append(sampling_params.return_log_probs)
            generated_log_probs = [-0.25, -0.5] if sampling_params.return_log_probs else None
            return {
                "uid": "request-1",
                "status": "COMPLETED",
                "generated_text": "ab",
                "prompt_length": 2,
                "num_cached_tokens": 1,
                "generated_tokens": [1, 2],
                "generated_log_probs": generated_log_probs,
                "log_probs": [-9.0, -9.0],
                "generated_top_n_logprobs": [{"a": -0.25}, {"b": -0.5}],
                "policy_epoch": [],
                "kv_cache_epoch": [],
                "events": [],
                "sampling_params": {"num_tokens_to_generate": 2},
                "routing_indices": None,
            }

    fake_client = FakeInferenceClient()
    app = quart.Quart(__name__)
    app.config.update(client=fake_client, tokenizer=FakeTokenizer(), parsers=[], verbose=False)
    app.config["multimodal_prompt_config"] = MultimodalPromptConfig()
    app.register_blueprint(bp)
    client = app.test_client()
    request_body = {
        "messages": [{"role": "user", "content": "prompt"}],
        "max_tokens": 2,
        "prevent_retokenization": False,
    }

    with_logprobs = await client.post(
        "/v1/chat/completions", json={**request_body, "logprobs": True, "top_logprobs": 1}
    )
    without_logprobs = await client.post(
        "/v1/chat/completions", json={**request_body, "logprobs": False}
    )

    assert with_logprobs.status_code == without_logprobs.status_code == 200
    with_payload = await with_logprobs.get_json()
    without_payload = await without_logprobs.get_json()
    assert fake_client.return_log_probs == [True, False]
    assert [entry["logprob"] for entry in with_payload["choices"][0]["logprobs"]["content"]] == [
        -0.25,
        -0.5,
    ]
    assert without_payload["choices"][0]["logprobs"] is None


@pytest.mark.asyncio
async def test_openai_stream_emits_delta_chunks_and_terminal_metadata():
    stream = AsyncStream(request_id=1, cancel=lambda: None)
    # The -inf logprobs ride the real wire path: formatters must clamp them to the
    # JSON-safe floor before json.dumps.
    stream.put(
        {
            "partial": {
                "request_id": 1,
                "new_tokens": [1, 2],
                "new_log_probs": [-0.1, NEG_INF],
                "new_top_n_logprobs": [{"a": -0.01}, {"b": NEG_INF}],
            }
        }
    )
    # Token 3 models a token completed before the engine's final reply and
    # therefore absent from its last partial frame.
    stream.put(
        {
            "final": {
                "prompt_tokens": [9, 9],
                "generated_tokens": [1, 2, 3],
                "generated_log_probs": [-0.1, NEG_INF, -0.3],
                "generated_top_n_logprobs": [{"a": -0.01}, {"b": -0.02}, {"c": -0.03}],
                "num_cached_tokens": 2,
                "sampling_params": {"num_tokens_to_generate": 3},
            }
        }
    )
    stream.finish()

    records = [
        record
        async for record in openai_stream(
            [stream],
            _Tokenizer(),
            [_IncrementalDetokenizer()],
            chat=False,
            return_log_probs=True,
            include_usage=True,
        )
    ]
    payloads = [json.loads(record.removeprefix("data: ")) for record in records[:-1]]

    first, reconciled, finished, usage = payloads
    assert first["choices"][0]["text"] == "ab"
    assert "generation_token_ids" not in first["choices"][0]
    assert "generation_log_probs" not in first["choices"][0]
    assert "generated_text" not in first["choices"][0]
    assert "generated_length" not in first["choices"][0]
    assert first["choices"][0]["logprobs"]["token_logprobs"] == [-0.1, JSON_SAFE_LOGPROB_FLOOR]
    assert first["choices"][0]["logprobs"]["top_logprobs"] == [
        {"a": -0.01},
        {"b": JSON_SAFE_LOGPROB_FLOOR},
    ]
    assert first["choices"][0]["logprobs"]["text_offset"] == [0, 1]
    assert reconciled["choices"][0]["text"] == "c"
    assert reconciled["choices"][0]["logprobs"]["top_logprobs"] == [{"c": -0.03}]
    assert "generation_token_ids" not in reconciled["choices"][0]
    assert "generation_log_probs" not in reconciled["choices"][0]
    assert "generated_text" not in reconciled["choices"][0]
    assert "generated_length" not in reconciled["choices"][0]
    assert finished["choices"][0]["finish_reason"] == "length"
    assert finished["choices"][0]["generation_token_ids"] == [1, 2, 3]
    assert finished["choices"][0]["generation_log_probs"] == [-0.1, JSON_SAFE_LOGPROB_FLOOR, -0.3]
    assert finished["choices"][0]["generated_text"] == "abc"
    assert finished["choices"][0]["generated_length"] == 3
    assert usage["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    assert records[-1] == "data: [DONE]\n\n"


def test_huggingface_fast_incremental_detokenizer_requires_optional_dependencies(monkeypatch):
    original_import = builtins.__import__

    def import_without_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("transformers is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_transformers)

    with pytest.raises(ImportError, match="requires the tokenizers and transformers packages"):
        HuggingFaceFastIncrementalDetokenizer(_Tokenizer(), [])


@pytest.mark.asyncio
async def test_openai_stream_echoes_completion_prompt_before_generated_text():
    stream = AsyncStream(request_id=1, cancel=lambda: None)
    stream.put(
        {
            "partial": {
                "request_id": 1,
                "new_tokens": [1],
                "new_log_probs": [-0.1],
                "new_top_n_logprobs": [{"a": -0.01}],
                "prompt_log_probs": [NEG_INF],
                "prompt_top_n_logprobs": [{"z": -0.04}],
            }
        }
    )
    stream.put(
        {
            "final": {
                "prompt_tokens": [26, 26],
                "generated_tokens": [1],
                "generated_log_probs": [-0.1],
                "generated_top_n_logprobs": [{"a": -0.01}],
                "sampling_params": {"num_tokens_to_generate": 2},
            }
        }
    )
    stream.finish()

    records = [
        record
        async for record in openai_stream(
            [stream],
            _Tokenizer(),
            [_IncrementalDetokenizer()],
            chat=False,
            return_log_probs=True,
            echo_prompts=["zz"],
            prompt_token_ids=[[26, 26]],
        )
    ]
    payloads = [json.loads(record.removeprefix("data: ")) for record in records[:-1]]

    echoed, generated, finished = payloads
    assert echoed["choices"][0]["text"] == "zz"
    assert echoed["choices"][0]["logprobs"] == {
        "tokens": ["z", "z"],
        "token_logprobs": [None, JSON_SAFE_LOGPROB_FLOOR],
        "top_logprobs": [None, {"z": -0.04}],
        "text_offset": [0, 1],
    }
    assert generated["choices"][0]["text"] == "a"
    assert generated["choices"][0]["logprobs"]["text_offset"] == [2]
    assert finished["choices"][0]["finish_reason"] == "stop"
    assert records[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_stream_surfaces_failed_final_without_success_terminator():
    stream = AsyncStream(request_id=1, cancel=lambda: None)
    stream.put(
        {
            "final": {
                "status": "FAILED",
                "events": [{"type": "ERROR_NONTRANSIENT", "payload": "context length exceeded"}],
            }
        }
    )
    stream.finish()

    records = [
        record
        async for record in openai_stream(
            [stream], _Tokenizer(), [_IncrementalDetokenizer()], chat=True
        )
    ]

    assert records == ['data: {"error": {"message": "context length exceeded"}}\n\n']


@pytest.mark.asyncio
async def test_openai_stream_preserves_chat_top_logprobs_with_parser():
    stream = AsyncStream(request_id=1, cancel=lambda: None)
    stream.put(
        {
            "partial": {
                "request_id": 1,
                "new_tokens": [1],
                "new_log_probs": [-0.1],
                "new_top_n_logprobs": [{"a": NEG_INF}],
            }
        }
    )
    stream.put(
        {
            "final": {
                "prompt_tokens": [9],
                "generated_tokens": [1],
                "generated_log_probs": [-0.1],
                "generated_top_n_logprobs": [{"a": NEG_INF}],
                "sampling_params": {"num_tokens_to_generate": 2},
            }
        }
    )
    stream.finish()
    parser = StreamingChatParser(lambda text: (text, {}))

    records = [
        record
        async for record in openai_stream(
            [stream],
            _Tokenizer(),
            [_IncrementalDetokenizer()],
            chat=True,
            return_log_probs=True,
            chat_parsers=[parser],
        )
    ]
    payloads = [json.loads(record.removeprefix("data: ")) for record in records[:-1]]

    role, content, finished = payloads
    assert role["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert content["choices"][0]["delta"] == {"content": "a"}
    assert content["choices"][0]["logprobs"]["content"][0]["top_logprobs"] == [
        {"token": "a", "logprob": JSON_SAFE_LOGPROB_FLOOR, "bytes": [97]}
    ]
    assert finished["choices"][0]["finish_reason"] == "stop"
    assert records[-1] == "data: [DONE]\n\n"


def test_streaming_chat_parser_emits_structured_stable_tool_call_deltas():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    def parse(text):
        return Qwen3CoderToolParser.parse(text, tools=tools)

    parser = StreamingChatParser(parse, marker_prefixes=Qwen3CoderToolParser.streaming_markers)
    model_output = (
        "Checking. <tool_call><function=get_weather>"
        "<parameter=city>\nSF\n</parameter></function></tool_call>"
    )
    deltas = []
    for end in range(1, len(model_output) + 1):
        deltas.extend(parser.parse(model_output[:end]))
    deltas.extend(parser.parse(model_output, finished=True))

    assert "".join(delta.get("content", "") for delta in deltas) == "Checking. "
    tool_deltas = [tool_call for delta in deltas for tool_call in delta.get("tool_calls", [])]
    name_deltas = [
        tool_call
        for tool_call in tool_deltas
        if tool_call.get("function", {}).get("name") is not None
    ]
    assert len(name_deltas) == 1
    assert name_deltas[0]["index"] == 0
    assert name_deltas[0]["type"] == "function"
    assert name_deltas[0]["id"].startswith("call_")
    assert name_deltas[0]["function"] == {"name": "get_weather"}

    argument_text = "".join(
        tool_call.get("function", {}).get("arguments", "") for tool_call in tool_deltas
    )
    assert json.loads(argument_text) == {"city": "SF"}
    assert all(tool_call["index"] == 0 for tool_call in tool_deltas)
    assert (
        parser.finish_reason(
            {"generated_tokens": [1], "sampling_params": {"num_tokens_to_generate": 2}}
        )
        == "tool_calls"
    )


def test_streaming_chat_parser_handles_single_multi_turn_tool_call_request():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    request_payload = {
        "stream": True,
        "tools": tools,
        "messages": [
            {"role": "user", "content": "What is the weather in SF?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_previous",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_previous", "content": "62 F"},
            {"role": "user", "content": "What about NYC?"},
        ],
    }

    sanitized_messages = _sanitize_messages_for_template(request_payload["messages"])

    assert sanitized_messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "SF"}
    assert sanitized_messages[2] == {
        "role": "tool",
        "tool_call_id": "call_previous",
        "content": "62 F",
    }
    assert sanitized_messages[3] == {"role": "user", "content": "What about NYC?"}
    assert (
        request_payload["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city": "SF"}'
    )

    parser = StreamingChatParser(
        lambda text: Qwen3CoderToolParser.parse(text, tools=tools),
        marker_prefixes=Qwen3CoderToolParser.streaming_markers,
    )
    model_output = (
        "<tool_call><function=get_weather>"
        "<parameter=city>\nNYC\n</parameter></function></tool_call>"
    )
    deltas = []
    for end in range(1, len(model_output) + 1):
        deltas.extend(parser.parse(model_output[:end]))
    deltas.extend(parser.parse(model_output, finished=True))

    tool_deltas = [tool_call for delta in deltas for tool_call in delta.get("tool_calls", [])]
    assert (
        sum(tool_call.get("function", {}).get("name") == "get_weather" for tool_call in tool_deltas)
        == 1
    )
    tool_call_ids = [tool_call["id"] for tool_call in tool_deltas if "id" in tool_call]
    assert len(tool_call_ids) == 1
    assert tool_call_ids[0].startswith("call_")
    assert all(tool_call["index"] == 0 for tool_call in tool_deltas)
    argument_text = "".join(
        tool_call.get("function", {}).get("arguments", "") for tool_call in tool_deltas
    )
    assert json.loads(argument_text) == {"city": "NYC"}
    assert (
        parser.finish_reason(
            {"generated_tokens": [1], "sampling_params": {"num_tokens_to_generate": 2}}
        )
        == "tool_calls"
    )


def test_huggingface_fast_incremental_detokenizer_preserves_utf8_boundaries():
    tokenizer = _make_byte_level_fast_tokenizer()
    huggingface_tokenizer = tokenizer._tokenizer.tokenizer
    token_ids = huggingface_tokenizer.encode("😀 café", add_special_tokens=False)
    detokenizer = HuggingFaceFastIncrementalDetokenizer(tokenizer, [])

    streamed_text = "".join(detokenizer.update([token_id]) for token_id in token_ids)

    assert streamed_text == huggingface_tokenizer.decode(token_ids, skip_special_tokens=False)
    assert detokenizer.text == streamed_text
    assert detokenizer.text_length == len(streamed_text)


def test_huggingface_fast_incremental_detokenizer_uses_prompt_context():
    tokenizer = _make_byte_level_fast_tokenizer()
    huggingface_tokenizer = tokenizer._tokenizer.tokenizer
    prompt_token_ids = huggingface_tokenizer.encode("hello", add_special_tokens=False)
    generated_token_ids = huggingface_tokenizer.encode(" world", add_special_tokens=False)
    detokenizer = HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_token_ids)

    streamed_text = "".join(detokenizer.update([token_id]) for token_id in generated_token_ids)
    full_text = huggingface_tokenizer.decode(
        prompt_token_ids + generated_token_ids, skip_special_tokens=False
    )
    prompt_text = huggingface_tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

    assert full_text.startswith(prompt_text)
    assert streamed_text == full_text[len(prompt_text) :]


def test_huggingface_fast_incremental_detokenizer_skips_special_tokens():
    tokenizer = _make_byte_level_fast_tokenizer()
    huggingface_tokenizer = tokenizer._tokenizer.tokenizer
    huggingface_tokenizer.add_special_tokens({"additional_special_tokens": ["<special>"]})
    token_ids = huggingface_tokenizer.encode("a<special>b", add_special_tokens=False)
    detokenizer = HuggingFaceFastIncrementalDetokenizer(tokenizer, [])

    streamed_text = "".join(detokenizer.update([token_id]) for token_id in token_ids)

    assert streamed_text == huggingface_tokenizer.decode(token_ids, skip_special_tokens=True)
    assert streamed_text == "ab"


def test_incremental_detokenizer_rejects_unsupported_tokenizer():
    with pytest.raises(
        ValueError, match="Streaming is currently supported only for Hugging Face fast tokenizers"
    ):
        HuggingFaceFastIncrementalDetokenizer(_Tokenizer(), [])


@pytest.mark.parametrize(
    "raw_kwargs,expected",
    [
        # A caller-supplied Jinja template must never survive sanitization: the
        # server renders it synchronously, so an expensive one (nested range(),
        # unbounded string multiplication) would pin the worker's event loop.
        (
            {
                "chat_template": (
                    "{% for _ in range(100000) %}{% for _ in range(100000) %}"
                    "{% endfor %}{% endfor %}"
                )
            },
            {},
        ),
        (
            {"chat_template": "{{ 'a' * 100000000 }}", "enable_thinking": False},
            {"enable_thinking": False},
        ),
        # Documented, template-consumed flags stay untouched.
        (
            {"enable_thinking": True, "force_nonempty_content": True},
            {"enable_thinking": True, "force_nonempty_content": True},
        ),
        ({}, {}),
        # Malformed bodies degrade to "no kwargs" rather than raising.
        (None, {}),
        ("chat_template", {}),
        ([{"chat_template": "x"}], {}),
    ],
)
def test_sanitize_chat_template_kwargs_strips_request_supplied_template(raw_kwargs, expected):
    assert _sanitize_chat_template_kwargs(raw_kwargs) == expected


def test_sanitize_chat_template_kwargs_does_not_mutate_caller_payload():
    raw_kwargs = {"chat_template": "{{ 'x' }}", "enable_thinking": False}

    sanitized = _sanitize_chat_template_kwargs(raw_kwargs)

    assert sanitized == {"enable_thinking": False}
    assert raw_kwargs["chat_template"] == "{{ 'x' }}"


def test_json_safe_helpers_clamp_only_non_finite():
    floor = JSON_SAFE_LOGPROB_FLOOR
    for value, expected in [
        (NEG_INF, floor),
        (float("inf"), floor),
        (float("nan"), floor),
        (0.0, 0.0),
        (-0.25, -0.25),
        # Values below the floor are already finite; they are not clamped.
        (floor - 1.0, floor - 1.0),
    ]:
        assert json_safe_logprob(value) == expected
    assert json_safe_logprobs([-0.5, NEG_INF]) == [-0.5, floor]
    assert json_safe_top_n_logprobs([{"a": -0.5, "b": NEG_INF}, None]) == [
        {"a": -0.5, "b": floor},
        None,
    ]
    # The hazard being guarded against: orjson encodes non-finite floats as null.
    orjson = pytest.importorskip("orjson")
    assert orjson.dumps(NEG_INF) == b"null"
    assert orjson.loads(orjson.dumps(json_safe_logprob(NEG_INF))) == floor
