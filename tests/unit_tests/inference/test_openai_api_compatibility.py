# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Is the inference HTTP frontend 1:1 compatible with the OpenAI API?

Responses are validated against the OpenAI SDK's own pydantic models
(``openai.types.Completion``, ``openai.types.chat.ChatCompletion``,
``ChatCompletionChunk``) rather than hand-written expectations. If a field is
renamed, retyped or dropped, the vendor's schema rejects it, which is a much
stronger statement about compatibility than asserting on keys we chose
ourselves.

Everything here runs against the real Quart app and the real endpoint handlers
via Quart's test client, with a fake InferenceClient in place of the engine.
There is no model, no coordinator and no GPU: the subject under test is the HTTP
layer's request parsing and response shaping.

The compatibility gaps this frontend has are recorded in
TestDocumentedOpenAIGaps rather than left implicit, so that closing one of them
fails a test and forces the list to be updated.
"""

import json

import pytest

pytest.importorskip("quart", reason="the OpenAI-compatible frontend requires Quart")
pytest.importorskip("openai", reason="response schemas come from the OpenAI SDK")

from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from megatron.core.inference.inference_request import Status
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (  # noqa: E501
    build_app,
)
from tests.unit_tests.inference.frontend_test_utils import (
    FakeInferenceClient,
    make_byte_level_fast_tokenizer,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def tokenizer():
    return make_byte_level_fast_tokenizer()


@pytest.fixture
def fake_client(tokenizer):
    return FakeInferenceClient(num_output_tokens=4, tokenizer=tokenizer)


@pytest.fixture
def http(fake_client, tokenizer):
    """A Quart test client wired to the real endpoint blueprints."""
    app = build_app(fake_client, tokenizer, parsers=None, verbose=False)
    return app.test_client()


def parse_sse(body: str):
    """Split an SSE body into (json_chunks, saw_done_terminator)."""
    chunks = []
    saw_done = False
    for block in body.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        data = block[len("data:") :].strip()
        if data == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(data))
    return chunks, saw_done


class TestResponseSchemas:
    """The response bodies must satisfy the OpenAI SDK's own models."""

    async def test_completions_response_matches_openai_schema(self, http):
        response = await http.post(
            "/v1/completions", json={"prompt": "Hello, world!", "max_tokens": 4}
        )
        assert response.status_code == 200
        body = await response.get_json()

        # The SDK model is the contract: it enforces required fields and types.
        parsed = Completion.model_validate(body)
        assert parsed.object == "text_completion"
        assert len(parsed.choices) == 1
        assert parsed.choices[0].finish_reason in ("stop", "length")
        assert parsed.usage.total_tokens == (
            parsed.usage.prompt_tokens + parsed.usage.completion_tokens
        )
        assert parsed.choices[0].text

    async def test_chat_completions_response_matches_openai_schema(self, http):
        response = await http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 4},
        )
        assert response.status_code == 200
        body = await response.get_json()

        parsed = ChatCompletion.model_validate(body)
        assert parsed.object == "chat.completion"
        assert len(parsed.choices) == 1
        assert parsed.choices[0].message.role == "assistant"
        assert parsed.choices[0].finish_reason in ("stop", "length", "tool_calls")
        assert parsed.usage.total_tokens == (
            parsed.usage.prompt_tokens + parsed.usage.completion_tokens
        )

    async def test_completions_logprobs_are_returned_and_aligned(self, http):
        """`logprobs` is an int on the completions API and shapes a Logprobs object.

        With echo=False every returned token is generated and therefore has a
        logprob, so all four parallel arrays must be the same length. A leading
        null in token_logprobs (which belongs to the echo=True case, where the
        first prompt token genuinely has none) both misaligns the arrays and
        fails strict validation against openai.types.Completion.
        """
        response = await http.post(
            "/v1/completions", json={"prompt": "Hello", "max_tokens": 4, "logprobs": 2}
        )
        assert response.status_code == 200
        body = await response.get_json()
        logprobs = body["choices"][0]["logprobs"]
        assert logprobs is not None
        assert len(logprobs["tokens"]) == 4
        assert len(logprobs["token_logprobs"]) == len(logprobs["tokens"])
        assert len(logprobs["text_offset"]) == len(logprobs["tokens"])
        assert logprobs["text_offset"] == sorted(logprobs["text_offset"])
        assert all(isinstance(value, float) for value in logprobs["token_logprobs"])
        Completion.model_validate(body)

    async def test_chat_completions_return_logprobs(self, http):
        """`logprobs: true` on the chat API must return per-token logprobs.

        The engine serializes this field as generated_log_probs. Reading any
        other name yields an empty logprobs.content that still satisfies the SDK
        schema, so only a value check catches it.
        """
        response = await http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 4,
                "logprobs": True,
                "top_logprobs": 2,
            },
        )
        assert response.status_code == 200
        parsed = ChatCompletion.model_validate(await response.get_json())
        assert parsed.choices[0].logprobs is not None
        assert len(parsed.choices[0].logprobs.content) == 4
        assert all(entry.logprob is not None for entry in parsed.choices[0].logprobs.content)


class TestStreamingSchemas:
    """Streamed chunks must satisfy the SDK chunk models and terminate correctly."""

    async def test_chat_stream_chunks_match_openai_schema(self, http):
        response = await http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 4,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.content_type.startswith("text/event-stream")

        chunks, saw_done = parse_sse((await response.get_data()).decode())
        assert saw_done, "stream must terminate with the OpenAI 'data: [DONE]' sentinel"
        assert chunks
        for chunk in chunks:
            ChatCompletionChunk.model_validate(chunk)
        # The first chunk announces the assistant role, and exactly one chunk
        # carries the terminal finish_reason.
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        finish_reasons = [
            choice["finish_reason"]
            for chunk in chunks
            for choice in chunk["choices"]
            if choice.get("finish_reason") is not None
        ]
        assert len(finish_reasons) == 1

    async def test_completions_stream_chunks_match_openai_schema(self, http):
        """Streamed completions reuse the non-streaming Completion shape.

        Only the terminal chunk is validated strictly: intermediate chunks carry
        ``finish_reason: null``, which the SDK's Completion model rejects but its
        streaming path tolerates because it constructs models without validating.
        The intermediate chunks are checked structurally instead.
        """
        response = await http.post(
            "/v1/completions", json={"prompt": "Hello", "max_tokens": 4, "stream": True}
        )
        assert response.status_code == 200
        chunks, saw_done = parse_sse((await response.get_data()).decode())
        assert saw_done
        assert chunks

        for chunk in chunks[:-1]:
            assert chunk["object"] == "text_completion"
            assert chunk["id"] and isinstance(chunk["created"], int)
            for choice in chunk["choices"]:
                assert choice["finish_reason"] is None
                assert isinstance(choice["text"], str)

        Completion.model_validate(chunks[-1])
        assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
        # Every chunk in a stream must share one response id.
        assert len({chunk["id"] for chunk in chunks}) == 1

    async def test_stream_usage_chunk_is_opt_in(self, http):
        """`stream_options.include_usage` adds a final usage-only chunk."""
        without = await http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "stream": True},
        )
        chunks, _ = parse_sse((await without.get_data()).decode())
        assert all(chunk.get("usage") is None for chunk in chunks)

        with_usage = await http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        chunks, saw_done = parse_sse((await with_usage.get_data()).decode())
        assert saw_done
        usage_chunks = [chunk for chunk in chunks if chunk.get("usage") is not None]
        assert len(usage_chunks) == 1
        # OpenAI sends the usage chunk last, with an empty choices list.
        assert usage_chunks[0] is chunks[-1]
        assert usage_chunks[0]["choices"] == []

    async def test_streamed_text_matches_non_streamed_text(self, http):
        """Concatenated deltas must equal the non-streaming completion text.

        These come from different code paths (incremental detokenizer vs. the
        engine's detokenized text), so they can silently diverge.
        """
        request = {"prompt": "Hello", "max_tokens": 6}
        non_streamed = await http.post("/v1/completions", json=request)
        expected = (await non_streamed.get_json())["choices"][0]["text"]

        streamed = await http.post("/v1/completions", json={**request, "stream": True})
        chunks, _ = parse_sse((await streamed.get_data()).decode())
        assembled = "".join(
            choice.get("text") or "" for chunk in chunks for choice in chunk["choices"]
        )
        assert assembled == expected


class TestRequestFieldMapping:
    """OpenAI request fields must reach the engine as the right sampling params."""

    @pytest.mark.parametrize(
        "request_body, attribute, expected",
        [
            ({"prompt": "hi", "max_tokens": 32}, "num_tokens_to_generate", 32),
            ({"prompt": "hi", "temperature": 0.7}, "temperature", 0.7),
            ({"prompt": "hi", "top_p": 0.9}, "top_p", 0.9),
            ({"prompt": "hi", "top_k": 5}, "top_k", 5),
            ({"prompt": "hi", "stop": "END"}, "stop_words", ["END"]),
            ({"prompt": "hi", "stop": ["A", "B"]}, "stop_words", ["A", "B"]),
            ({"prompt": "hi", "logprobs": 3}, "top_n_logprobs", 3),
            ({"prompt": "hi", "logprobs": 3}, "return_log_probs", True),
            # OpenAI has no ignore_eos; Megatron maps it to termination_id=-1.
            ({"prompt": "hi", "ignore_eos": True}, "termination_id", -1),
            # max_tokens defaults to 16 on the completions API.
            ({"prompt": "hi"}, "num_tokens_to_generate", 16),
        ],
    )
    async def test_completions_fields_map_to_sampling_params(
        self, http, fake_client, request_body, attribute, expected
    ):
        response = await http.post("/v1/completions", json=request_body)
        assert response.status_code == 200
        _, sampling_params = fake_client.submissions[-1]
        assert getattr(sampling_params, attribute) == expected

    async def test_temperature_zero_becomes_greedy(self, http, fake_client):
        """temperature=0 must select greedy decoding, as OpenAI clients expect."""
        response = await http.post("/v1/completions", json={"prompt": "hi", "temperature": 0.0})
        assert response.status_code == 200
        _, sampling_params = fake_client.submissions[-1]
        assert sampling_params.top_k == 1
        assert sampling_params.top_p == 0.0

    @pytest.mark.parametrize(
        "prompt, expected_choices",
        [
            ("single string", 1),
            (["first", "second"], 2),
            ([72, 101, 108], 1),
            ([[72, 101], [108, 111]], 2),
        ],
    )
    async def test_completions_accepts_every_openai_prompt_form(
        self, http, prompt, expected_choices
    ):
        """OpenAI allows str, list[str], list[int] and list[list[int]] prompts."""
        response = await http.post("/v1/completions", json={"prompt": prompt, "max_tokens": 2})
        assert response.status_code == 200
        parsed = Completion.model_validate(await response.get_json())
        assert len(parsed.choices) == expected_choices
        assert [choice.index for choice in parsed.choices] == list(range(expected_choices))

    async def test_chat_max_completion_tokens_supersedes_max_tokens(self, http, fake_client):
        """`max_completion_tokens` is OpenAI's replacement for the deprecated
        `max_tokens`, and must win when both are sent."""
        response = await http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "max_completion_tokens": 9,
            },
        )
        assert response.status_code == 200
        _, sampling_params = fake_client.submissions[-1]
        assert sampling_params.num_tokens_to_generate == 9

    async def test_chat_n_produces_n_choices(self, http):
        """The chat API honours `n` by fanning out to n engine requests."""
        response = await http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2, "n": 3},
        )
        assert response.status_code == 200
        parsed = ChatCompletion.model_validate(await response.get_json())
        assert len(parsed.choices) == 3
        assert [choice.index for choice in parsed.choices] == [0, 1, 2]


class TestErrorHandling:
    """Malformed and failing requests must produce the documented status codes."""

    @pytest.mark.parametrize(
        "endpoint, body",
        [
            ("/v1/completions", {}),
            ("/v1/completions", {"prompt": []}),
            ("/v1/completions", {"prompt": [{"nested": "object"}]}),
            ("/v1/chat/completions", {}),
            ("/v1/chat/completions", {"messages": "not-a-list"}),
        ],
    )
    async def test_invalid_requests_are_rejected_with_400(self, http, endpoint, body):
        response = await http.post(endpoint, json=body)
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "endpoint, body",
        [
            ("/v1/completions", {}),
            ("/v1/completions", {"prompt": "hi", "max_tokens": "not-an-int"}),
            ("/v1/chat/completions", {}),
            ("/v1/chat/completions", {"messages": "not-a-list"}),
        ],
    )
    async def test_errors_use_openai_error_envelope(self, http, endpoint, body):
        """Errors must be `{"error": {"message", "type", ...}}`.

        The OpenAI SDK reads error.message off the JSON body to build its
        exception. A plain-text body makes it raise a parse error instead, so the
        caller never sees why the request was rejected.
        """
        response = await http.post(endpoint, json=body)
        assert response.status_code >= 400
        error = (await response.get_json())["error"]
        assert error["message"]
        assert error["type"] == "invalid_request_error"

    async def test_nontransient_engine_failure_maps_to_400(self, http, fake_client):
        """A request the engine rejects outright is the caller's fault (400)."""
        fake_client.reply_override = {
            "status": "FAILED",
            "events": [{"type": "ERROR_NONTRANSIENT", "payload": "prompt too long"}],
        }
        response = await http.post("/v1/completions", json={"prompt": "hi"})
        assert response.status_code == 400
        assert "prompt too long" in (await response.get_json())["error"]["message"]

    async def test_transient_engine_failure_maps_to_500(self, http, fake_client):
        """A transient engine failure is the server's fault (500)."""
        fake_client.reply_override = {
            "status": "FAILED",
            "events": [{"type": "ERROR_TRANSIENT", "payload": "engine restarting"}],
        }
        response = await http.post("/v1/completions", json={"prompt": "hi"})
        assert response.status_code == 500
        assert (await response.get_json())["error"]["type"] == "server_error"

    async def test_context_length_error_keeps_its_exact_message(self, http, fake_client):
        """The max-context wording is load-bearing for Nemo-RL.

        chat_completions.py marks it DO NOT MODIFY. Wrapping errors in the OpenAI
        envelope moved it from the whole body to error.message, so this pins both
        the wording and its new location.
        """
        fake_client.reply_override = {
            "status": "FAILED",
            "events": [
                {"type": "ERROR_NONTRANSIENT", "payload": "MaxSequenceLengthOverflowError: 5000"}
            ],
        }
        response = await http.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 400
        error = (await response.get_json())["error"]
        assert "This model's maximum context length was exceeded." in error["message"]
        assert "Please reduce the length of the messages." in error["message"]
        assert error["code"] == "context_length_exceeded"


class TestDocumentedOpenAIGaps:
    """Known deviations from the OpenAI API.

    Each test asserts what an OpenAI client would expect and is marked xfail, so
    the gap is recorded here rather than in a comment somewhere. Closing a gap
    turns its test XPASS (strict), which fails the suite and forces this list to
    be updated -- that is the point.
    """

    @pytest.mark.xfail(strict=True, reason="GET /v1/models is not implemented")
    async def test_models_endpoint_exists(self, http):
        """openai.models.list() and many client health checks call /v1/models."""
        response = await http.get("/v1/models")
        assert response.status_code == 200
        body = await response.get_json()
        assert body["object"] == "list"
        assert body["data"]

    @pytest.mark.xfail(strict=True, reason="the completions endpoint ignores `n`")
    async def test_completions_honours_n(self, http):
        """`n` is read on /v1/chat/completions but silently dropped on
        /v1/completions, so a client asking for 4 completions gets 1."""
        response = await http.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 2, "n": 4}
        )
        assert response.status_code == 200
        parsed = Completion.model_validate(await response.get_json())
        assert len(parsed.choices) == 4

    @pytest.mark.xfail(strict=True, reason="completion ids use the chatcmpl- prefix")
    async def test_completion_id_uses_cmpl_prefix(self, http):
        """OpenAI ids are prefixed by endpoint: cmpl- for completions,
        chatcmpl- for chat. Both use chatcmpl- here."""
        response = await http.post("/v1/completions", json={"prompt": "hi"})
        body = await response.get_json()
        assert body["id"].startswith("cmpl-")

    @pytest.mark.xfail(strict=True, reason="/metrics is not implemented")
    async def test_metrics_endpoint_exists(self, http):
        """vLLM and OpenAI-compatible deployments expose Prometheus metrics."""
        response = await http.get("/metrics")
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "unsupported_field, value",
        [
            ("seed", 1234),
            ("logit_bias", {"50256": -100}),
            ("presence_penalty", 0.5),
            ("frequency_penalty", 0.5),
            ("best_of", 2),
            ("suffix", "!"),
        ],
    )
    async def test_unsupported_sampling_fields_are_silently_ignored(
        self, http, fake_client, unsupported_field, value
    ):
        """These OpenAI fields are accepted and dropped rather than rejected.

        Accepting-and-ignoring is deliberate (rejecting unknown fields would
        break clients that always send them), but `seed` in particular means a
        caller asking for reproducible output silently does not get it. This test
        pins the current behaviour so the choice stays a decision.
        """
        response = await http.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 2, unsupported_field: value}
        )
        assert response.status_code == 200
        _, sampling_params = fake_client.submissions[-1]
        assert not hasattr(sampling_params, unsupported_field)


class TestNonOpenAIExtensions:
    """Megatron-specific response fields that OpenAI clients must tolerate."""

    async def test_extra_choice_fields_do_not_break_sdk_parsing(self, http):
        """The completions endpoint adds prompt_token_ids, generation_token_ids
        and generation_log_probs to each choice. The SDK models allow extra
        fields, so this is additive rather than breaking -- but if a future
        field collided with an OpenAI field name, validation would catch it."""
        response = await http.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
        body = await response.get_json()
        choice = body["choices"][0]
        assert {"prompt_token_ids", "generation_token_ids", "generation_log_probs"} <= set(choice)
        Completion.model_validate(body)

    async def test_prompt_tokens_stay_off_the_wire_for_chat_by_default(self, http, fake_client):
        """Chat requests do not ask the engine for prompt tokens unless the
        caller wants them echoed, which keeps the prompt tensor off the reply."""
        await http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 2,
                "prevent_retokenization": False,
            },
        )
        _, sampling_params = fake_client.submissions[-1]
        assert sampling_params.return_prompt_tokens is False

    async def test_health_endpoint_reports_ready(self, http):
        """/health and /v1/health back the perf harness's readiness poll."""
        for path in ("/health", "/v1/health"):
            response = await http.get(path)
            assert response.status_code == 200
