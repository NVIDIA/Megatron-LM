# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for the legacy /completions endpoint of the static text generation server."""

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_restful")

from flask import Flask
from flask_restful import Api

from megatron.core.inference.text_generation_server.endpoints import completions
from megatron.core.tokenizers.text.libraries.null_tokenizer import NullTokenizer

GENERATED_TOKENS = [8, 9]
GENERATED_LOG_PROBS = [-0.25, -0.5]


class _Tokenizer:
    """The subset of the Megatron tokenizer interface that the endpoint uses."""

    def __init__(self):
        self._tokenizer = NullTokenizer(100)
        self.eod = self._tokenizer.eod

    def tokenize(self, text):
        return self._tokenizer.text_to_ids(text)

    def detokenize(self, ids):
        return self._tokenizer.ids_to_text(ids)

    def offsets(self, ids, text):
        return self._tokenizer.offsets(ids, text)


def _run_mcore_engine(tokenizer):
    """Mimic run_mcore_engine: log-probs are the prompt's (one per token after the first)
    followed by one per generated token."""

    def run(engine, prompts, *args, **kwargs):
        prompt_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]
        return {
            "text": [prompt + " " + tokenizer.detokenize(GENERATED_TOKENS) for prompt in prompts],
            "segments": [None] * len(prompts),
            "tokens": [tokens + GENERATED_TOKENS for tokens in prompt_tokens],
            "logprobs": [
                [-1.0] * (len(tokens) - 1) + GENERATED_LOG_PROBS for tokens in prompt_tokens
            ],
        }

    return run


@pytest.fixture
def client():
    tokenizer = _Tokenizer()
    app = Flask(__name__)
    engine = SimpleNamespace(controller=SimpleNamespace(tokenizer=tokenizer))
    Api(app).add_resource(
        completions.MegatronCompletions, "/completions", resource_class_args=[engine, None]
    )
    with (
        mock.patch.object(completions, "send_do_generate"),
        mock.patch.object(completions, "run_mcore_engine", _run_mcore_engine(tokenizer)),
    ):
        yield app.test_client()


def test_generation_logprobs_without_echo(client):
    response = client.post(
        "/completions", json={"prompt": "10 11 12", "max_tokens": 2, "echo": False, "stop": []}
    )

    choice = response.get_json()["choices"][0]
    assert choice["text"] == " 8 9"
    assert choice["logprobs"]["tokens"] == ["8", "9"]
    assert choice["logprobs"]["token_logprobs"] == GENERATED_LOG_PROBS


def test_token_id_prompt_uses_every_token(client):
    response = client.post(
        "/completions", json={"prompt": [10, 11, 12], "max_tokens": 2, "echo": True, "stop": []}
    )

    choice = response.get_json()["choices"][0]
    assert choice["text"] == "10 11 12 8 9"
    assert choice["logprobs"]["token_logprobs"] == [None, -1.0, -1.0] + GENERATED_LOG_PROBS
