# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for configurable defaults on the dynamic text generation server."""

import pytest

quart = pytest.importorskip("quart")
from quart import Quart

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    bp as chat_completions_blueprint,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions import (
    bp as completions_blueprint,
)


class _Tokenizer:
    chat_template = "test-template"
    bos = None

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [10, 11]

    def tokenize(self, prompt):
        del prompt
        return [10, 11]


class _CapturingClient:
    def __init__(self):
        self.sampling_params = []

    async def add_request(self, prompt_tokens, sampling_params, *, multi_modal_data=None):
        del prompt_tokens, multi_modal_data
        self.sampling_params.append(sampling_params)
        raise RuntimeError("stop after request submission")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "eval_mode",
        "request_overrides",
        "expected_top_p",
        "expected_top_k",
        "expected_prompt_tokens",
    ),
    [
        (False, {}, 0.95, 20, True),
        (True, {}, 0.95, 20, False),
        (True, {"top_p": 0.8, "top_k": 5, "prevent_retokenization": True}, 0.8, 5, True),
        (True, {"return_tokenized_data": True}, 0.95, 20, True),
    ],
)
async def test_chat_request_uses_server_defaults(
    eval_mode, request_overrides, expected_top_p, expected_top_k, expected_prompt_tokens
):
    app = Quart(__name__)
    inference_client = _CapturingClient()
    app.config.update(
        client=inference_client,
        tokenizer=_Tokenizer(),
        parsers=[],
        verbose=False,
        default_top_p=0.95,
        default_top_k=20,
        eval_mode=eval_mode,
    )
    app.register_blueprint(chat_completions_blueprint)

    payload = {"messages": [{"role": "user", "content": "hello"}], **request_overrides}
    response = await app.test_client().post("/v1/chat/completions", json=payload)

    assert response.status_code == 500
    assert len(inference_client.sampling_params) == 1
    sampling_params = inference_client.sampling_params[0]
    assert sampling_params.top_p == expected_top_p
    assert sampling_params.top_k == expected_top_k
    assert sampling_params.return_prompt_tokens is expected_prompt_tokens


@pytest.mark.asyncio
async def test_completions_request_uses_server_sampling_defaults():
    app = Quart(__name__)
    inference_client = _CapturingClient()
    app.config.update(
        client=inference_client,
        tokenizer=_Tokenizer(),
        verbose=False,
        default_top_p=0.9,
        default_top_k=0,
    )
    app.register_blueprint(completions_blueprint)

    response = await app.test_client().post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 500
    assert len(inference_client.sampling_params) == 1
    sampling_params = inference_client.sampling_params[0]
    assert sampling_params.top_p == 0.9
    assert sampling_params.top_k == 0
