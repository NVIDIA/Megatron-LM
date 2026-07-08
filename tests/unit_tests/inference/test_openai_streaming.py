# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.openai_streaming import (
    openai_stream,
)

pytestmark = pytest.mark.asyncio


class _Tokenizer:
    def detokenize(self, tokens):
        return "".join(chr(ord("a") + token - 1) for token in tokens)


async def test_openai_stream_exposes_cumulative_tokens_and_log_probs():
    stream = AsyncStream(request_id=1, cancel=lambda: None)
    stream.put({"partial": {"request_id": 1, "new_tokens": [1, 2], "new_log_probs": [-0.1, -0.2]}})
    # Token 3 models a token completed before the engine's final reply and
    # therefore absent from its last partial frame.
    stream.put(
        {
            "final": {
                "prompt_tokens": [9, 9],
                "generated_tokens": [1, 2, 3],
                "generated_log_probs": [-0.1, -0.2, -0.3],
                "num_cached_tokens": 2,
                "sampling_params": {"num_tokens_to_generate": 3},
            }
        }
    )
    stream.finish()

    records = [
        record
        async for record in openai_stream(
            [stream], _Tokenizer(), chat=False, return_log_probs=True, include_usage=True
        )
    ]
    payloads = [json.loads(record.removeprefix("data: ")) for record in records[:-1]]

    first, reconciled, finished, usage = payloads
    assert first["choices"][0]["text"] == "ab"
    assert first["choices"][0]["generation_token_ids"] == [1, 2]
    assert first["choices"][0]["generation_log_probs"] == [-0.1, -0.2]
    assert first["choices"][0]["generated_text"] == "ab"
    assert first["choices"][0]["generated_length"] == 2
    assert first["choices"][0]["logprobs"]["token_logprobs"] == [-0.1, -0.2]
    assert first["choices"][0]["logprobs"]["text_offset"] == [0, 1]
    assert reconciled["choices"][0]["text"] == "c"
    assert reconciled["choices"][0]["generation_token_ids"] == [1, 2, 3]
    assert reconciled["choices"][0]["generation_log_probs"] == [-0.1, -0.2, -0.3]
    assert finished["choices"][0]["finish_reason"] == "length"
    assert usage["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    assert records[-1] == "data: [DONE]\n\n"
