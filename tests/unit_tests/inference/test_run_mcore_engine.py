# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the response built by the legacy text generation server's `run_mcore_engine`."""

import json
from types import SimpleNamespace

import pytest
import torch

import megatron.core.inference.text_generation_server.run_mcore_engine as run_mcore_engine_module
from megatron.core.inference.text_generation_server.run_mcore_engine import run_mcore_engine


class _FakeTokenizer:
    def detokenize(self, tokens):
        return "prompt"


class _FakeEngine:
    def __init__(self, result):
        self.controller = SimpleNamespace(tokenizer=_FakeTokenizer())
        self.result = result

    def get_new_request_id(self):
        return 0

    def generate(self, inference_requests):
        return self.result


@pytest.mark.parametrize(
    ("prompt_tokens", "generated_tokens", "prompt_log_probs", "generated_log_probs"),
    [
        # Static engine backed by the dynamic engine: tensor prompt, list generations.
        (torch.tensor([5, 6, 7]), [8, 9], [-0.5, -1.5], [-2.5, -3.5]),
        # Legacy static engine: list prompt, tensor generations.
        ([5, 6, 7], torch.tensor([8, 9]), [-0.5, -1.5], [-2.5, -3.5]),
        # Tensor log probs on both sides.
        (
            torch.tensor([5, 6, 7]),
            torch.tensor([8, 9]),
            torch.tensor([-0.5, -1.5]),
            torch.tensor([-2.5, -3.5]),
        ),
    ],
)
def test_response_joins_tensor_and_list_fields(
    monkeypatch, prompt_tokens, generated_tokens, prompt_log_probs, generated_log_probs
):
    monkeypatch.setattr(
        run_mcore_engine_module,
        "broadcast_float_list",
        lambda size, float_list, data_parallel: torch.tensor(float_list),
    )
    monkeypatch.setattr(
        run_mcore_engine_module,
        "tokenize_prompts",
        lambda **kwargs: (torch.tensor([[5, 6, 7, 0]]), torch.tensor([3])),
    )
    monkeypatch.setattr(
        run_mcore_engine_module, "mpu", SimpleNamespace(is_pipeline_first_stage=lambda: True)
    )
    result = [
        SimpleNamespace(
            generated_text=" out",
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prompt_log_probs=prompt_log_probs,
            generated_log_probs=generated_log_probs,
            segments=["prompt", " out"],
        )
    ]

    response = run_mcore_engine(
        _FakeEngine(result), prompts=["prompt"], logprobs=True, tokens_to_generate=2
    )

    assert response["text"] == ["prompt out"]
    assert response["tokens"] == [[5, 6, 7, 8, 9]]
    assert all(type(token) is int for token in response["tokens"][0])
    assert response["logprobs"] == [[-0.5, -1.5, -2.5, -3.5]]
    # The server returns this dict as JSON, so every field must serialize.
    json.dumps(response)
