# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest

import megatron.inference._llm_base as base_mod
from megatron.inference.async_llm import MegatronAsyncLLM
from megatron.inference.serve_config import ServeConfig


@pytest.fixture
def mock_pipeline(monkeypatch):
    monkeypatch.setattr(base_mod, "DynamicInferenceContext", MagicMock())
    monkeypatch.setattr(base_mod, "GPTInferenceWrapper", MagicMock())
    monkeypatch.setattr(base_mod, "TextGenerationController", MagicMock())
    monkeypatch.setattr(base_mod, "DynamicInferenceEngine", MagicMock())


@pytest.fixture
def fake_model_and_tokenizer():
    model = MagicMock()
    model.config = MagicMock()
    tokenizer = MagicMock()
    return model, tokenizer


class TestAsyncLLMServeGuard:
    @pytest.mark.asyncio
    async def test_serve_requires_use_coordinator(
        self, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronAsyncLLM(model=model, tokenizer=tok)  # direct mode
        with pytest.raises(ValueError, match="requires use_coordinator=True"):
            await llm.serve(ServeConfig())

    @pytest.mark.asyncio
    async def test_direct_mode_generate_is_single_caller(
        self, mock_pipeline, fake_model_and_tokenizer, monkeypatch
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronAsyncLLM(model=model, tokenizer=tok)

        # Replace _generate_impl with a coroutine that holds until released,
        # so the first call stays in flight while the second one starts.
        gate = asyncio.Event()

        async def slow_impl(prompts, sp):
            await gate.wait()
            return [MagicMock() for _ in prompts]

        monkeypatch.setattr(llm, "_generate_impl", slow_impl)

        first = asyncio.create_task(llm.generate("hello"))
        # Yield control so ``first`` reaches the await on ``gate``.
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="single-caller"):
            await llm.generate("world")

        # Release the first call so the test cleans up.
        gate.set()
        await first
