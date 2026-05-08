# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

import pytest

import megatron.inference._llm_base as base_mod
from megatron.inference.async_llm import MegatronAsyncLLM
from megatron.inference.llm import MegatronLLM


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Stub out the engine pipeline so the constructor runs without torch/megatron."""
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


def _make_worker_instance(cls):
    """Build a coordinator-mode worker-rank instance without running the
    constructor's engine/runtime setup."""
    obj = cls.__new__(cls)
    obj._engine = MagicMock()
    obj._context = MagicMock()
    obj._controller = MagicMock()
    obj._use_coordinator = True
    obj._is_primary_rank = False
    obj._loop_manager = None
    obj._coord_runtime = None
    obj._shutdown_called = False
    if cls is MegatronAsyncLLM:
        obj._direct_generate_in_flight = False
        obj._serve_started = False
    return obj


class TestDirectModeLifecycleGuards:
    """Direct mode: pause/unpause/suspend/resume must raise; shutdown is a no-op."""

    def test_sync_lifecycle_raises_in_direct_mode(self, mock_pipeline, fake_model_and_tokenizer):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        for method in ("pause", "unpause", "suspend", "resume"):
            with pytest.raises(RuntimeError, match="use_coordinator=True"):
                getattr(llm, method)()
        # shutdown / wait_for_shutdown are no-ops, not errors.
        llm.shutdown()
        llm.wait_for_shutdown()

    @pytest.mark.asyncio
    async def test_async_lifecycle_raises_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronAsyncLLM(model=model, tokenizer=tok)
        for method in ("pause", "unpause", "suspend", "resume"):
            with pytest.raises(RuntimeError, match="use_coordinator=True"):
                await getattr(llm, method)()
        # shutdown / wait_for_shutdown are no-ops in direct mode.
        await llm.shutdown()
        await llm.wait_for_shutdown()


class TestCoordinatorWorkerRankGuards:
    """Coordinator mode + non-primary rank: generate must raise."""

    def test_sync_generate_raises_on_worker_rank(self):
        llm = _make_worker_instance(MegatronLLM)
        with pytest.raises(RuntimeError, match="primary rank"):
            llm.generate("hello")

    @pytest.mark.asyncio
    async def test_async_generate_raises_on_worker_rank(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        with pytest.raises(RuntimeError, match="primary rank"):
            await llm.generate("hello")


class TestShutdownIdempotence:
    def test_sync_shutdown_idempotent_in_direct_mode(self, mock_pipeline, fake_model_and_tokenizer):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        llm.shutdown()
        assert llm._shutdown_called is True
        # Second shutdown should be a no-op (idempotent).
        llm.shutdown()
        assert llm._shutdown_called is True

    @pytest.mark.asyncio
    async def test_async_shutdown_idempotent_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronAsyncLLM(model=model, tokenizer=tok)
        await llm.shutdown()
        assert llm._shutdown_called is True
        await llm.shutdown()
        assert llm._shutdown_called is True
