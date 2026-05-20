# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the high-level inference APIs (``MegatronLLM`` /
``MegatronAsyncLLM``). Tests run without torch/megatron init by stubbing
the engine pipeline; the worker-rank tests bypass ``__init__`` entirely
via ``cls.__new__``."""

from unittest.mock import MagicMock

import pytest

import megatron.core.inference.apis._llm_base as base_mod
from megatron.core.inference.apis._llm_base import _MegatronLLMBase
from megatron.core.inference.apis.async_llm import MegatronAsyncLLM
from megatron.core.inference.apis.llm import MegatronLLM


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
        obj._serve_started = False
    return obj


class TestConstructorValidation:
    """Constructor-time validation for both ``MegatronLLM`` and ``MegatronAsyncLLM``."""

    @pytest.mark.parametrize(
        "extra_kwargs",
        [{"coordinator_host": "x"}, {"coordinator_port": 5000}],
    )
    def test_coordinator_host_or_port_without_use_coordinator_raises(
        self, mock_pipeline, fake_model_and_tokenizer, extra_kwargs
    ):
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="coordinator_host/port require use_coordinator=True"):
            MegatronLLM(model=model, tokenizer=tok, use_coordinator=False, **extra_kwargs)

    def test_megatron_llm_direct_mode_succeeds(self, mock_pipeline, fake_model_and_tokenizer):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        assert llm.is_primary_rank is True
        assert llm._use_coordinator is False

    def test_async_llm_requires_use_coordinator(self, mock_pipeline, fake_model_and_tokenizer):
        """``MegatronAsyncLLM`` rejects direct mode at ``__init__`` -- the
        engine's loop-bound primitives would collide with the caller's
        running asyncio loop."""
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="requires use_coordinator=True"):
            MegatronAsyncLLM(model=model, tokenizer=tok)


class TestLifecycleGuards:
    """Direct-mode lifecycle guards (``MegatronLLM`` only -- async direct is
    rejected at construction), and coordinator-mode worker-rank guards."""

    @pytest.mark.parametrize("method", ["pause", "unpause", "suspend", "resume"])
    def test_sync_lifecycle_raises_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer, method
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        with pytest.raises(RuntimeError, match="use_coordinator=True"):
            getattr(llm, method)()

    def test_sync_shutdown_is_noop_and_idempotent_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        llm.shutdown()
        assert llm._shutdown_called is True
        llm.shutdown()  # second call is a no-op
        assert llm._shutdown_called is True
        llm.wait_for_shutdown()  # also a no-op

    def test_sync_generate_raises_on_worker_rank(self):
        llm = _make_worker_instance(MegatronLLM)
        with pytest.raises(RuntimeError, match="primary rank"):
            llm.generate("hello")

    @pytest.mark.asyncio
    async def test_async_generate_raises_on_worker_rank(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        with pytest.raises(RuntimeError, match="primary rank"):
            await llm.generate("hello")


class TestNormalizePrompts:
    """Input-shape normalization (str / list[int] / list[str] / list[list[int]])."""

    @staticmethod
    def _normalize(prompts):
        obj = _MegatronLLMBase.__new__(_MegatronLLMBase)
        return obj._normalize_prompts(prompts)

    @pytest.mark.parametrize(
        "prompts,expected",
        [
            ("abc", (["abc"], False)),
            ([1, 2, 3], ([[1, 2, 3]], False)),
            (["a", "b"], (["a", "b"], True)),
            ([[1, 2], [3, 4]], ([[1, 2], [3, 4]], True)),
            ([], ([], True)),
        ],
    )
    def test_valid_inputs(self, prompts, expected):
        assert self._normalize(prompts) == expected

    @pytest.mark.parametrize("bad_input", [{1, 2}, 1.5, [1.5], {"k": "v"}])
    def test_unsupported_inputs_raise_typeerror(self, bad_input):
        with pytest.raises(TypeError):
            self._normalize(bad_input)
