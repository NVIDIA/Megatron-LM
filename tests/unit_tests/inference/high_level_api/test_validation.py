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


@pytest.mark.parametrize("cls", [MegatronLLM, MegatronAsyncLLM])
class TestConstructorValidation:
    def test_coordinator_host_without_use_coordinator_raises(
        self, cls, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="coordinator_host/port require use_coordinator=True"):
            cls(model=model, tokenizer=tok, use_coordinator=False, coordinator_host="x")

    def test_coordinator_port_without_use_coordinator_raises(
        self, cls, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="coordinator_host/port require use_coordinator=True"):
            cls(model=model, tokenizer=tok, use_coordinator=False, coordinator_port=5000)

    def test_direct_mode_constructor_succeeds(self, cls, mock_pipeline, fake_model_and_tokenizer):
        model, tok = fake_model_and_tokenizer
        llm = cls(model=model, tokenizer=tok)
        assert llm.is_primary_rank is True
        assert llm._use_coordinator is False
