# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the framework-facing dynamic inference engine factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import megatron.core.inference.engine_factory as factory_module
from megatron.core.inference.config import InferenceConfig


class _DynamicEngine:
    requires_recurrent_state_dummy_slot = False

    def __init__(self, *, controller, context):
        self.controller = controller
        self.context = context


class _DisaggEngine(_DynamicEngine):
    requires_recurrent_state_dummy_slot = True


@pytest.fixture
def mock_pipeline(monkeypatch):
    context = MagicMock(name="context")
    wrapper = MagicMock(name="wrapper")
    controller = MagicMock(name="controller")
    monkeypatch.setattr(factory_module, "DynamicInferenceContext", MagicMock(return_value=context))
    wrapper_cls = MagicMock(return_value=wrapper)
    monkeypatch.setattr(
        factory_module, "TextGenerationController", MagicMock(return_value=controller)
    )
    monkeypatch.setattr(factory_module, "DynamicInferenceEngine", _DynamicEngine)
    monkeypatch.setattr(factory_module, "DisaggDynamicInferenceEngine", _DisaggEngine)
    configure = MagicMock()
    monkeypatch.setattr(factory_module, "configure_prebuilt_disagg_engine", configure)
    return context, wrapper_cls, controller, configure


def test_builds_standard_engine(mock_pipeline):
    context, wrapper_cls, controller, configure = mock_pipeline
    config = InferenceConfig()
    model = SimpleNamespace(config=MagicMock())

    engine = factory_module.build_dynamic_inference_engine(
        model=model,
        tokenizer="tokenizer",
        inference_config=config,
        inference_wrapper_cls=wrapper_cls,
    )

    assert type(engine) is _DynamicEngine
    assert engine.context is context
    assert engine.controller is controller
    assert config.reserve_recurrent_state_dummy_slot is False
    wrapper_cls.assert_called_once_with(model, context)
    configure.assert_not_called()


def test_builds_and_configures_disaggregated_engine(mock_pipeline):
    _, wrapper_cls, _, configure = mock_pipeline
    config = InferenceConfig(
        disaggregation_shards="tp=1,role=prefill+tp=1,role=decode", enable_prefix_caching=True
    )

    engine = factory_module.build_dynamic_inference_engine(
        model=SimpleNamespace(config=MagicMock()),
        tokenizer="tokenizer",
        inference_config=config,
        inference_wrapper_cls=wrapper_cls,
    )

    assert type(engine) is _DisaggEngine
    assert config.reserve_recurrent_state_dummy_slot is True
    configure.assert_called_once_with(engine)


def test_disaggregation_rejects_incompatible_engine_override(mock_pipeline):
    config = InferenceConfig(
        disaggregation_shards="tp=1,role=prefill+tp=1,role=decode", enable_prefix_caching=True
    )
    with pytest.raises(ValueError, match="requires a DisaggDynamicInferenceEngine"):
        factory_module.build_dynamic_inference_engine(
            model=SimpleNamespace(config=MagicMock()),
            tokenizer="tokenizer",
            inference_config=config,
            engine_cls=_DynamicEngine,
        )


def test_disaggregation_rejects_disabled_prefix_caching(mock_pipeline):
    config = InferenceConfig(disaggregation_shards="tp=1,role=prefill+tp=1,role=decode")
    with pytest.raises(ValueError, match="requires prefix caching"):
        factory_module.build_dynamic_inference_engine(
            model=SimpleNamespace(config=MagicMock()),
            tokenizer="tokenizer",
            inference_config=config,
        )
