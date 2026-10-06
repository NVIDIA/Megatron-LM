# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the framework-facing dynamic inference engine factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest

import megatron.core.inference.engine_factory as factory_module
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.disaggregation.engine import (
    DisaggDynamicInferenceEngine,
    StateHandoffDynamicInferenceEngine,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import Status


class _DynamicEngine:
    requires_recurrent_state_dummy_slot = False

    def __init__(self, *, controller, context):
        self.controller = controller
        self.context = context


@pytest.mark.parametrize("native", [False, True])
def test_prefill_reply_registers_external_ownership_or_sends_compact_native_metadata(native):
    cls = DisaggDynamicInferenceEngine if native else StateHandoffDynamicInferenceEngine
    engine = object.__new__(cls)
    engine._disagg_config = object() if native else None
    engine._kv_transfer_role = "prefill"
    engine.local_metadata_ledger_enabled = False
    engine.payload_stager = None
    engine.socket_for_receiving_requests = MagicMock()
    handoff = {"request_id": 7, "block_ids": [1], "kv_meta": {"agent": "prefill"}}
    serialized = {"request_id": 7, "prompt_tokens": [1] * 16384, "disaggregated_params": handoff}
    request = SimpleNamespace(
        request_id=7,
        status=Status.COMPLETED,
        disaggregated_params=handoff,
        serialize=MagicMock(return_value=serialized),
    )
    engine._send_requests_to_coordinator([request])
    calls = engine.socket_for_receiving_requests.mock_calls
    if native:
        assert len(calls) == 1
        request.serialize.assert_not_called()
        expected = {"request_id": 7, "disaggregated_params": handoff}
    else:
        assert msgpack.unpackb(calls[0].args[0]) == [Headers.REGISTER_KV.value, 7]
        expected = serialized
    reply = calls[-1].args[0]
    assert msgpack.unpackb(reply[0]) == [Headers.ENGINE_REPLY.value, [[7, False]]]
    assert msgpack.unpackb(reply[1]) == expected


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


@pytest.mark.parametrize("disaggregated", [False, True])
def test_builds_and_configures_engine(mock_pipeline, disaggregated):
    context, wrapper_cls, controller, configure = mock_pipeline
    config = InferenceConfig(
        disaggregation_shards="tp=1,role=prefill+tp=1,role=decode" if disaggregated else None,
        enable_prefix_caching=True,
    )
    model = SimpleNamespace(config=MagicMock())

    engine = factory_module.build_dynamic_inference_engine(
        model=model,
        tokenizer="tokenizer",
        inference_config=config,
        inference_wrapper_cls=wrapper_cls,
    )

    assert type(engine) is (_DisaggEngine if disaggregated else _DynamicEngine)
    assert engine.context is context
    assert engine.controller is controller
    assert config.reserve_recurrent_state_dummy_slot is disaggregated
    wrapper_cls.assert_called_once_with(model, context)
    if disaggregated:
        configure.assert_called_once_with(engine)
    else:
        configure.assert_not_called()


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
