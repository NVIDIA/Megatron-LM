# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the framework-facing dynamic inference engine factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import torch

import megatron.core.inference.engine_factory as factory_module
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
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


@pytest.mark.asyncio
async def test_native_disagg_requires_coordinator_launch():
    engine = object.__new__(DisaggDynamicInferenceEngine)
    engine._disagg_config = object()
    with pytest.raises(ValueError, match="launch_inference_coordinator=True"):
        await engine.start_listening_to_data_parallel_coordinator(
            launch_inference_coordinator=False
        )


@pytest.mark.parametrize("native, require_claim", [(True, False), (False, False), (False, True)])
def test_prefill_reply_registers_external_ownership_or_sends_compact_native_metadata(
    native, require_claim
):
    cls = DisaggDynamicInferenceEngine if native else StateHandoffDynamicInferenceEngine
    engine = object.__new__(cls)
    engine._initialize_disaggregation_state()
    engine._require_handoff_claim = require_claim
    engine._disagg_config = object() if native else None
    engine._kv_transfer_role = "prefill"
    engine.local_metadata_ledger_enabled = False
    engine.payload_stager = None
    engine.socket_for_receiving_requests = MagicMock()
    handoff = {"request_id": 7, "block_ids": [1], "kv_meta": {"agent": "prefill"}}
    serialized = {"request_id": 7, "prompt_tokens": [1] * 16384, "disaggregated_params": handoff}
    request = SimpleNamespace(
        request_id=7,
        prompt_tokens=torch.tensor(serialized["prompt_tokens"]),
        offload_params={"destination": "payload-store"},
        status=Status.COMPLETED,
        disaggregated_params=handoff,
        serialize=MagicMock(return_value=serialized),
    )
    engine._send_requests_to_coordinator([request])
    calls = engine.socket_for_receiving_requests.mock_calls
    if native:
        assert len(calls) == 1
        request.serialize.assert_not_called()
        expected = {
            "request_id": 7,
            "disaggregated_params": handoff,
            "prompt_frame": msgpack.packb(serialized["prompt_tokens"], use_bin_type=True),
            "offload_frame": msgpack.packb(request.offload_params, use_bin_type=True),
        }
    else:
        assert msgpack.unpackb(calls[0].args[0]) == [Headers.REGISTER_KV.value, 7, require_claim]
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


@pytest.mark.parametrize(
    "mtp_pattern, given, expected, draft_kv_plane",
    [
        ("M", None, ["M"], False),
        ("**-", None, ["*", "*", "-"], False),
        ("*-", None, ["*", "-"], True),
        ("M", ["*", "-"], ["*", "-"], True),
        (None, None, None, True),
    ],
    ids=["recurrent-head", "two-attention-head", "attention-head", "explicit-list", "non-hybrid"],
)
def test_mtp_layer_types_default_to_the_models_head(
    mock_pipeline, mtp_pattern, given, expected, draft_kv_plane
):
    """A hybrid MTP head's layer types come from the model unless the config names them.

    Left None, the context treats the head as one attention layer and gives a repeated head the
    draft KV plane; a recurrent or two-attention head would then run through it.
    """
    config = InferenceConfig(num_speculative_tokens=2, mtp_layer_type_list=given)
    model_config = SimpleNamespace(mtp_num_layers=1, mtp_use_repeated_layer=True)
    model = SimpleNamespace(config=model_config)
    if mtp_pattern is not None:
        model.mtp_pattern = mtp_pattern

    factory_module.build_dynamic_inference_engine(
        model=model,
        tokenizer="tokenizer",
        inference_config=config,
        inference_wrapper_cls=MagicMock(),
    )

    assert config.mtp_layer_type_list == expected
    assert (
        DynamicInferenceContext.should_enable_mtp_kv_cache(
            model_config, config.mtp_layer_type_list, config.num_speculative_tokens
        )
        is draft_kv_plane
    )
