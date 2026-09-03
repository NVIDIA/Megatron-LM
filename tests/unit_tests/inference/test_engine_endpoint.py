# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.inference.engine_endpoint import (
    InferenceEngineCapabilities,
    InferenceEngineEndpoint,
)


def _engine(tokenizer):
    return SimpleNamespace(
        context=SimpleNamespace(
            kv_block_allocator=SimpleNamespace(pool_size=17),
            max_sequence_length=8192,
            block_size_tokens=64,
            max_requests=8,
            max_tokens=1024,
            enable_prefix_caching=True,
        ),
        controller=SimpleNamespace(tokenizer=tokenizer),
    )


def test_endpoint_round_trip_describes_running_engine():
    endpoint = InferenceEngineEndpoint.from_engine(
        "tcp://127.0.0.1:5000", _engine(SimpleNamespace(eod=2)), logical_data_parallel_size=3
    )

    assert endpoint.capabilities.total_kv_blocks == 16
    assert endpoint.capabilities.bos_token_id == 2
    assert endpoint.capabilities.logical_data_parallel_size == 3
    assert InferenceEngineEndpoint.from_dict(endpoint.to_dict()) == endpoint


def test_capabilities_validate_registration_limits():
    values = InferenceEngineCapabilities.from_engine(
        _engine(SimpleNamespace(bos_token_id=1))
    ).to_dict()
    values["max_num_seqs"] = 0

    with pytest.raises(ValueError, match="max_num_seqs must be positive"):
        InferenceEngineCapabilities.from_dict(values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("context_length", "8192", "context_length must be an integer"),
        ("enable_prefix_caching", 1, "enable_prefix_caching must be a boolean"),
    ],
)
def test_capabilities_reject_coerced_wire_types(field_name, invalid_value, message):
    values = InferenceEngineCapabilities.from_engine(
        _engine(SimpleNamespace(bos_token_id=1))
    ).to_dict()
    values[field_name] = invalid_value

    with pytest.raises(TypeError, match=message):
        InferenceEngineCapabilities.from_dict(values)


def test_endpoint_rejects_non_string_coordinator_address():
    endpoint = InferenceEngineEndpoint.from_engine(
        "tcp://127.0.0.1:5000", _engine(SimpleNamespace(eod=2))
    ).to_dict()
    endpoint["coordinator_address"] = None

    with pytest.raises(TypeError, match="coordinator_address must be a string"):
        InferenceEngineEndpoint.from_dict(endpoint)
