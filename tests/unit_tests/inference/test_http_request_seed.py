# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest

from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.test_completions import (
    _ENDPOINTS,
    _build_app,
    _reply,
    _ReplyingClient,
)


class RecordingClient(_ReplyingClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.params = []

    def add_request_with_id(self, prompt_tokens, sampling_params, **kwargs):
        self.params.append(sampling_params.serialize())
        return super().add_request_with_id(prompt_tokens, sampling_params, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("blueprint,path,body", _ENDPOINTS)
@pytest.mark.parametrize("seed", [None, 0, 42, 2**63 - 1])
async def test_http_seed_reaches_engine(blueprint, path, body, seed):
    client = RecordingClient([_reply("r", [10], [12])])
    response = (
        await _build_app(blueprint, client).test_client().post(path, json={**body, "seed": seed})
    )
    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.params[0]["seed"] == seed
    assert SamplingParams.deserialize(client.params[0]).seed == seed


@pytest.mark.asyncio
@pytest.mark.parametrize("blueprint,path,body", _ENDPOINTS)
@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "42"])
async def test_invalid_seed_rejected_before_admission(blueprint, path, body, seed):
    client = RecordingClient([])
    response = (
        await _build_app(blueprint, client).test_client().post(path, json={**body, "seed": seed})
    )
    assert response.status_code == 400
    assert client.params == []


@pytest.mark.asyncio
@pytest.mark.parametrize("blueprint,path,body", _ENDPOINTS)
async def test_choices_have_stable_distinct_seeds(blueprint, path, body):
    payload = {**body, "seed": 2**63 - 1}
    if path.endswith("chat/completions"):
        payload["n"] = 2
    else:
        payload["prompt"] = ["hello", "hello"]
    client = RecordingClient([_reply("a", [10], [12]), _reply("b", [10], [12])])
    response = await _build_app(blueprint, client).test_client().post(path, json=payload)
    assert response.status_code == 200, await response.get_data(as_text=True)
    assert [p["seed"] for p in client.params] == [2**63 - 1, 0]
