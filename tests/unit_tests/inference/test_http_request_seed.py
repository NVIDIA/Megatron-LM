# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings

import pytest

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    sampling_params_for_choice,
)
from tests.unit_tests.inference.test_endpoints_common import (
    BODIES,
    CHAT_PATH,
    PATHS,
    ReplyingClient,
    build_app,
    completed_reply,
)

pytestmark = [pytest.mark.internal]


@pytest.mark.asyncio
@PATHS
@pytest.mark.parametrize("seed", [None, 0, 42, 2**63 - 1])
async def test_http_seed_reaches_engine(path, seed):
    client = ReplyingClient()
    response = (
        await build_app(path, client).test_client().post(path, json={**BODIES[path], "seed": seed})
    )
    assert response.status_code == 200, await response.get_data(as_text=True)
    (sampling_params,) = client.sampling_params
    assert sampling_params.seed == seed
    assert SamplingParams.deserialize(sampling_params.serialize()).seed == seed


@pytest.mark.asyncio
@PATHS
@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "42"])
async def test_invalid_seed_rejected_before_admission(path, seed):
    client = ReplyingClient([])
    response = (
        await build_app(path, client).test_client().post(path, json={**BODIES[path], "seed": seed})
    )
    assert response.status_code == 400
    assert client.sampling_params == []


@pytest.mark.asyncio
@PATHS
async def test_choices_have_stable_distinct_seeds(path):
    payload = {**BODIES[path], "seed": 2**63 - 1}
    if path == CHAT_PATH:
        payload["n"] = 2
    else:
        payload["prompt"] = ["hello", "hello"]
    client = ReplyingClient([completed_reply("a", [10], [12]), completed_reply("b", [10], [12])])
    response = await build_app(path, client).test_client().post(path, json=payload)
    assert response.status_code == 200, await response.get_data(as_text=True)
    assert [p.seed for p in client.sampling_params] == [2**63 - 1, 0]


def test_choice_preserves_derived_prompt_logprob_policy_without_warning():
    params = SamplingParams(seed=42, top_n_logprobs=3, skip_prompt_log_probs=False)
    assert params.return_prompt_top_n_logprobs
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        choice = sampling_params_for_choice(params, 1)
    assert choice.seed == 43
    assert choice.return_prompt_top_n_logprobs
    assert not choice.skip_prompt_log_probs
    assert choice.top_n_logprobs == 3
    assert params.seed == 42
