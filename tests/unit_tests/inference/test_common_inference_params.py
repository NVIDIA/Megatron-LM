# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import GREEDY_TEMPERATURE, SamplingParams


def _sampling_metadata(params):
    """The sampling metadata a dynamic context tracks for a request with these params."""
    if params.termination_id is None:
        params.termination_id = 0
    request = DynamicInferenceRequest(
        request_id=0, prompt="", prompt_tokens=torch.tensor([1]), sampling_params=params
    )
    names = [name for name, _ in request.get_metadata_types()]
    metadata = dict(zip(names, request.tracked_metadata))
    return {name: metadata[name] for name in ("temperature", "top_k", "top_p", "is_greedy")}


class TestSamplingParams:

    @pytest.mark.parametrize(
        'given,expected',
        [
            ({'top_p': 1.0}, {'top_p': 0.0}),
            ({'top_p': 1.5}, {'top_p': 0.0}),
            ({'top_p': 0.95}, {'top_p': 0.95}),
            ({'top_k': -1}, {'top_k': 0}),
        ],
    )
    def test_no_op_filters_normalize_to_disabled_sentinels(self, given, expected):
        # `deserialize` funnels through `add_attributes`, bypassing __post_init__,
        # so both construction paths must normalize.
        for params in (SamplingParams(**given), SamplingParams.deserialize(given)):
            for name, value in expected.items():
                assert getattr(params, name) == value

    @pytest.mark.parametrize(
        "given",
        [
            {"temperature": 0.0},
            {"temperature": 0.0, "top_k": 20, "top_p": 0.95},
            {"temperature": GREEDY_TEMPERATURE / 2},
            {"top_k": 1, "top_p": 0.9},
            {"is_greedy": True, "temperature": 0.7, "top_k": 20},
        ],
        ids=["no-filters", "model-card-filters", "tiny-temperature", "top-1", "flag"],
    )
    def test_greedy_requests(self, given):
        # Greedy (the lowest-index argmax) for every caller, not only the HTTP front ends. Its
        # filters read top_k=1 without top-p, as the front ends set them, and its temperature is
        # kept; the samplers read only `is_greedy`, over neutral sampling metadata.
        for params in (SamplingParams(**given), SamplingParams.deserialize(given)):
            assert params.is_greedy
            assert (params.top_k, params.top_p) == (1, 0.0)
            assert params.temperature == given.get("temperature", 1.0)
            assert _sampling_metadata(params) == {
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 0.0,
                "is_greedy": True,
            }

    @pytest.mark.parametrize(
        "given",
        [{}, {"temperature": GREEDY_TEMPERATURE}, {"temperature": 0.7, "top_k": 2, "top_p": 0.9}],
        ids=["defaults", "threshold-temperature", "filters"],
    )
    def test_sampled_requests_keep_their_sampling_metadata(self, given):
        params = SamplingParams(**given)
        assert not params.is_greedy
        assert _sampling_metadata(params) == {
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "is_greedy": False,
        }

    @pytest.mark.parametrize("temperature", [-0.5, -GREEDY_TEMPERATURE, math.nan, math.inf])
    def test_invalid_temperature_is_rejected(self, temperature):
        for build in (SamplingParams, SamplingParams.deserialize):
            given = {"temperature": temperature}
            with pytest.raises(ValueError, match="temperature"):
                build(**given) if build is SamplingParams else build(given)

    @pytest.mark.parametrize("temperature", [0.0, GREEDY_TEMPERATURE / 2, 0.7, 1.0])
    @pytest.mark.parametrize("top_k", [0, -1, 1, 20])
    def test_disabled_top_p_spelling_never_changes_greedy(self, temperature, top_k):
        # NeMo RL disables top-p as 1.0 and Megatron as 0.0 (#6993). Either spelling gives the
        # same greedy decision, the same params and the same sampling metadata, so the same
        # kernels and numerics.
        zero, one = (
            SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)
            for top_p in (0.0, 1.0)
        )
        assert zero.is_greedy == one.is_greedy == (temperature < GREEDY_TEMPERATURE or top_k == 1)
        assert (zero.temperature, zero.top_k, zero.top_p) == (one.temperature, one.top_k, one.top_p)
        assert one.top_p == 0.0
        assert _sampling_metadata(zero) == _sampling_metadata(one)

    def test_sampling_params(self):
        sampling_params = SamplingParams()
        sampling_params.add_attributes({"min_tokens": 45})
        assert (
            sampling_params.min_tokens == 45
        ), f"min tokens not set correctly. it is {sampling_params.min_tokens}"

    def test_streaming_interval(self):
        sampling_params = SamplingParams(streaming_interval=8)
        serialized = sampling_params.serialize()
        deserialized = SamplingParams.deserialize(serialized)

        assert SamplingParams().streaming_interval == 1
        assert serialized["streaming_interval"] == 8
        assert deserialized.streaming_interval == 8

    def test_streaming_interval_must_be_positive(self):
        try:
            SamplingParams(streaming_interval=0)
        except ValueError:
            pass
        else:
            raise AssertionError("streaming_interval=0 should be rejected")
