# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.sampling_params import SamplingParams


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
