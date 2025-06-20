from megatron.core.inference.sampling_params import SamplingParams


class TestSamplingParams:

    def test_sampling_params(self):
        sampling_params = SamplingParams()
        sampling_params.add_attributes({"min_tokens": 45})
        assert (
            sampling_params.min_tokens == 45
        ), f"min tokens not set correctly. it is {sampling_params.min_tokens}"
