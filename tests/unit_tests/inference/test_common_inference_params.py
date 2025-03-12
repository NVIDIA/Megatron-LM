from megatron.core.inference.sampling_params import SamplingParams


class TestSamplingParams:

    def test_inference_params(self):
        inference_parameters = SamplingParams()
        inference_parameters.add_attributes({"min_tokens": 45})
        assert (
            inference_parameters.min_tokens == 45
        ), f"min tokens not set correctly. it is {inference_parameters.min_tokens}"
