from megatron.core.inference.common_inference_params import CommonInferenceParams

class TestCommonInferenceParams:

    def test_inference_params(self):
        inference_parameters = CommonInferenceParams()
        inference_parameters.add_attributes({"min_tokens": 45})
        assert inference_parameters.min_tokens == 45, f"min tokens not set correctly. it is {inference_parameters.min_tokens}"