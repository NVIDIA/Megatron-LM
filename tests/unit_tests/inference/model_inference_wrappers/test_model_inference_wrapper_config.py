from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig

class TestModelInferenceWrapperConfig:

    def test_inference_params(self):
        inference_parameters = InferenceWrapperConfig()
        inference_parameters.add_attributes({"abc": 45})
        assert inference_parameters.abc == 45, f"min tokens not set correctly. it is {inference_parameters.min_tokens}"