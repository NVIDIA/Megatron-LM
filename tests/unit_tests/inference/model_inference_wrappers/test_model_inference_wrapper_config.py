import torch

from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)


class TestModelInferenceWrapperConfig:

    def test_inference_params(self):
        inference_parameters = InferenceWrapperConfig(
            hidden_size=10,
            inference_batch_times_seqlen_threshold=10,
            padded_vocab_size=10,
            params_dtype=torch.float,
            fp32_residual_connection=False,
        )
        inference_parameters.add_attributes({"abc": 45})
        assert (
            inference_parameters.abc == 45
        ), f"min tokens not set correctly. it is {inference_parameters.min_tokens}"
