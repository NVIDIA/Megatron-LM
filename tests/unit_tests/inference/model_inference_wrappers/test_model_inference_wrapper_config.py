import torch

from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)


class TestModelInferenceWrapperConfig:

    def test_inference_config(self):
        inference_config = InferenceWrapperConfig(
            hidden_size=10,
            inference_batch_times_seqlen_threshold=10,
            padded_vocab_size=10,
            params_dtype=torch.float,
            fp32_residual_connection=False,
        )
        inference_config.add_attributes({"abc": 45})
        assert (
            inference_config.abc == 45
        ), f"min tokens not set correctly. it is {inference_config.min_tokens}"
