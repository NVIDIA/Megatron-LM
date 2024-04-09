from typing import List

from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams


def common_generate(
    inference_backend: AbstractBackend,
    prompts: List[str] = None,
    common_inference_params: CommonInferenceParams = None,
) -> dict:
    """Common Generate function to call for inference

    This function will automatically chose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. 

    Args:
        inference_backend (Union[MCoreBackend, TRTLLMBackend]): The inference backend, that has the generate function.
        prompts (List[str], optional): The input prompts as a list of strings. Typically of length global batch size. Defaults to None.
        common_inference_params (CommonInferenceParams, optional): The usual inference parameters that are used for generation. Defaults to None.

    Returns:
        dict: The output dictionary containing the generated tokens, texts and log probs if required
    """
    output_dictionary = inference_backend.generate(
        prompts=prompts, common_inference_params=common_inference_params
    )

    return output_dictionary
