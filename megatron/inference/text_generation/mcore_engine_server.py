# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Union

from megatron.core import mpu
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.text_generation.communication import broadcast_float_list
from megatron.inference.text_generation.tokenization import tokenize_prompts
import torch
from megatron.training import get_tokenizer


class ModelInferenceWrapperServer(GPTInferenceWrapper):
    def __init__(self, model, inference_wrapper_config):
        super().__init__(model, inference_wrapper_config)

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """
        Slices out the tokens, position ids, and masking for the specific context window.
        This version also sets `runtime_gather_output` to be False to be compatible with
        the inference server in tools/run_text_generation_server.py, which expects parallel logits
        distributed across TP ranks.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window.
            context_end_position (int): End of the context window.

        Returns:
            Dict[str, Any]: Inputs used in the forward call.
        """
        inference_input = super().get_batch_for_context_window(inference_input,
                                                               context_start_position,
                                                               context_end_position)
        inference_input["runtime_gather_output"] = False
        return inference_input


def run_mcore_engine(
    engine, prompts=None, prompt_tokens=None, prompt_tokens_tensor=None, prompt_lengths_tensor = None,temperature=1.0, top_k=0, top_p=0.0, logprobs=True, tokens_to_generate=0
):
    """Server-compatible version of the MCore Engine, used in
    tools/run_text_generation_server.py."""

    values = [tokens_to_generate, logprobs, top_k, top_p, temperature]
    values_float_tensor = broadcast_float_list(len(values), float_list=values, data_parallel=False)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k = int(values_float_tensor[2].item())
    top_p = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_segments=True,
        return_log_probs=return_output_log_probs,
        num_tokens_to_generate=tokens_to_generate,
    )

    if prompt_tokens_tensor is not None:
        context_tokens_tensor = prompt_tokens_tensor
        context_length_tensor = prompt_lengths_tensor
    elif prompt_tokens is not None:
        # Find max length for padding
        max_length = max(len(tokens) for tokens in prompt_tokens)
        # Pad all sequences to the max length
        padded_tokens = []
        # Get tokenizer once outside the loop to find the padding token
        tokenizer = get_tokenizer()
        pad_token = tokenizer.eod

        prompts_length = [len(prompt_tokens_instance) for prompt_tokens_instance in prompt_tokens]
        # Get the max prompts length.
        max_prompt_len = max(prompts_length)
        # Number of tokens in the each sample of the batch.
        samples_length = max_prompt_len + tokens_to_generate
        # Now update the list of list to be of the same size: samples_length.
        for pt, pl in zip(prompt_tokens, prompts_length):
            padding_size = samples_length - pl
            pt.extend([pad_token] * padding_size)

        # Now we are in a structured format, we can convert to tensors.
        context_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.int64, device='cuda')
        context_length_tensor = torch.tensor(prompts_length, dtype=torch.int64, device='cuda')
    else:
        assert prompts is not None, "Either prompts or prompt_tokens must be provided"
        context_tokens_tensor, context_length_tensor = tokenize_prompts(
            prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=False, data_parallel=False
        )
    # print(context_tokens_tensor)

    tokenized_prompts = []
    for p, l in zip(context_tokens_tensor, context_length_tensor):
        tokenized_prompts.append(p[:l].cpu().numpy().tolist())

    # Detokenize prompts into strings to pass through the engine
    detokenized_prompts = [
        engine.text_generation_controller.tokenizer.detokenize(prompt)
        for prompt in tokenized_prompts
    ]

    requests = []
    for i in range(len(tokenized_prompts)):
        req = InferenceRequest(
            prompt=detokenized_prompts[i],
            prompt_tokens=tokenized_prompts[i],
            sampling_params=sampling_params,
            request_id=engine.get_new_request_id(),
        )
        requests.append(req)

    result = engine.generate(inference_requests=requests)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        response_dict = {"text": [x.prompt + x.generated_text for x in result]}
        if sampling_params.return_log_probs:
            response_logprobs = [x.prompt_log_probs + x.generated_log_probs for x in result]
            response_dict["logprobs"] = response_logprobs
        if sampling_params.return_segments:
            response_dict["segments"] = [x.segments for x in result]
        response_dict["tokens"] = [(x.prompt_tokens + x.generated_tokens.tolist()) for x in result]

        return response_dict
    return None
