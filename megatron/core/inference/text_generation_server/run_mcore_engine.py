# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import inspect

from megatron.core import mpu
from megatron.core.inference.communication_utils import broadcast_float_list
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.tokenization import tokenize_prompts


def run_mcore_engine(
    engine,
    prompts=None,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    logprobs=True,
    tokens_to_generate=0,
    top_n_logprobs=0,
    random_seed=-1,
):
    """Server-compatible version of the MCore Engine, used in
    tools/run_text_generation_server.py."""

    values = [tokens_to_generate, logprobs, top_k, top_p, temperature, top_n_logprobs, random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values, data_parallel=False)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k = int(values_float_tensor[2].item())
    top_p = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    top_n_logprobs = int(values_float_tensor[5].item())
    random_seed = int(values_float_tensor[6].item())

    if random_seed > 0:
        engine.controller.sampling_rng.manual_seed(random_seed)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_segments=True,
        return_log_probs=return_output_log_probs,
        num_tokens_to_generate=tokens_to_generate,
        top_n_logprobs=top_n_logprobs,
        skip_prompt_log_probs=False,
    )

    tokenizer = engine.controller.tokenizer
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        tokenizer=tokenizer,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        add_BOS=False,
        data_parallel=False,
    )

    tokenized_prompts = []
    for p, l in zip(context_tokens_tensor, context_length_tensor):
        tokenized_prompts.append(p[:l].cpu().numpy().tolist())

    # detect if detokenize supports skip_special_tokens or **kwargs
    sig_params = inspect.signature(tokenizer.detokenize).parameters.values()
    accepts_skip = any(
        p.name == "skip_special_tokens" or p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig_params
    )

    # Detokenize prompts into strings to pass through the engine
    detokenized_prompts = [
        (
            tokenizer.detokenize(p, skip_special_tokens=True)
            if accepts_skip
            else tokenizer.detokenize(p)
        )
        for p in tokenized_prompts
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
        response_dict = {
            "text": [x.prompt + x.generated_text for x in result],
            "tokens": [x.prompt_tokens + x.generated_tokens.tolist() for x in result],
        }
        if sampling_params.return_log_probs:
            response_logprobs = [x.prompt_log_probs + x.generated_log_probs for x in result]
            response_dict["logprobs"] = response_logprobs
        if sampling_params.return_segments:
            response_dict["segments"] = [x.segments for x in result]
        if sampling_params.top_n_logprobs > 0:
            # TODO(ksanthanam): Support enabling `skip_prompt_log_probs`
            assert (
                sampling_params.return_prompt_top_n_logprobs
            ), "skip_prompt_log_probs must be False"
            response_dict["top_n_logprobs"] = [
                x.prompt_top_n_logprobs + x.generated_top_n_logprobs for x in result
            ]

        return response_dict
    return None
