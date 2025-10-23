# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time
import uuid

from flask import Blueprint, current_app, jsonify, request

from megatron.core.inference.sampling_params import SamplingParams

bp = Blueprint('completions_api', __name__)


@bp.route('/completions', methods=['POST'])
@bp.route('/v1/completions', methods=['POST'])
async def completions():
    """Handles async POST requests for completions."""
    client = current_app.config['client']
    tokenizer = current_app.config['tokenizer']

    req = request.get_json()

    # --- 1. Parse Prompt ---
    prompt_data = req.get("prompt")
    if not prompt_data:
        return "Missing 'prompt' field", 400

    try:
        if isinstance(prompt_data, str):
            prompts_as_tokens = [tokenizer.tokenize(prompt_data)]
            prompts_as_strings = [prompt_data]
        elif isinstance(prompt_data, list):
            if not prompt_data:
                return "'prompt' list is empty", 400
            if all(isinstance(p, str) for p in prompt_data):
                prompts_as_tokens = [tokenizer.tokenize(p) for p in prompt_data]
                prompts_as_strings = prompt_data
            elif all(isinstance(p, int) for p in prompt_data):
                prompts_as_tokens = [prompt_data]
                prompts_as_strings = [tokenizer.detokenize(prompt_data)]
            elif all(
                isinstance(p, list) and all(isinstance(t, int) for t in p) for p in prompt_data
            ):
                prompts_as_tokens = prompt_data
                prompts_as_strings = [tokenizer.detokenize(p) for p in prompt_data]
            else:
                return (
                    (
                        "Invalid 'prompt' format. Must be str, list[str], "
                        "list[int], or list[list[int]]"
                    ),
                    400,
                )
        else:
            return "Invalid 'prompt' type. Must be str or list", 400
    except Exception as e:
        return f"Error tokenizing prompt: {e}", 500

    # --- 2. Parse Sampling Params ---
    try:
        temperature = float(req.get("temperature", 1.0))
        top_p = float(req.get("top_p", 1.0))
        top_k = int(req.get("top_k", 0))

        if temperature == 0.0:
            top_k = 1
            top_p = 0.0

        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_log_probs=bool(req.get("logprobs", None) is not None),
            num_tokens_to_generate=int(req.get("max_tokens", 16)),
        )
    except ValueError as e:
        return f"Invalid sampling parameter: {e}", 400

    # --- 3. Send Requests to Engine ---
    tasks = []
    for prompt_tokens in prompts_as_tokens:
        per_req_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            return_log_probs=sampling_params.return_log_probs,
            num_tokens_to_generate=sampling_params.num_tokens_to_generate,
        )
        tasks.append(client.add_request(prompt_tokens, per_req_params))

    start_time = time.perf_counter()
    try:
        batch_results = await asyncio.gather(*tasks)
    except Exception as e:
        return f"Error during inference: {e}", 500

    logging.info(
        f"Batch of {len(tasks)} requests processed in {time.perf_counter() - start_time:.2f}s"
    )

    # --- 4. Format OpenAI Response ---
    echo = bool(req.get("echo", False))
    choices = []
    total_completion_tokens = 0

    for i, result in enumerate(batch_results):
        full_text = result.generated_text
        text_output = (prompts_as_strings[i] + full_text) if echo else full_text

        logprobs_data = None
        if sampling_params.return_log_probs:
            token_logprobs = getattr(result, 'log_probs', [])
            tokens = [tokenizer.detokenize([tok]) for tok in result.generated_tokens]
            logprobs_data = {
                "token_logprobs": token_logprobs,
                "tokens": tokens,
                "text_offset": [],
                "top_logprobs": [],
            }

        choices.append(
            {"index": i, "text": text_output, "logprobs": logprobs_data, "finish_reason": "length"}
        )
        total_completion_tokens += len(result.generated_tokens)

    response = {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "choices": choices,
        "usage": {
            "prompt_tokens": sum(len(p) for p in prompts_as_tokens),
            "completion_tokens": total_completion_tokens,
            "total_tokens": sum(len(p) for p in prompts_as_tokens) + total_completion_tokens,
        },
    }
    return jsonify(response)
