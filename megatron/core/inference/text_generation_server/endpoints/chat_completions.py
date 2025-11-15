# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time

from flask import Blueprint, current_app, jsonify, request

from megatron.core.inference.sampling_params import SamplingParams

bp = Blueprint('chat_completions_api', __name__)

logger = logging.getLogger(__name__)


@bp.route('/chat/completions', methods=['POST'])
@bp.route('/v1/chat/completions', methods=['POST'])
async def chat_completions():
    """Handles async POST requests for chat completions."""
    client = current_app.config['client']
    tokenizer = current_app.config['tokenizer']

    req = request.get_json()

    # --- 1. Parse Messages ---
    messages = req.get("messages")
    if not messages:
        return "Missing 'messages' field", 400
    if not isinstance(messages, list):
        return "'messages' must be a list", 400

    try:
        prompt_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except AttributeError:
        return (
            "Tokenizer does not support 'apply_chat_template'. "
            "Chat completions requires a tokenizer with a configured chat template."
        ), 500
    except Exception as e:
        return f"Error processing 'messages': {e}", 500

    # --- 2. Parse Sampling Params ---
    try:
        temperature = float(req.get("temperature", 1.0))
        top_p = float(req.get("top_p", 1.0))
        top_k = int(req.get("top_k", 0))
        n = int(req.get("n", 1))  # Number of choices to generate

        if temperature == 0.0:
            top_k = 1
            top_p = 0.0

        # Check for 'logprobs' (bool)
        return_log_probs = bool(req.get("logprobs", False))

        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_log_probs=return_log_probs,
            num_tokens_to_generate=int(req.get("max_tokens", 16)),
        )
    except ValueError as e:
        return f"Invalid sampling parameter: {e}", 400

    # --- 3. Send Requests to Engine ---
    # For chat, we run the *same* prompt 'n' times.
    tasks = []
    for _ in range(n):
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

    logger.info(
        f"Batch of {len(tasks)} requests (n={n}) processed in "
        f"{time.perf_counter() - start_time:.2f}s"
    )

    # --- 4. Format OpenAI Response ---
    choices = []
    total_completion_tokens = 0
    prompt_token_count = len(prompt_tokens)  # Calculated once

    for i, result in enumerate(batch_results):
        text_output = result.generated_text

        logprobs_content = None
        if sampling_params.return_log_probs:
            token_logprobs = getattr(result, 'log_probs', [])
            tokens = [tokenizer.detokenize([tok]) for tok in result.generated_tokens]

            logprobs_content = []
            for tok, lp in zip(tokens, token_logprobs):
                # The original server's 'result' object only provides the
                # logprob for the *chosen* token, not 'top_logprobs'.
                # We map what we have to the new API format.
                entry = {
                    "token": tok,
                    "logprob": lp,
                    "bytes": list(tok.encode("utf-8")),  # OpenAI API includes this
                    "top_logprobs": [],  # We don't have this data from the engine
                }
                logprobs_content.append(entry)

        choice_data = {
            "index": i,
            "message": {"role": "assistant", "content": text_output},
            # 'logprobs' in chat API is an object containing 'content'
            "logprobs": {"content": logprobs_content} if logprobs_content else None,
            "finish_reason": "length",  # Original code hardcoded this.
        }
        choices.append(choice_data)
        total_completion_tokens += len(result.generated_tokens)

    response = {
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_token_count + total_completion_tokens,
        },
    }
    return jsonify(response)
