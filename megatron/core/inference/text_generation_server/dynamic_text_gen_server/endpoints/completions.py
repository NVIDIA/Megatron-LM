# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import logging

from megatron.core.inference.inference_request import unwrap_serialized_tensors

from .common import (
    add_moe_routing_to_choice,
    check_failed_requests,
    parse_sampling_params,
    run_inference,
)

logger = logging.getLogger(__name__)


def parse_prompt(prompt_data, tokenizer):
    """Parse a completions-style prompt into token and string representations.

    Handles the four OpenAI prompt formats:
      - ``str``: single text prompt
      - ``list[str]``: batch of text prompts
      - ``list[int]``: single prompt as pre-tokenized IDs
      - ``list[list[int]]``: batch of pre-tokenized prompts

    Args:
        prompt_data: The ``"prompt"`` field value from the request.
        tokenizer: Object with ``tokenize(str)`` and ``detokenize(list[int])``
            methods.

    Returns:
        ``(prompts_as_tokens, prompts_as_strings)`` — parallel lists.

    Raises:
        ValueError: If *prompt_data* is missing, empty, or has an
            unrecognised format.  Tokenizer exceptions propagate unchanged.
    """
    if not prompt_data:
        raise ValueError("Missing 'prompt' field")

    if isinstance(prompt_data, str):
        return [tokenizer.tokenize(prompt_data)], [prompt_data]

    if isinstance(prompt_data, list):
        if not prompt_data:
            raise ValueError("'prompt' list is empty")
        if all(isinstance(p, str) for p in prompt_data):
            return [tokenizer.tokenize(p) for p in prompt_data], list(prompt_data)
        if all(isinstance(p, int) for p in prompt_data):
            return [prompt_data], [tokenizer.detokenize(prompt_data)]
        if all(isinstance(p, list) and all(isinstance(t, int) for t in p) for p in prompt_data):
            return prompt_data, [tokenizer.detokenize(p) for p in prompt_data]
        raise ValueError(
            "Invalid 'prompt' format. Must be str, list[str], " "list[int], or list[list[int]]"
        )

    raise ValueError("Invalid 'prompt' type. Must be str or list")


def format_completions_response(
    batch_results, prompts_as_strings, sampling_params, echo, tokenizer
):
    """Format inference results into an OpenAI completions response body.

    Args:
        batch_results: Raw result dicts from the inference engine.
        prompts_as_strings: Original prompt strings (used when *echo* is True).
        sampling_params: The :class:`SamplingParams` used for inference.
        echo: Whether to prepend the prompt text to each generated output.
        tokenizer: Tokenizer with a ``detokenize`` method.

    Returns:
        ``dict`` with a ``"choices"`` key.
    """
    choices = []

    request_idx = 0
    for completed_request in batch_results:
        result = unwrap_serialized_tensors(completed_request)
        full_text = result["generated_text"] or ""
        text_output = (prompts_as_strings[request_idx] + full_text) if echo else full_text

        logprobs_data = None
        if sampling_params.return_log_probs:
            # Get prompt tokens and logprobs
            prompt_tokens_list = result["prompt_tokens"] or []

            prompt_log_probs = result.get('prompt_log_probs') or []
            prompt_top_n_logprobs = result.get('prompt_top_n_logprobs') or []

            # Get generated tokens and logprobs
            generated_tokens_list = result["generated_tokens"] or []
            generated_log_probs = result.get('generated_log_probs') or []
            generated_top_n_logprobs = result.get('generated_top_n_logprobs') or []

            if echo:
                # When echo=True, include prompt tokens and their logprobs
                # Prompt logprobs are for tokens [1:] (first token has no logprob)
                all_token_ids = prompt_tokens_list + generated_tokens_list
                tokens = [tokenizer.detokenize([tok]) for tok in all_token_ids]

                # Build token_logprobs: [None] for first token, then prompt logprobs,
                # then generated logprobs
                token_logprobs = [None] + list(prompt_log_probs) + list(generated_log_probs)

                # Build top_logprobs: [None] for first token, then prompt top_n,
                # then generated top_n
                top_logprobs = None
                if prompt_top_n_logprobs or generated_top_n_logprobs:
                    top_logprobs = (
                        [None] + list(prompt_top_n_logprobs) + list(generated_top_n_logprobs)
                    )

                # Calculate text_offset: cumulative character positions starting from 0
                text_offset = []
                current_offset = 0
                for tok_str in tokens:
                    text_offset.append(current_offset)
                    current_offset += len(tok_str)
            else:
                # When echo=False, only return generated tokens and their logprobs
                tokens = [tokenizer.detokenize([tok]) for tok in generated_tokens_list]

                # Prepend [None] to match OpenAI format
                token_logprobs = [None] + list(generated_log_probs)

                # Build top_logprobs
                top_logprobs = None
                if generated_top_n_logprobs:
                    top_logprobs = [None] + list(generated_top_n_logprobs)

                # Calculate text_offset for generated tokens only
                text_offset = []
                current_offset = 0
                for tok_str in tokens:
                    text_offset.append(current_offset)
                    current_offset += len(tok_str)

            logprobs_data = {
                "token_logprobs": token_logprobs,
                "tokens": tokens,
                "text_offset": text_offset,
                "top_logprobs": top_logprobs,
            }

        choices.append({"index": request_idx, "text": text_output, "logprobs": logprobs_data})
        add_moe_routing_to_choice(choices[-1], result)

        request_idx += 1

    return {"choices": choices}


try:
    from quart import Blueprint, current_app, jsonify, request

    bp = Blueprint('completions_api', __name__)

    @bp.route('/completions', methods=['POST'])
    @bp.route('/v1/completions', methods=['POST'])
    async def completions():
        """Handles async POST requests for completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']

        req = await request.get_json(force=True)
        if req is None:
            return "Invalid or missing JSON body", 400

        # --- 1. Parse Prompt ---
        try:
            prompts_as_tokens, prompts_as_strings = parse_prompt(req.get("prompt"), tokenizer)
        except ValueError as e:
            return str(e), 400
        except Exception as e:
            return f"Error tokenizing prompt: {e}", 500

        # --- 2. Parse Sampling Params ---
        try:
            sampling_params, extras = parse_sampling_params(req, completions_mode=True)
            echo = extras["echo"]
        except ValueError as e:
            return f"Invalid sampling parameter: {e}", 400

        # --- 3. Send Requests to Engine ---
        tasks = []
        for prompt_tokens in prompts_as_tokens:
            tasks.append(client.add_request(prompt_tokens, dataclasses.replace(sampling_params)))

        try:
            batch_results = await run_inference(tasks, current_app.config['verbose'])
        except Exception as e:
            return f"Error during inference: {e}", 500

        # --- 4. Check for failed requests ---
        failure = check_failed_requests(batch_results)
        if failure:
            error_detail, status = failure
            return f"Inference request(s) failed: {error_detail}", status

        # --- 5. Format Response ---
        return jsonify(
            format_completions_response(
                batch_results, prompts_as_strings, sampling_params, echo, tokenizer
            )
        )

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
