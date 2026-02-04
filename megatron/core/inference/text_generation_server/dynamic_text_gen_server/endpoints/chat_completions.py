# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time
import traceback

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

logger = logging.getLogger(__name__)

try:
    from flask import Blueprint, current_app, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']

        req = request.get_json()

        # --- 1. Parse Messages ---
        messages = req.get("messages")
        if not messages:
            return "Missing 'messages' field", 400
        if not isinstance(messages, list):
            return "'messages' must be a list", 400

        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, tools=req.get("tools", None)
            )
        except AttributeError:
            logger.warning(
                "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
            )
            prompt_tokens = tokenizer.tokenize(
                "\n".join([message["content"] for message in messages])
            )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
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

            # Check for 'logprobs' (bool) and 'top_logprobs' (int)
            return_log_probs = bool(req.get("logprobs", False))
            top_n_logprobs = int(req.get("top_logprobs", 0)) if return_log_probs else 0

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                num_tokens_to_generate=(
                    int(max_tokens)
                    if ((max_tokens := req.get("max_tokens", None)) is not None)
                    else None
                ),
                skip_prompt_log_probs=True,
            )
        except ValueError as e:
            return f"Invalid sampling parameter: {e}", 400

        # --- 3. Send Requests to Engine ---
        # For chat, we run the *same* prompt 'n' times.
        tasks = []
        for _ in range(n):
            tasks.append(client.add_request(prompt_tokens, sampling_params))

        start_time = time.perf_counter()
        try:
            batch_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"Error during inference: {e}", 500

        logger.info(
            f"Batch of {len(tasks)} requests (n={n}) processed in "
            f"{time.perf_counter() - start_time:.2f}s"
        )

        # --- 4. Format OpenAI Response ---
        choices = []
        total_completion_tokens = 0
        prompt_token_count = len(prompt_tokens)  # Calculated once

        request_idx = 0
        for record in batch_results:
            for result in record.requests:
                text_output = result.generated_text

                logprobs_content = None
                if sampling_params.return_log_probs:
                    token_logprobs = getattr(result, 'log_probs', [])
                    tokens = [tokenizer.detokenize([tok]) for tok in result.generated_tokens]

                    # Get top_n_logprobs if available
                    generated_top_n_logprobs = getattr(result, 'generated_top_n_logprobs', None)

                    logprobs_content = []
                    for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
                        # Build top_logprobs list for this token position
                        top_logprobs_list = []
                        if generated_top_n_logprobs and i < len(generated_top_n_logprobs):
                            top_n_dict = generated_top_n_logprobs[i]
                            for token_str, logprob in top_n_dict.items():
                                top_logprobs_list.append(
                                    {
                                        "token": token_str,
                                        "logprob": logprob,
                                        "bytes": list(token_str.encode("utf-8")),
                                    }
                                )

                        entry = {
                            "token": tok,
                            "logprob": lp,
                            "bytes": list(tok.encode("utf-8")),
                            "top_logprobs": top_logprobs_list,
                        }
                        logprobs_content.append(entry)

                metadata = {}
                message_text = text_output
                if parsers:
                    for parser in parsers:
                        if parser not in PARSER_MAPPING:
                            raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")
                        message_text, new_info = PARSER_MAPPING[parser].parse(
                            message_text, tools=req.get("tools", None)
                        )
                        assert not (
                            metadata.keys() & new_info.keys()
                        ), "Multiple parsers found the same information."
                        metadata.update(new_info)
                message = {"role": "assistant", "content": message_text}
                if "tool_calls" in metadata:
                    message["tool_calls"] = metadata["tool_calls"]
                if "reasoning" in metadata:
                    message["reasoning"] = metadata["reasoning"]

                choice_data = {
                    "index": request_idx,
                    "message": message,
                    "prompt_token_ids": prompt_tokens,
                    "generation_token_ids": result.generated_tokens,
                    "generation_log_probs": result.generated_log_probs,
                    "raw_text": result.prompt + result.generated_text,
                    # 'logprobs' in chat API is an object containing 'content'
                    "logprobs": {"content": logprobs_content} if logprobs_content else None,
                    "finish_reason": (
                        "tool_calls" if metadata.get("tool_calls", []) else "stop"
                    ),  # Original code hardcoded this.
                }
                choices.append(choice_data)
                total_completion_tokens += len(result.generated_tokens)
                request_idx += 0

        response = {
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_token_count + total_completion_tokens,
            },
        }
        return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import flask: {e}")
