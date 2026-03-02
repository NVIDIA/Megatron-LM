# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time
import traceback
import uuid

from ._request_utils import format_inference_result, parse_sampling_params, tokenize_messages

logger = logging.getLogger(__name__)

try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False


try:
    from quart import Blueprint, Response, current_app, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']

        req = await request.get_json()

        # --- 1. Parse Messages ---
        messages = req.get("messages")
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)

        try:
            prompt_tokens = tokenize_messages(
                tokenizer, messages,
                tools=req.get("tools", None),
                chat_template_kwargs=req.get("chat_template_kwargs", {}),
            )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

        # --- 2. Parse Sampling Params ---
        try:
            n = int(req.get("n", 1))
            sampling_params, prompt_tokens = parse_sampling_params(req, prompt_tokens, tokenizer)
        except ValueError as e:
            return Response(f"Invalid sampling parameter: {e}", status=400)

        # --- 3. Send Requests to Engine ---
        tasks = [client.add_request(prompt_tokens, sampling_params) for _ in range(n)]

        if current_app.config['verbose']:
            start_time = time.perf_counter()

        try:
            batch_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return Response(f"Error during inference: {e}", status=500)

        if current_app.config['verbose']:
            logging.info(
                f"Batch of {len(tasks)} requests (n={n}) processed in "
                f"{time.perf_counter() - start_time:.2f}s"
            )

        # --- 4. Format OpenAI Response ---
        choices = []
        total_completion_tokens = 0
        prompt_tokens_counts = []
        tools = req.get("tools", None)

        for request_idx, record in enumerate(batch_results):
            formatted = format_inference_result(
                record, sampling_params, tokenizer, parsers, tools,
                verbose=current_app.config['verbose'],
            )
            formatted["index"] = request_idx
            prompt_tokens_counts.append(formatted.pop("prompt_tokens_count"))
            total_completion_tokens += formatted.pop("completion_tokens_count")
            choices.append(formatted)

        prompt_token_count = max(prompt_tokens_counts) if prompt_tokens_counts else 0
        response = {
            "id": str(uuid.uuid4()),
            "created": int(time.time()),
            "model": "EMPTY",
            "object": "chat.completion",
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_token_count + total_completion_tokens,
            },
        }

        if HAVE_ORJSON:
            # Use orjson for faster serialization
            return Response(orjson.dumps(response), mimetype="application/json")
        else:
            return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
