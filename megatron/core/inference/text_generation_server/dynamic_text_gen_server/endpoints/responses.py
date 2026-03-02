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

# Per-process storage for background requests. With multiple server replicas,
# each process has its own dict. This works because HTTP/2 keep-alive connections
# route the same client to the same replica for the duration of the connection.
_pending = {}

try:
    from quart import Blueprint, Response, current_app, jsonify, request

    bp = Blueprint('responses_api', __name__)

    def _build_completed_response(response_id, formatted, created_at):
        """Build a completed Responses API response from formatted inference results."""
        prompt_tokens_count = formatted.pop("prompt_tokens_count")
        completion_tokens_count = formatted.pop("completion_tokens_count")

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": "EMPTY",
            "output": [],
            "usage": {
                "prompt_tokens": prompt_tokens_count,
                "completion_tokens": completion_tokens_count,
                "total_tokens": prompt_tokens_count + completion_tokens_count,
            },
        }
        response_data.update(formatted)
        if HAVE_ORJSON:
            return Response(orjson.dumps(response_data), mimetype="application/json")
        return jsonify(response_data)

    @bp.route('/responses', methods=['POST'])
    @bp.route('/v1/responses', methods=['POST'])
    async def create_response():
        """Submit an inference request via the Responses API.

        With background=True, returns immediately with status="queued" (202).
        Otherwise, awaits the result and returns the completed response.
        """
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']
        verbose = current_app.config['verbose']

        req = await request.get_json()

        # --- 1. Parse Messages (from 'input' field, Responses API convention) ---
        messages = req.get("input")
        if not messages:
            return jsonify({"error": "Missing 'input' field"}), 400
        if not isinstance(messages, list):
            return jsonify({"error": "'input' must be a list"}), 400

        try:
            prompt_tokens = tokenize_messages(
                tokenizer, messages,
                tools=req.get("tools", None),
                chat_template_kwargs=req.get("chat_template_kwargs", {}),
            )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return jsonify({"error": f"Error processing 'input': {e}"}), 500

        # --- 2. Parse Sampling Params ---
        try:
            sampling_params, prompt_tokens = parse_sampling_params(req, prompt_tokens, tokenizer)
        except ValueError as e:
            return jsonify({"error": f"Invalid sampling parameter: {e}"}), 400

        # --- 3. Submit Request ---
        response_id = str(uuid.uuid4())
        created_at = int(time.time())
        background = req.get("background", False)

        if background:
            # Non-blocking: schedule the request and return immediately.
            task = asyncio.ensure_future(client.add_request(prompt_tokens, sampling_params))
            _pending[response_id] = {
                "task": task,
                "sampling_params": sampling_params,
                "tools": req.get("tools", None),
                "parsers": parsers,
                "tokenizer": tokenizer,
                "verbose": verbose,
                "created_at": created_at,
            }
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "queued",
                "created_at": created_at,
                "model": "EMPTY",
                "output": [],
            }), 202

        # Blocking: await the result inline.
        try:
            record = await client.add_request(prompt_tokens, sampling_params)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return jsonify({"error": f"Error during inference: {e}"}), 500

        try:
            formatted = format_inference_result(
                record, sampling_params, tokenizer, parsers,
                tools=req.get("tools", None), verbose=verbose,
            )
        except Exception as e:
            logger.error(f"Response formatting failed: {traceback.format_exc()}")
            return jsonify({"id": response_id, "status": "failed", "error": str(e)}), 500

        return _build_completed_response(response_id, formatted, created_at)

    @bp.route('/responses/<response_id>', methods=['GET'])
    @bp.route('/v1/responses/<response_id>', methods=['GET'])
    async def get_response(response_id):
        """Poll for the result of a background inference request."""
        entry = _pending.get(response_id)
        if entry is None:
            return jsonify({"error": "Response not found", "id": response_id}), 404

        task = entry["task"]

        if not task.done():
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "created_at": entry["created_at"],
                "model": "EMPTY",
                "output": [],
            })

        # Check for exception (keep entry so retries get "failed" instead of 404).
        exc = task.exception()
        if exc is not None:
            logger.error(f"Response {response_id} failed: {exc}")
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": str(exc),
            })

        # Format completed result.
        try:
            record = task.result()
            formatted = format_inference_result(
                record, entry["sampling_params"], entry["tokenizer"],
                entry["parsers"], tools=entry["tools"], verbose=entry["verbose"],
            )
        except Exception as e:
            logger.error(f"Response {response_id} formatting failed: {traceback.format_exc()}")
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": str(e),
            })

        # Clean up on success.
        del _pending[response_id]
        return _build_completed_response(response_id, formatted, entry["created_at"])

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
