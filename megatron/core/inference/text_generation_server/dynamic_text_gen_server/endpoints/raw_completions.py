# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time
import uuid
from typing import Any

from megatron.core.inference.inference_request import unwrap_serialized_tensors
from megatron.core.inference.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def _is_int_token(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_token_ids(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{field_name}' must be a non-empty list of token ids")
    if not all(_is_int_token(token_id) for token_id in value):
        raise ValueError(f"'{field_name}' must contain only integer token ids")
    return value


def _parse_stop_token_sequences(req: dict[str, Any]) -> list[list[int]]:
    sequences: list[list[int]] = []

    stop_token_ids = req.get("stop_token_ids")
    if stop_token_ids is not None:
        if not isinstance(stop_token_ids, list):
            raise ValueError("'stop_token_ids' must be a list of token ids")
        for token_id in stop_token_ids:
            if not _is_int_token(token_id):
                raise ValueError("'stop_token_ids' must contain only integer token ids")
            sequences.append([token_id])

    stop_token_id_sequences = req.get("stop_token_id_sequences")
    if stop_token_id_sequences is not None:
        if not isinstance(stop_token_id_sequences, list):
            raise ValueError("'stop_token_id_sequences' must be a list of token-id lists")
        for sequence in stop_token_id_sequences:
            sequences.append(_parse_token_ids(sequence, "stop_token_id_sequences[]"))

    return sequences


def _trim_at_stop_sequences(
    token_ids: list[int], token_data: list[Any] | None, stop_sequences: list[list[int]]
) -> tuple[list[int], list[Any] | None, bool]:
    earliest: int | None = None
    for stop_sequence in stop_sequences:
        stop_len = len(stop_sequence)
        if stop_len == 0 or len(token_ids) < stop_len:
            continue
        for start in range(len(token_ids) - stop_len + 1):
            if token_ids[start : start + stop_len] == stop_sequence:
                earliest = start if earliest is None else min(earliest, start)
                break

    if earliest is None:
        return token_ids, token_data, False

    return token_ids[:earliest], None if token_data is None else token_data[:earliest], True


def _first_single_token_stop(stop_sequences: list[list[int]]) -> int | None:
    for sequence in stop_sequences:
        if len(sequence) == 1:
            return sequence[0]
    return None


def _parse_top_n_logprobs(
    req: dict[str, Any], prompt_logprobs: bool, logprobs: bool
) -> tuple[int, int, int]:
    top_logprobs = req.get("top_logprobs")
    top_prompt_logprobs = req.get("top_prompt_logprobs")

    if top_logprobs is not None and not logprobs:
        raise ValueError("'top_logprobs' requires 'logprobs' to be true")
    if top_prompt_logprobs is not None and not prompt_logprobs:
        raise ValueError("'top_prompt_logprobs' requires 'prompt_logprobs' to be true")

    generation_top_n = int(top_logprobs or 0)
    prompt_top_n = int(top_prompt_logprobs or 0)
    if generation_top_n < 0 or prompt_top_n < 0:
        raise ValueError("'top_logprobs' and 'top_prompt_logprobs' must be non-negative")
    return prompt_top_n, generation_top_n, max(prompt_top_n, generation_top_n)


def _truncate_top_logprobs(
    top_logprobs: list[dict[str, float]] | None, requested_count: int
) -> list[dict[str, float]] | None:
    if top_logprobs is None or requested_count == 0:
        return None
    return [
        dict(sorted(entry.items(), key=lambda item: item[1], reverse=True)[:requested_count])
        for entry in top_logprobs
    ]


try:
    from quart import Blueprint, current_app, jsonify, request

    bp = Blueprint("raw_completions_api", __name__)

    @bp.route("/raw_completions", methods=["POST"])
    @bp.route("/v1/raw_completions", methods=["POST"])
    async def raw_completions():
        """Handles token-in/token-out completion requests.

        This endpoint is intentionally tokenizer-free: callers pass prompt token
        ids and receive generated token ids plus optional sampled logprobs.
        """
        client = current_app.config["client"]

        req = await request.get_json(force=True)
        if req is None:
            return "Invalid or missing JSON body", 400
        if not isinstance(req, dict):
            return "JSON body must be an object", 400

        if req.get("images"):
            return "'images' are not supported by the Megatron raw completion endpoint", 400
        if req.get("audios"):
            return "'audios' are not supported by the Megatron raw completion endpoint", 400
        if "stop" in req and req["stop"]:
            return (
                "'stop' string sequences are not supported by the token-only "
                "Megatron raw completion endpoint; use stop_token_ids or "
                "stop_token_id_sequences",
                400,
            )

        try:
            prompt_tokens = _parse_token_ids(req.get("prompt"), "prompt")
            stop_sequences = _parse_stop_token_sequences(req)

            temperature = float(req.get("temperature", 1.0))
            top_p = float(req.get("top_p", 1.0))
            top_k = int(req.get("top_k", 0))
            max_tokens = int(req.get("max_tokens", 16))
            if max_tokens < 0:
                return "'max_tokens' must be non-negative", 400

            if temperature == 0.0:
                top_k = 1
                top_p = 0.0
            elif top_k < 0:
                top_k = 0

            prompt_logprobs = bool(req.get("prompt_logprobs", False))
            logprobs = bool(req.get("logprobs", False))
            prompt_top_n, generation_top_n, top_n_logprobs = _parse_top_n_logprobs(
                req, prompt_logprobs, logprobs
            )

            termination_id = None
            if bool(req.get("ignore_eos", False)):
                termination_id = -1
            else:
                stop_token_id = _first_single_token_stop(stop_sequences)
                if stop_token_id is not None:
                    termination_id = stop_token_id

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=prompt_logprobs or logprobs,
                top_n_logprobs=top_n_logprobs,
                skip_prompt_log_probs=not prompt_logprobs,
                num_tokens_to_generate=max_tokens,
                termination_id=termination_id,
            )
        except ValueError as e:
            return str(e), 400

        if current_app.config["verbose"]:
            start_time = time.perf_counter()

        try:
            completed_request = await asyncio.gather(
                client.add_request(prompt_tokens, sampling_params)
            )
        except Exception as e:
            return f"Error during inference: {e}", 500

        if current_app.config["verbose"]:
            logging.info(
                "Raw completion request processed in " f"{time.perf_counter() - start_time:.2f}s"
            )

        record = completed_request[0]
        if record.get("status") == "FAILED":
            events = record.get("events", [])
            error_events = [
                e for e in events if e.get("type") in ("ERROR_NONTRANSIENT", "ERROR_TRANSIENT")
            ]
            error_msg = (
                str(error_events[-1].get("payload", "Unknown error"))
                if error_events
                else "Unknown error"
            )
            has_nontransient_error = any(
                e.get("type") == "ERROR_NONTRANSIENT" for e in error_events
            )
            status = 400 if has_nontransient_error else 500
            logger.error(f"Inference request failed: {error_msg}")
            return f"Inference request failed: {error_msg}", status

        result = unwrap_serialized_tensors(record)
        generated_tokens = list(result.get("generated_tokens") or [])
        generated_log_probs = result.get("generated_log_probs")
        if generated_log_probs is not None:
            generated_log_probs = list(generated_log_probs)
        generated_top_logprobs = result.get("generated_top_n_logprobs")
        if generated_top_logprobs is not None:
            generated_top_logprobs = list(generated_top_logprobs)
        generated_tokens, generated_log_probs, stop_hit = _trim_at_stop_sequences(
            generated_tokens, generated_log_probs, stop_sequences
        )
        if generated_top_logprobs is not None:
            generated_top_logprobs = generated_top_logprobs[: len(generated_tokens)]

        prompt_tokens_result = list(result.get("prompt_tokens") or prompt_tokens)
        prompt_log_probs = result.get("prompt_log_probs")
        if prompt_log_probs is not None:
            prompt_log_probs = list(prompt_log_probs)
        prompt_top_logprobs = result.get("prompt_top_n_logprobs")
        if prompt_top_logprobs is not None:
            prompt_top_logprobs = list(prompt_top_logprobs)
        prompt_top_logprobs = _truncate_top_logprobs(prompt_top_logprobs, prompt_top_n)
        generated_top_logprobs = _truncate_top_logprobs(generated_top_logprobs, generation_top_n)
        finish_reason = "stop" if stop_hit else "length"
        if max_tokens == 0 or len(generated_tokens) < max_tokens:
            finish_reason = "stop"

        completion_tokens = len(generated_tokens)
        prompt_tokens_count = len(prompt_tokens_result)
        return jsonify(
            {
                "id": str(uuid.uuid4()),
                "object": "raw_completion",
                "created": int(time.time()),
                "model": "EMPTY",
                "choices": [
                    {
                        "index": 0,
                        "prompt_token_ids": prompt_tokens_result,
                        "generation_token_ids": generated_tokens,
                        "prompt_logprobs": prompt_log_probs,
                        "generation_logprobs": generated_log_probs,
                        "prompt_top_logprobs": prompt_top_logprobs,
                        "generation_top_logprobs": generated_top_logprobs,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens_count,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens_count + completion_tokens,
                },
            }
        )

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
