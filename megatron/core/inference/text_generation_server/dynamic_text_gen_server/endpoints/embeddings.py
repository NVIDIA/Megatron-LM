# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sequence embeddings endpoint.

POST /embeddings (or /v1/embeddings) takes the OpenAI embeddings body,

    {"input": [[1, 5, 7, ...], ...], "encoding_format": "float", "skip_tokens": 1}

and returns, for each input, the mean of the model's final hidden states (the output layer's
input) over the input's token positions. `input` is read as the completions endpoint's `prompt`.
`skip_tokens` (default 0) leaves that many leading positions out of the mean, for instance a
start token. `encoding_format` is "float" (default) or "base64" (little-endian float32).

The server must run with `--inference-dynamic-batching-prompt-hidden-states`. Each input runs as
one request that prefills the input and generates a single token, which is discarded.
"""

import base64
import logging

import numpy as np

from megatron.core.inference.sampling_params import SamplingParams

from .common import failed_requests_response, run_inference, submit_requests, unwrap_batch
from .completions import parse_prompt

logger = logging.getLogger(__name__)


def parse_embedding_options(req):
    """Return `(skip_tokens, encoding_format)` of a request body; ``ValueError`` if malformed."""
    skip_tokens = req.get("skip_tokens", 0)
    if isinstance(skip_tokens, bool) or not isinstance(skip_tokens, int) or skip_tokens < 0:
        raise ValueError("skip_tokens must be a nonnegative integer")
    encoding_format = req.get("encoding_format") or "float"
    if encoding_format not in ("float", "base64"):
        raise ValueError("encoding_format must be 'float' or 'base64'")
    return skip_tokens, encoding_format


def format_embeddings_response(batch_results, encoding_format):
    """Build the OpenAI embeddings body; ``ValueError`` names an input without an embedding."""
    results, response_uid, response_metadata = unwrap_batch(batch_results)
    data = []
    for index, result in enumerate(results):
        if result.get("prompt_embedding") is None:
            raise ValueError(
                f"input {index} has no embedding: its prompt's hidden states were not each "
                "seen exactly once (request recomputation is not supported)"
            )
        embedding = np.asarray(result["prompt_embedding"], dtype="<f4")
        if encoding_format == "base64":
            embedding = base64.b64encode(embedding.tobytes()).decode("ascii")
        else:
            embedding = embedding.tolist()
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": embedding,
                "prompt_token_ids": result.get("prompt_tokens") or [],
            }
        )
    prompt_tokens = sum(len(item["prompt_token_ids"]) for item in data)
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}
    return {**response_metadata, "id": response_uid, "object": "list", "data": data, "usage": usage}


try:
    from quart import Blueprint, current_app, jsonify, request

    bp = Blueprint('embeddings_api', __name__)

    @bp.route('/embeddings', methods=['POST'])
    @bp.route('/v1/embeddings', methods=['POST'])
    async def embeddings():
        """Handles async POST requests for sequence embeddings."""
        client = current_app.config['client']

        req = await request.get_json(force=True)
        if req is None:
            return "Invalid or missing JSON body", 400

        try:
            prompts, _ = parse_prompt(req.get("input"), current_app.config['tokenizer'])
            skip_tokens, encoding_format = parse_embedding_options(req)
        except ValueError as e:
            return str(e).replace("'prompt'", "'input'"), 400
        except Exception as e:
            return f"Error tokenizing input: {e}", 500
        short = [index for index, tokens in enumerate(prompts) if len(tokens) <= skip_tokens]
        if short:
            return f"inputs {short} have no token after skip_tokens={skip_tokens}", 400

        sampling_params = SamplingParams(
            num_tokens_to_generate=1,
            return_prompt_tokens=True,
            detokenize_generations=False,
            return_prompt_embedding=True,
            prompt_embedding_skip_tokens=skip_tokens,
        )
        try:
            request_ids, tasks = submit_requests(
                client, [(tokens, sampling_params) for tokens in prompts]
            )
            batch_results = await run_inference(
                client, request_ids, tasks, current_app.config['verbose'], "(embeddings)"
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"Error during inference: {e}", 500

        failure = failed_requests_response(batch_results, max(map(len, prompts), default=0))
        if failure is not None:
            return failure
        try:
            return jsonify(format_embeddings_response(batch_results, encoding_format))
        except ValueError as e:
            return str(e), 500

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
