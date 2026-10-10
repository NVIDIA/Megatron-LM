# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64
import dataclasses
import logging

from megatron.core.inference.inference_request import prepare_multimodal_data
from megatron.core.inference.utils import detokenize_tokens

from ..incremental_detokenizer import HuggingFaceFastIncrementalDetokenizer
from ..openai_streaming import (
    finish_reason,
    json_safe_logprobs,
    json_safe_top_n_logprobs,
    openai_stream,
)
from .common import (
    add_moe_routing_to_choice,
    build_response,
    build_usage,
    failed_requests_response,
    parse_sampling_params,
    reply_prompt_length,
    run_inference,
    submit_requests,
    unwrap_batch,
    validate_offload_params,
)

logger = logging.getLogger(__name__)


def parse_prompt(prompt_data, tokenizer):
    """Tokenize the ``prompt`` field (str, list[str], list[int] or list[list[int]]).

    Returns parallel ``(prompts_as_tokens, prompts_as_strings)`` lists. Raises ``ValueError`` for a
    missing or malformed prompt; tokenizer errors propagate unchanged.
    """
    if not prompt_data:
        raise ValueError("Missing 'prompt' field")
    if isinstance(prompt_data, str):
        return [tokenizer.tokenize(prompt_data)], [prompt_data]
    if isinstance(prompt_data, list):
        if all(isinstance(p, str) for p in prompt_data):
            return [tokenizer.tokenize(p) for p in prompt_data], prompt_data
        if all(isinstance(p, int) for p in prompt_data):
            return [prompt_data], [tokenizer.detokenize(prompt_data)]
        if all(isinstance(p, list) and all(isinstance(t, int) for t in p) for p in prompt_data):
            return prompt_data, [tokenizer.detokenize(p) for p in prompt_data]
        raise ValueError(
            "Invalid 'prompt' format. Must be str, list[str], list[int], or list[list[int]]"
        )
    raise ValueError("Invalid 'prompt' type. Must be str or list")


def parse_multi_modal_data(req):
    """Decode the optional vLLM-style ``multi_modal_data`` (base64 or data-URL strings) to bytes.

    Returns ``{modality: [bytes, ...]}`` for the one populated modality, or ``None``. HTTP callers
    provide encoded bytes; preprocessed tensors are direct-API only.
    """
    request_multi_modal_data = req.get("multi_modal_data") or {}
    if not isinstance(request_multi_modal_data, dict):
        raise ValueError("multi_modal_data must be a dictionary.")
    unsupported_modalities = set(request_multi_modal_data) - {"image", "video"}
    if unsupported_modalities:
        raise ValueError(f"Unsupported multimodal modalities: {sorted(unsupported_modalities)}.")
    populated_modalities = [
        modality for modality in ("image", "video") if request_multi_modal_data.get(modality)
    ]
    if len(populated_modalities) > 1:
        raise ValueError("A completions request cannot mix image and video inputs.")
    if not populated_modalities:
        return None
    modality = populated_modalities[0]
    encoded_media = request_multi_modal_data[modality]
    if isinstance(encoded_media, str):
        encoded_media = [encoded_media]
    if not isinstance(encoded_media, list) or any(
        not isinstance(item, str) for item in encoded_media
    ):
        raise ValueError(f"multi_modal_data.{modality} must be a string or list[str].")
    media_bytes = [
        base64.b64decode(
            item.split(",", 1)[1] if item.startswith("data:") and "," in item else item
        )
        for item in encoded_media
    ]
    return {modality: media_bytes}


def _completions_logprobs(result, tokenizer, echo):
    """OpenAI text-completion ``logprobs`` block; with ``echo`` the prompt tokens lead.

    Non-finite logprobs are clamped for JSON. The leading ``None`` matches the OpenAI format,
    where the first token carries no logprob.
    """
    generated_tokens = result["generated_tokens"] or []
    generated_log_probs = json_safe_logprobs(result.get("generated_log_probs") or [])
    generated_top_n_logprobs = json_safe_top_n_logprobs(
        result.get("generated_top_n_logprobs") or []
    )
    if echo:
        prompt_log_probs = json_safe_logprobs(result.get("prompt_log_probs") or [])
        prompt_top_n_logprobs = json_safe_top_n_logprobs(result.get("prompt_top_n_logprobs") or [])
        prompt_tokens = result["prompt_tokens"] or []
        token_ids = prompt_tokens + generated_tokens
        # Prompt scores cover tokens [1:] (P-1 convention); pad to the full prompt when the
        # engine skipped them (skip_prompt_log_probs) so both lists stay aligned with `tokens`.
        score_pad = [None] * (len(prompt_tokens) - 1 - len(prompt_log_probs))
        top_pad = [None] * (len(prompt_tokens) - 1 - len(prompt_top_n_logprobs))
        token_logprobs = [None] + prompt_log_probs + score_pad + generated_log_probs
        top_logprobs = (
            [None] + prompt_top_n_logprobs + top_pad + generated_top_n_logprobs
            if prompt_top_n_logprobs or generated_top_n_logprobs
            else None
        )
    else:
        token_ids = generated_tokens
        token_logprobs = [None] + generated_log_probs
        top_logprobs = [None] + generated_top_n_logprobs if generated_top_n_logprobs else None

    tokens = [tokenizer.detokenize([tok]) for tok in token_ids]
    text_offset = []
    current_offset = 0
    for tok_str in tokens:
        text_offset.append(current_offset)
        current_offset += len(tok_str)
    return {
        "token_logprobs": token_logprobs,
        "tokens": tokens,
        "text_offset": text_offset,
        "top_logprobs": top_logprobs,
    }


def format_completions_response(
    batch_results, prompts_as_strings, sampling_params, echo, tokenizer
):
    """Build the OpenAI text-completion body from the finished replies."""
    results, response_uid, response_metadata = unwrap_batch(batch_results)
    choices = []
    total_completion_tokens = 0
    prompt_tokens_counts = []
    cached_tokens_counts = []
    for request_idx, result in enumerate(results):
        generated_tokens = result.get("generated_tokens") or []
        full_text = detokenize_tokens(
            tokenizer, generated_tokens, remove_EOD=not sampling_params.detokenize_stop_sequence
        )
        total_completion_tokens += len(generated_tokens)
        prompt_len = reply_prompt_length(result)
        prompt_tokens_counts.append(prompt_len)
        cached_tokens_counts.append(result.get("num_cached_tokens", 0))
        # Under payload offload the engine dropped the per-token log probs from the reply.
        payload_offloaded = bool(result.get("payload_offloaded"))

        choice_data = {
            "index": request_idx,
            "text": (prompts_as_strings[request_idx] + full_text) if echo else full_text,
            "logprobs": (
                _completions_logprobs(result, tokenizer, echo)
                if sampling_params.return_log_probs and not payload_offloaded
                else None
            ),
            "finish_reason": finish_reason(result),
            "prompt_token_ids": result["prompt_tokens"],
            "generation_token_ids": result["generated_tokens"],
        }
        if not payload_offloaded:
            # Clamped: processed logprobs can be -inf, which JSON cannot carry.
            choice_data["generation_log_probs"] = json_safe_logprobs(
                result.get("generated_log_probs") or []
            )
        # Speculative decoding (e.g. MTP): per-engine-step emitted token counts, summing to the
        # generated token count; empty/None when spec decoding is off. `ttft` is the real
        # time-to-first-token in seconds; `tpot` is a SPARSE per-token step-time sample (only
        # populated on logging steps), so a dense TPOT must come from ttft + total latency.
        choice_data["acceptance_step_lengths"] = result.get("acceptance_step_lengths")
        choice_data["ttft"] = result.get("ttft")
        choice_data["tpot"] = result.get("tpot")
        add_moe_routing_to_choice(choice_data, result, prompt_len)
        choices.append(choice_data)

    usage = build_usage(prompt_tokens_counts, total_completion_tokens, cached_tokens_counts)
    return build_response(response_uid, "text_completion", choices, usage, response_metadata)


try:
    from quart import Blueprint, Response, current_app, jsonify, request

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

        # Opaque metadata forwarded to the engine's payload stager. Keys starting
        # with '_' are engine-owned and rejected here so a client cannot forge them.
        offload_params = req.get("offload_params")
        offload_params_error = validate_offload_params(offload_params)
        if offload_params_error is not None:
            return offload_params_error, 400

        # --- 1. Parse Prompt ---
        try:
            prompts_as_tokens, prompts_as_strings = parse_prompt(req.get("prompt"), tokenizer)
        except ValueError as e:
            return str(e), 400
        except Exception as e:
            return f"Error tokenizing prompt: {e}", 500

        # --- 2. Parse Sampling Params ---
        try:
            # This endpoint always echoes prompt_token_ids in its response, so the engine
            # must keep the prompt tokens on the payload.
            sampling_params, extras = parse_sampling_params(
                req, current_app.config, tokenizer, completions_mode=True, return_prompt_tokens=True
            )
            echo = extras["echo"]
            multi_modal_data = parse_multi_modal_data(req)
        except (ValueError, TypeError) as e:
            return f"Invalid sampling parameter: {e}", 400

        # --- 3. Send Requests to Engine ---
        stream_requested = bool(req.get("stream", False))
        incremental_detokenizers = []
        if stream_requested:
            # Streaming currently supports only Hugging Face fast tokenizers.
            try:
                incremental_detokenizers = [
                    HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_tokens)
                    for prompt_tokens in prompts_as_tokens
                ]
            except ValueError as error:
                return str(error), 400

        # TODO: streaming submissions made before a mid-loop failure are not aborted. Their
        # handles are AsyncStreams, not ids, and they abort through openai_stream's finally,
        # which never runs because the generator is never started.
        request_ids = []
        tasks = []
        streams = []
        try:
            # Hash and serialize shared media once before fanning it out across the prompts.
            prepared_multimodal_data = prepare_multimodal_data(multi_modal_data)
            submissions = [
                (
                    prompt_tokens,
                    dataclasses.replace(sampling_params, return_prompt_top_n_logprobs=False),
                )
                for prompt_tokens in prompts_as_tokens
            ]
            if stream_requested:
                streams = [
                    client.add_request_streaming(
                        prompt_tokens,
                        per_request_params,
                        multi_modal_data=prepared_multimodal_data,
                        offload_params=offload_params,
                    )
                    for prompt_tokens, per_request_params in submissions
                ]
            else:
                request_ids, tasks = submit_requests(
                    client,
                    submissions,
                    multi_modal_data=prepared_multimodal_data,
                    offload_params=offload_params,
                )
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            return f"Error submitting request: {e}", 500

        if stream_requested:
            include_usage = bool((req.get("stream_options") or {}).get("include_usage", False))
            response = Response(
                openai_stream(
                    streams,
                    tokenizer,
                    incremental_detokenizers,
                    chat=False,
                    return_log_probs=sampling_params.return_log_probs,
                    include_usage=include_usage,
                    echo_prompts=prompts_as_strings if echo else None,
                    prompt_token_ids=prompts_as_tokens if echo else None,
                ),
                content_type="text/event-stream",
            )
            response.timeout = None
            return response

        try:
            batch_results = await run_inference(
                client, request_ids, tasks, current_app.config['verbose']
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"Error during inference: {e}", 500

        # --- 4. Check for failed requests ---
        failure = failed_requests_response(
            batch_results, max((len(p) for p in prompts_as_tokens), default=0)
        )
        if failure is not None:
            return failure

        # --- 5. Format Response ---
        return jsonify(
            format_completions_response(
                batch_results, prompts_as_strings, sampling_params, echo, tokenizer
            )
        )

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
