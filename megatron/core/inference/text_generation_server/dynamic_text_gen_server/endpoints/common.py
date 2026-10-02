# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Iterable, Optional

from megatron.core.inference.inference_request import unwrap_serialized_tensors
from megatron.core.inference.sampling_params import SamplingParams

if TYPE_CHECKING:
    from megatron.core.inference.inference_client import InferenceClient

logger = logging.getLogger(__name__)


def abort_requests(client: "InferenceClient", request_ids: Iterable[int], reason: str) -> None:
    """Tell the coordinator to stop generating the given requests.

    Args:
        client: The InferenceClient the requests were submitted through.
        request_ids: Ids from add_request_with_id.
        reason: Logged so the abort can be told apart from a normal completion.
    """
    for request_id in request_ids:
        try:
            client.abort_request(request_id)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to abort request %s (%s)", request_id, reason, exc_info=True)
        else:
            logger.debug("Aborted request %s (%s)", request_id, reason)


def validate_offload_params(offload_params) -> Optional[str]:
    """Return an error message if client-supplied `offload_params` are malformed, else None."""
    if offload_params is None:
        return None
    if not isinstance(offload_params, dict):
        return "'offload_params' must be an object"
    reserved = sorted(key for key in offload_params if isinstance(key, str) and key.startswith("_"))
    if reserved:
        return f"'offload_params' keys starting with '_' are reserved: {reserved}"
    return None


def apply_optional_sampling_default(app_config, config_key, value) -> None:
    """Set `app_config[config_key]` only when `value` was actually provided.

    Startup passes `default_temperature`/`default_top_p`/`default_top_k`
    through as `Optional`, `None` when the operator didn't configure one.
    Setting the key unconditionally (even to a hardcoded fallback like `1.0`)
    would make `resolve_sampling_default`'s `config_key in app_config` check
    always true, permanently hiding the model's own `generation_config.json`
    tier behind a value nobody actually asked for.
    """
    if value is not None:
        app_config[config_key] = value


def resolve_sampling_default(app_config, gen_defaults, key, config_key, hardcoded):
    """Resolve one sampling default: app config > generation_config > hardcoded.

    Precedence for a field the request omits:

    1. an explicitly configured server default (`config_key` present in the app
       config) -- an operator who set it should not be overridden by a model file;
    2. the model's own `generation_config.json` declaration;
    3. the previous hardcoded fallback, so a model without a generation_config and a
       server without configured defaults behave exactly as before.

    Key presence is what distinguishes "operator set 1.0" from "unset", so
    `.get(config_key, default)` is deliberately not used here.
    """
    if config_key in app_config:
        return app_config[config_key]
    if key in gen_defaults:
        return gen_defaults[key]
    return hardcoded


def generation_config_sampling_defaults(tokenizer):
    """Sampling defaults declared by the model's `generation_config.json`.

    HF models ship sampling defaults (`temperature`, `top_p`, `top_k`, `do_sample`)
    in `generation_config.json`, and vLLM applies them when a request omits the
    field. These endpoints did not consult that file, so a client that omitted
    `top_p` got 1.0 here but the model-declared value (e.g. 0.95) under vLLM -- a
    silent per-engine difference in the sampling tail.

    Returns a dict with only the keys the config actually declares, so callers can
    fall back to their own defaults for the rest. Request values always win; this
    only supplies defaults.
    """
    gen_cfg = getattr(tokenizer, "generation_config", None)
    if not isinstance(gen_cfg, dict):
        return {}
    defaults = {}
    for key in ("temperature", "top_p", "top_k"):
        value = gen_cfg.get(key)
        # bool is an int subclass; reject it explicitly.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            defaults[key] = value
    # Greedy decoding is expressed here as top_k=1, matching the temperature==0 path.
    if gen_cfg.get("do_sample") is False:
        defaults["top_k"] = 1
    return defaults


_LOGGED_SAMPLING_DEFAULTS = False


def _log_sampling_defaults_once(tokenizer, resolved):
    """Log the resolved sampling defaults once per process.

    Mirrors the startup log for the EOS token set: it makes it possible to confirm
    from the server log that `generation_config.json` was found and applied, rather
    than inferring it from sampled output (which is hopeless on peaked
    distributions, where top_p barely truncates).
    """
    global _LOGGED_SAMPLING_DEFAULTS
    if _LOGGED_SAMPLING_DEFAULTS:
        return
    _LOGGED_SAMPLING_DEFAULTS = True

    gen_cfg = getattr(tokenizer, "generation_config", None)
    logging.info(
        "Sampling defaults: generation_config=%s -> defaults=%s; first request "
        "resolved to temperature=%s top_p=%s top_k=%s",
        (
            {k: gen_cfg.get(k) for k in ("temperature", "top_p", "top_k", "do_sample")}
            if isinstance(gen_cfg, dict)
            else None
        ),
        generation_config_sampling_defaults(tokenizer),
        resolved.get("temperature"),
        resolved.get("top_p"),
        resolved.get("top_k"),
    )


def _get_non_none(req, key, default):
    """Returns the value from the object or default if the key is missing or None."""
    val = req.get(key)
    return default if val is None else val


def parse_sampling_params(req, app_config, tokenizer, *, completions_mode, return_prompt_tokens):
    """Build the `SamplingParams` for an OpenAI-style request body."""
    gen_defaults = generation_config_sampling_defaults(tokenizer)
    temperature = float(
        _get_non_none(
            req,
            "temperature",
            resolve_sampling_default(
                app_config, gen_defaults, "temperature", "default_temperature", 1.0
            ),
        )
    )
    top_p = float(
        _get_non_none(
            req,
            "top_p",
            resolve_sampling_default(app_config, gen_defaults, "top_p", "default_top_p", 1.0),
        )
    )
    top_k = int(
        _get_non_none(
            req,
            "top_k",
            resolve_sampling_default(app_config, gen_defaults, "top_k", "default_top_k", 0),
        )
    )
    _log_sampling_defaults_once(
        tokenizer, {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    )
    if temperature == 0.0:
        top_k = 1
        top_p = 0.0

    if completions_mode:
        echo = bool(req.get("echo", False))
        logprobs_param = req.get("logprobs", None)
        return_log_probs = logprobs_param is not None
        top_n_logprobs = int(logprobs_param) if return_log_probs else 0
        # Prompt logprobs are only worth computing when echo will display them, unless the
        # client sets the engine knob explicitly.
        skip_prompt_log_probs = bool(
            _get_non_none(req, "skip_prompt_log_probs", not (echo and return_log_probs))
        )
        num_tokens_to_generate = int(_get_non_none(req, "max_tokens", 16))
        add_BOS = False
        extras = {"echo": echo}
    else:
        return_log_probs = bool(_get_non_none(req, "logprobs", False))
        top_n_logprobs = int(_get_non_none(req, "top_logprobs", 0)) if return_log_probs else 0
        skip_prompt_log_probs = bool(_get_non_none(req, "skip_prompt_log_probs", True))
        max_tokens = req.get("max_completion_tokens", None) or req.get("max_tokens", None)
        num_tokens_to_generate = int(max_tokens) if max_tokens is not None else None
        add_BOS = bool(_get_non_none(req, "add_BOS", False))
        extras = {"n": int(_get_non_none(req, "n", 1))}

    # OpenAI-style "stop" may be a string or list of strings; normalize.
    stop = req.get("stop", None)
    if isinstance(stop, str):
        stop = [stop]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=return_log_probs,
        top_n_logprobs=top_n_logprobs,
        skip_prompt_log_probs=skip_prompt_log_probs,
        num_tokens_to_generate=num_tokens_to_generate,
        stop_words=stop,
        add_BOS=add_BOS,
        termination_id=-1 if bool(req.get("ignore_eos", False)) else None,
        return_prompt_tokens=return_prompt_tokens,
        streaming_interval=int(_get_non_none(req, "streaming_interval", 1)),
        detokenize_generations=False,
    )
    return sampling_params, extras


def submit_requests(client, submissions, *, multi_modal_data=None, offload_params=None):
    """Submit each `(prompt_tokens, sampling_params)` pair; return `(request_ids, futures)`."""
    request_ids = []
    tasks = []
    try:
        for prompt_tokens, sampling_params in submissions:
            request_id, future = client.add_request_with_id(
                prompt_tokens,
                sampling_params,
                multi_modal_data=multi_modal_data,
                offload_params=offload_params,
            )
            request_ids.append(request_id)
            tasks.append(future)
    except Exception as e:
        abort_requests(client, request_ids, f"submission failed: {e}")
        raise
    return request_ids, tasks


async def run_inference(client, request_ids, tasks, verbose, log_label=""):
    """Await inference tasks with optional timing. Re-raises exceptions."""
    start_time = time.perf_counter()
    try:
        batch_results = await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        abort_requests(client, request_ids, "client disconnected")
        raise
    if verbose:
        label = f" {log_label}" if log_label else ""
        logger.info(
            f"Batch of {len(tasks)} requests{label} processed in "
            f"{time.perf_counter() - start_time:.2f}s"
        )
    return batch_results


def failed_requests_response(batch_results, prompt_token_count):
    """Return the error response if any reply failed.

    A non-transient engine error is a 400, anything else is a 500.
    """
    failed_errors = []
    has_nontransient_error = False
    for i, record in enumerate(batch_results):
        if record.get("status") != "FAILED":
            continue
        events = record.get("events", [])
        error_events = [
            e for e in events if e.get("type") in ("ERROR_NONTRANSIENT", "ERROR_TRANSIENT")
        ]
        if any(e.get("type") == "ERROR_NONTRANSIENT" for e in error_events):
            has_nontransient_error = True
        error_msg = (
            str(error_events[-1].get("payload", "Unknown error"))
            if error_events
            else "Unknown error"
        )
        failed_errors.append(f"Request {i}: {error_msg}")

    if not failed_errors:
        return None
    error_detail = "; ".join(failed_errors)
    logger.error(f"Inference request(s) failed: {error_detail}")

    # NOTE: This exact string is required for compatibility with Nemo-RL, DO NOT MODIFY.
    if "MaxSequenceLengthOverflowError" in error_detail:
        return (
            f"This model's maximum context length was exceeded. "
            f"Your messages resulted in {prompt_token_count} tokens. "
            f"Please reduce the length of the messages. {error_detail}",
            400,
        )
    return f"Inference request(s) failed: {error_detail}", 400 if has_nontransient_error else 500


def unwrap_batch(batch_results):
    """Unwrap every reply; the response id is the first reply's uid, stager metadata is merged."""
    results = [unwrap_serialized_tensors(record) for record in batch_results]
    response_metadata = {}
    for result in results:
        for key, value in (result.get("payload_stage_metadata") or {}).items():
            if key in response_metadata and response_metadata[key] != value:
                raise ValueError(
                    f"payload stager returned conflicting response metadata for {key!r}"
                )
            response_metadata[key] = value
    response_uid = results[0]["uid"] if results else None
    return results, response_uid, response_metadata


def reply_prompt_length(result):
    """Prompt-token count of one reply: `prompt_length`, falling back to `prompt_tokens` length."""
    length = result.get("prompt_length")
    if length is None:
        prompt_tokens = result.get("prompt_tokens")
        length = len(prompt_tokens) if prompt_tokens is not None else 0
    return length


def add_moe_routing_to_choice(choice_data, result, prompt_len):
    """Add MOE routing indices to a choice dict if present in the result."""
    if result["routing_indices"] is not None:
        choice_data["moe_topk_indices"] = result["routing_indices"]
        if prompt_len:
            choice_data["prompt_moe_topk_indices"] = result["routing_indices"][:prompt_len]


def build_usage(prompt_tokens_counts, total_completion_tokens, cached_tokens_counts):
    """Build the OpenAI usage block from per-choice prompt-token counts."""
    prompt_token_count = max(prompt_tokens_counts) if prompt_tokens_counts else 0
    cached_token_count = max(cached_tokens_counts) if cached_tokens_counts else 0
    return {
        "prompt_tokens": prompt_token_count,
        "completion_tokens": total_completion_tokens,
        "total_tokens": prompt_token_count + total_completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_token_count},
    }


def build_response(response_uid, object_type, choices, usage, response_metadata):
    """Assemble the OpenAI envelope and merge the stager's metadata at the top level.

    The OpenAI-shaped fields are reserved; a stager key colliding with one is a bug rather than
    something to overwrite silently.
    """
    response = {
        "id": response_uid,
        "object": object_type,
        "created": int(time.time()),
        "model": "EMPTY",
        "choices": choices,
        "usage": usage,
    }
    overlap = set(response).intersection(response_metadata)
    if overlap:
        raise ValueError(
            f"payload stager response metadata collides with reserved fields: {sorted(overlap)}"
        )
    response.update(response_metadata)
    return response
