# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import threading
import time

import torch

from megatron.core.inference.sampling_params import SamplingParams

GENERATE_NUM = 0
LOCK = threading.Lock()


def send_do_generate():
    """Broadcasts a message to perform a generation to all tensor parallel ranks."""
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.broadcast(choice, 0)


logger = logging.getLogger(__name__)


def parse_sampling_params(req, completions_mode=False):
    """Parse all sampling parameters from a request dict into a SamplingParams.

    Handles the different conventions between the two endpoint types:

    completions_mode=True (OpenAI /v1/completions):
      - ``logprobs`` is an integer (top-N) or absent; drives both
        ``return_log_probs`` and ``top_n_logprobs``.
      - ``skip_prompt_log_probs`` is derived from echo + return_log_probs.
      - ``max_tokens`` defaults to 16.
      - ``stop`` field is parsed into ``stop_words``.

    completions_mode=False (OpenAI /v1/chat/completions, default):
      - ``logprobs`` is a boolean; ``top_logprobs`` is a separate integer.
      - ``skip_prompt_log_probs`` is read directly (default True).
      - ``max_completion_tokens`` / ``max_tokens`` with no default.

    Returns:
        ``(SamplingParams, extras)`` where *extras* is a dict:
          - completions mode: ``{"echo": bool}``
          - chat mode: ``{"n": int}``
    """
    # --- Core sampling parameters ---
    temperature = float(req.get("temperature", 1.0))
    top_p = float(req.get("top_p", 1.0))
    top_k = int(req.get("top_k", 0))

    if temperature == 0.0:
        top_k = 1
        top_p = 0.0

    # --- Logprobs (endpoint-dependent interpretation) ---
    if completions_mode:
        echo = bool(req.get("echo", False))
        logprobs_param = req.get("logprobs", None)
        if logprobs_param is not None:
            top_n_logprobs = int(logprobs_param)
            return_log_probs = True
        else:
            top_n_logprobs = 0
            return_log_probs = False
        skip_prompt_log_probs = not (echo and return_log_probs)
    else:
        return_log_probs = bool(req.get("logprobs", False))
        top_n_logprobs = int(req.get("top_logprobs", 0)) if return_log_probs else 0
        skip_prompt_log_probs = bool(req.get("skip_prompt_log_probs", True))

    # --- Token generation limits ---
    if completions_mode:
        num_tokens_to_generate = int(req.get("max_tokens", 16))
    else:
        max_tokens = req.get("max_completion_tokens", None) or req.get("max_tokens", None)
        num_tokens_to_generate = int(max_tokens) if max_tokens is not None else None

    # --- Stop sequences ---
    stop = req.get("stop", None)
    if isinstance(stop, str):
        stop = [stop]

    # --- Remaining SamplingParams fields ---
    add_BOS = bool(req.get("add_BOS", False))
    return_segments = bool(req.get("return_segments", False))
    detokenize_stop_sequence = bool(req.get("detokenize_stop_sequence", False))

    num_tokens_total = req.get("num_tokens_total", None)
    if num_tokens_total is not None:
        num_tokens_total = int(num_tokens_total)

    termination_id = req.get("termination_id", None)
    if termination_id is not None:
        termination_id = int(termination_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=return_log_probs,
        top_n_logprobs=top_n_logprobs,
        skip_prompt_log_probs=skip_prompt_log_probs,
        num_tokens_to_generate=num_tokens_to_generate,
        num_tokens_total=num_tokens_total,
        termination_id=termination_id,
        add_BOS=add_BOS,
        stop_words=stop,
        return_segments=return_segments,
        detokenize_stop_sequence=detokenize_stop_sequence,
    )

    if completions_mode:
        extras = {"echo": echo}
    else:
        extras = {"n": int(req.get("n", 1))}

    return sampling_params, extras


async def run_inference(tasks, verbose):
    """Await inference tasks with optional timing. Re-raises exceptions."""
    if verbose:
        start_time = time.perf_counter()

    batch_results = await asyncio.gather(*tasks)

    if verbose:
        logging.info(
            f"Batch of {len(tasks)} requests processed in "
            f"{time.perf_counter() - start_time:.2f}s"
        )

    return batch_results


def check_failed_requests(batch_results):
    """Check batch results for failures.

    Returns (error_detail, status_code) if any request failed, else None.
    """
    failed_errors = []
    has_nontransient_error = False
    for i, record in enumerate(batch_results):
        if record.get("status") == "FAILED":
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

    if failed_errors:
        error_detail = "; ".join(failed_errors)
        status = 400 if has_nontransient_error else 500
        logger.error(f"Inference request(s) failed: {error_detail}")
        return error_detail, status
    return None


def add_moe_routing_to_choice(choice_data, result):
    """Add MOE routing indices to a choice dict if present in the result."""
    if result["routing_indices"] is not None:
        choice_data["moe_topk_indices"] = result["routing_indices"]
        prompt_tokens = result.get("prompt_tokens")
        prompt_length = len(prompt_tokens) if prompt_tokens is not None else 0
        if prompt_length:
            choice_data["prompt_moe_topk_indices"] = result["routing_indices"][:prompt_length]
