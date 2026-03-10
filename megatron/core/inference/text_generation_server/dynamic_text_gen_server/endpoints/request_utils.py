# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

logger = logging.getLogger(__name__)


def tokenize_messages(tokenizer, messages, tools=None, chat_template_kwargs=None):
    """Tokenize chat messages using the tokenizer's chat template.

    Falls back to plain tokenization if apply_chat_template is not supported.
    Raises on unexpected errors.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            **(chat_template_kwargs or {}),
        )
    except (AttributeError, AssertionError):
        warnings.warn(
            "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
        )
        return tokenizer.tokenize(
            "\n".join([message["content"] for message in messages])
        )


def parse_sampling_params(req, prompt_tokens, tokenizer):
    """Parse sampling parameters from a request dict.

    Returns (SamplingParams, prompt_tokens). prompt_tokens may be modified for BOS handling.
    Raises ValueError on invalid parameters.
    """
    temperature = float(req.get("temperature", 1.0))
    top_p = float(req.get("top_p", 1.0))
    top_k = int(req.get("top_k", 0))

    if temperature == 0.0:
        top_k = 1
        top_p = 0.0

    return_log_probs = bool(req.get("logprobs", False))
    top_n_logprobs = int(req.get("top_logprobs", 0)) if return_log_probs else 0
    skip_prompt_log_probs = bool(req.get("skip_prompt_log_probs", True))
    add_BOS = bool(req.get("add_BOS", False))

    # The engine only handles add_BOS for string prompts, not pre-tokenized
    # input. Since we pre-tokenize via apply_chat_template, we must handle
    # BOS ourselves, matching the logic in tokenize_prompt().
    if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
        start_idx = 0
        while start_idx < len(prompt_tokens) and prompt_tokens[start_idx] == tokenizer.bos:
            start_idx += 1
        if start_idx > 0:
            prompt_tokens = prompt_tokens[start_idx:]
        if add_BOS:
            prompt_tokens = [tokenizer.bos] + prompt_tokens

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
        skip_prompt_log_probs=skip_prompt_log_probs,
        add_BOS=add_BOS,
    )

    return sampling_params, prompt_tokens


def apply_parsers(text, tools, parsers_list):
    """Run CPU-intensive text parsing on generated text."""
    metadata = {}
    for parser in parsers_list:
        if parser not in PARSER_MAPPING:
            raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")
        text, new_info = PARSER_MAPPING[parser].parse(text, tools=tools)
        assert not (
            metadata.keys() & new_info.keys()
        ), "Multiple parsers found the same information."
        metadata.update(new_info)
    return text, metadata


def format_inference_result(record, sampling_params, tokenizer, parsers, tools, verbose=False):
    """Format a single inference engine result into a response dict.

    Returns a dict with common fields used by both chat completions and responses endpoints.
    Callers should pop 'prompt_tokens_count' and 'completion_tokens_count' for usage stats.
    """
    result = record.merge().serialize()
    # Unwrap ("tensor", [...]) tuples from serialize() into plain lists.
    result = {
        k: v[1] if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == "tensor" else v
        for k, v in result.items()
    }

    prompt_tokens = result["prompt_tokens"]
    text_output = result["generated_text"]
    prompt_tokens_count = len(prompt_tokens) if prompt_tokens is not None else 0

    # --- Logprobs ---
    logprobs_content = None
    if sampling_params.return_log_probs:
        token_logprobs = result.get('log_probs', [])
        tokens_to_decode = [[tok] for tok in result["generated_tokens"]]
        tokens = list(map(tokenizer.detokenize, tokens_to_decode))
        generated_top_n_logprobs = result.get('generated_top_n_logprobs')

        logprobs_content = []
        for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
            top_logprobs_list = []
            if generated_top_n_logprobs and i < len(generated_top_n_logprobs):
                top_n_dict = generated_top_n_logprobs[i]
                for token_str, logprob in top_n_dict.items():
                    top_logprobs_list.append({
                        "token": token_str,
                        "logprob": logprob,
                        "bytes": list(token_str.encode("utf-8")),
                    })
            logprobs_content.append({
                "token": tok,
                "logprob": lp,
                "bytes": list(tok.encode("utf-8")),
                "top_logprobs": top_logprobs_list,
            })

    # --- Parsers ---
    metadata = {}
    message_text = text_output
    if parsers:
        message_text, metadata = apply_parsers(message_text, tools, parsers)

    message = {"role": "assistant", "content": message_text}
    if "tool_calls" in metadata:
        message["tool_calls"] = metadata["tool_calls"]
    if "reasoning" in metadata:
        message["reasoning"] = metadata["reasoning"]

    # Replicate data in the message field for compatibility.
    message["prompt_token_ids"] = result["prompt_tokens"]
    message["generation_token_ids"] = result["generated_tokens"]
    message["generation_log_probs"] = result.get("generated_log_probs", [])

    # --- Evictions ---
    events = result.get("events")
    if events is not None:
        num_evictions = sum(1 for e in events if e.get("type") == "EVICT")
    else:
        num_evictions = 0

    formatted = {
        "message": message,
        "prompt_token_ids": result["prompt_tokens"],
        "generation_token_ids": result["generated_tokens"],
        "generation_log_probs": result.get("generated_log_probs", []),
        "raw_text": result["prompt"] + result["generated_text"],
        "logprobs": {"content": logprobs_content} if sampling_params.return_log_probs else None,
        "finish_reason": "tool_calls" if metadata.get("tool_calls", []) else "stop",
        "policy_staleness": result.get("policy_staleness"),
        "kv_cache_staleness": result.get("kv_cache_staleness"),
        "num_evictions": num_evictions,
        "prompt_tokens_count": prompt_tokens_count,
        "completion_tokens_count": len(result["generated_tokens"]),
    }

    if verbose:
        logging.info(result)

    if result["routing_indices"] is not None:
        formatted["moe_topk_indices"] = result["routing_indices"]
        if prompt_tokens_count:
            formatted["prompt_moe_topk_indices"] = result["routing_indices"][:prompt_tokens_count]

    return formatted
