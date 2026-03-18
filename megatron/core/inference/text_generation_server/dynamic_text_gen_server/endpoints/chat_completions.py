# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import json
import logging
import time
import traceback
import uuid
import warnings

from megatron.core.inference.inference_request import unwrap_serialized_tensors
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

logger = logging.getLogger(__name__)

# pylint: disable=line-too-long


def _get_field(obj, key, default=None):
    """Read a field from dict-like or object-like values."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_tool_calls(tool_calls):
    """Normalize tool calls to OpenAI-compatible JSON primitives."""
    normalized = []
    for call in tool_calls or []:
        fn = _get_field(call, "function", {}) or {}
        fn_name = _get_field(fn, "name")
        fn_args = _get_field(fn, "arguments", "")
        if fn_name is None:
            continue
        if not isinstance(fn_args, str):
            try:
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            except TypeError:
                fn_args = str(fn_args)
        normalized.append(
            {
                "id": str(_get_field(call, "id", f"call_{uuid.uuid4().hex[:24]}")),
                "type": "function",
                "function": {"name": str(fn_name), "arguments": fn_args},
            }
        )
    return normalized


def _coerce_arguments_mapping(arguments):
    """Coerce function.arguments to a mapping for HF/Jinja chat templates.

    Examples:
    - {"x": 1} -> {"x": 1}
    - '{"x": 1}' -> {"x": 1}
    - "[1, 2]" -> {}  # JSON parses, but not a mapping
    - "not-json" -> {}
    - None -> {}
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sanitize_messages_for_template(messages):
    """Prepare messages so tokenizer chat templates can safely consume them.

    This only normalizes tool-call argument payloads inside each message:
    - messages[*].tool_calls[*].function.arguments is coerced to a dict.

    Example transformation:
    Input:
      [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]}]
    Output:
      [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]}]

    Another example:
    - arguments: "[1,2,3]" -> arguments: {}
    """
    if not isinstance(messages, list):
        return messages
    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        msg_copy = dict(message)
        tool_calls = msg_copy.get("tool_calls")
        if isinstance(tool_calls, list):
            sanitized_tool_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    sanitized_tool_calls.append(call)
                    continue
                call_copy = dict(call)
                function = call_copy.get("function")
                if isinstance(function, dict):
                    function_copy = dict(function)
                    function_copy["arguments"] = _coerce_arguments_mapping(
                        function_copy.get("arguments", {})
                    )
                    call_copy["function"] = function_copy
                sanitized_tool_calls.append(call_copy)
            msg_copy["tool_calls"] = sanitized_tool_calls
        sanitized.append(msg_copy)
    return sanitized


def _sanitize_tools_for_template(tools):
    """Ensure tools payload is template-safe and has mapping parameters.

    Example transformations:
    - {"function": {"name": "f", "parameters": "not-a-dict"}}
      -> {"function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    - non-dict tool entries are dropped.
    - non-list input returns None.
    """
    if not isinstance(tools, list):
        return None

    sanitized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_copy = dict(tool)
        function = tool_copy.get("function")
        if isinstance(function, dict):
            function_copy = dict(function)
            if not isinstance(function_copy.get("parameters"), dict):
                function_copy["parameters"] = {"type": "object", "properties": {}}
            tool_copy["function"] = function_copy
        sanitized.append(tool_copy)
    return sanitized


def _reconstruct_reasoning_content(messages: list[dict]) -> list[dict]:
    """Reconstruct <think> tags from reasoning_content fields on assistant messages.

    For parity with vLLM, assistant messages may carry reasoning in the reasoning_content field.
    Before applying the chat template, we must inline those tags back into content.
    """
    for message in messages:
        if message.get("role") != "assistant":
            continue
        reasoning_content = message.pop("reasoning_content", None)
        if reasoning_content is not None:
            content = message.get("content") or ""
            message["content"] = f"<think>{reasoning_content}</think>{content}"
    return messages


def _replace_prefix_tokens(
    eos_token_id,
    previous_turn_token_ids,
    retokeenized_previous_turn_token_ids,
    current_turn_token_ids,
):
    """Replace the token ids that are associated with the previous turn with the actual tokens
    from the previous generation (rather than the ones from the chat template application)."""

    # Strip the EOS from the previous turn token ids if it exists
    if previous_turn_token_ids[-1] == eos_token_id:
        previous_turn_token_ids = previous_turn_token_ids[:-1]

    # Find the last EOS token id in the previous turn token ids
    last_eos_token_id_index = len(retokeenized_previous_turn_token_ids) - 1
    for i in reversed(range(len(retokeenized_previous_turn_token_ids))):
        if current_turn_token_ids[i] == eos_token_id:
            last_eos_token_id_index = i
            break

    # Replace the current turn token ids with the tokens from the previous generation
    current_turn_additional_token_ids = current_turn_token_ids[last_eos_token_id_index:]

    # Return the previous turn token ids + the current turn token ids
    return previous_turn_token_ids + current_turn_additional_token_ids


try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False


try:
    from quart import Blueprint, Response, current_app, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    def apply_parsers(message_text, tools, parsers_list, tools_requested):
        """Runs CPU-intensive text parsing."""
        meta = {}
        for parser in parsers_list:
            if parser not in PARSER_MAPPING:
                raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")

            prev_text = message_text
            parsed_text, new_info = PARSER_MAPPING[parser].parse(message_text, tools=tools)
            if "tool_calls" in new_info:
                new_info["tool_calls"] = _normalize_tool_calls(new_info.get("tool_calls", []))
                if not tools_requested:
                    # Ignore incidental tool-call syntax in plain chat mode.
                    parsed_text = prev_text
                    new_info.pop("tool_calls", None)
            message_text = parsed_text

            assert not (
                meta.keys() & new_info.keys()
            ), "Multiple parsers found the same information."
            meta.update(new_info)

        return message_text, meta

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']

        req = await request.get_json()
        tools = req.get("tools", None)
        tools_requested = bool(tools)
        messages = req.get("messages")
        chat_template_kwargs = req.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            logger.warning(
                "Ignoring non-dict chat_template_kwargs: %s", type(chat_template_kwargs).__name__
            )
            chat_template_kwargs = {}
        # --- 1. Parse Messages ---
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)
        template_messages = _sanitize_messages_for_template(messages)
        template_messages = _reconstruct_reasoning_content(template_messages)
        template_tools = _sanitize_tools_for_template(tools)

        try:
            if (
                hasattr(tokenizer, 'apply_chat_template')
                and getattr(tokenizer, "chat_template", None) is not None
            ):
                prompt_tokens = tokenizer.apply_chat_template(
                    template_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=template_tools,
                    **chat_template_kwargs,
                )

                if req.get("prevent_retokenization", True):
                    # If we are avoiding retokenization, we need to replace some prompt tokens with the prompt/generation tokens from the previous generation
                    # This improves prefix cache hits and reduces logprob variation between training and inference.

                    eos_token_id = tokenizer.eos_id
                    assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

                    warnings.warn(
                        "Avoiding prefix retokenization."
                        " This is a patch that ensures subsequent generations are not retokenized differently than the previous generation."
                        " This may cause unexpected behavior if messages (including system messages) are altered between generations."
                    )

                    # Find the last assistant message
                    last_assistant_message_idx = None
                    for i in reversed(range(len(template_messages))):
                        if template_messages[i]["role"] == "assistant":
                            last_assistant_message_idx = i
                            break

                    # If there was a previous assistant message, we need to replace the prefix tokens with the tokens from the previous generation
                    if last_assistant_message_idx is not None:
                        messages_to_last_assistant_message = template_messages[
                            : last_assistant_message_idx + 1
                        ]

                        # Get the templated tokenization of just the previous generation
                        retokenized_previous_turn_token_ids = tokenizer.apply_chat_template(
                            messages_to_last_assistant_message,
                            tokenize=True,
                            add_generation_prompt=False,
                            tools=template_tools,
                            **chat_template_kwargs,
                        )

                        # Replace the prefix tokens with the tokens from the previous generation
                        last_assistant_message = template_messages[last_assistant_message_idx]
                        assert (
                            "prompt_token_ids" in last_assistant_message
                            and "generation_token_ids" in last_assistant_message
                        ), "Last assistant message must have prompt_token_ids and generation_token_ids from previous generation to avoid prefix retokenization"
                        previous_turn_token_ids = (
                            last_assistant_message["prompt_token_ids"]
                            + last_assistant_message["generation_token_ids"]
                        )
                        prompt_tokens = _replace_prefix_tokens(
                            eos_token_id,
                            previous_turn_token_ids,
                            retokenized_previous_turn_token_ids,
                            prompt_tokens,
                        )

            else:
                warnings.warn(
                    "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
                )
                prompt_tokens = tokenizer.tokenize(
                    "\n".join([message["content"] for message in messages])
                )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

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

            max_tokens = req.get("max_completion_tokens", None) or req.get("max_tokens", None)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                num_tokens_to_generate=(int(max_tokens) if max_tokens is not None else None),
                skip_prompt_log_probs=skip_prompt_log_probs,
                add_BOS=add_BOS,
            )
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

        request_idx = 0
        for result_item in batch_results:
            result = result_item if isinstance(result_item, dict) else result_item.serialize()
            result = unwrap_serialized_tensors(result)

            if result["status"] == "FAILED":
                if result["sampling_params"]["num_tokens_to_generate"] <= 0:
                    return Response(
                        f"Request {request_idx} failed due to context length overflow", status=400
                    )
                else:
                    return Response(
                        f"Request {request_idx} failed due to internal error {result['events']}",
                        status=500,
                    )

            prompt_tokens_out = result["prompt_tokens"]  # The engine can modify prompt_tokens.
            text_output = result["generated_text"]
            prompt_tokens_count = len(prompt_tokens_out) if prompt_tokens_out is not None else 0
            prompt_tokens_counts.append(prompt_tokens_count)

            logprobs_content = None
            if sampling_params.return_log_probs:
                token_logprobs = result.get('log_probs', [])

                tokens_to_decode = [[tok] for tok in result["generated_tokens"]]
                tokens = list(map(tokenizer.detokenize, tokens_to_decode))

                # Get top_n_logprobs if available
                generated_top_n_logprobs = result.get('generated_top_n_logprobs')

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

                    logprobs_content.append(
                        {
                            "token": tok,
                            "logprob": lp,
                            "bytes": list(tok.encode("utf-8")),
                            "top_logprobs": top_logprobs_list,
                        }
                    )

            metadata = {}
            message_text = text_output

            if parsers:
                message_text, metadata = apply_parsers(
                    message_text, req.get("tools", None), parsers, tools_requested
                )

            message = {"role": "assistant", "content": message_text}
            if metadata.get("tool_calls", []):
                message["tool_calls"] = metadata["tool_calls"]
            if "reasoning" in metadata:
                message["reasoning_content"] = metadata["reasoning"]

            # Replicate data in the message field for compatibility.
            message["prompt_token_ids"] = result["prompt_tokens"]
            message["generation_token_ids"] = result["generated_tokens"]
            message["generation_log_probs"] = result.get("generated_log_probs", [])
            return_log_probs = sampling_params.return_log_probs

            finish_reason = "tool_calls" if metadata.get("tool_calls", []) else "stop"
            if (
                len(result["generated_tokens"])
                >= result["sampling_params"]["num_tokens_to_generate"]
            ):
                finish_reason = "length"

            choice_data = {
                "index": request_idx,
                "message": message,
                "prompt_token_ids": result["prompt_tokens"],
                "generation_token_ids": result["generated_tokens"],
                "generation_log_probs": result.get("generated_log_probs", []),
                "raw_text": result["prompt"] + result["generated_text"],
                # 'logprobs' in chat API is an object containing 'content'
                # "logprobs": {"content": logprobs_content} if logprobs_content else None,
                "logprobs": {"content": logprobs_content} if return_log_probs else None,
                "finish_reason": finish_reason,
            }
            choice_data["policy_epoch"] = result["policy_epoch"]
            choice_data["kv_cache_epoch"] = result["kv_cache_epoch"]
            choice_data["num_evictions"] = sum(
                1 for e in result["events"] if e.get("type") == "EVICT"
            )
            if current_app.config['verbose']:
                logging.info(result)

            if result["routing_indices"] is not None:
                choice_data["moe_topk_indices"] = result["routing_indices"]
                if prompt_tokens_count:
                    choice_data["prompt_moe_topk_indices"] = result["routing_indices"][
                        :prompt_tokens_count
                    ]

            choices.append(choice_data)
            if choice_data["generation_log_probs"] is None:
                logger.warning(
                    "Generation log probs is None for request:\n%s", json.dumps(result, indent=4)
                )
            total_completion_tokens += len(result["generated_tokens"])
            request_idx += 1

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
