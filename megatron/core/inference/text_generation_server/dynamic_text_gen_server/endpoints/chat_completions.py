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

_TOKEN_ID_FIELDS_TO_REDACT = {
    "prompt_tokens",
    "remaining_prompt_tokens",
    "generated_tokens",
    "prompt_token_ids",
    "generation_token_ids",
}

_INDEX_FIELDS_TO_REDACT = {"routing_indices", "moe_topk_indices", "prompt_moe_topk_indices"}

_HASH_FIELDS_TO_REDACT = {"precomputed_block_hashes"}

_NUMERIC_SERIES_FIELDS_TO_REDACT = {"tpot"}


def _is_int_list_like(value):
    """Return True for integer lists, including nested integer lists."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, int) or _is_int_list_like(item) for item in value)


def _is_numeric_list_like(value):
    """Return True for numeric lists, including nested numeric lists."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, (int, float)) or _is_numeric_list_like(item) for item in value)


def _redact_token_id_lists_for_logging(value):
    """Redact verbose token-id arrays from logs."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if (
                key in _TOKEN_ID_FIELDS_TO_REDACT
                or key in _INDEX_FIELDS_TO_REDACT
                or key in _HASH_FIELDS_TO_REDACT
                or key.endswith("_token_ids")
                or key.endswith("_topk_indices")
                or key.endswith("_hashes")
            ) and _is_int_list_like(item):
                redacted[key] = "...truncated..."
            elif key in _NUMERIC_SERIES_FIELDS_TO_REDACT and _is_numeric_list_like(item):
                redacted[key] = "...truncated..."
            else:
                redacted[key] = _redact_token_id_lists_for_logging(item)
        return redacted
    if isinstance(value, list):
        return [_redact_token_id_lists_for_logging(item) for item in value]
    return value


def _get_field(obj, key, default=None):
    """Read a field from dict-like or object-like values."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _try_parse_jsonish(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return value


def _extract_declared_types(schema):
    """Recursively extract declared JSON-schema type names."""
    declared = set()
    if not isinstance(schema, dict):
        return declared

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        declared.add(schema_type.strip().lower())
    elif isinstance(schema_type, list):
        for item in schema_type:
            if isinstance(item, str):
                declared.add(item.strip().lower())

    for combinator in ("anyOf", "oneOf", "allOf"):
        options = schema.get(combinator)
        if isinstance(options, list):
            for option in options:
                declared.update(_extract_declared_types(option))
    return declared


def _get_tool_argument_schemas(tools):
    """Build function-name to argument-schema mapping from request tools."""
    schemas = {}
    if not isinstance(tools, list):
        return schemas

    for tool in tools:
        function = _get_field(tool, "function", {}) or {}
        function_name = _get_field(function, "name")
        params = _get_field(function, "parameters", {})
        if not isinstance(function_name, str) or not isinstance(params, dict):
            continue
        if isinstance(params.get("properties"), dict):
            schemas[function_name] = params.get("properties")
        else:
            schemas[function_name] = params
    return schemas


def _normalize_structured_tool_arguments(arguments, function_name, tool_argument_schemas):
    """Coerce structured (array/object) args from JSON strings to native types."""
    if not isinstance(arguments, dict):
        return arguments

    function_schema = tool_argument_schemas.get(function_name, {})
    if not isinstance(function_schema, dict):
        return arguments

    normalized = dict(arguments)
    for key in normalized:
        param_schema = function_schema.get(key)
        declared_types = _extract_declared_types(param_schema)
        if not (declared_types & {"array", "arr", "object", "dict", "list"}):
            continue
        parsed = _try_parse_jsonish(normalized[key])
        if isinstance(parsed, (dict, list)):
            normalized[key] = parsed
    return normalized


def _normalize_tool_calls(tool_calls, tools=None):
    """Normalize tool calls to OpenAI-compatible JSON primitives."""
    tool_argument_schemas = _get_tool_argument_schemas(tools)
    normalized = []
    for call in tool_calls or []:
        fn = _get_field(call, "function", {}) or {}
        fn_name = _get_field(fn, "name")
        fn_args = _get_field(fn, "arguments", "")
        if fn_name is None:
            continue
        if isinstance(fn_args, str):
            try:
                parsed_args = json.loads(fn_args)
            except (TypeError, ValueError):
                parsed_args = None
            if isinstance(parsed_args, dict):
                fn_args = json.dumps(
                    _normalize_structured_tool_arguments(
                        parsed_args, fn_name, tool_argument_schemas
                    ),
                    ensure_ascii=False,
                )
        elif isinstance(fn_args, dict):
            fn_args = json.dumps(
                _normalize_structured_tool_arguments(fn_args, fn_name, tool_argument_schemas),
                ensure_ascii=False,
            )
        else:
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


def _maybe_filter_parallel_tool_calls(tool_calls, parallel_tool_calls):
    """Filter to first tool call only when parallel_tool_calls is False.

    Matches vLLM's maybe_filter_parallel_tool_calls behavior.
    """
    if parallel_tool_calls:
        return tool_calls
    if tool_calls:
        return tool_calls[:1]
    return tool_calls


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
        content = msg_copy.get("content")
        # OpenAI-style multimodal/text content may arrive as a list of blocks.
        # HF/Jinja chat templates used by this server expect plain strings.
        if isinstance(content, list):
            text_chunks = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text":
                        text_chunks.append(str(chunk.get("text", "")))
                    elif "text" in chunk:
                        text_chunks.append(str(chunk.get("text", "")))
                elif isinstance(chunk, str):
                    text_chunks.append(chunk)
            msg_copy["content"] = "".join(text_chunks)
        elif isinstance(content, dict):
            msg_copy["content"] = str(content.get("text", ""))
        elif content is None:
            msg_copy["content"] = ""
        elif not isinstance(content, str):
            msg_copy["content"] = str(content)

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
                new_info["tool_calls"] = _normalize_tool_calls(
                    new_info.get("tool_calls", []), tools=tools
                )
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
        tool_choice = req.get("tool_choice", None)
        parallel_tool_calls = req.get("parallel_tool_calls", True)
        tools_requested = bool(tools) and tool_choice != "none"
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

                    # Find the last assistant message
                    last_assistant_message_idx = None
                    for i in reversed(range(len(template_messages))):
                        if template_messages[i]["role"] == "assistant":
                            last_assistant_message_idx = i
                            break

                    last_assistant_message = (
                        template_messages[last_assistant_message_idx]
                        if last_assistant_message_idx is not None
                        else None
                    )

                    # Only proceed if the last assistant message has the token IDs from a previous generation.
                    # Dataset-provided conversation history won't have these fields.
                    if (
                        last_assistant_message is not None
                        and "prompt_token_ids" in last_assistant_message
                        and "generation_token_ids" in last_assistant_message
                    ):
                        eos_token_id = tokenizer.eos_id
                        assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

                        warnings.warn(
                            "Avoiding prefix retokenization."
                            " This is a patch that ensures subsequent generations are not retokenized differently than the previous generation."
                            " This may cause unexpected behavior if messages (including system messages) are altered between generations."
                        )

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

                        # Replace the prefix tokens with the tokens from the previous generation.
                        # If prior token IDs are unavailable, fall back to normal retokenized prompt
                        # instead of failing the request.
                        prompt_token_ids = last_assistant_message.get("prompt_token_ids")
                        generation_token_ids = last_assistant_message.get("generation_token_ids")

                        if isinstance(prompt_token_ids, list) and isinstance(
                            generation_token_ids, list
                        ):
                            previous_turn_token_ids = prompt_token_ids + generation_token_ids
                            prompt_tokens = _replace_prefix_tokens(
                                eos_token_id,
                                previous_turn_token_ids,
                                retokenized_previous_turn_token_ids,
                                prompt_tokens,
                            )
                        else:
                            logger.warning(
                                "Last assistant message missing prompt_token_ids/"
                                "generation_token_ids; skipping prefix replacement."
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

        # --- 4. Check for failed requests ---
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
            return Response(f"Inference request(s) failed: {error_detail}", status=status)

        # --- 5. Format OpenAI Response ---
        choices = []
        total_completion_tokens = 0
        prompt_tokens_counts = []

        request_idx = 0
        for result_item in batch_results:
            result = unwrap_serialized_tensors(result_item)

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
                    message_text, tools, parsers, tools_requested
                )

            normalized_tool_calls = metadata.get("tool_calls", [])

            # Apply parallel_tool_calls filtering (matches vLLM behavior)
            normalized_tool_calls = _maybe_filter_parallel_tool_calls(
                normalized_tool_calls, parallel_tool_calls
            )

            # Determine content based on tool_choice (matches vLLM behavior):
            # - Named tool choice or "required": content is empty string
            # - Otherwise: content is the parsed message text
            is_named_tool_choice = isinstance(tool_choice, dict) and "function" in tool_choice
            if normalized_tool_calls and (is_named_tool_choice or tool_choice == "required"):
                content = ""
            else:
                content = message_text if message_text is not None else ""

            message = {"role": "assistant", "content": content}
            if normalized_tool_calls:
                message["tool_calls"] = normalized_tool_calls
            if "reasoning" in metadata:
                message["reasoning_content"] = metadata["reasoning"]

            # Replicate data in the message field for compatibility.
            message["prompt_token_ids"] = result["prompt_tokens"]
            message["generation_token_ids"] = result["generated_tokens"]
            message["generation_log_probs"] = result.get("generated_log_probs", [])
            return_log_probs = sampling_params.return_log_probs

            # Determine finish_reason following vLLM conventions:
            # - "tool_calls" for auto or required tool choice when tools are called
            # - "stop" for named tool choice (even when tools are called)
            # - "length" when max tokens is reached
            if (
                len(result["generated_tokens"])
                >= result["sampling_params"]["num_tokens_to_generate"]
            ):
                finish_reason = "length"
            elif normalized_tool_calls and not is_named_tool_choice:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

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
                logging.info(_redact_token_id_lists_for_logging(result))

            if result["routing_indices"] is not None:
                choice_data["moe_topk_indices"] = result["routing_indices"]
                if prompt_tokens_count:
                    choice_data["prompt_moe_topk_indices"] = result["routing_indices"][
                        :prompt_tokens_count
                    ]

            choices.append(choice_data)
            if choice_data["generation_log_probs"] is None:
                logger.warning(
                    "Generation log probs is None for request:\n%s",
                    json.dumps(_redact_token_id_lists_for_logging(result), indent=4),
                )
            total_completion_tokens += len(result["generated_tokens"])
            request_idx += 1

        prompt_token_count = max(prompt_tokens_counts) if prompt_tokens_counts else 0
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
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
