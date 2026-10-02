# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import base64
import ipaddress
import json
import logging
import socket
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Optional

_MEDIA_FETCH_TIMEOUT_S = 5.0
_MEDIA_FETCH_USER_AGENT = "megatron-inference"

from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    prepare_multimodal_data,
)
from megatron.core.inference.utils import detokenize_tokens, model_eos_token_ids
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

from ..incremental_detokenizer import HuggingFaceFastIncrementalDetokenizer
from ..openai_streaming import (
    StreamingChatParser,
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


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject HTTP redirects so a 3xx to a private address can't bypass the
    pre-fetch allowlist check."""

    def http_error_301(self, req, fp, code, msg, headers):
        """Turn a 3xx redirect into an HTTPError so the fetch fails closed."""
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects disabled for image_url fetches", headers, fp
        )

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


_no_redirect_opener = urllib.request.build_opener(_NoRedirectHandler())


def _extract_media_url_bytes(url: str, *, max_fetch_bytes: int | None = None) -> bytes:
    """Extract bytes from an OpenAI-style media URL.

    Supports base64-encoded data URLs (``data:image/...;base64,<b64>``) and
    plain ``http(s)://`` URLs. Data URLs are already bounded by the server's
    request-body limit; remote responses bypass it, so ``max_fetch_bytes``
    bounds them instead.
    """
    if url.startswith("data:"):
        try:
            metadata, b64_data = url.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed media data URL: {url[:40]!r}") from exc
        return base64.b64decode(b64_data)
    if url.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(url)
        if not parsed.hostname:
            raise ValueError(f"Invalid media URL: {url[:40]!r}")
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
        except (socket.gaierror, ValueError) as exc:
            raise ValueError(f"Cannot resolve media URL host: {parsed.hostname}") from exc
        # Refuse SSRF-prone destinations (loopback, RFC1918, link-local,
        # multicast, reserved, unspecified). Public addresses only.
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Refusing to fetch media from non-public address: {parsed.hostname}")
        req = urllib.request.Request(url, headers={"User-Agent": _MEDIA_FETCH_USER_AGENT})
        with _no_redirect_opener.open(req, timeout=_MEDIA_FETCH_TIMEOUT_S) as response:
            if max_fetch_bytes is None:
                return response.read()
            data = response.read(max_fetch_bytes + 1)
        if len(data) > max_fetch_bytes:
            raise ValueError(f"Media at {parsed.hostname} exceeds {max_fetch_bytes} byte limit")
        return data
    raise ValueError(f"Unsupported media URL scheme: {url[:40]!r}")


def _extract_multimodal_from_messages(
    messages, prompt_config: MultimodalPromptConfig, max_fetch_bytes: int | None = None
):
    """Extract media bytes and replace structured blocks with internal slots.

    Remote image fetching is blocking, so callers must run this function off
    the event loop. ``max_fetch_bytes`` bounds each remote media response.
    """
    if not isinstance(messages, list):
        return messages, [], [], []

    rewritten = []
    image_bytes_list: list[bytes] = []
    video_bytes_list: list[bytes] = []
    media_slots = []

    def add_slot(modality, message_index):
        """Preserve a media block's position while the chat template renders text.

        The actual bytes travel separately. After rendering,
        _tokenize_with_media_slots_sync replaces this sentinel with the
        model-specific MediaPromptSpec tokens.
        """
        sentinel = f"__MCORE_MEDIA_SLOT_{len(media_slots)}__"
        media_slots.append((sentinel, modality, message_index))
        return {"type": "text", "text": sentinel}

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            rewritten.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue

        new_chunks = []
        found_modalities = set()
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "image_url":
                url = chunk.get("image_url", {}).get("url", "")
                if not url:
                    continue
                try:
                    image_bytes_list.append(
                        _extract_media_url_bytes(url, max_fetch_bytes=max_fetch_bytes)
                    )
                except Exception as e:
                    # Dropping the image would answer the request as if it were
                    # text-only, handing the client a confident answer about an
                    # image the model never saw. Surface it as a 400 instead.
                    raise ValueError(f"Failed to load image_url: {e}") from e
                new_chunks.append(add_slot("image", message_index))
                found_modalities.add("image")
            elif isinstance(chunk, dict) and chunk.get("type") in {"video_url", "input_video"}:
                video_value = chunk.get("video_url") or chunk.get("video")
                url = video_value.get("url", "") if isinstance(video_value, dict) else video_value
                if not isinstance(url, str) or not url.startswith("data:"):
                    raise ValueError("Megatron chat video inputs must be base64 data URLs.")
                try:
                    video_bytes_list.append(
                        _extract_media_url_bytes(url, max_fetch_bytes=max_fetch_bytes)
                    )
                except Exception as e:
                    raise ValueError(f"Failed to load video_url: {e}") from e
                new_chunks.append(add_slot("video", message_index))
                found_modalities.add("video")
            else:
                new_chunks.append(chunk)

        if found_modalities:
            input_markers = {
                prompt_config.get_spec(modality).input_marker for modality in found_modalities
            } - {None}
            for index, chunk in enumerate(new_chunks):
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    chunk = dict(chunk)
                    for marker in input_markers:
                        chunk["text"] = str(chunk.get("text", "")).replace(marker, "")
                    new_chunks[index] = chunk
            msg_copy = dict(message)
            msg_copy["content"] = new_chunks
            rewritten.append(msg_copy)
        else:
            rewritten.append(message)

    if image_bytes_list and video_bytes_list:
        raise ValueError("Mixing image and video blocks in one request is not supported.")
    return rewritten, image_bytes_list, video_bytes_list, media_slots


def _sanitize_messages_for_template(messages, media_slots=(), prompt_config=None):
    """Prepare messages so tokenizer chat templates can safely consume them.

    This lowers structured media content according to the model prompt contract
    and normalizes tool-call argument payloads inside each message:
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
    media_modalities_by_message = {}
    media_sentinels_by_message = {}
    for _sentinel, modality, message_index in media_slots:
        media_modalities_by_message.setdefault(message_index, set()).add(modality)
        media_sentinels_by_message.setdefault(message_index, set()).add(_sentinel)

    for message_index, message in enumerate(messages):
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
            if prompt_config is not None and prompt_config.content_part_order == "media_first":
                media_sentinels = media_sentinels_by_message.get(message_index, set())
                media_chunks = [chunk for chunk in text_chunks if chunk in media_sentinels]
                non_media_chunks = [chunk for chunk in text_chunks if chunk not in media_sentinels]
                text_chunks = media_chunks + non_media_chunks
            separator = ""
            message_modalities = media_modalities_by_message.get(message_index, set())
            if message_modalities:
                if prompt_config is None:
                    raise ValueError("Media content normalization requires a prompt config.")
                separators = {
                    prompt_config.get_spec(modality).content_part_separator
                    for modality in message_modalities
                }
                if len(separators) != 1:
                    raise ValueError(
                        "Media types in one message must use the same content-part separator."
                    )
                separator = separators.pop()
            msg_copy["content"] = separator.join(chunk for chunk in text_chunks if chunk)
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


# Keys a client may never supply via `chat_template_kwargs`.
#
# `chat_template` replaces the Jinja template that the server renders for the
# request. The template is server/tokenizer configuration (`--chat-template` or
# the tokenizer's own template), never request data. Honoring a caller-supplied
# template lets an unauthenticated POST hand us arbitrary Jinja that we then
# compile and render synchronously: the sandbox blocks attribute escapes, but it
# does not bound work, so nested `range()` loops or unbounded string
# multiplication pin the worker and stall every other in-flight request on it.
_DISALLOWED_CHAT_TEMPLATE_KWARGS = frozenset({"chat_template"})


def _sanitize_chat_template_kwargs(raw_kwargs):
    """Drop request-supplied `chat_template_kwargs` entries that are not caller-controllable.

    Returns a new dict; the caller's object is never mutated. Non-dict input
    (including `None`) yields an empty dict.
    """
    if not isinstance(raw_kwargs, dict):
        if raw_kwargs is not None:
            logger.warning("Ignoring non-dict chat_template_kwargs: %s", type(raw_kwargs).__name__)
        return {}

    sanitized = {k: v for k, v in raw_kwargs.items() if k not in _DISALLOWED_CHAT_TEMPLATE_KWARGS}
    rejected = sorted(set(raw_kwargs) - set(sanitized))
    if rejected:
        logger.warning(
            "Ignoring disallowed chat_template_kwargs key(s) from request: %s. "
            "The chat template is server configuration; use --chat-template to set it.",
            ", ".join(rejected),
        )
    return sanitized


def _replace_prefix_tokens(
    eos_token_ids,
    previous_turn_token_ids,
    retokenized_previous_turn_token_ids,
    current_turn_token_ids,
):
    """Replace the token ids that are associated with the previous turn with the actual tokens
    from the previous generation (rather than the ones from the chat template application)."""

    if not previous_turn_token_ids:
        return current_turn_token_ids

    eos_token_ids = _normalize_eos_token_ids(eos_token_ids)

    # Find the boundary of the current sequence's prompt.
    current_turn_additional_token_ids = _suffix_tokens_after_prefix(
        eos_token_ids, retokenized_previous_turn_token_ids, current_turn_token_ids
    )
    if previous_turn_token_ids[-1] in eos_token_ids:
        # Preserve the exact EOS emitted previously. The rendered suffix begins
        # with its own boundary EOS, which may be a different accepted EOS ID.
        current_turn_additional_token_ids = current_turn_additional_token_ids[1:]

    # Return the previous turn token ids + the current turn token ids
    return previous_turn_token_ids + current_turn_additional_token_ids


def _has_previous_turn_tokens(last_assistant_message):
    """True when the last assistant message carries the token ids of a previous
    Megatron-Inference response, so the endpoint can replace the prefix with the exact prior turn here.
    Dataset-provided conversation history won't have these fields."""
    if last_assistant_message is None:
        return False
    prompt_token_ids = last_assistant_message.get("prompt_token_ids")
    generation_token_ids = last_assistant_message.get("generation_token_ids")
    # Check that we have non-empty prompt or generation tokens.
    return (
        isinstance(prompt_token_ids, list)
        and isinstance(generation_token_ids, list)
        and bool(prompt_token_ids or generation_token_ids)
    )


def _last_assistant_message(template_messages):
    """Return ``(index, message)`` of the last assistant turn, or ``(None, None)``."""
    for i in reversed(range(len(template_messages))):
        if template_messages[i]["role"] == "assistant":
            return i, template_messages[i]
    return None, None


def _replace_prefix_tokens_metadata(eos_token_ids, template_prefix_token_ids, offload_params):
    """Ship the rendered prior-turn tokens so the engine's RequestPromptPreparer can replace the
    prefix with the exact prior tokens itself (NeMo RL's ``replace_prefix_tokens``)."""
    return {
        **offload_params,
        PREFIX_TEMPLATE_TOKEN_IDS_FIELD: list(template_prefix_token_ids),
        PREFIX_EOS_TOKEN_ID_FIELD: _serialize_eos_token_ids(eos_token_ids),
    }


def _expanded_prefix_stitching_metadata(prefix_media_count, expanded_prefix_token_count):
    """Mark the exact, already-expanded prefix so the engine only expands the tokens after it."""
    return {
        PREFIX_MEDIA_COUNT_FIELD: prefix_media_count,
        PREFIX_EXPANDED_TOKEN_COUNT_FIELD: expanded_prefix_token_count,
    }


def _normalize_eos_token_ids(eos_token_ids):
    """Normalize one or more EOS IDs to a validated set."""
    if type(eos_token_ids) is int:
        eos_token_ids = [eos_token_ids]
    if (
        not isinstance(eos_token_ids, (list, tuple, set, frozenset))
        or not eos_token_ids
        or not all(type(token_id) is int for token_id in eos_token_ids)
    ):
        raise ValueError("EOS token IDs must be a non-empty integer or collection of integers.")
    return frozenset(eos_token_ids)


def _serialize_eos_token_ids(eos_token_ids):
    """Serialize one or more EOS IDs as a JSON-compatible list."""
    return sorted(_normalize_eos_token_ids(eos_token_ids))


def _suffix_tokens_after_prefix(eos_token_ids, template_prefix_token_ids, current_tokens):
    """Return the current-turn suffix, beginning at the prefix's final EOS."""
    # Count the number of EOS tokens in the retokenized prefix.
    eos_token_ids = _normalize_eos_token_ids(eos_token_ids)
    eos_count = sum(token_id in eos_token_ids for token_id in template_prefix_token_ids)
    if eos_count <= 0:
        raise ValueError(
            "Could not locate an EOS-delimited previous turn: the chat template's turn "
            f"terminator is not among the model EOS token IDs {sorted(eos_token_ids)}."
        )

    # Scan current_tokens from beginning to end. Return the suffix after eos_count
    # EOS tokens have been seen.
    # This guards against changes in templating or even upstream modification of
    # the prefix tokens to ensure we retrieve the current turn's suffix / prompt.
    seen_eos = 0
    for position, token_id in enumerate(current_tokens):
        if token_id in eos_token_ids:
            seen_eos += 1
            if seen_eos == eos_count:
                return current_tokens[position:]
    raise ValueError(
        f"Expected {eos_count} EOS token(s) before the new turn, but found only {seen_eos}."
    )


def _contains_model_media_token(token_ids, tokenizer, prompt_config):
    """Whether exact prior model tokens still reference image/video embeddings."""
    if not isinstance(token_ids, list) or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return False

    media_token_ids = set()
    for modality in ("image", "video"):
        spec = prompt_config.get_spec(modality)
        token_id = tokenizer.convert_tokens_to_ids(spec.model_token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            media_token_ids.add(int(token_id))
    return any(type(token_id) is int and token_id in media_token_ids for token_id in token_ids)


def _apply_chat_template_sync(
    tokenizer, messages, tools, chat_template_kwargs, add_generation_prompt=True
):
    """Apply the chat template and coerce to `list[int]`, for use in a worker thread.

    The coercion runs here too: it walks every token, so leaving it on the event loop
    would keep part of the stall this offload exists to remove.
    """
    return _coerce_to_token_id_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **chat_template_kwargs,
        )
    )


def _coerce_to_token_id_list(result):
    """Convert the return value of `tokenizer.apply_chat_template` to `list[int]`.

    transformers >= 5.x.x sometimes returns a `BatchEncoding` object instead of a `list[int]`.
    """
    # BatchEncoding / dict-like with input_ids
    if isinstance(result, dict) or hasattr(result, "input_ids"):
        ids = result["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    # Fast-tokenizer Encoding object
    if hasattr(result, "ids"):
        ids = result.ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
    # Raw tensor / ndarray
    if hasattr(result, "tolist"):
        ids = result.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return ids
    # Plain list
    return list(result)


def _tokenize_with_media_slots_sync(
    chat_tok,
    messages,
    media_slots,
    prompt_config,
    *,
    tools,
    chat_template_kwargs,
    add_generation_prompt=True,
):
    """Render a chat template and lower internal media slots to model tokens.

    Synchronous, and run whole on the frontend's tokenize executor rather than
    piecewise. Rendering is only the first of 3N+1 tokenizer calls for N media
    slots -- each slot encodes the text before it, then the model token's prefix
    and suffix -- and awaiting just the render left the rest on the event loop,
    where they stall every other request this replica owns.

    One hop to the executor also means one thread touches the tokenizer for the
    whole request. HF tokenizers are not thread-safe, and this runs on the
    executor's private copy.
    """
    rendered = chat_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **chat_template_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("Multimodal chat template rendering must return a string.")

    positioned_slots = []
    for sentinel, modality, _message_index in media_slots:
        if rendered.count(sentinel) != 1:
            raise ValueError(f"Chat template did not preserve media slot {sentinel}.")
        positioned_slots.append((rendered.index(sentinel), sentinel, modality))

    prompt_tokens = []
    cursor = 0
    for position, sentinel, modality in sorted(positioned_slots):
        prompt_tokens.extend(
            _coerce_to_token_id_list(chat_tok(rendered[cursor:position], add_special_tokens=False))
        )
        spec = prompt_config.get_spec(modality)
        resolved_id = (
            chat_tok.convert_tokens_to_ids(spec.model_token)
            if hasattr(chat_tok, "convert_tokens_to_ids")
            else None
        )
        if resolved_id == getattr(chat_tok, "unk_token_id", None):
            resolved_id = None
        if resolved_id is None:
            raise ValueError(f"Tokenizer does not define media token {spec.model_token!r}.")
        prompt_tokens.extend(
            _coerce_to_token_id_list(chat_tok(spec.prefix, add_special_tokens=False))
        )
        prompt_tokens.append(int(resolved_id))
        prompt_tokens.extend(
            _coerce_to_token_id_list(chat_tok(spec.suffix, add_special_tokens=False))
        )
        cursor = position + len(sentinel)

    prompt_tokens.extend(
        _coerce_to_token_id_list(chat_tok(rendered[cursor:], add_special_tokens=False))
    )
    return prompt_tokens


def apply_parsers(
    message_text, tools, parsers_list, tools_requested, chat_template_kwargs=None, finished=True
):
    """Runs CPU-intensive text parsing."""
    for parser in parsers_list:
        if parser not in PARSER_MAPPING:
            raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")

    implicit_reasoning_end_markers = (
        tuple(
            marker
            for parser_name in parsers_list
            for marker in getattr(PARSER_MAPPING[parser_name], "implicit_reasoning_end_markers", ())
        )
        if tools_requested
        else ()
    )

    meta = {}
    for parser in parsers_list:
        prev_text = message_text
        parsed_text, new_info = PARSER_MAPPING[parser].parse(
            message_text,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            implicit_reasoning_end_markers=implicit_reasoning_end_markers,
            finished=finished,
        )
        if "tool_calls" in new_info:
            new_info["tool_calls"] = _normalize_tool_calls(
                new_info.get("tool_calls", []), tools=tools
            )
            if not tools_requested:
                # Ignore incidental tool-call syntax in plain chat mode.
                parsed_text = prev_text
                new_info.pop("tool_calls", None)
        message_text = parsed_text

        assert not (meta.keys() & new_info.keys()), "Multiple parsers found the same information."
        meta.update(new_info)

    return message_text, meta


def _chat_tokenizer(tokenizer):
    """Prefer the underlying HF tokenizer for chat-template work."""
    hf_tokenizer = getattr(getattr(tokenizer, "_tokenizer", None), "tokenizer", None)
    return hf_tokenizer if hf_tokenizer is not None else tokenizer


def _has_chat_template(tokenizer, chat_template_kwargs):
    """True when a template can be rendered: from the tokenizer or from server injection."""
    return hasattr(tokenizer, "apply_chat_template") and (
        getattr(tokenizer, "chat_template", None) is not None
        or chat_template_kwargs.get("chat_template") is not None
    )


class _TemplateRenderer:
    """Renders chat messages to token ids on the tokenize executor for one request."""

    def __init__(self, executor, tokenizer, prompt_config, tools, chat_template_kwargs) -> None:
        self.executor = executor
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        self.tools = tools
        self.chat_template_kwargs = chat_template_kwargs

    async def tokenize(self, messages, media_slots, *, add_generation_prompt):
        """Render and tokenize `messages`; media slots take the media-aware path."""
        if media_slots:
            render = partial(
                _tokenize_with_media_slots_sync,
                self.tokenizer,
                messages,
                media_slots,
                self.prompt_config,
                tools=self.tools,
                chat_template_kwargs=self.chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            render = partial(
                _apply_chat_template_sync,
                self.tokenizer,
                messages,
                self.tools,
                self.chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
            )
        return await asyncio.get_running_loop().run_in_executor(self.executor, render)


class _PrefixMode(Enum):
    """Where the exact tokens of the previous turn come from, if anywhere."""

    NONE = "none"
    # The client echoed back the token ids of a previous response on the last assistant message;
    # the prefix is replaced here.
    EXACT = "exact"
    # The client sent offload_params, so the exact tokens live in a store the engine's
    # RequestPromptPreparer can reach; we ship what the preparer needs.
    OFFLOADED = "offloaded"


@dataclass(frozen=True)
class _PrefixSource:
    """The previous-turn source decision together with the last assistant turn it refers to."""

    mode: _PrefixMode
    last_assistant_idx: Optional[int]
    last_assistant_message: Optional[dict]


def _resolve_prefix_source(template_messages, prevent_retokenization, offload_params):
    """Decide the previous-turn source; the two sources are mutually exclusive."""
    last_idx, last_message = _last_assistant_message(template_messages)
    has_previous_turn_tokens = _has_previous_turn_tokens(last_message)
    if has_previous_turn_tokens and offload_params is not None:
        raise ValueError(
            "prompt_token_ids/generation_token_ids on the last assistant message and "
            "'offload_params' are mutually exclusive prefix sources"
        )
    if prevent_retokenization and has_previous_turn_tokens:
        mode = _PrefixMode.EXACT
    elif offload_params is not None and last_message is not None:
        mode = _PrefixMode.OFFLOADED
    else:
        mode = _PrefixMode.NONE
    return _PrefixSource(mode, last_idx, last_message)


def _eos_token_ids(tokenizer):
    """Every EOS id the model may emit: the generation_config set plus the tokenizer's own."""
    eos_token_ids = set(model_eos_token_ids(tokenizer))
    if getattr(tokenizer, "eos_id", None) is not None:
        eos_token_ids.add(tokenizer.eos_id)
    assert eos_token_ids, "Your tokenizer must have an EOS token ID!"
    return eos_token_ids


async def _stitch_previous_turn(
    renderer,
    tokenizer,
    prefix_source,
    prompt_tokens,
    offload_params,
    template_messages,
    media_slots,
):
    """Replace the re-rendered previous turn with its exact tokens, here or via the engine."""
    last_idx = prefix_source.last_assistant_idx
    last_message = prefix_source.last_assistant_message
    messages_to_last_assistant = template_messages[: last_idx + 1]
    previous_media_slots = [slot for slot in media_slots if slot[2] <= last_idx]
    if (
        prefix_source.mode is _PrefixMode.EXACT
        and not previous_media_slots
        and _contains_model_media_token(
            last_message["prompt_token_ids"], renderer.tokenizer, renderer.prompt_config
        )
    ):
        raise ValueError(
            "The exact previous prompt contains media tokens, but its image/video "
            "payload is missing from message history. Preserve prior media content "
            "when using prevent_retokenization."
        )
    eos_token_ids = _eos_token_ids(tokenizer)
    warnings.warn(
        "Avoiding prefix retokenization. "
        "This is a patch that ensures subsequent generations are not retokenized "
        "differently than the previous generation. "
        "This may cause unexpected behavior if messages (including system messages) "
        "are altered between generations."
    )
    # The templated tokenization of just the previous generation.
    retokenized_prefix = await renderer.tokenize(
        messages_to_last_assistant, previous_media_slots, add_generation_prompt=False
    )

    if prefix_source.mode is _PrefixMode.OFFLOADED:
        offload_params = _replace_prefix_tokens_metadata(
            eos_token_ids, retokenized_prefix, offload_params
        )
        if previous_media_slots:
            # Post-expansion stitching needs the media count to know how many subsequent
            # media tokens to expand.
            offload_params[PREFIX_MEDIA_COUNT_FIELD] = len(previous_media_slots)
        return prompt_tokens, offload_params

    previous_turn_token_ids = (
        last_message["prompt_token_ids"] + last_message["generation_token_ids"]
    )
    prompt_tokens = _replace_prefix_tokens(
        eos_token_ids, previous_turn_token_ids, retokenized_prefix, prompt_tokens
    )
    if previous_media_slots:
        # The previous turn is already expanded; the engine only expands the media tokens after it.
        offload_params = _expanded_prefix_stitching_metadata(
            len(previous_media_slots), len(previous_turn_token_ids)
        )
    return prompt_tokens, offload_params


async def _render_prompt(
    renderer, tokenizer, template_messages, media_slots, prefix_source, offload_params
):
    """Tokenize the conversation; when a previous-turn source exists, stitch its exact tokens in."""
    prompt_tokens = await renderer.tokenize(
        template_messages, media_slots, add_generation_prompt=True
    )
    if prefix_source.mode is _PrefixMode.NONE:
        return prompt_tokens, offload_params
    return await _stitch_previous_turn(
        renderer,
        tokenizer,
        prefix_source,
        prompt_tokens,
        offload_params,
        template_messages,
        media_slots,
    )


def _tokenize_without_template(tokenizer, messages, media_slots):
    """Fallback for tokenizers without a chat template: join the message contents with newlines."""
    if media_slots:
        raise ValueError("Multimodal chat requests require a chat template.")
    warnings.warn("Tokenizer does not support 'apply_chat_template'. Using tokenize instead.")
    return tokenizer.tokenize("\n".join([message["content"] for message in messages]))


def _apply_bos_policy(prompt_tokens, tokenizer, add_BOS, offload_params):
    """Strip leading BOS tokens and re-add one if requested, keeping the expanded-prefix count."""
    if getattr(tokenizer, "bos", None) is None:
        return prompt_tokens
    length_before = len(prompt_tokens)
    start_idx = 0
    while start_idx < len(prompt_tokens) and prompt_tokens[start_idx] == tokenizer.bos:
        start_idx += 1
    prompt_tokens = prompt_tokens[start_idx:]
    if add_BOS:
        prompt_tokens = [tokenizer.bos] + prompt_tokens
    if offload_params and PREFIX_EXPANDED_TOKEN_COUNT_FIELD in offload_params:
        offload_params[PREFIX_EXPANDED_TOKEN_COUNT_FIELD] += len(prompt_tokens) - length_before
    return prompt_tokens


@dataclass(frozen=True)
class _ChatRequestOptions:
    """The non-sampling options of one chat request, read once from the body."""

    tools: Any
    tool_choice: Any
    parallel_tool_calls: bool
    tools_requested: bool
    chat_template_kwargs: dict
    prevent_retokenization: bool
    return_tokenized_data: bool
    return_raw_text: bool
    stream: bool
    include_usage: bool

    @property
    def is_named_tool_choice(self):
        """True when `tool_choice` names one function, which vLLM reports as a plain stop."""
        return isinstance(self.tool_choice, dict) and "function" in self.tool_choice

    @classmethod
    def from_request(cls, req, app_config):
        """Read the options off the request body, applying the server's defaults."""
        tools = req.get("tools", None)
        tool_choice = req.get("tool_choice", None)
        chat_template_kwargs = _sanitize_chat_template_kwargs(req.get("chat_template_kwargs"))
        # The server-configured chat template (e.g. pretraining.jinja for VLM checkpoints) is
        # loaded once at startup from --chat-template. It is the only path by which a template
        # reaches the renderer; requests cannot override it (see _sanitize_chat_template_kwargs).
        server_chat_template = app_config.get("chat_template", None)
        if server_chat_template:
            chat_template_kwargs["chat_template"] = server_chat_template
        prevent_retokenization = req.get(
            "prevent_retokenization", not app_config.get("eval_mode", False)
        )
        # Tolerate a malformed `stream_options` instead of 500ing requests that carry one.
        stream_options = req.get("stream_options")
        include_usage = (
            bool(stream_options.get("include_usage", False))
            if isinstance(stream_options, dict)
            else False
        )
        return cls(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=req.get("parallel_tool_calls", True),
            tools_requested=bool(tools) and tool_choice != "none",
            chat_template_kwargs=chat_template_kwargs,
            prevent_retokenization=prevent_retokenization,
            # The engine keeps prompt_tokens on the payload only when the client wants them back:
            # return_tokenized_data (implied by prevent_retokenization) echoes the ids and
            # return_raw_text detokenizes them.
            return_tokenized_data=req.get("return_tokenized_data", False) or prevent_retokenization,
            return_raw_text=req.get("return_raw_text", False),
            stream=bool(req.get("stream", False)),
            include_usage=include_usage,
        )


def _streaming_chat_parsers(parsers, options, n):
    """One `StreamingChatParser` per choice, or `None` when no parsers are configured."""
    if not parsers:
        return None
    marker_prefixes = (
        tuple(
            marker
            for parser_name in parsers
            for marker in getattr(PARSER_MAPPING[parser_name], "streaming_markers", ())
        )
        if options.tools_requested
        else ()
    )

    def parse_streaming_text(text, finished=False):
        parsed_text, metadata = apply_parsers(
            text,
            options.tools,
            parsers,
            options.tools_requested,
            chat_template_kwargs=options.chat_template_kwargs,
            finished=finished,
        )
        metadata["tool_calls"] = _maybe_filter_parallel_tool_calls(
            metadata.get("tool_calls", []), options.parallel_tool_calls
        )
        return parsed_text, metadata

    return [
        StreamingChatParser(
            parse_streaming_text,
            marker_prefixes=marker_prefixes,
            named_tool_choice=options.is_named_tool_choice,
        )
        for _ in range(n)
    ]


def _logprob_token(token, logprob, top_logprobs=None):
    """One OpenAI chat-logprobs entry; `top_logprobs` is attached only for the sampled token."""
    entry = {"token": token, "logprob": logprob, "bytes": list(token.encode("utf-8"))}
    if top_logprobs is not None:
        entry["top_logprobs"] = top_logprobs
    return entry


def _chat_logprobs_content(result, tokenizer):
    """The `logprobs.content` list for one choice; non-finite logprobs are clamped for JSON."""
    token_logprobs = json_safe_logprobs(result.get("generated_log_probs") or [])
    tokens = [tokenizer.detokenize([tok]) for tok in result["generated_tokens"]]
    generated_top_n_logprobs = json_safe_top_n_logprobs(
        result.get("generated_top_n_logprobs") or []
    )
    content = []
    for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
        top_logprobs = (
            [
                _logprob_token(token_str, logprob)
                for token_str, logprob in generated_top_n_logprobs[i].items()
            ]
            if i < len(generated_top_n_logprobs)
            else []
        )
        content.append(_logprob_token(tok, lp, top_logprobs=top_logprobs))
    return content


def format_chat_response(batch_results, sampling_params, tokenizer, options, *, parsers, verbose):
    """Build the OpenAI chat-completion body from the finished replies.

    `finish_reason` follows vLLM: "length" at the token limit,
    "tool_calls" when tools were called under auto/required tool choice,
    otherwise "stop" (a named tool choice also reports "stop" and empties `content`).
    """
    results, response_uid, response_metadata = unwrap_batch(batch_results)
    choices = []
    total_completion_tokens = 0
    prompt_tokens_counts = []
    cached_tokens_counts = []
    for request_idx, result in enumerate(results):
        text_output = detokenize_tokens(
            tokenizer,
            result["generated_tokens"],
            remove_EOD=not sampling_params.detokenize_stop_sequence,
        )
        prompt_len = reply_prompt_length(result)
        prompt_tokens_counts.append(prompt_len)
        cached_tokens_counts.append(result.get("num_cached_tokens", 0))
        # Under payload offload the engine dropped the per-token log probs from the reply.
        payload_offloaded = bool(result.get("payload_offloaded"))
        logprobs_content = None
        if sampling_params.return_log_probs and not payload_offloaded:
            logprobs_content = _chat_logprobs_content(result, tokenizer)

        message_text = text_output
        metadata = {}
        if parsers:
            message_text, metadata = apply_parsers(
                message_text,
                options.tools,
                parsers,
                options.tools_requested,
                chat_template_kwargs=options.chat_template_kwargs,
            )
        normalized_tool_calls = _maybe_filter_parallel_tool_calls(
            metadata.get("tool_calls", []), options.parallel_tool_calls
        )
        if normalized_tool_calls and (
            options.is_named_tool_choice or options.tool_choice == "required"
        ):
            content = ""
        else:
            content = message_text if message_text is not None else ""

        message = {"role": "assistant", "content": content}
        if normalized_tool_calls:
            message["tool_calls"] = normalized_tool_calls
        if "reasoning" in metadata:
            message["reasoning_content"] = metadata["reasoning"]
        if options.return_tokenized_data and not payload_offloaded:
            # Wire contract matches vLLM: prompt_token_ids are model-input tokens
            # (post vision/video expansion).
            message["prompt_token_ids"] = result["prompt_tokens"]
            message["generation_token_ids"] = result["generated_tokens"]
        if options.return_raw_text and not payload_offloaded:
            message["raw_text"] = tokenizer.detokenize(result["prompt_tokens"]) + text_output
        if not payload_offloaded:
            # Small RL/debug scalars (a few bytes each); harmless to keep for compatibility.
            message["generation_log_probs"] = result.get("generated_log_probs", [])

        reason = finish_reason(result)
        if reason == "stop" and normalized_tool_calls and not options.is_named_tool_choice:
            reason = "tool_calls"

        choice_data = {
            "index": request_idx,
            "message": message,
            # 'logprobs' in chat API is an object containing 'content'
            "logprobs": {"content": logprobs_content} if logprobs_content is not None else None,
            "finish_reason": reason,
        }
        add_moe_routing_to_choice(choice_data, result, prompt_len)
        choices.append(choice_data)

        if verbose:
            logger.info(_redact_token_id_lists_for_logging(result))
        if not payload_offloaded and result.get("generated_log_probs") is None:
            logger.warning(
                "Generation log probs is None for request:\n%s",
                json.dumps(_redact_token_id_lists_for_logging(result), indent=4),
            )
        total_completion_tokens += len(result["generated_tokens"])

    usage = build_usage(prompt_tokens_counts, total_completion_tokens, cached_tokens_counts)
    return build_response(response_uid, "chat.completion", choices, usage, response_metadata)


try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False


try:
    from quart import Blueprint, Response, current_app, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    def _streaming_response(
        client,
        tokenizer,
        parsers,
        options,
        prompt_tokens,
        sampling_params,
        n,
        *,
        multi_modal_data,
        offload_params,
    ):
        """SSE response streaming `n` choices of one prompt; 400 when cannot stream."""
        # Streaming currently supports only Hugging Face fast tokenizers.
        try:
            incremental_detokenizers = [
                HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_tokens) for _ in range(n)
            ]
        except ValueError as error:
            return Response(str(error), status=400)
        streams = [
            client.add_request_streaming(
                prompt_tokens,
                sampling_params,
                multi_modal_data=multi_modal_data,
                offload_params=offload_params,
            )
            for _ in range(n)
        ]
        response = Response(
            openai_stream(
                streams,
                tokenizer,
                incremental_detokenizers,
                chat=True,
                return_log_probs=sampling_params.return_log_probs,
                include_usage=options.include_usage,
                chat_parsers=_streaming_chat_parsers(parsers, options, n),
            ),
            content_type="text/event-stream",
        )
        response.timeout = None
        return response

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']
        prompt_config = current_app.config['multimodal_prompt_config']

        req = await request.get_json()
        offload_params = req.get("offload_params")
        offload_params_error = validate_offload_params(offload_params)
        if offload_params_error is not None:
            return Response(offload_params_error, status=400)
        options = _ChatRequestOptions.from_request(req, current_app.config)

        # --- 1. Parse Messages ---
        messages = req.get("messages")
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)
        # Extract structured media before template sanitization. Remote image fetches block, so
        # keep this work off the event loop; remote responses bypass Quart's request-body limit,
        # so apply the same bound to them.
        try:
            messages, image_bytes_list, video_bytes_list, media_slots = await asyncio.to_thread(
                _extract_multimodal_from_messages,
                messages,
                prompt_config,
                current_app.config.get("MAX_CONTENT_LENGTH"),
            )
        except ValueError as error:
            return Response(str(error), status=400)
        multi_modal_data = None
        if image_bytes_list:
            multi_modal_data = {"image": image_bytes_list}
        elif video_bytes_list:
            multi_modal_data = {"video": video_bytes_list}
        template_messages = _sanitize_messages_for_template(messages, media_slots, prompt_config)
        template_tools = _sanitize_tools_for_template(options.tools)

        try:
            prefix_source = _resolve_prefix_source(
                template_messages, options.prevent_retokenization, offload_params
            )
        except ValueError as error:
            return Response(str(error), status=400)

        # Capability checks read the shared tokenizer; rendering runs on the executor's private
        # copy, since HF tokenizers are not thread-safe. Callers that build the app config directly
        # register no copy, so the shared tokenizer is the fallback.
        chat_tok = _chat_tokenizer(tokenizer)
        renderer = _TemplateRenderer(
            current_app.config.get('tokenize_executor'),
            _chat_tokenizer(current_app.config.get('tokenizer_copy', tokenizer)),
            prompt_config,
            template_tools,
            options.chat_template_kwargs,
        )
        try:
            if _has_chat_template(chat_tok, options.chat_template_kwargs):
                prompt_tokens, offload_params = await _render_prompt(
                    renderer,
                    tokenizer,
                    template_messages,
                    media_slots,
                    prefix_source,
                    offload_params,
                )
            else:
                prompt_tokens = _tokenize_without_template(tokenizer, messages, media_slots)
        except ValueError as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Invalid 'messages': {e}", status=400)
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

        # --- 2. Parse Sampling Params ---
        try:
            sampling_params, extras = parse_sampling_params(
                req,
                current_app.config,
                tokenizer,
                completions_mode=False,
                return_prompt_tokens=options.return_tokenized_data or options.return_raw_text,
            )
            n = extras["n"]  # Number of choices to generate
            prompt_tokens = _apply_bos_policy(
                prompt_tokens, tokenizer, sampling_params.add_BOS, offload_params
            )
        except (ValueError, TypeError) as e:
            return Response(f"Invalid sampling parameter: {e}", status=400)

        # --- 3. Send Requests to Engine ---
        # Hash and serialize shared media once before fanning one prompt out to n independently
        # sampled choices; coordinator affinity keeps equivalent requests on the engine that owns
        # the cached vision embedding.
        prepared_multimodal_data = prepare_multimodal_data(multi_modal_data)
        if options.stream:
            return _streaming_response(
                client,
                tokenizer,
                parsers,
                options,
                prompt_tokens,
                sampling_params,
                n,
                multi_modal_data=prepared_multimodal_data,
                offload_params=offload_params,
            )

        # add_request_with_id, not add_request: a non-streaming response writes
        # nothing to the socket while generating, so a disconnect is never
        # discovered as a broken pipe. Aborting needs the request ids.
        try:
            request_ids, tasks = submit_requests(
                client,
                ((prompt_tokens, sampling_params) for _ in range(n)),
                multi_modal_data=prepared_multimodal_data,
                offload_params=offload_params,
            )
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            return Response(f"Error submitting request: {e}", status=500)

        try:
            batch_results = await run_inference(
                client, request_ids, tasks, current_app.config['verbose'], log_label=f"(n={n})"
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return Response(f"Error during inference: {e}", status=500)

        # --- 4. Check for failed requests ---
        failure = failed_requests_response(batch_results, len(prompt_tokens))
        if failure is not None:
            body, status = failure
            return Response(body, status=status)

        # --- 5. Format OpenAI Response ---
        response = format_chat_response(
            batch_results,
            sampling_params,
            tokenizer,
            options,
            parsers=parsers,
            verbose=current_app.config['verbose'],
        )

        if HAVE_ORJSON:
            # Use orjson for faster serialization
            return Response(orjson.dumps(response), mimetype="application/json")
        else:
            return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
