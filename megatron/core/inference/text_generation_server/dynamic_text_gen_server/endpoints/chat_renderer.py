# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chat rendering and reply building for the OpenAI-compatible chat endpoint.

`ChatRenderer` holds what `/v1/chat/completions` does before a prompt reaches the engine
(request options, media extraction, message/tool sanitization, chat template + tokenization,
BOS policy) and after the engine replies (detokenization, reasoning/tool parsers, the OpenAI
response body). It has no Quart, `current_app` or `InferenceClient` dependency, so frontends
that route token ids to engines themselves can reuse Megatron's exact chat contract.

Previous-turn prefix stitching is deliberately not part of the class: the endpoint keeps its own
splice, and other callers can splice with `RenderedChat.history_token_ids` and
`RenderedChat.eos_token_ids`.
"""

import asyncio
import base64
import ipaddress
import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Optional, Sequence

from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.inference_request import PREFIX_EXPANDED_TOKEN_COUNT_FIELD
from megatron.core.inference.utils import detokenize_tokens, model_eos_token_ids
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

from ..openai_streaming import json_safe_logprobs, json_safe_top_n_logprobs
from .common import (
    add_moe_routing_to_choice,
    build_response,
    build_usage,
    generation_finish_reason,
    parse_stream_options,
    reply_prompt_length,
    unwrap_batch,
)

logger = logging.getLogger(__name__)

_MEDIA_FETCH_TIMEOUT_S = 5.0
_MEDIA_FETCH_USER_AGENT = "megatron-inference"

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
        _tokenize_with_media_slots replaces this sentinel with the
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


def _last_assistant_message(template_messages):
    """Return ``(index, message)`` of the last assistant turn, or ``(None, None)``."""
    for i in reversed(range(len(template_messages))):
        if template_messages[i]["role"] == "assistant":
            return i, template_messages[i]
    return None, None


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


def _tokenize_with_media_slots(media_message_str: str, media_slots, chat_tok, prompt_config):
    """Tokenize and convert media text placeholders into model media token IDs."""
    positioned_slots = []
    for sentinel, modality, _message_index in media_slots:
        if media_message_str.count(sentinel) != 1:
            raise ValueError(f"Chat template did not preserve media slot {sentinel}.")
        positioned_slots.append((media_message_str.index(sentinel), sentinel, modality))

    prompt_tokens = []
    cursor = 0
    for position, sentinel, modality in sorted(positioned_slots):
        prompt_tokens.extend(
            _coerce_to_token_id_list(
                chat_tok(media_message_str[cursor:position], add_special_tokens=False)
            )
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
        _coerce_to_token_id_list(chat_tok(media_message_str[cursor:], add_special_tokens=False))
    )
    return prompt_tokens


def _tokenize_chat_template_sync(
    chat_tok,
    messages,
    media_slots,
    prompt_config,
    tools,
    chat_template_kwargs,
    add_generation_prompt=True,
):
    """Render a chat template and convert internal media slots to model media
    tokens defined in the MediaPromptSpec.

    Synchronous, and run whole on the frontend's tokenize executor rather than
    piecewise. Rendering is only the first of 3N+1 tokenizer calls for N media
    slots -- each slot encodes the text before it, then the model token's prefix
    and suffix -- and awaiting just the render left the rest on the event loop,
    where they stall every other request this replica owns.

    One hop to the executor also means one thread touches the tokenizer for the
    whole request. HF tokenizers are not thread-safe, and this runs on the
    executor's private copy.
    """
    # Tokenize the message with a chat template.
    rendered = chat_tok.apply_chat_template(
        messages,
        # Need to insert media placeholder tokens based on
        # position of the media sentinels in the messages
        # before tokenization.
        tokenize=False if media_slots else True,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **chat_template_kwargs,
    )

    if media_slots:
        if not isinstance(rendered, str):
            raise TypeError("Multimodal chat template rendering must return a string.")
        prompt_tokens = _tokenize_with_media_slots(rendered, media_slots, chat_tok, prompt_config)
    else:
        # Standardize to list[int].
        prompt_tokens = _coerce_to_token_id_list(rendered)

    if not isinstance(prompt_tokens, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in prompt_tokens
    ):
        raise TypeError(f"Tokenized output should be a list[int], got {prompt_tokens}.")
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


def _eos_token_ids(tokenizer):
    """Every EOS id the model may emit: the generation_config set plus the tokenizer's own."""
    eos_token_ids = set(model_eos_token_ids(tokenizer))
    if getattr(tokenizer, "eos_id", None) is not None:
        eos_token_ids.add(tokenizer.eos_id)
    assert eos_token_ids, "Your tokenizer must have an EOS token ID!"
    return eos_token_ids


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
class ChatRequestOptions:
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
        """Read the options off the request body; a malformed field raises ValueError."""
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
        include_usage = parse_stream_options(req)
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
        elif not normalized_tool_calls and not message_text and "reasoning" in metadata:
            # Match vLLM's reasoning-only response. Some agent clients distinguish
            # absent final content from an empty assistant message and otherwise
            # continue the episode after a reasoning-only completion.
            content = None
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
            message["generation_log_probs"] = result.get("generated_log_probs", [])
        if options.return_raw_text and not payload_offloaded:
            message["raw_text"] = tokenizer.detokenize(result["prompt_tokens"]) + text_output

        reason = generation_finish_reason(result)
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
        if (
            sampling_params.return_log_probs
            and not payload_offloaded
            and result.get("generated_log_probs") is None
        ):
            logger.warning(
                "Generation log probs is None for request:\n%s",
                json.dumps(_redact_token_id_lists_for_logging(result), indent=4),
            )
        total_completion_tokens += len(result["generated_tokens"])

    usage = build_usage(prompt_tokens_counts, total_completion_tokens, cached_tokens_counts)
    return build_response(response_uid, "chat.completion", choices, usage, response_metadata)


def _history_prefix(template_messages, media_slots, last_assistant_idx):
    """The messages up to and including the last assistant turn, and the media slots in them."""
    messages_to_last_assistant = template_messages[: last_assistant_idx + 1]
    previous_media_slots = [slot for slot in media_slots if slot[2] <= last_assistant_idx]
    return messages_to_last_assistant, previous_media_slots


@dataclass(frozen=True)
class RenderedChat:
    """Everything `ChatRenderer.render` derives from one chat request before submission.

    `prompt_token_ids` is the full conversation rendered with the generation prompt. It is
    *not* spliced with the exact tokens of a previous turn; callers that want that splice can
    find the previous turn's boundary with `history_token_ids` and `eos_token_ids`.
    """

    prompt_token_ids: list
    """Prompt token ids (media slots already lowered to model media tokens)."""

    history_token_ids: Optional[list]
    """Messages up to and including the last assistant message, rendered with
    `add_generation_prompt=False`; `None` when not requested, when there is no assistant
    message, or when the tokenizer has no chat template."""

    eos_token_ids: frozenset
    """Every EOS id the model may emit; the turn terminators prefix stitching counts."""

    options: ChatRequestOptions
    """The parsed non-sampling request options; pass them back to `build_response`."""

    messages: list
    """Request messages with structured media replaced by internal slot sentinels."""

    template_messages: list
    """`messages` after sanitization, as handed to the chat template."""

    template_tools: Optional[list]
    """Request tools after sanitization, as handed to the chat template."""

    media_slots: list
    """`(sentinel, modality, message_index)` per media block, in request order."""

    multi_modal_data: Optional[dict]
    """`{"image": [bytes, ...]}` or `{"video": [bytes, ...]}`; `None` for text-only requests."""


@dataclass
class ChatGeneration:
    """One generated choice, for building a response without a Megatron engine reply.

    `ChatRenderer.build_response` accepts either these or the engine's serialized request
    dicts; `to_result_dict` produces the subset of those dict keys the response reads.
    """

    generated_tokens: list
    """Generated token ids, including a trailing EOS if one was sampled."""

    finish_reason: Optional[str] = None
    """`"stop"` or `"length"`; `None` reports `"stop"`."""

    prompt_tokens: Optional[list] = None
    """Model-input prompt ids; echoed when the request asks for tokenized data or raw text."""

    prompt_length: Optional[int] = None
    """Prompt-token count for `usage`; defaults to `len(prompt_tokens)`."""

    generated_log_probs: Optional[list] = None
    """Per-token log probs of `generated_tokens`, read when `return_log_probs` is set."""

    generated_top_n_logprobs: Optional[list] = None
    """Per-token `{token_str: logprob}` maps, read when `return_log_probs` is set."""

    num_cached_tokens: int = 0
    """Prompt tokens served from the prefix cache, reported in `usage`."""

    routing_indices: Optional[list] = None
    """MoE routing indices for prompt + generation, reported when present."""

    uid: Any = None
    """Response id; the first choice's uid becomes the body's `id`."""

    def to_result_dict(self) -> dict:
        """The engine-reply dict shape that `format_chat_response` reads."""
        return asdict(self)


class ChatRenderer:
    """Megatron's chat-completion rendering and reply building, without the HTTP server.

    One instance per tokenizer/parser configuration; methods are reentrant except where they
    share `render_tokenizer` (Hugging Face tokenizers are not thread-safe, so give each thread
    its own renderer or tokenizer copy).

    Args:
        tokenizer: The Megatron (or Hugging Face) tokenizer used for capability checks,
            detokenization, EOS/BOS ids and the no-template fallback.
        parsers: Names into `PARSER_MAPPING` applied to generated text, in order.
        multimodal_prompt_config: How image/video blocks lower to prompt tokens; defaults to
            `MultimodalPromptConfig()`.
        render_tokenizer: Tokenizer that renders chat templates; defaults to `tokenizer`. The
            text-gen server passes a private copy owned by `executor`'s thread.
        executor: Executor that `tokenize_async` runs on; `None` uses the loop's default.
        chat_template: Server-configured chat template; requests cannot override it.
        eval_mode: Server default for a request that omits `prevent_retokenization`
            (`True` disables it).
        max_media_fetch_bytes: Bound on each remote media response; `None` is unbounded.
    """

    def __init__(
        self,
        tokenizer: Any,
        parsers: Optional[Sequence[str]] = None,
        multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
        *,
        render_tokenizer: Any = None,
        executor: Optional[Executor] = None,
        chat_template: Optional[str] = None,
        eval_mode: bool = False,
        max_media_fetch_bytes: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.parsers = parsers
        self.prompt_config = (
            multimodal_prompt_config
            if multimodal_prompt_config is not None
            else MultimodalPromptConfig()
        )
        self.render_tokenizer = _chat_tokenizer(
            render_tokenizer if render_tokenizer is not None else tokenizer
        )
        self.executor = executor
        self.chat_template = chat_template
        self.eval_mode = eval_mode
        self.max_media_fetch_bytes = max_media_fetch_bytes

    # --- request -> prompt ------------------------------------------------------------------

    def parse_options(self, request: dict) -> ChatRequestOptions:
        """Read the non-sampling options off a request body; a malformed field raises ValueError."""
        return ChatRequestOptions.from_request(
            request, {"chat_template": self.chat_template, "eval_mode": self.eval_mode}
        )

    def extract_media(self, messages: list) -> tuple:
        """Replace image/video blocks with slot sentinels and collect their bytes.

        Blocking (remote image URLs are fetched); async callers should run it off the event
        loop. A media block that cannot be loaded raises ValueError.

        Returns:
            `(messages, multi_modal_data, media_slots)`, see `RenderedChat`.
        """
        messages, image_bytes_list, video_bytes_list, media_slots = (
            _extract_multimodal_from_messages(
                messages, self.prompt_config, self.max_media_fetch_bytes
            )
        )
        multi_modal_data = None
        if image_bytes_list:
            multi_modal_data = {"image": image_bytes_list}
        elif video_bytes_list:
            multi_modal_data = {"video": video_bytes_list}
        return messages, multi_modal_data, media_slots

    def sanitize(self, messages: list, media_slots: list, tools: Any) -> tuple:
        """Make messages and tools safe for the chat template.

        Returns:
            `(template_messages, template_tools)`.
        """
        template_messages = _sanitize_messages_for_template(
            messages, media_slots, self.prompt_config
        )
        return template_messages, _sanitize_tools_for_template(tools)

    def has_chat_template(self, options: ChatRequestOptions) -> bool:
        """True when a chat template can be rendered: the tokenizer's own or the server's."""
        return _has_chat_template(_chat_tokenizer(self.tokenizer), options.chat_template_kwargs)

    def _render_call(
        self, messages, media_slots, tools, chat_template_kwargs, add_generation_prompt
    ):
        """Bind one synchronous template render, so sync and executor paths share it."""
        return partial(
            _tokenize_chat_template_sync,
            self.render_tokenizer,
            messages,
            media_slots,
            self.prompt_config,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs if chat_template_kwargs is not None else {},
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize(
        self,
        messages: list,
        media_slots: Sequence = (),
        *,
        tools: Optional[list] = None,
        chat_template_kwargs: Optional[dict] = None,
        add_generation_prompt: bool = True,
    ) -> list:
        """Render the chat template and tokenize it, lowering media slots to model tokens."""
        return self._render_call(
            messages, media_slots, tools, chat_template_kwargs, add_generation_prompt
        )()

    async def tokenize_async(
        self,
        messages: list,
        media_slots: Sequence = (),
        *,
        tools: Optional[list] = None,
        chat_template_kwargs: Optional[dict] = None,
        add_generation_prompt: bool = True,
    ) -> list:
        """`tokenize` on `executor`, in one hop so one thread touches the tokenizer per call."""
        render = self._render_call(
            messages, media_slots, tools, chat_template_kwargs, add_generation_prompt
        )
        return await asyncio.get_running_loop().run_in_executor(self.executor, render)

    def tokenize_without_template(self, messages: list, media_slots: Sequence = ()) -> list:
        """Fallback for tokenizers without a chat template: newline-joined message contents."""
        return _tokenize_without_template(self.tokenizer, messages, media_slots)

    def apply_bos_policy(
        self, prompt_token_ids: list, add_bos: bool, offload_params: Optional[dict] = None
    ) -> list:
        """Collapse leading BOS tokens and keep one only when `add_bos`.

        Apply it after any previous-turn splice. `offload_params`, when given, has its
        expanded-prefix token count shifted by the change in length.
        """
        return _apply_bos_policy(prompt_token_ids, self.tokenizer, add_bos, offload_params)

    def eos_token_ids(self) -> frozenset:
        """Every EOS id the model may emit: the generation_config set plus the tokenizer's own."""
        return frozenset(_eos_token_ids(self.tokenizer))

    @staticmethod
    def history_prefix(template_messages: list, media_slots: Sequence) -> Optional[tuple]:
        """`(messages, media_slots)` up to and including the last assistant message, if any."""
        last_idx, _ = _last_assistant_message(template_messages)
        if last_idx is None:
            return None
        return _history_prefix(template_messages, media_slots, last_idx)

    def render(
        self, request: dict, *, with_history: bool = False, add_bos: Optional[bool] = None
    ) -> RenderedChat:
        """Render an OpenAI chat-completions request body to prompt token ids.

        Runs the text-gen server's pre-submission steps synchronously, except the
        previous-turn splice. Blocking (may fetch remote images and runs the tokenizer).

        Args:
            request: The request body (`messages`, `tools`, `chat_template_kwargs`, ...).
            with_history: Also render the conversation up to the last assistant message.
            add_bos: Apply the BOS policy to `prompt_token_ids` (the server reads it from the
                request's `add_BOS`, default False). `None` leaves the template's BOS as is,
                for callers that splice first and call `apply_bos_policy` afterwards.

        Raises:
            ValueError: The request is malformed (bad options, missing or non-list
                `messages`, unloadable media, a template that loses a media slot, ...).
        """
        options = self.parse_options(request)
        messages = request.get("messages")
        if not messages:
            raise ValueError("Missing 'messages' field")
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list")
        messages, multi_modal_data, media_slots = self.extract_media(messages)
        template_messages, template_tools = self.sanitize(messages, media_slots, options.tools)

        history_token_ids = None
        if self.has_chat_template(options):
            prompt_token_ids = self.tokenize(
                template_messages,
                media_slots,
                tools=template_tools,
                chat_template_kwargs=options.chat_template_kwargs,
                add_generation_prompt=True,
            )
            history = self.history_prefix(template_messages, media_slots) if with_history else None
            if history is not None:
                history_messages, history_media_slots = history
                history_token_ids = self.tokenize(
                    history_messages,
                    history_media_slots,
                    tools=template_tools,
                    chat_template_kwargs=options.chat_template_kwargs,
                    add_generation_prompt=False,
                )
        else:
            prompt_token_ids = self.tokenize_without_template(messages, media_slots)
        if add_bos is not None:
            prompt_token_ids = self.apply_bos_policy(prompt_token_ids, add_bos)

        return RenderedChat(
            prompt_token_ids=prompt_token_ids,
            history_token_ids=history_token_ids,
            eos_token_ids=self.eos_token_ids(),
            options=options,
            messages=messages,
            template_messages=template_messages,
            template_tools=template_tools,
            media_slots=media_slots,
            multi_modal_data=multi_modal_data,
        )

    # --- generation -> response -------------------------------------------------------------

    def build_response(
        self,
        results: Sequence,
        sampling_params: Any,
        options: ChatRequestOptions,
        *,
        verbose: bool = False,
    ) -> dict:
        """Build the OpenAI `chat.completion` body for the generated choices.

        Args:
            results: One entry per choice: `ChatGeneration`s or the engine's serialized
                request dicts.
            sampling_params: Read for `detokenize_stop_sequence` and `return_log_probs`
                (a `SamplingParams`).
            options: The request options from `parse_options` / `RenderedChat.options`.
            verbose: Log each (redacted) reply.
        """
        batch_results = [
            result.to_result_dict() if isinstance(result, ChatGeneration) else result
            for result in results
        ]
        return format_chat_response(
            batch_results,
            sampling_params,
            self.tokenizer,
            options,
            parsers=self.parsers,
            verbose=verbose,
        )
