# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared OpenAI-compatible streaming response formatting."""

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Iterable

from megatron.core.inference.inference_request import unwrap_serialized_tensors

logger = logging.getLogger(__name__)

# JSON cannot represent non-finite floats: `orjson.dumps` encodes them as null and the
# std-json fallback emits `-Infinity` literals that strict parsers reject.
# Use the standard floor of -9999.0, which other inference backends also use, e.g. vLLM:
# https://github.com/vllm-project/vllm/blob/85c1365bd971/vllm/entrypoints/openai/completion/serving.py#L703
# https://github.com/vllm-project/vllm/blob/85c1365bd971/vllm/entrypoints/generate/base/serving.py#L376-L388
JSON_SAFE_LOGPROB_FLOOR = -9999.0


def json_safe_logprob(logprob: float) -> float:
    """Clamp one logprob to a finite, JSON-representable value."""
    return logprob if math.isfinite(logprob) else JSON_SAFE_LOGPROB_FLOOR


def json_safe_logprobs(logprobs: Iterable[float]) -> list[float]:
    """Clamp a sequence of logprobs elementwise."""
    return [json_safe_logprob(logprob) for logprob in logprobs]


def json_safe_top_n_logprobs(
    top_n_logprobs: Iterable[dict[str, float] | None],
) -> list[dict[str, float] | None]:
    """Clamp the values of per-position `{token: logprob}` dicts."""
    return [
        (
            {token: json_safe_logprob(logprob) for token, logprob in entry.items()}
            if isinstance(entry, dict)
            else entry
        )
        for entry in top_n_logprobs
    ]


def _top_logprob_entries(top_logprobs):
    if not isinstance(top_logprobs, dict):
        return []
    return [
        {
            "token": str(token),
            "logprob": json_safe_logprob(logprob),
            "bytes": list(str(token).encode("utf-8")),
        }
        for token, logprob in top_logprobs.items()
    ]


def _token_logprobs(tokenizer, token_ids, log_probs, top_log_probs, chat, start_offset):
    entries = []
    offsets = []
    offset = start_offset
    for i, token_id in enumerate(token_ids):
        token = tokenizer.detokenize([token_id])
        token_top_logprobs = (
            top_log_probs[i] if top_log_probs is not None and i < len(top_log_probs) else None
        )
        entries.append(
            {
                "token": token,
                "logprob": json_safe_logprob(log_probs[i]) if i < len(log_probs) else None,
                "bytes": list(token.encode("utf-8")),
                "top_logprobs": _top_logprob_entries(token_top_logprobs),
            }
        )
        offsets.append(offset)
        offset += len(token)
    if chat:
        return {"content": entries}
    return {
        "tokens": [entry["token"] for entry in entries],
        "token_logprobs": [entry["logprob"] for entry in entries],
        "top_logprobs": json_safe_top_n_logprobs(
            top_log_probs[i] if top_log_probs is not None and i < len(top_log_probs) else None
            for i in range(len(entries))
        ),
        "text_offset": offsets,
    }


def _prompt_logprobs(tokenizer, token_ids, log_probs, top_log_probs):
    token_ids = list(token_ids or [])
    tokens = [tokenizer.detokenize([token_id]) for token_id in token_ids]
    token_logprobs = [None] + json_safe_logprobs(log_probs or [])
    token_logprobs = token_logprobs[: len(tokens)]
    token_logprobs.extend([None] * (len(tokens) - len(token_logprobs)))
    prompt_top_logprobs = [None] + json_safe_top_n_logprobs(top_log_probs or [])
    prompt_top_logprobs = prompt_top_logprobs[: len(tokens)]
    prompt_top_logprobs.extend([None] * (len(tokens) - len(prompt_top_logprobs)))

    offsets = []
    offset = 0
    for token in tokens:
        offsets.append(offset)
        offset += len(token)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": prompt_top_logprobs,
        "text_offset": offsets,
    }


def _finish_reason(result):
    requested = (result.get("sampling_params") or {}).get("num_tokens_to_generate")
    generated = len(result.get("generated_tokens") or [])
    return "length" if requested is not None and generated >= requested else "stop"


def _safe_content_prefix(content, markers):
    """Hold text that may still become a parser control marker."""
    safe_end = len(content)
    for marker in markers:
        marker_index = content.find(marker)
        if marker_index >= 0:
            safe_end = min(safe_end, marker_index)
        for prefix_length in range(1, min(len(marker), len(content)) + 1):
            if content.endswith(marker[:prefix_length]):
                safe_end = min(safe_end, len(content) - prefix_length)
    return content[:safe_end]


class StreamingChatParser:
    """Convert accumulated parser output into stable OpenAI chat deltas."""

    def __init__(self, parse, *, marker_prefixes=(), named_tool_choice=False):
        self._parse = parse
        self._marker_prefixes = tuple(marker_prefixes)
        self._named_tool_choice = named_tool_choice
        # Retain emitted content/reasoning prefixes and stable per-tool IDs and emission flags.
        self._content_sent = ""
        self._reasoning_sent = ""
        self._tool_arguments_sent = []
        self._tool_ids = []
        self._tool_names_sent = []
        # Track whether a tool delta was emitted so the final finish reason can reflect it.
        self.tools_streamed = False

    def _append_state_for_tool(self):
        self._tool_arguments_sent.append(False)
        self._tool_ids.append(f"call_{uuid.uuid4().hex[:24]}")
        self._tool_names_sent.append(False)

    @staticmethod
    def _function(call):
        function = call.get("function", {}) if isinstance(call, dict) else {}
        return function if isinstance(function, dict) else {}

    def parse(self, text, *, finished=False):
        """Parse accumulated text and return zero or more structured deltas."""
        try:
            content, metadata = self._parse(text)
        except Exception:
            logger.exception("Failed to parse a streaming chat delta.")
            return []

        content = content or ""
        metadata = metadata or {}
        tool_calls = metadata.get("tool_calls") or []
        reasoning = metadata.get("reasoning") or ""
        deltas = []

        if reasoning.startswith(self._reasoning_sent):
            reasoning_delta = reasoning[len(self._reasoning_sent) :]
            if reasoning_delta:
                deltas.append({"reasoning_content": reasoning_delta})
                self._reasoning_sent = reasoning

        # Buffer partial control-marker prefixes so parser syntax is not emitted as content.
        safe_content = (
            content
            if finished or tool_calls
            else _safe_content_prefix(content, self._marker_prefixes)
        )
        if safe_content.startswith(self._content_sent):
            content_delta = safe_content[len(self._content_sent) :]
            if content_delta:
                deltas.append({"content": content_delta})
                self._content_sent = safe_content

        for index, call in enumerate(tool_calls):
            while len(self._tool_ids) <= index:
                self._append_state_for_tool()

            function = self._function(call)
            function_name = function.get("name")
            current_arguments = function.get("arguments", "")
            if not isinstance(current_arguments, str):
                current_arguments = json.dumps(current_arguments, ensure_ascii=False)

            if function_name and not self._tool_names_sent[index]:
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": self._tool_ids[index],
                                "type": "function",
                                "function": {"name": str(function_name)},
                            }
                        ]
                    }
                )
                self._tool_names_sent[index] = True
                self.tools_streamed = True

            if self._tool_names_sent[index]:
                call_is_complete = finished or index + 1 < len(tool_calls)
                # Re-serialized arguments are not append-stable while XML parameters arrive.
                # Emit the complete JSON object once instead of streaming corrupt prefixes.
                if call_is_complete and not self._tool_arguments_sent[index]:
                    deltas.append(
                        {
                            "tool_calls": [
                                {"index": index, "function": {"arguments": current_arguments}}
                            ]
                        }
                    )
                    self._tool_arguments_sent[index] = True

        return deltas

    def finish_reason(self, result):
        """Return the OpenAI finish reason for this parsed choice."""
        if self.tools_streamed and not self._named_tool_choice:
            return "tool_calls"
        return _finish_reason(result)


def _status_name(record):
    status = record.get("status") if isinstance(record, dict) else None
    return getattr(status, "name", str(status)).upper()


def _failure_message(record):
    events = record.get("events") or []
    error_events = [
        event for event in events if event.get("type") in ("ERROR_NONTRANSIENT", "ERROR_TRANSIENT")
    ]
    if error_events:
        return str(error_events[-1].get("payload", "Unknown error"))
    return "Unknown inference error"


async def openai_stream(
    streams,
    tokenizer,
    incremental_detokenizers,
    *,
    chat,
    return_log_probs=False,
    include_usage=False,
    chat_parsers=None,
    echo_prompts=None,
    prompt_token_ids=None,
):
    """Yield SSE records for one or more inference streams."""
    if len(streams) != len(incremental_detokenizers):
        raise ValueError("Each inference stream must have an incremental detokenizer.")
    if chat_parsers is not None and len(streams) != len(chat_parsers):
        raise ValueError("Each inference stream must have a streaming chat parser.")
    if echo_prompts is not None and len(streams) != len(echo_prompts):
        raise ValueError("Each inference stream must have an echo prompt.")
    if prompt_token_ids is not None and len(streams) != len(prompt_token_ids):
        raise ValueError("Each inference stream must have prompt token IDs.")

    response_id = f"chatcmpl-{uuid.uuid4().hex}" if chat else str(uuid.uuid4())
    created = int(time.time())
    queue = asyncio.Queue()
    states = [
        dict(
            tokens=[],
            log_probs=[],
            top_log_probs=[],
            detokenizer=detokenizer,
            final=None,
            role_sent=False,
            echo_sent=False,
            parser=chat_parsers[index] if chat_parsers is not None else None,
            echo_prompt=echo_prompts[index] if echo_prompts is not None else None,
            prompt_token_ids=prompt_token_ids[index] if prompt_token_ids is not None else [],
        )
        for index, detokenizer in enumerate(incremental_detokenizers)
    ]

    async def pump(index, stream):
        try:
            async for item in stream:
                await queue.put((index, item, None))
        except Exception as exc:  # Propagate listener failures through the SSE response.
            await queue.put((index, None, exc))
        finally:
            await queue.put((index, None, None))

    tasks = [asyncio.create_task(pump(index, stream)) for index, stream in enumerate(streams)]

    def sse(choices, usage=None):
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": created,
            "model": "EMPTY",
            "choices": choices,
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"

    try:
        remaining = len(streams)
        while remaining:
            index, item, error = await queue.get()
            if error is not None:
                yield f"data: {json.dumps({'error': {'message': str(error)}})}\n\n"
                return
            if item is None:
                remaining -= 1
                continue

            state = states[index]
            is_final = "final" in item
            if "partial" in item:
                partial = item["partial"]
                new_tokens = partial.get("new_tokens") or []
                new_log_probs = partial.get("new_log_probs") or []
                new_top_log_probs = partial.get("new_top_n_logprobs") or []
                prompt_log_probs = partial.get("prompt_log_probs") or []
                prompt_top_log_probs = partial.get("prompt_top_n_logprobs") or []
            else:
                final_record = item["final"]
                if _status_name(final_record) == "FAILED":
                    error_payload = {"error": {"message": _failure_message(final_record)}}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    return
                result = unwrap_serialized_tensors(final_record)
                state["final"] = result
                already = len(state["tokens"])
                new_tokens = (result.get("generated_tokens") or [])[already:]
                new_log_probs = (result.get("generated_log_probs") or [])[already:]
                new_top_log_probs = (result.get("generated_top_n_logprobs") or [])[already:]
                prompt_log_probs = result.get("prompt_log_probs") or []
                prompt_top_log_probs = result.get("prompt_top_n_logprobs") or []

            if chat and not state["role_sent"]:
                yield sse(
                    [
                        {
                            "index": index,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ]
                )
                state["role_sent"] = True

            if state["echo_prompt"] is not None and not state["echo_sent"]:
                yield sse(
                    [
                        {
                            "index": index,
                            "text": state["echo_prompt"],
                            "logprobs": (
                                _prompt_logprobs(
                                    tokenizer,
                                    state["prompt_token_ids"],
                                    prompt_log_probs,
                                    prompt_top_log_probs,
                                )
                                if return_log_probs
                                else None
                            ),
                            "finish_reason": None,
                        }
                    ]
                )
                state["echo_sent"] = True

            if new_tokens:
                state["tokens"].extend(new_tokens)
                state["log_probs"].extend(new_log_probs)
                state["top_log_probs"].extend(new_top_log_probs)
                start_offset = state["detokenizer"].text_length + len(state["echo_prompt"] or "")
                delta = state["detokenizer"].update(new_tokens)
                logprobs = (
                    _token_logprobs(
                        tokenizer, new_tokens, new_log_probs, new_top_log_probs, chat, start_offset
                    )
                    if return_log_probs
                    else None
                )
            else:
                delta = ""
                logprobs = None

            parser = state["parser"]
            if parser is not None:
                parsed_deltas = parser.parse(state["detokenizer"].text, finished=is_final)
                logprobs_attached = False
                for parsed_delta in parsed_deltas:
                    parsed_logprobs = None
                    if (
                        not logprobs_attached
                        and logprobs is not None
                        and (
                            parsed_delta.get("content") is not None
                            or parsed_delta.get("reasoning_content") is not None
                        )
                    ):
                        parsed_logprobs = logprobs
                        logprobs_attached = True
                    yield sse(
                        [
                            {
                                "index": index,
                                "delta": parsed_delta,
                                "logprobs": parsed_logprobs,
                                "finish_reason": None,
                            }
                        ]
                    )
            elif new_tokens:
                choice = {"index": index, "logprobs": logprobs, "finish_reason": None}
                choice["delta" if chat else "text"] = {"content": delta} if chat else delta
                yield sse([choice])

        prompt_tokens = completion_tokens = cached_token_count = 0
        for index, state in enumerate(states):
            result = state["final"] or {}
            prompt_len = result.get("prompt_length")
            if prompt_len is None:
                prompt_len = len(result.get("prompt_tokens") or [])
            prompt_tokens = max(prompt_tokens, prompt_len)
            completion_tokens += len(result.get("generated_tokens") or [])
            cached_token_count = max(cached_token_count, result.get("num_cached_tokens", 0))
            parser = state["parser"]
            choice = {
                "index": index,
                "logprobs": None,
                "finish_reason": (
                    parser.finish_reason(result) if parser is not None else _finish_reason(result)
                ),
                "generation_token_ids": list(state["tokens"]),
                "generation_log_probs": json_safe_logprobs(state["log_probs"]),
                "generated_text": state["detokenizer"].text,
                "generated_length": len(state["tokens"]),
            }
            choice["delta" if chat else "text"] = {} if chat else ""
            yield sse([choice])

        if include_usage:
            yield sse(
                [],
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_token_count},
                },
            )
        yield "data: [DONE]\n\n"
    finally:
        for task in tasks:
            task.cancel()
        for stream in streams:
            await stream.aclose()
