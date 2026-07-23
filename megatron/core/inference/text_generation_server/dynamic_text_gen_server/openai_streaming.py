# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared OpenAI-compatible streaming response formatting."""

import asyncio
import json
import time
import uuid

from megatron.core.inference.inference_request import unwrap_serialized_tensors


def _token_logprobs(tokenizer, token_ids, log_probs, chat, start_offset):
    entries = []
    offsets = []
    offset = start_offset
    for i, token_id in enumerate(token_ids):
        token = tokenizer.detokenize([token_id])
        entries.append(
            {
                "token": token,
                "logprob": log_probs[i] if i < len(log_probs) else None,
                "bytes": list(token.encode("utf-8")),
                "top_logprobs": [],
            }
        )
        offsets.append(offset)
        offset += len(token)
    if chat:
        return {"content": entries}
    return {
        "tokens": [entry["token"] for entry in entries],
        "token_logprobs": [entry["logprob"] for entry in entries],
        "top_logprobs": [None] * len(entries),
        "text_offset": offsets,
    }


def _finish_reason(result):
    requested = (result.get("sampling_params") or {}).get("num_tokens_to_generate")
    generated = len(result.get("generated_tokens") or [])
    return "length" if requested is not None and generated >= requested else "stop"


async def openai_stream(streams, tokenizer, *, chat, return_log_probs=False, include_usage=False):
    """Yield SSE records for one or more inference streams."""
    response_id = f"chatcmpl-{uuid.uuid4().hex}" if chat else str(uuid.uuid4())
    created = int(time.time())
    queue = asyncio.Queue()
    states = [dict(tokens=[], log_probs=[], text="", final=None) for _ in streams]

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
        if chat:
            for index in range(len(streams)):
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

        remaining = len(streams)
        while remaining:
            index, item, error = await queue.get()
            if error is not None:
                yield f"data: {json.dumps({'error': {'message': str(error)}})}\n\n"
                continue
            if item is None:
                remaining -= 1
                continue

            state = states[index]
            if "partial" in item:
                partial = item["partial"]
                new_tokens = partial.get("new_tokens") or []
                new_log_probs = partial.get("new_log_probs") or []
            else:
                result = unwrap_serialized_tensors(item["final"])
                state["final"] = result
                already = len(state["tokens"])
                new_tokens = (result.get("generated_tokens") or [])[already:]
                new_log_probs = (result.get("generated_log_probs") or [])[already:]

            if not new_tokens:
                continue
            state["tokens"].extend(new_tokens)
            state["log_probs"].extend(new_log_probs)
            start_offset = len(state["text"])
            full_text = tokenizer.detokenize(state["tokens"])
            delta = (
                full_text[len(state["text"]) :]
                if full_text.startswith(state["text"])
                else tokenizer.detokenize(new_tokens)
            )
            state["text"] = full_text
            choice = {
                "index": index,
                "logprobs": (
                    _token_logprobs(tokenizer, new_tokens, new_log_probs, chat, start_offset)
                    if return_log_probs
                    else None
                ),
                "finish_reason": None,
                "generation_token_ids": list(state["tokens"]),
                "generation_log_probs": list(state["log_probs"]),
                "generated_text": full_text,
                "generated_length": len(state["tokens"]),
            }
            choice["delta" if chat else "text"] = {"content": delta} if chat else delta
            yield sse([choice])

        prompt_tokens = completion_tokens = cached_token_count = 0
        for index, state in enumerate(states):
            result = state["final"] or {}
            prompt_tokens = max(prompt_tokens, len(result.get("prompt_tokens") or []))
            completion_tokens += len(result.get("generated_tokens") or [])
            cached_token_count = max(
                cached_token_count, result.get("num_cached_tokens", 0)
            )
            choice = {
                "index": index,
                "logprobs": None,
                "finish_reason": _finish_reason(result),
                "generation_token_ids": list(state["tokens"]),
                "generation_log_probs": list(state["log_probs"]),
                "generated_text": state["text"],
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
