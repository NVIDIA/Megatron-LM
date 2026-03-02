# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pydantic import BaseModel

from ..__init__ import Request


class LLMChatMessage(BaseModel):
    role: str
    content: str


class InferenceRequest(Request):
    prompt: list[LLMChatMessage]
    tools: list[dict] | None = None


class InferenceResponse(BaseModel):
    """The minimum required response for an inference interface."""

    response: LLMChatMessage
    raw_text: str | None = None
    token_ids: list[int] | None = None
    prompt_length: int | None = None
    logprobs: list[float] | None = None
    policy_staleness: list[int]
    kv_cache_staleness: list[int]
    completed_at_step: int
    num_evictions: int
