# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pydantic import BaseModel

from ..__init__ import Request


class LLMChatMessage(BaseModel):
    role: str
    content: str


class InferenceRequest(Request):
    prompt: list[str]
    n: int | None = None


class ChatInferenceRequest(InferenceRequest):
    prompt: list[list[LLMChatMessage]]
    tools: list[dict] | None = None


class GroupedInferenceRequest(InferenceRequest):
    group_size: int = 1


class InferenceResponse(BaseModel):
    """The minimum required response for an inference interface."""

    response: str
    raw_text: str | None = None
    token_ids: list[int] | None = None
    prompt_length: int | None = None
    logprobs: list[float] | None = None


class GroupedInferenceResponse(BaseModel):
    """An inference response which includes a list of responses."""

    responses: list[InferenceResponse]


class ChatInferenceResponse(InferenceResponse):
    """The minimum required response for a chat inference interface."""

    response: LLMChatMessage


class GroupedChatInferenceResponse(GroupedInferenceResponse):
    """A chat inference response which includes a list of responses."""

    responses: list[ChatInferenceResponse]
