# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

from pydantic import BaseModel

from ..__init__ import GenericGenerationArgs
from ..inference.api import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
)


class InferenceInterface(BaseModel):
    """Inference interface for chat models."""

    class Config:
        arbitrary_types_allowed = True

    def prepare_request(
        self, prompt: str | list[LLMChatMessage], generation_args: GenericGenerationArgs
    ) -> InferenceRequest:
        prompt = [LLMChatMessage(role='user', content=prompt)] if isinstance(prompt, str) else prompt
        return InferenceRequest(prompt=prompt, generation_args=generation_args)

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        assert NotImplementedError("Direct Inference Classes must implement the base_generate method.")

    async def agenerate(
        self, request: InferenceRequest
    ) -> InferenceResponse:
        return await self.base_generate(request)

    def generate(
        self, request: InferenceRequest
    ) -> InferenceResponse:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate(request))
        else:
            return loop.run_until_complete(self.agenerate(request))

class ReturnsRaw(InferenceInterface):
    """Mix-In for interface that supports returning complete string fed to the LLM."""

    # TODO: Should this be a mix-in or a class variable?


class ReturnsTokens(InferenceInterface):
    """Mix-In for interface that supports returning the complete list of tokens fed to the LLM."""

    # TODO: Should this be a mix-in or a class variable?


class ReturnsLogProbs(ReturnsTokens):
    """Mix-In for interface that supports returning the logprobs for a set of tokens."""

    # TODO: Should this be a mix-in or a class variable?
