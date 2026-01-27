# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from abc import abstractmethod
from itertools import zip_longest
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, ValidationError

from ..__init__ import GenericGenerationArgs
from ..inference.api import (
    GroupedInferenceResponse,
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
)


class InferenceInterface(BaseModel):
    """Inference interface for chat models."""

    class Config:
        arbitrary_types_allowed = True

    def prepare_request(
        self, prompts: list[str | list[LLMChatMessage]], generation_args: GenericGenerationArgs
    ) -> InferenceRequest:
        prompt = [
            [LLMChatMessage(role='user', content=p)] if isinstance(p, str) else p for p in prompts
        ]
        return InferenceRequest(prompt=prompt, generation_args=generation_args)

    async def base_generate(self, request: InferenceRequest) -> list[InferenceResponse]:
        assert NotImplementedError("Direct Inference Classes must implement the base_generate method.")

    async def agenerate(
        self, request: InferenceRequest
    ) -> list[InferenceResponse] | list[GroupedInferenceResponse]:
        return await self.base_generate(request)

    def generate(
        self, request: InferenceRequest
    ) -> list[InferenceResponse] | list[GroupedInferenceResponse]:
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
