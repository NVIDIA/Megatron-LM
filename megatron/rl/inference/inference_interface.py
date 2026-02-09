# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from abc import abstractmethod
from itertools import zip_longest
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, ValidationError

from ..__init__ import GenericGenerationArgs
from ..inference.api import (
    ChatInferenceRequest,
    ChatInferenceResponse,
    GroupedChatInferenceResponse,
    GroupedInferenceResponse,
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
)
from ..inference.chat_templates import ConversationTemplate


# Used when generating n resposnes for a single prompt
def grouper(iterable, n, fillvalue=None):
    """Fold an iterable into a list of lists of size n."""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class InferenceInterface(BaseModel):
    """Inference interface that for base language models."""

    class Config:
        arbitrary_types_allowed = True

    supports_n: ClassVar[bool] = False

    def prepare_request(
        self, prompts: list[str], generation_args: GenericGenerationArgs
    ) -> InferenceRequest:
        assert all(isinstance(p, str) for p in prompts), "Prompt must be a list of strings"
        return InferenceRequest(prompt=prompts, generation_args=generation_args)

    @abstractmethod
    async def base_generate(self, request: InferenceRequest) -> list[InferenceResponse]:
        raise NotImplementedError(
            "Direct Inference Classes must implement the base_generate method."
        )

    def duplicate_requests(self, request: InferenceRequest, n: int) -> list[InferenceRequest]:
        return request.model_copy(update={'prompt': request.prompt * n})

    def fold_responses(
        self, responses: list[InferenceResponse], n: int
    ) -> list[GroupedInferenceResponse]:
        return [GroupedInferenceResponse(responses=x) for x in list(grouper(responses, n))]

    async def agenerate(
        self, request: InferenceRequest
    ) -> list[InferenceResponse] | list[GroupedInferenceResponse]:
        if not self.supports_n and request.n is not None:
            request = self.duplicate_requests(request, request.n)

        generations = await self.base_generate(request)

        if request.n is not None:
            if self.supports_n:
                assert (
                    len(generations) == len(request.prompt) * request.n
                ), f"Number of generations ({len(generations)}) does not match number of prompts ({len(request.prompt)} * {request.n})."
            else:
                assert len(generations) == len(
                    request.prompt
                ), f"Number of generations ({len(generations)}) does not match number of prompts ({len(request.prompt)})."
            generations = self.fold_responses(generations, request.n)

        return generations

    def generate(
        self, request: InferenceRequest
    ) -> list[InferenceResponse] | list[GroupedInferenceResponse]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate(request))
        else:
            return loop.run_until_complete(self.agenerate(request))


def ensure_template(value: Any) -> ConversationTemplate:
    if isinstance(value, ConversationTemplate):
        return value
    elif isinstance(value, str):
        return ConversationTemplate.from_string(value)
    else:
        raise ValidationError(f"Invalid conversation template: {value}")


class ChatInferenceInterface(InferenceInterface):
    """Inference interface for chat models."""

    conversation_template: Annotated[ConversationTemplate, BeforeValidator(ensure_template)]

    def prepare_request(
        self, prompts: list[str | list[LLMChatMessage]], generation_args: GenericGenerationArgs
    ) -> ChatInferenceRequest:
        prompt = [
            [LLMChatMessage(role='user', content=p)] if isinstance(p, str) else p for p in prompts
        ]
        return ChatInferenceRequest(prompt=prompt, generation_args=generation_args)

    async def base_generate(self, request: ChatInferenceRequest) -> list[ChatInferenceResponse]:
        base_generate_results = await super().base_generate(
            InferenceRequest(
                prompt=[self.conversation_template.format(messages) for messages in request.prompt],
                generation_args=request.generation_args,
            )
        )
        chat_message_results = self.conversation_template.parse_response(base_generate_results)
        return [
            ChatInferenceResponse(
                response=chat_message, **response.model_dump(exclude={'response'})
            )
            for chat_message, response in zip(chat_message_results, base_generate_results)
        ]

    def generate(
        self, request: ChatInferenceRequest
    ) -> list[ChatInferenceResponse] | list[GroupedChatInferenceResponse]:
        return super().generate(request)

    async def agenerate(
        self, request: ChatInferenceRequest
    ) -> list[ChatInferenceResponse] | list[GroupedChatInferenceResponse]:
        return await super().agenerate(request)


class ReturnsRaw(InferenceInterface):
    """Mix-In for interface that supports returning complete string fed to the LLM."""

    # TODO: Should this be a mix-in or a class variable?


class ReturnsTokens(InferenceInterface):
    """Mix-In for interface that supports returning the complete list of tokens fed to the LLM."""

    # TODO: Should this be a mix-in or a class variable?


class ReturnsLogProbs(ReturnsTokens):
    """Mix-In for interface that supports returning the logprobs for a set of tokens."""

    # TODO: Should this be a mix-in or a class variable?
