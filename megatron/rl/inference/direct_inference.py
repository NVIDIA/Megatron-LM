# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import Annotated, Any, ClassVar

from pydantic import BeforeValidator, ValidationError


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


from .api import (
    ChatInferenceInterface,
    ChatInferenceRequest,
    ChatInferenceResponse,
    GroupedInferenceResponse,
    InferenceInterface,
    InferenceRequest,
    InferenceResponse,
)
from .chat_templates import ConversationTemplate


class DirectInferenceInterface(InferenceInterface, ABC):
    """Basic inference engines that operate directly on strings can extend this class and implement base_generate.

    This abstract base class then implements necessary LLM interfaces.
    """

    supports_n: ClassVar[bool] = False

    @abstractmethod
    async def base_generate(self, request: InferenceRequest) -> list[InferenceResponse]:
        raise NotImplementedError("Inference Classes must implement the base_generate method.")

    def duplicate_requests(self, request: InferenceRequest, n: int) -> list[InferenceRequest]:
        return request.model_copy(update={'prompt': request.prompt * n})

    def fold_responses(
        self, responses: list[InferenceResponse], n: int
    ) -> list[GroupedInferenceResponse]:
        return [GroupedInferenceResponse(responses=x) for x in list(grouper(responses, n))]

    async def agenerate(self, request: InferenceRequest) -> list[InferenceResponse]:
        return await self.base_generate(
            InferenceRequest.model_validate(request, from_attributes=True)
        )

    async def agroup_generate(
        self, request: InferenceRequest, group_size: int
    ) -> list[GroupedInferenceResponse]:
        if not self.supports_n:
            request = self.duplicate_requests(request, group_size)

        generations = await self.agenerate(request)

        generations = self.fold_responses(generations, group_size)

        return generations


def ensure_template(value: Any) -> ConversationTemplate:
    if isinstance(value, ConversationTemplate):
        return value
    elif isinstance(value, str):
        return ConversationTemplate.from_string(value)
    else:
        raise ValidationError(f"Invalid conversation template: {value}")


class DirectChatInferenceInterface(DirectInferenceInterface, ChatInferenceInterface):
    """Basic inference engines that operate directly on strings can extend this class and implement base_generate. This class implements necessary chat interfaces."""

    conversation_template: Annotated[ConversationTemplate, BeforeValidator(ensure_template)]

    async def base_generate(self, request: ChatInferenceRequest) -> list[ChatInferenceResponse]:
        base_generate_results = await super().base_generate(
            InferenceRequest(
                prompt=[self.conversation_template.format(messages) for messages in request.prompt],
                generation_args=request.generation_args,
            )
        )
        chat_message_results = self.conversation_template.parse_response(base_generate_results)
        return [
            ChatInferenceResponse(response=chat_message, **response.model_dump())
            for chat_message, response in zip(chat_message_results, base_generate_results)
        ]
