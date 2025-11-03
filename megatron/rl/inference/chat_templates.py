# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .api import InferenceResponse, LLMChatMessage


class ConversationTemplate(BaseModel):
    """Transformers tokenizer based template."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = Field(repr=False)
    stop_words: list[str] = []

    def format(self, messages: list[LLMChatMessage], tools: list[dict] | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, tools=tools
        )

    def parse_response(self, responses: list[InferenceResponse]) -> list[LLMChatMessage]:
        return [
            LLMChatMessage(role="assistant", content=response.response) for response in responses
        ]

    @classmethod
    def from_string(cls, tokenizer_name: str) -> 'ConversationTemplate':
        if tokenizer_name == "null":
            warnings.warn(
                "Using NullConversationTemplate. This provides no chat templating to Chat requests."
            )
            return NullConversationTemplate()
        return cls(tokenizer=AutoTokenizer.from_pretrained(tokenizer_name))


class NullConversationTemplate(ConversationTemplate):

    tokenizer: None = None

    def format(self, messages: list[LLMChatMessage], tools: list[dict] | None = None) -> str:
        return "\n".join([f"{message.content}" for message in messages]) + "\n"

    def parse_response(self, responses: list[InferenceResponse]) -> list[LLMChatMessage]:
        return [
            LLMChatMessage(role="assistant", content=response.response) for response in responses
        ]
