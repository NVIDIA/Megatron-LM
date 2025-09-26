# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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

    def format(self, messages: list[LLMChatMessage]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def parse_response(self, responses: list[InferenceResponse]) -> list[LLMChatMessage]:
        return [
            LLMChatMessage(role="assistant", content=response.response) for response in responses
        ]

    @classmethod
    def from_string(cls, tokenizer_name: str) -> 'ConversationTemplate':
        return cls(tokenizer=AutoTokenizer.from_pretrained(tokenizer_name))
