# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from openai import AsyncOpenAI
from pydantic import PrivateAttr

from .api import InferenceRequest, InferenceResponse, LLMChatMessage
from .inference_interface import ReturnsRaw, ReturnsTokens


class MegatronChatInterface(ReturnsTokens, ReturnsRaw):
    """Generates through the OpenAI chat endpoint of a Megatron text-generation server."""

    base_url: str
    add_bos: bool = False

    _openai_client: AsyncOpenAI | None = PrivateAttr(None)

    def openai_client(self) -> AsyncOpenAI:
        """The shared client, created on first use unless one was installed."""
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(base_url=self.base_url, api_key="NONE", timeout=None)
        return self._openai_client

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        temperature = request.generation_args.temperature
        response = await self.openai_client().chat.completions.create(
            model="",
            messages=[message.model_dump() for message in request.prompt],
            temperature=1.0 if temperature is None else temperature,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            extra_body={
                "skip_prompt_log_probs": True,
                "add_BOS": self.add_bos,
                # TODO: These are non-standard fields that add significant memory overheads to the
                # chat completions payload. return_raw_text also wastes a lot of CPU cycles
                # detokenizing prompt tokens, especially expensive for long prompts in agentic RL.
                # Set to False if not needed in MRL.
                "return_tokenized_data": True,
                "return_raw_text": True,
            },
        )

        choice = response.choices[0]

        return InferenceResponse(
            # TODO: Handle tool calls and reasoning in LLMChatMessage
            response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
            raw_text=choice.message.raw_text,
            token_ids=choice.message.prompt_token_ids + choice.message.generation_token_ids,
            logprobs=choice.message.generation_log_probs,
            finish_reason=choice.finish_reason,
            prompt_length=len(choice.message.prompt_token_ids),
            completion_id=response.id,
        )
