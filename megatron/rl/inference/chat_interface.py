# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import PrivateAttr

from .api import InferenceRequest, InferenceResponse, LLMChatMessage
from .inference_interface import ReturnsTokens


class MegatronChatInterface(ReturnsTokens):
    """Generates through the OpenAI chat endpoint of a Megatron text-generation server."""

    base_url: str
    add_bos: bool = False

    _openai_client: AsyncOpenAI | None = PrivateAttr(None)

    def openai_client(self) -> AsyncOpenAI:
        """The shared client, created on first use unless one was installed."""
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="NONE",
                http_client=DefaultAsyncHttpxClient(
                    timeout=None,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                ),
            )
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
                "return_raw_text": False,
            },
        )

        choice = response.choices[0]

        # The API response only carries the text; token IDs and logprobs are staged in the engine.
        return InferenceResponse(
            # TODO: Handle tool calls and reasoning in LLMChatMessage
            response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
            token_ids=None,
            logprobs=None,
            finish_reason=choice.finish_reason,
            prompt_length=None,
            completion_id=response.id,
        )
