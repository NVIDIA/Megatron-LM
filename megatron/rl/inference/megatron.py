# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging

import torch.distributed as dist
from pydantic import PrivateAttr

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tokenizer

from ..inference.inference_interface import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from ..server.api import InferenceServer

logger = logging.getLogger(__name__)


class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _server_task: asyncio.Task = PrivateAttr(None)
    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:

        assert self._server_task is not None, "Inference server is not initialized"
        tokenizer = get_tokenizer()
        args = get_args()

        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=f"http://{self.host}:{self.port}", api_key="NONE")

        # Things that may be problematic when doign this switch
        # - Add BOS token
        # - Skip prompt logprobs
        response = await client.chat.completions.create(
            model="",
            messages=[message.model_dump() for message in request.prompt],
            temperature=request.generation_args.temperature or 1.0,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            extra_body={
                "skip_prompt_log_probs": True,
                "add_BOS": (not args.rl_skip_bos_token and tokenizer.bos is not None),
            },
        )

        choice = response.choices[0]

        return InferenceResponse(
            # TODO: Handle tool calls and reasoning in LLMChatMessage
            response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
            raw_text=choice.raw_text,
            token_ids=choice.prompt_token_ids + choice.generation_token_ids,
            logprobs=choice.generation_log_probs,
            prompt_length=len(choice.prompt_token_ids),
        )

    @classmethod
    async def launch(cls, model: GPTModel, **kwargs):
        # Import here to avoid circular imports
        from megatron.inference.utils import get_dynamic_inference_engine

        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(model=model)
        dp_addr = await inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=41521, launch_inference_coordinator=True,
        )

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.flask_server import run_flask_server_on_client
            loop = asyncio.get_event_loop()
            client = InferenceClient(inference_coordinator_address=dp_addr)
            await client.start()
            server_task = loop.create_task(run_flask_server_on_client(
                client=client,
                tokenizer=inference_engine.controller.tokenizer,
                flask_port=kwargs.get('port', 8294),
                parsers=[],
                verbose=kwargs.get('verbose', False),
            ))
        else:
            client = None
            server_task = None
            
        launched_server = cls(**kwargs)
        launched_server._client = client
        launched_server._server_task = server_task
        launched_server._inference_engine = inference_engine

        return launched_server

    async def kill(self):
        if dist.get_rank() == 0:
            await self._client.stop_engines()
        await self._inference_engine.stopped.wait()

    async def suspend(self):
        if dist.get_rank() == 0:
            await self._client.pause_engines()
        await self._inference_engine.paused.wait()

    async def resume(self):
        if dist.get_rank() == 0:
            self._client.unpause_engines()
        await self._inference_engine.running.wait()
