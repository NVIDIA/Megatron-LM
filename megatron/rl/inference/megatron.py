# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
from argparse import Namespace

import torch.distributed as dist
from pydantic import PrivateAttr

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model, log_single_rank
from megatron.training import get_wandb_writer
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


## This code is copied from tools/run_text_generation_server.py
def get_static_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference.

    This function will automatically choose the TRTLLMBackend when possible,
    and default to Mcore backend if the user does not specify any backends.
    TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapped_model = GPTInferenceWrapper(model)
    pg_collection = get_attr_wrapped_model(model, "pg_collection")
    pp_group = pg_collection.pp
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer, pp_group=pp_group
    )
    return MCoreEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=(
            args.inference_max_requests if args.inference_max_requests is not None else 1
        ),
    )


class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _server_task: asyncio.Task = PrivateAttr(None)
    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)

    async def base_generate(self, request: InferenceRequest):

        assert self._server_task is not None, "Infernce server is not initialized"

        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=f"http://{self.host}:{self.port}", api_key="NONE")

        # Things that may be problematic when doign this switch
        # - Add BOS token
        # - Skip prompt logprobs
        generations = [ client.chat.completions.create(
            model="",
            messages=[message.model_dump() for message in prompt],
            temperature=request.generation_args.temperature or 1.0,
            top_p=request.generation_args.top_p or 0.0,
            n=request.generation_args.n or 1,
            logprobs=True,
        ) for prompt in request.prompt ]

        responses = await asyncio.gather(*generations)

        assert all(len(response.choices) == 1 for response in responses), "Still need to properly support requests with n > 1"

        return [
            InferenceResponse(
                response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
                raw_text=choice.raw_text,
                token_ids=choice.prompt_token_ids + choice.generation_token_ids,
                logprobs=choice.generation_log_probs,
                prompt_length=len(choice.prompt_token_ids),
            )
            for response in responses for choice in response.choices
        ]

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
            client = InferenceClient(inference_coordinator_addr=dp_addr)
            await client.start()
            server_task = loop.create_task(run_flask_server_on_client(
                client=client,
                tokenizer=inference_engine.controller.tokenizer,
                flask_port=8294,
                parsers=[]
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
