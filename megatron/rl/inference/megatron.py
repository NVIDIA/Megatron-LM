# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging

import httpx
import torch.distributed as dist
from openai import AsyncOpenAI, DefaultAioHttpClient
from pydantic import PrivateAttr

try:
    import h2  # noqa: F401
    use_http2 = True
except ImportError:
    use_http2 = False

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
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
logging.getLogger("httpx").setLevel(logging.WARNING)

class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)
    _rl_kv_cache_management_mode: KVCacheManagementMode = PrivateAttr(None)
    _openai_client: AsyncOpenAI = PrivateAttr(None)

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        tokenizer = get_tokenizer()
        args = get_args()

        # Use the shared, optimized client instead of spinning up a new one
        client = self._openai_client

        # Things that may be problematic when doing this switch
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
            policy_epoch=choice.policy_epoch,
            kv_cache_epoch=choice.kv_cache_epoch,
            num_evictions=getattr(choice, 'num_evictions', 0),
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
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import start_text_gen_server

            client = InferenceClient(inference_coordinator_address=dp_addr)
            client.start()

            start_text_gen_server(
                coordinator_addr=dp_addr,
                tokenizer=inference_engine.controller.tokenizer,
                rank=dist.get_rank(),
                server_port=kwargs.get('port', 8294),
                parsers=[],
                verbose=kwargs.get('verbose', False),
            )
        else:
            client = None

        launched_server = cls(**kwargs)
        launched_server._client = client
        launched_server._inference_engine = inference_engine
        launched_server._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )

        concurrency_limit = args.grpo_prompts_per_step * args.grpo_group_size * args.rl_parallel_generation_tasks
        custom_limits = httpx.Limits(
            max_connections=concurrency_limit,
            max_keepalive_connections=concurrency_limit,
        )
        http_client = DefaultAioHttpClient(
            timeout=None,
            limits=custom_limits,
            http2=use_http2
        )

        launched_server._openai_client = AsyncOpenAI(
            base_url=f"http://{launched_server.host}:{launched_server.port}",
            api_key="NONE",
            http_client=http_client
        )

        return launched_server

    async def kill(self):
        # Gracefully close the shared OpenAI client connections
        if self._openai_client is not None:
            await self._openai_client.close()

        if dist.get_rank() == 0:
            self._client.pause_engines()
        await self._inference_engine.wait_until(EngineState.PAUSED)

        if dist.get_rank() == 0:
            self._client.stop_engines()
        await self._inference_engine.wait_until(EngineState.STOPPED)

        if dist.get_rank() == 0:
            self._client.shutdown_coordinator()
            self._client.stop()

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import stop_text_gen_server
            stop_text_gen_server()

    def set_generation_epoch(self, generation_epoch: int):
        if dist.get_rank() == 0:
            self._client.set_generation_epoch(generation_epoch)

    async def suspend(self):
        if dist.get_rank() == 0:
            self._client.pause_engines()
        await self._inference_engine.wait_until(EngineState.PAUSED)

        if dist.get_rank() == 0:
            self._client.suspend_engines()
        await self._inference_engine.wait_until(EngineState.SUSPENDED)

    async def resume(self):
        if self._inference_engine._state_events[EngineState.RUNNING].is_set():
            return

        if dist.get_rank() == 0:
            self._client.resume_engines()
        await self._inference_engine.wait_until(EngineState.RESUMED)

        if dist.get_rank() == 0:
            self._client.unpause_engines()
        await self._inference_engine.wait_until(EngineState.RUNNING)
