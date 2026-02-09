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
    ChatInferenceInterface,
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

    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)

    async def base_generate(self, request: InferenceRequest):

        if any(isinstance(p, LLMChatMessage) for p in request.prompt):
            raise ValueError(
                "MegatronLocal does not support chat requests."
                "Use MegatronChatLocal to apply chat templating."
            )
        assert all(
            isinstance(p, str) for p in request.prompt
        ), "MegatronLocal only supports string prompts."

        assert self._client is not None, "Client is not initialized"

        tokenizer = get_tokenizer()
        args = get_args()

        sampling_params = SamplingParams(
            num_tokens_to_generate=None,
            num_tokens_total=request.generation_args.max_tokens,
            temperature=request.generation_args.temperature or 1.0,
            top_k=request.generation_args.top_k or 0,
            top_p=request.generation_args.top_p or 0.0,
            termination_id=self._inference_engine.controller.tokenizer.eod,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            add_BOS=(not args.rl_skip_bos_token and tokenizer.bos is not None),
        )
        requests = [
            self._client.add_request(prompt=prompt, sampling_params=sampling_params)
            for prompt in request.prompt
        ]
        records = await asyncio.gather(*requests)
        responses = [record[-1] for record in records]
        return [
            InferenceResponse(
                response=r.generated_text,
                raw_text=p + r.generated_text,
                token_ids=r.prompt_tokens.tolist() + r.generated_tokens,
                logprobs=r.generated_log_probs,
                prompt_length=len(r.prompt_tokens),
            )
            for p, r in zip(request.prompt, responses)
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
            # TODO: We have to do this only on the rank 0 process, should be fixed in the future when we have support for multiple inference clients. !2278
            client = InferenceClient(inference_coordinator_address=dp_addr)
            await client.start()
        else:
            client = None
        launched_server = cls(**kwargs)
        launched_server._client = client
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


class MegatronChatLocal(ChatInferenceInterface, MegatronLocal): ...
