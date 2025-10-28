# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
from argparse import Namespace

from pydantic import PrivateAttr

from megatron.core import parallel_state
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.coordinator import DynamicEngineCoordinator
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tokenizer

from ..inference.inference_interface import (
    ChatInferenceInterface,
    InferenceRequest,
    InferenceResponse,
    ReturnsRaw,
    ReturnsTokens,
)
from ..server.api import InferenceServer

logger = logging.getLogger(__name__)


## This code is copied from tools/run_text_generation_server.py
def get_static_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference.

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_seq_length=args.inference_max_seq_length,
        inference_max_requests=(
            args.inference_max_batch_size if args.inference_max_batch_size is not None else 1
        ),
        nccl_all_reduce_for_prefill=args.nccl_all_reduce_for_prefill,
    )

    inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return MCoreEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=(
            args.inference_max_batch_size if args.inference_max_batch_size is not None else 1
        ),
    )


## This code is copied from tools/run_text_generation_server.py
def get_dynamic_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference.

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    num_cuda_graphs = None
    if args.enable_cuda_graph:
        num_cuda_graphs = args.inference_dynamic_batching_num_cuda_graphs

    module = model.module.module if hasattr(model.module, "module") else model.module

    # Inference context.
    inference_context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=args.inference_max_seq_length,
        num_cuda_graphs=num_cuda_graphs,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        buffer_guaranteed_fraction=args.inference_dynamic_batching_buffer_guaranteed_fraction,
        chunk_size_tokens=args.inference_dynamic_batching_chunk_size,
        buffer_overflow_factor=args.inference_dynamic_batching_buffer_overflow_factor,
        max_requests_override=args.inference_dynamic_batching_max_requests_override,
        max_tokens_override=args.inference_dynamic_batching_max_tokens_override,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        materialize_only_last_token_logits=True,
        unified_memory_kvcache=args.inference_dynamic_batching_unified_memory_kvcache,
        is_hybrid_model=args.is_hybrid_model,
        layer_type_list=module.decoder.layer_type_list if args.is_hybrid_model else None,
        mamba_head_dim=args.mamba_head_dim,
        mamba_num_groups=args.mamba_num_groups,
        mamba_d_model=args.hidden_size,
        mamba_d_conv=4 if args.is_hybrid_model else None,
        mamba_d_state=args.mamba_state_dim,
    )

    inference_wrapped_model = GPTInferenceWrapper(model, args, inference_context)

    inference_wrapped_model.model_is_pipeline_parallel = not (
        parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
    )

    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )

    return DynamicInferenceEngine(
        controller=text_generation_controller,
        context=inference_context,
        enable_cuda_graph=args.enable_cuda_graph,
        random_seed=args.seed,
    )


class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    _coordinator: DynamicEngineCoordinator = PrivateAttr(None)
    _engine_task: asyncio.Task = PrivateAttr(None)
    _kill_engine: bool = PrivateAttr(False)

    async def base_generate(self, request: InferenceRequest):
        tokenizer = get_tokenizer()

        sampling_params = SamplingParams(
            num_tokens_to_generate=request.generation_args.max_tokens or 1024,
            temperature=request.generation_args.temperature or 1.0,
            top_k=request.generation_args.top_k or 0,
            top_p=request.generation_args.top_p or 0.0,
            termination_id=self._coordinator.engine.controller.tokenizer.eod,
            return_log_probs=True,
            skip_prompt_log_probs_for_dynamic_inference=True,
            add_BOS=tokenizer.bos is not None,
        )
        request_ids = [
            self._coordinator.schedule_request(prompt=prompt, sampling_params=sampling_params)
            for prompt in request.prompt
        ]
        responses = await asyncio.gather(
            *[self._coordinator.get_response(id) for id in request_ids]
        )
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
        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(args, model)
        coordinator = DynamicEngineCoordinator(
            inference_engine,
            inference_max_requests=inference_engine.context.max_requests,
            log_level=0,
        )
        launched_server = cls(**kwargs)
        launched_server._coordinator = coordinator

        loop = asyncio.get_running_loop()

        coordinator.startup(loop)

        return launched_server

    async def kill(self):
        await self._coordinator.shutdown()

    async def suspend(self):
        await self._coordinator.suspend_engine()

    def resume(self):
        self._coordinator.resume_engine()


class MegatronChatLocal(ChatInferenceInterface, MegatronLocal): ...
