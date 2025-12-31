# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
from argparse import Namespace

import torch.distributed as dist
from pydantic import PrivateAttr

from megatron.core import parallel_state
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.inference_client import InferenceClient
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
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_mamba_inference_state_config_from_model,
    get_pg_size,
    log_single_rank,
)
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
def get_dynamic_inference_engine(
    args: Namespace,
    model: MegatronModule,
    inference_logging_step_interval: int = 0,
    metrics_writer = None
) -> AbstractEngine:
    """Get the relevant backend for running inference.

    This function will automatically choose the TRTLLMBackend when possible,
    and default to Mcore backend if the user does not specify any backends.
    TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.
        inference_logging_step_interval (int): Step interval for logging inference metrics.
        metrics_writer: Metrics writer (wandb module) for logging.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    enable_cuda_graph = args.cuda_graph_impl == "local"

    mamba_inference_state_config = get_mamba_inference_state_config_from_model(model)

    # DynamicInferenceContext must use the inference model's TP size, not the
    # training TP size from global args. The inference model may have a custom
    # ProcessGroupCollection with a different TP size.
    pg_collection = get_attr_wrapped_model(model, "pg_collection")
    tp_group = getattr(pg_collection, 'tp', None) if pg_collection is not None else None
    if tp_group is not None:
        inference_tp_size = get_pg_size(tp_group)
    else:
        inference_tp_size = args.tensor_model_parallel_size

    # Inference context.
    inference_context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers // args.pipeline_model_parallel_size,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=args.inference_max_seq_length,
        num_cuda_graphs=(
            args.inference_dynamic_batching_num_cuda_graphs if enable_cuda_graph else None
        ),
        block_size_tokens=args.inference_dynamic_batching_block_size,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        max_tokens=args.inference_dynamic_batching_max_tokens,
        tensor_model_parallel_size=inference_tp_size,
        materialize_only_last_token_logits=True,
        mamba_inference_state_config=mamba_inference_state_config,
        cache_mla_latent=args.multi_latent_attention and args.cache_mla_latents,
        kv_lora_rank=args.kv_lora_rank if args.multi_latent_attention else None,
        qk_pos_emb_head_dim=args.qk_pos_emb_head_dim,
        use_cuda_graphs_for_non_decode_steps=not args.decode_only_cuda_graphs,
        use_flashinfer_fused_rope=None,
        unified_memory_level=args.inference_dynamic_batching_unified_memory_level,
        cuda_graph_max_tokens=args.inference_dynamic_batching_cuda_graph_max_tokens,
        cuda_graph_mixed_prefill_count=args.inference_dynamic_batching_cuda_graph_mixed_prefill_count,
        metrics_writer=metrics_writer,
    )

    inference_wrapped_model = GPTInferenceWrapper(model, args, inference_context)

    inference_wrapped_model.model_is_pipeline_parallel = not (
        is_pp_first_stage(pg_collection.pp) and is_pp_last_stage(pg_collection.pp)
    )

    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )

    return DynamicInferenceEngine(
        controller=text_generation_controller,
        context=inference_context,
        random_seed=args.seed,
        track_paused_request_events=args.inference_dynamic_batching_track_paused_request_events,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        inference_logging_step_interval=inference_logging_step_interval,
        pg_collection=pg_collection,
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

        sampling_params = SamplingParams(
            num_tokens_to_generate=None,
            num_tokens_total=request.generation_args.max_tokens,
            temperature=request.generation_args.temperature or 1.0,
            top_k=request.generation_args.top_k or 0,
            top_p=request.generation_args.top_p or 0.0,
            termination_id=self._inference_engine.controller.tokenizer.eod,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            add_BOS=tokenizer.bos is not None,
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
        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        # Get inference logging configuration from args
        log_inference_wandb = args.inference_wandb_logging
        inference_logging_step_interval = args.inference_logging_step_interval

        # Get metrics writer if logging is enabled and on the logging rank
        # Use the same rank convention as training (last rank logs)
        metrics_writer = None
        if (
            inference_logging_step_interval > 0
            and log_inference_wandb
            and args.rank == (args.world_size - 1)
        ):
            metrics_writer = get_wandb_writer()
            if metrics_writer is None:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "WARNING: --rl-inference-logging-step-interval is set but no metrics writer "
                    "wandb module is available. Inference logging will be disabled.",
                )

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(
            args, model, inference_logging_step_interval, metrics_writer
        )
        await inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=41521, launch_inference_coordinator=True,
        )
        if dist.get_rank() == 0:
            # TODO: We have to do this only on the rank 0 process, should be fixed in the future when we have support for multiple inference clients. !2278
            client = InferenceClient(inference_coordinator_port=41521)
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
