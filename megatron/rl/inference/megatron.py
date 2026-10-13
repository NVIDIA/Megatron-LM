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

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import FinishedRequestRecord
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import get_pg_size, log_single_rank
from megatron.training.global_vars import get_args, get_run_config, get_tokenizer
from megatron.training.utils import print_rank_0

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


def resolve_generation_lag(
    generation_lag: float | None,
    *,
    prompts_per_step: int,
    group_size: int,
    dp_size: int,
    max_requests: int,
) -> float:
    """Return the generation lag to run with. An unset `generation_lag` autotunes.
    """
    max_effective_groups = dp_size * max_requests // group_size
    max_effective_lag = max_effective_groups / prompts_per_step - 1
    if generation_lag is None:
        generation_lag = max_effective_lag
        print_rank_0(
            f"Autotuned rl-generation-lag={max_effective_lag:.2f} "
            f"(DP={dp_size}, max_requests={max_requests}, G={group_size}, P={prompts_per_step}).")
    else:
        print_rank_0(
            f"Using rl-generation-lag={generation_lag} "
            f"(max effective lag={max_effective_lag:.2f}; "
            f"DP={dp_size}, max_requests={max_requests}, G={group_size}, P={prompts_per_step}).")
    groups_in_flight = (generation_lag + 1) * prompts_per_step
    if groups_in_flight > max_effective_groups + 1e-6:
        print_rank_0(
            f"WARNING: {groups_in_flight:.1f} groups in flight oversubscribes the "
            f"inference engine (max effective lag is {max_effective_lag:.2f}). "
            f"Additional run-ahead beyond that point has no benefit.")
    if max_effective_lag < 0:
        print_rank_0(
            f"WARNING: max effective lag is {max_effective_lag:.2f} (negative) — the "
            f"inference engine cannot hold even one training step's worth of rollouts "
            f"({max_effective_groups} groups < P={prompts_per_step}). Even fully-synchronous GRPO "
            f"would oversubscribe. Consider scaling up inference resources.")
    return generation_lag


class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)
    _generation_lag: float = PrivateAttr(None)
    _openai_client: AsyncOpenAI = PrivateAttr(None)

    @property
    def generation_lag(self) -> float:
        """Trainer batches of rollout generation to run ahead with."""
        return self._generation_lag

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        tokenizer = get_tokenizer()
        cfg = get_run_config()

        # Use the shared, optimized client instead of spinning up a new one
        client = self._openai_client

        # Things that may be problematic when doing this switch
        # - Add BOS token
        # - Skip prompt logprobs
        temperature = request.generation_args.temperature
        response = await client.chat.completions.create(
            model="",
            messages=[message.model_dump() for message in request.prompt],
            temperature=1.0 if temperature is None else temperature,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            extra_body={
                "skip_prompt_log_probs": True,
                "add_BOS": (not cfg.rl.rl_skip_bos_token and tokenizer.bos is not None),
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

    @classmethod
    async def launch(cls, model: GPTModel, **kwargs):
        # Import here to avoid circular imports
        from megatron.inference.utils import get_dynamic_inference_engine

        args = get_args()
        cfg = get_run_config()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        # RL needs log probs, but not prompt log probs.
        args.return_log_probs = True
        args.skip_prompt_log_probs = True

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(model=model)
        inference_engine.local_metadata_ledger_enabled = True
        generation_lag = cfg.rl.rl_generation_lag
        if cfg.rl.rl_partial_rollouts:
            generation_lag = resolve_generation_lag(
                generation_lag,
                prompts_per_step=cfg.rl.grpo_prompts_per_step,
                group_size=cfg.rl.grpo_group_size,
                dp_size=get_pg_size(inference_engine.pg_collection.dp),
                max_requests=inference_engine.context.max_requests,
            )
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
                parsers=cfg.rl.rl_inference_parsers,
                verbose=kwargs.get('verbose', False),
                multimodal_prompt_config=(
                    inference_engine.controller.inference_wrapped_model.multimodal_prompt_config
                ),
            )
        else:
            client = None

        launched_server = cls(**kwargs)
        launched_server._client = client
        launched_server._inference_engine = inference_engine
        launched_server._generation_lag = generation_lag

        concurrency_limit = (
            get_pg_size(inference_engine.pg_collection.dp) * inference_engine.context.max_requests
        )
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

    def merge_global_request_ledgers(self) -> dict[str, FinishedRequestRecord]:
        """Union every engine's local-metadata ledger and clear them."""
        engine = self._inference_engine
        local, engine.local_metadata_ledger = engine.local_metadata_ledger, {}
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, local)
        merged: dict[str, FinishedRequestRecord] = {}
        for shard in shards:
            merged.update(shard)
        assert len(merged) == sum(len(shard) for shard in shards), (
            "finished-request ledger: duplicate uids across engine ledgers"
        )
        return merged

    async def resume(self):
        if self._inference_engine._state_events[EngineState.RUNNING].is_set():
            return

        if dist.get_rank() == 0:
            self._client.resume_engines()
        await self._inference_engine.wait_until(EngineState.RESUMED)

        if dist.get_rank() == 0:
            self._client.unpause_engines()
        await self._inference_engine.wait_until(EngineState.RUNNING)
