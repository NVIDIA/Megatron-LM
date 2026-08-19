# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy
import json
import logging
import os
import warnings
from asyncio import Lock
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field

from ..__init__ import GenericGenerationArgs
from ..inference import InferenceRequest, InferenceResponse, LLMChatMessage
from .api import (
    EpisodeResult,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    RewardEvaluationResult,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from .pass_at_evaluation_agent import PassAtEvaluationAgent

logger = logging.getLogger(__name__)


class _WireModel(BaseModel, extra='allow'):
    """Parsed subset of a server-owned wire object; may hold unknown fields."""


class RunOutputItem(_WireModel):
    """Responses output item."""

    type: str
    prompt_token_ids: list[int] | None = None
    generation_token_ids: list[int] | None = None
    generation_log_probs: list[float] | None = None
    completion_id: str | None = None

    def is_trainable(self) -> bool:
        return (
            self.prompt_token_ids is not None
            and self.generation_token_ids is not None
            and self.generation_log_probs is not None
        )


class RunModelResponse(_WireModel):
    """The Responses object inside a /run result."""

    output: list[RunOutputItem]


class RunResult(_WireModel):
    """An environment server's /run verify result: reward plus the finished episode.

    Fields past `response` are server-specific extensions; may be absent from some replies.
    """

    reward: float
    response: RunModelResponse
    mask_sample: bool = False
    instance_config: dict | None = None
    failure_reason: str | None = None
    final_eval_time: float | None = None
    responses_create_params: dict | None = None


class EnvRunRequest(InferenceRequest):
    """Payload to be sent to the environment server's /run endpoint."""

    prompt: list[LLMChatMessage] = []
    payload: dict[str, Any]
    server_name: str | None = None


class EnvRunResponse(InferenceResponse):
    """Result received from the environment server's /run endpoint."""

    response: LLMChatMessage = LLMChatMessage(role='assistant', content='[env: unused]')
    finish_reason: str = 'env_unused'
    result: RunResult | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None


class EnvConnectExhausted(RuntimeError):
    """Connection failures exceeded the cap; the episode fails as a value."""


class EnvHTTPError(RuntimeError):
    """HTTP failure with the response body preserved for classification."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def resolve_curriculum_training_state(args) -> tuple[int, int]:
    """Return the live training iteration and prompt groups per iteration."""
    iteration = getattr(args, 'curr_iteration', None)
    if iteration is None:
        iteration = getattr(args, 'iteration')

    prompts_per_iter = int(getattr(args, 'grpo_prompts_per_step', None) or 64)
    if prompts_per_iter <= 0:
        raise ValueError(f"grpo_prompts_per_step must be positive, got {prompts_per_iter}")
    return int(iteration), prompts_per_iter


def next_curriculum_index(
    curriculum_cursor: int | None, iteration: int, prompts_per_iter: int, prompt_share: float = 1.0
) -> tuple[int, int]:
    """Advance a monotonic prompt-group cursor seeded from training progress."""
    iteration_start = int(iteration * prompts_per_iter * prompt_share)
    if curriculum_cursor is None or curriculum_cursor < iteration_start:
        curriculum_cursor = iteration_start

    return curriculum_cursor, curriculum_cursor + 1


def generation_args_to_run_payload(generation_args: GenericGenerationArgs, prompt: dict) -> dict:
    """Merge sampling settings into a dataset row's ``responses_create_params``."""
    if generation_args.top_k is not None:
        warnings.warn("top_k is not supported by the OpenAI Responses API and will be ignored.")

    payload = copy.deepcopy(prompt)
    payload['responses_create_params']['temperature'] = generation_args.temperature
    payload['responses_create_params']['top_p'] = generation_args.top_p
    return payload


def _field(obj, name: str, default=None):
    """Read a field from a parsed-model attribute or a plain dict interchangeably."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_prompt_from_run_input(input_obj) -> str | list[LLMChatMessage]:
    """Convert a ``responses_create_params.input`` value into eval-loggable form."""
    if isinstance(input_obj, str):
        return input_obj

    messages = []
    for item in input_obj or []:
        if _field(item, 'type') != 'message':
            continue
        content = _field(item, 'content')
        if content is None:
            continue
        if isinstance(content, list):
            parts = [
                _field(block, 'text')
                for block in content
                if _field(block, 'type') == 'input_text' and _field(block, 'text') is not None
            ]
            content = ' '.join(parts)
        elif not isinstance(content, str):
            content = str(content)
        messages.append(LLMChatMessage(role=_field(item, 'role', 'user'), content=content))

    return messages


def extract_response_text(response: RunModelResponse) -> LLMChatMessage | str:
    """Best-effort assistant text of a finished episode, for eval logging only."""
    texts = []
    for item in response.output:
        if item.type != 'message' or _field(item, 'role') != 'assistant':
            continue
        content = _field(item, 'content')
        if isinstance(content, list):
            texts.extend(
                _field(block, 'text')
                for block in content
                if _field(block, 'type') == 'output_text' and _field(block, 'text') is not None
            )
        elif isinstance(content, str):
            texts.append(content)

    if texts:
        return LLMChatMessage(role='assistant', content='\n'.join(texts).strip())
    return ''


def run_result_to_evaluation_response(
    result: RunResult, problem_id: str | None, env_id: str
) -> EvaluationResponse[RewardEvaluationResult]:
    """Package a /run result as a single-result evaluation response."""
    params = result.responses_create_params or {}
    prompt = extract_prompt_from_run_input(params.get('input'))
    # Display-only: the reported string may not be faithful to the raw episode.
    response = extract_response_text(result.response)

    return EvaluationResponse[RewardEvaluationResult](
        results=[
            RewardEvaluationResult(
                prompt=prompt, response=response, reward=result.reward, problem_id=problem_id
            )
        ],
        env_id=env_id,
    )


class ResponsesEnvAgent(RolloutGenerator, GroupedRolloutGenerator, PassAtEvaluationAgent):
    """Rollout/evaluation agent over a remote Responses-API environment server."""

    agent_name: str
    dataset_file: str
    # Environment server the /run POSTs target. The static host:port default suits a single server;
    # directory-based discovery overrides resolve_run_url instead.
    env_server_host_port: str | None = None
    server_name: str | None = None
    env_id: str | None = None
    # Per-turn output-token caps stamped onto every /run body (training and eval).
    # max_output_tokens_per_step takes priority over defer_output_cap_to_engine.
    max_output_tokens_per_step: int | None = None
    defer_output_cap_to_engine: bool = False
    # Curriculum sampling assumes dataset rows are pre-sorted in curriculum order;
    # the cursor is seeded from the training iteration so restarts resume, not reset.
    curriculum_sampling: bool = Field(
        default_factory=lambda: os.environ.get('LANGRL_ENV_CURRICULUM', '0') == '1'
    )
    # Connection-establishment timeout; there is deliberately no read/total timeout,
    # so slow-but-alive episodes are never severed mid-flight.
    # rollout_timeout_s (opt-in) is the wall-clock bound; discards slow episodes' training signals.
    http_connect_timeout_s: float = Field(
        default_factory=lambda: float(os.environ.get('LANGRL_ENV_HTTP_CONNECT_TIMEOUT_S', '60'))
    )
    rollout_timeout_s: float | None = Field(
        default_factory=lambda: float(os.environ.get('LANGRL_ENV_ROLLOUT_TIMEOUT_S', '0')) or None
    )
    http_retries: int = Field(
        default_factory=lambda: int(os.environ.get('LANGRL_ENV_HTTP_RETRIES', '3'))
    )
    # Bounded connect retries: re-resolve routes every refresh_every consecutive failures
    # give up at failure_cap so the pipeline slot frees as a placeholder.
    connect_refresh_every: int = Field(
        default_factory=lambda: int(os.environ.get('LANGRL_ENV_CONNECT_REFRESH_EVERY', '10'))
    )
    connect_failure_cap: int = Field(
        default_factory=lambda: int(os.environ.get('LANGRL_ENV_CONNECT_FAILURE_CAP', '60'))
    )
    connect_retry_wait_s: float = Field(
        default_factory=lambda: float(os.environ.get('LANGRL_ENV_CONNECT_RETRY_WAIT_S', '5'))
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.dataset = self.load_dataset()
        if self.env_id is None:
            self.env_id = self.agent_name
        self._curriculum_cursor: int | None = None

    def load_dataset(self):
        """Load the JSONL dataset, expanding ``${VAR}`` placeholders in the path."""
        dataset_file = os.path.expandvars(self.dataset_file)
        if '${' in dataset_file:
            raise ValueError(
                f"dataset_file {self.dataset_file!r} contains an unexpanded environment variable; "
                "export it before launching."
            )
        data = []
        with open(dataset_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Error decoding dataset JSON on line: {line.strip()}. Error: {e}"
                    ) from e
        return data

    async def get_prompt(self, validation: bool) -> dict:
        """Next dataset row: curriculum order for training, uniform for validation."""
        if validation or not self.curriculum_sampling:
            return self.dataset[np.random.randint(0, len(self.dataset))]
        # On fallback (no Megatron state), sample uniformly rather than blocking.
        # Rollout collection only runs on one rank; DP peers must not advance the cursor.
        try:
            from megatron.training.global_vars import get_args

            iteration, prompts_per_iter = resolve_curriculum_training_state(get_args())
        except Exception:
            return self.dataset[np.random.randint(0, len(self.dataset))]

        idx_global, self._curriculum_cursor = next_curriculum_index(
            self._curriculum_cursor,
            iteration,
            prompts_per_iter,
            prompt_share=getattr(self, '_prompt_share', 1.0),
        )
        return self.dataset[idx_global % len(self.dataset)]

    def build_token_rollout(self, result: RunResult, problem_id: str | None) -> TokenRollout:
        """Convert a /run result into a (possibly multi-turn) TokenRollout.

        Each trainable output item becomes one trajectory row.
        Subclasses may override to post-process the built rollout (e.g. reward shaping).
        """
        trajectory: list[list[int]] = []
        generation_mask: list[list[bool]] = []
        logprobs: list[list[float]] = []
        completion_ids: list[str] = []

        for item in result.response.output:
            if not item.is_trainable():
                continue
            trajectory.append(item.prompt_token_ids + item.generation_token_ids)
            generation_mask.append(
                [False] * len(item.prompt_token_ids) + [True] * len(item.generation_token_ids)
            )
            logprobs.append(item.generation_log_probs)
            if item.completion_id is not None:
                completion_ids.append(item.completion_id)

        if not trajectory:
            # Dominant cause: an over-context request, answered with an episode that
            # carries no token metadata. One bad episode must not crash the GRPO step.
            logger.warning(
                "Env response had no trainable output items (likely over-context prompt): "
                "problem_id=%s env_id=%s reward=%s; padding with an empty placeholder.",
                problem_id,
                self.env_id,
                result.reward,
            )
            return TokenRollout.placeholder(self.env_id, failure_reason='no_trainable_output')

        mask_sample = result.mask_sample
        if not mask_sample and isinstance(result.instance_config, dict):
            mask_sample = bool(result.instance_config.get('mask_sample', False))
        if mask_sample:
            failure_reason = result.failure_reason or 'env_masked'
            logger.warning(
                "Env marked rollout non-trainable: problem_id=%s env_id=%s reason=%s",
                problem_id,
                self.env_id,
                failure_reason,
            )
            return TokenRollout.placeholder(
                self.env_id, failure_reason=failure_reason, rollout_status='masked'
            )

        # completion_ids join rollouts to engine request records (staleness/eviction telemetry);
        # stamp only a complete per-turn set, a partial one would misalign.
        if len(completion_ids) != len(trajectory):
            completion_ids = []

        return TokenRollout(
            trajectory=trajectory,
            generation_mask=generation_mask,
            reward=result.reward,
            logprobs=logprobs,
            env_id=self.env_id,
            problem_id=problem_id,
            completion_ids=completion_ids,
            generation_cap=self.max_output_tokens_per_step,
            rollout_status='graded' if result.final_eval_time is not None else 'ok',
        )

    def resolve_run_url(self, server_name: str | None) -> str:
        """Return the /run URL for one POST; called per attempt so overrides can re-route."""
        if self.env_server_host_port is None:
            raise ValueError(
                f"ResponsesEnvAgent[{self.agent_name}] has no env_server_host_port; set it "
                "or override resolve_run_url with a service-discovery implementation."
            )
        return f"http://{self.env_server_host_port}/run"

    async def refresh_routes(self) -> None:
        """Re-resolve server routes after repeated connection failures. Default: static, no-op."""

    def _http_client(self) -> httpx.AsyncClient:
        client = getattr(self, '_client', None)
        if client is None:
            # Episodes hold their connection for the full run duration;
            # admission is the environment server's job, so the client pool is unbounded.
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=self.http_connect_timeout_s),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
            )
            object.__setattr__(self, '_client', client)
        return client

    async def _refresh_routes_single_flight(self, seen_generation: int) -> None:
        """Run refresh_routes once per generation; concurrent callers piggyback."""
        lock = getattr(self, '_route_refresh_lock', None)
        if lock is None:
            lock = Lock()
            object.__setattr__(self, '_route_refresh_lock', lock)
        async with lock:
            if getattr(self, '_route_generation', 0) != seen_generation:
                return
            await self.refresh_routes()
            object.__setattr__(self, '_route_generation', seen_generation + 1)
            logger.warning(
                "ResponsesEnvAgent[%s]: refreshed server routes (generation %s)",
                self.agent_name,
                seen_generation + 1,
            )

    async def _post_run(self, payload: dict[str, Any], server_name: str | None = None) -> RunResult:
        inner = self._post_run_bounded(payload, server_name)
        if self.rollout_timeout_s is not None:
            return await asyncio.wait_for(inner, timeout=self.rollout_timeout_s)
        return await inner

    async def _post_run_bounded(
        self, payload: dict[str, Any], server_name: str | None = None
    ) -> RunResult:
        name = server_name or self.server_name
        client = self._http_client()
        consecutive_failures = 0
        while True:
            generation = getattr(self, '_route_generation', 0)
            url = self.resolve_run_url(name)
            try:
                response = await client.post(url, json=payload)
                body = response.text
                if response.status_code >= 400:
                    detail = body.strip() or response.reason_phrase or 'empty response body'
                    raise EnvHTTPError(response.status_code, detail[:4000])
                return RunResult.model_validate_json(body)
            except httpx.TransportError as e:
                # Connect refused/reset/timeout and mid-body read errors all land here:
                # re-POST, re-resolving periodically, and give up at the cap.
                consecutive_failures += 1
                if consecutive_failures >= self.connect_failure_cap:
                    raise EnvConnectExhausted(
                        f"{consecutive_failures} consecutive connection failures posting to "
                        f"{name or 'the env server'} (last url {url}): {type(e).__name__}: {e}"
                    ) from e
                if consecutive_failures % self.connect_refresh_every == 0:
                    logger.warning(
                        "ResponsesEnvAgent[%s]: %s consecutive connection failures on %s "
                        "(%s); re-resolving server routes",
                        self.agent_name,
                        consecutive_failures,
                        url,
                        type(e).__name__,
                    )
                    try:
                        await self._refresh_routes_single_flight(generation)
                    except Exception as refresh_error:
                        logger.warning(
                            "ResponsesEnvAgent[%s]: route refresh failed (%s: %s); "
                            "retrying with the old routes",
                            self.agent_name,
                            type(refresh_error).__name__,
                            refresh_error,
                        )
                await asyncio.sleep(self.connect_retry_wait_s)

    async def get_rollout_response(
        self,
        request: RolloutRequest | GroupedRolloutRequest | EvaluationRequest,
        inference_request: InferenceRequest,
    ) -> InferenceResponse:
        """Run one /run request for the pipeline's infer stage; failures travel as values."""
        assert isinstance(inference_request, EnvRunRequest)
        payload = inference_request.payload
        # Route to the server stamped at prepare time: a dispatching parent may run
        # this method on behalf of the child that built the payload.
        target_server = inference_request.server_name or self.server_name

        for attempt in range(self.http_retries):
            try:
                return EnvRunResponse(
                    result=await self._post_run(payload, server_name=target_server)
                )
            except EnvHTTPError as e:
                logger.warning(
                    "get_rollout_response HTTP %s from %s: %s", e.status, target_server, e.detail
                )
                if e.status >= 500 and attempt < self.http_retries - 1:
                    wait_s = 5 * (attempt + 1)
                    logger.warning(
                        "get_rollout_response retrying HTTP %s in %ss (attempt %s/%s)",
                        e.status,
                        wait_s,
                        attempt + 1,
                        self.http_retries,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                return EnvRunResponse(failure_reason=f"http_{e.status}", failure_detail=e.detail)
            except asyncio.TimeoutError as e:
                logger.warning(
                    "get_rollout_response attempt %s/%s timed out after %ss",
                    attempt + 1,
                    self.http_retries,
                    self.rollout_timeout_s,
                )
                if attempt < self.http_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                return EnvRunResponse(failure_reason='rollout_timeout', failure_detail=str(e))
            except Exception as e:
                logger.exception("get_rollout_response failed; padding with a placeholder")
                return EnvRunResponse(failure_reason=type(e).__name__, failure_detail=str(e))

        return EnvRunResponse(failure_reason='unknown', failure_detail='exhausted retry loop')

    def _stamp_output_caps(self, payload: dict) -> dict:
        """Apply this agent's output-token caps to a /run body in place."""
        params = payload['responses_create_params']
        if self.max_output_tokens_per_step is not None:
            current = params.get('max_output_tokens')
            cap = self.max_output_tokens_per_step
            params['max_output_tokens'] = min(current, cap) if current else cap
        elif self.defer_output_cap_to_engine:
            params['max_output_tokens'] = None
        return payload

    async def prepare_group_rollout(self, request: GroupedRolloutRequest) -> GroupRolloutParams:
        prompt = await self.get_prompt(validation=request.validation)
        payload = self._stamp_output_caps(
            generation_args_to_run_payload(request.generation_args, prompt)
        )
        problem_id = _field(prompt, 'problem_id')
        inference_request = EnvRunRequest(payload=payload, server_name=self.server_name)

        # The server runs the whole episode in one /run call: an episode is exactly one
        # request/response and the conversation lives (and stays) server-side.
        async def run_episode() -> EpisodeResult:
            response = await self.get_rollout_response(request, inference_request)
            return EpisodeResult(responses=[response], conversation=[])

        async def build_rollout(episode: EpisodeResult) -> TokenRollout:
            response = episode.responses[-1]
            assert isinstance(response, EnvRunResponse)
            if response.result is None:
                logger.warning(
                    "prepare_group_rollout: padding failed episode with placeholder for "
                    "env=%s: %s %s",
                    self.env_id,
                    response.failure_reason,
                    response.failure_detail,
                )
                return TokenRollout.placeholder(
                    self.env_id, failure_reason=response.failure_reason or 'unknown'
                )
            return self.build_token_rollout(response.result, problem_id)

        return GroupRolloutParams(run_episode=run_episode, build_rollout=build_rollout)

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        """N independent rollouts composed from the group-rollout primitives."""

        async def _single_rollout() -> TokenRollout:
            params = await self.prepare_group_rollout(request)
            return await params.build_rollout(await params.run_episode())

        return list(await asyncio.gather(*[_single_rollout() for _ in range(request.num_rollouts)]))

    async def _evaluation(
        self, prompt: dict, golden: None, request: EvaluationRequest
    ) -> EvaluationResponse[RewardEvaluationResult]:
        payload = self._stamp_output_caps(
            generation_args_to_run_payload(request.generation_args, prompt)
        )
        # Same bounded, re-resolving transport as training rollouts.
        result = await self._post_run(payload, server_name=self.server_name)
        return run_result_to_evaluation_response(result, _field(prompt, 'problem_id'), self.env_id)

    def evaluation_prompts(self, num_prompts: int, validation: bool = False) -> list[dict]:
        return [self.dataset[i] for i in range(num_prompts) if i < len(self.dataset)]

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        all_prompts = self.evaluation_prompts(request.num_prompts, request.validation)

        results = await asyncio.gather(*[self.evaluation(p, None, request) for p in all_prompts])

        return type(results[0])(
            results=sum([result.results for result in results], []), env_id=self.env_id
        )
