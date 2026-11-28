# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy
import json
import logging
import os
import re
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from .. import GenericGenerationArgs
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

# Gym wire constants: the run-body key a capture key rides on, the control-route prefix on the
# policy model server, and the failure reason of a call that never returned a completion
# (off-chain, no poison).
ROLLOUT_ID_KEY = '_ng_rollout_id'
CAPTURE_CONTROL_PREFIX = '/training-token-capture/control'
UNCOMMITTED_CALL_REASON = 'request_finished_without_staged_coordinates'


class RunOutputItem(BaseModel):
    """Responses output item; only what evaluation logging reads."""

    type: str
    role: str | None = None
    content: str | list[dict[str, Any]] | None = None


class RunModelResponse(BaseModel):
    """The Responses object inside a /run result; `id` is the served envelope Gym recorded."""

    id: str | None = None
    output: list[RunOutputItem] = []


class RunResult(BaseModel):
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
    # The served response id the harness declares it kept (Gym's terminal_response_id).
    terminal_response_id: str | None = None


class ManifestRow(BaseModel):
    """One call of Gym's manifest, in the fields terminal selection and the ledger join read."""

    model_call_id: str
    parent_call_id: str | None = None
    prev_len: int
    cum_len: int
    staging_key: str
    response_id: str
    admitted_at: float | None = None


@dataclass(frozen=True)
class EpisodeKeys:
    """What Gym's manifest says about one episode, in the engine ledger's terms."""

    # Staging keys (engine uids) of the terminal chain, root first: the trainable turns.
    turns: list[str] = field(default_factory=list)
    # Staging keys of every other recorded call: retries, dead branches, poisoned episodes.
    others: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    terminal_selection: str | None = None

    @property
    def all_keys(self) -> list[str]:
        return self.turns + self.others


class EnvRunRequest(InferenceRequest):
    """Payload to be sent to the environment server's /run endpoint."""

    prompt: list[LLMChatMessage] = []
    payload: dict[str, Any]


class EnvRunResponse(InferenceResponse):
    """Result received from the environment server's /run endpoint."""

    response: LLMChatMessage = LLMChatMessage(role='assistant', content='[env: unused]')
    finish_reason: str = 'env_unused'
    result: RunResult | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None
    # The capture key this episode ran under (_ng_rollout_id); the manifest read uses it.
    rollout_id: str | None = None


class EnvConnectExhausted(RuntimeError):
    """Connection failures exceeded the cap; the episode fails as a value."""


class EnvHTTPError(RuntimeError):
    """HTTP failure with the response body preserved for classification."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def new_rollout_id(prefix: str) -> str:
    """A fresh Gym capture key: sanitized prefix plus a uuid.

    The `r` before the uuid keeps the key from ever ending in Gym's attempt suffix, `-a<N>`.
    """
    stem = re.sub(r'[^A-Za-z0-9._-]+', '-', prefix).lstrip('._-') or 'episode'
    return f"{stem}-r{uuid.uuid4().hex}"


def select_terminal_call(rows: list[ManifestRow]) -> tuple[ManifestRow | None, str]:
    """Gym's parent-link heuristic: walk from the preferred root while exactly one child extends."""
    # The caller passes at least one row, and no two rows share a call id.
    call_ids = {row.model_call_id for row in rows}
    if any(row.parent_call_id is not None and row.parent_call_id not in call_ids for row in rows):
        return None, 'orphaned_row'
    children: dict[str, list[ManifestRow]] = {}
    roots: list[ManifestRow] = []
    for row in rows:
        if row.parent_call_id is None:
            roots.append(row)
        else:
            children.setdefault(row.parent_call_id, []).append(row)
    if not roots:
        return None, 'no_root'
    position = {row.model_call_id: index for index, row in enumerate(rows)}

    def root_key(row: ManifestRow):
        # Admission time, unstamped rows last, manifest position as the tie-break.
        return (row.admitted_at is None, row.admitted_at or 0.0, position[row.model_call_id])

    def survivors(candidates: list[ManifestRow]) -> list[ManifestRow]:
        extended = [row for row in candidates if children.get(row.model_call_id)]
        return extended or candidates

    node = min(survivors(roots), key=root_key)
    while True:
        candidates = children.get(node.model_call_id) or []
        if not candidates:
            return node, 'selected'
        pool = survivors(candidates)
        if len(pool) > 1:
            return None, 'ambiguous_terminal'
        node = pool[0]


def _select_terminal(
    rows: list[ManifestRow], scored_response_id: str | None, declared_response_id: str | None
) -> tuple[ManifestRow | None, str, str]:
    """The terminal row, how it was chosen (Gym's method names) and the reason when it was not."""
    if declared_response_id:
        # A declaration is authoritative: an id the ledger cannot confirm never falls back.
        matches = [row for row in rows if row.response_id == declared_response_id]
        if len(matches) != 1:
            return (
                None,
                'declared',
                'declared_ambiguous' if matches else 'declared_terminal_not_captured',
            )
        return matches[0], 'declared', ''
    if scored_response_id:
        matches = [row for row in rows if row.response_id == scored_response_id]
        if len(matches) == 1:
            return matches[0], 'response_id', ''
    row, reason = select_terminal_call(rows)
    return row, 'heuristic', reason


def _terminal_chain(
    rows: list[ManifestRow], terminal: ManifestRow
) -> tuple[list[ManifestRow] | None, str]:
    """Follow parent links from the terminal to its root; every link must be contiguous."""
    by_id = {row.model_call_id: row for row in rows}
    chain, seen, node = [], set(), terminal
    while True:
        if node.model_call_id in seen:
            return None, 'lineage_cycle'
        seen.add(node.model_call_id)
        chain.append(node)
        if node.parent_call_id is None:
            break
        parent = by_id.get(node.parent_call_id)
        if parent is None:
            return None, 'missing_parent'
        if parent.cum_len != node.prev_len:
            return None, 'parent_length_mismatch'
        node = parent
    chain.reverse()
    return chain, ''


def episode_keys_from_manifest(
    manifest: dict[str, Any],
    *,
    scored_response_id: str | None = None,
    declared_response_id: str | None = None,
) -> EpisodeKeys:
    """Turn Gym's manifest into the episode's turns and its other calls, failing closed.

    Mirrors NeMo RL's receipt assembly: a poisoning failure row, an unparsable row, an
    unattributable terminal or a broken chain leaves the episode untrainable, and the
    keys of every parsed row still come back so their ledger records can be drained.
    """
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    invalid = False
    for raw in manifest.get('records') or []:
        try:
            row = ManifestRow.model_validate(raw)
        except ValidationError:
            invalid = True
            continue
        if row.model_call_id in seen:
            continue
        seen.add(row.model_call_id)
        rows.append(row)
    all_keys = [row.staging_key for row in rows]

    def poisoned(reason: str, selection: str | None = None) -> EpisodeKeys:
        return EpisodeKeys(others=all_keys, failure_reason=reason, terminal_selection=selection)

    for failure in manifest.get('failures') or []:
        reason = str(failure.get('reason') or '') or 'capture_failed'
        if reason != UNCOMMITTED_CALL_REASON:
            return poisoned(reason)
    if invalid:
        return poisoned('invalid_manifest_row')
    if not rows:
        return poisoned('no_captured_calls')
    terminal, selection, reason = _select_terminal(rows, scored_response_id, declared_response_id)
    if terminal is None:
        return poisoned(reason, selection)
    chain, reason = _terminal_chain(rows, terminal)
    if chain is None:
        return poisoned(reason, selection)
    chain_ids = {row.model_call_id for row in chain}
    return EpisodeKeys(
        turns=[row.staging_key for row in chain],
        others=[row.staging_key for row in rows if row.model_call_id not in chain_ids],
        terminal_selection=selection,
    )


def curriculum_collections(
    iteration: int,
    prompts_per_collection: int,
    group_size: int = 1,
    grpo_iterations: int = 1,
    global_batch_size: int | None = None,
) -> int:
    """Completed data collections at `iteration`; the cursor advances per collection, not step."""
    if prompts_per_collection <= 0:
        raise ValueError(f"grpo_prompts_per_step must be positive, got {prompts_per_collection}")
    samples_per_collection = prompts_per_collection * max(1, group_size)
    global_batch_size = global_batch_size or samples_per_collection
    batches_per_collection = max(1, samples_per_collection // global_batch_size)
    return int(iteration) // max(1, max(1, grpo_iterations) * batches_per_collection)


def generation_args_to_run_payload(generation_args: GenericGenerationArgs, prompt: dict) -> dict:
    """Stamp the trainer's sampling settings over the row's `responses_create_params`"""
    if generation_args.top_k is not None and generation_args.top_k > 0:
        warnings.warn("top_k is not supported by the OpenAI Responses API and will be ignored.")

    payload = copy.deepcopy(prompt)
    payload['responses_create_params']['temperature'] = generation_args.temperature
    # Megatron uses top_p == 0.0 (and None) as its keep-all sentinel;
    # the OpenAI wire requires 0 < top_p <= 1 and vLLM-class engines reject 0.0 outright.
    top_p = generation_args.top_p
    payload['responses_create_params']['top_p'] = 1.0 if top_p is None or top_p <= 0.0 else top_p
    # generation_args.max_tokens is deliberately not forwarded: it dies at the client,
    # matching MegatronLocal; output caps are stamped by _stamp_output_caps.
    return payload


def extract_prompt_from_run_input(input_obj) -> str | list[LLMChatMessage]:
    """Convert a `responses_create_params.input` value into eval-loggable form."""
    if isinstance(input_obj, str):
        return input_obj

    messages = []
    for item in input_obj or []:
        if item.get('type') != 'message':
            continue
        content = item.get('content')
        if content is None:
            continue
        if isinstance(content, list):
            parts = [
                block.get('text')
                for block in content
                if block.get('type') == 'input_text' and block.get('text') is not None
            ]
            content = ' '.join(parts)
        elif not isinstance(content, str):
            content = str(content)
        messages.append(LLMChatMessage(role=item.get('role', 'user'), content=content))

    return messages


def extract_response_text(response: RunModelResponse) -> LLMChatMessage | str:
    """Best-effort assistant text of a finished episode, for eval logging only."""
    texts = []
    for item in response.output:
        if item.type != 'message' or item.role != 'assistant':
            continue
        if isinstance(item.content, list):
            texts.extend(
                block.get('text')
                for block in item.content
                if block.get('type') == 'output_text' and block.get('text') is not None
            )
        elif isinstance(item.content, str):
            texts.append(item.content)

    if texts:
        return LLMChatMessage(role='assistant', content='\n'.join(texts).strip())
    return ''


def run_result_to_evaluation_response(
    result: RunResult, problem_id: str | None, env_id: str, completion_ids: list[str] = ()
) -> EvaluationResponse[RewardEvaluationResult]:
    """Package a /run result as a single-result evaluation response."""
    params = result.responses_create_params or {}
    prompt = extract_prompt_from_run_input(params.get('input'))
    # Display-only: the reported string may not be faithful to the raw episode.
    response = extract_response_text(result.response)

    return EvaluationResponse[RewardEvaluationResult](
        results=[
            RewardEvaluationResult(
                prompt=prompt,
                response=response,
                reward=result.reward,
                problem_id=problem_id,
                completion_ids=list(completion_ids),
            )
        ],
        env_id=env_id,
    )


class ResponsesEnvAgent(RolloutGenerator, GroupedRolloutGenerator, PassAtEvaluationAgent):
    """Rollout/evaluation agent over a remote Responses-API environment server."""

    agent_name: str
    dataset_file: str
    # Environment server the /run POSTs target.
    env_server_host_port: str
    env_id: str | None = None
    # Gym's policy model server, which hosts the capture ledger's manifest and retire routes.
    # The bearer token comes from the environment.
    capture_control_host_port: str
    capture_control_token_env: str = 'NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN'
    capture_control_timeout_s: float = Field(default=60.0, gt=0)
    # Per-turn output-token caps stamped onto every /run body (training and eval).
    # max_output_tokens_per_step takes priority over defer_output_cap_to_engine.
    max_output_tokens_per_step: int | None = None
    defer_output_cap_to_engine: bool = False
    # Curriculum sampling assumes dataset rows are pre-sorted in curriculum order;
    # the cursor is seeded from the training iteration so restarts resume, not reset.
    curriculum_sampling: bool = False
    # Connection-establishment timeout; there is deliberately no read/total timeout,
    # so slow-but-alive episodes are never severed mid-flight.
    # rollout_timeout_s (opt-in) is the wall-clock bound; discards slow episodes' training signals.
    http_connect_timeout_s: float = Field(default=60.0, gt=0)
    rollout_timeout_s: float | None = Field(default=None, gt=0)
    http_retries: int = Field(default=3, ge=1)
    # Bounded connect retries: give up at failure_cap so the pipeline slot frees as a placeholder.
    connect_failure_cap: int = Field(default=60, ge=1)
    connect_retry_wait_s: float = Field(default=5.0, ge=0)
    # One warning per process when curriculum sampling degrades to uniform.
    _curriculum_fallback_logged: bool = PrivateAttr(False)
    # Next curriculum row, counted in prompt groups over the whole run.
    _curriculum_cursor: int = PrivateAttr(0)
    # This agent's share of each collection's prompt groups; WeightedMultiTask stamps it.
    _prompt_share: float = PrivateAttr(1.0)
    # Engine uids recorded by evaluation episodes since run_evaluation last reported them.
    _evaluation_completion_ids: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.dataset = self.load_dataset()
        if self.env_id is None:
            self.env_id = self.agent_name

    def load_dataset(self):
        """Load the JSONL dataset, expanding `${VAR}` placeholders in the path."""
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
        # Rollout collection only runs on one rank; DP peers must not advance the cursor.
        try:
            from megatron.training.global_vars import get_args, get_run_config, get_train_state

            rl_config = get_run_config().rl
            collections = curriculum_collections(
                get_train_state().iteration,
                rl_config.grpo_prompts_per_step,
                rl_config.grpo_group_size,
                rl_config.grpo_iterations,
                get_args().global_batch_size,
            )
        except (ImportError, AssertionError, AttributeError) as e:
            if not self._curriculum_fallback_logged:
                self._curriculum_fallback_logged = True
                logger.warning(
                    "ResponsesEnvAgent[%s]: curriculum_sampling is enabled but no training "
                    "state is available (%s: %s); falling back to uniform sampling.",
                    self.agent_name,
                    type(e).__name__,
                    e,
                )
            return self.dataset[np.random.randint(0, len(self.dataset))]

        # A restart seeds the cursor from the collections this agent already consumed; from
        # there it only moves forward.
        restart_seed = int(collections * rl_config.grpo_prompts_per_step * self._prompt_share)
        index = max(self._curriculum_cursor, restart_seed)
        self._curriculum_cursor = index + 1
        return self.dataset[index % len(self.dataset)]

    def _placeholder(
        self,
        failure_reason: str,
        problem_id: str | None,
        completion_ids: list[str],
        rollout_status: str = 'placeholder',
    ) -> TokenRollout:
        """Empty-trajectory rollout padding a failed episode, naming the calls it made."""
        return TokenRollout(
            trajectory=[],
            generation_mask=[],
            reward=0.0,
            logprobs=[],
            env_id=self.env_id,
            problem_id=problem_id,
            completion_ids=completion_ids,
            rollout_status=rollout_status,
            failure_reason=failure_reason,
        )

    def build_token_rollout(
        self,
        result: RunResult,
        problem_id: str | None,
        keys: EpisodeKeys | None,
        generation_cap: int | None = None,
    ) -> TokenRollout:
        """Convert a /run result and its manifest keys into a TokenRollout for the ledger join.

        The rollout carries no tokens: `completion_ids` are the terminal chain's engine uids
        and rank 0 rebuilds the turns from the finished-request ledger. `generation_cap` is the
        effective per-turn cap stamped on the wire. Subclasses may override to post-process the
        built rollout (e.g. reward shaping).
        """
        if keys is None:
            # Nothing is known about the episode's requests; see _episode_keys for the warning.
            return self._placeholder('manifest_unavailable', problem_id, [])
        if keys.failure_reason is not None:
            logger.warning(
                "Env episode is not trainable (capture %s, terminal selection %s): "
                "problem_id=%s env_id=%s reward=%s; padding with a placeholder.",
                keys.failure_reason,
                keys.terminal_selection,
                problem_id,
                self.env_id,
                result.reward,
            )
            return self._placeholder(keys.failure_reason, problem_id, keys.all_keys)

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
            return self._placeholder(
                failure_reason, problem_id, keys.all_keys, rollout_status='masked'
            )

        return TokenRollout(
            trajectory=None,
            generation_mask=None,
            logprobs=None,
            reward=result.reward,
            env_id=self.env_id,
            problem_id=problem_id,
            completion_ids=keys.turns,
            retired_completion_ids=keys.others,
            generation_cap=generation_cap,
            rollout_status='graded' if result.final_eval_time is not None else 'ok',
        )

    def _control_headers(self) -> dict[str, str]:
        token = os.environ.get(self.capture_control_token_env)
        if not token:
            raise ValueError(
                f"ResponsesEnvAgent[{self.agent_name}]: the capture control bearer token is not "
                f"set in ${self.capture_control_token_env}; the policy model server requires it."
            )
        return {'Authorization': f"Bearer {token}"}

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

    async def _post_run(self, payload: dict[str, Any]) -> RunResult:
        """POST /run, re-posting on connection failures up to the cap."""
        client = self._http_client()
        url = f"http://{self.env_server_host_port}/run"
        consecutive_failures = 0
        while True:
            try:
                response = await client.post(url, json=payload)
                body = response.text
                if response.status_code >= 400:
                    detail = body.strip() or response.reason_phrase or 'empty response body'
                    raise EnvHTTPError(response.status_code, detail[:4000])
                return RunResult.model_validate_json(body)
            except httpx.TransportError as e:
                # Connect refused/reset/timeout and mid-body read errors all land here:
                # re-POST and give up at the cap.
                consecutive_failures += 1
                if consecutive_failures >= self.connect_failure_cap:
                    raise EnvConnectExhausted(
                        f"{consecutive_failures} consecutive connection failures posting to "
                        f"{url}: {type(e).__name__}: {e}"
                    ) from e
                await asyncio.sleep(self.connect_retry_wait_s)

    async def _post_run_with_http_retry(self, payload: dict[str, Any]) -> RunResult:
        """POST /run with the 5xx/timeout retry budget shared by rollouts and eval.

        Raises the final error once retries are exhausted (immediately for
        non-retryable failures); callers decide value-vs-fatal semantics.
        """
        for attempt in range(self.http_retries):
            try:
                # rollout_timeout_s bounds each attempt; None waits without a limit.
                return await asyncio.wait_for(self._post_run(payload), self.rollout_timeout_s)
            except EnvHTTPError as e:
                logger.warning("/run HTTP %s: %s", e.status, e.detail)
                if e.status < 500 or attempt == self.http_retries - 1:
                    raise
            except asyncio.TimeoutError:
                logger.warning(
                    "/run attempt %s/%s timed out after %ss",
                    attempt + 1,
                    self.http_retries,
                    self.rollout_timeout_s,
                )
                if attempt == self.http_retries - 1:
                    raise
            wait_s = 5 * (attempt + 1)
            logger.warning(
                "retrying /run in %ss (attempt %s/%s)", wait_s, attempt + 1, self.http_retries
            )
            await asyncio.sleep(wait_s)
        raise RuntimeError("exhausted retry loop")  # unreachable: http_retries >= 1

    async def _control(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """One bearer-authenticated call to the capture control routes."""
        response = await self._http_client().request(
            method,
            f"http://{self.capture_control_host_port}{CAPTURE_CONTROL_PREFIX}{path}",
            headers=self._control_headers(),
            timeout=httpx.Timeout(self.capture_control_timeout_s),
            **kwargs,
        )
        if response.status_code >= 400:
            detail = response.text.strip() or response.reason_phrase or 'empty response body'
            raise EnvHTTPError(response.status_code, detail[:4000])
        return response.json()

    async def _episode_keys(
        self, rollout_id: str, result: RunResult | None, problem_id: str | None
    ) -> EpisodeKeys | None:
        """Read and retire the episode's manifest; None when it could not be read.

        The manifest is read even for a failed /run: whatever Gym recorded still has ledger
        records to drain. Retiring fences late writes and frees the ledger entry.
        """
        try:
            manifest = await self._control('GET', f"/rollouts/{rollout_id}/manifest")
        except (EnvHTTPError, httpx.HTTPError, ValueError) as e:
            logger.warning(
                "ResponsesEnvAgent[%s]: manifest of rollout %s unavailable (%s: %s); its "
                "requests' ledger records cannot be claimed: problem_id=%s",
                self.agent_name,
                rollout_id,
                type(e).__name__,
                e,
                problem_id,
            )
            return None
        keys = episode_keys_from_manifest(
            manifest,
            scored_response_id=None if result is None else result.response.id,
            declared_response_id=None if result is None else result.terminal_response_id,
        )
        try:
            await self._control('POST', '/rollouts/retire', json={'rollout_ids': [rollout_id]})
        except (EnvHTTPError, httpx.HTTPError, ValueError) as e:
            logger.warning(
                "ResponsesEnvAgent[%s]: could not retire rollout %s on the policy server (%s: %s)",
                self.agent_name,
                rollout_id,
                type(e).__name__,
                e,
            )
        return keys

    async def get_rollout_response(
        self,
        request: RolloutRequest | GroupedRolloutRequest | EvaluationRequest,
        inference_request: InferenceRequest,
    ) -> InferenceResponse:
        """Run one /run request for the pipeline's infer stage; failures travel as values."""
        assert isinstance(inference_request, EnvRunRequest)
        payload = inference_request.payload
        rollout_id = payload.get(ROLLOUT_ID_KEY)
        try:
            return EnvRunResponse(
                result=await self._post_run_with_http_retry(payload), rollout_id=rollout_id
            )
        except EnvHTTPError as e:
            return EnvRunResponse(
                failure_reason=f"http_{e.status}", failure_detail=e.detail, rollout_id=rollout_id
            )
        except asyncio.TimeoutError as e:
            return EnvRunResponse(
                failure_reason='rollout_timeout', failure_detail=str(e), rollout_id=rollout_id
            )
        except Exception as e:
            logger.exception("get_rollout_response failed; padding with a placeholder")
            return EnvRunResponse(
                failure_reason=type(e).__name__, failure_detail=str(e), rollout_id=rollout_id
            )

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

    def _run_payload(self, generation_args: GenericGenerationArgs, prompt: dict) -> dict:
        """The /run body shared by a group's episodes; each episode adds its own capture key."""
        return self._stamp_output_caps(generation_args_to_run_payload(generation_args, prompt))

    def _episode_request(self, payload: dict) -> EnvRunRequest:
        """One episode's /run request: the shared body under a fresh capture key."""
        return EnvRunRequest(payload={**payload, ROLLOUT_ID_KEY: new_rollout_id(self.env_id)})

    async def prepare_group_rollout(self, request: GroupedRolloutRequest) -> GroupRolloutParams:
        prompt = await self.get_prompt(validation=request.validation)
        payload = self._run_payload(request.generation_args, prompt)
        # The cap actually sent for this episode (row cap merged with the agent cap);
        # the rollout must self-report this value or the trainer's termination check breaks.
        effective_cap = payload['responses_create_params'].get('max_output_tokens')
        problem_id = prompt.get('problem_id')

        # The pipeline calls run_episode once per rollout of the group with these same params: the
        # server runs each episode in one /run call, and each episode gets its own capture key.
        async def run_episode() -> EpisodeResult:
            response = await self.get_rollout_response(request, self._episode_request(payload))
            return EpisodeResult(responses=[response], conversation=[])

        async def build_rollout(episode: EpisodeResult) -> TokenRollout:
            response = episode.responses[-1]
            assert isinstance(response, EnvRunResponse) and response.rollout_id
            keys = await self._episode_keys(response.rollout_id, response.result, problem_id)
            if response.result is None:
                logger.warning(
                    "prepare_group_rollout: padding failed episode with placeholder for "
                    "env=%s: %s %s",
                    self.env_id,
                    response.failure_reason,
                    response.failure_detail,
                )
                return self._placeholder(
                    response.failure_reason or 'unknown',
                    problem_id,
                    [] if keys is None else keys.all_keys,
                )
            return self.build_token_rollout(
                response.result, problem_id, keys, generation_cap=effective_cap
            )

        return GroupRolloutParams(run_episode=run_episode, build_rollout=build_rollout)

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        """N independent rollouts composed from the group-rollout primitives."""

        async def _single_rollout() -> TokenRollout:
            params = await self.prepare_group_rollout(request)
            return await params.build_rollout(await params.run_episode())

        return list(await asyncio.gather(*[_single_rollout() for _ in range(request.num_rollouts)]))

    async def _evaluation(
        self, prompt: dict, golden: dict | None, request: EvaluationRequest
    ) -> EvaluationResponse[RewardEvaluationResult]:
        episode = self._episode_request(self._run_payload(request.generation_args, prompt))
        rollout_id = episode.payload[ROLLOUT_ID_KEY]
        problem_id = prompt.get('problem_id')
        # Same transport and HTTP-retry classification as training rollouts;
        # terminal failures propagate and run_evaluation turns them into dropped prompts.
        try:
            result = await self._post_run_with_http_retry(episode.payload)
        finally:
            # Evaluation trains on nothing, but its requests were ledgered like any other:
            # every recorded key is reported so rank 0 can pop the records. A failed episode
            # has no result to carry its keys; run_evaluation reports those.
            keys = await self._episode_keys(rollout_id, None, problem_id)
            completion_ids = [] if keys is None else keys.all_keys
            self._evaluation_completion_ids.extend(completion_ids)
        return run_result_to_evaluation_response(result, problem_id, self.env_id, completion_ids)

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        """Pass@k-evaluate the leading dataset rows against the environment server."""
        assert request.rank_info is None, (
            "ResponsesEnvAgent evaluates the full prompt set; rank_info distribution "
            "is not supported"
        )
        all_prompts = self.dataset[: request.num_prompts]
        if not all_prompts:
            raise ValueError(
                f"ResponsesEnvAgent[{self.agent_name}]: no evaluation prompts "
                f"(num_prompts={request.num_prompts}, dataset rows={len(self.dataset)})."
            )
        if request.num_prompts > len(self.dataset):
            logger.warning(
                "run_evaluation[%s]: dataset has %d rows; evaluating all of them instead of the "
                "requested %d.",
                self.env_id,
                len(self.dataset),
                request.num_prompts,
            )
        results = await asyncio.gather(
            *[self.evaluation(prompt, prompt, request) for prompt in all_prompts],
            return_exceptions=True,
        )
        kept, failed = [], []
        for prompt, result in zip(all_prompts, results):
            if isinstance(result, BaseException):
                failed.append(prompt.get('problem_id'))
            else:
                kept.append(result)
        if failed:
            logger.warning(
                "run_evaluation[%s]: dropping %d/%d prompts after exhausted retries: %s",
                self.env_id,
                len(failed),
                len(results),
                failed,
            )
        if not kept:
            raise RuntimeError(
                f"run_evaluation[{self.env_id}]: 0 usable eval results; all "
                f"{len(results)} prompts failed."
            )
        response = type(kept[0])(
            results=sum([result.results for result in kept], []), env_id=self.env_id
        )
        # A dropped prompt has no result to carry its episodes' recorded keys, and rank 0 reads
        # keys only off results: the first kept result reports them.
        recorded, self._evaluation_completion_ids = self._evaluation_completion_ids, []
        reported = {cid for result in response.results for cid in result.completion_ids}
        response.results[0].completion_ids.extend(cid for cid in recorded if cid not in reported)
        return response
