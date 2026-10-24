# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.contexts.dynamic_context import (
    BlockOverflowError,
    MaxSequenceLengthOverflowError,
    PromptPreparationError,
    TokenOverflowError,
)
from megatron.core.inference.inference_request import (
    RESUMED_GENERATION_FIELD,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    LiveRequestSnapshot,
    ResumedGeneration,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import internal_api, nvtx_range_pop, nvtx_range_push

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Engine-owned control field written into ``offload_params`` on MP rank 0 when the
# RequestPromptPreparer fails, or by ``_admit_request`` for a resumed generation it
# cannot honour, and read back by ``_add_request`` which fails the request
# on every rank. The HTTP endpoints reject client-supplied ``offload_params`` keys
# that start with ``_`` (see ``endpoints.common.validate_offload_params``), so
# this namespace cannot be forged from a request body.
_PROMPT_PREPARATION_ERROR_FIELD = "_request_prompt_preparation_error"


def _weight_scoped_salt(weight_epoch: int, media_cache_key: Optional[str]) -> Optional[str]:
    """Scope a request's block hashes to the weight generation that will serve it.

    Under KVCacheManagementMode.PERSIST the prefix cache survives a refit:
    reinitialize_inference_state_buffers() only resets metadata on the RECOMPUTE
    path, so the KV and Mamba hash tables outlive suspend/resume and a request
    admitted afterwards can match blocks whose KV the previous weights computed.
    Staleness is then bounded only by eviction pressure, since the engine-side
    allocator has no TTL.

    Mixing the weight generation into the salt makes chains from different
    generations disjoint, so stale blocks become unmatchable rather than being
    freed -- nothing is mutated at refit time, leaving live requests, chunked
    prefill and pending Mamba restores untouched.

    Composed with the media key rather than replacing it: multimodal requests
    still need equal token placeholders backed by different media to stay
    unmatchable. Epoch 0 returns the media key unchanged, so an engine that never
    resumes hashes exactly as before.
    """
    if weight_epoch == 0:
        return media_cache_key
    if media_cache_key is None:
        return f"w{weight_epoch}"
    return f"w{weight_epoch}\x00{media_cache_key}"


@dataclass(kw_only=True)
class RequestEntry:
    """Entry in the engine's `self.requests` dict."""

    record: DynamicInferenceRequestRecord
    future: asyncio.Future[DynamicInferenceRequest]


@internal_api
# pylint: disable=line-too-long
class RequestIntakeMixin:
    """Request admission and prefill scheduling for `DynamicInferenceEngine`."""

    def _finalize_request(self, request_entry: RequestEntry) -> DynamicInferenceRequest:
        """Record the end of a request whose status is terminal and merge its record."""
        request = request_entry.record[-1]
        if request.status == Status.COMPLETED:
            request.add_event_finish()
            self.finished_request_count += 1
            self._metrics_counters["requests_finished_total"] += 1
        else:
            assert (
                request.status == Status.FAILED
            ), f"Request {request.request_id} ended in non-terminal status {request.status}."
            request.add_event_fail()
        return request_entry.record.merge()

    def _complete_request(
        self,
        request_entry: RequestEntry,
        finished_request: Optional[DynamicInferenceRequest] = None,
        *,
        reply_now: bool = False,
    ) -> DynamicInferenceRequest:
        """Resolve a request future, finalizing its record unless already finalized.

        Args:
            request_entry: Engine-owned record and completion future.
            finished_request: Result finalized before context cleanup, when available.
            reply_now: Send the result to the coordinator now. Used for a request that ends outside
                a step: one that failed before it ran a step, a resumed generation that arrived
                finished, or a KV import already satisfied by its transferred token.
        """
        assert not request_entry.future.done(), "Request future was already resolved."
        if finished_request is None:
            finished_request = self._finalize_request(request_entry)
        request_entry.future.set_result(finished_request)
        self.context.remove_vlm_request_data(finished_request.request_id)
        if reply_now and self.use_coordinator and self.is_mp_coordinator:
            self._send_requests_to_coordinator([finished_request])
        return finished_request

    def _handle_failed_request(self, request_id: int, *, stage: str = "be added to the engine"):
        """Handle a failed request by sending the reply immediately.

        The request is added to failed_request_ids so that the next bookkeeping pass can return it.

        Args:
            request_id (int): The failed request.
            stage (str): What the request failed to do, for the operator warning.
        """
        request_entry = self.requests[request_id]
        request = request_entry.record[-1]

        if self.rank == 0:
            errors = [
                e.payload
                for e in request.events
                if e.type
                in (
                    DynamicInferenceEventType.ERROR_NONTRANSIENT,
                    DynamicInferenceEventType.ERROR_TRANSIENT,
                )
            ]
            errors_str = (
                "; ".join(f"{type(e).__name__}: {e}" for e in errors) if errors else "unknown error"
            )
            warnings.warn(
                f"Request {request_id} failed to {stage} ({errors_str}). "
                f"Prompt Tokens: {len(request.prompt_tokens)} "
                f"Tokens to generate: {request.sampling_params.num_tokens_to_generate} "
                f"Max sequence length: {self.context.max_sequence_length} "
                f"Chunked prefill enabled: {self.enable_chunked_prefill}"
            )

        request.status = Status.FAILED
        self.failed_request_ids.append(request_id)
        self._complete_request(request_entry, reply_now=True)

    def _fail_submission(
        self, request_id: int, sampling_params: Optional[SamplingParams], exc: BaseException
    ) -> None:
        """Register a minimal failed request so a rejected admission still
        produces a client-visible failure reply.

        Called from the SUBMIT_REQUEST handler when image preprocessing or
        add_request raises. Registering a placeholder record with
        Status.FAILED lets ``_handle_failed_request`` publish the ENGINE_REPLY
        without leaving the client hanging or killing the engine loop.
        """
        if self.rank == 0:
            warnings.warn(
                f"Request {request_id} rejected before admission: " f"{type(exc).__name__}: {exc}"
            )
        # Empty prompt tokens are safe — the reply short-circuits at
        # Status.FAILED and the client sees the failure, not a completion.
        placeholder_request = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.empty(0, dtype=torch.int64),
            sampling_params=(sampling_params or SamplingParams()),
        )
        placeholder_request.status = Status.FAILED
        self.requests[request_id] = RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(placeholder_request),
            future=self._loop.create_future(),
        )
        self._handle_failed_request(request_id)

    def _collect_failed_requests(
        self, request_ids: Optional[set[int]] = None
    ) -> List[DynamicInferenceRequest]:
        """Remove and return a snapshot of synchronously failed requests.

        Args:
            request_ids: Optional ownership filter. Failed requests outside this
                set remain queued for their caller.

        Returns:
            Failed requests selected from the current queue snapshot.
        """
        failed_request_ids, self.failed_request_ids = self.failed_request_ids, []
        failed_requests = []
        for failed_request_id in failed_request_ids:
            if request_ids is not None and failed_request_id not in request_ids:
                self.failed_request_ids.append(failed_request_id)
                continue
            failed_entry = self.requests.pop(failed_request_id)
            assert (
                failed_entry.future.done()
            ), f"Failed request {failed_request_id} future has not been properly resolved."
            failed_requests.append(failed_entry.future.result())
        return failed_requests

    async def snapshot_live_requests(self) -> List[LiveRequestSnapshot]:
        """Snapshot every non-failed request for later resumption; must run on the
        engine loop whenever that loop is running (any loop may run it while the
        engine loop is stopped, as between a suspend and a resume)."""
        if self._loop.is_running() and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError(
                "snapshot_live_requests must run on the engine loop while it is running"
            )
        # A stop the engine has detected but has yet to complete sits on the record
        # as its finish reason, so the record alone says whether a request ended.
        snapshots = []
        for entry in self.requests.values():
            request = entry.record[-1]
            if request.status == Status.FAILED:
                continue
            snapshots.append(
                LiveRequestSnapshot.from_record(
                    entry.record,
                    # The blocks hold the routing of every token that ran forward; a request
                    # still in prefill has none that is complete.
                    routing_indices=(
                        self._routing_from_blocks(request) if request.generated_tokens else None
                    ),
                )
            )
        return snapshots

    def _routing_from_blocks(self, request: DynamicInferenceRequest) -> Optional[np.ndarray]:
        """A resident request's expert routing, read from its blocks as its completion reads it."""
        routing_replay = self.context.kv_block_allocator.routing_replay
        if not routing_replay.has_data():  # routing replay off: nothing was recorded
            return None
        if not hasattr(self.context, "request_ids"):
            return None
        index = self.context.get_index_of_request(request.request_id)
        if index < 0:
            return None
        total_tokens = len(request.prompt_tokens) + len(request.generated_tokens)
        return routing_replay.reconstruct_routing_from_blocks(
            self.context.get_committed_kv_block_ids(index), total_tokens - 1
        )

    def has_unfinished_requests(self) -> bool:
        """Test if context contains unfinished requests."""
        return self.context.has_unfinished_requests() or len(self.waiting_request_ids) > 0

    def get_request(self, request_id: int) -> DynamicInferenceRequest:
        """Get most recent request from a request record.

        Args:
            request_id (int): Request id.

        Returns:
            (DynamicInferenceRequest) The most recent request in the record.
        """
        return self.requests[request_id].record[-1]

    def _resolve_generation_budget(self, request: DynamicInferenceRequest) -> bool:
        """Resolve and clamp the generation budget; False when the request overflows the engine."""
        params = request.sampling_params
        remaining = self.context.max_sequence_length - len(request.prompt_tokens)
        if params.num_tokens_total is not None:
            params.num_tokens_to_generate = params.num_tokens_total - len(request.prompt_tokens)
            params.num_tokens_total = None
        elif params.num_tokens_to_generate is None:
            params.num_tokens_to_generate = remaining
        if params.num_tokens_to_generate < 0 or remaining < 0:
            return False
        if params.num_tokens_to_generate > remaining:
            # Clamp instead of rejecting, for compatibility with other frameworks.
            if self.rank == 0:
                warnings.warn(
                    f"Request {request.request_id} requested "
                    f"num_tokens_to_generate={params.num_tokens_to_generate} which exceeds the "
                    f"maximum sequence length of the engine. Clamping num_tokens_to_generate to "
                    f"{remaining}."
                )
            params.num_tokens_to_generate = remaining
        return True

    def _add_request(
        self,
        request: DynamicInferenceRequest,
        *,
        is_resume: bool = False,
        record: Optional[DynamicInferenceRequestRecord] = None,
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Add a request to the engine.

        Args:
            request (DynamicInferenceRequest): Request to add.
            is_resume (bool): Whether the newest segment of a record is being re-admitted,
                as after suspend/resume, rather than a new record opened.
            record (Optional[DynamicInferenceRequestRecord]): With `is_resume`, the record of
                a generation resumed from another engine.

        Returns:
            asyncio.Future[DynamicInferenceRequest]: Future completed when the request finishes.
        """

        assert record is None or is_resume, "A record is registered only on resume."
        request_id = request.request_id
        if is_resume:
            if record is not None:
                assert (
                    request_id not in self.requests
                ), f"Request ID {request_id} is already active."
                self.requests[request_id] = RequestEntry(
                    record=record, future=self._loop.create_future()
                )
            if request_id not in self.requests or self.get_request(request_id) is not request:
                raise ValueError(f"Cannot resume unknown request ID {request_id}.")
        elif request_id in self.requests:
            raise ValueError(f"Request ID {request_id} is already active.")
        else:
            self.requests[request_id] = RequestEntry(
                record=DynamicInferenceRequestRecord.from_request(request),
                future=self._loop.create_future(),
            )
            request.add_event_add_engine()  # Record when request enters engine

            # Stamp new request with the current generation epoch.
            if self._generation_epoch is not None:
                epoch = self._generation_epoch
                request.policy_epoch = [(0, epoch)]
                request.kv_cache_epoch = [(0, epoch)]

        if request.status is None:
            request.status = Status.ACTIVE_AND_GENERATING_TOKENS

        seed_error = None
        if request.sampling_params.seed is not None:
            if self.context.config.sampling_backend != "torch":
                seed_error = "Request-local seeds require sampling_backend='torch'"
            elif self.context.config.num_speculative_tokens:
                seed_error = "Request-local seeds do not yet support speculative decoding"
            elif "seed" not in self.context.request_metadata:
                seed_error = "Request-local seeds require seed in request metadata"
        if seed_error is not None:
            request.status = Status.FAILED
            request.add_event_error_nontransient(ValueError(seed_error))

        assert (
            request.sampling_params.num_tokens_to_generate is None
            or request.sampling_params.num_tokens_total is None
        )
        if request.sampling_params.top_n_logprobs > 0:
            assert (
                request.sampling_params.return_log_probs
            ), "top_n_logprobs requires sampling_params.return_log_probs to be True"
        if (
            request.sampling_params.return_log_probs
            and not request.sampling_params.skip_prompt_log_probs
        ):
            assert not self.materialize_only_last_token_logits, (
                "Prompt log probs cannot be calculated if only last token logits are materialized. "
                "Set materialize_only_last_token_logits to False in DynamicInferenceContext "
                "or skip_prompt_log_probs to True in SamplingParams."
            )

        error = (
            request.offload_params.get(_PROMPT_PREPARATION_ERROR_FIELD)
            if isinstance(request.offload_params, dict)
            else None
        )
        if error is not None:
            request.status = Status.FAILED
            request.add_event_error_nontransient(PromptPreparationError(request_id, error))
        if not self._resolve_generation_budget(request):
            request.status = Status.FAILED
            request.add_event_error_nontransient(MaxSequenceLengthOverflowError(request_id))
        if request.sampling_params.termination_id is None:
            try:
                eod = self.controller.tokenizer.eod
            except AttributeError:
                if self.rank == 0:
                    warnings.warn(
                        "Termination ID not specified, and tokenizer does not define eod."
                        "Defaulting to not using termination id."
                    )
                eod = -1
            request.sampling_params.termination_id = eod

        if len(request.prompt_tokens) > self.context.max_tokens and not self.enable_chunked_prefill:
            request.status = Status.FAILED
            request.add_event_error_nontransient(TokenOverflowError(request_id))

        # Check that the shared KV pool has enough blocks for this request's stored tokens:
        # the prompt, all generated tokens but the last, and the final decode step's drafts.
        max_stored_tokens = len(request.prompt_tokens)
        if request.sampling_params.num_tokens_to_generate > 1:
            max_stored_tokens += (
                request.sampling_params.num_tokens_to_generate
                - 1
                + self.context.num_speculative_tokens
            )
        request_block_count = math.ceil(max_stored_tokens / self.context.block_size_tokens)
        usable_blocks = self.context.kv_block_allocator.pool_size - 1
        if request_block_count > usable_blocks:
            request.status = Status.FAILED
            request.add_event_error_nontransient(BlockOverflowError(request_id))

        # Tokenize stop words if provided
        if request.sampling_params.stop_words:
            stop_word_ids = [
                self.controller.tokenize_prompt(self.controller.tokenizer, stop_word, add_BOS=False)
                for stop_word in request.sampling_params.stop_words
            ]
            request.stop_word_ids = stop_word_ids

        if request.status != Status.FAILED:
            self.waiting_request_ids.append(request_id)
        else:
            self._handle_failed_request(request_id)

        return self.requests[request_id].future

    def add_request(
        self,
        request_id: int,
        prompt: Union[str, List[int], Tensor],
        sampling_params: Optional[SamplingParams] = None,
        precomputed_block_hashes: Optional[List[int]] = None,
        *,
        imgs: Optional[Tensor] = None,
        num_tiles: Optional[Tensor] = None,
        num_img_embeddings_per_tile: int = 0,
        imgs_sizes: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        video_frame_indices: Optional[List[List[int]]] = None,
        video_fps: Optional[List[float]] = None,
        media_tokens_preexpanded: bool = False,
        offload_params: Optional[Dict] = None,
        media_cache_key: Optional[str] = None,
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Add request to inference context.

        Supports both text-only and multimodal requests. For text-only, call
        with just (request_id, prompt, sampling_params). For multimodal, also
        pass imgs and either (num_tiles + num_img_embeddings_per_tile) for static
        resolution or imgs_sizes for dynamic resolution.

        When multimodal kwargs are provided the method will:
        1. Expand compact media tokens, or derive a mask from pre-expanded tokens.
        2. Run the vision encoder to produce image embeddings.
        3. Store the embeddings and mask in the context for later use by the
           controller's forward step.

        Args:
            request_id (int): Unique ID of request.
            prompt (Union[str, Tensor]): Prompt as either a text string or token IDs.
            sampling_params (Optional[SamplingParams]): Sampling parameters for the request.
            precomputed_block_hashes (Optional[List[int]]): Prefix-cache hashes already
                computed for the prompt's complete blocks. Values must match
                ``compute_block_hashes_batched(prompt_tokens, block_size_tokens)``.
            imgs (Optional[Tensor]): Image tensor [num_tiles, C, H, W] or
                [1, total_patches, patch_features] (or None).
            num_tiles (Optional[Tensor]): Number of tiles per image (1-D tensor, or None).
                Static resolution.
            num_img_embeddings_per_tile (int): Number of image embeddings per tile.
                Static resolution.
            imgs_sizes (Optional[Tensor]): Per-image sizes [N, 2] with [H, W].
                Dynamic resolution.
            num_frames (Optional[Tensor]): Number of frames per image/video item.
            video_frame_indices (Optional[List[List[int]]]): Source indices for
                each sampled video frame.
            video_fps (Optional[List[float]]): Source FPS for each video item.
            media_tokens_preexpanded (bool): Whether prompt token IDs already contain
                one model token per projected media embedding.
            offload_params (Optional[Dict]): Opaque metadata forwarded to the payload stager.
            media_cache_key (Optional[str]): Media identity computed by the submitting
                inference client. Direct callers may omit it and let the engine derive
                an identity from the resolved media tensors.

        Return:
            Returns an asyncio `Future[DynamicInferenceRequest]` for the user to wait on.
        """
        if request_id in self.requests:
            raise ValueError(f"Request ID {request_id} is already active.")
        if sampling_params is None:
            sampling_params = SamplingParams()

        input_modalities = ["text"]
        cached_vision_entry = self._get_cached_vision_entry(media_cache_key)
        if cached_vision_entry is not None:
            input_modalities.append(cached_vision_entry.modality)
        elif num_frames is not None:
            input_modalities.append("video")
        elif imgs is not None:
            input_modalities.append("image")
        self.controller.inference_wrapped_model.validate_input_modalities(*input_modalities)

        prompt_str = None
        # Tokenize prompt if text. Move prompt tokens to the CUDA device for inference.
        if isinstance(prompt, str):
            # Tokenize prompt if text. Support legacy single-arg mocks.
            prompt_str = prompt
            try:
                prompt_token_ids = self.controller.tokenize_prompt(
                    self.controller.tokenizer, prompt, sampling_params.add_BOS
                )
            except TypeError:
                prompt_token_ids = self.controller.tokenize_prompt(
                    self.controller.tokenizer, prompt
                )
            tokens = torch.tensor(
                prompt_token_ids, dtype=torch.int64, device=torch.cuda.current_device()
            )
        elif isinstance(prompt, list):
            # Convert List[int] -> Tensor.
            tokens = torch.tensor(prompt, dtype=torch.int64, device=torch.cuda.current_device())
        elif isinstance(prompt, torch.Tensor):
            # Prompt already tokenized.
            assert prompt.dtype == torch.int64, prompt.dtype
            assert prompt.device == torch.device(
                f"cuda:{torch.cuda.current_device()}"
            ), prompt.device
            tokens = prompt

        else:
            raise Exception("specialize for <%s>." % type(prompt).__name__)

        if (
            imgs is not None
            or num_tiles is not None
            or num_img_embeddings_per_tile != 0
            or imgs_sizes is not None
            or num_frames is not None
            or media_cache_key is not None
        ):
            nvtx_range = "megatron.inference.multimodal.build_vlm_request"
            nvtx_range_push(nvtx_range)
            try:
                request = self._build_vlm_request(
                    request_id=request_id,
                    prompt_str=prompt_str,
                    tokens=tokens,
                    sampling_params=sampling_params,
                    imgs=imgs,
                    num_tiles=num_tiles,
                    num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                    imgs_sizes=imgs_sizes,
                    precomputed_block_hashes=precomputed_block_hashes,
                    num_frames=num_frames,
                    video_frame_indices=video_frame_indices,
                    video_fps=video_fps,
                    media_tokens_preexpanded=media_tokens_preexpanded,
                    offload_params=offload_params,
                    media_cache_key=media_cache_key,
                )
            finally:
                nvtx_range_pop(nvtx_range)
            # _build_vlm_request has already registered the image embeddings
            # and token mask into the context (add_vlm_request_data). If
            # admission now rejects the request (oversized prompt, cache
            # exhaustion, ...), those tensors would linger in the context
            # dicts and leak GPU memory. Clean them up on failure.
            try:
                return self._admit_request(request)
            except Exception:
                self.context.remove_vlm_request_data(request_id)
                raise
        else:
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt=prompt_str,
                prompt_tokens=tokens,
                sampling_params=sampling_params,
                offload_params=offload_params,
                block_size_tokens=self.context.block_size_tokens,
                enable_prefix_caching=self.context.enable_prefix_caching,
                precomputed_block_hashes=precomputed_block_hashes or [],
                # Text-only requests are the common case for a refit, so this is
                # the path the weight scoping exists for. Note that hashes handed
                # in by a caller are used as-is: __post_init__ only computes when
                # none were supplied, so a disaggregated handoff carries the
                # generation its sender hashed under.
                block_hash_salt=_weight_scoped_salt(self._weight_epoch, None),
            )
        return self._admit_request(request)

    def _admit_request(
        self, request: DynamicInferenceRequest
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Admit a built request, resuming the generation its preparer handed back, if any."""
        params = request.offload_params if isinstance(request.offload_params, dict) else {}
        wire = params.pop(RESUMED_GENERATION_FIELD, None)
        # New request path.
        if wire is None or _PROMPT_PREPARATION_ERROR_FIELD in params:
            return self._add_request(request)

        # Resumed request path.
        try:
            resumed = ResumedGeneration.from_wire(wire)
            record = (
                DynamicInferenceRequestRecord.from_resumed_generation(
                    request, resumed, generation_epoch=self._generation_epoch
                )
                if resumed.token_ids or resumed.finish_reason is not None
                else None  # nothing generated yet
            )
        except ValueError as invalid:
            # Refused the way rank zero refuses a prompt: _add_request fails the request.
            params[_PROMPT_PREPARATION_ERROR_FIELD] = str(invalid)
            return self._add_request(request)
        if record is None:  # admitted fresh, under the banked budget
            request.sampling_params.num_tokens_to_generate = resumed.num_tokens_to_generate
            request.sampling_params.num_tokens_total = None
            return self._add_request(request)

        if resumed.finish_reason is not None:  # ended on its first engine: nothing to schedule
            entry = RequestEntry(record=record, future=self._loop.create_future())
            self._complete_request(entry, reply_now=True)
            return entry.future
        return self._add_request(record[-1], is_resume=True, record=record)

    def get_prefix_coordination_metrics(self) -> dict:
        """Return prefix caching coordination metrics.

        Returns:
            Dict with coordination stats including the number of scheduling waits.
        """
        return {"waits": self._prefix_coordination_waits}

    def _mamba_batch_invariant_prefill_chunk_length(self, remaining: int, capacity: int) -> int:
        """Number of prompt tokens to compute this step, at most `capacity`.

        `remaining` counts the prompt tokens still to compute (excluding any
        cached prefix). Non-final calls must start and end at SSM chunk
        boundaries. The final prompt call may be shorter because it seeds the
        decode replay tail.
        """
        if capacity >= remaining:
            return remaining

        alignment = self.context.ssm_chunk_alignment
        computed_tokens = (capacity // alignment) * alignment
        if remaining - computed_tokens == 1:
            computed_tokens -= alignment
        if computed_tokens <= 0:
            return 0
        return computed_tokens

    def schedule_waiting_requests(self) -> None:
        """Try to schedule requests from the waiting pool."""
        # Waiting image requests whose admission deferred the vision encode are encoded
        # together here, before any of them can enter the context.
        self._encode_waiting_images()
        # Keep track of which requests get scheduled.
        waiting_before = set(self.waiting_request_ids)
        if self.enable_chunked_prefill:
            self.schedule_chunked_prefill()
        else:
            self.schedule_non_chunked_prefill()
        waiting_after = set(self.waiting_request_ids)

        # Re-stamp kv_cache_epoch on requests that were just scheduled.
        if self._generation_epoch is not None:
            for request_id in waiting_before - waiting_after:
                req = self.get_request(request_id)
                if req.kv_cache_epoch is None:
                    req.kv_cache_epoch = [(0, self._generation_epoch)]

    def _can_schedule_non_chunked_prefill(self, req, *, record_cg_wait: bool) -> bool:
        """Return whether the queue-head request can be admitted now.

        Args:
            req: Queue-head inference request.
            record_cg_wait (bool): Whether a CUDA-graph miss should update the
                request's wait counter.

        Returns:
            bool: Whether all request, token, KV-cache, and CUDA-graph checks pass.
        """
        if not all(self.context.check_availability(req)):
            return False

        if not self._cg_admission_gating_active():
            return True

        candidate = InferenceBatchDimensions(
            token_count=self.context.active_token_count + len(req.remaining_prompt_tokens),
            prefill_req_count=self.context.num_prefill_requests + 1,
            decode_req_count=self.context.num_decode_requests,
        )
        if record_cg_wait:
            return self._cg_admission_check(req, candidate)
        return self._matches_cg_admission(candidate)

    def _chunked_prefill_token_budget(self) -> int:
        """Return how many more prompt tokens this step can admit.

        The step capacity is `context.max_tokens`. Batch-invariant hybrid models also
        bound the prefill tokens alone by `context.max_prefill_tokens`, which leaves
        the rest of the capacity to this step's decode tokens.
        """
        context = self.context
        token_budget = context.max_tokens - context.active_token_count
        if context.max_prefill_tokens < context.max_tokens:
            assert context.batch_invariant_mode, (
                "max_prefill_tokens below max_tokens is set only for batch-invariant hybrid "
                "chunked prefill"
            )
            decode_token_count = context.num_decode_requests * (context.num_speculative_tokens + 1)
            prefill_token_count = context.active_token_count - decode_token_count
            token_budget = min(token_budget, context.max_prefill_tokens - prefill_token_count)
        return token_budget

    def _can_schedule_chunked_prefill(self, req) -> bool:
        """Return whether the queue-head request can admit at least one prompt token.

        Args:
            req: Queue-head inference request.

        Returns:
            bool: Whether request, token, and KV-cache capacity permit a chunk.
        """
        request_can_be_added, _, kv_cache_available = self.context.check_availability(
            req, prefill_chunk_length=1
        )
        is_continuing_chunk = self.context.chunked_prefill_request_id == req.request_id
        token_capacity_available = self._chunked_prefill_token_budget() > 0
        return (
            (is_continuing_chunk or request_can_be_added)
            and kv_cache_available
            and token_capacity_available
        )

    def _should_run_async_sched_overlap(self) -> bool:
        """Return whether this step should use overlap ordering.

        Returns:
            bool: Whether the next step can use overlap ordering.
        """
        # No-overlap also handles the first decode-only forward after prefill:
        # pending prefill output must be resolved before preparing its decode rows.
        # Paused requests and insufficient KV capacity likewise require complete
        # lifecycle bookkeeping before preparing the next batch.
        if not self.context.can_prepare_requests():
            return False
        if self.has_admittable_kv_import:
            return False
        if not self.waiting_request_ids:
            return True

        req = self.get_request(self.waiting_request_ids[0])
        if self.enable_chunked_prefill:
            return not self._can_schedule_chunked_prefill(req)
        return not self._can_schedule_non_chunked_prefill(req, record_cg_wait=False)

    def _pending_prefill_block_hashes(self) -> set[int]:
        """Return hashes reserved by work whose KV is not yet ready for reuse."""
        return set()

    def schedule_non_chunked_prefill(self) -> None:
        """Schedule non-chunked prefill requests."""
        prefix_caching_enabled = self.context.enable_prefix_caching
        if prefix_caching_enabled:
            pending_block_hashes = self._pending_prefill_block_hashes()
            pending_request_ids = []
        while self.waiting_request_ids:
            req = self.get_request(self.waiting_request_ids[0])

            # Check for conflicting block hashes.
            if prefix_caching_enabled:
                has_pending_hash = False
                for block_hash in req.precomputed_block_hashes:
                    if block_hash in pending_block_hashes:
                        has_pending_hash = True
                        break
                if has_pending_hash:
                    self._prefix_coordination_waits += 1
                    pending_request_ids.append(self.waiting_request_ids.popleft())
                    continue

            if self._can_schedule_non_chunked_prefill(req, record_cg_wait=True):
                # Add these hashes to pending.
                if prefix_caching_enabled:
                    for block_hash in req.precomputed_block_hashes:
                        if block_hash not in self.context.prefix_cache_registry.kv_hash_to_block_id:
                            pending_block_hashes.add(block_hash)
                self.context.add_request(req)
                self._loop.call_soon_threadsafe(
                    self._loop.create_task, self._notify_cond_for_new_request()
                )
                req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                req.add_event_add_context()
                self.waiting_request_ids.popleft()
            else:
                break

        # Prepend pending request ids to waiting queue.
        if prefix_caching_enabled and pending_request_ids:
            self.waiting_request_ids.extendleft(reversed(pending_request_ids))

    def _cg_admission_gating_active(self) -> bool:
        """Cudagraph-aware admission gating is active when --inference-cuda-graph-all-prefills
        is set, the engine has prefill/mixed CGs, and the batch-dim list is populated.

        All are required so legacy tests that exercise the scheduler without intending to run on
        captured graphs are unaffected. Gating is opt-in via `cuda_graph_all_prefills`.
        """
        return (
            self.cuda_graph_all_prefills
            and self.context.use_cuda_graphs_for_non_decode_steps
            and bool(self.context.cuda_graph_batch_dimensions_list)
        )

    def _find_cg_chunk_size(self, max_chunk_tokens: int) -> Optional[int]:
        """Return the largest chunk size <= max_chunk_tokens where batch matches a captured graph,
        or None if no graph covers any chunk in the budget.

        Walks the captured-CG list (sorted descending by token_count) and returns the largest chunk
        that falls within budget and produces an applicable batch_dim under the engine's matching
        mode (strict for hybrid models). Callers must explicitly handle the None case rather than
        scheduling eagerly.
        """
        active_tok = self.context.active_token_count
        active_p = self.context.num_prefill_requests
        active_d = self.context.num_decode_requests
        strict = self.context.is_hybrid_model

        best_chunk = None
        for cg in self.context.cuda_graph_batch_dimensions_list:
            chunk = min(max_chunk_tokens, cg.token_count - active_tok)
            if chunk < 1:
                continue
            candidate = InferenceBatchDimensions(
                token_count=active_tok + chunk,
                prefill_req_count=active_p + 1,
                decode_req_count=active_d,
            )
            if cg.is_applicable_for_batch_dim(candidate, strict=strict):
                if chunk == max_chunk_tokens:
                    return chunk
                best_chunk = max(best_chunk or 0, chunk)

        return best_chunk

    def _select_cg_chunk_size(
        self, req: DynamicInferenceRequest, max_chunk_tokens: int
    ) -> Optional[int]:
        """Select a graphed chunk size, or wait for the resident batch to drain.

        A deferred request remains at the queue head, so the resident batch can
        drain and change the graph shape. A lone in-flight continuation cannot
        defer, so it falls back to eager execution when no graph covers it.
        """
        chunk_size = self._find_cg_chunk_size(max_chunk_tokens)
        if chunk_size is not None:
            req.cg_wait_iters = 0
            return chunk_size

        if self.context.num_prefill_requests or self.context.num_decode_requests:
            self._register_cg_wait(req)
            return None

        if self.context.chunked_prefill_request_id == req.request_id:
            req.cg_wait_iters = 0
            return max_chunk_tokens

        raise RuntimeError(
            f"No captured CUDA graph can admit a {max_chunk_tokens}-token chunked prefill"
        )

    def _register_cg_wait(self, req) -> None:
        """Track a deferred admission attempt and throw a starvation warning at the threshold.

        Decode is bounded by the number of decode steps.
        Persistent waits past `_cg_admission_warn_after` consecutive steps signal a problem.
        """
        req.cg_wait_iters += 1
        if req.cg_wait_iters % self._cg_admission_warn_after == 0:
            logger.warning(
                "request %d has been deferred by CG-aware admission for %d steps — "
                "possible starvation (strict=%s, active P=%d D=%d tok=%d)",
                req.request_id,
                req.cg_wait_iters,
                self.context.is_hybrid_model,
                self.context.num_prefill_requests,
                self.context.num_decode_requests,
                self.context.active_token_count,
            )

    def _cg_admission_check(self, req, candidate: InferenceBatchDimensions) -> bool:
        """Return True if the candidate batch shape matches a captured cudagraph.

        On miss, registers a wait + warning via `_register_cg_wait`. On hit, resets the counter.
        Caller is responsible for breaking the scheduler loop on False.
        Passes match_ep_token_counts=False so this local admission probe doesn't force a per-attempt
        NCCL all-reduce — the step-time matcher does its own EP sync.

        Args:
            req: Request whose CUDA-graph wait state should be updated.
            candidate (InferenceBatchDimensions): Candidate batch after admission.

        Returns:
            bool: Whether a compatible captured graph exists.
        """
        if self._matches_cg_admission(candidate):
            req.cg_wait_iters = 0
            return True
        self._register_cg_wait(req)
        return False

    def _matches_cg_admission(self, candidate: InferenceBatchDimensions) -> bool:
        """Return whether a candidate batch matches a captured CUDA graph.

        Args:
            candidate (InferenceBatchDimensions): Candidate batch after admission.

        Returns:
            bool: Whether a compatible captured graph exists.
        """
        matched = CUDAGraphBatchDimensionBuilder.match_graph_config(
            real_batch_dim=candidate,
            cuda_graph_batch_dimensions_list=self.context.cuda_graph_batch_dimensions_list,
            strict=self.context.is_hybrid_model,
            match_ep_token_counts=False,
        )
        return matched is not None

    def schedule_chunked_prefill(self):
        """
        This function schedules chunked prefill requests.
        Invariant:
            - There are at most one chunked prefill request in the waiting pool,
                which should be the head
            - There are at most one chunked prefill request in the context,
                which should be the last active request
            - context.chunked_prefill_request_id == -1 if no chunked prefill request is scheduled,
                otherwise it is the request id of the chunked prefill request
            - For each request, finished_chunk_token_count is the number of tokens
                that have been prefilled for this request, non-zero means
                it is during a chunked prefill
            - For each request, remaining_prompt_tokens holds the **unprefilled** prompt tokens
        """
        prefix_caching_enabled = self.context.enable_prefix_caching
        if prefix_caching_enabled:
            pending_block_hashes = self._pending_prefill_block_hashes()
            pending_request_ids = []
        can_schedule = True
        while self.waiting_request_ids and can_schedule:
            can_schedule = False
            req = self.get_request(self.waiting_request_ids[0])

            # is_continuing_chunked_prefill is True if we are scheduling next
            # chunk of a existing chunked prefill request
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id >= 0
            batch_invariant_mamba_prefill = (
                self.context.batch_invariant_mode and self.context.is_hybrid_model
            )

            # Check for conflicting block hashes.
            if prefix_caching_enabled and not is_continuing_chunked_prefill:
                has_pending_hash = False
                for block_hash in req.precomputed_block_hashes:
                    # pylint: disable-next=possibly-used-before-assignment
                    if block_hash in pending_block_hashes:
                        has_pending_hash = True
                        break
                if has_pending_hash:
                    self._prefix_coordination_waits += 1
                    pending_request_ids.append(  # pylint: disable=possibly-used-before-assignment
                        self.waiting_request_ids.popleft()
                    )
                    can_schedule = True
                    continue

            # Use remaining prompt tokens for scheduling decisions
            remaining_len = len(req.remaining_prompt_tokens)

            if self._can_schedule_chunked_prefill(req):
                # How many tokens we can admit this step.
                token_budget = self._chunked_prefill_token_budget()

                # Prefix-cache skip: on a request's first chunk, the tokens covered
                # by a cached prefix are reused rather than recomputed, so they do
                # NOT consume the compute budget. Extend this chunk's SPAN to cover
                # the entire skippable prefix plus up to `token_budget` newly computed
                # tokens. Without this the span is capped at the budget, forcing the
                # rest of a long cached prefix to be re-prefilled over many chunks
                # (latency then scales with prompt length instead of the delta).
                # add_request() only computes `effective = span - skip` tokens.
                prefix_skip = 0
                if prefix_caching_enabled and not is_continuing_chunked_prefill:
                    prefix_skip = self.context._compute_prefix_match(
                        req, remaining_len
                    ).prefix_skip_tokens
                    prefix_skip = min(prefix_skip, remaining_len - 1)  # keep >=1 token to run

                computed_budget = min(remaining_len - prefix_skip, token_budget)

                if self._cg_admission_gating_active():
                    # Snap the COMPUTED chunk size to the largest captured-CG boundary
                    # within budget (skipped tokens don't affect the CG batch shape).
                    computed_chunk = self._select_cg_chunk_size(req, computed_budget)
                    if computed_chunk is None:
                        break
                else:
                    computed_chunk = computed_budget

                if batch_invariant_mamba_prefill:
                    computed_chunk = self._mamba_batch_invariant_prefill_chunk_length(
                        remaining_len - prefix_skip, computed_chunk
                    )
                    if computed_chunk == 0:
                        can_schedule = False
                        break
                prefill_chunk_length = prefix_skip + computed_chunk

                # Mamba prefix caching: keep chunk boundaries block-aligned.
                # compute_and_store_offsets() records a recurrent-state snapshot at a
                # KV-block boundary only when that boundary lands on a multiple of the
                # SSM chunk size measured FROM the start of the current prefill chunk
                # (it filters on `offset % mamba_chunk_size == 0`, where the chunk start
                # equals `finished_chunk_token_count` on continuation chunks). Block
                # boundaries are multiples of `block_size_tokens` (itself a multiple of
                # the SSM chunk size), so the filter only passes when
                # `finished_chunk_token_count` is block-aligned. If a chunk ends at an
                # arbitrary token offset, every candidate boundary in the following
                # chunks becomes unrecordable and the last-block snapshot that lets a
                # future request skip prefill is silently dropped. Stop a partial
                # (non-final) chunk short at the nearest lower block boundary so the
                # running `finished_chunk_token_count` stays block-aligned.
                if (
                    self.context.is_hybrid_model
                    and self.context.mamba_slot_allocator is not None
                    and prefill_chunk_length < remaining_len
                ):
                    block_size = self.context.block_size_tokens
                    chunk_end = req.finished_chunk_token_count + prefill_chunk_length
                    aligned_end = (chunk_end // block_size) * block_size
                    aligned_chunk_length = aligned_end - req.finished_chunk_token_count
                    # Only snap down when the aligned chunk still computes at least one
                    # token beyond the skipped prefix (a chunk whose budget is smaller
                    # than a block cannot be block-aligned; leave it unchanged).
                    if aligned_chunk_length > prefix_skip:
                        prefill_chunk_length = aligned_chunk_length

                # A bypass must finish prefill now. Starting a partial request would
                # keep it at the queue head after the earlier prefix conflict clears.
                if (
                    prefix_caching_enabled
                    and pending_request_ids
                    and prefill_chunk_length < remaining_len
                ):
                    break

                request_available, tokens_available, kv_available = self.context.check_availability(
                    req, prefill_chunk_length
                )
                if not (
                    (is_continuing_chunked_prefill or request_available)
                    and tokens_available
                    and kv_available
                ):
                    can_schedule = False
                    break

                # Add hashes to pending set (prefix-caching bookkeeping).
                if prefix_caching_enabled:
                    for block_hash in req.precomputed_block_hashes:
                        if block_hash not in self.context.prefix_cache_registry.kv_hash_to_block_id:
                            pending_block_hashes.add(block_hash)

                if prefill_chunk_length >= remaining_len:
                    self.context.chunked_prefill_request_id = -1
                    self.context.add_request(req)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
                    )
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                    req.add_event_add_context()
                    self.waiting_request_ids.popleft()
                    can_schedule = True
                else:
                    # Partial admit: schedule this chunk and keep the request at the queue head.
                    self.context.add_request(req, prefill_chunk_length=prefill_chunk_length)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
                    )
                    self.context.chunked_prefill_request_id = req.request_id
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens[prefill_chunk_length:]
                    req.finished_chunk_token_count += prefill_chunk_length

        # Prepend pending request ids to waiting queue.
        if prefix_caching_enabled and pending_request_ids:
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id >= 0
            if is_continuing_chunked_prefill:
                chunked_request_id = self.waiting_request_ids.popleft()
                self.waiting_request_ids.extendleft(reversed(pending_request_ids))
                self.waiting_request_ids.appendleft(chunked_request_id)
            else:
                self.waiting_request_ids.extendleft(reversed(pending_request_ids))
