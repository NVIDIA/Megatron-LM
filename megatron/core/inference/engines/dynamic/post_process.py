# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from itertools import repeat
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.utils import internal_api


@internal_api
# pylint: disable=line-too-long
class StepPostProcessingMixin:
    """Per-step request post-processing for `DynamicInferenceEngine`."""

    def post_process_requests(
        self,
        request_ids: torch.Tensor,
        finished_request_ids: torch.Tensor,
        evict_request_ids: torch.Tensor,
        step_time: float,
        sample: torch.Tensor,
        accepted_tokens: torch.Tensor,
        log_probs: torch.Tensor,
        consumed_chunked_prefill_request_id: int,
        top_n_logprobs: Optional[Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        pre_fwd_active_token_count: Optional[int] = None,
        pre_fwd_step_count: Optional[int] = None,
        finished_routing_block_ids: Optional[Dict[int, list[int]]] = None,
        finished_handoff_block_ids: Optional[Dict[int, list[int]]] = None,
        finished_handoff_ssm_slots: Optional[Dict[int, int]] = None,
        finished_handoff_decode_tokens: Optional[Dict[int, list[int]]] = None,
    ) -> Tuple[List[int], List[DynamicInferenceRequest]]:
        """
        Handles post-processing for requests after a step.

        Args:
            request_ids (torch.Tensor): A list of request_ids
            finished_request_ids (torch.Tensor): A list of finished request ids
            evict_request_ids (torch.Tensor): A list of evicted request ids.
            step_time (float): The latency of the last step
            sample: Tensor: The newly generated token for each request
            accepted_tokens: Tensor: The additional accepted tokens for each request
            log_probs: (List): Log probs for each request
            consumed_chunked_prefill_request_id (int): Chunked-prefill request ID
                associated with the consumed forward, or -1 if it had no partial chunk.
            top_n_logprobs: (Dict): Top-n log probs for each request. Maps request_idx to
                list of (top_n_logprobs, top_n_indices) tuples.
            pre_fwd_active_token_count (Optional[int]): Active token count for the
                consumed forward.
            pre_fwd_step_count (Optional[int]): Step count for the consumed forward.
            finished_routing_block_ids: (Dict[int, List[int]]): Block IDs for
                finished requests, saved before update_requests released them.
                Used for per-block routing reconstruction.
            finished_handoff_block_ids: Prompt KV block IDs retained for state handoff.
            finished_handoff_ssm_slots: Live SSM slots detached for state handoff.
            finished_handoff_decode_tokens: First sampled token and optional MTP proposals
                needed to resume directly from imported prefill state on decode.

        Returns:
            Active request IDs and completed requests.
        """
        active_request_ids: list[int] = []
        finished_request_ids = set(finished_request_ids.tolist())
        finished_requests: list[DynamicInferenceRequest] = []
        self.finished_request_count += len(finished_request_ids)
        if evict_request_ids is not None:
            self.evicted_request_count += evict_request_ids.numel()

        log_probs_iter = log_probs if log_probs else repeat(None)
        block_allocator = self.context.kv_block_allocator

        # Pre-compute step-level block stats (before the per-request loop)
        if self.track_generated_token_events:
            blocks_allocated = block_allocator.pool_size - block_allocator.pool_avail
            if block_allocator.enable_prefix_caching:
                blocks_hashed_active = int((block_allocator.block_ref_counts > 0).sum().item())
                blocks_ref_count = block_allocator.block_ref_counts.sum().item()
            else:
                blocks_hashed_active = blocks_allocated
                blocks_ref_count = None

        # When accepted_tokens is None (no speculative decoding), use repeat([]) to provide
        # empty lists for each request, so the zip produces the correct number of iterations
        accepted_tokens_iter = repeat([]) if accepted_tokens is None else accepted_tokens.tolist()

        if self.num_speculative_tokens > 0 and accepted_tokens is not None:
            self._spec_steps += 1

        # Convert the step's request IDs once, then batch-prepare handoff metadata for
        # finished requests using the KV blocks retained before context cleanup.
        request_id_list = request_ids.tolist()
        handoff_blocks_by_request = finished_handoff_block_ids or {}
        handoff_ssm_slots_by_request = finished_handoff_ssm_slots or {}
        prepared_handoff_metadata = self._prepare_handoff_metadata_batch(
            [
                (
                    self.get_request(request_id),
                    handoff_blocks_by_request.get(request_id, []),
                    handoff_ssm_slots_by_request.get(request_id),
                )
                for request_id in request_id_list
                if request_id in finished_request_ids
            ],
            finished_handoff_decode_tokens or {},
        )

        for req_idx, (request_id, tokens, accepted_tokens_list, request_log_probs) in enumerate(
            zip(request_id_list, sample.tolist(), accepted_tokens_iter, log_probs_iter)
        ):
            finished_entry = None

            # Ensure tokens is always a list for consistent handling
            if not isinstance(tokens, list):
                tokens = [tokens]

            request: DynamicInferenceRequest = self.get_request(request_id)

            if self.num_speculative_tokens > 0:
                accepted_tokens = list(filter(lambda tok: tok != -1, accepted_tokens_list))

                # The order `accepted_tokens + tokens` is correct here.
                # `accepted_tokens` contains the sequence of
                # successfully verified draft tokens. `tokens` (from `sample`) is the
                # brand new token generated by the target model based on that accepted prefix.
                # Therefore, the newly sampled token must go at the end of the sequence.
                tokens = accepted_tokens + tokens

            num_stop_word_trim = 0
            num_stop_word_prompt_score_trim = 0
            eos_mid_block_hit = False
            is_prefill = len(request.generated_tokens) == 0
            if request_id != consumed_chunked_prefill_request_id:
                tokens, request_log_probs, eos_mid_block_hit = self._truncate_at_mid_block_eos(
                    request, tokens, request_log_probs, top_n_logprobs, req_idx
                )

                # Skip appending token for requests being finished due to stop words
                # (they already have their final token from the previous step)
                # If the request already has more tokens, then we only append as much as is necessary
                if (
                    len(request.generated_tokens) + len(tokens)
                    >= request.sampling_params.num_tokens_to_generate
                ):
                    keep = request.sampling_params.num_tokens_to_generate - len(
                        request.generated_tokens
                    )
                    num_tokens_before_trim = len(tokens)
                    tokens = tokens[:keep]
                    # Drop only the excess *trailing* log probs / top-n so the counts stay
                    # in sync. We must trim from the end, not the front: on a prefill step
                    # request_log_probs covers the whole prompt and is laid out as
                    # [<prompt log probs...>, <sampled token log prob>], so front-slicing
                    # (e.g. [:keep] with keep == 0 when num_tokens_to_generate == 0) would
                    # discard the prompt log probs that echo+logprobs requests need. In a
                    # decode step all entries are generated, so trailing == front-equivalent.
                    num_dropped = num_tokens_before_trim - len(tokens)
                    if num_dropped > 0:
                        if request_log_probs is not None:
                            request_log_probs = request_log_probs[:-num_dropped]
                        if top_n_logprobs is not None and req_idx in top_n_logprobs:
                            top_n_logprobs[req_idx] = top_n_logprobs[req_idx][:-num_dropped]
                if request_id not in self.stop_word_being_finished_ids:
                    is_first_token = len(request.generated_tokens) == 0
                    request.generated_tokens += tokens
                    first_token_event = None
                    if self.track_generated_token_events:
                        for token in tokens:
                            if block_allocator.enable_prefix_caching:
                                event = request.add_event_generated_token(
                                    token,
                                    blocks_total=block_allocator.pool_size,
                                    blocks_hashed_total=blocks_allocated,
                                    blocks_hashed_active=blocks_hashed_active,
                                    blocks_ref_count=blocks_ref_count,
                                    pre_fwd_active_token_count=pre_fwd_active_token_count,
                                    pre_fwd_step_count=pre_fwd_step_count,
                                )
                            else:
                                event = request.add_event_generated_token(
                                    token,
                                    blocks_total=block_allocator.pool_size,
                                    blocks_hashed_total=blocks_allocated,
                                    blocks_hashed_active=blocks_hashed_active,
                                    pre_fwd_active_token_count=pre_fwd_active_token_count,
                                    pre_fwd_step_count=pre_fwd_step_count,
                                )
                            if first_token_event is None:
                                first_token_event = event
                    if is_first_token and tokens:
                        if not self.track_generated_token_events:
                            first_token_event = DynamicInferenceEvent(
                                type=DynamicInferenceEventType.GENERATED_TOKEN,
                                payload={"token_id": tokens[0]},
                            )
                        request.ttft = (
                            first_token_event.timestamp - request.event_add_engine.timestamp
                        )
                    # TPOT is observability-only. step_time is 0.0 on
                    # non-logging steps (async_forward skips the event sync),
                    # so gate the update to keep the metric a truthful sparse
                    # sample instead of polluting it with zeros.
                    if step_time > 0 and tokens:
                        per_token_step_time = step_time / len(tokens)
                        request.tpot.extend([per_token_step_time] * len(tokens))

                # Check for stop words (after token is appended).
                # With speculative decoding, a stop word may end before the last
                # appended token. The check truncates generated_tokens in-place and
                # returns how many trailing tokens were removed so we can also trim
                # the corresponding log probs below.
                num_new_tokens = (
                    len(tokens) if request_id not in self.stop_word_being_finished_ids else 0
                )
                stop_word_hit, num_stop_word_trim, num_stop_word_prompt_score_trim = (
                    self._check_stop_words_for_request_post_append(
                        request,
                        record=self.requests[request_id].record,
                        num_new_tokens=num_new_tokens,
                    )
                )

                # Record tokens emitted (and kept) this step so a client can reconstruct
                # per-step acceptance lengths. Runs AFTER the stop-word check, which may
                # truncate the last step in place, so subtract the trim to keep the list
                # summing to `generated_length`. The same reason excludes a consumed chunked
                # prefill: it samples tokens that are never appended. Spec decoding only.
                if (
                    self.num_speculative_tokens > 0
                    and request_id != consumed_chunked_prefill_request_id
                    and request_id not in self.stop_word_being_finished_ids
                ):
                    emitted = len(tokens) - num_stop_word_trim
                    if emitted > 0:
                        request.acceptance_step_lengths.append(emitted)
                    else:
                        # A stop sequence can reach back past this step's tokens, removing ones
                        # earlier entries already counted. Unwind them so the list keeps summing
                        # to `generated_length`; a client reconstructing acceptance from it
                        # would otherwise read a length longer than the output.
                        residual = -emitted
                        while residual > 0 and request.acceptance_step_lengths:
                            last = request.acceptance_step_lengths[-1]
                            if last > residual:
                                request.acceptance_step_lengths[-1] = last - residual
                                residual = 0
                            else:
                                residual -= last
                                request.acceptance_step_lengths.pop()

                # Track per-position acceptance statistics for logging.
                # Skip prefill requests: MTP heads only propose speculative tokens
                # for decode requests, so counting prefill requests would inflate
                # the denominator and artificially deflate the acceptance rate.
                if (
                    not is_prefill
                    and len(request.generated_tokens) > 0
                    and self.num_speculative_tokens > 0
                ):
                    actual_proposed = max(0, self.num_speculative_tokens - num_stop_word_trim)
                    self._spec_tokens_proposed_per_pos[:actual_proposed] += 1
                    accepted_t = torch.tensor(accepted_tokens_list[:actual_proposed])
                    self._spec_tokens_accepted_per_pos[:actual_proposed] += (
                        accepted_t != -1
                    ).long()

                if request_id in finished_request_ids:
                    # Reconstruct routing from per-block storage before popping.
                    if finished_routing_block_ids and request_id in finished_routing_block_ids:
                        block_ids = finished_routing_block_ids[request_id]
                        total_tokens = len(request.prompt_tokens) + len(request.generated_tokens)
                        request.routing_indices = (
                            self.context.kv_block_allocator.reconstruct_routing_from_blocks(
                                block_ids, total_tokens - 1
                            )
                        )

                    # Request finished by normal means (termination_id, max_length, or stop word from previous step)
                    request.generated_length = len(request.generated_tokens)
                    request.status = Status.COMPLETED
                    request.add_event_finish()
                    # Keep handoff blocks only when the request needs them.
                    handoff_blocks = handoff_blocks_by_request.get(request_id, [])
                    if request.sampling_params.do_kv_handoff:
                        self._capture_handoff_meta(
                            request, prepared_handoff_metadata.get(request_id)
                        )
                    # A prefill-role engine may also serve regular requests; release the
                    # temporary state ownership when no handoff was requested.
                    elif handoff_blocks or request_id in handoff_ssm_slots_by_request:
                        self._release_pinned_handoff_blocks(handoff_blocks)
                        self._release_pinned_handoff_ssm_slot(
                            handoff_ssm_slots_by_request.get(request_id)
                        )
                    finished_entry = self.requests[request_id]
                elif stop_word_hit or eos_mid_block_hit:
                    # Stop word or mid-speculative-block EOS detected - mark for removal in
                    # next step's bookkeeping. Don't pop yet; let the next step handle it
                    # properly via callback. Both share this deferred-finish channel
                    # because the context still has the request active at this point.
                    self.stop_word_finished_request_ids.add(request_id)
                    active_request_ids.append(request_id)
                else:
                    active_request_ids.append(request_id)
            else:
                # The chunked prefill produces useless tokens
                # so we are not appending them to the generated tokens.
                # Additionally, chunked prefill request do not finish.
                active_request_ids.append(request_id)

            # When a stop word was found mid-speculative-batch, trim log probs
            # and top_n_logprobs to match the truncated generated_tokens.
            num_stop_word_score_trim = num_stop_word_trim + num_stop_word_prompt_score_trim
            if num_stop_word_score_trim > 0:
                if request_log_probs is not None:
                    request_log_probs = request_log_probs[:-num_stop_word_score_trim]
                if top_n_logprobs is not None and req_idx in top_n_logprobs:
                    top_n_logprobs[req_idx] = top_n_logprobs[req_idx][:-num_stop_word_score_trim]

            # Process requested log_probs (unified for both regular and chunked prefill)
            # Skip for requests being finished due to stop words — tokens are not
            # appended for these requests, so log probs must also be skipped to keep
            # the two lists in sync.
            if (
                request.sampling_params.return_log_probs
                and request_id not in self.stop_word_being_finished_ids
            ):
                assert (
                    request_log_probs is not None
                ), f"Request {request_id} requested log probs, but none were produced."
                # Initialize lists if they don't exist
                if not request.prompt_log_probs:
                    request.prompt_log_probs = []
                if not request.generated_log_probs:
                    request.generated_log_probs = []

                is_chunked_prefill = request_id == consumed_chunked_prefill_request_id
                is_prefill = len(request.generated_log_probs) == 0

                if request.sampling_params.skip_prompt_log_probs:
                    # We only want decode log probs.
                    if is_chunked_prefill:
                        pass
                    elif is_prefill:
                        if request.generated_tokens and len(request_log_probs) > 0:
                            request.generated_log_probs.append(request_log_probs[-1])
                    else:
                        request.generated_log_probs.extend(request_log_probs)
                else:
                    # Split log probs between prompt and generated based on remaining prompt slots.
                    prompt_length = len(request.prompt_tokens)
                    total_accumulated = len(request.prompt_log_probs) + len(
                        request.generated_log_probs
                    )
                    remaining_prompt_slots = max(0, prompt_length - 1 - total_accumulated)
                    split_idx = min(remaining_prompt_slots, len(request_log_probs))

                    if split_idx > 0:
                        request.prompt_log_probs.extend(request_log_probs[:split_idx])
                    if split_idx < len(request_log_probs):
                        request.generated_log_probs.extend(request_log_probs[split_idx:])

            # Process top_n_logprobs if available (unified for both regular and chunked prefill)
            # Same stop-word guard as log probs above.
            if (
                top_n_logprobs is not None
                and req_idx in top_n_logprobs
                and request_id not in self.stop_word_being_finished_ids
                and not (
                    request_id == consumed_chunked_prefill_request_id
                    and request.sampling_params.skip_prompt_log_probs
                )
            ):
                # Initialize lists if they don't exist
                if request.prompt_top_n_logprobs is None:
                    request.prompt_top_n_logprobs = []
                if request.generated_top_n_logprobs is None:
                    request.generated_top_n_logprobs = []

                top_n_data_list = top_n_logprobs[req_idx]
                prompt_length = len(request.prompt_tokens)

                # Process each token's top-n logprobs
                for top_n_values, top_n_indices in top_n_data_list:
                    logit_dict = {}
                    for logprob, logprob_index in zip(
                        top_n_values.cpu().tolist(), top_n_indices.cpu().tolist()
                    ):
                        key = self.controller.tokenizer.detokenize([logprob_index])
                        logit_dict[key] = logprob

                    # Simple decision: check total count accumulated so far
                    total_accumulated = len(request.prompt_top_n_logprobs) + len(
                        request.generated_top_n_logprobs
                    )

                    # If skip_prompt_log_probs is False and we haven't reached prompt end,
                    # append to prompt_top_n_logprobs. Otherwise append to generated_top_n_logprobs.
                    if (
                        not request.sampling_params.skip_prompt_log_probs
                        and total_accumulated < prompt_length - 1
                    ):
                        request.prompt_top_n_logprobs.append(logit_dict)
                    else:
                        request.generated_top_n_logprobs.append(logit_dict)

            # Merge only after the final token's scores and metadata have been applied.
            if finished_entry is not None:
                popped_entry = self.requests.pop(request_id)
                assert popped_entry is finished_entry
                finished_requests.append(self._complete_request(finished_entry))

        # Handle evicted requests.
        if evict_request_ids is not None and evict_request_ids.numel() > 0:

            evict_request_ids = evict_request_ids.tolist()

            # Insert into waiting_request_ids after any chunk prefill request.
            self.waiting_request_ids.extendleft(evict_request_ids)
            if self.context.chunked_prefill_request_id != -1:
                chunked_prefill_id = self.waiting_request_ids[len(evict_request_ids)]
                del self.waiting_request_ids[len(evict_request_ids)]
                self.waiting_request_ids.appendleft(chunked_prefill_id)

            # Checkpoint requests (i.e., prompt += generations) + add eviction event.
            for request_id in evict_request_ids:
                self.requests[request_id].record.checkpoint()
                self.get_request(request_id).add_event_evict()

        # Clear the stop word being finished set after processing
        self.stop_word_being_finished_ids.clear()

        # Finished VLM data is no longer needed after the request has been flattened.
        for request in finished_requests:
            self.context.remove_vlm_request_data(request.request_id)

        return active_request_ids, finished_requests

    def _get_and_clear_stop_word_finished_ids(self, active_request_ids: list[int]) -> set[int]:
        """Get and clear the set of request IDs that should be finished due to stop words.

        This callback is called from the controller during bookkeeping to get request IDs
        that were detected as hitting stop words in the previous step's post_process_requests.

        Args:
            active_request_ids: List of currently active request IDs.

        Returns:
            Set of request IDs from active_request_ids that should be marked as finished.
        """
        if not self.stop_word_finished_request_ids:
            return set()

        # Find which stop word finished IDs are in the current active requests
        result = self.stop_word_finished_request_ids & set(active_request_ids)
        # Move to "being finished" set so post_process_requests can skip the extra token
        self.stop_word_being_finished_ids = result
        # Clear the IDs that we're returning (they'll be marked as finished)
        self.stop_word_finished_request_ids -= result
        return result

    def _terminating_token_ids(self, request: DynamicInferenceRequest) -> frozenset:
        """Token ids that end generation for this request.

        The CPU-side counterpart of the controller's per-step tensor check, resolved
        through the controller so every termination site shares one rule. Empty when
        termination is disabled (`ignore_eos`).

        Args:
            request (DynamicInferenceRequest): Request to resolve ids for.

        Returns:
            frozenset: Terminating token ids, empty when termination is disabled.
        """
        return self.controller.terminating_token_ids(request.sampling_params.termination_id)

    def _truncate_at_mid_block_eos(
        self,
        request: DynamicInferenceRequest,
        tokens: list[int],
        request_log_probs: Optional[list],
        top_n_logprobs: Optional[Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]],
        req_idx: int,
    ) -> Tuple[list, Optional[list], bool]:
        """Drop everything a speculative step emitted after an EOS.

        An EOS can land on an *accepted* speculative token rather than on the step's
        last token, which is the only one the controller's termination check sees
        (`sampled_tokens_cpu` is the target model's own sample at the last accepted
        position). Accepted tokens are verified target tokens, not rolled-back drafts,
        so an EOS among them is real and must end the request -- otherwise the rest of
        the block, plus every later step, is emitted after EOS. This mirrors the
        stop-word check, which truncates a stop sequence that ends mid-block.

        Trailing log probs / top-n are trimmed to match, keeping them aligned with
        `tokens`. `top_n_logprobs` is trimmed in place; log probs are returned.

        Args:
            request (DynamicInferenceRequest): Request the tokens belong to.
            tokens (list[int]): Tokens emitted by this step, in order.
            request_log_probs (Optional[list]): This step's log probs, or None.
            top_n_logprobs (Optional[Dict]): Per-request top-n log probs, or None.
            req_idx (int): This request's index into `top_n_logprobs`.

        Returns:
            Tuple[list, Optional[list], bool]: Truncated tokens, truncated log probs,
            and whether an EOS was found (the caller defers the finish by a step).
        """
        eos_idx = self._find_mid_block_eos(request, tokens)
        if eos_idx is None:
            return tokens, request_log_probs, False

        num_trim = len(tokens) - (eos_idx + 1)
        if num_trim > 0:
            if request_log_probs is not None:
                request_log_probs = request_log_probs[:-num_trim]
            if top_n_logprobs is not None and req_idx in top_n_logprobs:
                top_n_logprobs[req_idx] = top_n_logprobs[req_idx][:-num_trim]
        return tokens[: eos_idx + 1], request_log_probs, True

    def _find_mid_block_eos(
        self, request: DynamicInferenceRequest, tokens: list[int]
    ) -> Optional[int]:
        """Locate an EOS token inside a multi-token speculative step.

        Only meaningful when a step emits more than one token, i.e. under speculative
        decoding: the controller's termination check inspects just the step's last token,
        so an EOS on an accepted draft position is otherwise missed and generation runs
        past it. A single-token step was already checked by the controller, and a request
        the controller has already finished does not reach here as unfinished.

        Args:
            request (DynamicInferenceRequest): Request the tokens belong to.
            tokens (list[int]): Tokens emitted by this step, in order.

        Returns:
            Optional[int]: Index of the first EOS in `tokens`, or None if there is none.
        """
        if len(tokens) <= 1 or request.request_id in self.stop_word_being_finished_ids:
            return None
        terminating_ids = self._terminating_token_ids(request)
        if not terminating_ids:
            return None
        for idx, token in enumerate(tokens):
            if token in terminating_ids:
                return idx
        return None

    def _check_stop_words_for_request_post_append(
        self,
        request: DynamicInferenceRequest,
        *,
        record: Optional[DynamicInferenceRequestRecord] = None,
        num_new_tokens: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """Check if a request should stop due to stop words (after token is appended).

        This method is called from post_process_requests after the token has already
        been appended to request.generated_tokens. In the speculative decoding case,
        multiple tokens may have been appended at once. If a stop word is found in the
        middle of the speculative tokens, the trailing tokens after the stop word are
        truncated from generated_tokens.

        With speculative decoding, multiple tokens are appended at once. The stop word
        may end before the last appended token, leaving extra tokens that must be
        trimmed. When this happens, generated_tokens is truncated in-place and the
        number of trimmed tokens is returned so the caller can also trim log probs.

        Args:
            request: The request to check.
            record: Full checkpoint history for the request. Supplying the record
                lets stop sequences span checkpoint boundaries.
            num_new_tokens: Number of tokens appended in the current step. The
                returned trim count is limited to these tokens so the caller can
                trim pending log-probability results without deleting prompt data.

        Returns:
            Tuple of (stop_word_hit, num_new_tokens_trimmed,
            num_recomputed_prompt_scores_trimmed):
                stop_word_hit: True if the generated sequence contains a stop word.
                num_new_tokens_trimmed: Number of current-step tokens removed
                    from the end of generated_tokens.
                num_recomputed_prompt_scores_trimmed: Number of prompt-score
                    entries corresponding to stripped tokens from older
                    checkpoint segments.
        """
        if request.stop_word_ids is None or len(request.stop_word_ids) == 0:
            return False, 0, 0

        segments = record.requests if record is not None else [request]
        if record is not None:
            assert record[-1] is request

        # Legacy direct callers do not have pending score tensors to align and
        # expect the complete token trim count. The engine path always supplies
        # num_new_tokens.
        engine_call = num_new_tokens is not None
        endpoint_count = num_new_tokens if engine_call else self.num_speculative_tokens + 1
        pending_token_count = num_new_tokens if engine_call else len(request.generated_tokens)
        generated_tokens = []
        suffix_length = max(map(len, request.stop_word_ids)) + max(0, endpoint_count - 1)
        for segment in reversed(segments):
            take = min(suffix_length - len(generated_tokens), len(segment.generated_tokens))
            if take > 0:
                generated_tokens[:0] = segment.generated_tokens[-take:]
            if len(generated_tokens) == suffix_length:
                break
        tpot_is_token_aligned = {
            id(segment): len(segment.tpot) == len(segment.generated_tokens) for segment in segments
        }

        def trim_segment_tail(
            segment: DynamicInferenceRequest, trim: int, *, trim_stored_scores: bool
        ) -> None:
            """Trim token-correlated state from one request segment."""
            if trim == 0:
                return

            removed_tokens = list(segment.generated_tokens[-trim:])
            segment.generated_tokens = segment.generated_tokens[:-trim]

            if trim_stored_scores:
                for key in ("generated_log_probs", "generated_top_n_logprobs"):
                    values = getattr(segment, key, None)
                    if values is not None:
                        setattr(segment, key, values[:-trim])

            # TPOT is intentionally sparse on non-logging steps. Trim it only
            # when it is demonstrably one value per generated token.
            if tpot_is_token_aligned[id(segment)]:
                segment.tpot = segment.tpot[:-trim]

            generated_event_indexes = [
                idx
                for idx, event in enumerate(segment.events)
                if event.type == DynamicInferenceEventType.GENERATED_TOKEN
            ]
            if generated_event_indexes:
                assert len(generated_event_indexes) >= trim
                indexes_to_remove = generated_event_indexes[-trim:]
                event_tokens = [
                    segment.events[idx].payload["token_id"] for idx in indexes_to_remove
                ]
                assert event_tokens == removed_tokens
                for idx in reversed(indexes_to_remove):
                    del segment.events[idx]

            if segment.generated_length is not None:
                segment.generated_length = len(segment.generated_tokens)

        matched_stop = None
        for stop_word_ids in request.stop_word_ids:
            stop_len = len(stop_word_ids)
            if len(generated_tokens) < stop_len:
                continue
            # Search every endpoint produced in this step before mutating state.
            # A larger trailing-token count means this stop completed earlier;
            # at the same endpoint, prefer the longer stop sequence.
            for trailing_token_count in range(endpoint_count):
                end_idx = -trailing_token_count if trailing_token_count > 0 else None
                if (
                    list(generated_tokens[-stop_len - trailing_token_count : end_idx])
                    == stop_word_ids
                ):
                    candidate = (trailing_token_count, stop_len)
                    if matched_stop is None or candidate > matched_stop:
                        matched_stop = candidate

        if matched_stop is None:
            return False, 0, 0

        trailing_token_count, stop_len = matched_stop
        total_trim = (
            trailing_token_count
            if request.sampling_params.detokenize_stop_sequence
            else trailing_token_count + stop_len
        )
        pending_trim = min(total_trim, pending_token_count)
        if pending_trim > 0:
            trim_segment_tail(request, pending_trim, trim_stored_scores=False)

        # Any remainder belongs to tokens generated before this step. Remove
        # their already-stored result metadata from newest to oldest so merged
        # tokens, logprobs, and top-N values stay aligned.
        stored_trim = total_trim - pending_trim
        older_segment_trim = 0
        for segment in reversed(segments):
            if stored_trim == 0:
                break
            segment_trim = min(stored_trim, len(segment.generated_tokens))
            if segment_trim == 0:
                continue
            trim_segment_tail(segment, segment_trim, trim_stored_scores=True)
            if segment is not request:
                older_segment_trim += segment_trim
            stored_trim -= segment_trim
        assert stored_trim == 0, "Stop-word trim exceeds generated history."

        # Tokens from older checkpoint segments are duplicated at the end of
        # the active segment's cumulative prompt. Drop the same suffix so
        # terminal routing/serialization lengths describe the visible result.
        if older_segment_trim > 0:
            old_prompt_tokens = request.prompt_tokens
            old_remaining_prompt_tokens = request.remaining_prompt_tokens
            request.prompt_tokens = old_prompt_tokens[:-older_segment_trim]
            if old_remaining_prompt_tokens is not None:
                remaining_length = max(0, len(old_remaining_prompt_tokens) - older_segment_trim)
                request.remaining_prompt_tokens = (
                    request.prompt_tokens[-remaining_length:]
                    if remaining_length > 0
                    else request.prompt_tokens[:0]
                )

        prompt_score_trim = 0
        if older_segment_trim > 0 and not request.sampling_params.skip_prompt_log_probs:
            # Chunked prefill may already have stored some prompt scores. Remove
            # any suffix now outside the shortened prompt and ask the caller to
            # trim only the remainder from this step's pending score tensors.
            target_prompt_score_count = max(0, len(request.prompt_tokens) - 1)
            existing_prompt_score_count = len(request.prompt_log_probs or [])
            stored_prompt_score_trim = min(
                older_segment_trim, max(0, existing_prompt_score_count - target_prompt_score_count)
            )
            if stored_prompt_score_trim > 0:
                request.prompt_log_probs = request.prompt_log_probs[:-stored_prompt_score_trim]
                if request.prompt_top_n_logprobs is not None:
                    request.prompt_top_n_logprobs = request.prompt_top_n_logprobs[
                        :-stored_prompt_score_trim
                    ]
            prompt_score_trim = older_segment_trim - stored_prompt_score_trim
        return True, pending_trim, prompt_score_trim
