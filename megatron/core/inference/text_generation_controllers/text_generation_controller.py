# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import concurrent
import copy
import functools
import inspect
from collections import defaultdict
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts.dynamic_context import MaxSequenceLengthOverflowError
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import get_attention_mask, set_decode_expert_padding
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.utils import set_model_to_sequence_parallel
from megatron.core.utils import get_asyncio_loop, get_model_config, unwrap_model

try:
    import transformer_engine as te  # pylint: disable=unused-import

    HAVE_TE = True

except ImportError:
    HAVE_TE = False


class TextGenerationController:
    """The text generation controller (the main sampling loop)

    This class tokenizes the input, runs inference, samples from logits, and detokenizes the output.

    Args:
        inference_wrapped_model (AbstractModelInferenceWrapper): A model that
            is wrapped using the specs given in the abstract_model_inference_wrapper.py
        tokenizer (_type_): Tokenizer used for tokenizing and detokenizing the prompts
        pp_group (ProcessGroup): Process group for pipeline parallelism
    """

    def __init__(
        self,
        inference_wrapped_model: AbstractModelInferenceWrapper,
        tokenizer,
        pp_group: ProcessGroup = None,
    ):
        self.inference_wrapped_model = inference_wrapped_model
        self.tokenizer = tokenizer

        self.pp_group = pp_group

        # For models without pipeline parallelism, is_first_stage and is_last_stage returns True
        self.model_is_pipeline_parallel = not (
            is_pipeline_first_stage(self.pp_group) and is_pipeline_last_stage(self.pp_group)
        )

        model_config = get_model_config(self.inference_wrapped_model.model)
        self.sampling_rng = torch.Generator(device=torch.cuda.current_device())
        self.sampling_rng.manual_seed(model_config.inference_sampling_seed)

    def tokenize_prompt(self, prompt: str, add_BOS: bool = False) -> List[int]:
        """Utility to tokenize the input prompts.

        Args:
            prompt (str): The input prompt.

        Returns:
            List[int]: Returns the tokenized prompt.
        """

        prompt_tokens = self.tokenizer.tokenize(prompt)

        if add_BOS:
            assert self.tokenizer.bos is not None

        while prompt_tokens and prompt_tokens[0] == self.tokenizer.bos:
            prompt_tokens.pop(0)

        if add_BOS:
            prompt_tokens = [self.tokenizer.bos] + prompt_tokens

        return prompt_tokens

    def _detokenize(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Detokenize a sequence of token IDs, handling skip_special_tokens for
        different tokenizer APIs.

        On the first call, inspects `self.tokenizer.detokenize` to see if it accepts
        a `skip_special_tokens` keyword argument, and caches that result on `self`.
        Subsequent calls will use the cached flag to invoke `detokenize` with the
        correct signature (with or without `skip_special_tokens`).

        Args:
            tokens (List[int]): The token IDs to convert back to text.
            skip_special_tokens (bool): Whether to remove special tokens (e.g. BOS/EOS)
                during detokenization. Only passed through if the tokenizer supports it.

        Returns:
            str: The detokenized string.
        """
        # cache the check on first call
        if not hasattr(self, "_detok_accepts_skip"):
            sig_params = inspect.signature(self.tokenizer.detokenize).parameters.values()
            self._detok_accepts_skip = any(
                p.name == "skip_special_tokens" or p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig_params
            )
        if self._detok_accepts_skip:
            return self.tokenizer.detokenize(tokens, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.detokenize(tokens)

    def detokenize_generations(
        self,
        tokens_gpu_tensor: torch.Tensor,
        lengths_gpu_tensor: torch.Tensor,
        detokenize_segments: bool,
        skip_special_tokens: bool = True,
    ) -> tuple[str, Optional[List[List[str]]]]:
        """Detokenize the generated tokens.

        Args:
            tokens_gpu_tensor (torch.Tensor): Tensor containing the tokens
            lengths_gpu_tensor (torch.Tensor): Tensor containing the lengths of each sequence
            detokenize_segments (bool): If True, returns individually detokenized tokens. If False,
            returns None as second element. Helpful for understanding per-token boundaries in
            generated text.
            skip_special_tokens (bool): If True removes special tokens like bos
            during detokenization.

        Returns:
            tuple[str, List[str] | None]: A tuple containing:
            - str: The complete detokenized text
            - List[str] | None: List of segmented tokens if detokenize_segments is True, else None
        """
        # TODO(helenn): Unify with `detokenize_generations` from legacy textgen path

        if not detokenize_segments:
            tokens = tokens_gpu_tensor.tolist()
            return self._detokenize(tokens, skip_special_tokens=skip_special_tokens), None

        prompts_plus_generations: List[str] = []
        prompts_plus_generations_segments: List[List[str]] = []
        tokens_gpu_tensor = torch.unsqueeze(tokens_gpu_tensor, 0)
        tokens = tokens_gpu_tensor.tolist()
        lengths = lengths_gpu_tensor.tolist()

        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            detok_str = self._detokenize(sequence_tokens)
            prompts_plus_generations.append(detok_str)
            offsets = self.tokenizer.offsets(sequence_tokens, detok_str)
            words = [
                detok_str[start:end] for start, end in zip(offsets, offsets[1:] + [len(detok_str)])
            ]

            prompts_plus_generations_segments.append(words)

        text = self._detokenize(tokens[0], skip_special_tokens=skip_special_tokens)

        return text, prompts_plus_generations_segments

    def sample_from_logits(
        self,
        last_token_logits: torch.Tensor,
        sampling_params: Optional[SamplingParams] = None,
        vocab_size: Optional[int] = None,
        generation_started: Optional[torch.Tensor] = None,
        top_n_logprobs_dict: Dict[int, List[Dict[str, float]]] = None,
        logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it
        according to the parameters defined in sampling_params
        and returns the samples. If sampling parameters top_n_logprobs > 0
        at each step it also updates the top_n_logprobs dict.

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of
                size [batch_size, vocab_size]
            sampling_params (SamplingParams): The parameters to use for inference.
            vocab_size (int): Obtained from the tokenizer. Defaults to None
            generation_started (torch.Tensor): A boolean tensor of shape [batch_size]. True
                            indicates the prompt at that index has started generating tokens.
            top_n_logprobs_dict (top_n_logprobs_dict): The dict to be updated

        Returns:
            sampled_logits (torch.Tensor): 1D tensor with [batch_size] elements
            top_n_logprobs_this_step (torch.return_types.topk): a topk tensor with values as logits
                and indices as the top k elements. None if sampling params top_n_logprobs is 0.
        """

        if kwargs.get("common_inference_params"):
            sampling_params = kwargs["common_inference_params"]

        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        temperature = sampling_params.temperature

        assert isinstance(top_p, float)
        assert isinstance(top_k, int)
        assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"
        assert top_p <= 1.0, "top-p should be in (0,1]"

        def modify_logits_for_top_k_filtering(logits, top_k):
            """Set the logits for none top-k values to -inf."""
            filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits.masked_fill_(filter_, float("-Inf"))

        def modify_logits_for_top_p_filtering(logits, top_p):
            """Set the logits for none top-p values to -inf."""
            # First sort and calculate cumulative sum of probabilities.
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Filteration based on the cumulative sum.
            filter_ = cumulative_probs > top_p
            # This shift by 1 is weird and I cannot justify it. This existed
            # in the original implementation:
            #   https://github.com/ari-holtzman/degen/blob/master/gen.py
            # and I guess it is needed so keeping it for now.
            filter_[:, 1:] = filter_[:, :-1].clone()
            # Make sure we at least have one token to select from.
            filter_[..., 0] = 0

            # Fill in the filtered part
            filter_ = filter_.scatter(1, sorted_indices, filter_)
            logits.masked_fill_(filter_, float("-Inf"))

        if sampling_params.top_n_logprobs > 0:
            # NOTE : This thing can also be clubbed with where we compute log probs
            # when --return-log-probs is enabled. This is just more efficient
            assert generation_started is not None
            if logits is None:
                batch_size = last_token_logits.shape[0]
                last_token_log_probs = F.log_softmax(last_token_logits, dim=1).to(torch.float32)
                top_n_logits_this_step = torch.topk(
                    last_token_log_probs, k=sampling_params.top_n_logprobs
                )
                top_n_logprobs_this_step = top_n_logits_this_step.values.cpu()
                top_n_logprobs_indices = top_n_logits_this_step.indices.cpu()

                # If we return prompt top_n_log_probs then we always append to the
                # logprobs dict. Otherwise we only append for generated tokens.
                if sampling_params.return_prompt_top_n_logprobs:
                    mask = torch.ones(batch_size, dtype=torch.bool)
                else:
                    mask = generation_started.cpu()

                self._update_top_n_logprobs_dict(
                    top_n_logprobs_this_step, top_n_logprobs_indices, mask, top_n_logprobs_dict
                )
            else:
                assert sampling_params.return_prompt_top_n_logprobs

                # Compute the prompt logprobs
                batch_size, seq_length, _ = logits.shape
                log_probs = F.log_softmax(logits, dim=2).to(torch.float32)
                top_n_logits_this_step = torch.topk(log_probs, k=sampling_params.top_n_logprobs)

                # Move the token dimension to the front and then add each token logprobs
                # individually for every request in the batch
                top_n_logprobs_this_step = top_n_logits_this_step.values.permute(1, 0, 2).cpu()
                top_n_logprobs_indices = top_n_logits_this_step.indices.permute(1, 0, 2).cpu()

                # We append to the logprobs dict for every prompt token
                mask = torch.ones(batch_size, dtype=torch.bool)

                for i in range(seq_length):
                    self._update_top_n_logprobs_dict(
                        top_n_logprobs_this_step[i],
                        top_n_logprobs_indices[i],
                        mask,
                        top_n_logprobs_dict,
                    )

        # Greedy sampling
        if top_k == 1:
            sampled_logits = torch.argmax(last_token_logits, dim=-1)
        else:
            last_token_logits = last_token_logits.clone()
            if temperature != 1.0:
                last_token_logits.div_(temperature)
            if top_k > 1:
                assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
                if vocab_size:
                    assert top_k < vocab_size, "top-k is larger than vocab size."
                modify_logits_for_top_k_filtering(last_token_logits, top_k)

            elif top_p > 0.0:
                modify_logits_for_top_p_filtering(last_token_logits, top_p)

            # After filtering, we need to recalculate the distribution.
            probabilities = last_token_logits.softmax(dim=-1)

            sampled_logits = torch.multinomial(
                probabilities, num_samples=1, generator=self.sampling_rng
            ).view(-1)

            # If vocab size is provided, make sure the samples are in in the range [0, vocab-size).
            if vocab_size:
                sampled_logits = torch.clamp(sampled_logits, min=0, max=(vocab_size - 1))

        return sampled_logits

    def update_generation_status(
        self,
        updated_prompts_tokens: torch.Tensor,
        generation_started: torch.Tensor,
        current_context_end_position: int,
        is_generation_done_tensor: torch.Tensor,
        generated_sequence_lengths: torch.Tensor,
        termination_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Checks which prompts have reached an end condition

        We check which prompts have reached an end condition and set the corresponding
        flags of the is_generation_done_tensor to True. The generated sequence lengths
        increase as we keep generating, until that prompts hits an end condition. The
        generation_started tensor determines which prompts have started generating.

        Args:
            updated_prompts_tokens (torch.Tensor): The prompts tokens updated with the latest
                generated tokens. A tensor of shape [batch_size, max_seq_len]
                (i.e max_seq_len = max_prompt_len + tokens_to_generate)
            generation_started (torch.Tensor): A boolean tensor of shape [batch_size]. True
                indicates the prompt at that index has started generating tokens.
            current_context_end_position (int): An integer indicating which position to
                extract from the prompts tokens to get the latest generated tokens.
            is_generation_done_tensor (torch.Tensor): A boolean tensor of shape [batch_size].
                True indicates the prompt at that index has reached end condition.
            generated_sequence_lengths (torch.Tensor): A int tensor of shape [batch_size].
                Each value represents the generated sequence lengths for that prompt.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the boolean
                is_generation_done_tensor and the generated_sequence_lengths after updating it
        """
        if termination_id is None:
            termination_id = self.tokenizer.eod
        latest_samples = updated_prompts_tokens[:, current_context_end_position]
        # Make sure we are checking eod criterion only for prompts that have started generating
        # (i.e) We only look at the generated tokenns and not the input tokens.
        reached_eod = (latest_samples == termination_id) & generation_started
        is_generation_done_tensor = is_generation_done_tensor | reached_eod
        # We increment generated sequence lengths when that prompt has not hit the
        # EOD and generation has started
        generated_sequence_lengths += ~is_generation_done_tensor & generation_started

        return is_generation_done_tensor, generated_sequence_lengths.int()

    def pad_input_prompt_tokens(
        self,
        batch_prompt_tokens_list: List[List[int]],
        padded_batch_size: int,
        padded_sequence_length: int,
    ) -> torch.Tensor:
        """Method to pad input prompts

        Given a list of prompts, pad them all to uniform length

        Args:
            batch_prompt_tokens_list (List[List[int]]): A list containing the prompt tokens
            padded_batch_size (int): The maximum number of requests for this batch
            padded_sequence_length (int): The maximum number of input + output tokens for this batch

        Returns:
            torch.Tensor: A torch tensor of shape [padded_batch_size, padded_sequence_length]
        """
        batch_size = len(batch_prompt_tokens_list)

        # Pad existing tokens to maximum sequence length
        for prompt_tokens in batch_prompt_tokens_list:
            padding_size = padded_sequence_length - len(prompt_tokens)
            prompt_tokens.extend([self.tokenizer.eod] * padding_size)

        # Pad to maximum batch size
        padded_prompt_tokens_list = batch_prompt_tokens_list
        num_padded_requests = padded_batch_size - len(batch_prompt_tokens_list)
        padded_prompt_tokens_list += [
            [self.tokenizer.eod] * padded_sequence_length for _ in range(num_padded_requests)
        ]

        tokens = torch.tensor(padded_prompt_tokens_list, device=torch.cuda.current_device())

        return tokens

    def unpad_input_prompt_tokens(
        self, padded_batch_prompt_tokens: torch.Tensor, original_batch_size: int
    ):
        """Truncates the given input tensor back to the original prompt size before padding.

        Args:
            padded_batch_prompt_tokens (torch.Tensor): The padded tokens tensor
            original_batch_size (int): The original batch size before padding
        """
        return padded_batch_prompt_tokens[:original_batch_size]

    @torch.inference_mode()
    async def async_generate_output_tokens_dynamic_batch(
        self, sampling_params: SamplingParams, termination_id: int
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward step the model and update the inference context.

        Args:
            sampling_params (SamplingParams): Parameters for sampling logits.

        Return:
            (Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]) Current request IDs,
                paused request IDs, finished request IDs, new sample.
        """
        context = self.inference_wrapped_model.inference_context

        # Remove Float16Module wrapper if it exists
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)

        materialize_only_last_token_logits = context.materialize_only_last_token_logits
        if sampling_params.return_log_probs:
            skip_prompt_log_probs_for_dynamic_inference = getattr(
                sampling_params, "skip_prompt_log_probs_for_dynamic_inference", False
            )
            assert (
                skip_prompt_log_probs_for_dynamic_inference
                or materialize_only_last_token_logits is False
            ), "Materialize only last token logits must be false for returning log probs"

        # No tokens?
        if context.active_token_count == 0:
            return None

        # Initialize attention state.
        context.initialize_attention_state()
        cuda_graph_request_count = (
            context.padded_active_request_count if context.is_decode_only() else None
        )

        # Get flat tokens, position ids.
        input_ids, position_ids = context.current_input_and_position_ids()

        model_config = get_model_config(unwrapped_model)
        inference_wrapper_config = self.inference_wrapped_model.inference_wrapper_config

        # If using symmetric kernels and we are using using nccl
        # for prefill turn off symmetric kernels
        symmetric_ar_type = model_config.symmetric_ar_type
        nccl_all_reduce_for_prefill = inference_wrapper_config.nccl_all_reduce_for_prefill
        # Turning on/off MoE padding for prefill
        moe_pad_experts_for_cuda_graph_inference = (
            inference_wrapper_config.moe_pad_experts_for_cuda_graph_inference
        )
        if moe_pad_experts_for_cuda_graph_inference:
            if context.is_decode_only():
                capacity_factor = model_config.num_moe_experts / model_config.moe_router_topk
                set_decode_expert_padding(unwrapped_model, True, capacity_factor=capacity_factor)
            else:
                set_decode_expert_padding(unwrapped_model, False)

        if nccl_all_reduce_for_prefill and symmetric_ar_type is not None:
            if context.is_decode_only():
                # Turn on symmetric all reduce when in decode mode
                unwrapped_model.set_symmetric_ar(symmetric_ar_type)
            else:
                # Turn off symmetric all reduces for prefill
                unwrapped_model.set_symmetric_ar(None)

        # Forward pass -> logits.
        with torch.inference_mode():
            logits = self.inference_wrapped_model.run_one_forward_step(
                {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
            )

        if self.model_is_pipeline_parallel:
            batch_size = context.total_request_count - context.paused_request_count
            logits_seq_len = (
                batch_size if materialize_only_last_token_logits else input_ids.shape[1]
            )
            vocab_size = inference_wrapper_config.padded_vocab_size
            logits_shape = [1, logits_seq_len, vocab_size]

            if is_pipeline_last_stage(self.pp_group):
                assert logits is not None and torch.Size(logits_shape) == logits.shape

            # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
            # and then broadcast the sampled tokens rather than broadcasting the raw logits.
            logits = broadcast_from_last_pipeline_stage(
                logits_shape,
                dtype=inference_wrapper_config.params_dtype,
                tensor=logits,
                pp_group=self.pp_group,
            )

        # This is the best place to yield control back to event loop.
        # At this point we have enqueued FW pass GPU kernels asynchronously.
        # While they are running, we can do other useful CPU work.
        # Note: This can be moved further ahead if sampling can be made
        # asynchronous.
        # Todo [Siddharth]: Can we condition the sleep on a cuda event?
        await asyncio.sleep(0)

        # Last token logits.
        if materialize_only_last_token_logits:
            # When materialize_only_last_token_logits is true, last_token_logits is
            # already called in the forward pass of GPT.
            last_token_logits = logits.squeeze(0)
        else:
            last_token_logits = context.last_token_logits(logits)

        # Sample.
        # Use padded vocab size because tokenizer vocab size might not include padding
        # to nearest power of 2.
        vocab_size = inference_wrapper_config.padded_vocab_size
        new_sample = self.sample_from_logits(
            last_token_logits, sampling_params, vocab_size=vocab_size
        )

        # Active sequence lengths.
        active_request_ids = context.request_ids[
            context.paused_request_count : context.total_request_count
        ].long()
        active_sequence_lengths = context.get_active_sequence_lengths()
        active_sequence_lengths += 1  # Account for the token we just generated
        max_sequence_lengths = context.get_max_sequence_lengths()

        # Request finished if termination_id or length >= max_sequence_length.
        active_request_mask = (new_sample != termination_id).byte() & torch.less(
            active_sequence_lengths, max_sequence_lengths
        ).byte()
        finished_idxs = (
            torch.nonzero(active_request_mask == 0, as_tuple=True)[0] + context.paused_request_count
        )
        finished_request_ids = context.request_ids[finished_idxs]

        # New sample gets updated in update_requests, so we pass in a clone
        new_sample_copy = new_sample.clone()

        log_probs = None
        if sampling_params.return_log_probs:
            log_probs = context.calculate_log_probs(
                logits, new_sample_copy, only_last_token_logits=materialize_only_last_token_logits
            )

        # Update requests.
        newly_paused_request_ids = context.update_requests(active_request_mask, new_sample_copy)

        return {
            "active_request_ids": active_request_ids,
            "newly_paused_request_ids": newly_paused_request_ids,
            "finished_request_ids": finished_request_ids,
            "sample": new_sample,
            "log_probs": log_probs,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    @torch.inference_mode()
    def generate_output_tokens_dynamic_batch(
        self, sampling_params: SamplingParams, termination_id: int
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Synchronous wrapper for `self.async_generate_output_tokens_dynamic_batch."""
        loop = get_asyncio_loop()
        return loop.run_until_complete(
            self.async_generate_output_tokens_dynamic_batch(sampling_params, termination_id)
        )

    def _update_top_n_logprobs_dict(
        self,
        top_n_logprobs_this_step: torch.Tensor,
        top_n_logprobs_indices: torch.Tensor,
        mask: torch.Tensor,
        top_n_logprobs_dict: Dict[int, List[Dict[str, float]]],
    ):
        """Function to update the top_n_logprobs at each step

        This function goes through the topn logprobs generated for each, and for whichever
        batch has started generating tokens, it updates the top_n_logprobs_dict with the
        decoded token (string) as the key and the logit as the value.
        top_n_logprobs_dict has as keys the batch idx, the values is a list, where each element
        represents a dictionary of decoded token as key and logit as value generated at each step

        Args:
            top_n_logprobs_this_step (torch.Tensor): The top n logprob values
            top_n_logprobs_indices (torch.Tensor): The indices corresponding to the top n logprobs
            mask (torch.Tensor): A mask to indicate which requests should append to the dict
            top_n_logprobs_dict (top_n_logprobs_dict): The dict to be updated
        """
        for batch_idx, (logprob_values, logprob_indices) in enumerate(
            zip(top_n_logprobs_this_step, top_n_logprobs_indices)
        ):
            if mask[batch_idx]:
                logit_dict = {}
                for logprob, logprob_index in zip(logprob_values, logprob_indices):
                    key = self.tokenizer.detokenize([logprob_index.item()])
                    logit_dict[key] = logprob.item()
                top_n_logprobs_dict[batch_idx].append(logit_dict)

    @torch.inference_mode()
    def generate_all_output_tokens_static_batch(
        self,
        active_requests: OrderedDict[int, InferenceRequest],
        active_streams: Optional[OrderedDict[str, AsyncStream]] = None,
    ) -> OrderedDict[int, InferenceRequest]:
        """Utility to generate all the output tokens and probabilities for the prompts.

        This utility generates the output tokens for a static batch. It runs the forward steps till
        all prompts complete generation, updates the status of these requests to completed, adds
        the generated result and returns these requests

        Args:
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests.

        Returns:
            OrderedDict[int, InferenceRequest]: The result for each of the incoming requests
        """
        assert all(request.prompt_tokens is not None for request in active_requests.values())

        # Perform a deep copy so that the request prompt tokens do not get modified.
        batch_prompt_tokens_list: List[List[int]] = list(
            map(
                lambda request: copy.deepcopy(request.prompt_tokens),  # type: ignore[arg-type]
                active_requests.values(),
            )
        )
        prompt_lengths_in_batch = torch.tensor(
            [len(prompt_tokens) for prompt_tokens in batch_prompt_tokens_list],
            device=torch.cuda.current_device(),
        )
        max_prompt_length_in_batch = max(prompt_lengths_in_batch)
        min_prompt_length_in_batch = min(prompt_lengths_in_batch)

        # For batch inference the sampling params are the same for all request
        sampling_params: SamplingParams = list(active_requests.values())[0].sampling_params

        # Remove Float16Module wrapper if it exists
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        model_config = get_model_config(unwrapped_model)

        # We only need an attention mask if we are exclusively doing prefill over
        # prompts of variable length
        use_attention_mask = (
            sampling_params.num_tokens_to_generate == 0
            and min_prompt_length_in_batch != max_prompt_length_in_batch
        )

        # Check whether CUDA graphs are enabled
        enable_cuda_graph = (
            model_config.cuda_graph_impl == "local"
            and "full_iteration" not in model_config.cuda_graph_scope
        )

        # Pad batch tokens if necessary
        batch_size = len(active_requests)
        max_sequence_length = max_prompt_length_in_batch + sampling_params.num_tokens_to_generate
        inference_wrapper_config = self.inference_wrapped_model.inference_wrapper_config
        inference_max_batch_size = inference_wrapper_config.inference_max_requests
        inference_max_sequence_length = inference_wrapper_config.inference_max_seq_length
        padded_batch_size = inference_max_batch_size if enable_cuda_graph else batch_size
        if padded_batch_size > inference_max_batch_size:
            raise ValueError(
                f"Padded batch size {padded_batch_size} > max batch size {inference_max_batch_size}"
            )
        padded_batch_prompt_tokens = self.pad_input_prompt_tokens(
            batch_prompt_tokens_list,
            padded_batch_size=padded_batch_size,
            padded_sequence_length=max_sequence_length,
        )

        # Verify that output sequence length is within configured limit
        if max_sequence_length > inference_max_sequence_length:
            raise MaxSequenceLengthOverflowError(
                f"Maximum allowed sequence length was set to {inference_max_sequence_length} "
                f"tokens but requested generation of {max_sequence_length} tokens"
            )

        top_n_logprobs_dict = defaultdict(list)

        # Pre allocate log probs tensor
        output_log_probs = None
        if sampling_params.return_log_probs:
            output_log_probs = torch.empty(
                (batch_size, max_sequence_length - 1),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )

        # An array to check which of the prompts have reached end of generation condition
        is_generation_done_tensor = torch.zeros(
            batch_size, dtype=torch.bool, device=torch.cuda.current_device()
        )

        # An array to act as a counter to keep track of generated sequence lengths
        generated_sequence_lengths = torch.zeros(
            batch_size, device=torch.cuda.current_device()
        ).cuda()

        # Use padded vocab size because tokenizer vocab size might not include padding
        # to nearest power of 2
        vocab_size = inference_wrapper_config.padded_vocab_size

        # Check whether early termination is enabled
        no_early_termination = getattr(sampling_params, "no_early_termination", False)
        termination_id = -1 if no_early_termination else self.tokenizer.eod

        streaming_enabled = active_streams is not None and len(active_streams) > 0
        if streaming_enabled:
            # Start a separate thread for streaming tokens to avoid blocking the
            # main computation
            streaming_idx: List[int] = [
                i
                for (i, request_id) in enumerate(active_requests.keys())
                if request_id in active_streams
            ]
            streaming_request_ids: List[int] = list(active_streams.keys())
            streams: List[AsyncStream] = list(active_streams.values())
            streaming_requests: List[InferenceRequest] = [
                active_requests[request_id] for request_id in streaming_request_ids
            ]
            streaming_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            stream_tokens = functools.partial(self.stream_tokens, sampling_params)

        for request in active_requests.values():
            # Initialize to a list to store a latency measurement for each generated token.
            request.tpot = []
        timing_events = []

        with torch.inference_mode():
            self.inference_wrapped_model.prep_model_for_inference()

            inference_input: Dict[str, Any] = self.prep_inference_input(
                prompts_tokens=padded_batch_prompt_tokens,
                active_requests=active_requests,
                use_attention_mask=use_attention_mask,
            )

            assert (
                not self.inference_wrapped_model.inference_context.is_decode_only()
            ), f"Generation must start in prefill mode"

            # Sequence parallelism is required for MoE layers when using expert parallelism (EP)
            # becausethe expert routing mechanism relies on sequence parallelism's communication
            # infrastructure to distribute tokens across expert ranks. However, sequence parallelism
            # is not currently supported for non-MoE layers during inference, so we selectively
            # disable it for all other layer types. This is safe because MoE layers perform an
            # all-gather operation on sequences before passing data to subsequent layers, ensuring
            # that each rank has the complete sequence data needed for the next non-MoE layer.
            tp_size = model_config.tensor_model_parallel_size
            ep_size = model_config.expert_model_parallel_size
            model_is_tp_ep = tp_size > 1 and ep_size > 1
            if model_is_tp_ep:
                set_model_to_sequence_parallel(
                    unwrapped_model, False, exclude_modules=[BaseMoELayer]
                )
            elif model_config.sequence_parallel and (ep_size == 1 or tp_size == 1):
                raise NotImplementedError(
                    f"Sequence parallellism is only supported for static batching with MoE models"
                )

            # If using symmetric kernels and we are using using nccl
            # for prefill turn off symmetric kernels
            symmetric_ar_type = model_config.symmetric_ar_type
            nccl_all_reduce_for_prefill = inference_wrapper_config.nccl_all_reduce_for_prefill
            if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                unwrapped_model.set_symmetric_ar(None)

            # Turning off MoE padding for prefill
            moe_pad_experts_for_cuda_graph_inference = (
                inference_wrapper_config.moe_pad_experts_for_cuda_graph_inference
            )
            if moe_pad_experts_for_cuda_graph_inference:
                set_decode_expert_padding(unwrapped_model, False)

            context_start_position = 0

            # If we are exclusively doing prefill then we can process all prompt tokens
            # together even if the prompt lengths are different
            if sampling_params.num_tokens_to_generate == 0:
                context_end_position = max_prompt_length_in_batch
            else:
                context_end_position = min_prompt_length_in_batch

            # The initial iteration of this loop runs the prefill phase up to the shortest
            # prompt length in the batch. Then every subsequent iterations runs a decode step.
            # At least one new token will be generated in each iteration. The generated token
            # will be ignored for requests which have prompt length > the current generated
            # sequence length. Similarly, the generated token is ignored for requests which
            # have maximum total sequence length < the current generated sequence length.
            while True:
                # Add a timing event at the start of each iteration. The token generation
                # time will be the elapsed time between consective timing events.
                timing_events.append(torch.cuda.Event(enable_timing=True))
                timing_events[-1].record()

                # Pick the context window that we need to pass through the network.
                inference_input_for_context_window: Dict[str, Any] = (
                    self.inference_wrapped_model.get_batch_for_context_window(
                        inference_input, context_start_position, context_end_position
                    )
                )

                # Disable attention mask when using CUDA graphs for decode
                if (
                    enable_cuda_graph
                    and self.inference_wrapped_model.inference_context.is_decode_only()
                    and "attention_mask" in inference_input_for_context_window
                ):
                    inference_input_for_context_window["attention_mask"] = None
                elif use_attention_mask:
                    assert (
                        attention_mask := inference_input_for_context_window.get(
                            "attention_mask", None
                        )
                        is not None
                    )

                # Only materialize prompt log probs if the user requests log probs
                materialize_only_last_token_logits = (
                    self.inference_wrapped_model.inference_context.is_decode_only()
                    or not (sampling_params.return_log_probs or sampling_params.top_n_logprobs > 0)
                )
                inference_context = self.inference_wrapped_model.inference_context
                inference_context.materialize_only_last_token_logits = (
                    materialize_only_last_token_logits
                )

                # Returns the final logits of shape [batch_size, context_length, vocab_size]
                # Note: This is returned in all TP ranks or last PP stage in PP models
                logits = self.inference_wrapped_model.run_one_forward_step(
                    inference_input_for_context_window
                )

                # Undo padding if necessary
                batch_prompt_tokens = self.unpad_input_prompt_tokens(
                    padded_batch_prompt_tokens, batch_size
                )
                assert batch_prompt_tokens.shape[0] == batch_size, batch_prompt_tokens.shape[0]
                if is_pipeline_last_stage(self.pp_group):
                    logits = logits[:batch_size]

                if self.model_is_pipeline_parallel:
                    context_length = context_end_position - context_start_position
                    logits_seq_len = 1 if materialize_only_last_token_logits else context_length
                    logits_shape = [batch_size, logits_seq_len, vocab_size]
                    if is_pipeline_last_stage(self.pp_group):
                        assert logits is not None and torch.Size(logits_shape) == logits.shape
                    # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
                    # and then broadcast the sampled tokens rather than broadcasting the raw logits.
                    logits = broadcast_from_last_pipeline_stage(
                        [batch_size, logits_seq_len, vocab_size],
                        dtype=inference_wrapper_config.params_dtype,
                        tensor=logits,
                        pp_group=self.pp_group,
                    )

                # Turn on symmetric all reduce kernels for decode stage
                # if we turned it off for prefill
                if (
                    context_end_position == min_prompt_length_in_batch
                    and symmetric_ar_type is not None
                    and nccl_all_reduce_for_prefill
                ):
                    if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                        unwrapped_model.set_symmetric_ar(symmetric_ar_type)

                # Indicates which of the input prompts have started generating tokens.
                # A 1D boolean tensor with [batch_size] elements (i.e) The shortest
                # prompts will start generating first and so on
                generation_started = prompt_lengths_in_batch <= context_end_position
                last_token_logits = logits[:, -1, :]

                logits_for_top_n_prompt_logprobs = (
                    logits
                    if context_start_position == 0 and sampling_params.return_prompt_top_n_logprobs
                    else None
                )
                sampled_logits = self.sample_from_logits(
                    last_token_logits,
                    sampling_params,
                    vocab_size,
                    generation_started=generation_started,
                    top_n_logprobs_dict=top_n_logprobs_dict,
                    logits=logits_for_top_n_prompt_logprobs,
                )

                if sampling_params.num_tokens_to_generate > 0:
                    # Substitute the sampled logits only for the prompts that
                    # have started generating tokens
                    batch_prompt_tokens[generation_started, context_end_position] = sampled_logits[
                        generation_started
                    ]

                # Compute log probs
                if sampling_params.return_log_probs:
                    log_probs = F.log_softmax(logits, dim=2).to(torch.float32)

                    indices = torch.unsqueeze(
                        batch_prompt_tokens[
                            :, (context_start_position + 1) : (context_end_position + 1)
                        ],
                        2,
                    )
                    # Get the log probabilities for only the prompt tokens
                    assert output_log_probs is not None
                    output_log_probs[:, context_start_position:context_end_position] = torch.gather(
                        log_probs, 2, indices
                    ).squeeze(2)

                context_start_position = context_end_position

                if sampling_params.num_tokens_to_generate > 0:
                    # Check end of generation status for each tensor
                    # and update generated sequence lengths
                    (is_generation_done_tensor, generated_sequence_lengths) = (
                        self.update_generation_status(
                            updated_prompts_tokens=batch_prompt_tokens,
                            generation_started=generation_started,
                            current_context_end_position=context_end_position,
                            is_generation_done_tensor=is_generation_done_tensor,
                            generated_sequence_lengths=generated_sequence_lengths,
                            termination_id=termination_id,
                        )
                    )

                    # Stream intermediate outputs
                    if streaming_enabled:
                        streaming_executor.submit(
                            stream_tokens,
                            streaming_request_ids,
                            streaming_requests,
                            streams,
                            generation_started[streaming_idx].cpu(),
                            is_generation_done_tensor[streaming_idx].cpu(),
                            batch_prompt_tokens[streaming_idx].cpu(),
                            prompt_lengths_in_batch[streaming_idx].cpu(),
                            generated_sequence_lengths[streaming_idx].cpu(),
                            (
                                output_log_probs[streaming_idx].cpu()
                                if output_log_probs is not None
                                else [None] * len(streaming_idx)
                            ),
                        )

                # Boolean flag indicating if all prompts are finished
                all_prompts_done = torch.all(is_generation_done_tensor)
                if all_prompts_done:
                    break

                # Change to decode mode if all prefill is complete
                if torch.all(generation_started):
                    self.inference_wrapped_model.inference_context.enable_decode_mode()
                    # Turn on padding for decode if flag set
                    if moe_pad_experts_for_cuda_graph_inference:
                        capacity_factor = (
                            model_config.num_moe_experts / model_config.moe_router_topk
                        )
                        set_decode_expert_padding(
                            unwrapped_model, True, capacity_factor=capacity_factor
                        )

                context_end_position = context_start_position + 1
                if context_end_position >= max_sequence_length:
                    break

        # Add a final timing event to compute the latency of every loop iteration
        timing_events.append(torch.cuda.Event(enable_timing=True))
        timing_events[-1].record()

        # Close all streams
        if streaming_enabled:
            streaming_executor.shutdown()
            for stream in streams:
                stream.finish()

        # Include all the generated tokens
        batch_prompt_tokens_with_generations = padded_batch_prompt_tokens[
            :batch_size, : (context_end_position + 1)
        ]
        if sampling_params.return_log_probs:
            assert output_log_probs is not None
            output_log_probs = output_log_probs[:, :context_end_position]

        generated_sequence_lengths[
            generated_sequence_lengths > sampling_params.num_tokens_to_generate
        ] = sampling_params.num_tokens_to_generate

        timing_events[-1].synchronize()
        tpot = torch.tensor(
            [
                timing_events[i].elapsed_time(timing_events[i + 1]) / 1e3
                for i in range(len(timing_events) - 1)
            ],
            dtype=torch.float32,
        )

        for idx, request in enumerate(active_requests.values()):
            input_prompt_length = int(prompt_lengths_in_batch[idx])
            # Shorter prompts might have generated more than required tokens. So we trim them down
            required_sequence_length = int(
                min(generated_sequence_lengths[idx], sampling_params.num_tokens_to_generate)
            )
            # Extract only the generated tokens
            required_result_tokens = batch_prompt_tokens_with_generations[
                idx, input_prompt_length : (input_prompt_length + required_sequence_length)
            ]
            generated_sequence_lengths = generated_sequence_lengths.to(dtype=torch.int32)
            request.generated_sequence_lengths = generated_sequence_lengths.to(dtype=torch.int32)
            request.generated_length = required_sequence_length
            request.generated_tokens = required_result_tokens

            # Record the decode latencies for only the generated tokens
            request_tpot = tpot.clone()
            # Sum up the latencies of the first prompt tokens if the
            # request prompt length > minimum prompt length
            spill_length = input_prompt_length - min_prompt_length_in_batch
            if spill_length > 0:
                spill_latency = request_tpot[:spill_length].sum()
                request_tpot = torch.cat((spill_latency.unsqueeze(0), request_tpot[spill_length:]))

            # Remove the extraneous latencies if the
            # request sequence length < maximum sequence length
            request_tpot = request_tpot[:required_sequence_length]
            request.tpot = request_tpot.tolist()

            if output_log_probs is not None:
                request.prompt_log_probs = output_log_probs[idx, : input_prompt_length - 1].tolist()
                request.generated_log_probs = output_log_probs[
                    idx,
                    input_prompt_length - 1 : (input_prompt_length + required_sequence_length - 1),
                ].tolist()
            if sampling_params.top_n_logprobs > 0:
                if sampling_params.return_prompt_top_n_logprobs:
                    assert (
                        len(top_n_logprobs_dict[idx])
                        >= input_prompt_length + required_sequence_length - 1
                    ), (
                        "Did not collect required number of top-N logprobs: "
                        f"{len(top_n_logprobs_dict[idx])}"
                    )
                    request.prompt_top_n_logprobs = top_n_logprobs_dict[idx][
                        : input_prompt_length - 1
                    ]
                    request.generated_top_n_logprobs = top_n_logprobs_dict[idx][
                        input_prompt_length
                        - 1 : (input_prompt_length + required_sequence_length - 1)
                    ]
                else:
                    assert len(top_n_logprobs_dict[idx]) >= required_sequence_length, (
                        "Did not collect required number of top-N logprobs: "
                        f"{len(top_n_logprobs_dict[idx])}"
                    )
                    request.generated_top_n_logprobs = top_n_logprobs_dict[idx][
                        :required_sequence_length
                    ]

            request.status = Status.COMPLETED

            text, segments = self.detokenize_generations(
                batch_prompt_tokens_with_generations[
                    idx, : (input_prompt_length + required_sequence_length)
                ],
                input_prompt_length + generated_sequence_lengths,
                sampling_params.return_segments,
            )
            request.text = text  # Inference server returns prompts & generations together
            if sampling_params.return_segments:
                request.segments = segments[0]
            request.generated_text = text[len(request.prompt) :]
        return active_requests

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[int, InferenceRequest],
        use_attention_mask: bool = False,
    ) -> Dict[str, Any]:
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.

        Returns:
            A dict of the inference input for the current batch.
        """
        inference_input = self.inference_wrapped_model.prep_inference_input(prompts_tokens)

        if use_attention_mask and (
            attention_mask := inference_input.get("attention_mask", None) is None
        ):
            inference_input["attention_mask"] = get_attention_mask(prompts_tokens.size(1))

        return inference_input

    def stream_tokens(
        self,
        sampling_params: SamplingParams,
        request_ids: List[int],
        requests: List[InferenceRequest],
        streams: List[AsyncStream],
        generation_started: List[bool],
        is_generation_done: List[bool],
        tokens: torch.Tensor,
        prompt_lengths: List[int],
        generated_lengths: List[int],
        output_log_probs: Union[torch.Tensor, None],
    ):
        """Asynchronously streams tokens for the given requests.

        Args:
            sampling_params (SamplingParams): The sampling parameters.
            request_ids (List[int]): The request IDs.
            request (List[InferenceRequest]): The requests.
            stream (List[AsyncStream]): The streams over which to send tokens.
            generation_started (List[bool]): Whether the decode step has started.
            is_generation_done (List[bool]): Whether generation has completed.
            tokens (torch.Tensor): The tokens for this request.
            prompt_lengths (List[int]): The number of prompt tokens for each request.
            generated_lengths (List[int]): The number of output tokens for each request.
            output_log_probs (torch.Tensor, optional): The log probs for each request.
        """

        def stream_token(
            request_id: int,
            request: InferenceRequest,
            stream: AsyncStream,
            generation_started: bool,
            is_generation_done: bool,
            tokens: torch.Tensor,
            prompt_length: int,
            generated_length: int,
            output_log_probs: Union[torch.Tensor, None],
        ):
            """Asynchronously streams a token for the given request."""

            if (
                not generation_started
                or stream.finished
                or sampling_params.num_tokens_to_generate == 0
            ):
                return

            return_segments = sampling_params.return_segments
            detokenize_streaming_text = not getattr(
                sampling_params, "no_detokenize_streaming_text", False
            )

            generated_tokens = tokens[prompt_length : prompt_length + generated_length]

            if detokenize_streaming_text:
                generated_text, generated_segments = self.detokenize_generations(
                    generated_tokens, prompt_length + generated_length, return_segments
                )
            else:
                generated_text = ""
                generated_segments = []

            if output_log_probs is not None:
                generated_log_probs = output_log_probs[
                    prompt_length - 1 : prompt_length + generated_length - 1
                ].tolist()
            else:
                generated_log_probs = None

            stream.put(
                InferenceRequest(
                    request_id=request_id,
                    prompt=request.prompt,
                    sampling_params=request.sampling_params,
                    prompt_tokens=request.prompt_tokens,
                    arrival_time=request.arrival_time,
                    status=request.status,
                    encoder_prompt=request.encoder_prompt,
                    generated_text=generated_text,
                    generated_segments=generated_segments,
                    generated_tokens=generated_tokens,
                    generated_log_probs=generated_log_probs,
                    generated_length=generated_length,
                )
            )

            if is_generation_done or generated_length == sampling_params.num_tokens_to_generate:
                stream.finish()

        ret = map(
            stream_token,
            request_ids,
            requests,
            streams,
            generation_started,
            is_generation_done,
            tokens,
            prompt_lengths,
            generated_lengths,
            output_log_probs,
        )
        list(ret)
