# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import concurrent
import copy
import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts.dynamic_context import MaxSequenceLengthOverflowError
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import get_attention_mask, set_decode_expert_padding
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.utils import set_model_to_sequence_parallel
from megatron.core.utils import (
    accepts_parameter,
    get_asyncio_loop,
    get_model_config,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
    round_up_to_nearest_multiple,
    unwrap_model,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    HAVE_TE = True

except ImportError:
    HAVE_TE = False

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    mamba_state_selective_copy,
    prepare_next_forward_pass,
    rewind_kv_cache,
    verify_speculative_tokens,
)


# pylint: disable=line-too-long
class TextGenerationController:
    """The text generation controller (the main sampling loop)

    This class tokenizes the input, runs inference, samples from logits, and detokenizes the output.

    Args:
        inference_wrapped_model (AbstractModelInferenceWrapper): A model that
            is wrapped using the specs given in the abstract_model_inference_wrapper.py
        tokenizer (_type_): Tokenizer used for tokenizing and detokenizing the prompts
    """

    def __init__(self, inference_wrapped_model: AbstractModelInferenceWrapper, tokenizer):
        self.inference_wrapped_model = inference_wrapped_model
        self.model_config = self.inference_wrapped_model.model.config
        inference_config = self.inference_wrapped_model.inference_context.config
        self.tokenizer = tokenizer
        self.num_speculative_tokens = inference_config.num_speculative_tokens

        pg_collection = inference_config.pg_collection
        if pg_collection is not None:
            self.pp_group = pg_collection.pp
        else:
            self.pp_group = parallel_state.get_pipeline_model_parallel_group()

        self.model_is_pipeline_parallel = self.model_config.pipeline_model_parallel_size > 1

        # Use padded vocab size because tokenizer vocab size might pad to nearest power of 2.
        # TODO(ksanthanam): Consider deprecating this check if LLaVAModel is no longer used
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        if isinstance(unwrapped_model, LLaVAModel):
            self.vocab_size = unwrapped_model.language_model.vocab_size
        else:
            self.vocab_size = unwrapped_model.vocab_size

        self.sampling_rng = torch.Generator(device=torch.cuda.current_device())
        self.num_mtp_heads = self._get_mtp_num_heads()
        self.sampling_rng.manual_seed(self.model_config.inference_sampling_seed)

        if (
            self.model_config.cuda_graph_impl == "local"
            and self.model_config.expert_model_parallel_size > 1
            and self.model_config.transformer_impl != "inference_optimized"
        ):
            assert self.model_config.moe_pad_experts_for_cuda_graph_inference, (
                "--moe-pad-experts-for-cuda-graph-inference must be set when using "
                "CUDA graphs with expert parallelism"
            )

        if self.inference_wrapped_model.inference_context.is_dynamic_batching():
            self._init_dynamic_sampling_tensors()

    def _get_mtp_num_heads(self) -> int:
        """Get the number of MTP layers from the model config."""
        model = self.inference_wrapped_model.model
        if hasattr(model, 'config') and hasattr(model.config, 'mtp_num_layers'):
            return model.config.mtp_num_layers or 0
        return 0

    def set_stop_word_finished_ids_callback(self, callback):
        """Set a callback to get request IDs that should be marked as finished due to stop words.

        The callback should have signature: callback(active_request_ids: List[int]) -> Set[int]
        Returns a set of request IDs from active_request_ids that should be marked as finished.

        Args:
            callback: Function that returns request IDs to mark as finished.
        """
        self._get_stop_word_finished_ids_callback = callback

    def _init_dynamic_sampling_tensors(self):
        """Initialize tensors needed for dynamic sampling."""
        context = self.inference_wrapped_model.inference_context
        max_requests = context.max_requests
        if context.config.materialize_only_last_token_logits:
            # Under MTP, each decode request emits (num_speculative_tokens + 1) logit rows
            max_logits = max_requests * (self.num_speculative_tokens + 1)
        else:
            max_logits = context.max_tokens

        # Callback to get request IDs that should be marked as finished due to stop words
        self._get_stop_word_finished_ids_callback = None

        device = torch.cuda.current_device()
        logits_dtype = self.inference_wrapped_model.config.params_dtype

        self._sampling_backend = "torch"
        self._enable_cuda_graph = self.model_config.cuda_graph_impl == "local"

        # Initialize bookkeeping tensors.
        if self._enable_cuda_graph:
            self._all_logits_cuda = torch.zeros(
                (1, max_logits, self.vocab_size), dtype=logits_dtype, device=device
            )
        else:
            self._all_logits_cuda = None
        self._sampled_tokens_cuda = torch.empty(max_requests, dtype=torch.int64, device=device)

        # Used for inefficient torch sampling.
        if self._sampling_backend == "torch":
            self._torch_sampling_buckets: List[Tuple] = []

        # Cache values that are constant across inference steps.
        self._unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        self._is_last_pp_stage = is_pipeline_last_stage(self.pp_group)
        self._tp_size = get_pg_size(self.inference_wrapped_model.tp_group)
        self._sp_enabled = self.model_config.sequence_parallel and self._tp_size > 1

        self._init_mtp_sampling_tensors()

    def _init_mtp_sampling_tensors(self):
        """Pre-allocate MTP sampling tensors.

        Addresses must be stable across steps for CUDA graph capture.
        """
        if not self.num_speculative_tokens:
            self._sampled_mtp_tokens_cuda = None
            self._accepted_tokens_per_request = None
            self._last_accepted_seq_indices = None
            return

        context = self.inference_wrapped_model.inference_context
        max_requests = context.max_requests
        device = torch.cuda.current_device()
        self._sampled_mtp_tokens_cuda = torch.empty(
            [self.num_speculative_tokens, max_requests], dtype=torch.int64, device=device
        )
        self._accepted_tokens_per_request = (
            torch.ones(
                [max_requests, self.num_speculative_tokens], dtype=torch.int64, device=device
            )
            * -1
        )
        self._accepted_token_counts_per_request = torch.zeros(
            max_requests, dtype=torch.int64, device=device
        )
        self._last_accepted_seq_indices_buf = torch.empty(
            max_requests, dtype=torch.int64, device=device
        )
        self._last_accepted_seq_indices = None
        self._num_mtp_depths = min(self.num_speculative_tokens, self.num_mtp_heads)
        self._mtp_token_ids_buf = torch.empty([1, max_requests], dtype=torch.int64, device=device)
        self._mtp_position_ids_buf = torch.empty(
            [1, max_requests], dtype=torch.int64, device=device
        )

    @staticmethod
    def tokenize_prompt(tokenizer, prompt: str, add_BOS: bool = False) -> List[int]:
        """Utility to tokenize the input prompts.

        Args:
            tokenizer: The tokenizer to use.
            prompt (str): The input prompt.
            add_BOS (bool): Whether to add a BOS token.

        Returns:
            List[int]: Returns the tokenized prompt.
        """

        prompt_tokens = tokenizer.tokenize(prompt)

        if add_BOS:
            assert tokenizer.bos is not None

        while prompt_tokens and prompt_tokens[0] == tokenizer.bos:
            prompt_tokens.pop(0)

        if add_BOS:
            prompt_tokens = [tokenizer.bos] + prompt_tokens

        return prompt_tokens

    @staticmethod
    def detokenize(
        tokenizer, tokens: List[int], remove_EOD: bool = True, skip_special_tokens: bool = True
    ) -> str:
        """
        Detokenize a sequence of token IDs, optionally removing trailing EOD
        tokens and handling skip_special_tokens for different tokenizer APIs.

        Args:
            tokenizer: The tokenizer to use for detokenization.
            tokens (List[int]): The token IDs to convert back to text.
            remove_EOD (bool): Whether to remove trailing EOD tokens before
                detokenization. Defaults to True.
            skip_special_tokens (bool): Whether to remove special tokens (e.g. BOS/EOS)
                during detokenization. Only passed through if the tokenizer supports it.

        Returns:
            str: The detokenized string.
        """
        if remove_EOD and getattr(tokenizer, "eod", None) is not None:
            while tokens and tokens[-1] == tokenizer.eod:
                tokens = tokens[:-1]

        if accepts_parameter(tokenizer.detokenize, "skip_special_tokens"):
            return tokenizer.detokenize(tokens, skip_special_tokens=skip_special_tokens)
        else:
            return tokenizer.detokenize(tokens)

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
            return (
                self.detokenize(self.tokenizer, tokens, skip_special_tokens=skip_special_tokens),
                None,
            )

        prompts_plus_generations: List[str] = []
        prompts_plus_generations_segments: List[List[str]] = []
        tokens_gpu_tensor = torch.unsqueeze(tokens_gpu_tensor, 0)
        tokens = tokens_gpu_tensor.tolist()
        lengths = lengths_gpu_tensor.tolist()

        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            detok_str = self.detokenize(self.tokenizer, sequence_tokens)
            prompts_plus_generations.append(detok_str)
            offsets = self.tokenizer.offsets(sequence_tokens, detok_str)
            words = [
                detok_str[start:end] for start, end in zip(offsets, offsets[1:] + [len(detok_str)])
            ]

            prompts_plus_generations_segments.append(words)

        text = self.detokenize(self.tokenizer, tokens[0], skip_special_tokens=skip_special_tokens)

        return text, prompts_plus_generations_segments

    def _torch_sampling_func(
        self,
        last_token_logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        vocab_size: Optional[int] = None,
    ):
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it
        according to the parameters defined in sampling_params
        and returns the samples. If sampling parameters top_n_logprobs > 0
        at each step it also updates the top_n_logprobs dict.

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of
                size [batch_size, vocab_size].
            temperature (float): The temperature to use for sampling.
            top_k (int): The top-k value to use for sampling.
            top_p (float): The top-p value to use for sampling.
            vocab_size (int): Obtained from the tokenizer. Defaults to None.

        Returns:
            sampled_logits (torch.Tensor): 1D tensor with [batch_size] elements
        """
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
            # Clone needed: filter_[:, 1:] and filter_[:, :-1] are overlapping views;
            # without clone, each write would corrupt the next read during the shift.
            filter_[:, 1:] = filter_[:, :-1].clone()
            # Make sure we at least have one token to select from.
            filter_[..., 0] = 0

            # Fill in the filtered part
            filter_ = filter_.scatter(1, sorted_indices, filter_)
            logits.masked_fill_(filter_, float("-Inf"))

        # Greedy sampling
        if top_k == 1:
            sampled_logits = torch.argmax(last_token_logits, dim=-1)
        else:
            # Clone needed: .div_() and masked_fill_() below modify in-place,
            # which would mutate the caller's tensor without this clone.
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

                # If we skip prompt log_probs then we only append for generated tokens.
                # Otherwise we always append to the logprobs dict.
                if sampling_params.skip_prompt_log_probs:
                    mask = generation_started.cpu()
                else:
                    mask = torch.ones(batch_size, dtype=torch.bool)

                self._update_top_n_logprobs_dict(
                    top_n_logprobs_this_step, top_n_logprobs_indices, mask, top_n_logprobs_dict
                )
            else:
                assert not sampling_params.skip_prompt_log_probs

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

        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        temperature = sampling_params.temperature

        return self._torch_sampling_func(last_token_logits, temperature, top_k, top_p, vocab_size)

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

    def _dynamic_step_context_init(
        self,
        construct_graph_dimensions: Optional[InferenceBatchDimensions] = None,
        is_dummy_forward: bool = False,
    ):
        """Initializes the inference context for dynamic batching.

        Args:
            construct_graph_dimensions (Optional[InferenceBatchDimensions]): The graph config to use
                for constructing the cuda graphs.
            is_dummy_forward (bool): Whether we are running an expert parallel dummy forward pass

        Return:
            input_ids (Tensor): The active input IDs.
            position_ids (Tensor): The active position IDs.
        """
        context = self.inference_wrapped_model.inference_context

        # Remove Float16Module wrapper if it exists
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        model_config = get_model_config(unwrapped_model)

        # Initialize attention state.
        context.initialize_attention_state(
            construct_graph_dimensions=construct_graph_dimensions,
            is_expert_parallel_dummy_cuda_graph_step=is_dummy_forward,
        )

        # Derive the MTP padded batch size from the existing padded graph dimensions.
        # For MoE models this is post EP sync. In eager mode MTP uses locally SP-aligned
        # batch size instead.
        if context.using_cuda_graph_this_step():
            self._mtp_resolved_padded_count = context.padded_batch_dimensions.req_count
            if self._sp_enabled:
                self._mtp_resolved_padded_count = round_up_to_nearest_multiple(
                    self._mtp_resolved_padded_count, self._tp_size
                )
        else:
            self._mtp_resolved_padded_count = None

        # If using symmetric kernels and we are using using nccl
        # for prefill turn off symmetric kernels
        symmetric_ar_type = self.model_config.symmetric_ar_type
        nccl_all_reduce_for_prefill = self.model_config.nccl_all_reduce_for_prefill
        # Turning on/off MoE padding for cuda-graphs
        moe_pad_experts_for_cuda_graph_inference = (
            self.model_config.moe_pad_experts_for_cuda_graph_inference
        )
        is_inference_optimized = self.model_config.transformer_impl == "inference_optimized"
        if is_inference_optimized:
            assert not moe_pad_experts_for_cuda_graph_inference, (
                "moe_pad_experts_for_cuda_graph_inference cannot be True when "
                "transformer_impl is 'inference_optimized'"
            )
        if moe_pad_experts_for_cuda_graph_inference:
            if context.using_cuda_graph_this_step():
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

        # Get flat tokens, position ids.
        # If we are running a dummy forward step we want to use the token count agreed upon
        # by all EP ranks rather than the minimum number of tokens.
        if construct_graph_dimensions is not None and not is_dummy_forward:
            return context.current_input_and_position_ids(
                num_warmup_tokens=construct_graph_dimensions.token_count
            )
        else:
            return context.current_input_and_position_ids()

    def _dynamic_step_forward_logits(self, input_ids: Tensor, position_ids: Tensor):
        """Forward step the model to get logits for dynamic batching.

        This also handles logits-broadcasting for pipeline parallelism.

        Args:
            input_ids (Tensor): The input token IDs.
            position_ids (Tensor): The position IDs.
        """
        context = self.inference_wrapped_model.inference_context
        if context.config.materialize_only_last_token_logits:
            logits_seq_len = context.num_last_token_logits
        else:
            logits_seq_len = context.padded_active_token_count

        with torch.inference_mode():
            logits = self.inference_wrapped_model.run_one_forward_step(
                {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
            )
            # logits shape: [1, seq_len, vocab_size]

        if not context.config.materialize_only_last_token_logits:
            assert logits_seq_len == input_ids.shape[1]

        # Note: When speculative decoding is active (num_speculative_tokens > 0),
        # the model skips MTP computation during the forward pass. MTP logits
        # will be computed serially after verification to ensure they are
        # conditioned on verified tokens only.

        if self.model_is_pipeline_parallel:
            if context.config.materialize_only_last_token_logits:
                logits_seq_len = context.num_last_token_logits
            else:
                logits_seq_len = input_ids.shape[1]
            logits_shape = [1, logits_seq_len, self.vocab_size]

            if is_pipeline_last_stage(self.pp_group):
                assert logits is not None and torch.Size(logits_shape) == logits.shape

            logits = broadcast_from_last_pipeline_stage(
                logits_shape,
                dtype=self.model_config.params_dtype,
                tensor=logits,
                pp_group=self.pp_group,
            )

        # Copy logits to contiguous buffer.
        if self._enable_cuda_graph:
            self._all_logits_cuda[:, :logits_seq_len, :].copy_(logits[:, :logits_seq_len, :])
        else:
            self._all_logits_cuda = logits

    def _dynamic_step_sample_bookkeeping(self):
        """Perform bookkeeping necessary to sample logits for dynamic batching."""
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        if self._sampling_backend == "torch":
            # Bucketize the core sampling parameters.
            # Doing so via list comprehension is orders of magnitude faster than via torch.
            bucket_map = defaultdict(list)

            # Shorthands for the dictionary comprehension.
            temp = context.request_metadata["temperature"][:active_request_count].tolist()
            top_k = context.request_metadata["top_k"][:active_request_count].tolist()
            top_p = context.request_metadata["top_p"][:active_request_count].tolist()

            for request_index, (t, k, p) in enumerate(zip(temp, top_k, top_p)):
                sampling_params = (t, k, p)
                bucket_map[sampling_params].append(request_index)

            # Just unpack the key directly!
            device = torch.cuda.current_device()
            self._torch_sampling_buckets = [
                (indices, *sampling_params) for sampling_params, indices in bucket_map.items()
            ]
            # Pre-compute index tensors on GPU to avoid per-step H2D copies.
            self._torch_sampling_bucket_index_tensors = [
                torch.tensor(indices, device=device, dtype=torch.long)
                for indices, *_ in self._torch_sampling_buckets
            ]

    def _rewind_kv_cache(self) -> tuple:
        """Update the KV cache bookkeeping for speculative decoding.

        After forward pass with speculative tokens, some tokens may be rejected.
        This function "rewinds" the KV cache bookkeeping to reflect only the accepted
        tokens. The core bookkeeping is handled by a Triton kernel (one thread per
        request). Mamba hybrid-model state updates remain in PyTorch.

        Returns (blocks_to_release, remove_mask) for the caller to release blocks
        back to the allocator outside the compiled graph.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        active_request_slice = slice(0, active_request_count)

        accepted_tokens_per_request = self._accepted_token_counts_per_request[:active_request_count]

        request_in_prefill_status = context.request_in_prefill_status_tensor[active_request_slice]
        request_last_kv_block_offset = context.request_last_kv_block_offset[active_request_slice]
        request_kv_length_offsets = context.request_kv_length_offsets[active_request_slice]
        request_kv_block_counts = context.request_kv_block_counts[active_request_slice]
        request_last_kv_block_id = context.request_last_kv_block_id[active_request_slice]
        request_to_kv_block_ids = context.request_to_kv_block_ids[active_request_slice]

        # --- Triton kernel: core KV-cache rewind ---
        blocks_to_release, remove_mask = rewind_kv_cache(
            accepted_counts=accepted_tokens_per_request,
            prefill_status=request_in_prefill_status,
            last_kv_block_offset=request_last_kv_block_offset,
            kv_length_offsets=request_kv_length_offsets,
            kv_block_counts=request_kv_block_counts,
            last_kv_block_id=request_last_kv_block_id,
            kv_block_ids=request_to_kv_block_ids,
            num_speculative_tokens=self.num_speculative_tokens,
            block_size_tokens=context.block_size_tokens,
            num_active_requests=active_request_count,
        )

        # Mamba speculative rewind: copy accepted intermediate states in-place.
        if context.is_hybrid_model:
            mamba_state_idx = context.mamba_metadata.request_to_mamba_state_idx[
                active_request_slice
            ]
            mamba_state_selective_copy(
                intermediate_states=context.mamba_intermediate_conv_states,
                current_states=context.mamba_conv_states,
                prefill_status=request_in_prefill_status,
                state_idx=mamba_state_idx,
                accepted_counts=accepted_tokens_per_request,
                num_layers=context.num_mamba_layers,
            )
            mamba_state_selective_copy(
                intermediate_states=context.mamba_intermediate_ssm_states,
                current_states=context.mamba_ssm_states,
                prefill_status=request_in_prefill_status,
                state_idx=mamba_state_idx,
                accepted_counts=accepted_tokens_per_request,
                num_layers=context.num_mamba_layers,
            )

        return blocks_to_release, remove_mask

    def _sample_from_logits_2d(self, logits_2d: Tensor) -> Tensor:
        """Sample tokens from 2D logits using existing sampling parameters.

        Args:
            logits_2d (Tensor): Logits of shape [num_requests, vocab_size].

        Returns:
            Tensor: Sampled tokens of shape [num_requests].
        """
        spec_token_list = []
        for idx_tensor, (_, temp, top_k, top_p) in zip(
            self._torch_sampling_bucket_index_tensors, self._torch_sampling_buckets
        ):
            spec_token_list.append(
                self._torch_sampling_func(logits_2d[idx_tensor, :], temp, top_k, top_p)
            )

        spec_tokens = torch.empty(logits_2d.shape[0], device=logits_2d.device, dtype=torch.int64)
        for tokens, indices in zip(spec_token_list, self._torch_sampling_bucket_index_tensors):
            spec_tokens[indices] = tokens
        return spec_tokens

    def _compute_serial_mtp_and_sample(self):
        """Compute MTP logits serially after verification and sample speculative tokens.

        This ensures that MTP predictions are always conditioned on verified tokens.
        Each MTP depth receives the correctly sampled token from the previous depth
        (or the base token for depth 0) rather than stale speculative tokens from
        the previous step.

        When sequence parallelism is active, hidden states are kept in SP format
        (scattered along the first dimension) between MTP depths to avoid a
        redundant gather + scatter round-trip per depth.
        """
        nvtx_range_push("mtp-spec-decoding/serial-mtp-init")
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        active_slice = slice(0, active_request_count)

        unwrapped_model = self._unwrapped_model

        # On non-last pipeline stages, the model won't have decoder hidden states.
        has_mtp = self._is_last_pp_stage and hasattr(
            unwrapped_model, '_decoder_hidden_states_cache'
        )

        if has_mtp:
            # Get decoder hidden states at last accepted positions.
            hidden_states = unwrapped_model._decoder_hidden_states_cache

            # When SP is active the decoder output is in scattered format
            # [S/TP, B, H], but _last_accepted_seq_indices are indices into
            # the full (gathered) sequence.
            if self._sp_enabled:
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=self.inference_wrapped_model.tp_group
                )
            last_accepted_hidden = hidden_states[self._last_accepted_seq_indices, :, :]
            # Shape: [active_request_count, 1, hidden_size]
        else:
            last_accepted_hidden = None

        # Compute position IDs for the next tokens.
        # After rewind, request_kv_length_offsets has been adjusted. The actual
        # KV cache length is: adjusted_offset + processed_tokens.
        # The next position to predict starts at that cache length.
        adjusted_offsets = context.request_kv_length_offsets[active_slice]
        processed_tokens = context.request_query_lengths[active_slice]
        # Cast to int64 to match CUDA graph capture dtype expectations.
        base_position = (adjusted_offsets + processed_tokens).to(torch.int64)

        # Start with the freshly sampled base token.
        next_token_ids = self._sampled_tokens_cuda[:active_request_count].clone()
        current_hidden = last_accepted_hidden if has_mtp else None

        # Compute padding needed to make batch compatible with SP and CUDA graphs.
        if getattr(self, '_mtp_resolved_padded_count', None) is not None:
            # CUDA-graph path: use the EP-synced padded count.
            padded_count = self._mtp_resolved_padded_count
            assert not self._sp_enabled or padded_count % self._tp_size == 0
        elif has_mtp:
            # Eager path: pad only for SP alignment.
            padded_count = active_request_count
            if self._sp_enabled:
                padded_count = round_up_to_nearest_multiple(padded_count, self._tp_size)
        else:
            padded_count = active_request_count
        pad_count = padded_count - active_request_count

        # Pad hidden states and scatter for sequence parallelism.
        if has_mtp:
            current_hidden = F.pad(current_hidden, (0, 0, 0, 0, 0, pad_count))
            if self._sp_enabled:
                current_hidden = scatter_to_sequence_parallel_region(
                    current_hidden, group=self.inference_wrapped_model.tp_group
                )

        token_ids_buf = self._mtp_token_ids_buf[:, :padded_count]
        position_ids_buf = self._mtp_position_ids_buf[:, :padded_count]

        # Zero-fill padding slots so the embedding layer never sees out-of-range IDs.
        token_ids_buf[0, active_request_count:] = 0
        position_ids_buf[0, active_request_count:] = 0

        nvtx_range_pop("mtp-spec-decoding/serial-mtp-init")
        for depth in range(self._num_mtp_depths):
            nvtx_range_push(f"mtp-spec-decoding/depth-{depth}")

            token_ids_buf[0, :active_request_count] = next_token_ids
            position_ids_buf[0, :active_request_count] = base_position + depth

            mtp_logits_2d = None
            if has_mtp:
                nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/forward")
                mtp_depth = None if unwrapped_model.mtp.mtp_use_repeated_layer else depth
                current_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=current_hidden,
                    next_token_ids=token_ids_buf,
                    position_ids=position_ids_buf,
                    depth=mtp_depth,
                    eager=not context.using_cuda_graph_this_step(),
                    cache_key=(
                        ("mtp", padded_count, mtp_depth)
                        if context.using_cuda_graph_this_step()
                        else None
                    ),
                )
                nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}/forward")

                # Strip padding from logits only. Hidden states stay padded+SP
                # between depths to avoid redundant gather/scatter round-trips.
                mtp_logits = mtp_logits[:active_request_count]

                # mtp_logits: [active_request_count, 1, vocab_size]
                mtp_logits_2d = mtp_logits.squeeze(1)  # [active_request_count, vocab_size]

            # Broadcast MTP logits across pipeline stages.
            if self.model_is_pipeline_parallel:
                nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/pp-broadcast")
                mtp_logits_2d = broadcast_from_last_pipeline_stage(
                    [active_request_count, self.vocab_size],
                    dtype=self.model_config.params_dtype,
                    tensor=mtp_logits_2d,
                    pp_group=self.pp_group,
                )
                nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}/pp-broadcast")

            # Sample speculative token using the same sampling parameters.
            nvtx_range_push(f"mtp-spec-decoding/depth-{depth}/sample")
            spec_tokens = self._sample_from_logits_2d(mtp_logits_2d)
            self._sampled_mtp_tokens_cuda[depth, :active_request_count] = spec_tokens
            nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}/sample")

            # Use sampled token as input for the next depth.
            next_token_ids = spec_tokens
            nvtx_range_pop(f"mtp-spec-decoding/depth-{depth}")

        # Clean up cached hidden states.
        if has_mtp:
            del unwrapped_model._decoder_hidden_states_cache

    def _sample_speculative_logits(
        self, required_logits: Tensor, request_in_prefill_status_tensor: Tensor
    ) -> tuple:
        """Sample tokens from logits using sampling buckets.

        For torch sampling buckets: [request_indices, temp, top_k, top_p]

        Example with 5 requests:
            token_to_request_idx :              [ 0    0     0  |  1     1     1     |  2     2     2     |   3    |   4  ]
            required_logits :                   [ a5l  a6l  a7l |  b3l    b4l  b5l   |  c6l   c7l   c8l   |  d2l   | e4l  ]  # Shape [11, vocab_size]

            Sampling buckets: [[[0,2], temp1, top_k1, top_p1], [[1], temp3, top_k3, top_p3], [[3, 4], temp2, top_k2, top_p2]]

            Final output tokens : [a5s  a6s  a7s  c6s  c7s  c8s  b3s  b4s  b5s  d2s  e4s]  # Shape [11]
            (Rearranged from sampling bucket order back to input order using token_order)

        Returns:
            tuple: (output_tokens, repeats) where output_tokens has shape [total_required_tokens]
        """
        repeats = torch.where(
            request_in_prefill_status_tensor == 0, 1 + self.num_speculative_tokens, 1
        )
        token_to_request_index = torch.repeat_interleave(
            torch.arange(
                len(request_in_prefill_status_tensor),
                device=request_in_prefill_status_tensor.device,
            ),
            repeats,
        )

        output_tokens_jumbled_list = []
        token_order_list = []

        for idx_tensor, (_, temp, top_k, top_p) in zip(
            self._torch_sampling_bucket_index_tensors, self._torch_sampling_buckets
        ):
            required_indices = torch.where(torch.isin(token_to_request_index, idx_tensor))[0]
            output_tokens_jumbled_list.append(
                self._torch_sampling_func(required_logits[required_indices, :], temp, top_k, top_p)
            )
            token_order_list.append(required_indices)

        output_tokens_jumbled = torch.cat(output_tokens_jumbled_list, dim=0)
        output_tokens = torch.empty(
            len(output_tokens_jumbled),
            device=output_tokens_jumbled.device,
            dtype=output_tokens_jumbled.dtype,
        )
        token_order = torch.cat(token_order_list, dim=0)
        # Rearrange output tokens from sampling_bucket request order back to input ids order
        output_tokens[token_order] = output_tokens_jumbled

        return output_tokens, repeats

    def _verify_speculative_tokens(
        self,
        output_tokens: Tensor,
        input_tokens_required: Tensor,
        num_decode_requests: int,
        num_prefill_requests: int,
        active_request_count: int,
    ) -> tuple:
        """Verify speculative tokens against input tokens (Triton kernel)."""
        return verify_speculative_tokens(
            input_tokens=input_tokens_required,
            output_tokens=output_tokens,
            num_decode_requests=num_decode_requests,
            num_prefill_requests=num_prefill_requests,
            num_speculative_tokens=self.num_speculative_tokens,
        )

    def _dynamic_step_sample_logits_and_verify_tokens(self, input_ids: Tensor):
        """
        Sample tokens from logits for dynamic batching with speculative tokens and verify the tokens.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        request_in_prefill_status_tensor = context.request_in_prefill_status_tensor[
            :active_request_count
        ]

        # Get the logit indices for tokens that need sampling.
        # These indices are always needed for input_ids slicing and tracking
        # accepted sequence positions, even when logits are pre-sliced.
        nvtx_range_push("mtp-spec-decoding/verify/logit-indices")
        # Use pre-allocated buffer for CUDA graph compatibility.
        logits = self._all_logits_cuda
        required_logit_indices = context.speculative_required_logit_indices(logits.device)

        if context.config.materialize_only_last_token_logits:
            # last_token_logits already selected exactly the required positions.
            required_logits = logits.squeeze(0)
        else:
            required_logits = logits.squeeze(0)[
                required_logit_indices, :
            ]  # Shape [num_required, vocab_size]
        nvtx_range_pop("mtp-spec-decoding/verify/logit-indices")

        # Sample tokens from logits
        nvtx_range_push("mtp-spec-decoding/verify/sample")
        output_tokens, repeats = self._sample_speculative_logits(
            required_logits, request_in_prefill_status_tensor
        )
        nvtx_range_pop("mtp-spec-decoding/verify/sample")

        num_prefill_requests = context.num_prefill_requests
        num_decode_requests = active_request_count - num_prefill_requests

        # Verify speculative tokens against input tokens.
        nvtx_range_push("mtp-spec-decoding/verify/verify-tokens")
        input_tokens_required = input_ids[0, required_logit_indices]
        last_one_indices, accepted_tokens_mask, input_tokens_required = (
            self._verify_speculative_tokens(
                output_tokens,
                input_tokens_required,
                num_decode_requests,
                num_prefill_requests,
                active_request_count,
            )
        )
        nvtx_range_pop("mtp-spec-decoding/verify/verify-tokens")

        nvtx_range_push("mtp-spec-decoding/verify/prepare-next")
        self._prepare_speculative_tokens_for_next_forward_pass(
            num_decode_requests,
            output_tokens,
            required_logit_indices,
            last_one_indices,
            accepted_tokens_mask,
            input_tokens_required,
        )
        nvtx_range_pop("mtp-spec-decoding/verify/prepare-next")

    def _prepare_speculative_tokens_for_next_forward_pass(
        self,
        num_decode_requests: int,
        output_tokens: torch.Tensor,
        required_logit_indices: torch.Tensor,
        last_one_indices: torch.Tensor,
        accepted_tokens_mask: torch.Tensor,
        input_tokens_required: torch.Tensor,
    ):
        """Prepare accepted speculative tokens for the next forward pass (Triton kernel).

        Example:
          input_tokens_required:  [ a5  a6s  a7s |  b3   b4s  b5s  |  c6  c7s  c8s  |  d2  |  e4  ]
          Accepted tokens mask    [  1   1    0  |  1     1    1   |   1   0    0   |   1  |   1  ]
          Accepted tokens         [ [a6s  -1] | [b4s  b5s] | [-1  -1] ]  (decode only; prefill → -1)
          Accepted token counts   [     1     |      2     |     0    ]  (prefill defaults to 0)
        """
        active_request_count = last_one_indices.shape[0]
        prepare_next_forward_pass(
            num_decode_requests=num_decode_requests,
            output_tokens=output_tokens,
            required_logit_indices=required_logit_indices,
            last_one_indices=last_one_indices,
            accepted_tokens_mask=accepted_tokens_mask,
            input_tokens=input_tokens_required,
            sampled_tokens_buf=self._sampled_tokens_cuda,
            last_accepted_seq_buf=self._last_accepted_seq_indices_buf,
            accepted_tokens_per_request=self._accepted_tokens_per_request,
            accepted_token_counts=self._accepted_token_counts_per_request,
            num_speculative_tokens=self.num_speculative_tokens,
        )
        # Expose the active slice so downstream code sees the right length.
        self._last_accepted_seq_indices = self._last_accepted_seq_indices_buf[:active_request_count]

    def _dynamic_step_sample_logits(self):
        """Sample tokens from logits for dynamic batching."""
        # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
        # and then broadcast the sampled tokens rather than broadcasting the raw logits.

        # Last token logits.
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        if context.config.materialize_only_last_token_logits:
            # When materialize_only_last_token_logits is true, last_token_logits is
            # already called in the forward pass of GPT.
            required_token_logits = self._all_logits_cuda.squeeze(0)[:active_request_count, :]
        else:
            required_token_logits = context.last_token_logits(
                self._all_logits_cuda[:, : context.padded_active_token_count, :]
            )

        if self._sampling_backend == "torch":
            # Concatenate the outputs once to prevent repeated small writes.
            token_list = []
            indices_list = []

            # e.g torch sample buckets will be
            # i.e (for all unique comibnation of t, topk, topk what are the associated
            # requests indices (based on the active slices)
            # [ [req at index 0, req at index 2], t1, topk1, topp1 ]]
            # [ [req at index 1, req at index 3, req at index 4] , t2, topk2, topp2]
            for indices, temp, top_k, top_p in self._torch_sampling_buckets:
                token_list.append(
                    self._torch_sampling_func(required_token_logits[indices, :], temp, top_k, top_p)
                )
                indices_list.append(torch.tensor(indices))

            # Single write to the output tensor.
            sampled_tokens = torch.cat(token_list, dim=0)
            sampled_indices = torch.cat(indices_list, dim=0)

            self._sampled_tokens_cuda[sampled_indices] = sampled_tokens

    def _dynamic_step_log_probs_bookkeeping(self) -> Tuple[bool, bool]:
        """Perform bookkeeping necessary to compute log probs for dynamic batching.

        Returns:
            return_log_probs (bool): Whether to return the sampled log_probs.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        return (
            (context.request_metadata["return_log_probs"][:active_request_count]).any(),
            (context.request_metadata["top_n_logprobs"][:active_request_count] > 0).any(),
        )

    def _router_record_bookkeeping(self) -> Optional[np.ndarray]:
        """Collect flat routing indices for MoE router recording.

        Retrieves recorded routing decisions via the context's routing_metadata
        (which handles CUDA graph static buffers), performs the TP all-gather
        when sequence parallelism is active, strips CUDA padding, and returns
        a flat CPU numpy array aligned with the context's active-token layout.
        Must be called while context attributes are still valid (before request
        transitions).

        Returns:
            Optional[np.ndarray]: Flat routing array of shape
                [active_token_count, num_layers, topk], or None if routing
                replay is disabled or no routing data was recorded.
        """
        config = self.inference_wrapped_model.model.config
        if not config.moe_enable_routing_replay:
            return None

        # Get routing indices - use routing_metadata if available (handles CUDA graph static buffers)
        context = self.inference_wrapped_model.inference_context
        if context.moe_routing_metadata is None:
            return None

        stacked_routing = context.moe_routing_metadata.get_routing_indices()

        if stacked_routing is None:
            return None

        active_token_count = context.active_token_count

        # Get TP group for all-gather if using sequence parallelism
        # With sequence parallelism, each TP rank only sees a portion of the tokens,
        # so we need to gather routing indices across all TP ranks.
        tp_group = self.inference_wrapped_model.tp_group
        tp_size = get_pg_size(tp_group)

        # All-gather across TP group if using sequence parallelism (tp_size > 1)
        if tp_size > 1 and get_model_config(self.inference_wrapped_model.model).sequence_parallel:
            # With SP, the model processes padded_active_token_count tokens total,
            # scattered evenly across TP ranks. Each rank routes
            # padded_active_token_count // tp_size tokens through MoE layers.
            #
            # The CUDA-graph static buffer path in get_routing_indices() may return
            # a tensor sliced to active_token_count (the global unpadded count),
            # which can be larger than the per-rank valid count. Truncate to the
            # true per-rank count before the all-gather so we only gather valid
            # routing data and reconstruct the full sequence in the correct order.
            local_token_count = context.padded_active_token_count // tp_size

            stacked_routing = stacked_routing[:local_token_count]
            # gather_from_sequence_parallel_region gathers along dim 0
            # [local_token_count, num_layers, topk] -> [padded_token_count, num_layers, topk]
            stacked_routing = gather_from_sequence_parallel_region(stacked_routing, group=tp_group)

        # Slice to real tokens (remove CUDA padding), move to CPU as numpy with target dtype
        _ri_dtype = np.int16 if (config.num_moe_experts or 0) <= 32768 else np.int32
        return stacked_routing[:active_token_count].cpu().numpy().astype(_ri_dtype)

    def _dynamic_step_calculate_log_probs(self) -> Optional[Tensor]:
        """Calculate log probs from logits."""
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        # This code cannot be reached when we are using speculative decode.
        assert self.num_speculative_tokens == 0
        logits_seq_len = (
            active_request_count
            if context.config.materialize_only_last_token_logits
            else context.padded_active_token_count
        )

        return context.calculate_log_probs(
            self._all_logits_cuda[:, :logits_seq_len, :],
            self._sampled_tokens_cuda[:active_request_count],
            only_last_token_logits=context.config.materialize_only_last_token_logits,
        )

    def _dynamic_step_calculate_log_probs_speculative(self) -> Tuple[List[List[float]], Tensor]:
        """Calculate log probs from logits for speculative decoding.

        For decode requests, computes log probs for each accepted speculative token
        and the newly sampled token using the main model logits. For prefill requests,
        handles prompt log probs the same way as non-speculative decoding.

        The main model logits at position j predict the token at position j+1. So:
        - log_prob(accepted_token[j]) comes from logits at position j
        - log_prob(newly_sampled_token) comes from logits at position accepted_count

        Returns:
            Tuple of (log_probs_list, log_probs_tensor):
                log_probs_list: List of lists, one per active request, containing
                    log probs for the tokens emitted in this step.
                log_probs_tensor: Full log_softmax tensor for top-n computation.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        request_in_prefill_status_tensor = context.request_in_prefill_status_tensor[
            :active_request_count
        ]
        request_query_lengths = context.active_request_query_lengths[:active_request_count]

        num_prefill_requests = request_in_prefill_status_tensor.sum().item()
        num_decode_requests = active_request_count - num_prefill_requests

        only_last = context.config.materialize_only_last_token_logits
        # Use pre-allocated buffer for CUDA graph compatibility.
        logits = self._all_logits_cuda
        logits_squeezed = logits.squeeze(0).float()
        if only_last:
            log_probs_tensor = F.log_softmax(logits_squeezed, dim=-1)
        else:
            log_probs_tensor = F.log_softmax(logits_squeezed[: context.active_token_count], dim=-1)

        log_probs_list_decode = []

        if num_decode_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            decode_log_probs = log_probs_tensor[:decode_len].reshape(
                num_decode_requests, self.num_speculative_tokens + 1, -1
            )
            accepted_counts = self._accepted_token_counts_per_request[:num_decode_requests]

            # Build a [num_decode, num_spec+1] token ID matrix for gathering.
            # Columns 0..num_spec-1 hold accepted speculative tokens (clamped to 0
            # where rejected, since those positions will be masked out).
            # At column accepted_count[i], place the newly sampled token.
            gather_tokens = torch.zeros(
                num_decode_requests,
                self.num_speculative_tokens + 1,
                device=logits.device,
                dtype=torch.long,
            )
            gather_tokens[:, : self.num_speculative_tokens] = self._accepted_tokens_per_request[
                :num_decode_requests
            ].clamp(min=0)
            gather_tokens[
                torch.arange(num_decode_requests, device=logits.device), accepted_counts
            ] = self._sampled_tokens_cuda[:num_decode_requests]

            # Gather: [num_decode, num_spec+1]
            gathered_log_probs = decode_log_probs.gather(2, gather_tokens.unsqueeze(-1)).squeeze(-1)

            log_probs_list_decode = [
                gathered_log_probs[i, : accepted_counts[i].item() + 1].tolist()
                for i in range(num_decode_requests)
            ]

        log_probs_list_prefill = []
        if num_prefill_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            prefill_log_probs = log_probs_tensor[decode_len:]

            if only_last:
                # Only last-token logits were materialized per prefill request.
                prefill_new_tokens = self._sampled_tokens_cuda[
                    num_decode_requests:active_request_count
                ]
                selected_log_probs = prefill_log_probs[
                    torch.arange(num_prefill_requests, device=logits.device), prefill_new_tokens
                ]
                log_probs_list_prefill = [[lp.item()] for lp in selected_log_probs]
            else:
                prefill_token_ids = context.token_to_input_ids[
                    decode_len : context.active_token_count
                ].roll(-1, 0)
                prefill_query_lengths = request_query_lengths[request_in_prefill_status_tensor == 1]
                new_token_idx = prefill_query_lengths.cumsum(0) - 1
                prefill_new_tokens = self._sampled_tokens_cuda[
                    num_decode_requests:active_request_count
                ]
                prefill_token_ids[new_token_idx] = prefill_new_tokens

                prefill_token_count = context.active_token_count - decode_len
                seq_idx = torch.arange(prefill_token_count, device=logits.device)
                selected_log_probs = prefill_log_probs[seq_idx, prefill_token_ids]

                prefill_log_probs_split = selected_log_probs.cpu().split(
                    prefill_query_lengths.tolist(), dim=0
                )
                log_probs_list_prefill = [lp.tolist() for lp in prefill_log_probs_split]

        log_probs_list = log_probs_list_decode + log_probs_list_prefill

        return log_probs_list, log_probs_tensor

    def _dynamic_step_calculate_top_n_logprobs_speculative(
        self, log_probs_tensor: Tensor
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """Calculate top-n log probs for speculative decoding.

        For decode requests, computes top-n at each position that produced an
        emitted token (accepted speculative positions + the newly sampled position).
        For prefill requests, behaves identically to the non-speculative path.

        Args:
            log_probs_tensor (Tensor): Pre-computed log_softmax tensor from
                _dynamic_step_calculate_log_probs_speculative.

        Returns:
            A dictionary mapping request_idx to list of (top_n_values, top_n_indices)
            tuples, one per emitted token position.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        request_in_prefill_status_tensor = context.request_in_prefill_status_tensor[
            :active_request_count
        ]
        request_query_lengths = context.active_request_query_lengths[:active_request_count]

        num_prefill_requests = request_in_prefill_status_tensor.sum().item()
        num_decode_requests = active_request_count - num_prefill_requests

        top_n_results = {}

        if num_decode_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            decode_log_probs = log_probs_tensor[:decode_len].reshape(
                num_decode_requests, self.num_speculative_tokens + 1, -1
            )
            accepted_counts = self._accepted_token_counts_per_request[:num_decode_requests]
            top_n_per_request = context.request_metadata["top_n_logprobs"][:num_decode_requests]
            max_top_n = int(top_n_per_request.max().item())

            if max_top_n > 0:

                # Single batched topk on GPU: [num_decode, num_spec+1, max_top_n]
                topk_results = torch.topk(decode_log_probs, k=max_top_n, dim=-1)

                # Single CPU transfer instead of O(num_decode * num_spec) transfers
                topk_values_cpu = topk_results.values.cpu()
                topk_indices_cpu = topk_results.indices.cpu()

                for i in range(num_decode_requests):
                    top_n = int(top_n_per_request[i].item())
                    if top_n > 0:
                        num_valid = accepted_counts[i].item() + 1
                        top_n_results[i] = [
                            (topk_values_cpu[i, j, :top_n], topk_indices_cpu[i, j, :top_n])
                            for j in range(num_valid)
                        ]

        if num_prefill_requests > 0:
            only_last = context.config.materialize_only_last_token_logits
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            prefill_log_probs = log_probs_tensor[decode_len:]

            # Batch metadata reads: single CPU transfer for all prefill requests.
            prefill_top_n = context.request_metadata["top_n_logprobs"][
                num_decode_requests:active_request_count
            ].tolist()
            max_top_n_prefill = int(max(prefill_top_n)) if prefill_top_n else 0

            if max_top_n_prefill > 0:
                if only_last:
                    # One logit row per prefill request — single batched topk.
                    topk_results_prefill = torch.topk(
                        prefill_log_probs, k=max_top_n_prefill, dim=-1
                    )
                    topk_vals_cpu = topk_results_prefill.values.cpu()
                    topk_idxs_cpu = topk_results_prefill.indices.cpu()

                    for i in range(num_prefill_requests):
                        top_n = int(prefill_top_n[i])
                        if top_n > 0:
                            req_idx = num_decode_requests + i
                            top_n_results[req_idx] = [
                                (topk_vals_cpu[i, :top_n], topk_idxs_cpu[i, :top_n])
                            ]
                else:
                    prefill_query_lengths = request_query_lengths[
                        request_in_prefill_status_tensor == 1
                    ]
                    prefill_log_probs_per_request = prefill_log_probs.split(
                        prefill_query_lengths.tolist(), dim=0
                    )
                    prefill_skip_prompt = context.request_metadata["skip_prompt_log_probs"][
                        num_decode_requests:active_request_count
                    ].tolist()

                    for i in range(num_prefill_requests):
                        top_n = int(prefill_top_n[i])
                        if top_n > 0:
                            req_idx = num_decode_requests + i
                            request_lp = prefill_log_probs_per_request[i]
                            skip_prompt = bool(prefill_skip_prompt[i])

                            if skip_prompt and request_lp.size(0) > 1:
                                top_n_logits = torch.topk(request_lp[-1], k=top_n)
                                top_n_results[req_idx] = [
                                    (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                                ]
                            else:
                                top_n_logits = torch.topk(request_lp, k=top_n, dim=-1)
                                top_n_values_cpu = top_n_logits.values.cpu()
                                top_n_indices_cpu = top_n_logits.indices.cpu()
                                top_n_results[req_idx] = [
                                    (top_n_values_cpu[t], top_n_indices_cpu[t])
                                    for t in range(request_lp.size(0))
                                ]

        return top_n_results if top_n_results else None

    def _dynamic_step_calculate_top_n_logprobs(
        self, log_probs_tensor: Optional[Tensor] = None
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """Calculate top-n log probs from logits for dynamic batching.

        Args:
            log_probs_tensor (Optional[Tensor]): Pre-computed log probabilities tensor.
                If provided, avoids recomputing log_softmax. Should be the tensor
                returned by calculate_log_probs.

        Returns:
            A dictionary mapping request_idx to list of (top_n_logprobs, top_n_indices) tuples.
            Each tuple in the list represents one token position.
        """
        assert log_probs_tensor is not None, (
            "log_probs_tensor must be provided. This should be guaranteed by the calling code "
            "computing log_probs when return_top_n_logprobs is True."
        )

        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        # Handle decode-only mode (only last token)
        if context.config.materialize_only_last_token_logits or context.is_decode_only():
            # In decode mode or when only last token logits are materialized,
            # logits already represent only the last tokens
            log_probs = log_probs_tensor[:active_request_count]

            top_n_results = {}
            for req_idx in range(active_request_count):
                top_n = int(context.request_metadata["top_n_logprobs"][req_idx].item())
                if top_n > 0:
                    # Get top-n logprobs and indices for this request (single token)
                    top_n_logits = torch.topk(log_probs[req_idx], k=top_n)
                    top_n_results[req_idx] = [
                        (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                    ]
            return top_n_results if top_n_results else None

        # Handle prefill mode - need to extract top-n for tokens per request
        # This follows the same pattern as calculate_log_probs in dynamic_context.py
        # Note: logits may be padded, so we only take the first active_token_count tokens
        log_probs = log_probs_tensor[: context.active_token_count]

        active_query_lengths = context.active_request_query_lengths[:active_request_count]

        # Split log_probs across request boundaries
        # log_probs has shape [active_token_count, vocab_size]
        log_probs_per_request = log_probs.split(active_query_lengths.tolist(), dim=0)

        top_n_results = {}
        for req_idx in range(active_request_count):
            top_n = int(context.request_metadata["top_n_logprobs"][req_idx].item())
            if top_n > 0:
                request_log_probs = log_probs_per_request[
                    req_idx
                ]  # [num_tokens_for_request, vocab_size]
                skip_prompt = bool(
                    context.request_metadata["skip_prompt_log_probs"][req_idx].item()
                )

                # If skip_prompt_log_probs is True, only compute for last token
                if skip_prompt and request_log_probs.size(0) > 1:
                    # Only compute top-n for the last token (first generated token)
                    top_n_logits = torch.topk(request_log_probs[-1], k=top_n)
                    top_n_results[req_idx] = [
                        (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                    ]
                else:
                    # Compute top-n for all tokens in the request
                    top_n_per_token = []
                    for token_idx in range(request_log_probs.size(0)):
                        top_n_logits = torch.topk(request_log_probs[token_idx], k=top_n)
                        top_n_per_token.append(
                            (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                        )
                    top_n_results[req_idx] = top_n_per_token

        return top_n_results if top_n_results else None

    def dummy_forward(self):
        """Perform a dummy forward pass. This is used in expert model parallelism
        on ranks that do not have any real requests. It may run in eager mode."""

        context = self.inference_wrapped_model.inference_context
        # if no cuda graphs, directly use dummy forward
        if not context.cuda_graph_batch_dimensions_list:
            self.inference_wrapped_model.dummy_forward()

            # Disable MoE padding for MTP computation.
            # No CUDA graphs in this path (cuda_graph_batch_dimensions_list is empty).
            if self.model_config.moe_pad_experts_for_cuda_graph_inference:
                unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
                set_decode_expert_padding(unwrapped_model, False)

            self._dummy_serial_mtp_forward()

            return

        # attempt to use cuda-graph if possible
        input_ids, position_ids = self._dynamic_step_context_init(is_dummy_forward=True)

        # _dynamic_step_context_init tries to find a cuda-graph that is compatible
        # with all EP ranks. It can also return no match, in which case
        # we run in eager mode.

        if context.using_cuda_graph_this_step():
            # we found a cuda-graph to run
            self._dynamic_step_forward_logits(input_ids, position_ids)
        else:
            # fallback to eager dummy forward
            self.inference_wrapped_model.dummy_forward()

        # Disable MoE padding for MTP computation, unless CUDA graphs
        # are active (the graphs were captured with padding enabled).
        if self.model_config.moe_pad_experts_for_cuda_graph_inference:
            if not context.using_cuda_graph_this_step():
                unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
                set_decode_expert_padding(unwrapped_model, False)

        # When speculative decoding is active, the real EP ranks perform serial
        # MTP forward passes after the main forward pass. MTP layers may contain
        # MoE sublayers (inherited from the decoder spec), which require EP
        # all-to-all collectives. The dummy rank must participate in these
        # collectives to avoid a hang.
        self._dummy_serial_mtp_forward()

        # clear the context of any temporary state from the dummy forward
        context.reset()

    @torch.inference_mode()
    def _dummy_serial_mtp_forward(self):
        """Run dummy MTP forward passes to participate in EP collectives.

        When speculative decoding is active and MTP layers contain MoE sublayers
        (inherited from the decoder layer spec), each serial MTP step triggers
        EP all-to-all collectives. The dummy EP rank must issue matching
        collective calls so the real ranks do not hang.

        This mirrors the structure of ``_compute_serial_mtp_and_sample``:
        - On the last PP stage (where MTP resides): run ``compute_mtp_single_step``
          with dummy tensors so the MoE all-to-all is executed.
        - When PP > 1: participate in the ``broadcast_from_last_pipeline_stage``
          that the real ranks also perform.
        """
        if self.num_speculative_tokens == 0 or self.num_mtp_heads == 0:
            return
        if self.model_config.expert_model_parallel_size <= 1:
            return

        unwrapped_model = self._unwrapped_model

        has_mtp = self._is_last_pp_stage and hasattr(
            unwrapped_model, '_decoder_hidden_states_cache'
        )
        if not has_mtp and not self.model_is_pipeline_parallel:
            # No MTP on this rank and no PP broadcast to participate in.
            return

        device = torch.cuda.current_device()
        dtype = self.model_config.params_dtype
        hidden_size = self.model_config.hidden_size

        # Use precomputed MTP CUDA graph batch size when available;
        # otherwise use minimal SP-compatible size.
        if getattr(self, '_mtp_resolved_padded_count', None) is not None:
            padded_count = self._mtp_resolved_padded_count
            assert not self._sp_enabled or padded_count % self._tp_size == 0
        elif has_mtp:
            # Eager path: use TP-aligned minimum size for dummy tensors.
            padded_count = self._tp_size if self._sp_enabled else 1

        dummy_hidden = None
        if has_mtp:
            # Minimal dummy tensors to drive the MTP layer forward
            # so that the MoE all-to-all collectives are issued.
            dummy_hidden = torch.zeros((padded_count, 1, hidden_size), device=device, dtype=dtype)
            if self._sp_enabled:
                dummy_hidden = scatter_to_sequence_parallel_region(
                    dummy_hidden, group=self.inference_wrapped_model.tp_group
                )
            dummy_token_ids = torch.zeros((1, padded_count), device=device, dtype=torch.long)
            dummy_position_ids = torch.zeros((1, padded_count), device=device, dtype=torch.long)

        context = self.inference_wrapped_model.inference_context

        for depth in range(self._num_mtp_depths):
            nvtx_range_push(f"mtp-spec-decoding/dummy-depth-{depth}")
            mtp_logits_2d = None
            if has_mtp:
                mtp_depth = None if unwrapped_model.mtp.mtp_use_repeated_layer else depth
                dummy_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=dummy_hidden,
                    next_token_ids=dummy_token_ids,
                    position_ids=dummy_position_ids,
                    depth=mtp_depth,
                    eager=not context.using_cuda_graph_this_step(),
                    cache_key=(
                        ("mtp", padded_count, mtp_depth)
                        if context.using_cuda_graph_this_step()
                        else None
                    ),
                )
                mtp_logits_2d = mtp_logits.squeeze(1)  # [padded_count, vocab_size]

            # Match the PP broadcast that real ranks do in _compute_serial_mtp_and_sample.
            if self.model_is_pipeline_parallel:
                broadcast_from_last_pipeline_stage(
                    [padded_count, self.vocab_size],
                    dtype=dtype,
                    tensor=mtp_logits_2d,
                    pp_group=self.pp_group,
                )
            nvtx_range_pop(f"mtp-spec-decoding/dummy-depth-{depth}")

    def _dynamic_step_context_bookkeeping(self) -> Dict[str, Tensor]:
        """Update the dynamic inference context after sampling.

        Args:
            new_sample (Tensor): The newly sampled tokens.
            request_metadata (Optional[Dict[str, Tensor]]): An override for the tensors
                that manage request metadata, such as sampling parameters. By default, this
                metadata is retrieved from the context.

        Return:
            Dict [str, Tensor]: A dictionary containing:
                active_request_ids (Tensor): Current active request IDs.
                newly_paused_request_ids (Tensor): Newly paused request IDs.
                finished_request_ids (Tensor): Finished request IDs.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        # Active sequence lengths.
        # Use the snapshot taken during build_active_slices.
        active_request_ids = context.active_request_ids[:active_request_count]
        active_sequence_lengths = context.get_active_sequence_lengths()

        # After the forward pass and KV-cache rewind, get_active_sequence_lengths()
        # returns kv_offsets + query_lengths which already includes all accepted
        # speculative tokens (they were part of the query and survived the rewind).
        # Only the newly sampled base token is not yet in the KV cache, so add 1.
        active_sequence_lengths += 1
        max_sequence_lengths = context.get_max_sequence_lengths()

        # Request finished if termination_id or length >= max_sequence_length.
        # Note: termination_id tensor has per-request termination IDs from mixed sampling
        active_request_mask = (
            self._sampled_tokens_cuda[:active_request_count]
            != context.request_metadata["termination_id"][:active_request_count]
        ).byte() & torch.less(active_sequence_lengths, max_sequence_lengths).byte()

        # Mark requests as finished if they hit stop words
        # (detected in previous step's post_process_requests)
        if self._get_stop_word_finished_ids_callback is not None:
            request_ids_list = active_request_ids.tolist()
            stop_word_finished_ids = self._get_stop_word_finished_ids_callback(request_ids_list)
            if stop_word_finished_ids:
                for idx, request_id in enumerate(request_ids_list):
                    if request_id in stop_word_finished_ids:
                        active_request_mask[idx] = 0

        finished_idxs = torch.nonzero(active_request_mask == 0, as_tuple=True)[0]
        finished_request_ids = context.active_request_ids[finished_idxs]

        # Save block IDs for finished requests before update_requests releases them.
        # Needed for per-block routing reconstruction in the engine.
        finished_routing_block_ids = {}
        if context.kv_block_allocator.block_routing and finished_idxs.numel() > 0:
            for fidx in finished_idxs.tolist():
                req_id = int(context.request_ids[fidx].item())
                blocks = context.request_to_kv_block_ids[fidx]
                valid = blocks[blocks >= 0].tolist()
                if valid:
                    finished_routing_block_ids[req_id] = valid

        # Clone needed: update_requests mutates next_tokens in-place via tensor_swap,
        # which would corrupt the reused _sampled_tokens_cuda buffer.
        new_sample_copy = self._sampled_tokens_cuda[:active_request_count].clone()

        # Update requests.
        # _sampled_mtp_tokens_cuda has shape [num_speculative_tokens, max_requests]
        if self.num_speculative_tokens > 0:
            sampled_mtp_tokens_cuda = self._sampled_mtp_tokens_cuda[:, :active_request_count]
        else:
            sampled_mtp_tokens_cuda = None
        update_result = context.update_requests(
            active_request_mask, new_sample_copy, sampled_mtp_tokens_cuda
        )

        return {
            "active_request_ids": active_request_ids,
            "finished_request_ids": finished_request_ids,
            "finished_routing_block_ids": finished_routing_block_ids,
            **(update_result or {}),
        }

    async def async_generate_output_tokens_dynamic_batch(
        self, skip_bookkeeping: Optional[bool] = False
    ) -> Optional[Dict]:
        """Forward step the model and update the inference context.

        Args:
            skip_bookkeeping (Optional[bool]): If true, skip the context bookkeeping step.

        Return:
            (Optional[Dict]): A dictionary containing:
                active_request_ids (Tensor): Current active request IDs.
                newly_paused_request_ids (Tensor): Newly paused request IDs.
                finished_request_ids (Tensor): Finished request IDs.
                sample (Tensor): New sample.
                log_probs (Optional[Tensor]): Log probabilities of the new sample, if requested.
                cuda_graph_request_count (Optional[int]): Size of cuda graph used for this step.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        # No tokens and no active requests?
        if context.active_token_count == 0 and active_request_count == 0:
            return None

        with torch.inference_mode():
            input_ids, position_ids = self._dynamic_step_context_init()

            cuda_graph_request_count = (
                context.padded_active_request_count
                if context.using_cuda_graph_this_step()
                else None
            )

            # Enable routing recording before forward pass if routing replay is enabled
            config = self.inference_wrapped_model.model.config
            if config.moe_enable_routing_replay:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

            # Forward pass produces only base logits. When speculative decoding is
            # active, MTP logits are computed serially after verification.
            self._dynamic_step_forward_logits(input_ids, position_ids)

            # Commit Mamba intermediate states before update_requests, which
            # may swap request indices. The Python lists tracking EOS block IDs
            # and intermediate offsets are not swapped along with tensors, so
            # commit must run while indices are still valid.
            if context.is_hybrid_model and context.mamba_slot_allocator is not None:
                context.mamba_slot_allocator.commit_intermediate_states()

            # Collect flat routing indices and scatter them into per-block storage.
            # Must be done before update_requests while token-to-block mappings are valid.
            # Reconstruction happens from blocks at request completion.
            context.kv_block_allocator.store_routing_per_block(self._router_record_bookkeeping())

        # This is the best place to yield control back to event loop.
        # At this point we have enqueued FW pass GPU kernels asynchronously.
        # While they are running, we can do other useful CPU work.
        # Note: This can be moved further ahead if sampling can be made
        # asynchronous.
        # Todo [Siddharth]: Can we condition the sleep on a cuda event?
        # NOTE [TDE]: This will be moved once CPU and GPU methods are separated.
        await asyncio.sleep(0)

        with torch.inference_mode():
            return_log_probs, return_top_n_logprobs = self._dynamic_step_log_probs_bookkeeping()

            self._dynamic_step_sample_bookkeeping()

            if self.num_speculative_tokens > 0:
                # Phase 1: Verify speculative tokens using base logits only.
                nvtx_range_push("mtp-spec-decoding/verify")
                self._dynamic_step_sample_logits_and_verify_tokens(input_ids)
                nvtx_range_pop("mtp-spec-decoding/verify")
                # Phase 2: Rewind KV cache for rejected tokens.
                nvtx_range_push("mtp-spec-decoding/rewind-kv-cache")
                blocks_to_release, remove_mask = self._rewind_kv_cache()
                nvtx_range_pop("mtp-spec-decoding/rewind-kv-cache")

                # Disable MoE padding for MTP computation, unless CUDA graphs
                # are active (the graphs were captured with padding enabled).
                if self.model_config.moe_pad_experts_for_cuda_graph_inference:
                    if not context.using_cuda_graph_this_step():
                        set_decode_expert_padding(self._unwrapped_model, False)

                # Phase 3: Compute MTP serially with correct (verified) inputs.
                nvtx_range_push("mtp-spec-decoding/serial-mtp")
                self._compute_serial_mtp_and_sample()
                nvtx_range_pop("mtp-spec-decoding/serial-mtp")

                # Phase 4: Release freed blocks. Deferred from Phase 2 so the
                # data-dependent boolean-mask sync overlaps with MTP GPU work.
                context.kv_block_allocator.release_memory_blocks(blocks_to_release[remove_mask])
            else:
                self._dynamic_step_sample_logits()

            log_probs = None
            top_n_logprobs = None
            if return_log_probs or return_top_n_logprobs:
                if self.num_speculative_tokens > 0:
                    log_probs, log_probs_tensor = (
                        self._dynamic_step_calculate_log_probs_speculative()
                    )
                    if return_top_n_logprobs:
                        top_n_logprobs = self._dynamic_step_calculate_top_n_logprobs_speculative(
                            log_probs_tensor
                        )
                else:
                    log_probs, log_probs_tensor = self._dynamic_step_calculate_log_probs()
                    if return_top_n_logprobs:
                        top_n_logprobs = self._dynamic_step_calculate_top_n_logprobs(
                            log_probs_tensor
                        )

            if skip_bookkeeping:
                request_bookkeeping = {}
            else:
                request_bookkeeping = self._dynamic_step_context_bookkeeping()

            ret = {
                # Clone needed: _sampled_tokens_cuda is a reused buffer overwritten each step.
                "sample": self._sampled_tokens_cuda[:active_request_count].clone(),
                "accepted_tokens": (
                    # Clone needed: .fill_(-1) on line 1480 would corrupt the returned value.
                    self._accepted_tokens_per_request.clone()
                    if self.num_speculative_tokens > 0
                    else None
                ),
                "log_probs": log_probs,
                "top_n_logprobs": top_n_logprobs,
                "cuda_graph_request_count": cuda_graph_request_count,
            }
            if self.num_speculative_tokens > 0:
                self._accepted_tokens_per_request.fill_(-1)
                self._accepted_token_counts_per_request.fill_(0)
            ret.update(request_bookkeeping)
            return ret

    @torch.inference_mode()
    def generate_output_tokens_dynamic_batch(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Optional[Dict]:
        """Synchronous wrapper for `self.async_generate_output_tokens_dynamic_batch."""
        loop = get_asyncio_loop(loop)
        return loop.run_until_complete(self.async_generate_output_tokens_dynamic_batch())

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
            and CudaGraphScope.full_iteration not in model_config.cuda_graph_scope
        )

        # Pad batch tokens if necessary
        batch_size = len(active_requests)
        max_sequence_length = max_prompt_length_in_batch + sampling_params.num_tokens_to_generate
        context = self.inference_wrapped_model.inference_context
        assert isinstance(context, StaticInferenceContext)
        inference_max_batch_size = context.max_batch_size
        inference_max_sequence_length = context.max_sequence_length
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
            symmetric_ar_type = self.model_config.symmetric_ar_type
            nccl_all_reduce_for_prefill = self.model_config.nccl_all_reduce_for_prefill
            if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                unwrapped_model.set_symmetric_ar(None)

            # Turning off MoE padding for prefill
            moe_pad_experts_for_cuda_graph_inference = (
                self.model_config.moe_pad_experts_for_cuda_graph_inference
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
                inference_context.config.materialize_only_last_token_logits = (
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
                    logits_shape = [batch_size, logits_seq_len, self.vocab_size]
                    if is_pipeline_last_stage(self.pp_group):
                        assert logits is not None and torch.Size(logits_shape) == logits.shape
                    # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
                    # and then broadcast the sampled tokens rather than broadcasting the raw logits.
                    logits = broadcast_from_last_pipeline_stage(
                        [batch_size, logits_seq_len, self.vocab_size],
                        dtype=self.model_config.params_dtype,
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
                    if context_start_position == 0 and not sampling_params.skip_prompt_log_probs
                    else None
                )
                sampled_logits = self.sample_from_logits(
                    last_token_logits,
                    sampling_params,
                    self.vocab_size,
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
                if not sampling_params.skip_prompt_log_probs:
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
