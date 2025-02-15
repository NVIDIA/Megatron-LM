# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import concurrent
import copy
import functools
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.utils import get_model_config


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
        self.tokenizer = tokenizer

        # For models without pipeline parallelism, is_first_stage and is_last_stage returns True
        self.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

    def tokenize_prompt(
        self, prompt: str, add_BOS: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to tokenize the input prompts

        Args:
            prompt (str): The input prompt

        Returns:
            torch.Tensor: Returns the tokenized prompt
        """
        prompt_tokens = self.tokenizer.tokenize(prompt)

        if add_BOS:
            prompt_tokens = [self.tokenizer.bos] + prompt_tokens

        return prompt_tokens

    def detokenize_generations(
        self,
        tokens_gpu_tensor: torch.Tensor,
        lengths_gpu_tensor: torch.Tensor,
        detokenize_segments: bool,
    ) -> tuple[str, Optional[List[List[str]]]]:
        """Detokenize the generated tokens.

        Args:
            tokens_gpu_tensor (torch.Tensor): Tensor containing the tokens
            lengths_gpu_tensor (torch.Tensor): Tensor containing the lengths of each sequence
            detokenize_segments (bool): If True, returns individually detokenized tokens. If False,
            returns None as second element. Helpful for understanding per-token boundaries in
            generated text.

        Returns:
            tuple[str, List[str] | None]: A tuple containing:
            - str: The complete detokenized text
            - List[str] | None: List of segmented tokens if detokenize_segments is True, else None
        """
        # TODO(helenn): Unify with `detokenize_generations` from legacy textgen path

        if not detokenize_segments:
            tokens = tokens_gpu_tensor.cpu().numpy().tolist()
            return self.tokenizer.detokenize(tokens), None

        prompts_plus_generations: List[str] = []
        prompts_plus_generations_segments: List[List[str]] = []

        tokens_gpu_tensor = torch.unsqueeze(tokens_gpu_tensor, 0)
        tokens = tokens_gpu_tensor.cpu().numpy().tolist()
        lengths = lengths_gpu_tensor.cpu().numpy().tolist()

        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            detok_str = self.tokenizer.detokenize(sequence_tokens)
            prompts_plus_generations.append(detok_str)
            offsets = self.tokenizer.offsets(sequence_tokens, detok_str)
            words = [
                detok_str[start:end] for start, end in zip(offsets, offsets[1:] + [len(detok_str)])
            ]

            prompts_plus_generations_segments.append(words)

        text = self.tokenizer.detokenize(tokens[0])

        return text, prompts_plus_generations_segments

    def sample_from_logits(
        self,
        last_token_logits: torch.Tensor,
        sampling_params: Optional[SamplingParams] = None,
        vocab_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it
        according to the parameters defined in sampling_params
        and returns the samples

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of
                size [batch_size, vocab_size]
            sampling_params (SamplingParams): The parameters to use for inference.
            vocab_size (int): Obtained from the tokenizer. Defaults to None

        Returns:
            torch.Tensor: 1D tensor of the sampled logits with [batch_size] elements
        """

        if kwargs.get('common_inference_params'):
            sampling_params = kwargs['common_inference_params']

        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        temperature = sampling_params.temperature

        assert not (top_k > 0 and top_p > 0), 'Cannot have top-p and top-k both greater than zero'
        assert top_p <= 1.0, 'top-p should be in (0,1]'

        def modify_logits_for_top_k_filtering(logits, top_k):
            """Set the logits for none top-k values to -inf."""
            filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits.masked_fill_(filter_, float('-Inf'))

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
            logits.masked_fill_(filter_, float('-Inf'))

        # Greedy sampling
        if top_k == 1:
            sampled_logits = torch.argmax(last_token_logits, dim=-1)
        else:
            last_token_logits = last_token_logits.clone()
            if temperature != 1.0:
                last_token_logits.div_(temperature)

            if top_k > 1:
                assert top_k <= last_token_logits.size(1), 'top-k is larger than logit size.'
                if vocab_size:
                    assert top_k < vocab_size, 'top-k is larger than vocab size.'
                modify_logits_for_top_k_filtering(last_token_logits, top_k)

            elif top_p > 0.0:
                modify_logits_for_top_p_filtering(last_token_logits, top_p)

            # After filtering, we need to recalculate the distribution.
            probabilities = last_token_logits.softmax(dim=-1)
            sampled_logits = torch.multinomial(probabilities, num_samples=1).view(-1)

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
        latest_samples = updated_prompts_tokens[:, current_context_end_position]
        # Make sure we are checking eod criterion only for prompts that have started generating
        # (i.e) We only look at the generated tokenns and not the input tokens.
        reached_eod = (latest_samples == self.tokenizer.eod) & generation_started
        is_generation_done_tensor = is_generation_done_tensor | reached_eod
        # We increment generated sequence lengths when that prompt has not hit the
        # EOD and generation has started
        generated_sequence_lengths += ~is_generation_done_tensor & generation_started

        return is_generation_done_tensor, generated_sequence_lengths.int()

    def pad_input_prompt_tokens(
        self,
        batch_prompt_tokens_list: List[List[int]],
        max_prompt_length_in_batch: int,
        num_tokens_to_generate: int,
    ) -> torch.Tensor:
        """Method to pad input prompts

        Given a list of prompts, pad them all to uniform length

        Args:
            batch_prompt_tokens_list (List[List[int]]): A list containing the prompt tokens
            max_prompt_length_in_batch (int): Maximum of the length of the input prompt tokens
            num_tokens_togenerate (int): The number of tokens to generate for each prompt

        Returns:
            torch.Tensor: A torch tensor of shape [bs, max_seq_len] (i.e)
            max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate,
        """
        max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate

        for prompt_tokens in batch_prompt_tokens_list:
            padding_size = max_seq_len - len(prompt_tokens)
            prompt_tokens.extend([self.tokenizer.eod] * padding_size)

        return torch.tensor(batch_prompt_tokens_list, device=torch.cuda.current_device())

    def generate_output_tokens_dynamic_batch(
        self, active_requests: OrderedDict[str, InferenceRequest]
    ) -> OrderedDict[str, InferenceRequest]:
        """Utility to generate the output tokens and probabilities for the prompts

        This utility generates the output tokens for a dynamic batch. It will run one forward step
        at a time, and pass control back to the engine, which will update the request pool and call
        this method again.

        Args:
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests.

        Returns:
            OrderedDict[str, InferenceRequest]: The result for each of the incoming requests
            after running one forward step.
        """
        raise Exception("Not implemented yet")

    def generate_all_output_tokens_static_batch(
        self,
        active_requests: OrderedDict[str, InferenceRequest],
        active_streams: Optional[OrderedDict[str, AsyncStream]] = None,
    ) -> OrderedDict[str, InferenceRequest]:
        """Utility to generate the all the output tokens and probabilities for the prompts .

        This utility generates the output tokens for a static batch. It runs the forward steps till
        all prompts complete generation, updates the status of these requests to completed, adds
        the generated result and returns these requests

        Args:
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests.

        Returns:
            OrderedDict[str, InferenceRequest]: The result for each of the incoming requests
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

        # For batch inference the inference params are the same for all request
        sampling_params: SamplingParams = list(active_requests.values())[0].inference_parameters

        # max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate
        batch_prompt_tokens = self.pad_input_prompt_tokens(
            batch_prompt_tokens_list,
            max_prompt_length_in_batch=max_prompt_length_in_batch,
            num_tokens_to_generate=sampling_params.num_tokens_to_generate,
        )
        batch_size, max_sequence_length = batch_prompt_tokens.shape

        # Verify that output sequence length is within configured limit
        # TODO(ksanthanam): Raise TokenOverflowError once !2518 is merged
        inference_max_sequence_length = (
            self.inference_wrapped_model.inference_wrapper_config.inference_max_seq_length
        )
        assert max_sequence_length <= inference_max_sequence_length, (
            f"Maximum allowed sequence length was set to {inference_max_sequence_length} tokens "
            f"but requested generation of {max_sequence_length} tokens"
        )

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
        vocab_size = self.inference_wrapped_model.inference_wrapper_config.padded_vocab_size

        # Check whether CUDA graphs are enabled
        enable_cuda_graph = get_model_config(self.inference_wrapped_model.model).enable_cuda_graph

        streaming_enabled = active_streams is not None and len(active_streams) > 0
        if streaming_enabled:
            # Start a separate thread for streaming tokens to avoid blocking the
            # main computation
            streaming_idx: List[int] = [
                i
                for (i, request_id) in enumerate(active_requests.keys())
                if request_id in active_streams
            ]
            streaming_request_ids: List[str] = list(active_streams.keys())
            streams: List[AsyncStream] = list(active_streams.values())
            streaming_requests: List[InferenceRequest] = [
                active_requests[request_id] for request_id in streaming_request_ids
            ]
            streaming_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            stream_tokens = functools.partial(self.stream_tokens, sampling_params)

        with torch.no_grad():

            self.inference_wrapped_model.prep_model_for_inference(
                prompts_tokens=batch_prompt_tokens
            )

            inference_input: Dict[str, Any] = self.prep_inference_input(
                prompts_tokens=batch_prompt_tokens, active_requests=active_requests
            )

            assert (
                not self.inference_wrapped_model.inference_params.decode_mode
            ), f"Generation must start in prefill mode"

            context_start_position = 0
            # Pick the context window that we need to pass through the network.
            for context_end_position in range(min_prompt_length_in_batch, max_sequence_length):

                inference_input_for_context_window: Dict[str, Any] = (
                    self.inference_wrapped_model.get_batch_for_context_window(
                        inference_input, context_start_position, context_end_position
                    )
                )

                # Disable attention mask when using CUDA graphs for decode
                if (
                    enable_cuda_graph
                    and self.inference_wrapped_model.inference_params.decode_mode
                    and "attention_mask" in inference_input_for_context_window
                ):
                    inference_input_for_context_window["attention_mask"] = None

                # Returns the final logits of shape [batch_size, context_length, vocab_size]
                # Note: This is returned in all TP ranks or last PP stage in PP models
                logits = self.inference_wrapped_model.run_one_forward_step(
                    inference_input_for_context_window
                )

                if enable_cuda_graph:
                    create_cudagraphs()

                if self.model_is_pipeline_parallel:
                    context_length = context_end_position - context_start_position
                    logits = broadcast_from_last_pipeline_stage(
                        [batch_size, context_length, vocab_size],
                        dtype=self.inference_wrapped_model.inference_wrapper_config.params_dtype,
                        tensor=logits,
                    )

                # Indicates which of the input prompts have started generating tokens.
                # A 1D boolean tensor with [batch_size] elements (i.e) The shortest
                # prompts will start generating first and so on
                generation_started = prompt_lengths_in_batch <= context_end_position
                last_token_logits = logits[:, -1, :]
                sampled_logits = self.sample_from_logits(
                    last_token_logits, sampling_params, vocab_size
                )

                # Substitute the sampled logits only for the prompts that
                # have started generating tokens
                batch_prompt_tokens[generation_started, context_end_position] = sampled_logits[
                    generation_started
                ]

                if sampling_params.return_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
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

                # Check end of generation status for each tensor
                # and update generated sequence lengths
                (is_generation_done_tensor, generated_sequence_lengths) = (
                    self.update_generation_status(
                        updated_prompts_tokens=batch_prompt_tokens,
                        generation_started=generation_started,
                        current_context_end_position=context_end_position,
                        is_generation_done_tensor=is_generation_done_tensor,
                        generated_sequence_lengths=generated_sequence_lengths,
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
                    self.inference_wrapped_model.inference_params.enable_decode_mode()

        # Close all streams
        if streaming_enabled:
            streaming_executor.shutdown()
            for stream in streams:
                stream.finish()

        # Include all the generated tokens
        batch_prompt_tokens_with_generations = batch_prompt_tokens[:, : (context_end_position + 1)]
        if sampling_params.return_log_probs:
            assert output_log_probs is not None
            output_log_probs = output_log_probs[:, :context_end_position]

        generated_sequence_lengths[
            generated_sequence_lengths > sampling_params.num_tokens_to_generate
        ] = sampling_params.num_tokens_to_generate

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

            request.prompt_log_probs = (
                None
                if output_log_probs is None
                else output_log_probs[idx, :input_prompt_length].cpu().numpy().tolist()
            )

            request.generated_log_probs = (
                None
                if output_log_probs is None
                else output_log_probs[
                    idx,
                    input_prompt_length - 1 : (input_prompt_length + required_sequence_length - 1),
                ]
                .cpu()
                .numpy()
                .tolist()
            )
            request.status = Status.COMPLETED

            text, segments = self.detokenize_generations(
                batch_prompt_tokens_with_generations[idx],
                input_prompt_length + generated_sequence_lengths,
                sampling_params.return_segments,
            )
            request.text = text  # Inference server returns prompts & generations together
            if sampling_params.return_segments:
                request.segments = segments[0]
            request.generated_text = text[len(request.prompt) :]
        return active_requests

    def prep_inference_input(
        self, prompts_tokens: torch.Tensor, active_requests: OrderedDict[str, InferenceRequest]
    ) -> Dict[str, Any]:
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests

        Returns:
            A dict of the inference input for the current batch.
        """
        return self.inference_wrapped_model.prep_inference_input(prompts_tokens)

    def stream_tokens(
        self,
        sampling_params: SamplingParams,
        request_ids: List[str],
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
            request_ids (List[str]): The request IDs.
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
            request_id: str,
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

            if not generation_started or stream.finished:
                return

            num_tokens_to_generate = sampling_params.num_tokens_to_generate
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
                generated_log_probs = (
                    output_log_probs[prompt_length - 1 : prompt_length + generated_length - 1]
                    .cpu()
                    .numpy()
                    .tolist()
                )
            else:
                generated_log_probs = None

            stream.put(
                InferenceRequest(
                    request_id=request_id,
                    prompt=request.prompt,
                    inference_parameters=request.inference_parameters,
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

            if is_generation_done or generated_length == num_tokens_to_generate:
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
