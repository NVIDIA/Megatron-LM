from typing import List, OrderedDict, Tuple

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.inference.inference_model_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.inference_request import InferenceRequest, Status


class SimpleTextGenerationController:
    def __init__(self, inference_wrapped_model: AbstractModelInferenceWrapper, tokenizer):
        """The basic text generation controller

        This class is responsible for tokenizing the input , running the inference, sampling and also detokenizing the output

        Args:
            inference_wrapped_model (AbstractModelInferenceWrapper): A model that is wrapped using the specs given in the abstract_model_inference_wrapper.py
            tokenizer (_type_): Tokenizer used for tokenizing and detokenizing the prompts
        """
        self.inference_wrapped_model = inference_wrapped_model
        self.tokenizer = tokenizer

        # Only for TP models both is_first_stage and is_large_stage returns True
        self.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

    def tokenize_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to tokenize the input prompts

        Args:
            prompt (str): The input prompt

        Returns:
            torch.Tensor: Returns the tokenized prompt 
        """
        return self.tokenizer.tokenize(prompt)

    def detokenize_generations(self, prompt_tokens_with_generated_tokens: torch.Tensor) -> str:
        """Detokenize the output generations

        Args:
            prompt_tokens_with_generated_tokens (torch.Tensor): The input prompt tokens plus the generated tokens

        Returns:
            str: The detokenized output
        """
        tokens = prompt_tokens_with_generated_tokens.cpu().numpy().tolist()
        return self.tokenizer.detokenize(tokens)

    def sample_from_logits(
        self,
        last_token_logits: torch.Tensor,
        common_inference_params: CommonInferenceParams,
        vocab_size: int = None,
    ) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it according to the parameters defined in common_inference_params and returns the samples

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of size [batch_size, vocab_size]
            common_inference_params (CommonInferenceParams): The paramters to use for inference
            vocab_size (int): Obtained from the tokenizer. Defaults to None

        Returns:
            torch.Tensor: 1D tensor of the sampled logits with [batch_size] elements 
        """

        top_p = common_inference_params.top_p
        top_k = common_inference_params.top_k
        temperature = common_inference_params.temperature

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Function to check which prompts have reached an end condition

        We check which prompts have reached an end condition and set the corresponding flags of the is_generation_done_tensor to True . The generated sequence lengths increases as we keep generating, until that prompts hits an eod condition. The generation started status tensor helps us determine which prompts have started generating

        Args:
            updated_prompts_tokens (torch.Tensor): The prompts tokens updated with the latest generated tokens. A tensor of shape [batch_size, max_seq_len] (i.e max_seq_len = max_prompt_len + tokens_to_generate)
            generation_started (torch.Tensor): A boolean tensor of shape [batch_size]. True indicates the prompt at that index has started generating tokens. 
            current_context_end_position (int): An intiger showing which position to extract from the prompts tokens to get the latest generated tokens. 
            is_generation_done_tensor (torch.Tensor): A boolean tensor of shape [batch_size]. True indicates the prompt at that index has reached end condition.  
            generated_sequence_lengths (torch.Tensor): A int tensor of shape [batch_size]. Each value represents the generated sequence lengths for that prompt.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns the boolean is_generation_done_tensor and the generated_sequence_lengths after updating it  
        """
        latest_samples = updated_prompts_tokens[:, current_context_end_position]
        # Make sure we are checking eod criterion only for prompts that have started generating (i.e) We only look at the generated tokenns and not the input tokens.
        reached_eod = (latest_samples == self.tokenizer.eod) & generation_started
        is_generation_done_tensor = is_generation_done_tensor | reached_eod
        # We increment generated sequence lengths when that prompt has not hit the EOD and generation has started
        generated_sequence_lengths += ~is_generation_done_tensor & generation_started

        return is_generation_done_tensor, generated_sequence_lengths

    def pad_input_prompt_tokens(
        self,
        batch_prompt_tokens_list: List[List[int]],
        max_prompt_length_in_batch: int,
        num_tokens_to_generate: int,
    ) -> torch.Tensor:
        """Method to pad input prompts

        Given a bunch of prompt tokens, we pad them such that they all have uniform length

        Args:
            batch_prompt_tokens_list (List[List[int]]): A list containing the prompt tokens
            max_prompt_length_in_batch (int): Maximum of the length of the input prompt tokens
            num_tokens_togenerate (int): The number of tokens to generate for each prompt

        Returns:
            torch.Tensor: A torch tensor of shape [bs, max_seq_len] (i.e) max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate, with extra indices for each tensor padded with mask id. 
        """
        max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate

        for prompt_tokens in batch_prompt_tokens_list:
            padding_size = max_seq_len - len(prompt_tokens)
            prompt_tokens.extend([self.tokenizer.eod] * padding_size)

        return torch.tensor(batch_prompt_tokens_list).cuda()

    def generate_output_tokens_dynamic_batch(
        self, active_requests: OrderedDict[int, InferenceRequest],
    ) -> OrderedDict[int, InferenceRequest]:
        """Utility to generate the output tokens and probabilities for the prompts

        This utility generates the output tokens for a dynamic batch. It will run one forward step at a time, and pass control back to the engine, which will update the request pool and call this method again.  
        
        Args:
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests. 

        Returns:
            OrderedDict[int, InferenceRequest]: The result for each of the incoming requests after running one forward step. 
        """
        raise Exception("Not implemented yet")

    def generate_all_output_tokens_static_batch(
        self, active_requests: OrderedDict[int, InferenceRequest],
    ) -> OrderedDict[int, InferenceRequest]:
        """Utility to generate the all the output tokens and probabilities for the prompts .

        This utility generates the output tokens for a static batch. It runs the forward steps till all prompts complete generation, updates the status of these requests to completed, adds the generated result and returns these requests

        Args:
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests. 

        Returns:
            OrderedDict[int, InferenceRequest]: The result for each of the incoming requests
        """
        batch_prompt_tokens_list = list(
            map(lambda request: request.prompt_tokens, active_requests.values())
        )
        prompt_lengths_in_batch = torch.tensor(
            [len(prompt_tokens) for prompt_tokens in batch_prompt_tokens_list]
        ).cuda()
        max_prompt_length_in_batch = max(prompt_lengths_in_batch)
        min_prompt_length_in_batch = min(prompt_lengths_in_batch)

        # For batch inference the inference params are the same for all request
        common_inference_params: CommonInferenceParams = list(active_requests.values())[
            0
        ].inference_parameters

        # max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate
        batch_prompt_tokens = self.pad_input_prompt_tokens(
            batch_prompt_tokens_list,
            max_prompt_length_in_batch=max_prompt_length_in_batch,
            num_tokens_to_generate=common_inference_params.num_tokens_to_generate,
        )
        batch_size, max_sequence_length = batch_prompt_tokens.shape

        # Pre allocate log probs tensor
        output_log_probs = None
        if common_inference_params.return_log_probs:
            output_log_probs = torch.empty(
                (batch_size, max_sequence_length - 1), dtype=torch.float32
            ).cuda()

        # An array to check which of the prompts have reached end of generation condition
        is_generation_done_tensor = torch.zeros(batch_size, dtype=torch.bool).cuda()

        # An array to act as a counter to keep track of generated sequence lengths
        generated_sequence_lengths = torch.zeros(batch_size).cuda()

        with torch.no_grad():
            self.inference_wrapped_model.prep_model_for_inference(
                prompts_tokens=batch_prompt_tokens
            )

            context_start_position = 0
            # Pick the context window that we need to pass through the network.
            for context_end_position in range(min_prompt_length_in_batch, max_sequence_length):
                
                inference_input = self.inference_wrapped_model.get_batch_for_context_window(
                    context_start_position, context_end_position
                )

                # Returns the final logits of shape [batch_size, context_length, vocab_size]
                # Note: This is returned in all TP ranks or last PP stage in PP models
                logits = self.inference_wrapped_model.run_one_forward_step(inference_input)
                if self.model_is_pipeline_parallel:
                    context_length = context_end_position - context_start_position
                    logits = broadcast_from_last_pipeline_stage(
                        [batch_size, context_length, self.tokenizer.vocab_size],
                        dtype=torch.float32,
                        tensor=logits,
                    )

                # Indicates which of the input prompts have started generating tokens. A 1D boolean tensor with [batch_size] elements (i.e) The shortest prompts will start generating first and so on
                generation_started = prompt_lengths_in_batch <= context_end_position
                last_token_logits = logits[:, -1, :]
                sampled_logits = self.sample_from_logits(
                    last_token_logits, common_inference_params, self.tokenizer.vocab_size
                )
                
                # Substitute the sampled logits only for only the prompts that have started generating tokens
                batch_prompt_tokens[generation_started, context_end_position] = sampled_logits[
                    generation_started
                ]

                if common_inference_params.return_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
                    indices = torch.unsqueeze(
                        batch_prompt_tokens[
                            :, (context_start_position + 1) : (context_end_position + 1)
                        ],
                        2,
                    )
                    # Gather the log probabilities only along the indices of the prompt tokens
                    # i.e Get the log probablitiles for the prompt tokens alone
                    output_log_probs[:, context_start_position:context_end_position] = torch.gather(
                        log_probs, 2, indices
                    ).squeeze(2)

                context_start_position = context_end_position

                # Check end of generation status for each tensor and update generated sequence lengths
                (
                    is_generation_done_tensor,
                    generated_sequence_lengths,
                ) = self.update_generation_status(
                    updated_prompts_tokens=batch_prompt_tokens,
                    generation_started=generation_started,
                    current_context_end_position=context_end_position,
                    is_generation_done_tensor=is_generation_done_tensor,
                    generated_sequence_lengths=generated_sequence_lengths,
                )
                # Boolean flag indicating if all prompts are finished
                all_prompts_done = torch.all(is_generation_done_tensor)
                if all_prompts_done:
                    break
        
        # Include all the generated tokens
        batch_prompt_tokens_with_generations = batch_prompt_tokens[:, : (context_end_position + 1)]
        if common_inference_params.return_log_probs:
            output_log_probs = output_log_probs[:, :context_end_position]

        generated_sequence_lengths[
            generated_sequence_lengths > common_inference_params.num_tokens_to_generate
        ] = common_inference_params.num_tokens_to_generate

        for idx, request in enumerate(active_requests.values()):
            input_prompt_length = int(prompt_lengths_in_batch[idx])
            # Shorter prompts might have generated more than required tokens. So we trim them down
            required_sequence_length = int(
                min(generated_sequence_lengths[idx], common_inference_params.num_tokens_to_generate)
            )
            required_result_tokens = batch_prompt_tokens_with_generations[
                idx, input_prompt_length:required_sequence_length
            ]

            request.generated_length = required_sequence_length
            request.generated_tokens = required_result_tokens
            request.generated_log_probs = (
                None
                if output_log_probs is None
                else output_log_probs[idx, input_prompt_length:required_sequence_length]
            )
            request.status = Status.COMPLETED
            request.generated_text = self.detokenize_generations(required_result_tokens)

        return active_requests
