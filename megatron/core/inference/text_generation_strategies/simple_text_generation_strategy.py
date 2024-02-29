from typing import List, Tuple
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.communication_utils import copy_from_last_to_first_pipeline_stage, synchronize_list_across_all_ranks, synchronize_tensor_across_all_ranks
from megatron.core.inference.text_generation_strategies.abstract_text_generation_strategy import AbstractTextGenerationStrategy
import torch
import torch.nn.functional as F

from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.global_vars import get_num_microbatches
from megatron.core import parallel_state

class SimpleTextGenerationStrategy(AbstractTextGenerationStrategy):
    def __init__(self, model:callable, tokenizer):
        """The basic text generation strategy

        This class is responsible for tokenizing the input , running the inference and also detokenizing the output

        Args:
            model (callable): A callable instance (Can be a megatron model or a wrapped model with __call__ implemented)
            tokenizer (_type_): Tokenizer used for tokenizing and detokenizing the prompts
        """
        self.model = model
        self.tokenizer = tokenizer

    def tokenize_and_pad_input_prompts(self, prompts: List[str], num_tokens_to_generate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to tokenize and pad the input prompts

        Tokenizes the input prompts, pads them to required length and returns the tokenized tensor and also the original prompt lengths. 

        Args:
            prompts (List[str]): A list of the prompts as strings
            num_tokens_to_generate (int): The number of output tokens to generate for the prompts 

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the padded and tokenized prompts of dimension [batch_size, max_seq_length] (i.e max_seq_length = max prompt len + num_tokens_to_generate) and 1D tensor containing the lenghts of each prompt
        """        
        tokenizer = self.tokenizer
        sizes_list = None
        prompts_tokens_tensor = None
        prompts_length_tensor = None


        if torch.distributed.get_rank() == 0:
            # tokenize
            prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]
            prompts_lengths = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
            max_prompt_len = max(prompts_lengths)
            
            samples_length = max_prompt_len + num_tokens_to_generate

            # padding
            for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_lengths):
                padding_size = samples_length - prompt_length
                prompt_tokens.extend([tokenizer.eod] * padding_size)

            prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.long, device='cuda')
            prompts_length_tensor = torch.tensor(prompts_lengths, dtype=torch.long, device='cuda')

            sizes_list = [prompts_tokens_tensor.size(0), # batch_size
                      prompts_tokens_tensor.size(1)] # max_seq_length (max prompt len + num_tokens_to_generate)

        # Synchronize the prompt tokens and lengths tensor across all gpus  
        sizes_tensor = synchronize_list_across_all_ranks(size = 2, list_values=sizes_list, dtype=torch.int64)

        sizes = sizes_tensor.tolist()
        prompts_tokens_tensor = synchronize_tensor_across_all_ranks(
            sizes, torch.int64, tensor=prompts_tokens_tensor)
        prompts_length_tensor = synchronize_tensor_across_all_ranks(
            sizes[0], torch.int64, tensor=prompts_length_tensor) 
    
        return prompts_tokens_tensor , prompts_length_tensor
    

    def build_attention_mask_and_position_ids(self, prompts_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the full attention mask and position ids for the input tokens

        Args:
            tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The attention mask of shape [1, 1, max_seq_len, max_seq_len] and position ids of shape [batch_size, max_seq_len]
        """
        seq_length = prompts_tokens.size(1)
        attention_mask = torch.tril(torch.ones(
        (1, seq_length, seq_length), device=prompts_tokens.device)).view(
            1, 1, seq_length, seq_length)  
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=prompts_tokens.device).unsqueeze(0).expand_as(prompts_tokens)    
        return attention_mask, position_ids  

    def sanity_check_inference_params(self, common_inference_params:CommonInferenceParams):
        """Sanity checking the common inference parameters 

        Args:
            common_inference_params (CommonInferenceParams): The inference parameters
        """    
        if common_inference_params.use_greedy:
            assert common_inference_params.top_k == 0, 'Cannot use greedy sampling and have top_k greater than 0'
            assert common_inference_params.top_p == 0, 'Cannot use greedy sampling and have top_p greater than 0'
        
        if common_inference_params.top_k > 0:
            assert common_inference_params.top_p == 0, 'Cannot have a non zero top_k and top_p value. Set one of these to zero.'
        
        assert common_inference_params.top_p <= 1.0, 'top-p should be in (0, 1].'

    def sample_from_logits(self, last_token_logits:torch.Tensor, common_inference_params:CommonInferenceParams, vocab_size:int) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it according to the parameters defined in common_inference_params and returns the samples

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of size [batch_size, vocab_size]
            common_inference_params (CommonInferenceParams): The paramters to use for inference
            vocab_size (int): Obtained from the tokenizer. 

        Returns:
            torch.Tensor: 1D tensor of the sampled logits with [batch_size] elements 
        """

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

        self.sanity_check_inference_params(common_inference_params=common_inference_params)

        if common_inference_params.top_k == 1:
            sampled_logits = torch.argmax(last_token_logits, dim=-1)
        else:
            last_token_logits = last_token_logits.clone()
            if common_inference_params.temperature != 1.0:
                last_token_logits.div_(common_inference_params.temperature)

            if common_inference_params.top_k > 1:
                assert common_inference_params.top_k <= last_token_logits.size(1), 'top-k is larger than logit size.'
                if vocab_size:
                    assert common_inference_params.top_k < vocab_size, 'top-k is larger than vocab size.'
                modify_logits_for_top_k_filtering(last_token_logits, common_inference_params.top_k)

            elif common_inference_params.top_p > 0.0:               
                modify_logits_for_top_p_filtering(last_token_logits, common_inference_params.top_p)

            # After filtering, we need to recalculate the distribution.
            probabilities = last_token_logits.softmax(dim=-1)
            sampled_logits = torch.multinomial(probabilities, num_samples=1).view(-1)

            # If vocab size is provided, make sure the samples are in in the range [0, vocab-size).
            if vocab_size:
                sampled_logits = torch.clamp(sampled_logits, min=0, max=(vocab_size - 1))
        return sampled_logits

    def generate_output_tokens(self, prompts_tokens: torch.Tensor, prompts_lengths: torch.Tensor, common_inference_params: CommonInferenceParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility to generate the output tokens and probabilities for the prompts

        This utility generates the output tokens. It uses the model wrapper to generate the outputs internally

        Args:
            prompts_tokens (torch.Tensor): Prompt tokens of dimension [batch_size, max_seq_len] (i.e max_seq_len = max_prompt_len + tokens_to_generate)
            prompts_lengths (torch.Tensor): 1D tensor with [batch_size] elements with each element representing the length of the tokenized prompt
            common_inference_params (CommonInferenceParams): The inference params used for generation

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns the output tokens, the generated sequence lengths and the output log probabilitites
        """

        batch_size, max_sequence_length = prompts_tokens.size(0), prompts_tokens.size(1)
        min_prompt_length = prompts_lengths.min().item()
    
        output_log_probs = None
        if common_inference_params.return_log_probs:
            output_log_probs = torch.empty((batch_size, max_sequence_length - 1),
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
            
        # For tensor parallel models both of these return True.
        model_is_not_pipeline_parallel = parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        model_is_pipeline_parallel = not model_is_not_pipeline_parallel

        if model_is_not_pipeline_parallel or parallel_state.is_pipeline_last_stage():
            if common_inference_params.return_log_probs:
                # Pre allocate memory for output log probabilities
                output_log_probs = torch.empty((batch_size, max_sequence_length - 1),
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        
        with torch.no_grad():
            attention_mask, position_ids = self.build_attention_mask_and_position_ids(prompts_tokens)

            context_start_position = 0           
            # Pick the slice that we need to pass through the network.
            for context_end_position in range(min_prompt_length, max_sequence_length):

                tokens2use = prompts_tokens[:, context_start_position:context_end_position]
                positions2use = position_ids[:, context_start_position:context_end_position]
                attention_mask2use = attention_mask[..., context_start_position:context_end_position, :context_end_position]

                # Returns the logits of shape [batch_size, context_length, vocab_size]
                # NOTE: Can pass in a simple model or a model wrapper here. 
                # TODO : Maybe just pass in a data iterator, and then in the __call__ get the inputs rather than passing them individually to make it more generalizable. 
                logits = self.model(tokens2use, positions2use, attention_mask2use, max_sequence_length)
                
                if model_is_not_pipeline_parallel or parallel_state.is_pipeline_last_stage():
                    last_token_logits  = logits[:, -1 , :]
                    sampled_logits = self.sample_from_logits(last_token_logits, common_inference_params, self.tokenizer.vocab_size)

                    # Indicates which of the input prompts have started generating tokens. A 1D boolean tensor with [batch_size] elements
                    started = prompts_lengths < context_end_position

                    # Substitute the sampled logits only for only the prompts that have started generating tokens
                    prompts_tokens[started, context_end_position]  = sampled_logits[started]   

                    if common_inference_params.return_log_probs:
                        log_probs = F.log_softmax(logits, dim=2)
                        indices = torch.unsqueeze(prompts_tokens[:,(context_start_position+1):(context_end_position+1)], 2)
                        output_log_probs[:, context_start_position:context_end_position] = torch.gather(log_probs, 2, indices).squeeze(2)
                        
                if model_is_pipeline_parallel:
                    copy_from_last_to_first_pipeline_stage(batch_size, torch.int64, prompts_tokens)

                context_start_position = context_end_position

                #TODO : Need to add condition to check early stopping  and update generated sequence lengths

        # Include all the generated tokens
        prompts_tokens_with_generations = prompts_tokens[:,:(context_end_position+1)]
        if model_is_not_pipeline_parallel or parallel_state.is_pipeline_last_stage():
            if common_inference_params.return_log_probs:
                output_log_probs = output_log_probs[:, :context_end_position] 

        generated_sequence_lengths = prompts_lengths + common_inference_params.num_tokens_to_generate

        return prompts_tokens_with_generations, generated_sequence_lengths, output_log_probs

    def detokenize_generations(self, prompt_tokens_with_generations: torch.Tensor, generated_sequence_lengths: torch.Tensor)-> List[str]:
        """Detokenize the output generations

        This function takes the prompts with the generated tokens, and detokenizes it and trims off according to the generated sequence length param

        Args:
            prompt_tokens_with_generations (torch.Tensor): The input prompt tokens plus the generated tokens of shape [batch_size, max_seq_len] (i.e max_seq_len = max_prompt_len + tokens_to_generate)
            generated_sequence_lengths (torch.Tensor): A 1D tensor of with [batch_size]  elements consisting of the generated sequence lengths.

        Returns:
            List[str]: The detokenized outputs
        """
        
        prompts_plus_generations_detokenized = []  

        tokens = prompt_tokens_with_generations.cpu().numpy().tolist()
        lengths = generated_sequence_lengths.cpu().numpy().tolist()

        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            prompts_plus_generations_detokenized.append(
                self.tokenizer.detokenize(sequence_tokens))

        return prompts_plus_generations_detokenized