from argparse import Namespace
from typing import Iterable, List, Tuple, Union
from megatron.core import parallel_state
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.core.inference_params import InferenceParams
import math 
import torch
from megatron.model import GPTModel
import megatron.model

class GPTInferenceWrapper:
    def __init__(self, model: Union[GPTModel, megatron.model.GPTModel], args: Namespace):
        """Constructor for the model inference wrapper

        The wrapper is in charge of preparing the model for inference, providing the required in put data and running the forward pass

        Args:
            model (Union[GPTModel, megatron.model.GPTModel]): The actual GPT model (MCore or MLM)
            args (Namespace): The commadline arguments that were passed
        """
        assert not isinstance(model, Iterable), 'interleaving schedule is not supported for inference'
        self.model = model
        self.args = args

    def prep_model_for_inference(self,  prompts_tokens: torch.Tensor):
        """A utility function for preparing model for inference

        The function gets called before you get the inference data and running forward pass. Use it to put the model in eval mode, build position ids ,attention mask etc, so that required slices can be extracted during the forward pass. 

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
        """
        self.model.eval()
        # For TP only model both is_pp_first_stage and _is_pp_last_stage returns True
        self.model_is_pipeline_parallel = not (parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage())
        self.attention_mask, self.position_ids = self.build_attention_mask_and_position_ids(prompts_tokens)
        self.prompt_tokens = self.prompt_tokens

    def build_attention_mask_and_position_ids(self, prompts_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the full attention mask and position ids for the input tokens

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]

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
    
    def get_batch_for_context_window(self, context_start_position:int, context_end_position:int) -> List:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data. 

        Args:
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length. 

        Returns:
            List: A list of inputs that will be used by your model in the forward step
        """
        tokens2use = self.prompt_tokens[:, context_start_position:context_end_position]
        positions2use = self.position_ids[:, context_start_position:context_end_position]
        attention_mask2use = self.attention_mask[..., context_start_position:context_end_position, :context_end_position]

        batch_size, max_sequence_length = self.prompt_tokens.size
        inference_params = InferenceParams(batch_size, max_sequence_length)

        data_at_step_idx = [tokens2use, positions2use, attention_mask2use, inference_params]
        return data_at_step_idx

    
    def forward_pass_without_pipeline_parallel(self,  inference_input:List, inference_params:InferenceParams) -> torch.Tensor:
        """Utility to carry out forward pass for DP or TP only models

        Runs the forward pass for models which are not pipeline parallel 

        Args:
            tokens (torch.Tensor): Tokens tensor of shape [batch_size, inference_context_length]
            position_ids (torch.Tensor): A tensor of shape [batch_size, seq_len] containing the position ids
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, 1, seq_len, seq_len]
            inference_params (InferenceParams): The inference params passed to the forward pass for efficient computation of kv_cache

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens, position_ids, attention_mask = inference_input
        logits = self.model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)
        self.inference_params.sequence_len_offset += tokens.size(1)
        return logits

    def forward_pass_with_pipeline_parallel(self, inference_input:List, inference_params:InferenceParams) -> torch.Tensor:
        """Utility to carry out forward pass PP models

        Runs the forward pass for models which are pipeline parallel.

        Args:
            inference_input (List): A list containg the inputs for the gpt model [tokens, position ids, attention mask]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        def _allocate_recv_buffer(batch_size, seq_len):
            """Receive happens between the layers with size [seq_len, batch_size, hidden_size]."""    
            recv_size = (batch_size, seq_len, self.args.hidden_size)
            dtype = torch.float if self.args.fp32_residual_connection else self.args.params_dtype
            return torch.empty(recv_size, dtype=dtype, device=torch.cuda.current_device())

        is_pipeline_first_stage = parallel_state.is_pipeline_first_stage()
        is_pipeline_last_stage = parallel_state.is_pipeline_last_stage()

        tokens, position_ids, attention_mask = inference_input
        batch_size, seq_len = tokens.shape
        micro_batch_size = 1
        if batch_size * seq_len > self.args.inference_batch_times_seqlen_threshold:
            micro_batch_size = max(1, self.args.inference_batch_times_seqlen_threshold // tokens.size(1))
        # Round up to account for tge last partial micro batch if present
        num_micro_batches = math.ceil(batch_size/micro_batch_size)
        
        logits = None
        # Preallocate memory for output logits.
        if is_pipeline_last_stage:
            logits = torch.empty((batch_size, seq_len, self.args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device()) 
        
        recv_buffer = None
        if not is_pipeline_first_stage:
            recv_buffer = _allocate_recv_buffer(batch_size, seq_len)
            
        for micro_batch_index in range(num_micro_batches):
            start = micro_batch_index * micro_batch_size 
            end = min(start + micro_batch_size, batch_size)
            tokens2use = tokens[start:end, ...]
            position_ids2use = position_ids[start:end, ...]
            current_micro_batch_size = end-start

            # Need to change recv buffer shape for the last partial microbatch (if exists)
            if current_micro_batch_size != micro_batch_size:
                recv_buffer = _allocate_recv_buffer(current_micro_batch_size, seq_len)

            if not is_pipeline_first_stage:
                recv_from_prev_pipeline_rank_(recv_buffer)

            self.model.set_input_tensor(recv_buffer)
            output_tensor = self.model(tokens2use, position_ids2use, attention_mask,
                          inference_params=inference_params)
            
            if not is_pipeline_last_stage:
                send_to_next_pipeline_rank(output_tensor)
                logits[start:end, ...] = output_tensor

            inference_params.batch_size_offset += current_micro_batch_size
                
        #Once done with all micro batches, we reset batch size offset and seq len offset   
        inference_params.sequence_len_offset += seq_len
        inference_params.batch_size_offset = 0

        #NOTE: Only returns the logits on the last pipeline stage
        return logits

    #TODO : Should maybe use the parallel schedules to do this instead of doing manually
    def __call__(self , inference_input:List) -> torch.Tensor:
        """The forward pass of the model for inference

        Appropriate utility is called for the forward pass depending on the type of model parallelism used

        Args:
            inference_input (List): A list containg the inputs for the gpt model [tokens, position ids, attention mask, inference_params]
            
        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]. The logits are returned only in the last pipeline stage for PP models. 
        """
        logits  = None
        if self.model_is_pipeline_parallel:
            logits = self.forward_pass_with_pipeline_parallel(inference_input)
        else:
            logits = self.forward_pass_without_pipeline_parallel(inference_input)
        return logits
