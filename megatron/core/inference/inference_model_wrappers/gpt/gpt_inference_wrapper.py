

from argparse import Namespace
from typing import Iterable, Union
from megatron.core import parallel_state
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.core.inference_params import InferenceParams
import math 
import torch
from megatron.model import GPTModel
import megatron.model

class GPTInferenceWrapper:
    def __init__(self, model: Union[GPTModel, megatron.model.GPTModel], args: Namespace):
        """Constructor for the model inference wrapper

        Here put the model in an eval mode and also check if it is pipeline paralle which decides how the forward step happens

        Args:
            model (Union[GPTModel, megatron.model.GPTModel]): The actual GPT model (MCore or MLM)
            args (Namespace): The commadline arguments that were passed
        """
        assert not isinstance(model, Iterable), 'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        # For TP only model both is_pp_first_stage and _is_pp_last_stage returns True
        self.model_is_pipeline_parallel = not (parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage())
        self.args = args
    
    def forward_pass_without_pipeline_parallel(self, tokens:torch.Tensor, position_ids:torch.Tensor, attention_mask:torch.Tensor, inference_params:InferenceParams) -> torch.Tensor:
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
        logits = self.model(tokens, position_ids, attention_mask,
                          inference_params=inference_params)
        self.inference_params.sequence_len_offset += tokens.size(1)
        return logits

    def forward_pass_with_pipeline_parallel(self, tokens:torch.Tensor, position_ids:torch.Tensor, attention_mask:torch.Tensor, inference_params:InferenceParams) -> torch.Tensor:
        """Utility to carry out forward pass PP models

        Runs the forward pass for models which are pipeline parallel.

        Args:
            tokens (torch.Tensor): Tokens tensor of shape [batch_size, inference_context_length]
            position_ids (torch.Tensor): A tensor of shape [batch_size, seq_len] containing the position ids
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, 1, seq_len, seq_len]
            inference_params (InferenceParams): The inference params passed to the forward pass for efficient computation of kv_cache

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
    def __call__(self , tokens:torch.Tensor, position_ids:torch.Tensor, attention_mask:torch.Tensor, max_sequence_length:int) -> torch.Tensor:
        """The forward pass of the model for inference

        Appropriate utility is called for the forward pass depending on the type of model parallelism used

        Args:
            tokens (torch.Tensor): Tokens tensor of shape [batch_size, inference_context_length]
            position_ids (torch.Tensor): A tensor of shape [batch_size, seq_len] containing the position ids
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, 1, seq_len, seq_len]
            max_sequence_length (int) : max_input_prompt_len + tokens_to_generate

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]. The logits are returned only in the last pipeline stage for PP models. 
        """
        batch_size = tokens.shape[0]
        inference_params = InferenceParams(batch_size, max_sequence_length)
        logits  = None
        if self.model_is_pipeline_parallel:
            logits = self.forward_pass_with_pipeline_parallel(tokens, position_ids, attention_mask, inference_params)
        else:
            logits = self.forward_pass_without_pipeline_parallel(tokens, position_ids, attention_mask, inference_params)
        return logits
