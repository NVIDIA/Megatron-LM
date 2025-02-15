# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import abc
import math
from typing import Any, Dict, Iterable, Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.inference.communication_utils import (
    recv_from_prev_pipeline_rank_,
    send_to_next_pipeline_rank,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference_params import InferenceParams
from megatron.core.models.gpt.gpt_model import GPTModel


# pylint: disable=line-too-long
class AbstractModelInferenceWrapper(abc.ABC):
    """Abstract inference wrapper

    Extend this to create a version for your model.
    """

    def __init__(
        self,
        model: Union['LegacyGPTModel', GPTModel],  # type: ignore[name-defined]
        inference_wrapper_config: InferenceWrapperConfig,
    ):
        """Constructor for the model inference wrapper

        The wrapper prepares the model for inference, provides the required input data and runs the forward pass.

        Args:
            model (Union[GPTModel, LegacyGPTModel]): The actual GPT model (MCore or MLM)
            inference_wrapper_config (InferenceWrapperConfig): Has info like hidden size, vocab size etc.
        """
        assert not isinstance(
            model, Iterable
        ), 'interleaving schedule is not supported for inference'
        self.model = model
        self.inference_wrapper_config = inference_wrapper_config
        self.pipeline_communication_dtype = (
            torch.float
            if self.inference_wrapper_config.fp32_residual_connection
            else self.inference_wrapper_config.params_dtype
        )

        max_batch_size = self.inference_wrapper_config.inference_max_requests
        max_sequence_length = self.inference_wrapper_config.inference_max_seq_length
        self.inference_params = InferenceParams(max_batch_size, max_sequence_length)

    def prep_model_for_inference(self, prompts_tokens: torch.Tensor):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop.
        It puts the model in eval mode.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]

        """
        self.model.eval()

        # For TP only model both is_pp_first_stage and _is_pp_last_stage returns True
        self.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        self.inference_params.reset()

    @abc.abstractmethod
    def prep_inference_input(self, prompt_tokens) -> Dict[str, Any]:
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]

        Returns:
            A dict with all the inference input needed for the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_batch_for_context_window(self, *args, **kwargs) -> Dict[str, Any]:
        """Returns the input data for inference

        This function gets called iteratively in the inference loop . It can be used to extract relevant input from the prompt tokens, attention mask etc. required for each step in inference.

        """
        raise NotImplementedError()

    def _forward(self, inference_input):
        """Runs a forward pass of the model.

        Args:
            inference_input(Dict[str, Any]): The input data.
            inference_params(InferenceParams): The inference parameters.

        Returns:
            The model output logits.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        return self.model(
            tokens, position_ids, attention_mask, inference_params=self.inference_params
        )

    def _get_batch_size_and_seq_len(
        self, tokens: torch.Tensor, recv_buffer_seq_len: Optional[int] = None
    ):
        """
        Returns the batch size and sequence length based on the tokens tensor and recv_buffer_seq_len.

        Args:
            tokens (torch.Tensor): The input tensor of shape (batch_size, seq_len).
            recv_buffer_seq_len (int, optional): An optional recv buffer sequence length.

        Returns:
            tuple: A tuple (batch_size, seq_len), where batch_size is the first dimension of tokens
                   and seq_len is either the second dimension or recv_buffer_seq_len.
        """
        batch_size = tokens.shape[0]
        seq_len = recv_buffer_seq_len if recv_buffer_seq_len is not None else tokens.shape[1]
        return batch_size, seq_len

    def _allocate_recv_buffer(self, batch_size, seq_len):
        """Receive happens between the layers with size [seq_len, batch_size, hidden_size]."""
        recv_size = (seq_len, batch_size, self.inference_wrapper_config.hidden_size)
        return torch.empty(
            recv_size, dtype=self.pipeline_communication_dtype, device=torch.cuda.current_device()
        )

    def forward_pass_without_pipeline_parallel(
        self, inference_input: Dict[str, Any]
    ) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without any parallelism or only tensor parallelism.

        Args:
            inference_input (Dict[str, Any]): A dict containg the inputs for the gpt model [tokens, position ids, attention mask]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens = inference_input["tokens"]
        logits = self._forward(inference_input)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        self.inference_params.sequence_len_offset += tokens.size(1)

        return logits

    def forward_pass_with_pipeline_parallel_small_input_batch(
        self, inference_input: Dict[str, Any], recv_buffer_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Utility to carry out forward pass for PP models with very small inputs

        If a model is pipeline parallel, yet, the input global batch is very small, we compute a foward pass on the entire global batch, rather than splitting it up into micro batches and doing something more complex as in the forward_pass_with_pipeline_parallel_large_input_batch method

        Args:
            inference_input (Dict[str, Any]): A dict containing the inputs for the gpt model [tokens, position ids, attention mask]
            recv_buffer_seq_len (int): An optional sequence length for the pipeline parallel recv buffer.

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        batch_size, seq_len = self._get_batch_size_and_seq_len(tokens, recv_buffer_seq_len)
        recv_buffer = None
        if not parallel_state.is_pipeline_first_stage():
            recv_buffer = self._allocate_recv_buffer(batch_size, seq_len)
            recv_from_prev_pipeline_rank_(recv_buffer)

        self.model.set_input_tensor(recv_buffer)
        output_tensor = self._forward(inference_input)

        if not parallel_state.is_pipeline_last_stage():
            send_to_next_pipeline_rank(output_tensor.type(dtype=self.pipeline_communication_dtype))

        self.inference_params.sequence_len_offset += seq_len

        logits = None
        if parallel_state.is_pipeline_last_stage():
            logits = output_tensor
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

            # Explicitly cast logits to expected dtype
            logits = logits.to(self.inference_wrapper_config.params_dtype)

        return logits

    def forward_pass_with_pipeline_parallel_large_input_batch(
        self, inference_input: Dict[str, Any], recv_buffer_seq_len=None
    ) -> torch.Tensor:
        """Utility to carry out forward pass PP models.

        Runs the forward pass for models which are pipeline parallel.
        This is more complex than forward_pass_with_pipeline_parallel_small_input_batch because
        this splits the global batch into small micro batches and runs them through the model.

        Args:
            inference_input (Dict[str, Any]): A dict containg the inputs for the gpt model [tokens, position ids, attention mask]
            recv_buffer_seq_len (int): An optional sequence length for the pipeline parallel recv buffer.

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        micro_batch_size = max(
            1,
            self.inference_wrapper_config.inference_batch_times_seqlen_threshold // tokens.size(1),
        )
        batch_size, seq_len = self._get_batch_size_and_seq_len(tokens, recv_buffer_seq_len)
        # Round up to account for the last partial micro batch if present
        num_micro_batches = math.ceil(batch_size / micro_batch_size)

        logits = None
        # Preallocate memory for output logits.
        if parallel_state.is_pipeline_last_stage():
            logits = torch.empty(
                (batch_size, seq_len, self.inference_wrapper_config.padded_vocab_size),
                dtype=self.pipeline_communication_dtype,
                device=torch.cuda.current_device(),
            )

        recv_buffer = None
        if not parallel_state.is_pipeline_first_stage():
            recv_buffer = self._allocate_recv_buffer(micro_batch_size, seq_len)
        for micro_batch_index in range(num_micro_batches):
            start = micro_batch_index * micro_batch_size
            end = min(start + micro_batch_size, batch_size)
            tokens2use = tokens[start:end, ...]
            position_ids2use = position_ids[start:end, ...]
            current_micro_batch_size = end - start

            # Need to change recv buffer shape for the last partial microbatch (if exists)
            if current_micro_batch_size != micro_batch_size:
                recv_buffer = self._allocate_recv_buffer(current_micro_batch_size, seq_len)

            if not parallel_state.is_pipeline_first_stage():
                recv_from_prev_pipeline_rank_(recv_buffer)

            self.model.set_input_tensor(recv_buffer)
            output_tensor = self._forward(
                {
                    "tokens": tokens2use,
                    "position_ids": position_ids2use,
                    "attention_mask": attention_mask,
                }
            )

            if not parallel_state.is_pipeline_last_stage():
                send_to_next_pipeline_rank(output_tensor)

            self.inference_params.batch_size_offset += current_micro_batch_size

            if parallel_state.is_pipeline_last_stage():
                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(
                    output_tensor
                )
                assert logits is not None
                logits[start:end, ...] = output_tensor

                # Explicitly cast logits to expected dtype
                logits = logits.to(self.inference_wrapper_config.params_dtype)

        # Once done with all micro batches, we reset batch size offset and seq len offset
        self.inference_params.sequence_len_offset += seq_len
        self.inference_params.batch_size_offset = 0

        # NOTE: Only returns the logits on the last pipeline stage
        return logits

    def run_one_forward_step(
        self, inference_input: Dict[str, Any], recv_buffer_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """The forward pass of the model for inference

        Appropriate utility is called for the forward pass depending on the type of model parallelism used

        Args:
            inference_input (Dict[str, Any]): A dict containg the inputs for the gpt model [tokens, position ids, attention mask]
            recv_buffer_seq_len (int): An optional sequence length for the pipeline parallel recv buffer.

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]. The logits are returned only in the last pipeline stage for PP models.
        """
        if self.model_is_pipeline_parallel:
            tokens = inference_input["tokens"]
            current_batch_size, seq_len = self._get_batch_size_and_seq_len(
                tokens, recv_buffer_seq_len
            )
            # If input batch is large, we need to split into micro batches and run the forward pass
            if (
                current_batch_size * seq_len
                > self.inference_wrapper_config.inference_batch_times_seqlen_threshold
            ):
                return self.forward_pass_with_pipeline_parallel_large_input_batch(
                    inference_input, recv_buffer_seq_len
                )
            else:
                # If input batch is very small we can do a simple forward pass on the entire global batch
                return self.forward_pass_with_pipeline_parallel_small_input_batch(
                    inference_input, recv_buffer_seq_len
                )
        else:
            return self.forward_pass_without_pipeline_parallel(inference_input)
