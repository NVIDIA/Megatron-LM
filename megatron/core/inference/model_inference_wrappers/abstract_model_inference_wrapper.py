# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import abc
from typing import Any, Dict, Iterable, Optional, Union

import torch

from megatron.core.fp8_utils import prepare_model_for_fp8_inference
from megatron.core.inference.communication_utils import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    recv_from_prev_pipeline_rank_,
    send_to_next_pipeline_rank,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_attr_wrapped_model, get_model_config


class AbstractModelInferenceWrapper(abc.ABC):
    """Abstract inference wrapper

    Extend this to create a version for your model.

    The wrapper prepares the model for inference, provides the required input data and
    runs the forward pass.

    Args:
        model (Union[GPTModel, LegacyGPTModel]): The actual GPT model (MCore
            or MLM).
        inference_context (BaseInferenceContext): Context for managing KV
            cache and other inference params.
        pg_collection (ProcessGroupCollection): Process groups for model communication.
    """

    def __init__(
        self,
        model: Union['LegacyGPTModel', GPTModel],  # type: ignore[name-defined]
        inference_context: BaseInferenceContext,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        assert not isinstance(
            model, Iterable
        ), 'interleaving schedule is not supported for inference'
        self.model = model
        self.config = get_model_config(self.model)
        self.pipeline_communication_dtype = (
            torch.float if self.config.fp32_residual_connection else self.config.params_dtype
        )
        self.sequence_parallel = self.config.sequence_parallel

        self.inference_context = inference_context

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp
        self.tp_size = torch.distributed.get_world_size(self.tp_group)

        if self.config.fp8 is not None:
            self.model = prepare_model_for_fp8_inference(self.model)

        # TODO(ksanthanam): Add support for fp4

    def prep_model_for_inference(self):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop.
        It puts the model in eval mode.

        """
        self.model.eval()

        # For TP only model both is_pp_first_stage and _is_pp_last_stage returns True
        self.model_is_pipeline_parallel = not (
            is_pipeline_first_stage(self.pp_group) and is_pipeline_last_stage(self.pp_group)
        )

        self.inference_context.reset()

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

        This function gets called iteratively in the inference loop.
        It can be used to extract relevant input from the prompt tokens, attention mask etc.
        required for each step in inference.

        """
        raise NotImplementedError()

    def _forward(self, inference_input):
        """Runs a forward pass of the model.

        Args:
            inference_input(Dict[str, Any]): The input data.

        Returns:
            The model output logits.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        return self.model(
            tokens,
            position_ids,
            attention_mask,
            inference_context=self.inference_context,
            runtime_gather_output=True,  # Inference should always gather the logits
        )

    @torch.no_grad()
    def dummy_forward(self):
        """Run a dummy forward pass through the model, with a single token.
        Use-case: Used in EP on ranks which do not have any work, but are needed
        for the all-to-all communication."""
        # we use num_dummy_tokens equal to tensor model parallel size
        # so that the dummy forward pass will work with sequence parallel
        num_dummy_tokens = self.tp_size
        tokens = torch.zeros(
            (1, num_dummy_tokens), dtype=torch.long, device=torch.cuda.current_device()
        )
        position_ids = torch.zeros(
            (1, num_dummy_tokens), dtype=torch.long, device=torch.cuda.current_device()
        )
        attention_mask = None
        return self.model(tokens, position_ids, attention_mask)

    def _get_batch_size_and_seq_len(
        self, tokens: torch.Tensor, recv_buffer_seq_len: Optional[int] = None
    ):
        """
        Returns the batch size and sequence length based on the tokens tensor and
        recv_buffer_seq_len.

        Args:
            tokens (torch.Tensor): The input tensor of shape (batch_size, seq_len).
            recv_buffer_seq_len (int, optional): An optional recv buffer sequence length.

        Returns:
            tuple: A tuple (batch_size, seq_len), where batch_size is the first dimension of
                tokens and seq_len is either the second dimension or recv_buffer_seq_len.
        """
        batch_size = tokens.shape[0]
        seq_len = recv_buffer_seq_len if recv_buffer_seq_len is not None else tokens.shape[1]
        return batch_size, seq_len

    def _allocate_recv_buffer(self, batch_size, seq_len):
        """Receive happens between the layers with size [seq_len, batch_size, hidden_size]."""
        if self.sequence_parallel and self.inference_context.is_dynamic_batching():
            # For dynamic inference we need to explicitly adjust the recv buffer size here for
            # sequence parallelism. Static batching does not support sequence parallelism
            # except for the MoE layers which is handled separately.
            seq_len = seq_len // self.tp_size
        recv_size = (seq_len, batch_size, self.config.hidden_size)
        return torch.empty(
            recv_size, dtype=self.pipeline_communication_dtype, device=torch.cuda.current_device()
        )

    def forward_pass_without_pipeline_parallel(
        self, inference_input: Dict[str, Any]
    ) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without any
        parallelism or only tensor parallelism.

        Args:
            inference_input (Dict[str, Any]): A dict containg the inputs for the gpt model
                [tokens, position ids, attention mask]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens = inference_input["tokens"]
        logits = self._forward(inference_input)
        self.inference_context.increment_sequence_len_offset(tokens.size(1))

        return logits

    def forward_pass_with_pipeline_parallel(
        self, inference_input: Dict[str, Any], recv_buffer_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Utility to carry out forward pass for PP models

        TODO: Add support for asynchronous microbatches

        Args:
            inference_input (Dict[str, Any]): A dict containing the inputs for the gpt model
                [tokens, position ids, attention mask]
            recv_buffer_seq_len (int): An optional sequence length for the pipeline parallel
                recv buffer.

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]

        batch_size, seq_len = self._get_batch_size_and_seq_len(tokens, recv_buffer_seq_len)
        recv_buffer = None
        if not is_pipeline_first_stage(self.pp_group):
            recv_buffer = self._allocate_recv_buffer(batch_size, seq_len)
            recv_from_prev_pipeline_rank_(recv_buffer, self.pp_group)

        set_input_tensor = get_attr_wrapped_model(self.model, "set_input_tensor")
        set_input_tensor(recv_buffer)
        output_tensor = self._forward(inference_input)

        if not is_pipeline_last_stage(self.pp_group):
            send_to_next_pipeline_rank(
                output_tensor.type(dtype=self.pipeline_communication_dtype), self.pp_group
            )

        self.inference_context.increment_sequence_len_offset(seq_len)

        logits = None
        if is_pipeline_last_stage(self.pp_group):
            logits = output_tensor

            # Explicitly cast logits to expected dtype
            logits = logits.to(self.config.params_dtype)

        return logits

    @torch.inference_mode()
    def run_one_forward_step(
        self, inference_input: Dict[str, Any], recv_buffer_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """The forward pass of the model for inference

        Appropriate utility is called for the forward pass depending on the type of model
        parallelism used

        Args:
            inference_input (Dict[str, Any]): A dict containing the inputs for the gpt model
                [tokens, position ids, attention mask]
            recv_buffer_seq_len (int): An optional sequence length for the pipeline parallel
                recv buffer.

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size].
            The logits are returned only in the last pipeline stage for PP models.
        """
        # Check if we are in a PP model
        if not (is_pipeline_first_stage(self.pp_group) and is_pipeline_last_stage(self.pp_group)):
            tokens = inference_input["tokens"]
            current_batch_size, seq_len = self._get_batch_size_and_seq_len(
                tokens, recv_buffer_seq_len
            )
            return self.forward_pass_with_pipeline_parallel(inference_input, recv_buffer_seq_len)
        else:
            return self.forward_pass_without_pipeline_parallel(inference_input)
