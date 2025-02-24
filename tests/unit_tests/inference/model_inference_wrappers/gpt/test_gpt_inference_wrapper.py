from argparse import Namespace

import torch

from megatron.core import parallel_state
from megatron.core.device_utils import get_current_device
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
import torch
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed

class TestGPTInferenceWrapper:

    def setup_model(self, tensor_parallel_size, pipeline_parallel_size):
        Utils.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size,pipeline_model_parallel_size=pipeline_parallel_size)
        model_parallel_device_manual_seed(123)
        self.vocab_size = 100
        self.batch_size = 4
        self.sequence_length = 32
        hidden_size = 12

        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        gpt_model = GPTModel(
            config=transformer_config, 
            transformer_layer_spec=get_gpt_layer_local_spec(), 
            vocab_size=self.vocab_size, 
            max_sequence_length=self.sequence_length, 
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            parallel_output = True).to(device=get_current_device())

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=hidden_size,
            inference_batch_times_seqlen_threshold=20,
            inference_max_requests=self.batch_size,
            fp32_residual_connection=False,
            params_dtype=torch.float,
            padded_vocab_size=self.vocab_size,
        )

        self.inference_wrapped_model = GPTInferenceWrapper(gpt_model, inference_wrapper_config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # This will call the inference_wrapped_model.forward_pass_with_pipeline_parallel_small_input_batch()
    def test_inference_pipeline_parallel_small_size(self):
        self.setup_model(tensor_parallel_size=2, pipeline_parallel_size=2)
        
        batch_prompt_tokens = torch.randint(low = 0, high = self.vocab_size, size=(self.batch_size, self.sequence_length)).int().to(device=get_current_device())
        self.inference_wrapped_model.prep_model_for_inference(prompts_tokens=batch_prompt_tokens)
        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )
        # Logits are not returned in all ranks in PP
        if parallel_state.is_pipeline_last_stage():
            assert logits.shape == (
                self.batch_size,
                5,
                self.vocab_size,
            ), f"Shape mismatch . Expected {(self.batch_size, 5, self.vocab_size)}, but got {logits.shape}"

    # This will call the inference_wrapped_model.forward_pass_with_pipeline_parallel_large_input_batch()
    def test_inference_pipeline_parallel_large__size(self):
        self.setup_model(tensor_parallel_size=2, pipeline_parallel_size=2)
        
        batch_prompt_tokens = torch.randint(low = 0, high = self.vocab_size, size=(self.batch_size, self.sequence_length)).int().to(device=get_current_device())
        self.inference_wrapped_model.prep_model_for_inference(prompts_tokens=batch_prompt_tokens)
        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 10)
        )

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        if parallel_state.is_pipeline_last_stage():
            assert logits.shape == (
                self.batch_size,
                10,
                self.vocab_size,
            ), f"Shape mismatch . Expected {(self.batch_size,10, self.vocab_size)}, but got {logits.shape}"

    def test_inference_only_tensor_parallel(self):
        self.setup_model(tensor_parallel_size=4, pipeline_parallel_size=1)
    
        batch_prompt_tokens = torch.randint(low = 0, high = self.vocab_size, size=(self.batch_size, self.sequence_length)).int().to(device=get_current_device())
        self.inference_wrapped_model.prep_model_for_inference(prompts_tokens=batch_prompt_tokens)
        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )
        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        assert logits.shape == (
            self.batch_size,
            5,
            self.vocab_size,
        ), f"Shape mismatch . Expected {(self.batch_size, 5, self.vocab_size)}, but got {logits.shape}"
