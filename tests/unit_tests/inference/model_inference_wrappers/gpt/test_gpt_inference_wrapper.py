from argparse import Namespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestGPTInferenceWrapper:

    def setup_model(self, tensor_parallel_size, pipeline_parallel_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
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
            parallel_output=True,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=hidden_size,
            inference_batch_times_seqlen_threshold=20,
            inference_max_requests=self.batch_size,
            fp32_residual_connection=False,
            params_dtype=torch.float,
            padded_vocab_size=self.vocab_size,
        )

        inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

        self.inference_wrapped_model = GPTInferenceWrapper(
            gpt_model, inference_wrapper_config, inference_context
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # This will call the inference_wrapped_model.forward_pass_with_pipeline_parallel_small_input_batch()
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_pipeline_parallel_small_size(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=2, pipeline_parallel_size=2)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.materialize_only_last_token_logits = (
            materialize_only_last_token_logits
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )

        logits_seq_len = 1 if materialize_only_last_token_logits else 5

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )
        # Logits are not returned in all ranks in PP
        if parallel_state.is_pipeline_last_stage():
            assert logits.shape == (
                self.batch_size,
                logits_seq_len,
                self.vocab_size,
            ), f"Shape mismatch . Expected {(self.batch_size, logits_seq_len, self.vocab_size)}, but got {logits.shape}"

    # This will call the inference_wrapped_model.forward_pass_with_pipeline_parallel_large_input_batch()
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_pipeline_parallel_large_size(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=2, pipeline_parallel_size=2)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.materialize_only_last_token_logits = (
            materialize_only_last_token_logits
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 10)
        )

        logits_seq_len = 1 if materialize_only_last_token_logits else 10

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        if parallel_state.is_pipeline_last_stage():
            assert logits.shape == (
                self.batch_size,
                logits_seq_len,
                self.vocab_size,
            ), f"Shape mismatch . Expected {(self.batch_size, logits_seq_len, self.vocab_size)}, but got {logits.shape}"

    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_only_tensor_parallel(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=4, pipeline_parallel_size=1)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.materialize_only_last_token_logits = (
            materialize_only_last_token_logits
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )

        logits_seq_len = 1 if materialize_only_last_token_logits else 5

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        assert logits.shape == (
            self.batch_size,
            logits_seq_len,
            self.vocab_size,
        ), f"Shape mismatch . Expected {(self.batch_size, logits_seq_len, self.vocab_size)}, but got {logits.shape}"
