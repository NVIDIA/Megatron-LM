# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import te_general_gemm
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestGPTInferenceWrapper:

    def setup_model(
        self, tensor_parallel_size, pipeline_parallel_size, logit_dtype=None, bf16=False
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.vocab_size = 100
        self.batch_size = 4
        self.sequence_length = 32
        hidden_size = 32

        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=bf16,
            params_dtype=torch.bfloat16 if bf16 else torch.float32,
            pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        )

        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=self.vocab_size,
            max_sequence_length=self.sequence_length,
            parallel_output=True,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            logit_dtype=logit_dtype,
        ).cuda()

        # `logit_dtype` defaults to the output-layer input dtype, i.e. `params_dtype`.
        self.logit_dtype = logit_dtype or transformer_config.params_dtype

        if bf16:
            gpt_model = Float16Module(transformer_config, gpt_model)

        inference_context = StaticInferenceContext(self.batch_size, self.sequence_length)

        self.inference_wrapped_model = GPTInferenceWrapper(gpt_model, inference_context)

        InferenceMode.set_active()

    def teardown_method(self, method):
        InferenceMode.unset_active()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_pipeline_parallel(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=2, pipeline_parallel_size=2)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.config.materialize_only_last_token_logits = (
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
            assert (
                logits.dtype == self.logit_dtype
            ), f"Expected logits dtype {self.logit_dtype}, but got {logits.dtype}"

    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_only_tensor_parallel(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=4, pipeline_parallel_size=1)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.config.materialize_only_last_token_logits = (
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
        assert (
            logits.dtype == self.logit_dtype
        ), f"Expected logits dtype {self.logit_dtype}, but got {logits.dtype}"

    @pytest.mark.parametrize(
        "logit_dtype,expected_dtype",
        [
            # Unset: a bf16 model keeps bf16 logits rather than Float16Module's fp32 upcast.
            pytest.param(None, torch.bfloat16, id="default"),
            pytest.param(
                torch.float32,
                torch.float32,
                id="fp32",
                marks=pytest.mark.skipif(
                    te_general_gemm is None,
                    reason="Transformer Engine general_gemm is not available",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "tp_pp", [pytest.param((2, 2), id="tp2-pp2"), pytest.param((4, 1), id="tp4-pp1")]
    )
    def test_logit_dtype_bf16_model(self, tp_pp, logit_dtype, expected_dtype):
        tensor_parallel_size, pipeline_parallel_size = tp_pp
        self.setup_model(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            logit_dtype=logit_dtype,
            bf16=True,
        )
        assert self.logit_dtype == expected_dtype
        assert self.inference_wrapped_model.logit_dtype == expected_dtype

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )
        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        # Logits are only returned on the last pipeline stage.
        if parallel_state.is_pipeline_last_stage():
            assert (
                logits.dtype == expected_dtype
            ), f"Expected logits dtype {expected_dtype}, but got {logits.dtype}"
        else:
            assert logits is None
