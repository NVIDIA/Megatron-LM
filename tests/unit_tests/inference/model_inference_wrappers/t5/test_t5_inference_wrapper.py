from argparse import Namespace
from copy import deepcopy
from unittest import mock

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.t5.t5_inference_wrapper import (
    T5InferenceWrapper,
)
from megatron.core.models.T5.t5_model import T5Model
from megatron.core.models.T5.t5_spec import (
    get_t5_decoder_with_transformer_engine_block_spec,
    get_t5_encoder_with_transformer_engine_block_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestT5InferenceWrapper:

    def setup_model(self, tensor_parallel_size, pipeline_parallel_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.vocab_size = 100
        self.batch_size = 8
        self.encoder_sequence_length = 32
        self.decoder_sequence_length = 16
        hidden_size = 768

        transformer_config = TransformerConfig(
            num_layers=12,
            hidden_size=hidden_size,
            num_attention_heads=12,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            attention_backend=AttnBackend.unfused,
        )

        encoder_config = deepcopy(transformer_config)
        encoder_config.num_layers = transformer_config.num_layers

        encoder_layers_per_pipeline = (
            encoder_config.num_layers // encoder_config.pipeline_model_parallel_size
        )
        decoder_layers_per_pipeline = (
            transformer_config.num_layers // transformer_config.pipeline_model_parallel_size
        )
        en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(
            encoder_layers_per_pipeline
        )
        de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(
            decoder_layers_per_pipeline
        )

        t5_model = T5Model(
            config=transformer_config,
            encoder_config=encoder_config,
            transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=de_block_spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.encoder_sequence_length,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        ).cuda()

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=hidden_size,
            inference_batch_times_seqlen_threshold=-1,
            fp32_residual_connection=False,
            params_dtype=torch.float,
            padded_vocab_size=self.vocab_size,
        )

        self.inference_wrapped_model = T5InferenceWrapper(t5_model, inference_wrapper_config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_inference_only_tensor_parallel(self):
        self.setup_model(tensor_parallel_size=4, pipeline_parallel_size=1)

        batch_prompt_tokens = (
            torch.randint(
                low=0, high=self.vocab_size, size=(self.batch_size, self.decoder_sequence_length)
            )
            .int()
            .cuda()
        )
        batch_encoder_prompts = ["sample prompt encoders"] * self.batch_size
        mock_tokenizer = mock.Mock()
        mock_tokenizer.pad = self.vocab_size - 1
        mock_tokenizer.additional_special_tokens_ids = list(range(100))
        mock_tokenizer.tokenize.return_value = np.random.randint(
            self.vocab_size, size=self.encoder_sequence_length
        ).tolist()

        self.inference_wrapped_model.prep_model_for_inference(prompts_tokens=batch_prompt_tokens)

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens,
            encoder_prompts=batch_encoder_prompts,
            tokenizer=mock_tokenizer,
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(
                inference_input, 0, self.decoder_sequence_length
            )
        )

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        assert logits.shape == (
            self.batch_size,
            self.decoder_sequence_length,
            self.vocab_size,
        ), f"Shape mismatch . Expected {(self.batch_size, self.decoder_sequence_length, self.vocab_size)}, but got {logits.shape}"
