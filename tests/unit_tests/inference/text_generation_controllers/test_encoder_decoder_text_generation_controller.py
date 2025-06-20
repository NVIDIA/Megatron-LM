import random
import string
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict
from unittest import mock

import numpy as np
import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.t5.t5_inference_wrapper import (
    T5InferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.encoder_decoder_text_generation_controller import (
    EncoderDecoderTextGenerationController,
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


class TestEncoderDecoderTextGenerationController:

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=4, pipeline_model_parallel_size=1
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
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
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

        inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

        inference_wrapped_model = T5InferenceWrapper(
            t5_model, inference_wrapper_config, inference_context
        )

        self.mock_tokenizer = mock.Mock()

        self.text_generation_controller = EncoderDecoderTextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_generate_all_output_tokens_static_batch(self):
        self.mock_tokenizer.vocab_size = self.vocab_size
        self.mock_tokenizer.eod = self.vocab_size - 1
        self.mock_tokenizer.pad = self.vocab_size - 2
        self.mock_tokenizer.additional_special_tokens_ids = list(range(100))
        self.mock_tokenizer.detokenize.return_value = ''.join(
            random.choices(string.ascii_letters, k=random.randint(4, 10))
        )
        self.mock_tokenizer.tokenize.return_value = np.random.randint(
            self.vocab_size, size=(self.encoder_sequence_length - 5)
        ).tolist()

        active_requests: Dict[str, InferenceRequest] = OrderedDict()
        for i in range(self.batch_size):
            prompt = "decoder_sample"
            prompt_tokens = np.random.randint(
                self.vocab_size, size=self.decoder_sequence_length
            ).tolist()
            encoder_prompt = "encoder_sample"
            inference_request = InferenceRequest(
                request_id=i,
                prompt=prompt,
                encoder_prompt=encoder_prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                arrival_time=time.time(),
                prompt_tokens=prompt_tokens,
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests[i] = inference_request

        requests = self.text_generation_controller.generate_all_output_tokens_static_batch(
            active_requests
        )

        for request_id, request in requests.items():
            assert (
                request.status == Status.COMPLETED
            ), f"Status should be completed but its {request.status}"
            assert request.generated_length > 0, f"Generated length should be greater than zero"
            assert request.generated_text is not None, "Generated text should not be None"
