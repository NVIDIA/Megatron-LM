import copy
import os
import random
import string
import time
from argparse import Namespace
from collections import OrderedDict
from typing import Dict
from unittest import mock

import pytest
import torch

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_request import InferenceRequest, Status, VLMInferenceRequest
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.vlm_text_generation_controller import (
    VLMTextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.legacy.model import Float16Module
from tests.unit_tests.test_utilities import Utils


class TestVLMTextGenerationController:

    @pytest.mark.internal  # The model is under active development and its methods may change.
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.language_hidden_size = 64
        self.language_num_attention_heads = 4
        self.language_vocab_size = 8192
        self.language_max_sequence_length = 4096
        self.img_h = 336
        self.img_w = 336

        language_config = TransformerConfig(
            num_layers=3,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=False,
        )
        vision_config = TransformerConfig(
            num_layers=2, hidden_size=16, num_attention_heads=2, use_cpu_initialization=False
        )
        vision_projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=False,
        )

        language_layer_spec = get_gpt_layer_local_spec()
        vision_layer_spec = copy.deepcopy(language_layer_spec)
        vision_projection_spec = copy.deepcopy(language_layer_spec.submodules.mlp.submodules)

        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "clip"
        self.model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=self.language_vocab_size,
            language_max_sequence_length=self.language_max_sequence_length,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=14,
        ).cuda()
        self.image_token_index = self.model.image_token_index
        self.model = Float16Module(self.model, Namespace(fp16=False, bf16=True))

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=self.language_hidden_size,
            inference_batch_times_seqlen_threshold=-1,
            fp32_residual_connection=False,
            params_dtype=torch.float,
            padded_vocab_size=self.language_vocab_size,
        )

        inference_wrapped_model = VLMInferenceWrapper(self.model, inference_wrapper_config)

        self.mock_tokenizer = mock.Mock()

        self.text_generation_controller = VLMTextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_generate_all_output_tokens_static_batch(self):
        self.mock_tokenizer.vocab_size = self.language_vocab_size
        self.mock_tokenizer.eod = self.language_vocab_size - 1
        self.mock_tokenizer.detokenize.return_value = ''.join(
            random.choices(string.ascii_letters, k=random.randint(4, 10))
        )

        batch_size: int = 1
        num_img_embeddings_per_tile: int = 576
        imgs: torch.Tensor = torch.randn(1, 3, self.img_h, self.img_w).cuda()
        num_tiles: torch.Tensor = torch.Tensor([1]).int()
        decoder_seq_length: int = self.language_max_sequence_length

        active_requests: Dict[str, InferenceRequest] = OrderedDict()
        all_prompt_tokens: Dict[str, List[int]] = OrderedDict()
        for i in range(batch_size):
            prompt = "sample" * (i + 1)
            self.mock_tokenizer.tokenize.return_value = torch.randn(
                batch_size, self.language_vocab_size
            ).cuda()
            prompt_tokens = torch.randint(
                low=0, high=self.language_vocab_size - 1, size=(len(prompt),)
            ).tolist()
            prompt_tokens[3] = self.image_token_index

            request_id = str(i)
            inference_request = VLMInferenceRequest(
                request_id=request_id,
                prompt=prompt,
                inference_parameters=CommonInferenceParams(num_tokens_to_generate=10),
                arrival_time=time.time(),
                prompt_tokens=prompt_tokens,
                num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                imgs=imgs,
                num_tiles=num_tiles,
                decoder_seq_length=decoder_seq_length,
                status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
            )
            active_requests[request_id] = inference_request
            all_prompt_tokens[request_id] = copy.deepcopy(prompt_tokens)

        requests = self.text_generation_controller.generate_all_output_tokens_static_batch(
            active_requests
        )

        for request_id, request in requests.items():
            assert (
                request.status == Status.COMPLETED
            ), f"Status should be completed but its {request.status}"
            assert request.generated_length > 0, f"Generated length should be greater than zero"
            assert request.generated_text is not None, "Generated text should not be None"
            assert (
                all_prompt_tokens[request_id] == request.prompt_tokens
            ), "Prompt tokens should not have changed during generation"
