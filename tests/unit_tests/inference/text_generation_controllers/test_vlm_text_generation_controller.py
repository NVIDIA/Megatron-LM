# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import random
import string
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List
from unittest import mock

import pytest
import torch

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.inference_request import InferenceRequest, Status, VLMInferenceRequest
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.utils import (
    dynamic_media_embedding_counts,
    dynamic_media_replacement_counts,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.vlm_text_generation_controller import (
    VLMTextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
def test_dynamic_video_embedding_counts_support_video_and_tubelet_markers():
    frame_counts = dynamic_media_embedding_counts(
        torch.tensor([[448, 576]] * 4),
        patch_dim=16,
        pixel_shuffle=True,
    )
    assert frame_counts == [252] * 4

    assert dynamic_media_replacement_counts(
        frame_counts,
        num_frames=torch.tensor([4]),
        temporal_patch_size=2,
        placeholder_count=2,
    ) == [252, 252]
    assert dynamic_media_replacement_counts(
        frame_counts,
        num_frames=torch.tensor(4),
        temporal_patch_size=2,
        placeholder_count=1,
    ) == [504]


@pytest.mark.internal
def test_dynamic_video_embedding_counts_reject_misaligned_placeholders():
    with pytest.raises(ValueError, match="must match either"):
        dynamic_media_replacement_counts(
            [252] * 4,
            num_frames=torch.tensor([4]),
            temporal_patch_size=2,
            placeholder_count=3,
        )


@pytest.mark.internal
def test_vlm_wrapper_expands_one_video_marker_to_all_tubelet_embeddings():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace()
    wrapper.model.module = SimpleNamespace(
        image_token_index=-200,
        _dynamic_resolution=True,
        patch_dim=16,
        _pixel_shuffle=True,
        _conv_merging=False,
        _drop_vision_class_token=True,
        temporal_patch_dim=2,
    )

    expanded, masks = wrapper.expand_image_tokens(
        [[11, -200, 12]],
        imgs_sizes=torch.tensor([[448, 576]] * 4),
        num_frames=torch.tensor([4]),
    )

    assert expanded == [[11] + [-1] * 504 + [12]]
    assert masks[0][0] is None
    assert masks[0][1:-1] == list(range(504))
    assert masks[0][-1] is None


@pytest.mark.internal
def test_vlm_and_omni_wrappers_expand_tubelet_markers_consistently():
    model = SimpleNamespace(
        image_token_index=-200,
        _dynamic_resolution=True,
        dynamic_resolution=True,
        patch_dim=16,
        _pixel_shuffle=True,
        _conv_merging=False,
        _drop_vision_class_token=True,
        temporal_patch_dim=2,
    )
    model.vision_model = SimpleNamespace(temporal_patch_dim=2)

    vlm_wrapper = object.__new__(VLMInferenceWrapper)
    vlm_wrapper.model = SimpleNamespace(module=model)
    omni_wrapper = object.__new__(NemotronOmniInferenceWrapper)
    omni_wrapper.model = model

    kwargs = {
        "imgs_sizes": torch.tensor([[448, 576]] * 4),
        "num_frames": torch.tensor([4]),
    }
    vlm_expanded, vlm_masks = vlm_wrapper.expand_image_tokens(
        [[11, -200, -200, 12]], **kwargs
    )
    omni_expanded, omni_masks = omni_wrapper.expand_image_tokens(
        [[11, -200, -200, 12]], **kwargs
    )

    assert vlm_expanded == [[11] + [-1] * 504 + [12]]
    assert omni_expanded == [[11] + [-1] * 504 + [12]]
    assert vlm_masks == omni_masks
    assert vlm_masks[0][1:-1] == list(range(504))


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
            bf16=True,
        )
        vision_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=False,
            bf16=True,
        )
        vision_projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=False,
            bf16=True,
        )

        language_layer_submodules = get_gpt_layer_local_submodules()
        vision_layer_spec = ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(language_layer_submodules)
        )
        vision_projection_spec = copy.deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)

        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "clip"
        self.model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
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
        self.model = Float16Module(self.model.config, self.model)

        inference_context = StaticInferenceContext(max_batch_size=8, max_sequence_length=2560)

        inference_wrapped_model = VLMInferenceWrapper(self.model, inference_context)

        self.mock_tokenizer = mock.Mock()

        self.text_generation_controller = VLMTextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=self.mock_tokenizer
        )

        InferenceMode.set_active()

    def teardown_method(self, method):
        InferenceMode.unset_active()
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

            request_id = i
            inference_request = VLMInferenceRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=10),
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
