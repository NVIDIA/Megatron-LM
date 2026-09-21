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

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.inference_request import InferenceRequest, Status, VLMInferenceRequest
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.utils import (
    dynamic_media_embedding_counts,
    dynamic_media_replacement_counts,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
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
def test_vlm_wrapper_builds_preexpanded_media_token_mask():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace(image_token_index=-200)

    mask = wrapper.build_preexpanded_media_token_mask(torch.tensor([10, -200, -200, 20]), "image")

    assert mask.tolist() == [-1, 0, 1, -1]


@pytest.mark.internal
def test_vlm_wrapper_resolves_media_id_from_tokenizer_not_model_sentinel():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace(image_token_index=-200)
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: 99 if token == "<image>" else 0, unk_token_id=0
    )

    assert wrapper.multimodal_prompt_config.image_spec.model_token == "<image>"
    assert wrapper.resolve_media_token_id(tokenizer, "image") == 99


@pytest.mark.internal
def test_vlm_wrapper_resolves_media_id_with_encode_fallback():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace(image_token_index=-200)
    tokenizer = SimpleNamespace(encode=lambda token, add_special_tokens: [99])

    assert wrapper.resolve_media_token_id(tokenizer, "image") == 99


@pytest.mark.internal
def test_vlm_dynamic_forward_sanitizes_preexpanded_media_sentinel():
    embedding = mock.Mock(return_value=torch.zeros(3, 1, 4))
    model = SimpleNamespace(
        image_token_index=-200,
        language_model=SimpleNamespace(embedding=embedding),
        forward_lm_only=mock.Mock(return_value=torch.zeros(1, 3, 8)),
    )
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = model
    wrapper.pp_group = mock.Mock()
    wrapper._recv_only_vision_embeds = False
    wrapper.inference_context = None

    with mock.patch(
        "megatron.core.inference.model_inference_wrappers.multimodal."
        "vlm_inference_wrapper.is_pipeline_first_stage",
        return_value=True,
    ):
        wrapper._forward_dynamic(
            {
                "tokens": torch.tensor([[10, -200, -200]]),
                "position_ids": torch.tensor([[0, 1, 2]]),
                "image_token_mask": torch.tensor([[-1, 0, 1]]),
                "image_embeddings": torch.zeros(2, 1, 4),
                "attention_mask": None,
            }
        )

    assert torch.equal(embedding.call_args.kwargs["input_ids"], torch.tensor([[10, 0, 0]]))


@pytest.mark.internal
@pytest.mark.parametrize("is_video", [False, True])
def test_vlm_vision_forward_casts_images_and_passes_tensor_input_ids(is_video):
    module = SimpleNamespace(
        image_token_index=-200,
        vision_model=SimpleNamespace(config=SimpleNamespace(params_dtype=torch.bfloat16)),
        dynamic_resolution=False,
        add_decoder=True,
    )

    class WrappedVisionModel:
        def __init__(self):
            self.module = module
            self.call_args = None

        def __call__(self, *args, **kwargs):
            self.call_args = (args, kwargs)
            return torch.zeros(2, 1, 4, dtype=torch.bfloat16), None

    model = WrappedVisionModel()
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = model
    wrapper.inference_context = None

    output = wrapper._forward_vision_encoder(
        torch.ones(1, 3, 2, 2, dtype=torch.float32),
        num_image_tiles=torch.tensor([1]),
        num_frames=torch.tensor([1]) if is_video else None,
    )

    args, _ = model.call_args
    assert args[0].dtype == torch.bfloat16
    assert isinstance(args[1], torch.Tensor)
    assert args[1].shape == (1, 0)
    assert args[1].device == args[0].device
    assert module.add_decoder is True
    assert output.dtype == torch.bfloat16


@pytest.mark.internal
def test_dynamic_video_embedding_counts_group_one_placeholder_per_video():
    frame_counts = dynamic_media_embedding_counts(
        torch.tensor([[448, 576]] * 4), patch_dim=16, pixel_shuffle=True
    )
    assert frame_counts == [252] * 4

    assert dynamic_media_replacement_counts(
        frame_counts, num_frames=torch.tensor(4), temporal_patch_size=2
    ) == [504]
    assert dynamic_media_replacement_counts(
        frame_counts, num_frames=torch.tensor(4), temporal_patch_size=2, aggregate_videos=False
    ) == [252, 252]


@pytest.mark.internal
def test_nemotron_video_expansion_adds_timestamped_tubelet_wrappers():
    wrapper = object.__new__(NemotronOmniInferenceWrapper)
    wrapper.multimodal_prompt_config = MultimodalPromptConfig(
        video_spec=MediaPromptSpec(
            model_token="<image>",
            prefix="<img>",
            suffix="</img>",
            expansion_mode="temporal_patch",
            include_frame_timestamps_for_nemotron_vl=True,
        )
    )
    wrapper.model = SimpleNamespace(
        image_token_index=-200,
        dynamic_resolution=True,
        patch_dim=16,
        vision_model=SimpleNamespace(temporal_patch_dim=2),
    )

    class _Tokenizer:
        def __init__(self):
            self.rendered = None

        def tokenize(self, text):
            if text == "<img>":
                return [77]
            if text == "</img>":
                return [78]
            self.rendered = text
            return [7, 77, 99, 78, 7, 77, 99, 78]

    tokenizer = _Tokenizer()
    expanded, masks = wrapper.expand_image_tokens(
        [[11, 77, 99, 78, 12]],
        imgs_sizes=torch.tensor([[32, 32]] * 4),
        num_frames=torch.tensor([4]),
        image_token_id=99,
        tokenizer=tokenizer,
        video_frame_indices=[[0, 30, 60, 90]],
        video_fps=[29.97],
    )

    assert tokenizer.rendered == (
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.99 seconds: "
        "<img><image></img>\n"
        "Frame 3 sampled at 1.98 seconds and frame 4 sampled at 2.97 seconds: "
        "<img><image></img>"
    )
    assert expanded == [[11, 7, 77, -1, 78, 7, 77, -1, 78, 12]]
    assert masks == [[None, None, None, 0, None, None, None, 1, None, None]]


@pytest.mark.internal
def test_dynamic_video_embedding_counts_reject_misaligned_frames():
    with pytest.raises(ValueError, match="must partition"):
        dynamic_media_replacement_counts(
            [252] * 4, num_frames=torch.tensor([3]), temporal_patch_size=2
        )


@pytest.mark.internal
def test_super_video_geometry_has_32_tubelets_and_8192_embeddings():
    counts = dynamic_media_replacement_counts(
        [256] * 64, num_frames=torch.tensor([64]), temporal_patch_size=2, aggregate_videos=False
    )

    assert len(counts) == 32
    assert sum(counts) == 8192


@pytest.mark.internal
def test_vlm_wrapper_expands_one_video_marker_to_all_tubelet_embeddings():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace()
    wrapper.model.module = SimpleNamespace(
        image_token_index=-200,
        dynamic_resolution=True,
        patch_dim=16,
        _pixel_shuffle=True,
        _conv_merging=False,
        _drop_vision_class_token=True,
        temporal_patch_dim=2,
    )

    expanded, masks = wrapper.expand_image_tokens(
        [[11, 99, 12]],
        imgs_sizes=torch.tensor([[448, 576]] * 4),
        num_frames=torch.tensor([4]),
        image_token_id=99,
    )

    assert expanded == [[11] + [-1] * 504 + [12]]
    assert masks[0][0] is None
    assert masks[0][1:-1] == list(range(504))
    assert masks[0][-1] is None


@pytest.mark.internal
def test_vlm_wrapper_rejects_multiple_compact_markers_for_one_video():
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace()
    wrapper.model.module = SimpleNamespace(
        image_token_index=-200,
        dynamic_resolution=True,
        patch_dim=16,
        _pixel_shuffle=True,
        _conv_merging=False,
        _drop_vision_class_token=True,
        temporal_patch_dim=2,
    )

    with pytest.raises(ValueError, match="one compact placeholder per video"):
        wrapper.expand_image_tokens(
            [[11, 99, 99, 12]],
            imgs_sizes=torch.tensor([[448, 576]] * 4),
            num_frames=torch.tensor([4]),
            image_token_id=99,
        )


@pytest.mark.internal
def test_omni_text_forward_does_not_treat_generated_image_token_as_media():
    """A decode token matching the image-token ID has no projected feature."""
    tokens = torch.tensor([[18]])
    position_ids = torch.tensor([[7]])
    language_embeddings = torch.empty((1, 1, 8))
    output = torch.empty((1, 1, 16))
    language_model = mock.Mock(return_value=output)
    language_model.embedding = mock.Mock(return_value=language_embeddings)
    model = SimpleNamespace(
        image_token_index=18, language_model=language_model, sequence_parallel_lm=False
    )
    wrapper = object.__new__(NemotronOmniInferenceWrapper)
    wrapper.model = model
    wrapper.inference_context = mock.sentinel.inference_context

    result = wrapper._forward(
        {"tokens": tokens, "position_ids": position_ids, "attention_mask": None}
    )

    assert result is output
    assert language_model.embedding.call_args.kwargs["input_ids"] is tokens
    language_model.assert_called_once()


@pytest.mark.internal
def test_omni_raw_image_forward_calls_full_model():
    """Manually supplied raw images use NemotronOmniModel.forward."""
    tokens = torch.tensor([[7, 18, 9]])
    position_ids = torch.tensor([[0, 1, 2]])
    images = torch.empty((1, 3, 16, 16))
    imgs_sizes = torch.tensor([[16, 16]])
    output = torch.empty((1, 3, 16))
    wrapper = object.__new__(NemotronOmniInferenceWrapper)
    wrapper.model = mock.Mock(return_value=(output, None))
    wrapper.inference_context = mock.sentinel.inference_context

    result = wrapper._forward(
        {
            "tokens": tokens,
            "position_ids": position_ids,
            "attention_mask": None,
            "images": images,
            "imgs_sizes": imgs_sizes,
        }
    )

    assert result is output
    call_kwargs = wrapper.model.call_args.kwargs
    assert call_kwargs["images"] is images
    assert call_kwargs["input_ids"] is tokens
    assert call_kwargs["imgs_sizes"] is imgs_sizes


@pytest.mark.internal
@pytest.mark.parametrize("wrapper_cls", [VLMInferenceWrapper, NemotronOmniInferenceWrapper])
def test_image_token_mask_takes_precedence_over_raw_images(wrapper_cls):
    """A dynamic media mask selects the LM-only path even when raw images are present."""
    tokens = torch.tensor([[7, -200, 9]])
    position_ids = torch.tensor([[0, 1, 2]])
    images = torch.empty((1, 3, 16, 16))
    output = torch.empty((1, 3, 16))
    wrapper = object.__new__(wrapper_cls)
    wrapper.model = mock.Mock()
    wrapper.inference_context = mock.sentinel.inference_context
    wrapper._forward_dynamic = mock.Mock(return_value=output)

    inference_input = {
        "tokens": tokens,
        "position_ids": position_ids,
        "attention_mask": None,
        "images": images,
        "num_tiles": torch.tensor([1]),
        "image_token_mask": torch.tensor([[-1, 0, -1]]),
        "image_embeddings": torch.empty((1, 1, 16)),
    }
    result = wrapper._forward(inference_input)

    assert result is output
    wrapper._forward_dynamic.assert_called_once_with(inference_input)
    wrapper.model.assert_not_called()


@pytest.mark.internal
def test_llava_text_forward_embeds_generated_image_token_as_text():
    """Media-free decode must not replace a generated image-token ID with token 0."""
    tokens = torch.tensor([[18]])
    position_ids = torch.tensor([[7]])
    language_embeddings = torch.empty((1, 1, 8))
    output = torch.empty((1, 1, 16))
    embedding = mock.Mock(return_value=language_embeddings)
    model = SimpleNamespace(
        image_token_index=18,
        language_model=SimpleNamespace(embedding=embedding),
        forward_lm_only=mock.Mock(return_value=output),
    )
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = model
    wrapper.inference_context = mock.sentinel.inference_context
    wrapper.pp_group = None
    wrapper._recv_only_vision_embeds = False

    with mock.patch(
        "megatron.core.inference.model_inference_wrappers.multimodal."
        "vlm_inference_wrapper.is_pipeline_first_stage",
        return_value=True,
    ):
        result = wrapper._forward(
            {"tokens": tokens, "position_ids": position_ids, "attention_mask": None}
        )

    assert result is output
    embedded_input_ids = embedding.call_args.kwargs["input_ids"]
    assert torch.equal(embedded_input_ids, tokens)
    assert embedded_input_ids is not tokens
    model.forward_lm_only.assert_called_once()


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
