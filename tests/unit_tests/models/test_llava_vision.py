# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Focused image and video tests for ``LLaVAModel``."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.radio import RADIOViTModel


def _minimal_llava_forward(model, **kwargs):
    return LLaVAModel.forward(
        model,
        images=kwargs.pop("images", torch.ones(2, 3, 2, 2)),
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        attention_mask=None,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("imgs_sizes", "num_frames", "error"),
    [
        (None, [2], "Video inputs require imgs_sizes"),
        (torch.tensor([[2, 2], [2, 2]]), [0, 2], "num_frames entries must be positive"),
        (torch.tensor([[2, 2], [2, 2]]), [1], "num_frames must partition imgs_sizes exactly"),
    ],
)
def test_forward_rejects_invalid_video_frame_partitions(imgs_sizes, num_frames, error):
    model = object.__new__(LLaVAModel)
    model.add_encoder = True
    model.temporal_patch_dim = 1
    model.vision_model = SimpleNamespace(dynamic_resolution=False)

    with pytest.raises(ValueError, match=error):
        _minimal_llava_forward(model, imgs_sizes=imgs_sizes, num_frames=num_frames)


def test_forward_temporal_video_groups_tubelet_counts_per_placeholder():
    vision_model = object.__new__(RADIOViTModel)
    torch.nn.Module.__init__(vision_model)
    vision_model.dynamic_resolution = False
    vision_model.patch_dim = 2
    vision_model.class_token_len = 0
    vision_model.add_class_token = False
    vision_calls = []

    def vision_forward(images, *, imgs_sizes, packed_seq_params, num_frames):
        vision_calls.append((images, imgs_sizes, packed_seq_params, num_frames))
        post_sizes = torch.tensor([[4, 4], [4, 4]], dtype=torch.int32)
        return torch.arange(16, dtype=torch.float32).reshape(1, 8, 2), post_sizes, None

    vision_model.forward = vision_forward

    class _LanguageModel(torch.nn.Module):
        def embedding(self, input_ids, position_ids):
            del position_ids
            return torch.zeros(input_ids.shape[1], input_ids.shape[0], 2)

        def forward(self, **kwargs):
            return kwargs["decoder_input"]

    model = object.__new__(LLaVAModel)
    torch.nn.Module.__init__(model)
    model.add_encoder = True
    model.add_decoder = True
    model.pre_process = True
    model.temporal_patch_dim = 2
    model.vision_model = vision_model
    model.vision_projection = torch.nn.Identity()
    model.language_model = _LanguageModel()
    model.image_token_index = -200
    model.sound_token_index = -300
    model.context_parallel_lm = 1
    model.sequence_parallel_lm = False
    model._drop_vision_class_token = True
    model._pixel_shuffle = False
    model._conv_merging = False
    model._tile_tags = None
    captured = {}

    def preprocess(*args, **kwargs):
        captured["media_token_counts"] = kwargs["media_token_counts"]
        return torch.zeros(2, 1, 2), None, None, None, None

    model._preprocess_data = preprocess

    output, loss_mask = _minimal_llava_forward(
        model,
        images=torch.ones(4, 3, 4, 4),
        imgs_sizes=torch.tensor([[4, 4]] * 4, dtype=torch.int32),
        num_frames=[4],
    )

    assert vision_calls[0][3] == [4]
    assert captured["media_token_counts"].tolist() == [8]
    assert output.shape == (2, 1, 2)
    assert loss_mask is None


@pytest.mark.parametrize(("modality", "num_media_embeddings"), [("image", 4), ("video", 8)])
def test_forward_image_and_video_inference_prefill_reuses_kv_cache(modality, num_media_embeddings):
    class _VisionModel(torch.nn.Module):
        dynamic_resolution = False
        class_token_len = 0

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, images, **kwargs):
            self.calls.append((images, kwargs))
            return torch.ones(images.shape[0], 4, 2)

    class _LanguageModel(torch.nn.Module):
        def embedding(self, input_ids, position_ids):
            del position_ids
            return torch.zeros(input_ids.shape[1], input_ids.shape[0], 2)

        def forward(self, **kwargs):
            return kwargs["decoder_input"]

    model = object.__new__(LLaVAModel)
    torch.nn.Module.__init__(model)
    model.add_encoder = True
    model.add_decoder = True
    model.pre_process = True
    model.post_process = True
    model.vision_model = _VisionModel()
    model.vision_projection = torch.nn.Identity()
    model.sound_model = None
    model.sound_projection = None
    model.language_model = _LanguageModel()
    model.encoder_hidden_state = None
    model.image_token_index = -200
    model.sound_token_index = -300
    model.temporal_patch_dim = 1
    model.img_seq_len = 4
    model.patch_dim = 2
    model.dynamic_resolution = False
    model._drop_vision_class_token = False
    model._pixel_shuffle = False
    model._conv_merging = False
    model._tile_tags = None
    model._max_num_tiles = 1
    model._language_max_sequence_length = 64
    model._language_is_pipeline_parallel = False
    model.context_parallel_lm = 1
    model.sequence_parallel_lm = False

    if modality == "video":
        images = torch.ones(2, 3, 4, 4)
        imgs_sizes = torch.tensor([[4, 4], [4, 4]], dtype=torch.int32)
        num_frames = [2]
    else:
        images = torch.ones(1, 3, 4, 4)
        imgs_sizes = None
        num_frames = None

    inference_context = SimpleNamespace(key_value_memory_dict={})
    prefill_output, _ = LLaVAModel.forward(
        model,
        images=images,
        input_ids=torch.tensor([[10, model.image_token_index, 11]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=None,
        imgs_sizes=imgs_sizes,
        num_frames=num_frames,
        inference_context=inference_context,
    )

    assert len(model.vision_model.calls) == 1
    assert inference_context.key_value_memory_dict["image_tokens_count"] == num_media_embeddings
    assert prefill_output.shape == (num_media_embeddings + 2, 1, 2)
    assert int(torch.all(prefill_output == 1, dim=-1).sum()) == num_media_embeddings

    decode_output, _ = LLaVAModel.forward(
        model,
        images=None,
        input_ids=torch.tensor([[12]]),
        position_ids=torch.tensor([[3]]),
        attention_mask=None,
        inference_context=inference_context,
    )

    assert len(model.vision_model.calls) == 1
    assert decode_output.shape == (1, 1, 2)
