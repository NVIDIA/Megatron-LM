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
        input_ids=kwargs.pop("input_ids", torch.tensor([[1, 2]], dtype=torch.long)),
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


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("conv_merging", [False, True])
@pytest.mark.parametrize("balance_by_tokens", [False, True])
def test_forward_video_accounts_for_native_merging(
    monkeypatch, cp_size, conv_merging, balance_by_tokens
):
    """Count merged frame tokens and provide merger-compatible CP dummy images."""
    from megatron.core.models.multimodal import context_parallel, llava_model

    patch_dim = 2
    num_frames = 2
    tokens_per_frame = 1 if conv_merging else 4
    total_tokens = num_frames * tokens_per_frame
    vision_sizes = []
    gathered_padding = []

    class _VisionModel(torch.nn.Module):
        dynamic_resolution = True
        class_token_len = 0

        def __init__(self):
            super().__init__()
            self.patch_dim = patch_dim

        def forward(self, images, *, imgs_sizes, packed_seq_params):
            vision_sizes.extend(imgs_sizes.tolist())
            patch_hw = imgs_sizes // patch_dim
            if conv_merging:
                assert torch.all(patch_hw % 2 == 0), "native merger requires a 2x2 patch grid"
                patch_hw = patch_hw // 2
            token_count = int(patch_hw.prod(dim=-1).sum())
            return torch.ones(1, token_count, 2)

    def gather(local_embeddings, num_padded_ranks):
        gathered_padding.append(num_padded_ranks)
        assert local_embeddings.shape[0] == (1 if cp_size > num_frames else tokens_per_frame)
        return torch.ones(total_tokens, 1, 2)

    monkeypatch.setattr(context_parallel, "get_context_parallel_world_size", lambda: cp_size)
    # For CP=4, the last rank owns a dummy image; for CP=2 it owns a real frame.
    monkeypatch.setattr(context_parallel, "get_context_parallel_rank", lambda: cp_size - 1)
    monkeypatch.setattr(llava_model, "gather_from_context_parallel_ranks_dynamic_res", gather)

    model = object.__new__(LLaVAModel)
    torch.nn.Module.__init__(model)
    model.add_encoder = True
    model.add_decoder = True
    model.pre_process = False
    model.temporal_patch_dim = 1
    model.patch_dim = patch_dim
    model.vision_model = _VisionModel()
    model.vision_projection = torch.nn.Identity()
    model.language_model = lambda **kwargs: kwargs["decoder_input"]
    model.image_token_index = -200
    model.context_parallel_lm = cp_size
    model.sequence_parallel_lm = False
    model.use_loss_scaling = False
    model._drop_vision_class_token = True
    model._pixel_shuffle = False
    model._conv_merging = conv_merging
    model._tile_tags = None
    model._vision_fp8 = False
    model._vision_fp8_recipe = None
    model._vision_projection_fp8 = False
    model._balance_vision_context_parallel_by_tokens = balance_by_tokens
    model._profile_vision_context_parallel_partition = False
    captured = {}

    def preprocess(image_embeddings, *args, **kwargs):
        captured["media_token_counts"] = kwargs["media_token_counts"]
        assert image_embeddings.shape == (total_tokens, 1, 2)
        return image_embeddings, None, None, None, None

    model._preprocess_data = preprocess
    model._process_embedding_token_parallel = lambda *args: args

    output, loss_mask = _minimal_llava_forward(
        model,
        input_ids=torch.tensor([[1, model.image_token_index]], dtype=torch.long),
        images=torch.ones(1, num_frames * 4, 3 * patch_dim**2),
        imgs_sizes=torch.tensor([[4, 4]] * num_frames, dtype=torch.int32),
        num_frames=[num_frames],
    )

    assert captured["media_token_counts"].tolist() == [total_tokens]
    assert output.shape == (total_tokens, 1, 2)
    assert loss_mask is None
    if cp_size == 1:
        assert vision_sizes == [[4, 4], [4, 4]]
        assert gathered_padding == []
    else:
        dummy_size = 2 * patch_dim if conv_merging else patch_dim
        expected_size = dummy_size if cp_size > num_frames else 4
        assert vision_sizes == [[expected_size, expected_size]]
        assert gathered_padding == [max(0, cp_size - num_frames)]


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
        input_ids=torch.tensor([[1, model.image_token_index]], dtype=torch.long),
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
