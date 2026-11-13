# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native MIMO geometry and strict architecture-export initialization."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.distributed.checkpoint.metadata import TensorProperties, TensorStorageMetadata

from examples.mimo.model_providers.vision_encoder import load_native_vision_export
from examples.mimo.vision import (
    configure_vision_args,
    resolve_encoder_spec,
    vision_initialization_checkpoint,
)
from examples.multimodal.data_loading.conversation_sample import ImageMedia
from examples.multimodal.data_loading.image_processing import DynamicResolutionImageTilingStrategy


def _args(**options):
    values = dict(
        vision_model_type="pixtral-vit-1p6b",
        patch_dim=None,
        img_h=None,
        img_w=None,
        pixel_shuffle=False,
        dynamic_resolution=True,
        disable_vision_class_token=True,
        dataset_provider="energon",
        load_vision_from=None,
        load=None,
        pretrained_checkpoint=None,
        exit_on_missing_checkpoint=False,
    )
    values.update(options)
    return SimpleNamespace(**values)


def _count(width, height, *, spec, **kwargs):
    return spec.num_spatial_embeddings(height, width)


def _strategy():
    spec = resolve_encoder_spec(_args())
    return DynamicResolutionImageTilingStrategy(
        vision_model_type=spec.name,
        min_num_patches=4,
        max_num_patches=12100,
        patch_size=spec.patch_dim,
        spatial_merge_size=spec.spatial_merge_size,
        max_image_size=(spec.default_img_h, spec.default_img_w),
        get_num_embeddings=partial(_count, spec=spec),
    )


def _media(width, height, value=None):
    return ImageMedia(value=value, metadata={"width": width, "height": height})


def test_selected_geometry_resolves_before_validation_and_rejects_mismatch():
    args = _args()
    spec = configure_vision_args(args)
    assert spec.num_spatial_embeddings(args.img_h, args.img_w) == 3025
    assert (args.patch_dim, spec.spatial_merge_size) == (14, 2)
    with pytest.raises(ValueError, match="requires patch_dim=14"):
        configure_vision_args(_args(patch_dim=16))
    with pytest.raises(ValueError, match="pixel shuffle"):
        configure_vision_args(_args(pixel_shuffle=True))
    with pytest.raises(ValueError, match="Energon prepared-data"):
        configure_vision_args(_args(dataset_provider="mock"))


def test_native_budget_counts_outputs_after_merging_across_images():
    plans = _strategy().compute_params([_media(112, 112), _media(112, 112)], 32)
    assert [plan.num_embeddings for plan in plans] == [16, 16]
    assert [plan.patch_size for plan in plans] == [(8, 8), (8, 8)]


@pytest.mark.parametrize("width,height", [(11200, 5600), (5600, 11200)])
def test_native_axis_limits_preserve_aspect_ratio(width, height):
    (plan,) = _strategy().compute_params([_media(width, height)], 3025)
    resized_width, resized_height = (value * 14 for value in plan.patch_size)
    assert 0 < resized_width <= 1540 and 0 < resized_height <= 1540
    assert resized_width % 28 == resized_height % 28 == 0
    assert plan.num_embeddings <= 3025
    assert abs(resized_width / resized_height - width / height) < 0.05


@pytest.mark.parametrize("width,height", [(1, 1), (1, 100000), (100000, 1)])
def test_native_minimum_budget_and_extreme_aspect_ratios(width, height):
    (plan,) = _strategy().compute_params([_media(width, height)], 1)
    assert plan.patch_size == (2, 2)
    assert plan.num_embeddings == 1


def test_native_pixels_keep_clip_normalization_and_premerge_attention_boundaries():
    strategy = _strategy()
    source = Image.new("RGB", (56, 56), (255, 0, 0))
    (plan,) = strategy.compute_params([_media(56, 56, source)], 4)
    tiles = strategy.apply_params(plan)
    pixels, sizes, boundaries, maximum = strategy.stack(tiles)
    assert plan.num_embeddings == 4
    assert sizes.device.type == "cpu" and sizes.tolist() == [[56, 56]]
    assert pixels.shape == (1, 16, 3 * 14 * 14)
    assert boundaries.tolist() == [0, 16] and int(maximum) == 16
    expected = torch.tensor(
        [(1 - 0.48145466) / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711]
    )
    torch.testing.assert_close(tiles[0][:, 0, 0], expected)


def test_full_resume_cannot_silently_fall_back_to_frozen_random_weights():
    args = _args(load_vision_from="vision-export", load="training-checkpoint")
    with pytest.raises(ValueError, match="exit-on-missing-checkpoint"):
        configure_vision_args(args)
    args.exit_on_missing_checkpoint = True
    assert vision_initialization_checkpoint(args) is None
    args.load = None
    assert vision_initialization_checkpoint(args) == "vision-export"
    args.pretrained_checkpoint = "strict-full-model-source"
    assert vision_initialization_checkpoint(args) is None


class _RuntimeStateLinear(torch.nn.Linear):
    def get_extra_state(self):
        return {"runtime": "preserved"}

    def set_extra_state(self, value):
        assert value == {"runtime": "preserved"}


def _export_fixture(monkeypatch, model):
    import torch.distributed.checkpoint as checkpoint

    group = object()
    values = {
        "model.vision_model." + name: torch.full_like(parameter, index + 1, dtype=torch.bfloat16)
        for index, (name, parameter) in enumerate(model.named_parameters())
    }
    entries = {
        name: TensorStorageMetadata(
            properties=TensorProperties(dtype=value.dtype), size=value.shape, chunks=[]
        )
        for name, value in values.items()
    }
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda candidate: 1 if candidate is group else 2
    )
    monkeypatch.setattr(
        checkpoint,
        "FileSystemReader",
        lambda directory: SimpleNamespace(
            read_metadata=lambda: SimpleNamespace(state_dict_metadata=entries)
        ),
    )
    calls = []

    def load(slots, *, checkpoint_id, process_group):
        assert process_group is group, "language ranks must never enter this load"
        calls.append(checkpoint_id)
        for name, target in slots.items():
            assert target.device.type == "cpu"
            target.copy_(values[name])

    monkeypatch.setattr(checkpoint, "load", load)
    return group, values, entries, calls


def test_export_load_preserves_parameters_and_runtime_state(monkeypatch):
    model = _RuntimeStateLinear(3, 2)
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    group, values, _, calls = _export_fixture(monkeypatch, model)
    load_native_vision_export(model, "export", group)
    assert calls == ["export"]
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert model.get_extra_state() == {"runtime": "preserved"}
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, values["model.vision_model." + name].float())


@pytest.mark.parametrize("invalid", ["key", "shape", "dtype"])
def test_export_load_rejects_incomplete_or_incompatible_parameters(monkeypatch, invalid):
    model = _RuntimeStateLinear(3, 2)
    group, _, entries, calls = _export_fixture(monkeypatch, model)
    name = "model.vision_model.weight"
    if invalid == "key":
        del entries[name]
    elif invalid == "shape":
        entries[name].size = torch.Size([1])
    else:
        entries[name].properties.dtype = torch.float32
    with pytest.raises(ValueError, match="every native encoder parameter|shape/dtype"):
        load_native_vision_export(model, "export", group)
    assert calls == []


def test_export_load_rejects_missing_group_and_meta_parameters(monkeypatch):
    model = _RuntimeStateLinear(3, 2)
    group, _, _, calls = _export_fixture(monkeypatch, model)
    with pytest.raises(ValueError, match="explicit encoder TP1"):
        load_native_vision_export(model, "export", None)
    with pytest.raises(ValueError, match="materialized"):
        load_native_vision_export(_RuntimeStateLinear(3, 2, device="meta"), "export", group)
    assert calls == []
