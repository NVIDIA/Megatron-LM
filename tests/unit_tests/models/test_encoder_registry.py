# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Spatial geometry contracts for existing vision encoder defaults."""

from dataclasses import replace

import pytest

from megatron.core.models.vision.encoder_registry import EncoderSpec, get_spec


@pytest.mark.parametrize(
    ("name", "reduction", "expected"),
    [
        ("clip", 1, 1024),
        ("post-c-radio-omni", 2, 256),
        ("pixtral-vit-large", 2, 3025),
        ("qwen-vl", 2, 576),
        ("kimi-vit", 2, 1024),
    ],
)
def test_registered_spatial_geometry(name, reduction, expected):
    """Existing defaults account for plain patches, pixel shuffle and native merging."""
    spec = get_spec(name)
    assert spec.spatial_reduction == reduction
    assert spec.num_spatial_embeddings(spec.default_img_h, spec.default_img_w) == expected


@pytest.mark.parametrize(
    ("patch_dim", "spatial_merge_size", "expected"),
    [(14, 1, 512), (16, 1, 392), (14, 2, 128), (16, 2, 98)],
)
def test_rectangular_spatial_geometry(patch_dim, spatial_merge_size, expected):
    """Identical image dimensions yield counts specific to patch size and reduction."""
    spec = EncoderSpec(
        name="custom",
        patch_dim=patch_dim,
        default_img_h=224,
        default_img_w=448,
        has_native_spatial_merge=spatial_merge_size > 1,
        spatial_merge_size=spatial_merge_size,
    )
    assert spec.num_spatial_embeddings(img_h=224, img_w=448) == expected


def test_class_tokens_excluded():
    """Class tokens do not contribute to spatial outputs."""
    plain = replace(get_spec("clip"), class_token_len=8)
    assert plain.num_spatial_embeddings(img_h=112, img_w=224) == 98


def test_explicit_merge_geometry():
    """The numeric merger factor controls counting, including composed spatial reductions."""
    spec = EncoderSpec(
        name="custom",
        patch_dim=4,
        default_img_h=24,
        default_img_w=36,
        has_native_spatial_merge=True,
        spatial_merge_size=3,
    )
    assert spec.spatial_reduction == 3
    assert spec.num_spatial_embeddings(img_h=24, img_w=36) == 6
    composed = replace(get_spec("pixtral-vit-large"), pixel_shuffle=True)
    assert composed.spatial_reduction == 4
    assert composed.num_spatial_embeddings(img_h=112, img_w=224) == 8


@pytest.mark.parametrize(
    ("name", "img_h", "img_w"),
    [
        ("clip", 15, 32),
        ("clip", 32, 15),
        ("pixtral-vit-large", 42, 56),
        ("pixtral-vit-large", 56, 42),
        ("post-c-radio-omni", 48, 64),
        ("post-c-radio-omni", 64, 48),
        ("pixtral-vit-large", 0, 112),
        ("pixtral-vit-large", 112, 0),
        ("pixtral-vit-large", -28, 112),
        ("pixtral-vit-large", 112, -28),
        ("pixtral-vit-large", 112.0, 112),
        ("pixtral-vit-large", True, 112),
    ],
)
def test_invalid_image_geometry(name, img_h, img_w):
    """Each axis must be a positive integer aligned before and after spatial reduction."""
    with pytest.raises(ValueError):
        get_spec(name).num_spatial_embeddings(img_h, img_w)


@pytest.mark.parametrize(("field", "value"), [("patch_dim", 0), ("spatial_merge_size", 0)])
def test_invalid_spec_geometry(field, value):
    """Invalid geometry fails at counting instead of dividing by zero."""
    spec = replace(get_spec("clip"), **{field: value})
    with pytest.raises(ValueError):
        spec.num_spatial_embeddings(112, 224)
