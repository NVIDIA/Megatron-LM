# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_model import (
    KimiViTModel,
    Pixtral2DRotaryEmbedding,
    QwenVLViTModel,
    ViTModel,
    _dynamic_patch_grid_lists,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, defer_parameter_gathers


@pytest.mark.parametrize("model_cls", [ViTModel, QwenVLViTModel, KimiViTModel])
def test_dynamic_patches_wait_for_projection(model_cls):
    """Packed patches bypass both patch_embed and its projection's forward hooks."""
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        kwargs = {}
        if model_cls is QwenVLViTModel:
            kwargs = dict(out_hidden_size=16, num_pos_per_side=2)
        elif model_cls is KimiViTModel:
            kwargs = dict(pos_embed_height=2, pos_embed_width=2)
        model = model_cls(
            config,
            get_gpt_layer_with_transformer_engine_spec(),
            patch_dim=2,
            img_h=4,
            img_w=4,
            **kwargs,
        ).cuda()
        patches = torch.randn(1, 4, 12, device="cuda", requires_grad=True)
        reference = model(patches, imgs_sizes=[(4, 4)])
        with defer_parameter_gathers(model, model.patch_embed.proj.parameters()):
            actual = model(patches, imgs_sizes=[(4, 4)])
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        actual.square().mean().backward()
        assert torch.isfinite(patches.grad).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "imgs_sizes", [[(28, 28), (28, 56)], torch.tensor([[28, 28], [28, 56]], dtype=torch.int32)]
)
def test_dynamic_patch_grid_lists_keeps_metadata_on_cpu(imgs_sizes):
    patch_hw, seq_lens = _dynamic_patch_grid_lists(imgs_sizes, patch_dim=14)

    assert patch_hw == [(2, 2), (2, 4)]
    assert seq_lens == [4, 8]


def test_dynamic_patch_grid_lists_rejects_cuda_metadata():
    imgs_sizes = torch.tensor([[28, 28]], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="must remain on CPU"):
        _dynamic_patch_grid_lists(imgs_sizes, patch_dim=14)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("patch_grid", [(7, 11), (110, 110)])
@pytest.mark.parametrize("conversion", ["cast_before_cuda", "cuda_before_cast", "float16_module"])
def test_vision_rope_preserves_fp32_angles(dtype, head_dim, patch_grid, conversion):
    """Model conversion must preserve FP32 RoPE phases and the resulting Q/K rotation."""
    device = torch.device("cuda", torch.cuda.current_device())
    h_patches, w_patches = patch_grid
    reference = Pixtral2DRotaryEmbedding(head_dim, 110)
    expected = reference(h_patches, w_patches, device)

    model = torch.nn.Module()
    model.rope = Pixtral2DRotaryEmbedding(head_dim, 110)
    model.proj = torch.nn.Linear(head_dim, head_dim, bias=False)
    if conversion == "cast_before_cuda":
        model.to(dtype=dtype).to(device=device)
    elif conversion == "cuda_before_cast":
        model.to(device=device).to(dtype=dtype)
    else:
        model.to(device=device)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=head_dim,
            num_attention_heads=1,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
        )
        Float16Module(config, model)

    actual = model.rope(h_patches, w_patches, device)
    assert model.proj.weight.dtype == dtype
    assert model.rope.freqs.dtype == torch.float32
    assert model.rope.freqs.device == device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "rope.freqs" not in model.state_dict()

    queries = torch.randn(h_patches * w_patches, 1, 2, head_dim, device=device, dtype=dtype)
    rotated = _apply_rotary_pos_emb_bshd(queries, actual, rotary_interleaved=True)
    expected_rotated = _apply_rotary_pos_emb_bshd(queries, expected, rotary_interleaved=True)
    assert rotated.dtype == dtype
    torch.testing.assert_close(rotated, expected_rotated, rtol=0, atol=0)

    # Repeated images reuse the device copy, including after another model cast.
    cached_freqs = model.rope.freqs
    model.to(dtype=dtype)
    repeated = model.rope(h_patches, w_patches, device)
    assert model.rope.freqs is cached_freqs
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)


def test_vision_rope_moves_constants_to_input_device():
    """Lazy constant placement must follow inputs after moving the parent module."""
    model = torch.nn.Module()
    model.rope = Pixtral2DRotaryEmbedding(128, 110)
    expected = model.rope(7, 11, torch.device("cpu"))
    model.to(device="cuda", dtype=torch.bfloat16)
    actual = model.rope(7, 11, torch.device("cuda", torch.cuda.current_device()))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    model.cpu()
    actual = model.rope(7, 11, torch.device("cpu"))
    assert model.rope.freqs.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
