# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU forward/backward test for the RADIO vision encoder wrapper.

Builds the real ``RADIOEncoderWrapper`` (RADIOViTModel + TE) via
``radio_vision_encoder_spec`` and runs forward + backward on synthetic input,
exercising the class-token-drop and pixel-shuffle flags (which change the output
shape) plus the dynamic-resolution packed-tile path. Needs 1 GPU:

    WORLD_SIZE=1 python -m torch.distributed.run --nproc_per_node=1 -m pytest \
        tests/unit_tests/models/mimo/test_radio_encoder.py
"""

from types import SimpleNamespace

import pytest
import torch

from examples.mimo.model_providers.radio_encoder import (
    RADIOEncoderWrapper,
    radio_vision_encoder_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

IMG = 224
PATCH = 14
CLASS_TOKENS = 8
HIDDEN = 64
PATCHES = (IMG // PATCH) ** 2  # 16 * 16 = 256


def _build_wrapper(
    *,
    apply_pixel_shuffle,
    drop_class_token,
    dynamic_resolution,
    params_dtype=torch.float32,
    attention_backend=AttnBackend.auto,
):
    """Build the wrapper through the production spec builder, then instantiate it."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=params_dtype,
        bf16=params_dtype == torch.bfloat16,
        attention_backend=attention_backend,
    )
    args = SimpleNamespace(
        img_h=IMG,
        img_w=IMG,
        patch_dim=PATCH,
        class_token_len=CLASS_TOKENS,
        pixel_shuffle=apply_pixel_shuffle,
        disable_vision_class_token=drop_class_token,
        freeze_vit=False,
        dynamic_resolution=dynamic_resolution,
    )
    spec = radio_vision_encoder_spec(args, config, pg_collection=None)
    assert spec.module is RADIOEncoderWrapper
    return spec.module(**spec.params).cuda()


def _has_finite_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in module.parameters()
        if p.requires_grad
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RADIO encoder forward needs a GPU")
class TestRADIOEncoderWrapper:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "apply_pixel_shuffle,drop_class_token,expected_seq,expected_hidden",
        [
            # Raw RADIO output keeps the class tokens.
            (False, False, PATCHES + CLASS_TOKENS, HIDDEN),
            # Class-token drop removes class_token_len tokens.
            (False, True, PATCHES, HIDDEN),
            # Drop + 0.5x-per-axis pixel shuffle: seq /= 4, hidden *= 4.
            (True, True, PATCHES // 4, HIDDEN * 4),
        ],
    )
    def test_fixed_resolution_forward_backward(
        self, apply_pixel_shuffle, drop_class_token, expected_seq, expected_hidden
    ):
        wrapper = _build_wrapper(
            apply_pixel_shuffle=apply_pixel_shuffle,
            drop_class_token=drop_class_token,
            dynamic_resolution=False,
        )
        x = torch.randn(2, 3, IMG, IMG, device="cuda")

        out = wrapper(x)
        assert out.shape == torch.Size([2, expected_seq, expected_hidden])

        out.sum().backward()
        assert _has_finite_grad(wrapper)

    def test_dynamic_resolution_forward_backward(self):
        # Packed variable-tile path: one square tile of rows*cols patches, fed as
        # pre-patchified features (matches the dynamic-resolution data builder).
        # The packed (thd) attention path requires bf16 + a flash/fused backend
        # (the fixed sbhd path tolerates fp32; this one does not). TE fused attn
        # needs cu_seqlens on CUDA (mirrors training/step.py::move_batch_to_cuda,
        # which moves the PackedSeqParams index tensors to the device); max_seqlen
        # is passed as plain ints; imgs_sizes stays on CPU since RADIOViTModel reads
        # it via .tolist()/Python iteration. RADIOViTModel itself adds
        # class_token_len per tile to cu_seqlens.
        wrapper = _build_wrapper(
            apply_pixel_shuffle=True,
            drop_class_token=True,
            dynamic_resolution=True,
            params_dtype=torch.bfloat16,
            attention_backend=AttnBackend.flash,
        )
        rows = cols = 8
        patches = rows * cols
        feat_dim = 3 * PATCH * PATCH
        x = torch.randn(1, patches, feat_dim, device="cuda", dtype=torch.bfloat16)
        imgs_sizes = torch.tensor([[rows * PATCH, cols * PATCH]], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, patches], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=patches,
            max_seqlen_kv=patches,
        )

        out = wrapper(x, imgs_sizes=imgs_sizes, packed_seq_params=packed)
        assert out.dim() == 3 and out.shape[0] == 1

        out.sum().backward()
        assert _has_finite_grad(wrapper)
