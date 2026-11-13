# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility contracts for encoder wrappers shared with the MIMO example."""

from unittest.mock import patch

import pytest
import torch

from examples.mimo.model_providers import radio_encoder as example_radio
from examples.mimo.training.runtime import _EncoderFloat16Module
from megatron.core.models.mimo.model import MimoEncoderFloat16Module
from megatron.core.models.mimo.submodules import radio_encoder as core_radio
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_example_import_compatibility():
    assert example_radio.RADIOEncoderWrapper is core_radio.RADIOEncoderWrapper
    assert example_radio.RADIO_ENCODER_MODULE_NAME == core_radio.RADIO_ENCODER_MODULE_NAME
    assert example_radio._pixel_shuffle_dynamic_res is core_radio._pixel_shuffle_dynamic_res
    assert _EncoderFloat16Module is MimoEncoderFloat16Module


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_encoder_precision_and_checkpoint_compatibility(dtype):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            params_dtype=dtype,
        )
        layer = torch.nn.Linear(4, 4, device='cuda')
        wrapper = MimoEncoderFloat16Module(config, layer)
        x = torch.randn(2, 4, device='cuda')
        expected = layer(x.to(dtype))
        torch.testing.assert_close(wrapper(x), expected)
        torch.testing.assert_close(wrapper(x, fp32_output=True), expected.float())
        assert wrapper(x).dtype == dtype
        assert wrapper(x, fp32_output=True).dtype == torch.float32
        assert set(wrapper.state_dict()) == set(layer.state_dict())
        wrapper(x).sum().backward()
        assert layer.weight.grad is not None
    finally:
        Utils.destroy_model_parallel()


class _VisionBackbone(torch.nn.Module):
    """Supply deterministic embeddings to isolate wrapper postprocessing."""

    def __init__(self, **kwargs):
        super().__init__()
        self.embedder = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.patch_dim = kwargs['patch_dim']

    def forward(self, x, **kwargs):
        return self.embedder(x)


@pytest.mark.parametrize('dynamic_resolution', [False, True])
def test_radio_image_postprocessing_and_checkpoint_compatibility(dynamic_resolution):
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    with patch.object(core_radio, 'RADIOViTModel', _VisionBackbone):
        wrapper = core_radio.RADIOEncoderWrapper(
            config,
            ModuleSpec(module=torch.nn.Identity),
            None,
            img_h=2,
            img_w=2,
            patch_dim=1,
            class_token_len=1,
            dynamic_resolution=dynamic_resolution,
        )
    # One class token followed by a 2x2 image's embeddings.
    x = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    with torch.no_grad():
        wrapper.radio_model.embedder.weight.copy_(torch.eye(4))
    geometry = torch.tensor([[2, 2]]) if dynamic_resolution else None
    actual = wrapper(x, imgs_sizes=geometry)
    expected = x[:, 1:].reshape(1, 1, 16).bfloat16()
    torch.testing.assert_close(actual, expected)
    assert set(wrapper.state_dict()) == {'radio_model.embedder.weight'}
    clone = wrapper.state_dict()
    wrapper.load_state_dict(clone, strict=True)
    actual.sum().backward()
    assert wrapper.radio_model.embedder.weight.grad is not None
