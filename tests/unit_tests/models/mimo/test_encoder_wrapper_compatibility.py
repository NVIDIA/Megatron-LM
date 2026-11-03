# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility contracts for encoder wrappers shared with the MIMO example."""

from unittest.mock import patch

import pytest
import torch

from examples.mimo.model_providers import radio_encoder as example_radio
from examples.mimo.training.runtime import _EncoderFloat16Module
from megatron.core.models.mimo.model import MimoEncoderFloat16Module
from megatron.core.models.mimo.submodules import radio_encoder as core_radio
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
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
        self.temporal_patch_dim = kwargs['temporal_patch_dim']

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


# Exercise real RADIO packing with lightweight projections/decoder; these tests
# do not replace full Transformer Engine model-inference coverage.
class _Projection(torch.nn.Linear):
    def forward(self, x, **kwargs):
        return super().forward(x), None

    def gather_tensor_parallel_output(self, x):
        return x


class _Decoder(torch.nn.Module):
    def forward(self, x, **kwargs):
        return x


class _LightweightRADIO(RADIOViTModel):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.force_eval_mode = kwargs['force_eval_mode']
        self.patch_dim = kwargs['patch_dim']
        self.temporal_patch_dim = kwargs['temporal_patch_dim']
        self.dynamic_resolution = kwargs['dynamic_resolution']
        self.separate_video_embedder = kwargs['separate_video_embedder']
        self.class_token_len = kwargs['class_token_len']
        self.add_class_token = True
        self.class_token = torch.nn.Parameter(torch.full((1, self.class_token_len, 1), -99.0))
        width = 3 if self.separate_video_embedder else 3 * self.temporal_patch_dim
        self.embedder = _Projection(width, 1, bias=False)
        self.video_embedder = _Projection(3 * self.temporal_patch_dim, 1, bias=False)
        for layer in (self.embedder, self.video_embedder):
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight[0, 0] = 1
        self.decoder = _Decoder()
        self.ln_pre = self.ln_post = None
        if self.force_eval_mode:
            self.eval()

    def apply_pos_enc(self, x, input_size):
        return x, None


def _wrapper(**kwargs):
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    with patch.object(core_radio, 'RADIOViTModel', _LightweightRADIO):
        return core_radio.RADIOEncoderWrapper(
            config,
            ModuleSpec(module=torch.nn.Identity),
            None,
            img_h=4,
            img_w=4,
            patch_dim=1,
            class_token_len=1,
            dynamic_resolution=True,
            temporal_patch_dim=2,
            **kwargs,
        )


def _batch(device="cpu"):
    geometry = [(2, 4)] + [(4, 4)] * 3
    chunks = [
        torch.arange(n * 3, device=device).reshape(1, n, 3).float() + i * 100
        for i, n in enumerate([8, 16, 16, 16])
    ]
    boundaries = torch.tensor([0, 8, 24, 40, 56], dtype=torch.int32, device=device)
    packed = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )
    return torch.cat(chunks, dim=1), geometry, packed, chunks


@pytest.mark.parametrize(
    'separate_video_embedder,host_geometry,apply_pixel_shuffle,device',
    [
        (False, True, True, 'cuda'),
        (True, False, True, 'cuda'),
        (False, False, False, 'cpu'),
        (True, True, False, 'cuda'),
    ],
)
def test_mixed_image_video_grouping(
    separate_video_embedder, host_geometry, apply_pixel_shuffle, device
):
    device = torch.device(device, Utils.local_rank) if device == "cuda" else torch.device(device)
    wrapper = _wrapper(
        separate_video_embedder=separate_video_embedder, apply_pixel_shuffle=apply_pixel_shuffle
    ).to(device)
    x, geometry, packed, chunks = _batch(device)
    if not host_geometry:
        geometry = torch.tensor(geometry)
    counts = [2, 4, 4] if apply_pixel_shuffle else [8, 16, 16]
    out = wrapper(x, geometry, packed, num_frames=[1, 3], expected_output_counts=counts)
    # Tubelets use frames (0, 1), then (2, repeated 2); the projection
    # selects channel zero of the first frame, so frame 1 does not appear.
    expected_chunks = [chunks[i][..., :1] for i in [0, 1, 3]]
    if apply_pixel_shuffle:
        shuffled = []
        for chunk, (h, w) in zip(expected_chunks, [(2, 4), (4, 4), (4, 4)]):
            grid = chunk.reshape(h, w)
            neighborhoods = [
                grid[y : y + 2, x : x + 2].reshape(-1)
                for y in range(0, h, 2)
                for x in range(0, w, 2)
            ]
            shuffled.append(torch.stack(neighborhoods).unsqueeze(0))
        expected_chunks = shuffled
    torch.testing.assert_close(out, torch.cat(expected_chunks, dim=1))
    out.sum().backward()
    assert wrapper.radio_model.embedder.weight.grad is not None
    if separate_video_embedder:
        assert wrapper.radio_model.video_embedder.weight.grad is not None
    assert 'radio_model.class_token' in wrapper.state_dict()


def test_keep_class_tokens():
    wrapper = _wrapper(drop_class_token=False, apply_pixel_shuffle=False)
    x, geometry, packed, chunks = _batch()
    out = wrapper(x, geometry, packed, num_frames=[1, 3], expected_output_counts=[9, 17, 17])
    assert out[0, [0, 9, 26], 0].tolist() == [-99, -99, -99]


@pytest.mark.parametrize('counts', [[3, 3, 4], [2, 4], [2.5, 4, 4], [[2, 4, 4]], [-2, 4, 4]])
def test_invalid_output_counts(counts):
    wrapper = _wrapper()
    x, geometry, packed, _ = _batch()
    with pytest.raises(ValueError, match='counts'):
        wrapper(x, geometry, packed, num_frames=[1, 3], expected_output_counts=counts)


@pytest.mark.parametrize('geometry', [[[0, 4]], [[3, 4]], [[2.5, 4]], [[2, 4, 6]]])
def test_invalid_geometry(geometry):
    wrapper = _wrapper()
    x = torch.ones(1, 12, 3)
    with pytest.raises(ValueError):
        wrapper(x, geometry, num_frames=[1])


def test_frozen_encoder():
    wrapper = _wrapper(force_eval_mode=True)
    wrapper.train()
    assert not wrapper.radio_model.training
    x, geometry, packed, _ = _batch()
    out = wrapper(x.requires_grad_(), geometry, packed, num_frames=[1, 3])
    assert not out.requires_grad


def test_constructor_options():
    options = dict(
        force_cpe_eval_mode=True,
        interpolate_only_cpe=True,
        cpe_aspect_ratio_select=True,
        disable_cpe=True,
        temporal_ckpt_compat=True,
        separate_video_embedder=True,
    )
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    with patch.object(core_radio, 'RADIOViTModel') as backbone:
        core_radio.RADIOEncoderWrapper(
            config,
            ModuleSpec(module=torch.nn.Identity),
            None,
            img_h=4,
            img_w=4,
            patch_dim=1,
            class_token_len=1,
            temporal_patch_dim=2,
            **options,
        )
    actual = backbone.call_args.kwargs
    assert actual['has_cpe'] is False
    assert actual['temporal_patch_dim'] == 2
    for name, value in options.items():
        if name != 'disable_cpe':
            assert actual[name] == value


def test_actual_output_length_mismatch():
    wrapper = _wrapper(apply_pixel_shuffle=False)
    x, geometry, packed, _ = _batch()

    def append_embedding(module, inputs, output):
        embeddings, sizes, frames = output
        return torch.cat([embeddings, embeddings[:, -1:]], dim=1), sizes, frames

    with wrapper.radio_model.register_forward_hook(append_embedding):
        with pytest.raises(ValueError, match='output length'):
            wrapper(x, geometry, packed, num_frames=[1, 3], expected_output_counts=[8, 16, 16])


def test_pixel_shuffle_rejects_retained_class_tokens():
    wrapper = _wrapper(drop_class_token=False)
    x, geometry, packed, _ = _batch()
    with pytest.raises(ValueError, match='class tokens'):
        wrapper(x, geometry, packed, num_frames=[1, 3])
