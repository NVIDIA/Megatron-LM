# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal.mimo_checkpoint_model import (
    MimoCheckpointLLaVAModel,
    _unfused_fast_gelu,
    vision_config,
    vision_projection_config,
)
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    vlm_dynamic_inference,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.transformer.transformer_config import TransformerConfig

_MODALITY = 'modality_submodules.images.module.module.'
_LANGUAGE_KEY = 'language_model.module.module.embedding.word_embeddings.weight'
_ENCODER_KEY = f'{_MODALITY}encoders.vision_encoder.patch_embed.weight'
_PROJECTION_KEY = f'{_MODALITY}input_projections.0.encoder.linear_fc1.weight'
_PREFIX_MAP = {
    'language_model.': 'language_model.module.module.',
    'vision_model.': f'{_MODALITY}encoders.vision_encoder.',
    'vision_projection.': f'{_MODALITY}input_projections.0.',
}


def _sharded(key):
    return ShardedTensor.from_rank_offsets(key, torch.zeros(1))


@pytest.fixture
def checkpoint_keys(monkeypatch):
    """Serve the given keys as the tensor keys of the loaded checkpoint."""
    keys = []
    monkeypatch.setattr(vlm_dynamic_inference, '_checkpoint_tensor_keys', lambda args: keys)
    return keys


def test_mimo_checkpoint_prefix_map(checkpoint_keys):
    args = SimpleNamespace()
    checkpoint_keys += [_LANGUAGE_KEY]
    assert vlm_dynamic_inference._mimo_checkpoint_prefix_map(args) is None

    checkpoint_keys += [_ENCODER_KEY, _PROJECTION_KEY]
    assert vlm_dynamic_inference._mimo_checkpoint_prefix_map(args) == _PREFIX_MAP

    checkpoint_keys += [f'{_MODALITY}encoders.other_encoder.patch_embed.weight']
    with pytest.raises(ValueError, match='one vision encoder'):
        vlm_dynamic_inference._mimo_checkpoint_prefix_map(args)


def test_sharded_state_dict_uses_checkpoint_keys(monkeypatch):
    monkeypatch.setattr(
        LLaVAModel,
        'sharded_state_dict',
        lambda self, prefix='', sharded_offsets=(), metadata=None: {
            name: _sharded(f'{prefix}{name}')
            for name in ('language_model.embedding.weight', 'vision_model.patch_embed.weight')
        },
    )
    # Skip the full model build; only the key mapping is under test.
    model = MimoCheckpointLLaVAModel.__new__(MimoCheckpointLLaVAModel)
    object.__setattr__(model, 'checkpoint_prefix_map', _PREFIX_MAP)

    keys = {value.key for value in model.sharded_state_dict().values()}
    assert keys == {
        'language_model.module.module.embedding.weight',
        f'{_MODALITY}encoders.vision_encoder.patch_embed.weight',
    }


def test_resolve_mimo_vision_args():
    args = SimpleNamespace(
        vision_model_type='pixtral-vit-large', patch_dim=16, pixel_shuffle=True, image_token_id=7
    )
    with pytest.raises(ValueError, match='--vision-model-type'):
        vlm_dynamic_inference._resolve_mimo_vision_args(args, SimpleNamespace(), set())

    checkpoint_args = SimpleNamespace(image_token_id=3)
    vlm_dynamic_inference._resolve_mimo_vision_args(
        args, checkpoint_args, {'vision_model_type', 'patch_dim'}
    )
    # CLI values win; the rest come from the encoder registry and the checkpoint.
    assert args.patch_dim == 16
    assert (args.pixel_shuffle, args.conv_merging, args.dynamic_resolution) == (False, True, True)
    assert args.image_token_id == 3
    sources = {record['attr']: record['source'] for record in args._vlm_arg_resolution}
    assert sources['patch_dim'] == 'cli'
    assert sources['img_h'] == 'encoder registry'
    assert sources['image_token_id'] == 'checkpoint'


def test_check_mimo_checkpoint_fully_loaded(monkeypatch, checkpoint_keys):
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(
        torch.distributed, 'all_gather_object', lambda out, obj: out.__setitem__(0, obj)
    )
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda objs, src: None)
    args = SimpleNamespace(
        mimo_checkpoint_prefix_map={
            'language_model.': 'language_model.module.module.',
            'vision_model.': f'{_MODALITY}encoders.vision_encoder.',
        }
    )
    factory_key = 'language_model.module.module.decoder.layers.0.mixer.in_proj.weight'
    model = SimpleNamespace(
        sharded_state_dict=lambda: {'a': _sharded(_LANGUAGE_KEY), 'b': _sharded(factory_key)}
    )
    checkpoint_keys += [
        _LANGUAGE_KEY,
        f'{factory_key}.z',
        'language_model.module.module.decoder.layers.1.mlp.router.qb_bin_bounds',
        _PROJECTION_KEY,  # Outside the mapped prefixes.
    ]
    # Every checkpoint tensor under the mapped prefixes is accounted for.
    vlm_dynamic_inference._check_mimo_checkpoint_fully_loaded(args, model)

    checkpoint_keys += [_ENCODER_KEY]
    with pytest.raises(RuntimeError, match='patch_embed'):
        vlm_dynamic_inference._check_mimo_checkpoint_fully_loaded(args, model)


def test_mimo_vision_configs():
    language_config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
    args = SimpleNamespace(
        vision_model_type='pixtral-vit-large',
        vision_num_layers=2,
        vision_hidden_size=64,
        vision_ffn_hidden_size=128,
        vision_num_attention_heads=4,
        vision_num_query_groups=None,
        vision_kv_channels=None,
        vision_projection_activation='fast_gelu',
    )
    config = vision_config(args, language_config)
    # Sizes come from the arguments, head layout follows them, structure from the registry.
    assert (config.num_layers, config.hidden_size, config.ffn_hidden_size) == (2, 64, 128)
    assert (config.num_attention_heads, config.num_query_groups, config.kv_channels) == (4, 4, 16)
    assert config.normalization == 'RMSNorm' and config.gated_linear_unit
    assert not config.add_bias_linear
    assert config.vision_model_type == 'pixtral-vit-large'

    projection = vision_projection_config(args, language_config, config.add_bias_linear)
    assert (projection.hidden_size, projection.ffn_hidden_size) == (32, 32)
    assert projection.activation_func is _unfused_fast_gelu
    assert not projection.add_bias_linear
